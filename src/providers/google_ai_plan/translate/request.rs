//! Anthropic Messages request → Interactions API request.
//!
//! Every request is stateless (`store: false`) and carries the whole
//! conversation as typed steps: `user_input`, `thought`, `model_output`,
//! `function_call` and `function_result`.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::{Value, json};

use super::thought_signature::{SKIP_THOUGHT_SIGNATURE_VALIDATOR, decode_envelope};
use crate::anthropic::schema::MessagesRequest;
use crate::providers::google_ai_plan::models::{MAX_OUTPUT_TOKENS, ResolvedModel, ThinkingLevel};
use crate::providers::translate_shared::{
    ContentBlock, flatten_system_text, normalize_content, read_effort,
};
use crate::providers::vertex::translate::request::{
    FunctionCallingMode, inline_image, read_function_declarations, read_tool_choice,
    render_tool_result,
};

/// How this provider is named in request errors.
const PROVIDER_LABEL: &str = "Google AI Ultra plan";

/// The `interactions:createPerUserQuota` request body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InteractionRequest {
    /// The upstream model ID.
    pub model: String,
    /// Always `false`: the conversation is sent whole every time.
    pub store: bool,
    /// Always `true`: the response is an SSE stream of step events.
    pub stream: bool,
    /// The conversation as steps.
    pub input: Vec<Step>,
    /// System prompt, from the request's `system` and `system` / `developer`
    /// messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<String>,
    /// Function tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Thinking, output length and tool-choice settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Whether the client enabled thinking, which is whether the response
    /// translation forwards thought summaries as thinking text. The body
    /// always asks for summaries, so this is not derived from it. Never sent.
    #[serde(skip)]
    pub emit_thinking: bool,
}

/// One step of the conversation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Step {
    /// What the user sent: text and images.
    UserInput {
        /// The parts, never empty.
        content: Vec<Content>,
    },
    /// A thought the model had, replayed with its signature.
    Thought {
        /// The opaque signature, or the skip sentinel.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// The thought summary the client kept.
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<Vec<Content>>,
    },
    /// Text the model wrote.
    ModelOutput {
        /// The parts, never empty.
        content: Vec<Content>,
    },
    /// A function call the model made.
    FunctionCall {
        /// The call ID, which the result's `call_id` names.
        id: String,
        /// The function name.
        name: String,
        /// The arguments object.
        arguments: Value,
    },
    /// The result of a function call.
    FunctionResult {
        /// The ID of the call this answers.
        call_id: String,
        /// The function name.
        name: String,
        /// The result text, or a content list when it has images.
        result: FunctionResultValue,
        /// Whether the call failed.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

/// A text or image part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Plain text.
    Text {
        /// The text.
        text: String,
    },
    /// A base64 image.
    Image {
        /// The media type, for example `image/png`.
        mime_type: String,
        /// The base64 payload.
        data: String,
    },
}

/// `function_result.result`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum FunctionResultValue {
    /// A result without images.
    Text(String),
    /// A result with images: its text first, when there is any.
    Content(Vec<Content>),
}

/// A function the model may call.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tool {
    /// Always `function`.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The function name.
    pub name: String,
    /// What the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The Anthropic `input_schema`, sent unchanged as JSON Schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// `generation_config`. No sampling settings: the plan's request shape has
/// none, so the client's `temperature`, `top_p` and `top_k` are dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GenerationConfig {
    /// The thinking level, always one the model accepts.
    pub thinking_level: ThinkingLevel,
    /// Always `auto`, so the stream carries thought summaries.
    pub thinking_summaries: &'static str,
    /// Output token limit, from `max_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// How the model may call tools; sent only with tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// `generation_config.tool_choice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// The model must call a tool.
    Any,
    /// The model must not call a tool.
    None,
}

/// Translates an Anthropic Messages request for `model`.
///
/// # Errors
///
/// Fails, with a message meant for a 400 `invalid_request_error`, for what
/// the Interactions API has no counterpart for or cannot receive: URL images,
/// Anthropic server tools, `stop_sequences`, a function name Gemini does not
/// accept, a `tool_choice` naming a tool the request does not declare, a
/// `tool_result` that matches no earlier `tool_use`, an unknown message role,
/// a conversation without steps, a `max_tokens` of 0, or malformed fields.
pub fn prepare_request(
    body: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<InteractionRequest> {
    let mut system = Vec::new();
    if let Some(text) = request_system_text(body.extra.get("system")) {
        system.push(text);
    }
    let mut builder = StepsBuilder::new(&model.upstream_model);
    for message in &body.messages {
        let blocks = normalize_content(&message.content, json!({}));
        match message.role.as_str() {
            "system" | "developer" => {
                let text = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } if !text.trim().is_empty() => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    system.push(text);
                }
            }
            "user" => builder.push_user(&blocks)?,
            "assistant" => builder.push_model(&blocks)?,
            other => anyhow::bail!("unexpected message role: {other}"),
        }
    }
    let input = builder.finish();
    if input.is_empty() {
        anyhow::bail!("messages must contain at least one non-empty user or assistant turn");
    }
    reject_stop_sequences(body)?;
    // An explicit 0 has no counterpart: leaving the limit out would run the
    // request with the provider's default instead.
    if body.max_tokens == Some(0) {
        anyhow::bail!("max_tokens must be at least 1");
    }

    let mut tools: Vec<Tool> = read_function_declarations(body, PROVIDER_LABEL)?
        .into_iter()
        .map(|declaration| Tool {
            kind: "function",
            name: declaration.name,
            description: declaration.description,
            parameters: declaration.parameters_json_schema,
        })
        .collect();
    let choice = read_tool_choice(body)?;
    let tool_choice = match choice {
        None => None,
        Some(config) => {
            if tools.is_empty() && config.mode != FunctionCallingMode::None {
                anyhow::bail!("tool_choice requires at least one tool");
            }
            // The Interactions tool choice names no function, so a choice
            // of one tool sends `any` with only that tool declared.
            if let Some(name) = config
                .allowed_function_names
                .as_ref()
                .and_then(|names| names.first())
            {
                tools.retain(|tool| &tool.name == name);
                if tools.is_empty() {
                    anyhow::bail!(
                        "tool_choice names the tool {name}, which tools does not declare"
                    );
                }
            }
            Some(match config.mode {
                FunctionCallingMode::Auto => ToolChoice::Auto,
                FunctionCallingMode::Any => ToolChoice::Any,
                FunctionCallingMode::None => ToolChoice::None,
            })
        }
    };
    let (thinking_level, emit_thinking) = read_thinking(body, model)?;

    Ok(InteractionRequest {
        model: model.upstream_model.clone(),
        store: false,
        stream: true,
        input,
        system_instruction: (!system.is_empty()).then(|| system.join("\n\n")),
        generation_config: Some(GenerationConfig {
            thinking_level,
            thinking_summaries: "auto",
            max_output_tokens: body.max_tokens.map(|value| value.min(MAX_OUTPUT_TOKENS)),
            tool_choice: tool_choice.filter(|_| !tools.is_empty()),
        }),
        tools: (!tools.is_empty()).then_some(tools),
        emit_thinking,
    })
}

/// Builds `input`, keeping consecutive messages of one role together.
struct StepsBuilder<'a> {
    model: &'a str,
    steps: Vec<Step>,
    /// Function results of the user turn being collected, in order.
    results: Vec<Step>,
    /// Text and images of the user turn being collected, in order.
    user_content: Vec<Content>,
    /// Function name of every `tool_use` seen so far, by ID.
    tool_names: HashMap<String, String>,
    /// Whether the current model turn already has a signed thought step.
    signed: bool,
}

impl<'a> StepsBuilder<'a> {
    fn new(model: &'a str) -> Self {
        Self {
            model,
            steps: Vec::new(),
            results: Vec::new(),
            user_content: Vec::new(),
            tool_names: HashMap::new(),
            signed: false,
        }
    }

    fn push_model(&mut self, blocks: &[ContentBlock]) -> anyhow::Result<()> {
        self.flush_user();
        for block in blocks {
            match block {
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    // Only this model's own envelopes are replayed; thinking
                    // from another model or provider is dropped whole.
                    let Some(decoded) = signature
                        .as_deref()
                        .and_then(decode_envelope)
                        .filter(|decoded| decoded.model == self.model)
                    else {
                        continue;
                    };
                    let summary = (!thinking.trim().is_empty()).then(|| {
                        vec![Content::Text {
                            text: thinking.clone(),
                        }]
                    });
                    let signature =
                        (!decoded.signature.is_empty()).then(|| decoded.signature.to_string());
                    if signature.is_none() && summary.is_none() {
                        continue;
                    }
                    self.signed |= signature.is_some();
                    self.steps.push(Step::Thought { signature, summary });
                }
                ContentBlock::Text { text } if !text.is_empty() => {
                    let part = Content::Text { text: text.clone() };
                    match self.steps.last_mut() {
                        Some(Step::ModelOutput { content }) => content.push(part),
                        _ => self.steps.push(Step::ModelOutput {
                            content: vec![part],
                        }),
                    }
                }
                ContentBlock::ToolUse { id, name, input } => {
                    if id.is_empty() || name.is_empty() {
                        anyhow::bail!("assistant tool_use requires non-empty id and name");
                    }
                    if !input.is_object() {
                        anyhow::bail!("assistant tool_use {id} input must be an object");
                    }
                    // Gemini rejects a function call of a model turn without
                    // a signed thought step before it, for example after the
                    // client switched models mid tool loop; Google's sentinel
                    // skips the check.
                    if !self.signed {
                        self.steps.push(Step::Thought {
                            signature: Some(SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_string()),
                            summary: None,
                        });
                        self.signed = true;
                    }
                    self.tool_names.insert(id.clone(), name.clone());
                    self.steps.push(Step::FunctionCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    });
                }
                ContentBlock::Text { .. }
                | ContentBlock::Image { .. }
                | ContentBlock::ToolResult { .. } => {}
            }
        }
        Ok(())
    }

    fn push_user(&mut self, blocks: &[ContentBlock]) -> anyhow::Result<()> {
        for block in blocks {
            match block {
                ContentBlock::Text { text } if !text.is_empty() => {
                    self.user_content.push(Content::Text { text: text.clone() });
                }
                ContentBlock::Image { source } => {
                    let image = inline_image(source, PROVIDER_LABEL)?;
                    self.user_content.push(Content::Image {
                        mime_type: image.mime_type,
                        data: image.data,
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let name = self.tool_names.get(tool_use_id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "tool_result {tool_use_id} does not match any earlier tool_use"
                        )
                    })?;
                    let rendered = render_tool_result(content, PROVIDER_LABEL)?;
                    let result = if rendered.images.is_empty() {
                        FunctionResultValue::Text(rendered.text)
                    } else {
                        let text = (!rendered.text.is_empty()).then_some(Content::Text {
                            text: rendered.text,
                        });
                        FunctionResultValue::Content(
                            text.into_iter()
                                .chain(rendered.images.into_iter().map(|image| Content::Image {
                                    mime_type: image.mime_type,
                                    data: image.data,
                                }))
                                .collect(),
                        )
                    };
                    self.results.push(Step::FunctionResult {
                        call_id: tool_use_id.clone(),
                        name: name.clone(),
                        result,
                        is_error: is_error.unwrap_or(false),
                    });
                }
                ContentBlock::Text { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::ToolUse { .. } => {}
            }
        }
        Ok(())
    }

    /// Writes the collected user turn: function results first, so they
    /// follow the calls they answer, then one `user_input` step with the
    /// turn's text and images. A user turn that produced steps ends the
    /// model turn before it.
    fn flush_user(&mut self) {
        let results = std::mem::take(&mut self.results);
        let content = std::mem::take(&mut self.user_content);
        if results.is_empty() && content.is_empty() {
            return;
        }
        self.signed = false;
        self.steps.extend(results);
        if !content.is_empty() {
            self.steps.push(Step::UserInput { content });
        }
    }

    fn finish(mut self) -> Vec<Step> {
        self.flush_user();
        self.steps
    }
}

/// The request's `system` field as text, or `None` when it has no text that
/// is not blank. Blank text blocks are dropped before the blocks are joined,
/// so whitespace is never sent as, or joined into, the instruction.
fn request_system_text(system: Option<&Value>) -> Option<String> {
    let is_blank_block = |block: &Value| {
        block
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.trim().is_empty())
    };
    let text = match system? {
        Value::Array(blocks) if blocks.iter().any(is_blank_block) => {
            let kept = blocks
                .iter()
                .filter(|block| !is_blank_block(block))
                .cloned()
                .collect();
            flatten_system_text(Some(&Value::Array(kept)))
        }
        system => flatten_system_text(Some(system)),
    }?;
    (!text.trim().is_empty()).then_some(text)
}

fn reject_stop_sequences(body: &MessagesRequest) -> anyhow::Result<()> {
    match body.extra.get("stop_sequences") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(values)) if values.is_empty() => Ok(()),
        Some(_) => anyhow::bail!("stop_sequences is not supported by the {PROVIDER_LABEL}"),
    }
}

/// Maps Claude Code's effort and thinking settings to a thinking level,
/// then moves it to the nearest level the model accepts:
///
/// | effort | thinking | requested level |
/// |---|---|---|
/// | low / medium / high / xhigh | any | the same level |
/// | max | any | xhigh |
/// | none | on | high |
/// | none | off or absent | minimal |
///
/// Thinking is on for any `thinking.type` other than `disabled`;
/// `budget_tokens` is not used. No plan model can turn thinking off, so a
/// level is always sent. The second value is whether thinking is on.
fn read_thinking(
    body: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<(ThinkingLevel, bool)> {
    let thinking_on = match body.extra.get("thinking") {
        None | Some(Value::Null) => false,
        Some(Value::Object(thinking)) => {
            thinking.get("type").and_then(Value::as_str) != Some("disabled")
        }
        Some(_) => anyhow::bail!("thinking must be an object"),
    };
    let requested = match read_effort(body)? {
        Some("low") => ThinkingLevel::Low,
        Some("medium") => ThinkingLevel::Medium,
        Some("high") => ThinkingLevel::High,
        Some(_) => ThinkingLevel::Xhigh,
        // A level is always named: the request shape pairs it with
        // `thinking_summaries`, and summaries without a level is a request
        // nobody has seen accepted. `high` is a level all four models take.
        None if thinking_on => ThinkingLevel::High,
        // Thinking cannot be turned off; the clamp picks the lowest level.
        None => ThinkingLevel::Minimal,
    };
    Ok((model.thinking.clamp(requested), thinking_on))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::google_ai_plan::models::resolve;
    use crate::providers::google_ai_plan::translate::thought_signature::encode_envelope;
    use crate::providers::vertex::translate::thought_signature as vertex;

    const MODEL: &str = "gemini-4-hydrogen";
    const SIG: &str = "c2lnbmF0dXJl";

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("test request deserializes")
    }

    fn prepared_for(model: &str, value: Value) -> InteractionRequest {
        let model = resolve(&format!("google-ai-plan/{model}")).expect("model resolves");
        prepare_request(&request(value), &model)
            .unwrap_or_else(|error| panic!("request should translate: {error}"))
    }

    fn wire_for(model: &str, value: Value) -> Value {
        serde_json::to_value(prepared_for(model, value)).expect("request serializes")
    }

    fn wire(value: Value) -> Value {
        wire_for(MODEL, value)
    }

    fn error(value: Value) -> String {
        let model = resolve(&format!("google-ai-plan/{MODEL}")).expect("model resolves");
        prepare_request(&request(value), &model)
            .expect_err("request should be rejected")
            .to_string()
    }

    fn steps(wire: &Value) -> &[Value] {
        wire["input"].as_array().expect("input is an array")
    }

    fn envelope(model: &str, signature: &str) -> String {
        encode_envelope(model, signature).expect("envelope encodes")
    }

    /// The conversation of the reference replay test: a user turn, a model
    /// turn with a thinking block and a call, and the call's result.
    fn replay(thinking: Value) -> Value {
        json!({"messages":[
            {"role":"user","content":"Read a.py"},
            {"role":"assistant","content":[
                thinking,
                {"type":"tool_use","id":"call_1","name":"read_file","input":{"path":"a.py"}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"x = 1"}]}
            ]}
        ]})
    }

    #[test]
    fn stateless_request_with_a_clamped_level_and_no_sampling_settings() {
        let wire = wire(json!({
            "system":"Be brief.",
            "temperature":0.5,
            "top_p":0.9,
            "top_k":40,
            "max_tokens":1024,
            "tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
            "messages":[{"role":"user","content":"Hi"}]
        }));
        assert_eq!(
            wire,
            json!({
                "model":"gemini-4-hydrogen",
                "store":false,
                "stream":true,
                "input":[{"type":"user_input","content":[{"type":"text","text":"Hi"}]}],
                "system_instruction":"Be brief.",
                "tools":[{"type":"function","name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}],
                "generation_config":{"thinking_level":"low","thinking_summaries":"auto","max_output_tokens":1024}
            }),
            "a request without thinking asks for the lowest level Hydrogen accepts, not minimal"
        );
        let rendered = wire.to_string();
        for absent in ["temperature", "top_p", "top_k", "emit_thinking"] {
            assert!(
                !rendered.contains(absent),
                "{absent} must not be sent: {rendered}"
            );
        }
    }

    #[test]
    fn replays_the_models_own_signed_thought_before_its_function_call() {
        let wire = wire(replay(
            json!({"type":"thinking","thinking":"Read the file first.","signature":envelope(MODEL, SIG)}),
        ));
        assert_eq!(
            wire["input"],
            json!([
                {"type":"user_input","content":[{"type":"text","text":"Read a.py"}]},
                {"type":"thought","signature":SIG,"summary":[{"type":"text","text":"Read the file first."}]},
                {"type":"function_call","id":"call_1","name":"read_file","arguments":{"path":"a.py"}},
                {"type":"function_result","call_id":"call_1","name":"read_file","result":"x = 1"}
            ])
        );
    }

    #[test]
    fn function_calls_without_this_models_signature_get_the_skip_sentinel() {
        let foreign = [
            (
                "an Anthropic signature",
                json!({"type":"thinking","thinking":"Read the file first.","signature":"ErUBCkYIBRgCIkA8YWZb"}),
            ),
            (
                "another plan model",
                json!({"type":"thinking","thinking":"x","signature":envelope("gemini-4-neon", SIG)}),
            ),
            (
                "a Vertex AI envelope for the same model",
                json!({"type":"thinking","thinking":"x","signature":vertex::encode_envelope(MODEL, SIG).expect("encodes")}),
            ),
            ("no signature", json!({"type":"thinking","thinking":""})),
        ];
        for (case, thinking) in foreign {
            let wire = wire(replay(thinking));
            assert_eq!(
                steps(&wire)[1..3],
                [
                    json!({"type":"thought","signature":"skip_thought_signature_validator"}),
                    json!({"type":"function_call","id":"call_1","name":"read_file","arguments":{"path":"a.py"}}),
                ],
                "{case}"
            );
            assert!(
                !wire.to_string().contains(SIG),
                "{case}: foreign signature leaked"
            );
        }
    }

    #[test]
    fn unsigned_envelope_keeps_its_summary_and_still_gets_the_sentinel() {
        let wire = wire(replay(
            json!({"type":"thinking","thinking":"Plan.","signature":envelope(MODEL, "")}),
        ));
        assert_eq!(
            steps(&wire)[1..4],
            [
                json!({"type":"thought","summary":[{"type":"text","text":"Plan."}]}),
                json!({"type":"thought","signature":"skip_thought_signature_validator"}),
                json!({"type":"function_call","id":"call_1","name":"read_file","arguments":{"path":"a.py"}}),
            ]
        );
    }

    #[test]
    fn one_signed_thought_covers_every_call_of_the_model_turn_and_resets_after_results() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"","signature":envelope(MODEL, SIG)},
                {"type":"tool_use","id":"a","name":"Read","input":{}}
            ]},
            {"role":"assistant","content":[{"type":"tool_use","id":"b","name":"Grep","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content":"1"},
                {"type":"tool_result","tool_use_id":"b","content":"2","is_error":true}
            ]},
            {"role":"assistant","content":[
                {"type":"text","text":"again"},
                {"type":"tool_use","id":"c","name":"Read","input":{}}
            ]}
        ]}));
        let kinds: Vec<String> = wire["input"]
            .as_array()
            .expect("input is an array")
            .iter()
            .map(|step| match step["signature"].as_str() {
                Some(signature) => format!("thought:{signature}"),
                None => step["type"].as_str().unwrap_or_default().to_string(),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "user_input",
                &format!("thought:{SIG}"),
                "function_call",
                "function_call",
                "function_result",
                "function_result",
                "model_output",
                "thought:skip_thought_signature_validator",
                "function_call",
            ]
        );
        assert_eq!(wire["input"][5]["is_error"], json!(true));
        assert_eq!(wire["input"][4].get("is_error"), None, "false is not sent");
    }

    #[test]
    fn tool_results_come_before_text_and_carry_images_as_content() {
        let image = json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}});
        let wire = wire(json!({"messages":[
            {"role":"user","content":"look"},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Screenshot","input":{}}]},
            {"role":"user","content":[
                {"type":"text","text":"also this"},
                {"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"captured"}, image]},
                image
            ]}
        ]}));
        assert_eq!(
            steps(&wire)[3..],
            [
                json!({"type":"function_result","call_id":"t1","name":"Screenshot","result":[
                    {"type":"text","text":"captured"},
                    {"type":"image","mime_type":"image/png","data":"iVBORw0KGgo="}
                ]}),
                json!({"type":"user_input","content":[
                    {"type":"text","text":"also this"},
                    {"type":"image","mime_type":"image/png","data":"iVBORw0KGgo="}
                ]}),
            ]
        );
        let image_only = wire_for(
            MODEL,
            json!({"messages":[
                {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Shot","input":{}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[image]}]}
            ]}),
        );
        assert_eq!(
            image_only["input"][2]["result"],
            json!([{"type":"image","mime_type":"image/png","data":"iVBORw0KGgo="}]),
            "an image-only result has no empty text part"
        );
    }

    #[test]
    fn consecutive_model_text_merges_into_one_model_output() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"a"},
            {"role":"user","content":"b"},
            {"role":"assistant","content":"c"},
            {"role":"assistant","content":[{"type":"text","text":""},{"type":"text","text":"d"}]}
        ]}));
        assert_eq!(
            wire["input"],
            json!([
                {"type":"user_input","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
                {"type":"model_output","content":[{"type":"text","text":"c"},{"type":"text","text":"d"}]}
            ])
        );
    }

    #[test]
    fn system_prompt_and_system_messages_join_into_the_system_instruction() {
        let wire = wire(json!({
            "system":[
                {"type":"text","text":"x-anthropic-billing-header: dropped"},
                {"type":"text","text":"first"}
            ],
            "messages":[
                {"role":"system","content":"second"},
                {"role":"developer","content":[{"type":"text","text":"third"}]},
                {"role":"user","content":"hi"}
            ]
        }));
        assert_eq!(
            wire["system_instruction"],
            json!("first\n\nsecond\n\nthird")
        );
    }

    #[test]
    fn a_blank_system_prompt_sends_no_system_instruction() {
        let tests = [
            ("an empty string", json!({"system":""})),
            ("a whitespace-only string", json!({"system":" \n\t "})),
            (
                "blank text blocks",
                json!({"system":[{"type":"text","text":""},{"type":"text","text":"  \n"}]}),
            ),
            (
                "a whitespace-only system message",
                json!({"messages":[{"role":"system","content":"  "}]}),
            ),
            (
                "blank text blocks of a developer message",
                json!({"messages":[{"role":"developer","content":[{"type":"text","text":" "},{"type":"text","text":"\n"}]}]}),
            ),
        ];
        for (case, extra) in tests {
            let mut body = json!({"messages":[]});
            if let Some(system) = extra.get("system") {
                body["system"] = system.clone();
            }
            let messages = body["messages"].as_array_mut().expect("messages");
            if let Some(leading) = extra.get("messages").and_then(Value::as_array) {
                messages.extend(leading.iter().cloned());
            }
            messages.push(json!({"role":"user","content":"hi"}));
            let wire = wire(body);
            assert_eq!(wire.get("system_instruction"), None, "{case}: {wire}");
        }
    }

    #[test]
    fn blank_system_blocks_are_not_joined_into_the_system_instruction() {
        let tests = [
            (
                json!([{"type":"text","text":""},{"type":"text","text":"first"}]),
                "first",
            ),
            (
                json!([{"type":"text","text":" first "},{"type":"text","text":"\n"},{"type":"text","text":"second"}]),
                " first \n\nsecond",
            ),
            (json!("Be brief."), "Be brief."),
            (json!("  Be brief.\n"), "  Be brief.\n"),
        ];
        for (system, expected) in tests {
            let wire = wire(json!({
                "system":system,
                "messages":[
                    {"role":"system","content":[{"type":"text","text":" "},{"type":"text","text":"rules"}]},
                    {"role":"user","content":"hi"}
                ]
            }));
            assert_eq!(
                wire["system_instruction"],
                json!(format!("{expected}\n\nrules")),
                "{system}"
            );
        }
    }

    #[test]
    fn tool_choice_forms() {
        let tools = json!([
            {"name":"Read","input_schema":{"type":"object"}},
            {"name":"Grep","input_schema":{"type":"object"}}
        ]);
        let tests = [
            (
                "auto",
                json!({"type":"auto"}),
                Some("auto"),
                vec!["Read", "Grep"],
            ),
            (
                "any",
                json!({"type":"any","disable_parallel_tool_use":true}),
                Some("any"),
                vec!["Read", "Grep"],
            ),
            (
                "none",
                json!({"type":"none"}),
                Some("none"),
                vec!["Read", "Grep"],
            ),
            (
                "one tool",
                json!({"type":"tool","name":"Grep"}),
                Some("any"),
                vec!["Grep"],
            ),
            ("absent", Value::Null, None, vec!["Read", "Grep"]),
        ];
        for (case, choice, expected, names) in tests {
            let wire = wire(json!({
                "tools":tools,
                "tool_choice":choice,
                "messages":[{"role":"user","content":"hi"}]
            }));
            assert_eq!(
                wire["generation_config"]
                    .get("tool_choice")
                    .and_then(Value::as_str),
                expected,
                "{case}"
            );
            let sent: Vec<&str> = wire["tools"]
                .as_array()
                .expect("tools are sent")
                .iter()
                .map(|tool| tool["name"].as_str().unwrap_or_default())
                .collect();
            assert_eq!(sent, names, "{case}");
        }
        let without_tools = wire(json!({
            "tool_choice":{"type":"none"},
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(without_tools.get("tools"), None);
        assert_eq!(without_tools["generation_config"].get("tool_choice"), None);
    }

    #[test]
    fn each_models_level_is_clamped_to_what_it_accepts() {
        let tests = [
            // (model, effort, thinking type, level, emit_thinking)
            ("gemini-4-hydrogen", None, None, "low", false),
            ("gemini-4-hydrogen", None, Some("enabled"), "high", true),
            (
                "gemini-4-hydrogen",
                Some("xhigh"),
                Some("adaptive"),
                "xhigh",
                true,
            ),
            ("gemini-4-hydrogen", Some("max"), None, "xhigh", false),
            ("gemini-4-neon", Some("low"), Some("disabled"), "low", false),
            ("gemini-4-neon", Some("max"), Some("enabled"), "xhigh", true),
            ("gemini-4-carbon", Some("xhigh"), None, "high", false),
            ("gemini-4-carbon", Some("max"), None, "high", false),
            ("gemini-4-carbon", None, None, "low", false),
            ("gemini-4-carbon", None, Some("enabled"), "high", true),
            ("gemini-4-neon", None, Some("adaptive"), "high", true),
            ("gemini-4-neon", None, None, "low", false),
            ("gemini-3.5-flash-lite", None, Some("enabled"), "high", true),
            ("gemini-4-carbon", Some("medium"), None, "medium", false),
            ("gemini-3.5-flash-lite", None, None, "minimal", false),
            ("gemini-3.5-flash-lite", Some("low"), None, "low", false),
            (
                "gemini-3.5-flash-lite",
                Some("xhigh"),
                Some("enabled"),
                "high",
                true,
            ),
            ("gemini-4-argon", Some("xhigh"), None, "high", false),
            ("gemini-4-argon", None, None, "low", false),
        ];
        for (model, effort, thinking, level, emit_thinking) in tests {
            let mut body = json!({"messages":[{"role":"user","content":"hi"}]});
            if let Some(effort) = effort {
                body["output_config"] = json!({"effort":effort});
            }
            if let Some(kind) = thinking {
                body["thinking"] = json!({"type":kind,"budget_tokens":1024});
            }
            let prepared = prepared_for(model, body);
            let case = format!("{model} effort={effort:?} thinking={thinking:?}");
            assert_eq!(
                serde_json::to_value(&prepared).expect("serializes")["generation_config"]["thinking_level"],
                json!(level),
                "{case}"
            );
            assert_eq!(prepared.emit_thinking, emit_thinking, "{case}");
        }
    }

    #[test]
    fn max_tokens_is_capped_and_omitted_when_absent() {
        let tests = [
            ("absent", None, None),
            ("one", Some(1), Some(1)),
            (
                "the limit",
                Some(MAX_OUTPUT_TOKENS),
                Some(MAX_OUTPUT_TOKENS),
            ),
            ("past the limit", Some(200_000), Some(MAX_OUTPUT_TOKENS)),
            ("the largest value", Some(u32::MAX), Some(MAX_OUTPUT_TOKENS)),
        ];
        for (case, max_tokens, expected) in tests {
            let mut body = json!({"messages":[{"role":"user","content":"hi"}]});
            if let Some(max_tokens) = max_tokens {
                body["max_tokens"] = json!(max_tokens);
            }
            let wire = wire(body);
            assert_eq!(
                wire["generation_config"].get("max_output_tokens"),
                expected.map(|value| json!(value)).as_ref(),
                "{case}"
            );
        }
        assert_eq!(MAX_OUTPUT_TOKENS, 65536);
    }

    #[test]
    fn max_tokens_of_zero_is_rejected() {
        let message = error(json!({"max_tokens":0,"messages":[{"role":"user","content":"hi"}]}));
        assert_eq!(message, "max_tokens must be at least 1");
    }

    #[test]
    fn unsupported_requests_are_rejected_with_the_feature_named() {
        let user = json!([{"role":"user","content":"hi"}]);
        let tests = [
            (
                json!({"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}),
                "Google AI Ultra plan does not accept image URLs",
            ),
            (
                json!({"tools":[{"type":"web_search_20250305","name":"web_search"}],"messages":user}),
                "Google AI Ultra plan does not support the Anthropic server tool web_search_20250305",
            ),
            (
                json!({"tools":[{"name":"1bad"}],"messages":user}),
                "is not accepted by Google AI Ultra plan",
            ),
            (
                json!({"stop_sequences":["END"],"messages":user}),
                "stop_sequences is not supported",
            ),
            (
                json!({"tool_choice":{"type":"any"},"messages":user}),
                "tool_choice requires at least one tool",
            ),
            (
                json!({"tools":[{"name":"Read"}],"tool_choice":{"type":"tool","name":"Grep"},"messages":user}),
                "tool_choice names the tool Grep",
            ),
            (
                json!({"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"r"}]}]}),
                "tool_result x does not match any earlier tool_use",
            ),
            (
                json!({"messages":[{"role":"tool","content":"x"}]}),
                "unexpected message role: tool",
            ),
            (
                json!({"messages":[{"role":"user","content":""}]}),
                "at least one non-empty",
            ),
            (
                json!({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Read","input":"x"}]}]}),
                "input must be an object",
            ),
            (
                json!({"output_config":{"effort":"extreme"},"messages":user}),
                "Invalid output_config.effort",
            ),
        ];
        for (body, expected) in tests {
            let message = error(body.clone());
            assert!(message.contains(expected), "{body}: {message}");
        }
        let empty_stops = wire(json!({"stop_sequences":[],"messages":user}));
        assert!(empty_stops.get("stop_sequences").is_none());
    }
}
