//! Anthropic Messages request → Cloud Code PA `v1internal` `GenerateContentRequest`.
//!
//! Every request wraps `google.cloud.aiplatform.master.GenerateContentRequest`
//! inside the Cloud Code PA `v1internal.GenerateContentRequest` envelope,
//! matching `agy`'s `ccpa.Codec.EncodeRequest` and `ccpa.ComputeRequestID`.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::{Value, json};

use super::thought_signature::{SKIP_THOUGHT_SIGNATURE_VALIDATOR, signature_for_model};
use crate::anthropic::schema::MessagesRequest;
use crate::providers::google_ai_plan::models::{MAX_OUTPUT_TOKENS, ResolvedModel, ThinkingLevel};
use crate::providers::translate_shared::{
    ContentBlock, flatten_system_text, normalize_content, read_effort,
};
pub use crate::providers::vertex::translate::request::{
    Blob, Content, FunctionCall, FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration,
    FunctionResponse, FunctionResponsePart, Part, Role, Tool, ToolConfig, inline_image,
    read_function_declarations, read_tool_choice, render_tool_result,
};

/// How this provider is named in request errors.
const PROVIDER_LABEL: &str = "Google AI Ultra plan";

/// The Cloud Code PA `v1internal:streamGenerateContent` request envelope.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InteractionRequest {
    /// The Cloud AI Companion project ID (`cloudaicompanionProject`).
    pub project: String,
    /// Per-request ID (`agent/<uuid_v4>`).
    pub request_id: String,
    /// The upstream model ID.
    pub model: String,
    /// Per-prompt UUIDv4.
    pub user_prompt_id: String,
    /// Always `"antigravity"`.
    pub user_agent: &'static str,
    /// Always `"agent"`.
    pub request_type: &'static str,
    /// The inner `google.cloud.aiplatform.master.GenerateContentRequest`.
    pub request: InnerGenerateContentRequest,
    /// Whether the client enabled thinking, which is whether the response
    /// translation forwards thought summaries as thinking text. The body
    /// always asks for summaries, so this is not derived from it. Never sent.
    #[serde(skip)]
    pub emit_thinking: bool,
}

impl InteractionRequest {
    /// Sets the Cloud AI Companion project ID on the request envelope.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = project.into();
        self
    }
}

/// The inner `google.cloud.aiplatform.master.GenerateContentRequest` inside
/// [`InteractionRequest`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InnerGenerateContentRequest {
    /// The conversation, alternating `user` and `model` turns.
    pub contents: Vec<Content>,
    /// System prompt, from the request's `system` and `system` / `developer`
    /// messages. In `agy`, `systemInstruction` has `"role": "user"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Function declarations, all in one `Tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Function-calling mode, from `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Thinking and output length settings.
    pub generation_config: GenerationConfig,
}

/// `request.generationConfig`. No sampling settings (`temperature`, `top_p`,
/// `top_k`) are forwarded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Output token limit, from `max_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Thinking level and summaries.
    pub thinking_config: ThinkingConfig,
}

/// `request.generationConfig.thinkingConfig`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Always `true` so the stream carries thought summaries.
    pub include_thoughts: bool,
    /// The thinking level in uppercase (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`).
    pub thinking_level: WireThinkingLevel,
}

/// `google.cloud.aiplatform.master.GenerationConfig.ThinkingConfig.ThinkingLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WireThinkingLevel {
    /// Minimal.
    Minimal,
    /// Low.
    Low,
    /// Medium.
    Medium,
    /// High.
    High,
}

impl From<ThinkingLevel> for WireThinkingLevel {
    fn from(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal => Self::Minimal,
            ThinkingLevel::Low => Self::Low,
            ThinkingLevel::Medium => Self::Medium,
            ThinkingLevel::High | ThinkingLevel::Xhigh => Self::High,
        }
    }
}

/// Translates an Anthropic Messages request for `model`.
///
/// # Errors
///
/// Fails, with a message meant for a 400 `invalid_request_error`, for what
/// the Cloud Code PA endpoint has no counterpart for or cannot receive: URL
/// images, Anthropic server tools, `stop_sequences`, a function name Gemini
/// does not accept, a `tool_choice` naming a tool the request does not
/// declare, a `tool_result` that matches no earlier `tool_use`, an unknown
/// message role, a conversation without contents, a `max_tokens` of 0, or
/// malformed fields.
pub fn prepare_request(
    body: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<InteractionRequest> {
    let mut system = Vec::new();
    if let Some(text) = request_system_text(body.extra.get("system")) {
        system.push(text);
    }
    let mut builder = ContentsBuilder::new(&model.upstream_model);
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
    let contents = builder.finish();
    if contents.is_empty() {
        anyhow::bail!("messages must contain at least one non-empty user or assistant turn");
    }
    reject_stop_sequences(body)?;
    if body.max_tokens == Some(0) {
        anyhow::bail!("max_tokens must be at least 1");
    }

    let declarations = read_function_declarations(body, PROVIDER_LABEL)?;
    let choice = read_tool_choice(body)?;
    if let Some(config) = choice.as_ref() {
        if declarations.is_empty() && config.mode != FunctionCallingMode::None {
            anyhow::bail!("tool_choice requires at least one tool");
        }
        if let Some(name) = config
            .allowed_function_names
            .as_ref()
            .and_then(|names| names.first())
            && !declarations.iter().any(|decl| &decl.name == name)
        {
            anyhow::bail!("tool_choice names the tool {name}, which tools does not declare");
        }
    }
    let has_tools = !declarations.is_empty();
    let (thinking_level, emit_thinking) = read_thinking(body, model)?;

    Ok(InteractionRequest {
        project: String::new(),
        request_id: format!("agent/{}", uuid::Uuid::new_v4()),
        model: model.upstream_model.clone(),
        user_prompt_id: uuid::Uuid::new_v4().to_string(),
        user_agent: "antigravity",
        request_type: "agent",
        request: InnerGenerateContentRequest {
            contents,
            system_instruction: (!system.is_empty()).then(|| Content {
                role: Some(Role::User),
                parts: vec![Part {
                    text: Some(system.join("\n\n")),
                    ..Part::default()
                }],
            }),
            tools: has_tools.then(|| {
                vec![Tool {
                    function_declarations: declarations,
                }]
            }),
            tool_config: choice
                .filter(|_| has_tools)
                .map(|function_calling_config| ToolConfig {
                    function_calling_config,
                }),
            generation_config: GenerationConfig {
                max_output_tokens: body.max_tokens.map(|value| value.min(MAX_OUTPUT_TOKENS)),
                thinking_config: ThinkingConfig {
                    include_thoughts: true,
                    thinking_level: WireThinkingLevel::from(thinking_level),
                },
            },
        },
        emit_thinking,
    })
}

/// Builds `contents`, merging consecutive messages of one role and dropping
/// turns without parts. Text and images never share a user content with
/// function responses (see `flush_user`).
struct ContentsBuilder<'a> {
    model: &'a str,
    contents: Vec<Content>,
    user: UserTurn,
    /// Function name of every `tool_use` seen so far, by ID.
    tool_names: HashMap<String, String>,
    /// `tool_use` IDs of the latest model turn, in order.
    model_calls: Vec<String>,
    /// A signature from this model, waiting for the next function call.
    held_signature: Option<String>,
}

#[derive(Default)]
struct UserTurn {
    /// Function responses with the position of their call in the latest
    /// model turn (`None` when it is not there).
    responses: Vec<(Option<usize>, Part)>,
    /// Text and images, in order.
    others: Vec<Part>,
}

impl<'a> ContentsBuilder<'a> {
    fn new(model: &'a str) -> Self {
        Self {
            model,
            contents: Vec::new(),
            user: UserTurn::default(),
            tool_names: HashMap::new(),
            model_calls: Vec::new(),
            held_signature: None,
        }
    }

    fn push_model(&mut self, blocks: &[ContentBlock]) -> anyhow::Result<()> {
        self.flush_user();
        for block in blocks {
            match block {
                ContentBlock::Text { text } if !text.is_empty() => {
                    self.push_model_part(Part {
                        text: Some(text.clone()),
                        ..Part::default()
                    });
                }
                ContentBlock::Thinking { signature, .. } => {
                    // Only this model's own `ccp:google-ai-plan:v1:` envelopes
                    // with a non-empty signature are kept for the next call.
                    if let Some(signature) = signature
                        .as_deref()
                        .and_then(|value| signature_for_model(value, self.model))
                    {
                        self.held_signature = Some(signature.to_string());
                    }
                }
                ContentBlock::ToolUse { id, name, input } => {
                    if id.is_empty() || name.is_empty() {
                        anyhow::bail!("assistant tool_use requires non-empty id and name");
                    }
                    if !input.is_object() {
                        anyhow::bail!("assistant tool_use {id} input must be an object");
                    }
                    self.ensure_model_content();
                    let first_call = self.model_calls.is_empty();
                    let thought_signature = self.held_signature.take().or_else(|| {
                        first_call.then(|| SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_string())
                    });
                    self.tool_names.insert(id.clone(), name.clone());
                    self.model_calls.push(id.clone());
                    self.push_model_part(Part {
                        function_call: Some(FunctionCall {
                            name: name.clone(),
                            args: input.clone(),
                        }),
                        thought_signature,
                        ..Part::default()
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
        self.held_signature = None;
        for block in blocks {
            match block {
                ContentBlock::Text { text } if !text.is_empty() => self.user.others.push(Part {
                    text: Some(text.clone()),
                    ..Part::default()
                }),
                ContentBlock::Image { source } => self.user.others.push(Part {
                    inline_data: Some(inline_image(source, PROVIDER_LABEL)?),
                    ..Part::default()
                }),
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
                    let key = if is_error.unwrap_or(false) {
                        "error"
                    } else {
                        "output"
                    };
                    let position = self.model_calls.iter().position(|id| id == tool_use_id);
                    self.user.responses.push((
                        position,
                        Part {
                            function_response: Some(FunctionResponse {
                                name: name.clone(),
                                response: json!({ key: rendered.text }),
                                parts: (!rendered.images.is_empty()).then(|| {
                                    rendered
                                        .images
                                        .into_iter()
                                        .map(|inline_data| FunctionResponsePart { inline_data })
                                        .collect()
                                }),
                            }),
                            ..Part::default()
                        },
                    ));
                }
                ContentBlock::Text { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::ToolUse { .. } => {}
            }
        }
        Ok(())
    }

    fn flush_user(&mut self) {
        let mut turn = std::mem::take(&mut self.user);
        turn.responses
            .sort_by_key(|(position, _)| position.unwrap_or(usize::MAX));
        let responses: Vec<Part> = turn.responses.into_iter().map(|(_, part)| part).collect();
        self.push_user_parts(responses);
        let last_holds_responses = matches!(
            self.contents.last(),
            Some(Content {
                role: Some(Role::User),
                parts,
            }) if parts.iter().any(|part| part.function_response.is_some())
        );
        if last_holds_responses && !turn.others.is_empty() {
            self.contents.push(Content {
                role: Some(Role::User),
                parts: turn.others,
            });
        } else {
            self.push_user_parts(turn.others);
        }
    }

    fn push_user_parts(&mut self, parts: Vec<Part>) {
        if parts.is_empty() {
            return;
        }
        match self.contents.last_mut() {
            Some(Content {
                role: Some(Role::User),
                parts: existing,
            }) => existing.extend(parts),
            _ => self.contents.push(Content {
                role: Some(Role::User),
                parts,
            }),
        }
    }

    fn ensure_model_content(&mut self) {
        if !matches!(
            self.contents.last(),
            Some(Content {
                role: Some(Role::Model),
                ..
            })
        ) {
            self.contents.push(Content {
                role: Some(Role::Model),
                parts: Vec::new(),
            });
            self.model_calls.clear();
        }
    }

    fn push_model_part(&mut self, part: Part) {
        self.ensure_model_content();
        if let Some(content) = self.contents.last_mut() {
            content.parts.push(part);
        }
    }

    fn finish(mut self) -> Vec<Content> {
        self.flush_user();
        self.contents
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
/// then moves it to the nearest level the model accepts.
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
        None if thinking_on => ThinkingLevel::High,
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
            .with_project("test-project")
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

    fn contents(wire: &Value) -> &[Value] {
        wire["request"]["contents"]
            .as_array()
            .expect("contents is an array")
    }

    fn envelope(model: &str, signature: &str) -> String {
        encode_envelope(model, signature).expect("envelope encodes")
    }

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
    fn cloud_code_envelope_with_clamped_level_and_no_sampling_settings() {
        let wire = wire(json!({
            "system":"Be brief.",
            "temperature":0.5,
            "top_p":0.9,
            "top_k":40,
            "max_tokens":1024,
            "tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
            "messages":[{"role":"user","content":"Hi"}]
        }));
        assert_eq!(wire["project"], json!("test-project"));
        assert_eq!(wire["model"], json!("gemini-4-hydrogen"));
        assert_eq!(wire["userAgent"], json!("antigravity"));
        assert_eq!(wire["requestType"], json!("agent"));
        assert!(
            wire["requestId"]
                .as_str()
                .is_some_and(|id| id.starts_with("agent/")),
            "{wire}"
        );
        assert!(
            wire["userPromptId"]
                .as_str()
                .is_some_and(|id| !id.is_empty()),
            "{wire}"
        );
        assert_eq!(
            wire["request"],
            json!({
                "contents":[{"role":"user","parts":[{"text":"Hi"}]}],
                "systemInstruction":{"role":"user","parts":[{"text":"Be brief."}]},
                "tools":[{"functionDeclarations":[{"name":"read_file","description":"Read a file","parametersJsonSchema":{"type":"object","properties":{"path":{"type":"string"}}}}]}],
                "generationConfig":{
                    "maxOutputTokens":1024,
                    "thinkingConfig":{"includeThoughts":true,"thinkingLevel":"LOW"}
                }
            }),
            "a request without thinking asks for the lowest level Hydrogen accepts, mapped to uppercase"
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
    fn replays_the_models_own_signed_thought_on_its_function_call() {
        let wire = wire(replay(
            json!({"type":"thinking","thinking":"Read the file first.","signature":envelope(MODEL, SIG)}),
        ));
        assert_eq!(
            wire["request"]["contents"],
            json!([
                {"role":"user","parts":[{"text":"Read a.py"}]},
                {"role":"model","parts":[{"functionCall":{"name":"read_file","args":{"path":"a.py"}},"thoughtSignature":SIG}]},
                {"role":"user","parts":[{"functionResponse":{"name":"read_file","response":{"output":"x = 1"}}}]}
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
            (
                "an unsigned envelope",
                json!({"type":"thinking","thinking":"Plan.","signature":envelope(MODEL, "")}),
            ),
        ];
        for (case, thinking) in foreign {
            let wire = wire(replay(thinking));
            assert_eq!(
                contents(&wire)[1],
                json!({
                    "role":"model",
                    "parts":[{"functionCall":{"name":"read_file","args":{"path":"a.py"}},"thoughtSignature":"skip_thought_signature_validator"}]
                }),
                "{case}"
            );
            assert!(
                !wire.to_string().contains(SIG),
                "{case}: foreign signature leaked"
            );
        }
    }

    #[test]
    fn one_signed_thought_covers_parallel_calls_of_the_model_turn_and_resets_after_results() {
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
        assert_eq!(
            wire["request"]["contents"],
            json!([
                {"role":"user","parts":[{"text":"q"}]},
                {"role":"model","parts":[
                    {"functionCall":{"name":"Read","args":{}},"thoughtSignature":SIG},
                    {"functionCall":{"name":"Grep","args":{}}}
                ]},
                {"role":"user","parts":[
                    {"functionResponse":{"name":"Read","response":{"output":"1"}}},
                    {"functionResponse":{"name":"Grep","response":{"error":"2"}}}
                ]},
                {"role":"model","parts":[
                    {"text":"again"},
                    {"functionCall":{"name":"Read","args":{}},"thoughtSignature":"skip_thought_signature_validator"}
                ]}
            ])
        );
    }

    #[test]
    fn tool_results_come_before_text_and_carry_images_as_parts() {
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
            contents(&wire)[2..],
            [
                json!({"role":"user","parts":[{
                    "functionResponse":{
                        "name":"Screenshot",
                        "response":{"output":"captured"},
                        "parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw0KGgo="}}]
                    }
                }]}),
                json!({"role":"user","parts":[
                    {"text":"also this"},
                    {"inlineData":{"mimeType":"image/png","data":"iVBORw0KGgo="}}
                ]}),
            ]
        );
    }

    #[test]
    fn consecutive_messages_of_one_role_merge_into_one_content() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"a"},
            {"role":"user","content":"b"},
            {"role":"assistant","content":"c"},
            {"role":"assistant","content":[{"type":"text","text":""},{"type":"text","text":"d"}]}
        ]}));
        assert_eq!(
            wire["request"]["contents"],
            json!([
                {"role":"user","parts":[{"text":"a"},{"text":"b"}]},
                {"role":"model","parts":[{"text":"c"},{"text":"d"}]}
            ])
        );
    }

    #[test]
    fn system_prompt_and_system_messages_join_into_user_role_system_instruction() {
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
            wire["request"]["systemInstruction"],
            json!({"role":"user","parts":[{"text":"first\n\nsecond\n\nthird"}]})
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
            assert_eq!(
                wire["request"].get("systemInstruction"),
                None,
                "{case}: {wire}"
            );
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
                wire["request"]["systemInstruction"],
                json!({"role":"user","parts":[{"text":format!("{expected}\n\nrules")}]}),
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
            ("auto", json!({"type":"auto"}), Some(json!({"mode":"AUTO"}))),
            (
                "any",
                json!({"type":"any","disable_parallel_tool_use":true}),
                Some(json!({"mode":"ANY"})),
            ),
            ("none", json!({"type":"none"}), Some(json!({"mode":"NONE"}))),
            (
                "one tool",
                json!({"type":"tool","name":"Grep"}),
                Some(json!({"mode":"ANY","allowedFunctionNames":["Grep"]})),
            ),
            ("absent", Value::Null, None),
        ];
        for (case, choice, expected) in tests {
            let wire = wire(json!({
                "tools":tools,
                "tool_choice":choice,
                "messages":[{"role":"user","content":"hi"}]
            }));
            assert_eq!(
                wire["request"]
                    .get("toolConfig")
                    .and_then(|tc| tc.get("functionCallingConfig")),
                expected.as_ref(),
                "{case}"
            );
        }
        let without_tools = wire(json!({
            "tool_choice":{"type":"none"},
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(without_tools["request"].get("tools"), None);
        assert_eq!(without_tools["request"].get("toolConfig"), None);
    }

    #[test]
    fn each_models_level_is_clamped_and_mapped_to_uppercase_thinking_level() {
        let tests = [
            // (model, effort, thinking type, wire thinkingLevel, emit_thinking)
            ("gemini-4-hydrogen", None, None, "LOW", false),
            ("gemini-4-hydrogen", None, Some("enabled"), "HIGH", true),
            (
                "gemini-4-hydrogen",
                Some("xhigh"),
                Some("adaptive"),
                "HIGH",
                true,
            ),
            ("gemini-4-hydrogen", Some("max"), None, "HIGH", false),
            ("gemini-4-neon", Some("low"), Some("disabled"), "LOW", false),
            ("gemini-4-neon", Some("max"), Some("enabled"), "HIGH", true),
            ("gemini-4-carbon", Some("xhigh"), None, "HIGH", false),
            ("gemini-4-carbon", Some("max"), None, "HIGH", false),
            ("gemini-4-carbon", None, None, "LOW", false),
            ("gemini-4-carbon", None, Some("enabled"), "HIGH", true),
            ("gemini-4-neon", None, Some("adaptive"), "HIGH", true),
            ("gemini-4-neon", None, None, "LOW", false),
            ("gemini-3.5-flash-lite", None, Some("enabled"), "HIGH", true),
            ("gemini-4-carbon", Some("medium"), None, "MEDIUM", false),
            ("gemini-3.5-flash-lite", None, None, "MINIMAL", false),
            ("gemini-3.5-flash-lite", Some("low"), None, "LOW", false),
            (
                "gemini-3.5-flash-lite",
                Some("xhigh"),
                Some("enabled"),
                "HIGH",
                true,
            ),
            ("gemini-4-argon", Some("xhigh"), None, "HIGH", false),
            ("gemini-4-argon", None, None, "LOW", false),
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
            let serialized = serde_json::to_value(&prepared).expect("serializes");
            assert_eq!(
                serialized["request"]["generationConfig"]["thinkingConfig"],
                json!({"includeThoughts":true,"thinkingLevel":level}),
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
                wire["request"]["generationConfig"].get("maxOutputTokens"),
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
        assert!(empty_stops["request"].get("stopSequences").is_none());
    }
}
