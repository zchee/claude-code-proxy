//! Anthropic Messages request → Vertex AI `generateContent` request.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::{Number, Value, json};

use super::thought_signature::{SKIP_THOUGHT_SIGNATURE_VALIDATOR, signature_for_model};
use crate::anthropic::schema::MessagesRequest;
use crate::providers::translate_shared::{
    ContentBlock, ImageSource, flatten_system_text, normalize_content, read_effort,
};
use crate::providers::vertex::models::ResolvedModel;

/// `maxOutputTokens` sent when the request has no `max_tokens`.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 32_000;

/// Largest `maxOutputTokens` sent. The registered Gemini models document
/// 65,536 output tokens per response.
pub const MAX_OUTPUT_TOKENS: u32 = 65_536;

/// Longest function name `FunctionDeclaration` accepts.
const MAX_FUNCTION_NAME_CHARS: usize = 128;

/// The `generateContent` / `streamGenerateContent` request body.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    /// The conversation, alternating `user` and `model` turns.
    pub contents: Vec<Content>,
    /// System prompt, from the request's `system` and `system` / `developer`
    /// messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    /// Function declarations, all in one `Tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Function-calling mode, from `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    /// Sampling, output length and thinking settings.
    pub generation_config: GenerationConfig,
}

impl GenerateContentRequest {
    /// Whether the request asks for thought summaries, which is also whether
    /// the response translation should forward them as thinking blocks.
    pub fn includes_thoughts(&self) -> bool {
        self.generation_config
            .thinking_config
            .as_ref()
            .and_then(|config| config.include_thoughts)
            .unwrap_or(false)
    }
}

/// One turn of the conversation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Content {
    /// `user` or `model`; omitted for the system instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// The turn's parts; never empty.
    pub parts: Vec<Part>,
}

/// The author of a [`Content`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The client (Anthropic `user`).
    User,
    /// The model (Anthropic `assistant`).
    Model,
}

/// One part of a [`Content`]; exactly one data field is set.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// Plain text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Inline media (base64 image).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,
    /// A function call the model made earlier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    /// The result of a function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    /// The thought signature replayed with a function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

/// Base64 media with its MIME type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    /// The media type, for example `image/png`.
    pub mime_type: String,
    /// The base64 payload.
    pub data: String,
}

/// A replayed function call. `id` is not sent: whether a client-chosen ID is
/// accepted is not documented, and responses are matched by order instead.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionCall {
    /// The function name.
    pub name: String,
    /// The arguments object.
    pub args: Value,
}

/// A function result. `response` holds `output`, or `error` for a failed
/// call; media from the result goes in `parts`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionResponse {
    /// The name of the function that was called.
    pub name: String,
    /// `{"output": …}` or `{"error": …}`.
    pub response: Value,
    /// Images the tool returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<FunctionResponsePart>>,
}

/// Media inside a [`FunctionResponse`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponsePart {
    /// The base64 media.
    pub inline_data: Blob,
}

/// A set of function declarations.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// The declared functions.
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// A function the model may call.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    /// The function name.
    pub name: String,
    /// What the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The Anthropic `input_schema`, sent unchanged as JSON Schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_json_schema: Option<Value>,
}

/// Tool-use settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// How the model may call functions.
    pub function_calling_config: FunctionCallingConfig,
}

/// The function-calling mode and the functions it applies to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// The mode.
    pub mode: FunctionCallingMode,
    /// With [`FunctionCallingMode::Any`], the only functions the model may
    /// call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// `FunctionCallingConfig.mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    /// The model decides.
    Auto,
    /// The model must call a function.
    Any,
    /// The model must not call a function.
    None,
}

/// `generationConfig`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Output token limit.
    pub max_output_tokens: u32,
    /// Sequences that stop generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Sampling temperature, as sent by the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Number>,
    /// Nucleus sampling, as sent by the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Number>,
    /// Top-k sampling, as sent by the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<Number>,
    /// Thinking level and summaries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

/// `generationConfig.thinkingConfig`. `thinkingBudget` is never sent: it is
/// deprecated for Gemini 3 and rejected by Gemini 4.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// The thinking level; omitted to use the model's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
    /// Whether thought summaries are returned as `thought: true` parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

/// `thinkingConfig.thinkingLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ThinkingLevel {
    /// Accepted only by models with `supports_minimal`.
    Minimal,
    /// Low.
    Low,
    /// Medium.
    Medium,
    /// High.
    High,
}

/// Translates an Anthropic Messages request for `model`.
///
/// # Errors
///
/// Fails, with a message meant for a 400 `invalid_request_error`, when the
/// request uses something Gemini cannot receive in the documented form: URL
/// images, Anthropic server tools, an invalid function name, a `tool_result`
/// that matches no earlier `tool_use`, an unknown message role, or malformed
/// fields.
pub fn prepare_request(
    req: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<GenerateContentRequest> {
    let mut system = Vec::new();
    if let Some(text) = flatten_system_text(req.extra.get("system")) {
        system.push(text);
    }
    let mut builder = ContentsBuilder::new(&model.upstream_model);
    for message in &req.messages {
        let blocks = normalize_content(&message.content, json!({}));
        match message.role.as_str() {
            "system" | "developer" => {
                let text = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } if !text.is_empty() => Some(text.as_str()),
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

    let declarations = read_tools(req)?;
    let mode = read_tool_choice(req)?;
    if declarations.is_empty()
        && mode
            .as_ref()
            .is_some_and(|config| config.mode != FunctionCallingMode::None)
    {
        anyhow::bail!("tool_choice requires at least one tool");
    }
    let has_tools = !declarations.is_empty();

    Ok(GenerateContentRequest {
        contents,
        system_instruction: (!system.is_empty()).then(|| Content {
            role: None,
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
        tool_config: mode
            .filter(|_| has_tools)
            .map(|function_calling_config| ToolConfig {
                function_calling_config,
            }),
        generation_config: GenerationConfig {
            max_output_tokens: req
                .max_tokens
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
                .min(MAX_OUTPUT_TOKENS),
            stop_sequences: read_stop_sequences(req)?,
            temperature: read_number(req, "temperature")?,
            top_p: read_number(req, "top_p")?,
            top_k: read_number(req, "top_k")?,
            thinking_config: read_thinking(req, model)?,
        },
    })
}

/// Builds `contents`, merging consecutive messages of one role and dropping
/// turns without parts.
struct ContentsBuilder<'a> {
    model: &'a str,
    contents: Vec<Content>,
    /// The user turn being collected; flushed when the model speaks or at the
    /// end, so the ordering rule sees every function result of the turn.
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
                    // The thinking text is not sent; only a signature this
                    // model produced is kept, for the next function call.
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
                    // Gemini signs only the first call of a parallel step, so
                    // a missing signature is replaced only there; a signature
                    // the model did attach goes back on the part it came
                    // with, wherever that is.
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
                    inline_data: Some(inline_image(source)?),
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
                    let rendered = render_tool_result(content)?;
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

    /// Writes the collected user turn. Function responses come first, in the
    /// order of the calls in the preceding model turn (results whose call is
    /// not there keep their relative order after them); text and images
    /// follow. Gemini rejects responses interleaved with other parts.
    fn flush_user(&mut self) {
        let mut turn = std::mem::take(&mut self.user);
        if turn.responses.is_empty() && turn.others.is_empty() {
            return;
        }
        turn.responses
            .sort_by_key(|(position, _)| position.unwrap_or(usize::MAX));
        let parts: Vec<Part> = turn
            .responses
            .into_iter()
            .map(|(_, part)| part)
            .chain(turn.others)
            .collect();
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

struct RenderedToolResult {
    text: String,
    images: Vec<Blob>,
}

fn render_tool_result(content: &Value) -> anyhow::Result<RenderedToolResult> {
    let mut texts = Vec::new();
    let mut images = Vec::new();
    match content {
        Value::Null => {}
        Value::String(text) => texts.push(text.clone()),
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            texts.push(text.to_string());
                        }
                    }
                    Some("image") => {
                        let block = normalize_content(&Value::Array(vec![part.clone()]), json!({}))
                            .into_iter()
                            .next();
                        match block {
                            Some(ContentBlock::Image { source }) => {
                                images.push(inline_image(&source)?);
                            }
                            _ => anyhow::bail!("tool_result image block has no source"),
                        }
                    }
                    Some(kind) => {
                        texts.push(format!("[unsupported tool result block omitted: {kind}]"));
                    }
                    None => texts.push("[unsupported tool result block omitted]".to_string()),
                }
            }
        }
        other => texts.push(other.to_string()),
    }
    Ok(RenderedToolResult {
        text: texts.join("\n"),
        images,
    })
}

fn inline_image(source: &ImageSource) -> anyhow::Result<Blob> {
    if source.source_type == "url" {
        anyhow::bail!("Vertex AI Gemini does not accept image URLs; send the image as base64 data");
    }
    if source.data.is_empty() {
        anyhow::bail!("image block has no data");
    }
    Ok(Blob {
        mime_type: source.media_type.clone(),
        data: source.data.clone(),
    })
}

fn read_tools(req: &MessagesRequest) -> anyhow::Result<Vec<FunctionDeclaration>> {
    let tools = match req.extra.get("tools") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(tools)) => tools,
        Some(_) => anyhow::bail!("tools must be an array"),
    };
    tools
        .iter()
        .map(|tool| {
            let tool = tool
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("each tool must be an object"))?;
            match tool.get("type") {
                None | Some(Value::Null) => {}
                Some(Value::String(kind)) if kind == "custom" => {}
                Some(Value::String(kind)) => anyhow::bail!(
                    "Vertex AI Gemini does not support the Anthropic server tool {kind}"
                ),
                Some(_) => anyhow::bail!("tool type must be a string"),
            }
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool name must be a non-empty string"))?;
            if !is_valid_function_name(name) {
                anyhow::bail!(
                    "tool name {name:?} is not accepted by Vertex AI Gemini: a name starts with \
                     a letter or '_', contains only letters, digits, '_', '.', ':' and '-', and \
                     has at most {MAX_FUNCTION_NAME_CHARS} characters"
                );
            }
            let description = match tool.get("description") {
                None | Some(Value::Null) => None,
                Some(Value::String(description)) => Some(description.clone()),
                Some(_) => anyhow::bail!("tool {name} description must be a string"),
            };
            Ok(FunctionDeclaration {
                name: name.to_string(),
                description,
                parameters_json_schema: tool
                    .get("input_schema")
                    .filter(|schema| !schema.is_null())
                    .cloned(),
            })
        })
        .collect()
}

fn is_valid_function_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    name.len() <= MAX_FUNCTION_NAME_CHARS
        && (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
}

fn read_tool_choice(req: &MessagesRequest) -> anyhow::Result<Option<FunctionCallingConfig>> {
    let choice = match req.extra.get("tool_choice") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(choice)) => choice,
        Some(_) => anyhow::bail!("tool_choice must be an object"),
    };
    // `disable_parallel_tool_use` has no counterpart in FunctionCallingConfig
    // and is ignored.
    let (mode, allowed_function_names) = match choice.get("type").and_then(Value::as_str) {
        Some("auto") => (FunctionCallingMode::Auto, None),
        Some("any") => (FunctionCallingMode::Any, None),
        Some("none") => (FunctionCallingMode::None, None),
        Some("tool") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool_choice.name must be a non-empty string"))?;
            (FunctionCallingMode::Any, Some(vec![name.to_string()]))
        }
        Some(kind) => anyhow::bail!("unsupported tool_choice type: {kind}"),
        None => anyhow::bail!("tool_choice.type must be a string"),
    };
    Ok(Some(FunctionCallingConfig {
        mode,
        allowed_function_names,
    }))
}

fn read_stop_sequences(req: &MessagesRequest) -> anyhow::Result<Option<Vec<String>>> {
    let values = match req.extra.get("stop_sequences") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(values)) => values,
        Some(_) => anyhow::bail!("stop_sequences must be an array of strings"),
    };
    let sequences = values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("stop_sequences must be an array of strings"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((!sequences.is_empty()).then_some(sequences))
}

fn read_number(req: &MessagesRequest, key: &str) -> anyhow::Result<Option<Number>> {
    match req.extra.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => Ok(Some(number.clone())),
        Some(_) => anyhow::bail!("{key} must be a number"),
    }
}

/// Maps Claude Code's effort and thinking settings to `thinkingConfig`:
///
/// | effort | thinking | `thinkingLevel` | `includeThoughts` |
/// |---|---|---|---|
/// | low / medium | any | LOW / MEDIUM | true when thinking is on |
/// | high / xhigh / max | any | HIGH | true when thinking is on |
/// | none | on | model default (omitted) | true |
/// | none | off or absent | MINIMAL, or LOW without `supports_minimal` | omitted |
///
/// Thinking is on for any `thinking.type` other than `disabled`;
/// `budget_tokens` is not used.
fn read_thinking(
    req: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<Option<ThinkingConfig>> {
    let thinking_on = match req.extra.get("thinking") {
        None | Some(Value::Null) => false,
        Some(Value::Object(thinking)) => {
            thinking.get("type").and_then(Value::as_str) != Some("disabled")
        }
        Some(_) => anyhow::bail!("thinking must be an object"),
    };
    let thinking_level = match read_effort(req)? {
        Some("low") => Some(ThinkingLevel::Low),
        Some("medium") => Some(ThinkingLevel::Medium),
        Some(_) => Some(ThinkingLevel::High),
        None if thinking_on => None,
        None if model.supports_minimal => Some(ThinkingLevel::Minimal),
        None => Some(ThinkingLevel::Low),
    };
    let include_thoughts = thinking_on.then_some(true);
    Ok(
        (thinking_level.is_some() || include_thoughts.is_some()).then_some(ThinkingConfig {
            thinking_level,
            include_thoughts,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::vertex::models::resolve;
    use crate::providers::vertex::translate::thought_signature::encode_envelope;

    const MODEL: &str = "gemini-3.5-flash";
    const SIG_A: &str = "c3ludGhldGljLXNpZ25hdHVyZS1h";
    const SIG_B: &str = "c3ludGhldGljLXNpZ25hdHVyZS1i";
    const SIG_C: &str = "c3ludGhldGljLXNpZ25hdHVyZS1j";

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    fn wire(value: Value) -> Value {
        wire_for(MODEL, value)
    }

    fn wire_for(model: &str, value: Value) -> Value {
        let model = resolve(model).unwrap();
        let prepared = prepare_request(&request(value), &model)
            .unwrap_or_else(|error| panic!("request should translate: {error}"));
        serde_json::to_value(prepared).unwrap()
    }

    fn error(value: Value) -> String {
        let model = resolve(MODEL).unwrap();
        prepare_request(&request(value), &model)
            .expect_err("request should be rejected")
            .to_string()
    }

    fn envelope(model: &str, signature: &str) -> String {
        encode_envelope(model, signature).unwrap()
    }

    fn signed_thinking(signature: &str) -> Value {
        json!({"type":"thinking","thinking":"","signature":envelope(MODEL, signature)})
    }

    fn tool_use(id: &str, name: &str) -> Value {
        json!({"type":"tool_use","id":id,"name":name,"input":{"id":id}})
    }

    fn tool_result(id: &str, text: &str) -> Value {
        json!({"type":"tool_result","tool_use_id":id,"content":text})
    }

    fn signatures(content: &Value) -> Vec<Option<String>> {
        content["parts"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|part| part.get("functionCall").is_some())
            .map(|part| part["thoughtSignature"].as_str().map(str::to_string))
            .collect()
    }

    #[test]
    fn minimal_request_has_one_user_content_and_default_limits() {
        let wire = wire(json!({"model":MODEL,"messages":[{"role":"user","content":"hi"}]}));
        assert_eq!(
            wire,
            json!({
                "contents":[{"role":"user","parts":[{"text":"hi"}]}],
                "generationConfig":{
                    "maxOutputTokens":32000,
                    "thinkingConfig":{"thinkingLevel":"MINIMAL"}
                }
            })
        );
    }

    #[test]
    fn system_prompt_and_system_messages_become_the_system_instruction() {
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
            wire["systemInstruction"],
            json!({"parts":[{"text":"first\n\nsecond\n\nthird"}]})
        );
        assert_eq!(wire["contents"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn consecutive_roles_merge_and_empty_turns_are_dropped() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"a"},
            {"role":"user","content":[{"type":"text","text":""},{"type":"text","text":"b"}]},
            {"role":"assistant","content":[{"type":"thinking","thinking":"hidden","signature":"foreign"}]},
            {"role":"user","content":"c"},
            {"role":"assistant","content":"d"},
            {"role":"assistant","content":[{"type":"text","text":"e"}]},
            {"role":"user","content":[]},
            {"role":"assistant","content":""}
        ]}));
        assert_eq!(
            wire["contents"],
            json!([
                {"role":"user","parts":[{"text":"a"},{"text":"b"},{"text":"c"}]},
                {"role":"model","parts":[{"text":"d"},{"text":"e"}]}
            ])
        );
    }

    #[test]
    fn thinking_text_is_never_sent() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"private reasoning","signature":envelope(MODEL, SIG_A)},
                {"type":"text","text":"answer"}
            ]},
            {"role":"user","content":"next"}
        ]}));
        let rendered = wire.to_string();
        assert!(!rendered.contains("private reasoning"), "{rendered}");
        assert!(!rendered.contains(SIG_A), "{rendered}");
        assert_eq!(
            wire["contents"][1],
            json!({"role":"model","parts":[{"text":"answer"}]})
        );
    }

    #[test]
    fn signed_single_call_replays_its_signature() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[signed_thinking(SIG_A), tool_use("t1","Read")]},
            {"role":"user","content":[tool_result("t1","done")]}
        ]}));
        assert_eq!(
            wire["contents"][1],
            json!({"role":"model","parts":[{
                "functionCall":{"name":"Read","args":{"id":"t1"}},
                "thoughtSignature":SIG_A
            }]})
        );
        assert_eq!(
            wire["contents"][2],
            json!({"role":"user","parts":[{
                "functionResponse":{"name":"Read","response":{"output":"done"}}
            }]})
        );
    }

    #[test]
    fn parallel_calls_carry_the_signature_on_the_first_call_only() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[
                signed_thinking(SIG_A),
                tool_use("t1","Read"),
                tool_use("t2","Grep"),
                tool_use("t3","Glob")
            ]},
            {"role":"user","content":[
                tool_result("t1","one"),
                tool_result("t2","two"),
                tool_result("t3","three")
            ]}
        ]}));
        assert_eq!(
            signatures(&wire["contents"][1]),
            vec![Some(SIG_A.to_string()), None, None]
        );
    }

    #[test]
    fn sequential_steps_each_replay_their_own_signature_including_a_later_parallel_step() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[signed_thinking(SIG_A), tool_use("t1","Read")]},
            {"role":"user","content":[tool_result("t1","r1")]},
            {"role":"assistant","content":[signed_thinking(SIG_B), tool_use("t2","Edit")]},
            {"role":"user","content":[tool_result("t2","r2")]},
            {"role":"assistant","content":[
                {"type":"text","text":"running both"},
                signed_thinking(SIG_C),
                tool_use("t3","Bash"),
                tool_use("t4","Bash")
            ]},
            {"role":"user","content":[tool_result("t3","r3"), tool_result("t4","r4")]}
        ]}));
        let contents = wire["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 7);
        assert_eq!(signatures(&contents[1]), vec![Some(SIG_A.to_string())]);
        assert_eq!(signatures(&contents[3]), vec![Some(SIG_B.to_string())]);
        assert_eq!(
            signatures(&contents[5]),
            vec![Some(SIG_C.to_string()), None]
        );
        assert_eq!(contents[5]["parts"][0], json!({"text":"running both"}));
    }

    #[test]
    fn missing_signatures_fall_back_to_the_documented_skip_value() {
        // Thinking blocks stripped (restart, compaction), an empty envelope,
        // a foreign signature and another model's envelope all leave the
        // first call of the step without a usable signature.
        let cases = [
            (
                "stripped",
                vec![tool_use("t1", "Read"), tool_use("t2", "Read")],
            ),
            (
                "empty envelope",
                vec![
                    json!({"type":"thinking","thinking":"x","signature":envelope(MODEL, "")}),
                    tool_use("t1", "Read"),
                    tool_use("t2", "Read"),
                ],
            ),
            (
                "anthropic signature",
                vec![
                    json!({"type":"thinking","thinking":"x","signature":"ErUBCkYIBRgCIkA-real"}),
                    tool_use("t1", "Read"),
                    tool_use("t2", "Read"),
                ],
            ),
            (
                "other model",
                vec![
                    json!({"type":"thinking","thinking":"","signature":envelope("gemini-4-argon", SIG_A)}),
                    tool_use("t1", "Read"),
                    tool_use("t2", "Read"),
                ],
            ),
        ];
        for (name, assistant) in cases {
            let wire = wire(json!({"messages":[
                {"role":"user","content":"q"},
                {"role":"assistant","content":assistant},
                {"role":"user","content":[tool_result("t1","a"), tool_result("t2","b")]}
            ]}));
            assert_eq!(
                signatures(&wire["contents"][1]),
                vec![Some(SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_string()), None],
                "{name}"
            );
            assert!(!wire.to_string().contains(SIG_A), "{name}");
        }
    }

    #[test]
    fn signature_from_the_model_before_a_switch_is_replaced_by_the_skip_value() {
        let history = json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"","signature":envelope("gemini-3.5-flash", SIG_A)},
                tool_use("t1","Read")
            ]},
            {"role":"user","content":[tool_result("t1","a")]}
        ]});
        let same = wire_for("vertex/gemini-3.5-flash", history.clone());
        assert_eq!(
            signatures(&same["contents"][1]),
            vec![Some(SIG_A.to_string())]
        );
        let switched = wire_for("gemini-3.8-flash", history);
        assert_eq!(
            signatures(&switched["contents"][1]),
            vec![Some(SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_string())]
        );
    }

    #[test]
    fn function_responses_follow_call_order_and_precede_text_and_images() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("t1","Read"), tool_use("t2","Grep")]},
            {"role":"user","content":[
                {"type":"text","text":"note first"},
                tool_result("t2","two"),
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0K"}},
                tool_result("t1","one")
            ]}
        ]}));
        assert_eq!(
            wire["contents"][2]["parts"],
            json!([
                {"functionResponse":{"name":"Read","response":{"output":"one"}}},
                {"functionResponse":{"name":"Grep","response":{"output":"two"}}},
                {"text":"note first"},
                {"inlineData":{"mimeType":"image/png","data":"iVBORw0K"}}
            ])
        );
    }

    #[test]
    fn tool_result_names_come_from_earlier_tool_uses_and_unknown_ids_are_rejected() {
        let message = error(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("t1","Read")]},
            {"role":"user","content":[tool_result("t9","orphan")]}
        ]}));
        assert!(message.contains("t9"), "{message}");

        // A result for a call from an older turn still resolves its name and
        // goes after the results that match the latest turn.
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("old","Read")]},
            {"role":"user","content":"keep going"},
            {"role":"assistant","content":[tool_use("new","Grep")]},
            {"role":"user","content":[tool_result("old","late"), tool_result("new","fresh")]}
        ]}));
        assert_eq!(
            wire["contents"][4]["parts"],
            json!([
                {"functionResponse":{"name":"Grep","response":{"output":"fresh"}}},
                {"functionResponse":{"name":"Read","response":{"output":"late"}}}
            ])
        );
    }

    #[test]
    fn error_results_use_the_error_key_and_text_parts_are_joined() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("t1","Bash")]},
            {"role":"user","content":[{
                "type":"tool_result","tool_use_id":"t1","is_error":true,
                "content":[{"type":"text","text":"line 1"},{"type":"text","text":"line 2"},{"type":"document"}]
            }]}
        ]}));
        assert_eq!(
            wire["contents"][2]["parts"][0]["functionResponse"]["response"],
            json!({"error":"line 1\nline 2\n[unsupported tool result block omitted: document]"})
        );
    }

    #[test]
    fn parallel_image_results_keep_each_image_inside_its_own_function_response() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("t1","Read"), tool_use("t2","Read")]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":[
                    {"type":"text","text":"first screenshot"},
                    {"type":"image","source":{"type":"base64","media_type":"image/png","data":"UE5HLW9uZQ=="}}
                ]},
                {"type":"tool_result","tool_use_id":"t2","content":[
                    {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"SlBFRy10d28="}}
                ]}
            ]}
        ]}));
        assert_eq!(
            wire["contents"][2]["parts"],
            json!([
                {"functionResponse":{
                    "name":"Read",
                    "response":{"output":"first screenshot"},
                    "parts":[{"inlineData":{"mimeType":"image/png","data":"UE5HLW9uZQ=="}}]
                }},
                {"functionResponse":{
                    "name":"Read",
                    "response":{"output":""},
                    "parts":[{"inlineData":{"mimeType":"image/jpeg","data":"SlBFRy10d28="}}]
                }}
            ])
        );
    }

    #[test]
    fn url_images_are_rejected_in_messages_and_tool_results() {
        let message = error(json!({"messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"url","url":"https://example.invalid/a.png"}}
        ]}]}));
        assert!(message.contains("image URLs"), "{message}");
        let message = error(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("t1","Read")]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"image","source":{"type":"url","url":"https://example.invalid/a.png"}}
            ]}]}
        ]}));
        assert!(message.contains("image URLs"), "{message}");
    }

    #[test]
    fn tools_become_function_declarations_with_the_schema_unchanged() {
        let schema = json!({
            "type":"object",
            "properties":{"path":{"type":"string","format":"uri"}},
            "required":["path"],
            "additionalProperties":false,
            "$schema":"http://json-schema.org/draft-07/schema#"
        });
        let wire = wire(json!({
            "messages":[{"role":"user","content":"q"}],
            "tools":[
                {"name":"Read","description":"Read a file","input_schema":schema},
                {"type":"custom","name":"mcp__server__do.thing:v2-x","input_schema":{"type":"object"}},
                {"name":"_bare"}
            ]
        }));
        assert_eq!(
            wire["tools"],
            json!([{"functionDeclarations":[
                {"name":"Read","description":"Read a file","parametersJsonSchema":schema},
                {"name":"mcp__server__do.thing:v2-x","parametersJsonSchema":{"type":"object"}},
                {"name":"_bare"}
            ]}])
        );
    }

    #[test]
    fn server_tools_and_invalid_function_names_are_rejected() {
        let message = error(json!({
            "messages":[{"role":"user","content":"q"}],
            "tools":[{"type":"web_search_20250305","name":"web_search"}]
        }));
        assert!(message.contains("web_search_20250305"), "{message}");
        let too_long = "a".repeat(129);
        for name in [
            "1starts-with-digit",
            "has space",
            "slash/name",
            "-dash",
            too_long.as_str(),
        ] {
            let message = error(json!({
                "messages":[{"role":"user","content":"q"}],
                "tools":[{"name":name,"input_schema":{"type":"object"}}]
            }));
            assert!(message.contains(name), "{name}: {message}");
        }
        let longest = "a".repeat(128);
        let wire = wire(json!({
            "messages":[{"role":"user","content":"q"}],
            "tools":[{"name":longest}]
        }));
        assert_eq!(wire["tools"][0]["functionDeclarations"][0]["name"], longest);
    }

    #[test]
    fn tool_choice_maps_to_function_calling_modes() {
        let tests = [
            (json!({"type":"auto"}), json!({"mode":"AUTO"})),
            (json!({"type":"any"}), json!({"mode":"ANY"})),
            (json!({"type":"none"}), json!({"mode":"NONE"})),
            (
                json!({"type":"tool","name":"Read","disable_parallel_tool_use":true}),
                json!({"mode":"ANY","allowedFunctionNames":["Read"]}),
            ),
        ];
        for (choice, expected) in tests {
            let wire = wire(json!({
                "messages":[{"role":"user","content":"q"}],
                "tools":[{"name":"Read","input_schema":{"type":"object"}}],
                "tool_choice":choice
            }));
            assert_eq!(
                wire["toolConfig"],
                json!({"functionCallingConfig":expected}),
                "{choice}"
            );
        }
        let message = error(json!({
            "messages":[{"role":"user","content":"q"}],
            "tool_choice":{"type":"any"}
        }));
        assert!(message.contains("requires at least one tool"), "{message}");
        let wire = wire(json!({
            "messages":[{"role":"user","content":"q"}],
            "tool_choice":{"type":"none"}
        }));
        assert!(wire.get("toolConfig").is_none());
        assert!(wire.get("tools").is_none());
    }

    #[test]
    fn thinking_settings_follow_the_mapping_table() {
        let tests = [
            // effort alone, thinking off: level only.
            (
                MODEL,
                json!({"output_config":{"effort":"low"}}),
                json!({"thinkingLevel":"LOW"}),
            ),
            (
                MODEL,
                json!({"output_config":{"effort":"medium"}}),
                json!({"thinkingLevel":"MEDIUM"}),
            ),
            (
                MODEL,
                json!({"output_config":{"effort":"high"}}),
                json!({"thinkingLevel":"HIGH"}),
            ),
            (
                MODEL,
                json!({"output_config":{"effort":"xhigh"}}),
                json!({"thinkingLevel":"HIGH"}),
            ),
            (
                MODEL,
                json!({"output_config":{"effort":"max"}}),
                json!({"thinkingLevel":"HIGH"}),
            ),
            // effort with thinking on: level plus summaries.
            (
                MODEL,
                json!({"output_config":{"effort":"low"},"thinking":{"type":"adaptive"}}),
                json!({"thinkingLevel":"LOW","includeThoughts":true}),
            ),
            (
                MODEL,
                json!({"output_config":{"effort":"max"},"thinking":{"type":"enabled","budget_tokens":2048}}),
                json!({"thinkingLevel":"HIGH","includeThoughts":true}),
            ),
            // no effort, thinking on: model default level, summaries.
            (
                MODEL,
                json!({"thinking":{"type":"enabled","budget_tokens":4096}}),
                json!({"includeThoughts":true}),
            ),
            (
                MODEL,
                json!({"thinking":{"type":"adaptive"}}),
                json!({"includeThoughts":true}),
            ),
            // no effort, thinking off or absent: the model's smallest level.
            (MODEL, json!({}), json!({"thinkingLevel":"MINIMAL"})),
            (
                MODEL,
                json!({"thinking":{"type":"disabled"}}),
                json!({"thinkingLevel":"MINIMAL"}),
            ),
            ("gemini-4-argon", json!({}), json!({"thinkingLevel":"LOW"})),
            (
                "gemini-4-hydrogen",
                json!({}),
                json!({"thinkingLevel":"LOW"}),
            ),
            (
                "vertex/gemini-4-neon",
                json!({"thinking":{"type":"disabled"}}),
                json!({"thinkingLevel":"LOW"}),
            ),
            (
                "gemini-3.8-flash",
                json!({"thinking":{"type":"disabled"}}),
                json!({"thinkingLevel":"LOW"}),
            ),
            (
                "vertex/gemini-9-unregistered",
                json!({}),
                json!({"thinkingLevel":"LOW"}),
            ),
        ];
        for (model, extra, expected) in tests {
            let mut body = json!({"messages":[{"role":"user","content":"q"}]});
            for (key, value) in extra.as_object().unwrap() {
                body[key] = value.clone();
            }
            let wire = wire_for(model, body);
            assert_eq!(
                wire["generationConfig"]["thinkingConfig"], expected,
                "{model} {extra}"
            );
            assert!(
                !wire.to_string().contains("thinkingBudget"),
                "{model} {extra}"
            );
        }
    }

    #[test]
    fn includes_thoughts_reports_whether_summaries_were_requested() {
        let model = resolve(MODEL).unwrap();
        let on = prepare_request(
            &request(
                json!({"messages":[{"role":"user","content":"q"}],"thinking":{"type":"adaptive"}}),
            ),
            &model,
        )
        .unwrap();
        assert!(on.includes_thoughts());
        let off = prepare_request(
            &request(json!({"messages":[{"role":"user","content":"q"}],"output_config":{"effort":"high"}})),
            &model,
        )
        .unwrap();
        assert!(!off.includes_thoughts());
    }

    #[test]
    fn invalid_effort_is_rejected() {
        let message = error(json!({
            "messages":[{"role":"user","content":"q"}],
            "output_config":{"effort":"extreme"}
        }));
        assert!(message.contains("effort"), "{message}");
    }

    #[test]
    fn max_tokens_defaults_and_is_capped() {
        let tests = [
            (json!(null), 32_000),
            (json!(0), 32_000),
            (json!(1024), 1024),
            (json!(65_536), 65_536),
            (json!(128_000), 65_536),
        ];
        for (max_tokens, expected) in tests {
            let mut body = json!({"messages":[{"role":"user","content":"q"}]});
            if !max_tokens.is_null() {
                body["max_tokens"] = max_tokens.clone();
            }
            let wire = wire(body);
            assert_eq!(
                wire["generationConfig"]["maxOutputTokens"], expected,
                "{max_tokens}"
            );
        }
    }

    #[test]
    fn sampling_fields_pass_through_and_unsupported_fields_are_ignored() {
        let wire = wire(json!({
            "messages":[{"role":"user","content":[
                {"type":"text","text":"q","cache_control":{"type":"ephemeral"}}
            ]}],
            "temperature":0.25,
            "top_p":0.9,
            "top_k":40,
            "stop_sequences":["END","STOP"],
            "metadata":{"user_id":"someone"},
            "service_tier":"auto"
        }));
        let config = &wire["generationConfig"];
        assert_eq!(config["temperature"], json!(0.25));
        assert_eq!(config["topP"], json!(0.9));
        assert_eq!(config["topK"], json!(40));
        assert_eq!(config["stopSequences"], json!(["END", "STOP"]));
        let rendered = wire.to_string();
        for absent in [
            "cache_control",
            "metadata",
            "someone",
            "service_tier",
            "safetySettings",
        ] {
            assert!(!rendered.contains(absent), "{absent}: {rendered}");
        }
    }

    #[test]
    fn malformed_fields_are_rejected() {
        let cases = [
            json!({"messages":[{"role":"tool","content":"x"}]}),
            json!({"messages":[]}),
            json!({"messages":[{"role":"user","content":"q"}],"stop_sequences":[1]}),
            json!({"messages":[{"role":"user","content":"q"}],"temperature":"hot"}),
            json!({"messages":[{"role":"user","content":"q"}],"tools":{"name":"x"}}),
            json!({"messages":[{"role":"user","content":"q"}],"thinking":true}),
            json!({"messages":[
                {"role":"user","content":"q"},
                {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":"text"}]}
            ]}),
            json!({"messages":[
                {"role":"user","content":"q"},
                {"role":"assistant","content":[{"type":"tool_use","id":"","name":"Read","input":{}}]}
            ]}),
        ];
        for case in cases {
            let model = resolve(MODEL).unwrap();
            assert!(
                prepare_request(&request(case.clone()), &model).is_err(),
                "{case}"
            );
        }
    }

    #[test]
    fn no_function_call_carries_an_id() {
        let wire = wire(json!({"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[tool_use("toolu_abc","Read")]},
            {"role":"user","content":[tool_result("toolu_abc","ok")]}
        ]}));
        assert!(!wire.to_string().contains("toolu_abc"), "{wire}");
    }
}
