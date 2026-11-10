use std::collections::HashSet;
use std::fmt::Write as _;

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::model_allowlist::map_effort;
use super::reasoning_signature::decode_reasoning_signature;
use crate::anthropic::schema::{Message, MessagesRequest};
use crate::providers::translate_shared::{flatten_system_text, parallel_tool_calls, read_effort};
use crate::request_identity::ConversationIdentity;

/// Largest `max_output_tokens` the catalog lists for every Muse model.
pub const MAX_OUTPUT_TOKENS: u32 = 128_000;

/// Asks the upstream to return the opaque reasoning blob, which is what lets
/// a thinking block be replayed on the next turn.
const INCLUDE_ENCRYPTED_REASONING: &str = "reasoning.encrypted_content";

/// Effort sent when the client disables thinking without naming an effort.
/// The upstream rejects `none`, so the lowest accepted value stands in.
const DISABLED_THINKING_EFFORT: &str = "minimal";

/// Body of `POST {base_url}/responses`.
#[derive(Debug, Clone, Serialize)]
pub struct MuseResponsesRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: Vec<MuseInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<MuseTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<MuseToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    pub reasoning: MuseReasoning,
    pub include: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    pub store: bool,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum MuseInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: Vec<MuseContentPart>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<Value>,
        encrypted_content: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum MuseContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct MuseTool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum MuseToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct MuseReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'static str>,
    pub summary: &'static str,
}

/// Translates an Anthropic Messages request into the Meta Responses dialect.
///
/// `model` must already have passed `assert_allowed_model`. The conversation
/// identity, when present, becomes a hashed `prompt_cache_key`; the raw
/// session id never leaves the proxy.
pub fn translate_request(
    req: &MessagesRequest,
    model: &str,
    conversation_identity: Option<&ConversationIdentity>,
) -> anyhow::Result<MuseResponsesRequest> {
    reject_unknown_top_level(req)?;
    let tools = parse_tools(req.extra.get("tools"))?;
    let tool_choice = parse_tool_choice(req.extra.get("tool_choice"), tools.as_deref())?;
    let effort = match read_effort(req)? {
        Some(effort) => Some(map_effort(model, effort)),
        None if thinking_disabled(req) => Some(DISABLED_THINKING_EFFORT),
        None => None,
    };
    let mut input = Vec::new();
    let mut open_calls = HashSet::new();
    let mut seen_calls = HashSet::new();
    for message in &req.messages {
        parse_message(message, &mut input, &mut open_calls, &mut seen_calls)?;
    }
    Ok(MuseResponsesRequest {
        model: model.to_string(),
        instructions: flatten_system_text(req.extra.get("system")),
        input,
        tools,
        tool_choice,
        parallel_tool_calls: parallel_tool_calls(req),
        reasoning: MuseReasoning {
            effort,
            summary: "auto",
        },
        include: vec![INCLUDE_ENCRYPTED_REASONING],
        max_output_tokens: req.max_tokens.map(|tokens| tokens.min(MAX_OUTPUT_TOKENS)),
        prompt_cache_key: conversation_identity.map(prompt_cache_key),
        store: false,
        stream: true,
    })
}

fn prompt_cache_key(identity: &ConversationIdentity) -> String {
    let material = match identity {
        ConversationIdentity::Main(session) => format!("main\0{session}"),
        ConversationIdentity::Agent(session, agent) => format!("agent\0{session}\0{agent}"),
    };
    let digest = Sha256::digest(material.as_bytes());
    let mut key = String::from("ccp:");
    for byte in &digest[..16] {
        let _ = write!(key, "{byte:02x}");
    }
    key
}

fn thinking_disabled(req: &MessagesRequest) -> bool {
    req.extra
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        == Some("disabled")
}

/// Fields the translator either uses or accepts and drops, matching the Grok
/// translator so the two Responses providers treat a request alike.
fn reject_unknown_top_level(req: &MessagesRequest) -> anyhow::Result<()> {
    for key in req.extra.keys() {
        if ![
            "system",
            "tools",
            "tool_choice",
            "context_management",
            "diagnostics",
            "metadata",
            "output_config",
            "thinking",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "service_tier",
        ]
        .contains(&key.as_str())
        {
            anyhow::bail!("unsupported Muse request field: {key}");
        }
    }
    if !valid_diagnostics(req.extra.get("diagnostics")) {
        anyhow::bail!("unsupported diagnostics");
    }
    Ok(())
}

fn valid_diagnostics(value: Option<&Value>) -> bool {
    let Some(value) = value else { return true };
    let Some(object) = value.as_object() else {
        return value.is_null();
    };
    object.keys().all(|key| key == "previous_message_id")
        && object.get("previous_message_id").is_none_or(|id| {
            id.is_null() || id.as_str().is_some_and(|previous| !previous.is_empty())
        })
}

fn parse_tools(value: Option<&Value>) -> anyhow::Result<Option<Vec<MuseTool>>> {
    let Some(value) = value else { return Ok(None) };
    let tools = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tools must be an array"))?;
    let mut names = HashSet::new();
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let object = tool
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("tool must be an object"))?;
        match object.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(kind)) if kind == "custom" => {}
            Some(Value::String(kind)) => {
                anyhow::bail!("Muse does not support Anthropic server tools: {kind}")
            }
            Some(_) => anyhow::bail!("tool type is invalid"),
        }
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| anyhow::anyhow!("tool name is invalid"))?;
        if !names.insert(name) {
            anyhow::bail!("duplicate tool name");
        }
        let parameters = object
            .get("input_schema")
            .filter(|schema| schema.is_object())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("tool input_schema must be an object"))?;
        out.push(MuseTool {
            kind: "function",
            name: name.to_string(),
            description: object
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            parameters,
            strict: false,
        });
    }
    Ok(Some(out))
}

fn parse_tool_choice(
    value: Option<&Value>,
    tools: Option<&[MuseTool]>,
) -> anyhow::Result<Option<MuseToolChoice>> {
    let Some(value) = value else { return Ok(None) };
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("tool_choice must be an object"))?;
    if object
        .get("disable_parallel_tool_use")
        .is_some_and(|value| !value.is_boolean())
    {
        anyhow::bail!("tool_choice disable_parallel_tool_use must be boolean");
    }
    let choice = match object.get("type").and_then(Value::as_str) {
        Some("auto") => MuseToolChoice::Mode("auto"),
        Some("any") => MuseToolChoice::Mode("required"),
        Some("none") => MuseToolChoice::Mode("none"),
        Some("tool") => {
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool_choice name is invalid"))?;
            if !tools.is_some_and(|tools| tools.iter().any(|tool| tool.name == name)) {
                anyhow::bail!("tool_choice references an unknown tool");
            }
            MuseToolChoice::Function {
                kind: "function",
                name: name.to_string(),
            }
        }
        _ => anyhow::bail!("unsupported tool_choice"),
    };
    Ok(Some(choice))
}

fn parse_message(
    message: &Message,
    out: &mut Vec<MuseInputItem>,
    open_calls: &mut HashSet<String>,
    seen_calls: &mut HashSet<String>,
) -> anyhow::Result<()> {
    let role = message.role.as_str();
    if !["system", "user", "assistant"].contains(&role) {
        anyhow::bail!("unsupported message role");
    }
    let text_block;
    let blocks: &[Value] = match &message.content {
        Value::String(text) => {
            text_block = [serde_json::json!({"type": "text", "text": text})];
            &text_block
        }
        Value::Array(blocks) => blocks,
        _ => anyhow::bail!("message content must be text or blocks"),
    };
    let mut content = Vec::new();
    for block in blocks {
        let object = block
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("content block must be an object"))?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("content block type is invalid"))?;
        match (role, kind) {
            (_, "text") => {
                let text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("text block is invalid"))?;
                content.push(if role == "assistant" {
                    MuseContentPart::OutputText { text: text.into() }
                } else {
                    MuseContentPart::InputText { text: text.into() }
                });
            }
            ("assistant", "thinking") => {
                // Only this proxy's own signatures can be replayed. Thinking
                // from another provider, or a signature that no longer
                // decodes, is dropped rather than failing the turn.
                let Some(replay) = object
                    .get("signature")
                    .and_then(Value::as_str)
                    .and_then(decode_reasoning_signature)
                else {
                    continue;
                };
                flush_message(role, &mut content, out);
                out.push(MuseInputItem::Reasoning {
                    id: replay.id,
                    summary: Vec::new(),
                    encrypted_content: replay.encrypted_content,
                });
            }
            (_, "thinking" | "redacted_thinking") => {}
            // Hosted search blocks from another provider's turn cannot be
            // replayed to Muse and carry nothing the model needs.
            ("assistant", "server_tool_use")
            | (_, "web_search_tool_result" | "x_search_tool_result") => {}
            ("user", "image") => content.push(MuseContentPart::InputImage {
                image_url: image_url(object)?,
            }),
            ("assistant", "tool_use") => {
                flush_message(role, &mut content, out);
                out.push(parse_tool_use(object, open_calls, seen_calls)?);
            }
            ("user", "tool_result") => {
                flush_message(role, &mut content, out);
                parse_tool_result(object, out, open_calls)?;
            }
            _ => anyhow::bail!("unsupported content block: {kind}"),
        }
    }
    flush_message(role, &mut content, out);
    Ok(())
}

fn parse_tool_use(
    object: &Map<String, Value>,
    open_calls: &mut HashSet<String>,
    seen_calls: &mut HashSet<String>,
) -> anyhow::Result<MuseInputItem> {
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| anyhow::anyhow!("tool call id is invalid"))?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow::anyhow!("tool call name is invalid"))?;
    let input = object
        .get("input")
        .filter(|input| input.is_object())
        .ok_or_else(|| anyhow::anyhow!("tool call input must be an object"))?;
    if !seen_calls.insert(id.to_string()) {
        anyhow::bail!("duplicate tool call id");
    }
    open_calls.insert(id.to_string());
    Ok(MuseInputItem::FunctionCall {
        call_id: id.to_string(),
        name: name.to_string(),
        arguments: serde_json::to_string(input)?,
    })
}

/// Pushes a `function_call_output` with the result's text. Images in the
/// result cannot travel in the string output, so they follow in a user
/// message of their own.
fn parse_tool_result(
    object: &Map<String, Value>,
    out: &mut Vec<MuseInputItem>,
    open_calls: &mut HashSet<String>,
) -> anyhow::Result<()> {
    let id = object
        .get("tool_use_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| anyhow::anyhow!("tool result id is invalid"))?;
    if !open_calls.remove(id) {
        anyhow::bail!("tool result references an unknown or resolved tool call");
    }
    let mut texts = Vec::new();
    let mut images = Vec::new();
    match object.get("content") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => texts.push(text.clone()),
        Some(Value::Array(parts)) => {
            for part in parts {
                let part = part
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("tool result child must be an object"))?;
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => texts.push(
                        part.get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| anyhow::anyhow!("tool result text is invalid"))?
                            .to_string(),
                    ),
                    Some("image") => images.push(MuseContentPart::InputImage {
                        image_url: image_url(part)?,
                    }),
                    Some("tool_reference") => {}
                    _ => anyhow::bail!("tool result supports text and image children only"),
                }
            }
        }
        Some(_) => anyhow::bail!("tool result content must be text or blocks"),
    }
    out.push(MuseInputItem::FunctionCallOutput {
        call_id: id.to_string(),
        output: texts.join("\n"),
    });
    if !images.is_empty() {
        out.push(MuseInputItem::Message {
            role: "user".into(),
            content: images,
        });
    }
    Ok(())
}

fn image_url(object: &Map<String, Value>) -> anyhow::Result<String> {
    let source = object
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("image source is invalid"))?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .filter(|media_type| !media_type.is_empty())
                .ok_or_else(|| anyhow::anyhow!("image media_type is invalid"))?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .filter(|data| !data.is_empty())
                .ok_or_else(|| anyhow::anyhow!("image data is invalid"))?;
            Ok(format!("data:{media_type};base64,{data}"))
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("image url is invalid")),
        _ => anyhow::bail!("image source is invalid"),
    }
}

fn flush_message(role: &str, content: &mut Vec<MuseContentPart>, out: &mut Vec<MuseInputItem>) {
    if !content.is_empty() {
        out.push(MuseInputItem::Message {
            role: role.to_string(),
            content: std::mem::take(content),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_allowlist::MUSE_MODELS;
    use super::super::reasoning_signature::{ReasoningReplay, encode_reasoning_signature};
    use super::*;
    use serde_json::json;

    // Identifiers and values from the observed two-round tool exchange.
    const CALL_ID: &str = "call_01a1c315c9c87310a50529eaae77c51e";
    const REASONING_ID: &str = "rs_6af274a0aeedd7aeeddd44e4:rs_01a1c31794f476dfb01deceb9b9be238";
    const TOOL_OUTPUT: &str = "{\"temp_c\": 11, \"sky\": \"cloudy\"}";

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid Anthropic request")
    }

    fn translate(value: Value) -> Value {
        let translated =
            translate_request(&request(value), "muse-spark-1.3", None).expect("translates");
        serde_json::to_value(translated).expect("serializes")
    }

    fn weather_tool() -> Value {
        json!({
            "name": "get_weather",
            "description": "Get the weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            }
        })
    }

    fn signature() -> String {
        encode_reasoning_signature(&ReasoningReplay {
            id: REASONING_ID.into(),
            encrypted_content: "gAAAA-observed-length-1426-stand-in".into(),
        })
        .expect("encodable")
    }

    #[test]
    fn every_request_carries_the_fixed_fields() {
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "max_tokens": 256,
            "stream": false,
            "messages": [{"role": "user", "content": "Reply with the single word ok"}]
        }));
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true, "upstream is always streamed");
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["reasoning"], json!({"summary": "auto"}));
        assert_eq!(body["model"], "muse-spark-1.3");
        assert_eq!(body["max_output_tokens"], 256);
        assert_eq!(
            body["input"],
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with the single word ok"}]
            }])
        );
        for absent in [
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "prompt_cache_key",
            "temperature",
            "previous_response_id",
        ] {
            assert!(body.get(absent).is_none(), "{absent} should be absent");
        }
    }

    #[test]
    fn system_becomes_instructions_without_the_billing_header() {
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Answer in English."}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(body["instructions"], "Be brief.\n\nAnswer in English.");
    }

    #[test]
    fn observed_tool_round_trip_matches_the_cli_history_layout() {
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "tools": [weather_tool()],
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": signature()},
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": CALL_ID, "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": CALL_ID, "content": TOOL_OUTPUT}
                ]}
            ]
        }));
        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                },
                "strict": false
            }])
        );
        assert_eq!(
            body["input"],
            json!([
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "What is the weather in Paris?"}
                ]},
                {"type": "reasoning", "id": REASONING_ID, "summary": [],
                 "encrypted_content": "gAAAA-observed-length-1426-stand-in"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Let me check."}
                ]},
                {"type": "function_call", "call_id": CALL_ID, "name": "get_weather",
                 "arguments": "{\"city\":\"Paris\"}"},
                {"type": "function_call_output", "call_id": CALL_ID, "output": TOOL_OUTPUT}
            ])
        );
    }

    #[test]
    fn thinking_without_a_muse_signature_is_dropped_without_failing() {
        let codex = signature().replacen("ccp:muse:v1:", "ccp:codex:v1:", 1);
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "foreign", "signature": "ErUBCkYIBRgCIkA-real"},
                    {"type": "thinking", "thinking": "other proxy", "signature": codex},
                    {"type": "thinking", "thinking": "broken", "signature": "ccp:muse:v1:***:blob"},
                    {"type": "thinking", "thinking": "unsigned"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "hello"}
                ]},
                {"role": "user", "content": "again"}
            ]
        }));
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 3, "{input:?}");
        assert!(input.iter().all(|item| item["type"] == "message"));
        assert_eq!(input[1]["content"][0]["text"], "hello");
    }

    #[test]
    fn image_blocks_become_data_url_input_images() {
        let png = "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAA";
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What colour is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}
            ]}]
        }));
        assert_eq!(
            body["input"][0]["content"],
            json!([
                {"type": "input_text", "text": "What colour is this?"},
                {"type": "input_image", "image_url": format!("data:image/png;base64,{png}")}
            ])
        );
    }

    #[test]
    fn synthetic_tool_result_with_an_image_moves_the_image_to_a_user_message() {
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "tools": [weather_tool()],
            "messages": [
                {"role": "user", "content": "screenshot please"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": [
                        {"type": "text", "text": "captured"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
                        {"type": "text", "text": "done"}
                    ]},
                    {"type": "text", "text": "describe it"}
                ]}
            ]
        }));
        let input = &body["input"];
        assert_eq!(
            input[2],
            json!({"type": "function_call_output", "call_id": "call_1", "output": "captured\ndone"})
        );
        assert_eq!(
            input[3],
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"}
            ]})
        );
        assert_eq!(input[4]["content"][0]["text"], "describe it");
    }

    #[test]
    fn synthetic_tool_choice_variants_map_to_responses_values() {
        let cases = [
            (json!({"type": "auto"}), json!("auto")),
            (json!({"type": "any"}), json!("required")),
            (json!({"type": "none"}), json!("none")),
            (
                json!({"type": "tool", "name": "get_weather"}),
                json!({"type": "function", "name": "get_weather"}),
            ),
        ];
        for (choice, expected) in cases {
            let body = translate(json!({
                "model": "muse-spark-1.3",
                "tools": [weather_tool()],
                "tool_choice": choice,
                "messages": [{"role": "user", "content": "weather?"}]
            }));
            assert_eq!(body["tool_choice"], expected, "{choice}");
        }
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "tools": [weather_tool()],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
            "messages": [{"role": "user", "content": "weather?"}]
        }));
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn synthetic_tool_choice_for_an_unknown_tool_is_rejected() {
        let error = translate_request(
            &request(json!({
                "model": "muse-spark-1.3",
                "tools": [weather_tool()],
                "tool_choice": {"type": "tool", "name": "missing"},
                "messages": [{"role": "user", "content": "x"}]
            })),
            "muse-spark-1.3",
            None,
        )
        .expect_err("unknown tool");
        assert!(error.to_string().contains("unknown tool"), "{error}");
    }

    #[test]
    fn effort_follows_each_model_and_never_sends_none_or_ultra() {
        for model in MUSE_MODELS {
            for effort in ["low", "medium", "high", "xhigh", "max"] {
                let translated = translate_request(
                    &request(json!({
                        "model": model,
                        "output_config": {"effort": effort},
                        "messages": [{"role": "user", "content": "x"}]
                    })),
                    model,
                    None,
                )
                .expect("translates");
                let expected = if *model == "muse-spark-1.2" && effort == "max" {
                    "xhigh"
                } else {
                    effort
                };
                assert_eq!(
                    translated.reasoning.effort,
                    Some(expected),
                    "{model} {effort}"
                );
            }
        }
        for effort in ["none", "ultra", "minimal"] {
            let result = translate_request(
                &request(json!({
                    "model": "muse-spark-1.3",
                    "output_config": {"effort": effort},
                    "messages": [{"role": "user", "content": "x"}]
                })),
                "muse-spark-1.3",
                None,
            );
            assert!(result.is_err(), "{effort} must be rejected");
        }
    }

    #[test]
    fn missing_effort_uses_the_server_default_and_disabled_thinking_uses_minimal() {
        let default = translate(json!({
            "model": "muse-spark-1.3",
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "x"}]
        }));
        assert_eq!(default["reasoning"], json!({"summary": "auto"}));
        let disabled = translate(json!({
            "model": "muse-spark-1.3",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "x"}]
        }));
        assert_eq!(
            disabled["reasoning"],
            json!({"effort": "minimal", "summary": "auto"})
        );
        let disabled_with_effort = translate(json!({
            "model": "muse-spark-1.3",
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": "x"}]
        }));
        assert_eq!(disabled_with_effort["reasoning"]["effort"], "low");
    }

    #[test]
    fn max_tokens_is_capped_at_the_catalog_output_limit() {
        for (requested, expected) in [(16, 16), (128_000, 128_000), (200_000, 128_000)] {
            let body = translate(json!({
                "model": "muse-spark-1.3",
                "max_tokens": requested,
                "messages": [{"role": "user", "content": "x"}]
            }));
            assert_eq!(body["max_output_tokens"], expected, "{requested}");
        }
    }

    #[test]
    fn server_tools_are_rejected_and_custom_tools_are_functions() {
        let error = translate_request(
            &request(json!({
                "model": "muse-spark-1.3",
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": "x"}]
            })),
            "muse-spark-1.3",
            None,
        )
        .expect_err("server tool");
        assert!(error.to_string().contains("server tools"), "{error}");

        let body = translate(json!({
            "model": "muse-spark-1.3",
            "tools": [{"type": "custom", "name": "Bash", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "x"}]
        }));
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "Bash");
    }

    #[test]
    fn accepted_and_dropped_fields_never_reach_the_upstream_body() {
        let body = translate(json!({
            "model": "muse-spark-1.3",
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 5,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "u"},
            "service_tier": "auto",
            "context_management": {"edits": []},
            "diagnostics": {"previous_message_id": "msg_1"},
            "messages": [{"role": "user", "content": "x"}]
        }));
        for dropped in [
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "service_tier",
            "context_management",
            "diagnostics",
        ] {
            assert!(body.get(dropped).is_none(), "{dropped} leaked upstream");
        }
        let error = translate_request(
            &request(json!({
                "model": "muse-spark-1.3",
                "mcp_servers": [],
                "messages": [{"role": "user", "content": "x"}]
            })),
            "muse-spark-1.3",
            None,
        )
        .expect_err("unknown field");
        assert!(error.to_string().contains("mcp_servers"), "{error}");
    }

    #[test]
    fn prompt_cache_key_is_a_hash_of_the_conversation_identity() {
        let req = request(json!({
            "model": "muse-spark-1.3",
            "messages": [{"role": "user", "content": "x"}]
        }));
        let session = "0199a8c4-5b1e-7c3d-9f00-123456789abc";
        let main = ConversationIdentity::Main(session.into());
        let agent = ConversationIdentity::Agent(session.into(), "agent-7".into());
        let main_key = translate_request(&req, "muse-spark-1.3", Some(&main))
            .expect("translates")
            .prompt_cache_key
            .expect("main key");
        let agent_key = translate_request(&req, "muse-spark-1.3", Some(&agent))
            .expect("translates")
            .prompt_cache_key
            .expect("agent key");
        let none = translate_request(&req, "muse-spark-1.3", None).expect("translates");
        assert_eq!(none.prompt_cache_key, None);

        let expected = |material: &str| {
            format!(
                "ccp:{}",
                hex::encode(&Sha256::digest(material.as_bytes())[..16])
            )
        };
        assert_eq!(main_key, expected(&format!("main\0{session}")));
        assert_eq!(agent_key, expected(&format!("agent\0{session}\0agent-7")));
        assert_eq!(main_key.len(), "ccp:".len() + 32);
        assert_ne!(main_key, agent_key);
        assert!(!main_key.contains(session));
    }

    #[test]
    fn synthetic_tool_history_errors_are_reported() {
        let cases = [
            (
                json!([{"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_x", "content": "orphan"}
                ]}]),
                "unknown or resolved",
            ),
            (
                json!([{"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_1", "name": "a", "input": {}},
                    {"type": "tool_use", "id": "call_1", "name": "a", "input": {}}
                ]}]),
                "duplicate tool call id",
            ),
            (
                json!([{"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_1", "name": "a", "input": "not an object"}
                ]}]),
                "must be an object",
            ),
            (
                json!([{"role": "user", "content": [
                    {"type": "document", "source": {"type": "text", "data": "x"}}
                ]}]),
                "unsupported content block: document",
            ),
        ];
        for (messages, expected) in cases {
            let error = translate_request(
                &request(json!({"model": "muse-spark-1.3", "messages": messages})),
                "muse-spark-1.3",
                None,
            )
            .expect_err("invalid history");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}
