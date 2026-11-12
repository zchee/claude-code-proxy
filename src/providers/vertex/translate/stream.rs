//! Vertex AI `streamGenerateContent?alt=sse` chunks → Anthropic Messages.
//!
//! One [`LiveStreamTranslator`] state drives both the streaming response
//! ([`stream_body`]) and the non-streaming one ([`accumulate_response`]), so
//! the two decide termination the same way.
//!
//! Termination is decided in this order; the first matching row wins:
//!
//! | order | condition | result |
//! |---|---|---|
//! | 1 | a chunk with an `error` key, `promptFeedback.blockReason`, or a `finishReason` other than `STOP` / `MAX_TOKENS` (unknown values included) | error, even when tool calls were already sent |
//! | 2 | the stream ends without a `finishReason` | error |
//! | 3 | `MAX_TOKENS` | `max_tokens`; tool calls sent so far are complete, because each `functionCall` arrives whole in one part |
//! | 4 | `STOP` after at least one `functionCall` | `tool_use` |
//! | 5 | `STOP` without a `functionCall` | `end_turn` |
//!
//! After the `finishReason`, a chunk with content is an error and a
//! usage-only chunk is accepted. The response is final when the upstream
//! closes, or [`POST_TERMINAL_READ_LIMIT`] after the `finishReason` if it
//! does not.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::thought_signature::encode_envelope;
use crate::anthropic::sse::{encode_sse_event, parse_sse_events};
use crate::monitor::{MonitorHandle, usage_from_anthropic_sse};
use crate::providers::codex::native::NativeResponseOutcome;
use crate::providers::grok::translate::stream::SseDecoder;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

/// Largest SSE frame accepted. A `functionCall` arrives whole in one chunk,
/// so a large tool argument (a file being written) makes a large frame.
pub const MAX_SSE_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// How long the upstream may stay open after the `finishReason` before the
/// response is completed without it. The final chunk may be followed by a
/// usage-only chunk; an upstream that neither sends it nor closes must not
/// hold the response open.
pub const POST_TERMINAL_READ_LIMIT: Duration = Duration::from_secs(2);

const SUMMARY_NAME: &str = "061-vertex-stream-summary";

/// Longest upstream enum token (`finishReason`, `blockReason`, `status`) kept
/// in error text.
const MAX_TOKEN_CHARS: usize = 64;

/// Placeholder for an upstream enum token that is not a plain upper-case
/// identifier.
const UNRECOGNIZED_TOKEN: &str = "UNRECOGNIZED";

/// Cleans upstream prose before it reaches a client, the monitor, a log or a
/// traffic capture. The provider supplies it; it removes credentials, long
/// base64 runs and other material the upstream may echo.
pub type Sanitizer = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// What the translator needs to know about the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslatorConfig {
    /// The Anthropic message ID; tool-use IDs are derived from it.
    pub message_id: String,
    /// The model name echoed to the client (as requested).
    pub model: String,
    /// The upstream model ID recorded in thought-signature envelopes; the
    /// next request compares it with its own model.
    pub signature_model: String,
    /// Local input-token estimate shown in `message_start`, and used as the
    /// non-streaming input usage when the upstream reports none.
    pub estimated_input_tokens: u64,
    /// Whether thought summaries are forwarded as thinking text (the client
    /// enabled thinking). Signatures are carried either way.
    pub emit_thinking: bool,
}

/// Why a stream could not be translated into a successful response.
///
/// `Display` contains only proxy-authored words and upper-case enum tokens.
/// Upstream prose is kept in the `upstream_message` and `finish_message`
/// fields and is added, sanitised, only by [`StreamError::message_with`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    /// The stream carried an `error` object.
    #[error("Vertex AI stream reported an error: {status}{}", http_suffix(.code))]
    Upstream {
        /// The HTTP-style code, when it is in 100..=599.
        code: Option<u16>,
        /// The `google.rpc.Code` name, or `UNKNOWN`.
        status: String,
        /// The upstream `error.message`. Unsanitised.
        upstream_message: Option<String>,
    },
    /// `promptFeedback.blockReason` was set.
    #[error("Vertex AI blocked the prompt: blockReason {block_reason}")]
    PromptBlocked {
        /// The block reason token.
        block_reason: String,
        /// The upstream `promptFeedback.blockReasonMessage`. Unsanitised.
        upstream_message: Option<String>,
    },
    /// A `finishReason` other than `STOP` or `MAX_TOKENS`.
    #[error("Vertex AI stopped the response: finishReason {finish_reason}")]
    Stopped {
        /// The finish reason token.
        finish_reason: String,
        /// The upstream `candidates[0].finishMessage`. Unsanitised.
        finish_message: Option<String>,
    },
    /// The upstream closed before any `finishReason`.
    #[error("Vertex AI stream ended without a finishReason")]
    Incomplete,
    /// A chunk with content arrived after the `finishReason`.
    #[error("Vertex AI sent content after the finishReason")]
    ContentAfterFinish,
    /// Two chunks carried different `finishReason` values.
    #[error("Vertex AI changed the finishReason from {first} to {second}")]
    FinishReasonChanged {
        /// The first value.
        first: String,
        /// The later, different value.
        second: String,
    },
    /// Reading the upstream body failed.
    #[error("Vertex AI stream was interrupted")]
    Transport,
    /// The stream broke the documented shape.
    #[error("Vertex AI stream is invalid: {0}")]
    Malformed(&'static str),
}

fn http_suffix(code: &Option<u16>) -> String {
    code.map(|code| format!(" (HTTP {code})"))
        .unwrap_or_default()
}

impl StreamError {
    /// The full message for a client or a log: [`Display`] plus the
    /// upstream prose, passed through `sanitize` first.
    ///
    /// [`Display`]: std::fmt::Display
    pub fn message_with(&self, sanitize: &dyn Fn(&str) -> String) -> String {
        let prose = match self {
            Self::Upstream {
                upstream_message, ..
            }
            | Self::PromptBlocked {
                upstream_message, ..
            } => upstream_message.as_deref(),
            Self::Stopped { finish_message, .. } => finish_message.as_deref(),
            _ => None,
        };
        match prose.map(sanitize).filter(|text| !text.trim().is_empty()) {
            Some(text) => format!("{self}: {text}"),
            None => self.to_string(),
        }
    }

    /// The Anthropic error type for this failure.
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::Upstream { code, status, .. } => match (code, status.as_str()) {
                (Some(429), _) | (_, "RESOURCE_EXHAUSTED") => "rate_limit_error",
                (Some(401), _) | (_, "UNAUTHENTICATED") => "authentication_error",
                (Some(403), _) | (_, "PERMISSION_DENIED") => "permission_error",
                (Some(400 | 404), _)
                | (_, "INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "NOT_FOUND") => {
                    "invalid_request_error"
                }
                _ => "api_error",
            },
            Self::PromptBlocked { .. } => "invalid_request_error",
            _ => "api_error",
        }
    }

    /// A short label for traffic-capture records.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Upstream { .. } => "upstream_error",
            Self::PromptBlocked { .. } => "prompt_blocked",
            Self::Stopped { .. } => "finish_reason",
            Self::Incomplete => "incomplete_stream",
            Self::ContentAfterFinish => "content_after_finish",
            Self::FinishReasonChanged { .. } => "finish_reason_changed",
            Self::Transport => "upstream_stream",
            Self::Malformed(_) => "invalid_event",
        }
    }
}

/// The Anthropic `error` SSE event for `error`, with upstream prose passed
/// through `sanitize`.
pub fn error_event(error: &StreamError, sanitize: &dyn Fn(&str) -> String) -> Vec<u8> {
    let data = json!({
        "type":"error",
        "error":{"type":error.error_type(),"message":error.message_with(sanitize)},
    });
    encode_sse_event(Some("error"), &data.to_string())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Chunk {
    #[serde(default)]
    candidates: Option<Vec<Candidate>>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    finish_message: Option<String>,
    #[serde(default)]
    index: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Option<Vec<ResponsePart>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
    #[serde(default)]
    thought_signature: Option<String>,
    #[serde(default)]
    function_call: Option<ResponseFunctionCall>,
    #[serde(default)]
    inline_data: Option<Value>,
    #[serde(default)]
    file_data: Option<Value>,
    #[serde(default)]
    function_response: Option<Value>,
    #[serde(default)]
    executable_code: Option<Value>,
    #[serde(default)]
    code_execution_result: Option<Value>,
}

impl ResponsePart {
    fn has_content(&self) -> bool {
        self.text.as_deref().is_some_and(|text| !text.is_empty())
            || self.function_call.is_some()
            || self.has_untranslated_data()
    }

    fn has_untranslated_data(&self) -> bool {
        self.inline_data.is_some()
            || self.file_data.is_some()
            || self.function_response.is_some()
            || self.executable_code.is_some()
            || self.code_execution_result.is_some()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResponseFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    args: Option<Value>,
    #[serde(default)]
    partial_args: Option<Value>,
    #[serde(default)]
    will_continue: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: Option<u64>,
    #[serde(default)]
    candidates_token_count: Option<u64>,
    #[serde(default)]
    thoughts_token_count: Option<u64>,
    #[serde(default)]
    cached_content_token_count: Option<u64>,
}

impl UsageMetadata {
    /// `(input, cache_read, output)`. Cached tokens are part of
    /// `promptTokenCount`; thought tokens are counted separately from
    /// `candidatesTokenCount` and are output.
    fn anthropic(self) -> (u64, u64, u64) {
        let cached = self.cached_content_token_count.unwrap_or(0);
        let input = self.prompt_token_count.unwrap_or(0).saturating_sub(cached);
        let output = self
            .candidates_token_count
            .unwrap_or(0)
            .saturating_add(self.thoughts_token_count.unwrap_or(0));
        (input, cached, output)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptFeedback {
    #[serde(default)]
    block_reason: Option<String>,
    #[serde(default)]
    block_reason_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Finish {
    Stop,
    MaxTokens,
}

impl Finish {
    fn token(self) -> &'static str {
        match self {
            Self::Stop => "STOP",
            Self::MaxTokens => "MAX_TOKENS",
        }
    }
}

#[derive(Debug)]
enum Block {
    Thinking {
        text: String,
        signature: String,
    },
    Text {
        text: String,
    },
    Tool {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    Thinking(usize),
    Text(usize),
}

struct TranslationState {
    config: TranslatorConfig,
    message_started: bool,
    finished: bool,
    blocks: Vec<Block>,
    open: Option<Open>,
    tool_count: usize,
    finish: Option<Finish>,
    usage: Option<UsageMetadata>,
}

impl TranslationState {
    fn new(config: TranslatorConfig) -> Self {
        Self {
            config,
            message_started: false,
            finished: false,
            blocks: Vec::new(),
            open: None,
            tool_count: 0,
            finish: None,
            usage: None,
        }
    }

    fn apply_chunk(&mut self, chunk: Chunk) -> Result<Vec<u8>, StreamError> {
        if self.finished {
            return Err(StreamError::Malformed(
                "an event arrived after the response completed",
            ));
        }
        if let Some(feedback) = chunk.prompt_feedback
            && let Some(block_reason) = feedback.block_reason
        {
            return Err(StreamError::PromptBlocked {
                block_reason: upstream_token(&block_reason),
                upstream_message: feedback.block_reason_message,
            });
        }
        if let Some(usage) = chunk.usage_metadata {
            self.usage = Some(usage);
        }
        let mut candidates = chunk.candidates.unwrap_or_default();
        if candidates.len() > 1 {
            return Err(StreamError::Malformed(
                "the response has more than one candidate",
            ));
        }
        let Some(candidate) = candidates.pop() else {
            return Ok(Vec::new());
        };
        if candidate.index.is_some_and(|index| index != 0) {
            return Err(StreamError::Malformed(
                "the response has a candidate other than the first",
            ));
        }
        let parts = candidate
            .content
            .and_then(|content| content.parts)
            .unwrap_or_default();
        if self.finish.is_some() && parts.iter().any(ResponsePart::has_content) {
            return Err(StreamError::ContentAfterFinish);
        }
        let mut out = Vec::new();
        for part in parts {
            self.apply_part(part, &mut out)?;
        }
        if let Some(reason) = candidate.finish_reason {
            let finish = match reason.as_str() {
                "STOP" => Finish::Stop,
                "MAX_TOKENS" => Finish::MaxTokens,
                _ => {
                    return Err(StreamError::Stopped {
                        finish_reason: upstream_token(&reason),
                        finish_message: candidate.finish_message,
                    });
                }
            };
            if let Some(first) = self.finish
                && first != finish
            {
                return Err(StreamError::FinishReasonChanged {
                    first: first.token().to_string(),
                    second: finish.token().to_string(),
                });
            }
            self.finish = Some(finish);
        }
        Ok(out)
    }

    fn apply_part(&mut self, part: ResponsePart, out: &mut Vec<u8>) -> Result<(), StreamError> {
        if part.has_untranslated_data() {
            return Err(StreamError::Malformed(
                "the response has a part type the proxy does not translate",
            ));
        }
        if let Some(call) = part.function_call {
            if call.will_continue == Some(true) || call.partial_args.is_some() {
                return Err(StreamError::Malformed(
                    "the response streams function-call arguments, which the proxy did not request",
                ));
            }
            let name = call
                .name
                .filter(|name| !name.is_empty())
                .ok_or(StreamError::Malformed("a functionCall has no name"))?;
            let input = call.args.unwrap_or_else(|| json!({}));
            if !input.is_object() {
                return Err(StreamError::Malformed("functionCall args is not an object"));
            }
            let signature = part.thought_signature.filter(|value| !value.is_empty());
            self.function_call(name, input, signature.as_deref(), out);
            return Ok(());
        }
        // Signatures on parts other than functionCall are not validated
        // upstream and are dropped.
        let text = part.text.unwrap_or_default();
        if text.is_empty() {
            return Ok(());
        }
        if part.thought == Some(true) {
            if self.config.emit_thinking {
                self.thinking_delta(&text, out);
            }
        } else {
            self.text_delta(&text, out);
        }
        Ok(())
    }

    fn thinking_delta(&mut self, text: &str, out: &mut Vec<u8>) {
        let index = match self.open {
            Some(Open::Thinking(index)) => index,
            _ => {
                self.close_open(out);
                self.start_block(
                    out,
                    Block::Thinking {
                        text: String::new(),
                        signature: String::new(),
                    },
                    json!({"type":"thinking","thinking":"","signature":""}),
                )
            }
        };
        self.open = Some(Open::Thinking(index));
        if let Some(Block::Thinking { text: stored, .. }) = self.blocks.get_mut(index) {
            stored.push_str(text);
        }
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":text}}),
        );
    }

    fn text_delta(&mut self, text: &str, out: &mut Vec<u8>) {
        let index = match self.open {
            Some(Open::Text(index)) => index,
            _ => {
                self.close_open(out);
                self.start_block(
                    out,
                    Block::Text {
                        text: String::new(),
                    },
                    json!({"type":"text","text":""}),
                )
            }
        };
        self.open = Some(Open::Text(index));
        if let Some(Block::Text { text: stored }) = self.blocks.get_mut(index) {
            stored.push_str(text);
        }
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
        );
    }

    /// Emits a complete tool_use block. A signature on the call goes into
    /// the thinking block open just before it, or into a new empty thinking
    /// block, so the client sends it back with the history.
    fn function_call(
        &mut self,
        name: String,
        input: Value,
        signature: Option<&str>,
        out: &mut Vec<u8>,
    ) {
        let envelope = signature
            .and_then(|signature| encode_envelope(&self.config.signature_model, signature));
        match (self.open, envelope) {
            (Some(Open::Thinking(index)), Some(envelope)) => {
                self.open = None;
                self.close_thinking(index, envelope, out);
            }
            (_, Some(envelope)) => {
                self.close_open(out);
                let index = self.start_block(
                    out,
                    Block::Thinking {
                        text: String::new(),
                        signature: String::new(),
                    },
                    json!({"type":"thinking","thinking":"","signature":""}),
                );
                self.close_thinking(index, envelope, out);
            }
            (_, None) => self.close_open(out),
        }
        let id = format!("toolu_{}_{}", self.config.message_id, self.tool_count);
        self.tool_count += 1;
        let arguments = input.to_string();
        let index = self.start_block(
            out,
            Block::Tool {
                id: id.clone(),
                name: name.clone(),
                input,
            },
            json!({"type":"tool_use","id":id,"name":name,"input":{}}),
        );
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":arguments}}),
        );
        emit(
            out,
            "content_block_stop",
            json!({"type":"content_block_stop","index":index}),
        );
    }

    fn start_block(&mut self, out: &mut Vec<u8>, block: Block, content_block: Value) -> usize {
        self.ensure_message_start(out);
        let index = self.blocks.len();
        self.blocks.push(block);
        emit(
            out,
            "content_block_start",
            json!({"type":"content_block_start","index":index,"content_block":content_block}),
        );
        index
    }

    fn close_thinking(&mut self, index: usize, envelope: String, out: &mut Vec<u8>) {
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"signature_delta","signature":envelope}}),
        );
        emit(
            out,
            "content_block_stop",
            json!({"type":"content_block_stop","index":index}),
        );
        if let Some(Block::Thinking { signature, .. }) = self.blocks.get_mut(index) {
            *signature = envelope;
        }
    }

    /// Closes the open thinking or text block. A thinking block closed here
    /// had no signature and gets the empty envelope.
    fn close_open(&mut self, out: &mut Vec<u8>) {
        match self.open.take() {
            Some(Open::Thinking(index)) => {
                let envelope =
                    encode_envelope(&self.config.signature_model, "").unwrap_or_default();
                self.close_thinking(index, envelope, out);
            }
            Some(Open::Text(index)) => emit(
                out,
                "content_block_stop",
                json!({"type":"content_block_stop","index":index}),
            ),
            None => {}
        }
    }

    fn ensure_message_start(&mut self, out: &mut Vec<u8>) {
        if self.message_started {
            return;
        }
        self.message_started = true;
        emit(
            out,
            "message_start",
            json!({"type":"message_start","message":{"id":self.config.message_id,"type":"message","role":"assistant","model":self.config.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":self.config.estimated_input_tokens,"output_tokens":0}}}),
        );
    }

    fn stop_reason(&self) -> Result<&'static str, StreamError> {
        match self.finish {
            None => Err(StreamError::Incomplete),
            Some(Finish::MaxTokens) => Ok("max_tokens"),
            Some(Finish::Stop) if self.tool_count > 0 => Ok("tool_use"),
            Some(Finish::Stop) => Ok("end_turn"),
        }
    }

    fn finalize(&mut self) -> Result<Vec<u8>, StreamError> {
        if self.finished {
            return Ok(Vec::new());
        }
        let stop_reason = self.stop_reason()?;
        let mut out = Vec::new();
        self.close_open(&mut out);
        self.ensure_message_start(&mut out);
        // Input usage is published only when the upstream reported it, so the
        // client keeps the seeded estimate otherwise.
        let usage = match self.usage.map(UsageMetadata::anthropic) {
            Some((input, cache_read, output)) => json!({
                "input_tokens":input,
                "output_tokens":output,
                "cache_creation_input_tokens":0,
                "cache_read_input_tokens":cache_read,
            }),
            None => json!({"output_tokens":0}),
        };
        emit(
            &mut out,
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":stop_reason,"stop_sequence":null},"usage":usage}),
        );
        emit(&mut out, "message_stop", json!({"type":"message_stop"}));
        self.finished = true;
        Ok(out)
    }

    fn response(&self) -> Result<Value, StreamError> {
        if !self.finished {
            return Err(StreamError::Incomplete);
        }
        let content: Vec<Value> = self
            .blocks
            .iter()
            .map(|block| match block {
                Block::Thinking { text, signature } => {
                    json!({"type":"thinking","thinking":text,"signature":signature})
                }
                Block::Text { text } => json!({"type":"text","text":text}),
                Block::Tool { id, name, input } => {
                    json!({"type":"tool_use","id":id,"name":name,"input":input})
                }
            })
            .collect();
        let (input, cache_read, output) = self.usage.map(UsageMetadata::anthropic).unwrap_or((
            self.config.estimated_input_tokens,
            0,
            0,
        ));
        Ok(json!({
            "id":self.config.message_id,
            "type":"message",
            "role":"assistant",
            "model":self.config.model,
            "content":content,
            "stop_reason":self.stop_reason()?,
            "stop_sequence":null,
            "usage":{
                "input_tokens":input,
                "output_tokens":output,
                "cache_creation_input_tokens":0,
                "cache_read_input_tokens":cache_read,
            },
        }))
    }
}

/// Keeps a `google.rpc` / Gemini enum token only when it looks like one.
fn upstream_token(value: &str) -> String {
    let well_formed = (1..=MAX_TOKEN_CHARS).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_');
    if well_formed {
        value.to_string()
    } else {
        UNRECOGNIZED_TOKEN.to_string()
    }
}

fn upstream_error(error: &Value) -> StreamError {
    let code = error
        .get("code")
        .and_then(Value::as_u64)
        .filter(|code| (100..=599).contains(code))
        .and_then(|code| u16::try_from(code).ok());
    let status = error
        .get("status")
        .and_then(Value::as_str)
        .map_or_else(|| "UNKNOWN".to_string(), upstream_token);
    let upstream_message = match error {
        Value::String(message) => Some(message.clone()),
        _ => error
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string),
    };
    StreamError::Upstream {
        code,
        status,
        upstream_message,
    }
}

fn parse_chunk(data: &str) -> Result<Chunk, StreamError> {
    let value: Value = serde_json::from_str(data)
        .map_err(|_| StreamError::Malformed("an event is not valid JSON"))?;
    let value = match value {
        Value::Object(_) => value,
        // An error may also arrive as a one-element array, as in HTTP error
        // bodies.
        Value::Array(items) => {
            return Err(match items.as_slice() {
                [item] => match item.get("error") {
                    Some(error) => upstream_error(error),
                    None => StreamError::Malformed("an event is an array"),
                },
                _ => StreamError::Malformed("an event is an array"),
            });
        }
        _ => return Err(StreamError::Malformed("an event is not a JSON object")),
    };
    if let Some(error) = value.get("error") {
        return Err(upstream_error(error));
    }
    serde_json::from_value(value).map_err(|_| {
        StreamError::Malformed("an event does not have the GenerateContentResponse shape")
    })
}

fn emit(out: &mut Vec<u8>, event: &str, value: Value) {
    out.extend(encode_sse_event(Some(event), &value.to_string()));
}

/// Incremental translator from upstream SSE bytes to Anthropic SSE bytes.
pub struct LiveStreamTranslator {
    decoder: SseDecoder,
    state: TranslationState,
}

impl LiveStreamTranslator {
    /// A translator for one response.
    pub fn new(config: TranslatorConfig) -> Self {
        Self {
            decoder: SseDecoder::with_max_frame_bytes(MAX_SSE_FRAME_BYTES),
            state: TranslationState::new(config),
        }
    }

    /// Feeds upstream bytes (any chunk boundary) and returns the Anthropic
    /// SSE bytes they produce. After an error the translator must not be
    /// used again.
    ///
    /// # Errors
    ///
    /// Any [`StreamError`] except `Incomplete` and `Transport`.
    pub fn push(&mut self, input: &[u8]) -> Result<Vec<u8>, StreamError> {
        let events = self.decoder.push(input).map_err(|_| {
            StreamError::Malformed("the SSE framing is invalid or a frame exceeds 16 MiB")
        })?;
        let mut out = Vec::new();
        for event in events {
            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            out.extend(self.state.apply_chunk(parse_chunk(data)?)?);
        }
        Ok(out)
    }

    /// Completes the response after the upstream closed.
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] when no `finishReason` arrived, and
    /// [`StreamError::Malformed`] when the stream ended inside a frame.
    pub fn finish(&mut self) -> Result<Vec<u8>, StreamError> {
        if self.state.finished {
            return Ok(Vec::new());
        }
        if self.state.finish.is_none() {
            return Err(StreamError::Incomplete);
        }
        self.decoder
            .finish()
            .map_err(|_| StreamError::Malformed("the stream ended inside an SSE frame"))?;
        self.state.finalize()
    }

    /// Completes the response when the upstream stayed open for
    /// [`POST_TERMINAL_READ_LIMIT`] after the `finishReason`. A partial frame
    /// still buffered is discarded.
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] when called before a `finishReason`.
    pub fn finish_after_read_limit(&mut self) -> Result<Vec<u8>, StreamError> {
        self.state.finalize()
    }

    /// Whether a `finishReason` has arrived; from then on the caller reads
    /// for at most [`POST_TERMINAL_READ_LIMIT`].
    pub fn is_terminal(&self) -> bool {
        self.state.finish.is_some()
    }

    /// Whether the response has been completed.
    pub fn is_finished(&self) -> bool {
        self.state.finished
    }

    /// The non-streaming Anthropic message, once the response is completed.
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] before completion.
    pub fn response(&self) -> Result<Value, StreamError> {
        self.state.response()
    }
}

/// Translates a whole buffered upstream body into a non-streaming Anthropic
/// message.
///
/// # Errors
///
/// The same conditions that fail the streaming response.
pub fn accumulate_response(input: &[u8], config: TranslatorConfig) -> Result<Value, StreamError> {
    let mut translator = LiveStreamTranslator::new(config);
    translator.push(input)?;
    translator.finish()?;
    translator.response()
}

/// Where the streaming response reports progress, captures and failures.
pub struct StreamSinks {
    /// The proxy request ID.
    pub req_id: String,
    /// Monitor for progress and token counts.
    pub monitor: Option<MonitorHandle>,
    /// Traffic capture for upstream and downstream events.
    pub traffic: Option<Arc<TrafficCapture>>,
    /// Records a failed stream so the server logs the request as failed with
    /// the message the client saw.
    pub outcome: Option<NativeResponseOutcome>,
    /// Applied to upstream prose before it reaches the client, the outcome
    /// or the capture.
    pub sanitize: Sanitizer,
}

/// The Anthropic SSE body for a streaming client.
///
/// The upstream is read until it closes, or for at most
/// [`POST_TERMINAL_READ_LIMIT`] after the `finishReason`. A failure ends the
/// body with one `error` event whose message went through
/// [`StreamSinks::sanitize`].
pub fn stream_body<S, E>(upstream: S, config: TranslatorConfig, sinks: StreamSinks) -> Body
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let state = VertexStreamState {
        upstream: Box::pin(upstream),
        translator: LiveStreamTranslator::new(config),
        capture_decoder: SseDecoder::with_max_frame_bytes(MAX_SSE_FRAME_BYTES),
        read_deadline: None,
        done: false,
        bytes: 0,
        chunks: 0,
        stream_capture: sinks
            .traffic
            .as_ref()
            .map(|traffic| traffic.stream_capture()),
        sinks,
    };
    let stream = futures_util::stream::unfold(state, |mut state| async move {
        state
            .next_output()
            .await
            .map(|bytes| (Ok::<Bytes, Infallible>(Bytes::from(bytes)), state))
    });
    Body::from_stream(stream)
}

type UpstreamStream<E> = Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send>>;

struct VertexStreamState<E> {
    upstream: UpstreamStream<E>,
    translator: LiveStreamTranslator,
    capture_decoder: SseDecoder,
    read_deadline: Option<tokio::time::Instant>,
    done: bool,
    bytes: u64,
    chunks: u64,
    stream_capture: Option<StreamTrafficCapture>,
    sinks: StreamSinks,
}

impl<E> VertexStreamState<E> {
    async fn next_output(&mut self) -> Option<Vec<u8>> {
        if self.done {
            return None;
        }
        loop {
            let next = if self.translator.is_terminal() {
                let deadline = *self
                    .read_deadline
                    .get_or_insert_with(|| tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT);
                match tokio::time::timeout_at(deadline, self.upstream.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        let result = self.translator.finish_after_read_limit();
                        return self.complete(result);
                    }
                }
            } else {
                self.upstream.next().await
            };
            let chunk = match next {
                Some(Ok(chunk)) => chunk,
                Some(Err(_)) => return Some(self.fail(&StreamError::Transport)),
                None => {
                    let result = self.translator.finish();
                    return self.complete(result);
                }
            };
            if self.bytes == 0
                && let Some(monitor) = self.sinks.monitor.as_ref()
            {
                monitor.generation_started(&self.sinks.req_id);
            }
            self.bytes = self.bytes.saturating_add(chunk.len() as u64);
            self.chunks = self.chunks.saturating_add(1);
            self.capture_upstream(&chunk);
            match self.translator.push(&chunk) {
                Ok(output) if output.is_empty() => {}
                Ok(output) => {
                    self.report_progress(&output);
                    self.capture_downstream(&output);
                    return Some(output);
                }
                Err(error) => return Some(self.fail(&error)),
            }
        }
    }

    fn complete(&mut self, result: Result<Vec<u8>, StreamError>) -> Option<Vec<u8>> {
        match result {
            Ok(output) => {
                self.done = true;
                if !output.is_empty() {
                    self.report_progress(&output);
                    self.capture_downstream(&output);
                }
                self.finish_capture(true);
                (!output.is_empty()).then_some(output)
            }
            Err(error) => Some(self.fail(&error)),
        }
    }

    fn fail(&mut self, error: &StreamError) -> Vec<u8> {
        self.done = true;
        let sanitize = self.sinks.sanitize.clone();
        if let Some(outcome) = self.sinks.outcome.as_ref() {
            outcome.fail(error.message_with(&*sanitize));
        }
        let output = error_event(error, &*sanitize);
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.malformed("translation", error.kind());
        }
        self.capture_downstream(&output);
        self.finish_capture(false);
        output
    }

    fn report_progress(&self, output: &[u8]) {
        if let Some(monitor) = self.sinks.monitor.as_ref() {
            let (input_tokens, output_tokens) = usage_from_anthropic_sse(output);
            monitor.stream_progress(
                &self.sinks.req_id,
                output.len() as u64,
                parse_sse_events(output).len() as u64,
                input_tokens,
                output_tokens,
            );
        }
    }

    fn capture_upstream(&mut self, bytes: &[u8]) {
        if self.stream_capture.is_none() {
            return;
        }
        let events = match self.capture_decoder.push(bytes) {
            Ok(events) => events,
            Err(_) => {
                if let Some(capture) = self.stream_capture.as_mut() {
                    capture.malformed("decoder", "malformed_sse");
                }
                return;
            }
        };
        let sanitize = self.sinks.sanitize.clone();
        let Some(capture) = self.stream_capture.as_mut() else {
            return;
        };
        for event in events {
            match serde_json::from_str::<Value>(&event.data) {
                Ok(value) => {
                    capture.upstream_event(
                        event.event.as_deref(),
                        &sanitize_capture(&value, &*sanitize),
                    );
                }
                Err(_) => capture.malformed("json", "malformed_event"),
            }
        }
    }

    fn capture_downstream(&mut self, bytes: &[u8]) {
        let Some(capture) = self.stream_capture.as_mut() else {
            return;
        };
        for event in parse_sse_events(bytes) {
            if let Ok(value) = serde_json::from_str(&event.data) {
                capture.downstream_event(event.event.as_deref().unwrap_or("message"), value);
            }
        }
    }

    fn finish_capture(&mut self, completed: bool) {
        if let (Some(capture), Some(traffic)) =
            (self.stream_capture.take(), self.sinks.traffic.as_ref())
        {
            capture.finish_named(
                traffic,
                json!({
                    "kind":if completed { "stream_completion" } else { "stream_error" },
                    "bytes":self.bytes,
                    "chunks":self.chunks,
                }),
                SUMMARY_NAME,
            );
        }
    }
}

impl<E> Drop for VertexStreamState<E> {
    fn drop(&mut self) {
        if self.done || self.stream_capture.is_none() {
            return;
        }
        if let (Some(capture), Some(traffic)) =
            (self.stream_capture.take(), self.sinks.traffic.as_ref())
        {
            capture.finish_named(
                traffic,
                json!({
                    "kind":"stream_abandoned",
                    "reason":"downstream_body_dropped",
                    "bytes":self.bytes,
                    "chunks":self.chunks,
                }),
                SUMMARY_NAME,
            );
        }
    }
}

/// Prepares an upstream event for a traffic capture: the fields that carry
/// upstream prose (`error.message`, `candidates[].finishMessage`,
/// `promptFeedback.blockReasonMessage`) go through `sanitize`, and an error
/// object keeps only `code`, `status` and the sanitised `message`.
/// Key-based redaction in the capture cannot see material embedded in prose.
pub fn sanitize_capture(value: &Value, sanitize: &dyn Fn(&str) -> String) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| sanitize_capture(item, sanitize))
                .collect(),
        ),
        Value::Object(object) => {
            let mut object = object.clone();
            if let Some(error) = object.get_mut("error") {
                *error = sanitize_error(error, sanitize);
            }
            if let Some(Value::Array(candidates)) = object.get_mut("candidates") {
                for candidate in candidates {
                    if let Some(Value::String(message)) = candidate.get_mut("finishMessage") {
                        *message = sanitize(message);
                    }
                }
            }
            if let Some(Value::String(message)) = object
                .get_mut("promptFeedback")
                .and_then(|feedback| feedback.get_mut("blockReasonMessage"))
            {
                *message = sanitize(message);
            }
            Value::Object(object)
        }
        other => other.clone(),
    }
}

fn sanitize_error(error: &Value, sanitize: &dyn Fn(&str) -> String) -> Value {
    match error {
        Value::String(message) => Value::String(sanitize(message)),
        Value::Object(object) => {
            let mut out = Map::new();
            if let Some(code) = object.get("code").filter(|code| code.is_number()) {
                out.insert("code".into(), code.clone());
            }
            if let Some(status) = object.get("status").and_then(Value::as_str) {
                out.insert("status".into(), Value::String(upstream_token(status)));
            }
            if let Some(message) = object.get("message").and_then(Value::as_str) {
                out.insert("message".into(), Value::String(sanitize(message)));
            }
            Value::Object(out)
        }
        _ => Value::String("[redacted]".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::vertex::translate::thought_signature::{
        ENVELOPE_PREFIX, signature_for_model,
    };

    const MODEL: &str = "gemini-3.5-flash";
    const SIG_A: &str = "c3ludGhldGljLXNpZ25hdHVyZS1h";
    const SIG_B: &str = "c3ludGhldGljLXNpZ25hdHVyZS1i";

    fn config(emit_thinking: bool) -> TranslatorConfig {
        TranslatorConfig {
            message_id: "msg_1".to_string(),
            model: format!("vertex/{MODEL}"),
            signature_model: MODEL.to_string(),
            estimated_input_tokens: 42,
            emit_thinking,
        }
    }

    fn frame(data: &str) -> String {
        format!("data: {data}\n\n")
    }

    fn frames(chunks: &[&str]) -> String {
        chunks.iter().map(|chunk| frame(chunk)).collect()
    }

    fn text_chunk(text: &str) -> String {
        json!({"candidates":[{"content":{"role":"model","parts":[{"text":text}]}}]}).to_string()
    }

    fn call_chunk(name: &str, signature: Option<&str>) -> String {
        let mut part = json!({"functionCall":{"name":name,"args":{"path":"src/main.rs"}}});
        if let Some(signature) = signature {
            part["thoughtSignature"] = json!(signature);
        }
        json!({"candidates":[{"content":{"role":"model","parts":[part]}}]}).to_string()
    }

    fn finish_chunk(reason: &str) -> String {
        json!({
            "candidates":[{"content":{"role":"model","parts":[]},"finishReason":reason}],
            "usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"thoughtsTokenCount":5,"cachedContentTokenCount":30,"totalTokenCount":125},
            "modelVersion":MODEL
        })
        .to_string()
    }

    /// Runs the live translator over `input` and returns the events, or the
    /// first error.
    fn live(input: &str, emit_thinking: bool) -> Result<Vec<(String, Value)>, StreamError> {
        let mut translator = LiveStreamTranslator::new(config(emit_thinking));
        let mut out = translator.push(input.as_bytes())?;
        out.extend(translator.finish()?);
        Ok(parse_sse_events(&out)
            .into_iter()
            .map(|event| {
                (
                    event.event.unwrap_or_default(),
                    serde_json::from_str(&event.data).unwrap(),
                )
            })
            .collect())
    }

    fn accumulated(input: &str, emit_thinking: bool) -> Result<Value, StreamError> {
        accumulate_response(input.as_bytes(), config(emit_thinking))
    }

    /// Rebuilds the content and stop reason a client would assemble from the
    /// live events.
    fn rebuild(events: &[(String, Value)]) -> (Vec<Value>, Value) {
        let mut content: Vec<Value> = Vec::new();
        let mut partial: Vec<String> = Vec::new();
        let mut stop_reason = Value::Null;
        for (name, data) in events {
            match name.as_str() {
                "content_block_start" => {
                    content.push(data["content_block"].clone());
                    partial.push(String::new());
                }
                "content_block_delta" => {
                    let index = data["index"].as_u64().unwrap() as usize;
                    let delta = &data["delta"];
                    let block = &mut content[index];
                    match delta["type"].as_str().unwrap() {
                        "text_delta" => {
                            let text = block["text"].as_str().unwrap().to_string()
                                + delta["text"].as_str().unwrap();
                            block["text"] = json!(text);
                        }
                        "thinking_delta" => {
                            let text = block["thinking"].as_str().unwrap().to_string()
                                + delta["thinking"].as_str().unwrap();
                            block["thinking"] = json!(text);
                        }
                        "signature_delta" => block["signature"] = delta["signature"].clone(),
                        "input_json_delta" => {
                            partial[index].push_str(delta["partial_json"].as_str().unwrap());
                        }
                        other => panic!("unexpected delta {other}"),
                    }
                }
                "content_block_stop" => {
                    let index = data["index"].as_u64().unwrap() as usize;
                    if content[index]["type"] == "tool_use" {
                        content[index]["input"] = serde_json::from_str(&partial[index]).unwrap();
                    }
                }
                "message_delta" => stop_reason = data["delta"]["stop_reason"].clone(),
                _ => {}
            }
        }
        (content, stop_reason)
    }

    /// Asserts that the live and accumulated translations agree and returns
    /// the accumulated message.
    fn both_succeed(input: &str, emit_thinking: bool) -> Value {
        let events = live(input, emit_thinking).unwrap_or_else(|error| panic!("live: {error}"));
        let message = accumulated(input, emit_thinking)
            .unwrap_or_else(|error| panic!("accumulated: {error}"));
        let (content, stop_reason) = rebuild(&events);
        assert_eq!(json!(content), message["content"], "content differs");
        assert_eq!(stop_reason, message["stop_reason"], "stop_reason differs");
        assert_eq!(events.first().unwrap().0, "message_start");
        assert_eq!(events.last().unwrap().0, "message_stop");
        message
    }

    fn both_fail(input: &str) -> StreamError {
        let live_error = live(input, true).expect_err("live should fail");
        let accumulated_error = accumulated(input, true).expect_err("accumulated should fail");
        assert_eq!(live_error, accumulated_error);
        live_error
    }

    #[test]
    fn synthetic_text_only_stream_ends_the_turn_with_mapped_usage() {
        let input = frames(&[&text_chunk("Hel"), &text_chunk("lo"), &finish_chunk("STOP")]);
        let message = both_succeed(&input, false);
        assert_eq!(message["content"], json!([{"type":"text","text":"Hello"}]));
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(message["model"], "vertex/gemini-3.5-flash");
        // input = prompt - cached, cache read = cached, output = candidates +
        // thoughts.
        assert_eq!(
            message["usage"],
            json!({"input_tokens":70,"output_tokens":25,"cache_creation_input_tokens":0,"cache_read_input_tokens":30})
        );
        let events = live(&input, false).unwrap();
        assert_eq!(events[0].1["message"]["usage"]["input_tokens"], 42);
        let delta = events
            .iter()
            .find(|(name, _)| name == "message_delta")
            .unwrap();
        assert_eq!(delta.1["usage"]["input_tokens"], 70);
        assert_eq!(delta.1["usage"]["output_tokens"], 25);
        assert_eq!(delta.1["usage"]["cache_read_input_tokens"], 30);
    }

    #[test]
    fn synthetic_split_at_every_byte_boundary_gives_the_same_output() {
        let input = frames(&[
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"think","thought":true}]}}]}).to_string(),
            &text_chunk("answer"),
            &call_chunk("Read", Some(SIG_A)),
            &finish_chunk("STOP"),
        ]);
        let mut whole = LiveStreamTranslator::new(config(true));
        let mut expected = whole.push(input.as_bytes()).unwrap();
        expected.extend(whole.finish().unwrap());
        for split in 0..=input.len() {
            let mut translator = LiveStreamTranslator::new(config(true));
            let mut out = translator.push(&input.as_bytes()[..split]).unwrap();
            out.extend(translator.push(&input.as_bytes()[split..]).unwrap());
            out.extend(translator.finish().unwrap());
            assert_eq!(out, expected, "split at {split}");
        }
    }

    #[test]
    fn synthetic_thought_summary_then_text_becomes_thinking_and_text() {
        let input = frames(&[
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"**Planning**","thought":true}]}}]}).to_string(),
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":" the answer","thought":true}]}}]}).to_string(),
            &text_chunk("42"),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, true);
        let empty_envelope = format!("{ENVELOPE_PREFIX}{MODEL}:");
        assert_eq!(
            message["content"],
            json!([
                {"type":"thinking","thinking":"**Planning** the answer","signature":empty_envelope},
                {"type":"text","text":"42"}
            ])
        );
        assert_eq!(message["stop_reason"], "end_turn");
    }

    #[test]
    fn synthetic_thought_summary_is_dropped_when_thinking_was_not_requested() {
        let input = frames(&[
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"hidden","thought":true}]}}]}).to_string(),
            &text_chunk("42"),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, false);
        assert_eq!(message["content"], json!([{"type":"text","text":"42"}]));
        assert!(!message.to_string().contains("hidden"));
    }

    #[test]
    fn synthetic_signed_call_without_thinking_gets_an_empty_signed_thinking_block() {
        let input = frames(&[&call_chunk("Read", Some(SIG_A)), &finish_chunk("STOP")]);
        let message = both_succeed(&input, false);
        let content = message["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "");
        let signature = content[0]["signature"].as_str().unwrap();
        assert_eq!(signature_for_model(signature, MODEL), Some(SIG_A));
        assert_eq!(
            content[1],
            json!({"type":"tool_use","id":"toolu_msg_1_0","name":"Read","input":{"path":"src/main.rs"}})
        );
        assert_eq!(message["stop_reason"], "tool_use");
    }

    #[test]
    fn synthetic_signature_goes_into_the_thinking_block_open_before_the_call() {
        let input = frames(&[
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"I should read it","thought":true}]}}]}).to_string(),
            &call_chunk("Read", Some(SIG_A)),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, true);
        let content = message["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["thinking"], "I should read it");
        assert_eq!(
            signature_for_model(content[0]["signature"].as_str().unwrap(), MODEL),
            Some(SIG_A)
        );
        assert_eq!(content[1]["type"], "tool_use");
    }

    #[test]
    fn synthetic_parallel_calls_sign_only_the_first_and_number_the_ids() {
        // Gemini attaches the signature to the first functionCall part of a
        // parallel step only.
        let input = frames(&[
            &json!({"candidates":[{"content":{"role":"model","parts":[
                {"functionCall":{"name":"Read","args":{"path":"a"}},"thoughtSignature":SIG_A},
                {"functionCall":{"name":"Read","args":{"path":"b"}}}
            ]}}]})
            .to_string(),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, true);
        let content = message["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[1]["id"], "toolu_msg_1_0");
        assert_eq!(content[1]["input"], json!({"path":"a"}));
        assert_eq!(content[2]["id"], "toolu_msg_1_1");
        assert_eq!(content[2]["input"], json!({"path":"b"}));
        assert_eq!(message["stop_reason"], "tool_use");
    }

    #[test]
    fn synthetic_sequential_signed_calls_in_one_response_each_get_a_thinking_block() {
        let input = frames(&[
            &call_chunk("Read", Some(SIG_A)),
            &text_chunk("then"),
            &call_chunk("Edit", Some(SIG_B)),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, false);
        let content = message["content"].as_array().unwrap();
        let types: Vec<&str> = content
            .iter()
            .map(|block| block["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            types,
            ["thinking", "tool_use", "text", "thinking", "tool_use"]
        );
        assert_eq!(
            signature_for_model(content[0]["signature"].as_str().unwrap(), MODEL),
            Some(SIG_A)
        );
        assert_eq!(
            signature_for_model(content[3]["signature"].as_str().unwrap(), MODEL),
            Some(SIG_B)
        );
    }

    #[test]
    fn synthetic_signatures_on_text_parts_are_dropped() {
        let input = frames(&[
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"hi","thoughtSignature":SIG_A}]}}]}).to_string(),
            &json!({"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":SIG_B}]}}]}).to_string(),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(message["content"], json!([{"type":"text","text":"hi"}]));
        let rendered = message.to_string();
        assert!(
            !rendered.contains(SIG_A) && !rendered.contains(SIG_B),
            "{rendered}"
        );
    }

    #[test]
    fn synthetic_usage_only_chunk_after_the_finish_reason_is_accepted() {
        let input = frames(&[
            &text_chunk("done"),
            &json!({"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}).to_string(),
            &json!({"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3}}).to_string(),
            &json!({"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":4}}).to_string(),
        ]);
        let message = both_succeed(&input, false);
        assert_eq!(message["usage"]["input_tokens"], 11);
        assert_eq!(message["usage"]["output_tokens"], 4);
    }

    #[test]
    fn synthetic_missing_usage_keeps_the_estimate() {
        let input = frames(&[
            &text_chunk("done"),
            &json!({"candidates":[{"finishReason":"STOP"}]}).to_string(),
        ]);
        let message = both_succeed(&input, false);
        assert_eq!(message["usage"]["input_tokens"], 42);
        assert_eq!(message["usage"]["output_tokens"], 0);
        let events = live(&input, false).unwrap();
        let delta = events
            .iter()
            .find(|(name, _)| name == "message_delta")
            .unwrap();
        assert_eq!(delta.1["usage"], json!({"output_tokens":0}));
    }

    #[test]
    fn synthetic_cached_tokens_never_make_input_negative() {
        let input = frames(&[
            &text_chunk("x"),
            &json!({"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"cachedContentTokenCount":9}}).to_string(),
        ]);
        let message = both_succeed(&input, false);
        assert_eq!(message["usage"]["input_tokens"], 0);
        assert_eq!(message["usage"]["cache_read_input_tokens"], 9);
    }

    /// Every terminal condition, with and without a tool call before it, in
    /// both modes. Errors must win over tool calls already sent.
    #[test]
    fn synthetic_ordered_termination_cross_product() {
        enum Expect {
            Error(&'static str),
            Stop(&'static str, &'static str),
        }
        let block =
            json!({"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"blocked"}})
                .to_string();
        let upstream_error =
            json!({"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}).to_string();
        let stopped = |reason: &str| {
            json!({"candidates":[{"finishReason":reason,"finishMessage":"why it stopped"}]})
                .to_string()
        };
        let terminals: Vec<(&str, String, Expect)> = vec![
            (
                "error chunk",
                upstream_error,
                Expect::Error("upstream_error"),
            ),
            ("blockReason", block, Expect::Error("prompt_blocked")),
            ("SAFETY", stopped("SAFETY"), Expect::Error("finish_reason")),
            (
                "RECITATION",
                stopped("RECITATION"),
                Expect::Error("finish_reason"),
            ),
            ("OTHER", stopped("OTHER"), Expect::Error("finish_reason")),
            (
                "BLOCKLIST",
                stopped("BLOCKLIST"),
                Expect::Error("finish_reason"),
            ),
            (
                "PROHIBITED_CONTENT",
                stopped("PROHIBITED_CONTENT"),
                Expect::Error("finish_reason"),
            ),
            ("SPII", stopped("SPII"), Expect::Error("finish_reason")),
            (
                "MALFORMED_FUNCTION_CALL",
                stopped("MALFORMED_FUNCTION_CALL"),
                Expect::Error("finish_reason"),
            ),
            (
                "MODEL_ARMOR",
                stopped("MODEL_ARMOR"),
                Expect::Error("finish_reason"),
            ),
            (
                "UNEXPECTED_TOOL_CALL",
                stopped("UNEXPECTED_TOOL_CALL"),
                Expect::Error("finish_reason"),
            ),
            (
                "TOO_MANY_TOOL_CALLS",
                stopped("TOO_MANY_TOOL_CALLS"),
                Expect::Error("finish_reason"),
            ),
            (
                "MALFORMED_RESPONSE",
                stopped("MALFORMED_RESPONSE"),
                Expect::Error("finish_reason"),
            ),
            (
                "unknown value",
                stopped("SOMETHING_NEW"),
                Expect::Error("finish_reason"),
            ),
            (
                "end without finishReason",
                String::new(),
                Expect::Error("incomplete_stream"),
            ),
            (
                "MAX_TOKENS",
                finish_chunk("MAX_TOKENS"),
                Expect::Stop("max_tokens", "max_tokens"),
            ),
            (
                "STOP",
                finish_chunk("STOP"),
                Expect::Stop("tool_use", "end_turn"),
            ),
        ];
        for with_tool in [true, false] {
            let before = if with_tool {
                call_chunk("Read", Some(SIG_A))
            } else {
                text_chunk("partial")
            };
            for (name, terminal, expect) in &terminals {
                let mut chunks = vec![before.as_str()];
                if !terminal.is_empty() {
                    chunks.push(terminal.as_str());
                }
                let input = frames(&chunks);
                let label = format!("{name}, tool call before: {with_tool}");
                match expect {
                    Expect::Error(kind) => {
                        let error = both_fail(&input);
                        assert_eq!(error.kind(), *kind, "{label}");
                    }
                    Expect::Stop(with, without) => {
                        let message = both_succeed(&input, true);
                        let expected = if with_tool { with } else { without };
                        assert_eq!(message["stop_reason"], *expected, "{label}");
                        let has_tool = message["content"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|block| block["type"] == "tool_use");
                        assert_eq!(has_tool, with_tool, "{label}");
                    }
                }
            }
        }
    }

    #[test]
    fn synthetic_content_after_the_finish_reason_is_an_error() {
        let input = frames(&[&finish_chunk("STOP"), &text_chunk("late")]);
        assert_eq!(both_fail(&input), StreamError::ContentAfterFinish);
        let input = frames(&[&finish_chunk("STOP"), &call_chunk("Read", None)]);
        assert_eq!(both_fail(&input), StreamError::ContentAfterFinish);
    }

    #[test]
    fn synthetic_changed_finish_reason_is_an_error() {
        let input = frames(&[
            &text_chunk("x"),
            &finish_chunk("STOP"),
            &finish_chunk("MAX_TOKENS"),
        ]);
        assert_eq!(
            both_fail(&input),
            StreamError::FinishReasonChanged {
                first: "STOP".to_string(),
                second: "MAX_TOKENS".to_string(),
            }
        );
    }

    #[test]
    fn synthetic_malformed_events_are_errors() {
        let cases = [
            ("broken JSON", "{\"candidates\":[".to_string()),
            ("not an object", "42".to_string()),
            ("array without error", "[{\"candidates\":[]}]".to_string()),
            (
                "second candidate",
                json!({"candidates":[{"content":{"parts":[{"text":"a"}]}},{"content":{"parts":[{"text":"b"}]}}]}).to_string(),
            ),
            (
                "candidate index 1",
                json!({"candidates":[{"index":1,"content":{"parts":[{"text":"b"}]}}]}).to_string(),
            ),
            (
                "inline data part",
                json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw0K"}}]}}]}).to_string(),
            ),
            (
                "executable code part",
                json!({"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"1"}}]}}]}).to_string(),
            ),
            (
                "streamed arguments",
                json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"Read","willContinue":true}}]}}]}).to_string(),
            ),
            (
                "args not an object",
                json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"Read","args":[1]}}]}}]}).to_string(),
            ),
            (
                "call without name",
                json!({"candidates":[{"content":{"parts":[{"functionCall":{"args":{}}}]}}]}).to_string(),
            ),
            ("done marker", "[DONE]".to_string()),
        ];
        for (name, chunk) in cases {
            let input = frames(&[&chunk, &finish_chunk("STOP")]);
            let error = both_fail(&input);
            assert!(
                matches!(error, StreamError::Malformed(_)),
                "{name}: {error:?}"
            );
        }
    }

    #[test]
    fn synthetic_error_chunk_in_array_form_is_an_upstream_error() {
        let input = frames(&[
            &text_chunk("x"),
            &json!([{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}]).to_string(),
        ]);
        let error = both_fail(&input);
        assert_eq!(
            error,
            StreamError::Upstream {
                code: Some(429),
                status: "RESOURCE_EXHAUSTED".to_string(),
                upstream_message: Some("Quota exceeded".to_string()),
            }
        );
        assert_eq!(error.error_type(), "rate_limit_error");
    }

    #[test]
    fn synthetic_frames_larger_than_the_shared_default_limit_are_accepted() {
        let content = "x".repeat(2 * 1024 * 1024);
        let input = frames(&[
            &json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"Write","args":{"content":content}}}]}}]}).to_string(),
            &finish_chunk("STOP"),
        ]);
        let message = both_succeed(&input, false);
        assert_eq!(
            message["content"][0]["input"]["content"]
                .as_str()
                .unwrap()
                .len(),
            content.len()
        );
    }

    #[test]
    fn synthetic_end_inside_a_frame_after_the_finish_reason_is_an_error() {
        let mut input = frames(&[&text_chunk("x"), &finish_chunk("STOP")]);
        input.push_str("data: {\"usageMetadata\":");
        assert!(matches!(both_fail(&input), StreamError::Malformed(_)));
        // When the upstream stays open instead, the partial frame is dropped.
        let mut translator = LiveStreamTranslator::new(config(false));
        translator.push(input.as_bytes()).unwrap();
        assert!(translator.is_terminal());
        let out = translator.finish_after_read_limit().unwrap();
        assert!(String::from_utf8(out).unwrap().contains("end_turn"));
        assert!(translator.is_finished());
    }

    #[test]
    fn display_keeps_upstream_prose_out_and_message_with_sanitizes_it() {
        let error = StreamError::Stopped {
            finish_reason: "SAFETY".to_string(),
            finish_message: Some("echoed TOKEN-123 here".to_string()),
        };
        assert_eq!(
            error.to_string(),
            "Vertex AI stopped the response: finishReason SAFETY"
        );
        let sanitize = |text: &str| text.replace("TOKEN-123", "[redacted]");
        assert_eq!(
            error.message_with(&sanitize),
            "Vertex AI stopped the response: finishReason SAFETY: echoed [redacted] here"
        );
        let upstream = StreamError::Upstream {
            code: Some(503),
            status: "UNAVAILABLE".to_string(),
            upstream_message: Some("TOKEN-123".to_string()),
        };
        assert_eq!(
            upstream.to_string(),
            "Vertex AI stream reported an error: UNAVAILABLE (HTTP 503)"
        );
        assert!(!upstream.message_with(&sanitize).contains("TOKEN-123"));
        assert_eq!(
            StreamError::Incomplete.message_with(&sanitize),
            StreamError::Incomplete.to_string()
        );
    }

    #[test]
    fn synthetic_unusual_tokens_are_replaced_before_they_reach_display() {
        let input = frames(&[
            &json!({"candidates":[{"finishReason":"stop because <script>"}]}).to_string(),
        ]);
        let error = both_fail(&input);
        assert_eq!(
            error,
            StreamError::Stopped {
                finish_reason: UNRECOGNIZED_TOKEN.to_string(),
                finish_message: None,
            }
        );
        let input = frames(&[
            &json!({"error":{"code":99999,"status":"lower case","message":"m"}}).to_string(),
        ]);
        assert_eq!(
            both_fail(&input),
            StreamError::Upstream {
                code: None,
                status: UNRECOGNIZED_TOKEN.to_string(),
                upstream_message: Some("m".to_string()),
            }
        );
    }

    #[test]
    fn error_types_follow_the_upstream_status() {
        let tests = [
            (Some(429), "RESOURCE_EXHAUSTED", "rate_limit_error"),
            (None, "RESOURCE_EXHAUSTED", "rate_limit_error"),
            (Some(401), "UNAUTHENTICATED", "authentication_error"),
            (Some(403), "PERMISSION_DENIED", "permission_error"),
            (Some(400), "INVALID_ARGUMENT", "invalid_request_error"),
            (Some(404), "NOT_FOUND", "invalid_request_error"),
            (Some(500), "INTERNAL", "api_error"),
            (Some(503), "UNAVAILABLE", "api_error"),
            (None, "UNKNOWN", "api_error"),
        ];
        for (code, status, expected) in tests {
            let error = StreamError::Upstream {
                code,
                status: status.to_string(),
                upstream_message: None,
            };
            assert_eq!(error.error_type(), expected, "{code:?} {status}");
        }
        let blocked = StreamError::PromptBlocked {
            block_reason: "SAFETY".to_string(),
            upstream_message: None,
        };
        assert_eq!(blocked.error_type(), "invalid_request_error");
        assert_eq!(StreamError::Incomplete.error_type(), "api_error");
    }

    #[test]
    fn capture_values_have_upstream_prose_sanitized() {
        let sanitize = |text: &str| text.replace("SECRET", "[redacted]");
        let value = json!({
            "error":{"code":400,"status":"INVALID_ARGUMENT","message":"bad SECRET","details":[{"reason":"SECRET"}]},
            "candidates":[{"finishReason":"SAFETY","finishMessage":"SECRET stop"}],
            "promptFeedback":{"blockReason":"OTHER","blockReasonMessage":"SECRET block"}
        });
        let sanitized = sanitize_capture(&value, &sanitize);
        assert!(!sanitized.to_string().contains("SECRET"), "{sanitized}");
        assert_eq!(
            sanitized["error"],
            json!({"code":400,"status":"INVALID_ARGUMENT","message":"bad [redacted]"})
        );
        assert_eq!(sanitized["candidates"][0]["finishReason"], "SAFETY");
        let array = sanitize_capture(&json!([{"error":{"message":"SECRET"}}]), &sanitize);
        assert_eq!(array, json!([{"error":{"message":"[redacted]"}}]));
    }

    fn sinks(outcome: &NativeResponseOutcome) -> StreamSinks {
        StreamSinks {
            req_id: "req_1".to_string(),
            monitor: None,
            traffic: None,
            outcome: Some(outcome.clone()),
            sanitize: Arc::new(|text: &str| text.replace("SECRET", "[redacted]")),
        }
    }

    async fn body_text(body: Body) -> String {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_stream_body_completes_after_the_read_limit_when_upstream_stays_open() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(frame(&text_chunk("hi")))),
            Ok(Bytes::from(frame(&finish_chunk("STOP")))),
        ];
        let upstream = futures_util::stream::iter(chunks).chain(futures_util::stream::pending());
        let outcome = NativeResponseOutcome::default();
        let started = tokio::time::Instant::now();
        let text = body_text(stream_body(upstream, config(false), sinks(&outcome))).await;
        assert!(started.elapsed() >= POST_TERMINAL_READ_LIMIT);
        assert!(text.contains("\"stop_reason\":\"end_turn\""), "{text}");
        assert!(
            text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            "{text}"
        );
        assert_eq!(outcome.failure(), None);
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_stream_body_completes_without_waiting_when_upstream_closes() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(frame(&text_chunk("hi")))),
            Ok(Bytes::from(frame(&finish_chunk("MAX_TOKENS")))),
        ];
        let outcome = NativeResponseOutcome::default();
        let started = tokio::time::Instant::now();
        let text = body_text(stream_body(
            futures_util::stream::iter(chunks),
            config(false),
            sinks(&outcome),
        ))
        .await;
        assert!(started.elapsed() < POST_TERMINAL_READ_LIMIT);
        assert!(text.contains("\"stop_reason\":\"max_tokens\""), "{text}");
    }

    #[tokio::test]
    async fn synthetic_stream_body_reports_a_sanitized_error_event_and_outcome() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(frame(&call_chunk("Read", Some(SIG_A))))),
            Ok(Bytes::from(frame(
                &json!({"candidates":[{"finishReason":"SAFETY","finishMessage":"leaked SECRET"}]})
                    .to_string(),
            ))),
        ];
        let outcome = NativeResponseOutcome::default();
        let text = body_text(stream_body(
            futures_util::stream::iter(chunks),
            config(false),
            sinks(&outcome),
        ))
        .await;
        assert!(!text.contains("SECRET"), "{text}");
        assert!(!text.contains("message_delta"), "{text}");
        let events = parse_sse_events(text.as_bytes());
        let last = events.last().unwrap();
        assert_eq!(last.event.as_deref(), Some("error"));
        let data: Value = serde_json::from_str(&last.data).unwrap();
        assert_eq!(data["error"]["type"], "api_error");
        assert_eq!(
            data["error"]["message"],
            "Vertex AI stopped the response: finishReason SAFETY: leaked [redacted]"
        );
        assert_eq!(
            outcome.failure().as_deref(),
            Some("Vertex AI stopped the response: finishReason SAFETY: leaked [redacted]")
        );
    }

    #[tokio::test]
    async fn synthetic_stream_body_reports_a_transport_failure() {
        let chunks = vec![
            Ok(Bytes::from(frame(&text_chunk("hi")))),
            Err(std::io::Error::other("reset")),
        ];
        let outcome = NativeResponseOutcome::default();
        let text = body_text(stream_body(
            futures_util::stream::iter(chunks),
            config(false),
            sinks(&outcome),
        ))
        .await;
        assert!(text.contains("Vertex AI stream was interrupted"), "{text}");
        assert!(!text.contains("reset"), "{text}");
        assert!(outcome.failure().is_some());
    }
}
