//! Cloud Code PA `v1internal:streamGenerateContent?alt=sse` chunks → Anthropic Messages.
//!
//! One [`LiveStreamTranslator`] drives both the streaming response and the
//! non-streaming one ([`accumulate_response`]), so the two decide content
//! and stop reason the same way.
//!
//! Each SSE `data:` frame from Cloud Code PA is a JSON object wrapping a
//! `GenerateContentResponse` under `"response"` (plus optional `"traceId"`),
//! or a flat `GenerateContentResponse` / error object.

use std::collections::HashSet;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::thought_signature::encode_envelope;
use crate::anthropic::sse::encode_sse_event;
use crate::providers::grok::translate::stream::SseDecoder;
pub use crate::providers::vertex::translate::stream::{
    BLOCK_CHARGE, MAX_CONTENT_BYTES, MAX_SSE_FRAME_BYTES, TranslatorConfig,
};

/// Longest upstream token (a status, finishReason, blockReason or error code)
/// kept in error text.
const MAX_TOKEN_CHARS: usize = 64;

/// Placeholder for an upstream token that is not a plain identifier.
const UNRECOGNIZED_TOKEN: &str = "unrecognized";

/// Longest upstream call ID passed through as the `tool_use` ID.
const MAX_CALL_ID_BYTES: usize = 256;

/// Documented top-level fields of a `GenerateContentResponse` or Cloud Code PA
/// wrapper chunk.
const RESPONSE_FIELDS: [&str; 7] = [
    "candidates",
    "usageMetadata",
    "promptFeedback",
    "modelVersion",
    "createTime",
    "responseId",
    "traceId",
];

/// Documented fields of a response `Part`.
const PART_FIELDS: [&str; 9] = [
    "text",
    "thought",
    "thoughtSignature",
    "functionCall",
    "inlineData",
    "fileData",
    "functionResponse",
    "executableCode",
    "codeExecutionResult",
];

/// Why a stream could not be translated into a successful response.
///
/// `Display` contains only proxy-authored words and identifier tokens.
/// Upstream prose is kept in `upstream_message` / `finish_message` and is
/// added, sanitised, only by [`StreamError::message_with`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    /// The stream carried an `error` object.
    #[error("Google AI Ultra plan stream reported an error{}", code_suffix(.code))]
    Upstream {
        /// The error `code` or `status` token, such as `invalid_request` or
        /// `INVALID_ARGUMENT`.
        code: Option<String>,
        /// The upstream `error.message`. Unsanitised.
        upstream_message: Option<String>,
    },
    /// `promptFeedback.blockReason` was set.
    #[error("Google AI Ultra plan blocked the prompt: blockReason {block_reason}")]
    PromptBlocked {
        /// The block reason token.
        block_reason: String,
        /// The upstream `promptFeedback.blockReasonMessage`. Unsanitised.
        upstream_message: Option<String>,
    },
    /// `candidates[0].finishReason` carried a value other than `STOP` or
    /// `MAX_TOKENS`.
    #[error("Google AI Ultra plan stopped the response: finishReason {finish_reason}")]
    Stopped {
        /// The finish reason token.
        finish_reason: String,
        /// The upstream `candidates[0].finishMessage`. Unsanitised.
        finish_message: Option<String>,
    },
    /// The upstream closed before any `finishReason`.
    #[error("Google AI Ultra plan stream ended without a finishReason")]
    Incomplete,
    /// Reading the upstream body failed.
    #[error("Google AI Ultra plan stream was interrupted")]
    Transport,
    /// The stream broke the documented shape.
    #[error("Google AI Ultra plan stream is invalid: {0}")]
    Malformed(&'static str),
}

fn code_suffix(code: &Option<String>) -> String {
    code.as_deref()
        .map(|code| format!(" ({code})"))
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
            Self::Upstream {
                code: Some(code), ..
            } => match code.as_str() {
                "too_many_requests" | "resource_exhausted" | "RESOURCE_EXHAUSTED" | "429" => {
                    "rate_limit_error"
                }
                "invalid_request"
                | "invalid_argument"
                | "INVALID_ARGUMENT"
                | "failed_precondition"
                | "FAILED_PRECONDITION"
                | "not_found"
                | "NOT_FOUND"
                | "400"
                | "404" => "invalid_request_error",
                "unauthenticated" | "UNAUTHENTICATED" | "401" => "authentication_error",
                "permission_denied" | "PERMISSION_DENIED" | "403" => "permission_error",
                _ => "api_error",
            },
            Self::PromptBlocked { .. } => "invalid_request_error",
            _ => "api_error",
        }
    }

    /// A short label for logs and traffic-capture records.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Upstream { .. } => "upstream_error",
            Self::PromptBlocked { .. } => "prompt_blocked",
            Self::Stopped { .. } => "finish_reason",
            Self::Incomplete => "incomplete_stream",
            Self::Transport => "upstream_stream",
            Self::Malformed(_) => "invalid_event",
        }
    }

    /// The one Anthropic `error` SSE event that ends a failed stream, with
    /// the message passed through `sanitize`.
    pub fn sse_event(&self, sanitize: &dyn Fn(&str) -> String) -> Vec<u8> {
        error_event(self.error_type(), &self.message_with(sanitize))
    }
}

/// The Anthropic `error` SSE event. `message` is what the client reads and
/// must already be sanitised.
pub fn error_event(error_type: &str, message: &str) -> Vec<u8> {
    let data = json!({"type":"error","error":{"type":error_type,"message":message}});
    encode_sse_event(Some("error"), &data.to_string())
}

#[derive(Debug, Default, Deserialize)]
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
    id: Option<String>,
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
    /// `(input, cache_read, output)`: cached tokens are part of
    /// `promptTokenCount`, and thought tokens are output.
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

/// A finished content block, kept for the non-streaming response.
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

#[derive(Debug)]
enum Open {
    Thinking { index: usize, signature: String },
    Text(usize),
}

struct TranslationState {
    config: TranslatorConfig,
    keep_content: bool,
    content_bytes: usize,
    message_started: bool,
    finished: bool,
    blocks: Vec<Block>,
    block_count: usize,
    open: Option<Open>,
    tool_count: usize,
    tool_ids: HashSet<String>,
    finish: Option<Finish>,
    usage: Option<UsageMetadata>,
}

impl TranslationState {
    fn new(config: TranslatorConfig, keep_content: bool) -> Self {
        Self {
            config,
            keep_content,
            content_bytes: 0,
            message_started: false,
            finished: false,
            blocks: Vec::new(),
            block_count: 0,
            open: None,
            tool_count: 0,
            tool_ids: HashSet::new(),
            finish: None,
            usage: None,
        }
    }

    fn count_content(&mut self, len: usize) -> Result<(), StreamError> {
        self.content_bytes = self.content_bytes.saturating_add(len);
        if self.content_bytes > MAX_CONTENT_BYTES {
            return Err(StreamError::Malformed(
                "the response content exceeds the 16 MiB limit",
            ));
        }
        Ok(())
    }

    fn apply_chunk(&mut self, chunk: Chunk) -> Result<Vec<u8>, StreamError> {
        if let Some(feedback) = chunk.prompt_feedback
            && let Some(block_reason) = feedback.block_reason
        {
            return Err(StreamError::PromptBlocked {
                block_reason: upstream_token(&block_reason),
                upstream_message: feedback.block_reason_message,
            });
        }
        if let Some(usage) = chunk.usage_metadata {
            let prev = self.usage.unwrap_or_default();
            self.usage = Some(UsageMetadata {
                prompt_token_count: usage.prompt_token_count.or(prev.prompt_token_count),
                candidates_token_count: usage
                    .candidates_token_count
                    .or(prev.candidates_token_count),
                thoughts_token_count: usage.thoughts_token_count.or(prev.thoughts_token_count),
                cached_content_token_count: usage
                    .cached_content_token_count
                    .or(prev.cached_content_token_count),
            });
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
        // Once the finish reason has arrived, only trailing usage-only chunks
        // are expected; further content is a malformed stream.
        if self.finish.is_some() && !parts.is_empty() {
            return Err(StreamError::Malformed(
                "the response streams content after the finish reason",
            ));
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
            // Hold the terminal events: the final chunk may be followed by a
            // usage-only chunk, so they are emitted when the stream closes, a
            // `[DONE]` marker arrives, or the post-terminal read window ends.
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
            let raw_id = call.id.unwrap_or_default();
            let signature = part.thought_signature.filter(|value| !value.is_empty());
            return self.function_call(raw_id, name, input, signature, out);
        }

        let text = part.text.unwrap_or_default();
        let signature = part.thought_signature.filter(|value| !value.is_empty());
        if part.thought == Some(true) {
            if !text.is_empty() && self.config.emit_thinking {
                self.thinking_delta(&text, out)?;
            }
            if let Some(sig) = signature {
                self.thinking_signature(&sig, out)?;
            }
            return Ok(());
        }

        if let Some(sig) = signature
            && (matches!(self.open, Some(Open::Thinking { .. })) || text.is_empty())
        {
            self.thinking_signature(&sig, out)?;
        }
        if !text.is_empty() {
            self.text_delta(&text, out)?;
        }
        Ok(())
    }

    /// The open thinking block, or a new one. A signature ends its block:
    /// thought text or a signature arriving after it starts a new block, so
    /// two independently signed parts never share one block or have their
    /// signatures joined.
    fn ensure_thinking_open(&mut self, out: &mut Vec<u8>) -> Result<usize, StreamError> {
        match &self.open {
            Some(Open::Thinking { index, signature }) if signature.is_empty() => Ok(*index),
            _ => {
                self.close_open(out)?;
                let index = self.start_block(
                    out,
                    Block::Thinking {
                        text: String::new(),
                        signature: String::new(),
                    },
                    json!({"type":"thinking","thinking":"","signature":""}),
                )?;
                self.open = Some(Open::Thinking {
                    index,
                    signature: String::new(),
                });
                Ok(index)
            }
        }
    }

    fn thinking_delta(&mut self, text: &str, out: &mut Vec<u8>) -> Result<(), StreamError> {
        self.count_content(text.len())?;
        let index = self.ensure_thinking_open(out)?;
        if self.keep_content
            && let Some(Block::Thinking { text: stored, .. }) = self.blocks.get_mut(index)
        {
            stored.push_str(text);
        }
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":text}}),
        );
        Ok(())
    }

    fn thinking_signature(&mut self, sig: &str, out: &mut Vec<u8>) -> Result<(), StreamError> {
        self.count_content(sig.len())?;
        let _ = self.ensure_thinking_open(out)?;
        if let Some(Open::Thinking { signature, .. }) = &mut self.open {
            signature.push_str(sig);
        }
        Ok(())
    }

    fn text_delta(&mut self, text: &str, out: &mut Vec<u8>) -> Result<(), StreamError> {
        self.count_content(text.len())?;
        let index = match self.open {
            Some(Open::Text(index)) => index,
            _ => {
                self.close_open(out)?;
                let index = self.start_block(
                    out,
                    Block::Text {
                        text: String::new(),
                    },
                    json!({"type":"text","text":""}),
                )?;
                self.open = Some(Open::Text(index));
                index
            }
        };
        if self.keep_content
            && let Some(Block::Text { text: stored }) = self.blocks.get_mut(index)
        {
            stored.push_str(text);
        }
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
        );
        Ok(())
    }

    fn function_call(
        &mut self,
        raw_id: String,
        name: String,
        input: Value,
        signature: Option<String>,
        out: &mut Vec<u8>,
    ) -> Result<(), StreamError> {
        if let Some(sig) = signature {
            self.count_content(sig.len())?;
            match &mut self.open {
                Some(Open::Thinking {
                    signature: existing,
                    ..
                }) if existing.is_empty() => {
                    *existing = sig;
                    self.close_open(out)?;
                }
                _ => {
                    self.close_open(out)?;
                    let index = self.start_block(
                        out,
                        Block::Thinking {
                            text: String::new(),
                            signature: String::new(),
                        },
                        json!({"type":"thinking","thinking":"","signature":""}),
                    )?;
                    self.open = Some(Open::Thinking {
                        index,
                        signature: sig,
                    });
                    self.close_open(out)?;
                }
            }
        } else {
            self.close_open(out)?;
        }

        let compact = input.to_string();
        let id = if is_valid_call_id(&raw_id) && !self.tool_ids.contains(&raw_id) {
            raw_id
        } else {
            self.unused_call_id()
        };
        self.count_content(
            id.len()
                .saturating_add(name.len())
                .saturating_add(compact.len()),
        )?;
        self.tool_ids.insert(id.clone());
        self.tool_count += 1;
        let content_block = json!({"type":"tool_use","id":id,"name":name,"input":{}});
        let index = self.start_block(out, Block::Tool { id, name, input }, content_block)?;
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":compact}}),
        );
        emit(
            out,
            "content_block_stop",
            json!({"type":"content_block_stop","index":index}),
        );
        Ok(())
    }

    fn unused_call_id(&self) -> String {
        let mut suffix = self.tool_count;
        loop {
            let id = format!("toolu_{}_{suffix}", self.config.message_id);
            if !self.tool_ids.contains(&id) {
                return id;
            }
            suffix = suffix.saturating_add(1);
        }
    }

    fn start_block(
        &mut self,
        out: &mut Vec<u8>,
        block: Block,
        content_block: Value,
    ) -> Result<usize, StreamError> {
        self.count_content(BLOCK_CHARGE)?;
        self.ensure_message_start(out);
        let index = self.block_count;
        self.block_count += 1;
        if self.keep_content {
            self.blocks.push(block);
        }
        emit(
            out,
            "content_block_start",
            json!({"type":"content_block_start","index":index,"content_block":content_block}),
        );
        Ok(index)
    }

    fn close_open(&mut self, out: &mut Vec<u8>) -> Result<(), StreamError> {
        match self.open.take() {
            Some(Open::Thinking { index, signature }) => {
                let envelope = encode_envelope(&self.config.signature_model, &signature)
                    .or_else(|| encode_envelope(&self.config.signature_model, ""))
                    .unwrap_or_default();
                // The signature bytes were charged while the block was open.
                self.count_content(envelope.len().saturating_sub(signature.len()))?;
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
                if self.keep_content
                    && let Some(Block::Thinking {
                        signature: stored, ..
                    }) = self.blocks.get_mut(index)
                {
                    *stored = envelope;
                }
            }
            Some(Open::Text(index)) => {
                emit(
                    out,
                    "content_block_stop",
                    json!({"type":"content_block_stop","index":index}),
                );
            }
            None => {}
        }
        Ok(())
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

    fn finalize(&mut self, out: &mut Vec<u8>) -> Result<(), StreamError> {
        let stop_reason = self.stop_reason()?;
        self.close_open(out)?;
        self.ensure_message_start(out);
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
            out,
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":stop_reason,"stop_sequence":null},"usage":usage}),
        );
        emit(out, "message_stop", json!({"type":"message_stop"}));
        self.finished = true;
        Ok(())
    }

    fn response(&self) -> Result<Value, StreamError> {
        if !self.keep_content {
            return Err(StreamError::Malformed(
                "the streaming translator keeps no content to build a message from",
            ));
        }
        if !self.finished {
            return Err(StreamError::Incomplete);
        }
        let stop_reason = self.stop_reason()?;
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
            "stop_reason":stop_reason,
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

fn is_valid_call_id(id: &str) -> bool {
    (1..=MAX_CALL_ID_BYTES).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Keeps an upstream token (a status, finishReason, blockReason or error code)
/// only when it looks like one.
fn upstream_token(value: &str) -> String {
    let well_formed = (1..=MAX_TOKEN_CHARS).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.');
    if well_formed {
        value.to_string()
    } else {
        UNRECOGNIZED_TOKEN.to_string()
    }
}

fn upstream_error(error: Option<&Value>) -> StreamError {
    let code = error.and_then(|error| {
        if let Some(code) = error.get("code").and_then(Value::as_str) {
            return Some(upstream_token(code));
        }
        if let Some(status) = error.get("status").and_then(Value::as_str) {
            return Some(upstream_token(status));
        }
        if let Some(code) = error.get("code").and_then(Value::as_u64) {
            return Some(upstream_token(&code.to_string()));
        }
        None
    });
    let upstream_message = match error {
        Some(Value::String(message)) => Some(message.clone()),
        Some(error) => error
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string),
        None => None,
    };
    StreamError::Upstream {
        code,
        upstream_message,
    }
}

fn parse_chunk(data: &str) -> Result<Chunk, StreamError> {
    let value: Value = serde_json::from_str(data)
        .map_err(|_| StreamError::Malformed("an event is not valid JSON"))?;
    let mut object = match value {
        Value::Object(object) => object,
        Value::Array(items) => {
            return Err(match items.as_slice() {
                [item] if item.get("error").is_some() => upstream_error(item.get("error")),
                _ => StreamError::Malformed("an event is not a JSON object"),
            });
        }
        _ => return Err(StreamError::Malformed("an event is not a JSON object")),
    };
    if let Some(error) = object.get("error") {
        return Err(upstream_error(Some(error)));
    }
    let response = match object.remove("response") {
        Some(Value::Object(inner)) => inner,
        Some(_) => {
            return Err(StreamError::Malformed(
                "response field in SSE event is not a JSON object",
            ));
        }
        None => object,
    };
    if let Some(error) = response.get("error") {
        return Err(upstream_error(Some(error)));
    }
    check_known_shape(&response)?;
    serde_json::from_value(Value::Object(response)).map_err(|_| {
        StreamError::Malformed("an event does not have the GenerateContentResponse shape")
    })
}

fn check_known_shape(object: &Map<String, Value>) -> Result<(), StreamError> {
    let has_any = |map: &Map<String, Value>, fields: &[&str]| {
        fields.iter().any(|field| map.contains_key(*field))
    };
    if !has_any(object, &RESPONSE_FIELDS) {
        return Err(StreamError::Malformed(
            "an event has none of the GenerateContentResponse fields",
        ));
    }
    check_parts_shape(object)
}

fn check_parts_shape(object: &Map<String, Value>) -> Result<(), StreamError> {
    let has_any = |map: &Map<String, Value>, fields: &[&str]| {
        fields.iter().any(|field| map.contains_key(*field))
    };
    let parts = object
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|candidate| candidate.pointer("/content/parts")?.as_array())
        .flatten();
    for part in parts {
        if let Some(part) = part.as_object()
            && !has_any(part, &PART_FIELDS)
        {
            return Err(StreamError::Malformed(
                "a part has none of the documented Part fields",
            ));
        }
    }
    Ok(())
}

fn emit(out: &mut Vec<u8>, event: &str, value: Value) {
    out.extend(encode_sse_event(Some(event), &value.to_string()));
}

/// Incremental translator from Cloud Code PA SSE bytes to Anthropic SSE bytes.
pub struct LiveStreamTranslator {
    decoder: SseDecoder,
    state: TranslationState,
}

impl LiveStreamTranslator {
    /// A translator for one non-streaming response: it keeps the content
    /// for [`Self::response`].
    pub fn new(config: TranslatorConfig) -> Self {
        Self::with_content(config, true)
    }

    /// A translator for one streaming response. It emits the same events as
    /// [`Self::new`] but keeps no content, so [`Self::response`] is an error.
    pub fn streaming(config: TranslatorConfig) -> Self {
        Self::with_content(config, false)
    }

    fn with_content(config: TranslatorConfig, keep_content: bool) -> Self {
        Self {
            decoder: SseDecoder::with_max_frame_bytes(MAX_SSE_FRAME_BYTES),
            state: TranslationState::new(config, keep_content),
        }
    }

    /// Feeds upstream bytes, split at any boundary, and returns the
    /// Anthropic SSE bytes they produce. Input after the response completed
    /// is ignored.
    pub fn push(&mut self, input: &[u8]) -> Result<Vec<u8>, StreamError> {
        if self.state.finished {
            return Ok(Vec::new());
        }
        let events = self.decoder.push(input).map_err(|_| {
            StreamError::Malformed("the SSE framing is invalid or a frame exceeds 16 MiB")
        })?;
        let mut out = Vec::new();
        for event in events {
            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                // The upstream has ended the stream. Emit the held terminal
                // events if a finish reason arrived, and read nothing further.
                if self.state.finish.is_some() && !self.state.finished {
                    self.state.finalize(&mut out)?;
                }
                break;
            }
            let chunk = parse_chunk(data)?;
            out.extend(self.state.apply_chunk(chunk)?);
            if self.state.finished {
                break;
            }
        }
        Ok(out)
    }

    /// Completes the response after the upstream closed, emitting the held
    /// terminal events.
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] when no `finishReason` arrived.
    pub fn finish(&mut self) -> Result<Vec<u8>, StreamError> {
        if self.state.finished {
            return Ok(Vec::new());
        }
        if self.state.finish.is_none() {
            return Err(StreamError::Incomplete);
        }
        let mut out = Vec::new();
        self.state.finalize(&mut out)?;
        Ok(out)
    }

    /// Completes the response when the upstream stayed open for the
    /// post-terminal read window after the `finishReason`, emitting the held
    /// terminal events with whatever usage arrived in that window.
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] when called before a `finishReason`.
    pub fn finish_after_read_limit(&mut self) -> Result<Vec<u8>, StreamError> {
        let mut out = Vec::new();
        self.state.finalize(&mut out)?;
        Ok(out)
    }

    /// Whether a `finishReason` has arrived but the terminal events are still
    /// held, so the caller reads for at most the post-terminal window.
    pub fn has_finish(&self) -> bool {
        self.state.finish.is_some() && !self.state.finished
    }

    /// Whether the response has been completed.
    pub fn is_finished(&self) -> bool {
        self.state.finished
    }

    /// How many `tool_use` blocks were emitted.
    pub fn tool_calls(&self) -> usize {
        self.state.tool_count
    }

    /// The model name echoed to the client.
    pub fn model(&self) -> &str {
        &self.state.config.model
    }

    /// The non-streaming Anthropic message, once the response is completed.
    pub fn response(&self) -> Result<Value, StreamError> {
        self.state.response()
    }
}

/// Translates a whole buffered upstream body into a non-streaming Anthropic
/// message.
pub fn accumulate_response(input: &[u8], config: TranslatorConfig) -> Result<Value, StreamError> {
    let mut translator = LiveStreamTranslator::new(config);
    translator.push(input)?;
    translator.finish()?;
    translator.response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::sse::parse_sse_events;
    use crate::providers::google_ai_plan::translate::thought_signature::{
        ENVELOPE_PREFIX, signature_for_model,
    };

    const MODEL: &str = "gemini-4-hydrogen";

    fn config(emit_thinking: bool) -> TranslatorConfig {
        TranslatorConfig {
            message_id: "msg_1".to_string(),
            model: format!("google-ai-plan/{MODEL}"),
            signature_model: MODEL.to_string(),
            estimated_input_tokens: 42,
            emit_thinking,
        }
    }

    fn frame(data: Value) -> String {
        format!("data: {data}\n\n")
    }

    fn frames(events: &[Value]) -> String {
        events.iter().cloned().map(frame).collect()
    }

    fn ccpa_parts(parts: Value) -> Value {
        json!({
            "response": {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": parts
                    }
                }]
            },
            "traceId": "trace-1"
        })
    }

    fn ccpa_finish(parts: Value, reason: &str, usage: Option<Value>) -> Value {
        let mut resp = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": parts
                },
                "finishReason": reason
            }]
        });
        if let Some(usage) = usage {
            resp["usageMetadata"] = usage;
        }
        json!({
            "response": resp,
            "traceId": "trace-1"
        })
    }

    fn recorded_tool_turn() -> String {
        frames(&[
            ccpa_parts(json!([{"thought": true, "text": "Need the file."}])),
            ccpa_parts(json!([
                {"thought": true, "thoughtSignature": "c2ln"},
                {"text": "Reading it."}
            ])),
            ccpa_finish(
                json!([{
                    "functionCall": {
                        "id": "call_1",
                        "name": "read_file",
                        "args": {"path": "a.py"}
                    }
                }]),
                "STOP",
                Some(json!({
                    "promptTokenCount": 114,
                    "cachedContentTokenCount": 10,
                    "candidatesTokenCount": 26,
                    "thoughtsTokenCount": 44,
                    "totalTokenCount": 184
                })),
            ),
        ]) + "data: [DONE]\n\n"
    }

    fn live_chunks(
        chunks: &[&[u8]],
        emit_thinking: bool,
    ) -> Result<Vec<(String, Value)>, StreamError> {
        let mut translator = LiveStreamTranslator::streaming(config(emit_thinking));
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(translator.push(chunk)?);
        }
        out.extend(translator.finish()?);
        Ok(parse_sse_events(&out)
            .into_iter()
            .map(|event| {
                (
                    event.event.unwrap_or_default(),
                    serde_json::from_str(&event.data).expect("translator emits JSON"),
                )
            })
            .collect())
    }

    fn live(input: &str, emit_thinking: bool) -> Result<Vec<(String, Value)>, StreamError> {
        live_chunks(&[input.as_bytes()], emit_thinking)
    }

    fn rebuild(events: &[(String, Value)]) -> (Vec<Value>, Value, Value) {
        let mut content: Vec<Value> = Vec::new();
        let mut partial: Vec<String> = Vec::new();
        let mut open: Option<usize> = None;
        let mut stop_reason = Value::Null;
        let mut usage = Value::Null;
        for (name, data) in events {
            match name.as_str() {
                "content_block_start" => {
                    assert_eq!(
                        open, None,
                        "a block started while another was open: {events:?}"
                    );
                    let index = data["index"].as_u64().expect("index") as usize;
                    assert_eq!(index, content.len(), "block indexes are consecutive");
                    open = Some(index);
                    content.push(data["content_block"].clone());
                    partial.push(String::new());
                }
                "content_block_delta" => {
                    let index = data["index"].as_u64().expect("index") as usize;
                    assert_eq!(
                        open,
                        Some(index),
                        "a delta went to a block that is not open"
                    );
                    let delta = &data["delta"];
                    let block = &mut content[index];
                    match delta["type"].as_str().expect("delta type") {
                        "text_delta" => {
                            let text = block["text"].as_str().unwrap_or_default().to_string()
                                + delta["text"].as_str().expect("text");
                            block["text"] = json!(text);
                        }
                        "thinking_delta" => {
                            let text = block["thinking"].as_str().unwrap_or_default().to_string()
                                + delta["thinking"].as_str().expect("thinking");
                            block["thinking"] = json!(text);
                        }
                        "signature_delta" => block["signature"] = delta["signature"].clone(),
                        "input_json_delta" => {
                            partial[index].push_str(delta["partial_json"].as_str().expect("json"));
                        }
                        other => panic!("unexpected delta {other}"),
                    }
                }
                "content_block_stop" => {
                    let index = data["index"].as_u64().expect("index") as usize;
                    assert_eq!(open, Some(index), "a stop for a block that is not open");
                    open = None;
                    if content[index]["type"] == "tool_use" {
                        content[index]["input"] =
                            serde_json::from_str(&partial[index]).expect("tool input is JSON");
                    }
                }
                "message_delta" => {
                    stop_reason = data["delta"]["stop_reason"].clone();
                    usage = data["usage"].clone();
                }
                _ => {}
            }
        }
        assert_eq!(open, None, "a block was left open");
        (content, stop_reason, usage)
    }

    fn both_succeed(input: &str, emit_thinking: bool) -> Value {
        let events = live(input, emit_thinking).unwrap_or_else(|error| panic!("live: {error}"));
        let message = accumulate_response(input.as_bytes(), config(emit_thinking))
            .unwrap_or_else(|error| panic!("accumulated: {error}"));
        let (content, stop_reason, usage) = rebuild(&events);
        assert_eq!(json!(content), message["content"], "content differs");
        assert_eq!(stop_reason, message["stop_reason"], "stop_reason differs");
        if usage.get("input_tokens").is_some() {
            assert_eq!(usage, message["usage"], "usage differs");
        } else {
            assert_eq!(
                usage,
                json!({"output_tokens":0}),
                "usage without upstream usage"
            );
            assert_eq!(
                message["usage"]["input_tokens"],
                json!(42),
                "estimate in the message"
            );
        }
        assert_eq!(
            events.first().map(|event| event.0.as_str()),
            Some("message_start")
        );
        assert_eq!(
            events.last().map(|event| event.0.as_str()),
            Some("message_stop")
        );
        assert_eq!(events[0].1["message"]["usage"]["input_tokens"], json!(42));
        message
    }

    fn both_fail(input: &str) -> StreamError {
        let live = live(input, true).expect_err("live translation should fail");
        let accumulated = accumulate_response(input.as_bytes(), config(true))
            .expect_err("accumulated translation should fail");
        assert_eq!(live, accumulated, "both modes fail the same way");
        live
    }

    fn envelope(signature: &str) -> String {
        encode_envelope(MODEL, signature).expect("envelope encodes")
    }

    #[test]
    fn thought_text_and_function_call_parts_become_one_message() {
        let message = both_succeed(&recorded_tool_turn(), true);
        assert_eq!(
            message["content"],
            json!([
                {"type":"thinking","thinking":"Need the file.","signature":envelope("c2ln")},
                {"type":"text","text":"Reading it."},
                {"type":"tool_use","id":"call_1","name":"read_file","input":{"path":"a.py"}}
            ])
        );
        assert_eq!(message["stop_reason"], json!("tool_use"));
        assert_eq!(
            message["usage"],
            json!({"input_tokens":104,"output_tokens":70,"cache_creation_input_tokens":0,"cache_read_input_tokens":10}),
            "input excludes cached tokens and output includes thought tokens"
        );
    }

    #[test]
    fn the_signature_reaches_the_client_when_thinking_was_not_requested() {
        let message = both_succeed(&recorded_tool_turn(), false);
        assert_eq!(
            message["content"][0],
            json!({"type":"thinking","thinking":"","signature":envelope("c2ln")}),
            "the summary is dropped and the signature kept"
        );
        assert_eq!(
            signature_for_model(
                message["content"][0]["signature"]
                    .as_str()
                    .unwrap_or_default(),
                MODEL
            ),
            Some("c2ln")
        );
        let unsigned = frames(&[
            ccpa_parts(json!([{"thought": true, "text": "hidden"}])),
            ccpa_finish(json!([{"text": "answer"}]), "STOP", None),
        ]);
        let message = both_succeed(&unsigned, false);
        assert_eq!(message["content"], json!([{"type":"text","text":"answer"}]));
        let message = both_succeed(&unsigned, true);
        assert_eq!(
            message["content"][0],
            json!({"type":"thinking","thinking":"hidden","signature":envelope("")}),
            "an unsigned thought gets the empty envelope"
        );
    }

    #[test]
    fn a_signature_on_a_function_call_part_is_carried() {
        let input = frames(&[ccpa_finish(
            json!([{
                "functionCall": {"name": "read_file", "args": {"path": "a.py"}},
                "thoughtSignature": "c3RhcnQ="
            }]),
            "STOP",
            None,
        )]);
        let message = both_succeed(&input, false);
        assert_eq!(
            message["content"],
            json!([
                {"type":"thinking","thinking":"","signature":envelope("c3RhcnQ=")},
                {"type":"tool_use","id":"toolu_msg_1_0","name":"read_file","input":{"path":"a.py"}}
            ])
        );
    }

    #[test]
    fn independently_signed_thought_parts_stay_separate_blocks() {
        const SIG_A: &str = "Zmlyc3Q=";
        const SIG_B: &str = "c2Vjb25kIQ==";
        let first = json!({"thought": true, "text": "First thought.", "thoughtSignature": SIG_A});
        let second = json!({"thought": true, "text": "Second thought.", "thoughtSignature": SIG_B});
        let finish = ccpa_finish(json!([{"text": "answer"}]), "STOP", None);
        let tests = [
            (
                "both parts in one chunk",
                frames(&[ccpa_parts(json!([first, second])), finish.clone()]),
            ),
            (
                "each part in its own chunk",
                frames(&[
                    ccpa_parts(json!([first])),
                    ccpa_parts(json!([second])),
                    finish.clone(),
                ]),
            ),
            (
                "each signature on a part after its text",
                frames(&[
                    ccpa_parts(json!([
                        {"thought": true, "text": "First thought."},
                        {"thought": true, "thoughtSignature": SIG_A}
                    ])),
                    ccpa_parts(json!([
                        {"thought": true, "text": "Second thought."},
                        {"thought": true, "thoughtSignature": SIG_B}
                    ])),
                    finish.clone(),
                ]),
            ),
        ];
        for (name, input) in &tests {
            let message = both_succeed(input, true);
            println!("{name}: {}", message["content"]);
            assert_eq!(
                message["content"],
                json!([
                    {"type":"thinking","thinking":"First thought.","signature":envelope(SIG_A)},
                    {"type":"thinking","thinking":"Second thought.","signature":envelope(SIG_B)},
                    {"type":"text","text":"answer"}
                ]),
                "{name}"
            );
            for (index, signature) in [(0, SIG_A), (1, SIG_B)] {
                assert_eq!(
                    signature_for_model(
                        message["content"][index]["signature"]
                            .as_str()
                            .unwrap_or_default(),
                        MODEL
                    ),
                    Some(signature),
                    "{name}: block {index}"
                );
            }

            let hidden = both_succeed(input, false);
            assert_eq!(
                hidden["content"],
                json!([
                    {"type":"thinking","thinking":"","signature":envelope(SIG_A)},
                    {"type":"thinking","thinking":"","signature":envelope(SIG_B)},
                    {"type":"text","text":"answer"}
                ]),
                "{name}: thinking not requested"
            );
        }

        // A signature on a non-thought part after a signed thought gets its
        // own block as well.
        let input = frames(&[
            ccpa_parts(json!([first, {"text": "", "thoughtSignature": SIG_B}])),
            finish,
        ]);
        assert_eq!(
            both_succeed(&input, true)["content"],
            json!([
                {"type":"thinking","thinking":"First thought.","signature":envelope(SIG_A)},
                {"type":"thinking","thinking":"","signature":envelope(SIG_B)},
                {"type":"text","text":"answer"}
            ])
        );
    }

    #[test]
    fn finish_reasons_map_to_stop_reasons_or_errors() {
        let text_end = frames(&[ccpa_finish(json!([{"text": "hi"}]), "STOP", None)]);
        assert_eq!(
            both_succeed(&text_end, true)["stop_reason"],
            json!("end_turn")
        );

        let tool_end = frames(&[ccpa_finish(
            json!([{"functionCall": {"id": "c1", "name": "Read", "args": {}}}]),
            "STOP",
            None,
        )]);
        assert_eq!(
            both_succeed(&tool_end, true)["stop_reason"],
            json!("tool_use")
        );

        let max_tokens = frames(&[ccpa_finish(json!([{"text": "hi"}]), "MAX_TOKENS", None)]);
        assert_eq!(
            both_succeed(&max_tokens, true)["stop_reason"],
            json!("max_tokens")
        );

        let safety = frames(&[ccpa_finish(json!([{"text": "hi"}]), "SAFETY", None)]);
        assert_eq!(
            both_fail(&safety),
            StreamError::Stopped {
                finish_reason: "SAFETY".to_string(),
                finish_message: None,
            }
        );

        let blocked = frames(&[json!({
            "response": {
                "promptFeedback": {
                    "blockReason": "SAFETY",
                    "blockReasonMessage": "Blocked."
                }
            }
        })]);
        assert_eq!(
            both_fail(&blocked),
            StreamError::PromptBlocked {
                block_reason: "SAFETY".to_string(),
                upstream_message: Some("Blocked.".to_string()),
            }
        );
    }

    #[test]
    fn a_usage_only_chunk_after_the_finish_reason_is_counted() {
        // The usage arrives in a separate chunk after the finish reason, not
        // in the final content chunk; both modes must still report it.
        let input = frames(&[
            ccpa_finish(json!([{"text": "hi"}]), "STOP", None),
            json!({
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15
                    }
                },
                "traceId": "trace-1"
            }),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(message["stop_reason"], json!("end_turn"));
        assert_eq!(
            message["usage"],
            json!({
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }),
            "the trailing usage chunk is counted, not output_tokens 0"
        );
    }

    #[test]
    fn content_after_the_finish_reason_is_malformed() {
        let input = frames(&[
            ccpa_finish(json!([{"text": "hi"}]), "STOP", None),
            ccpa_parts(json!([{"text": "late"}])),
        ]);
        assert_eq!(
            both_fail(&input),
            StreamError::Malformed("the response streams content after the finish reason")
        );
    }

    #[test]
    fn a_stream_that_closes_before_finish_reason_fails() {
        let partial = frames(&[ccpa_parts(json!([{"text": "Partial"}]))]);
        let error = both_fail(&partial);
        assert_eq!(error, StreamError::Incomplete);
        assert_eq!(
            error.to_string(),
            "Google AI Ultra plan stream ended without a finishReason"
        );
        assert_eq!(error.kind(), "incomplete_stream");
        assert_eq!(both_fail(""), StreamError::Incomplete);
    }

    #[test]
    fn split_frames_at_every_byte_give_the_same_output() {
        let input = recorded_tool_turn();
        let whole = live(&input, true).unwrap_or_else(|error| panic!("{error}"));
        let bytes = input.as_bytes();
        let single: Vec<&[u8]> = bytes.chunks(1).collect();
        assert_eq!(live_chunks(&single, true).expect("byte chunks"), whole);
        for split in [1, 7, 33, bytes.len() / 2, bytes.len() - 1] {
            let (a, b) = bytes.split_at(split);
            assert_eq!(
                live_chunks(&[a, b], true).expect("two chunks"),
                whole,
                "split at {split}"
            );
        }
        let crlf = input.replace('\n', "\r\n");
        assert_eq!(live(&crlf, true).expect("CRLF framing"), whole);
    }

    #[test]
    fn events_after_completion_are_not_read() {
        let input = frames(&[ccpa_finish(json!([]), "STOP", None)])
            + "data: [DONE]\n\n"
            + &frames(&[ccpa_parts(json!([{"text": "late"}]))]);
        let message = both_succeed(&input, true);
        assert_eq!(message["content"], json!([]));
    }

    #[test]
    fn an_error_event_mid_stream_fails_with_the_mapped_type() {
        let error_frame = "data: {\"error\":{\"message\":\"Function call is missing a thought_signature.\",\"code\":\"invalid_request\"}}\n\n";
        let input = frames(&[ccpa_parts(json!([{"text": "partial"}]))]) + error_frame;
        let error = both_fail(&input);
        assert_eq!(
            error,
            StreamError::Upstream {
                code: Some("invalid_request".to_string()),
                upstream_message: Some("Function call is missing a thought_signature.".to_string()),
            }
        );
        assert_eq!(error.error_type(), "invalid_request_error");
        assert_eq!(error.kind(), "upstream_error");
        let event = parse_sse_events(
            &error.sse_event(&|text: &str| text.replace("thought_signature", "[x]")),
        );
        assert_eq!(event.len(), 1);
        assert_eq!(event[0].event.as_deref(), Some("error"));
        assert_eq!(
            serde_json::from_str::<Value>(&event[0].data).expect("JSON"),
            json!({"type":"error","error":{"type":"invalid_request_error","message":"Google AI Ultra plan stream reported an error (invalid_request): Function call is missing a [x]."}})
        );
        let tests = [
            (
                json!({"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"slow down"}}),
                "rate_limit_error",
            ),
            (
                json!([{"error":{"code":401,"status":"UNAUTHENTICATED"}}]),
                "authentication_error",
            ),
            (
                json!({"error":{"code":403,"status":"PERMISSION_DENIED"}}),
                "permission_error",
            ),
            (
                json!({"error":{"code":500,"status":"INTERNAL"}}),
                "api_error",
            ),
        ];
        for (event, expected) in tests {
            let error = both_fail(&frames(std::slice::from_ref(&event)));
            assert_eq!(error.error_type(), expected, "{event}");
        }
    }

    #[test]
    fn malformed_streams_fail() {
        let tests = [
            (
                "data: not json\n\n".to_string(),
                "an event is not valid JSON",
            ),
            (
                "data: [1,2]\n\n".to_string(),
                "an event is not a JSON object",
            ),
            (
                frames(&[json!({"unexpected": 1})]),
                "an event has none of the GenerateContentResponse fields",
            ),
            (
                frames(&[json!({"response": {"unexpected": 1}})]),
                "an event has none of the GenerateContentResponse fields",
            ),
            (
                frames(&[ccpa_parts(json!([{"functionCall": {"args": {}}}]))]),
                "a functionCall has no name",
            ),
            (
                frames(&[ccpa_parts(
                    json!([{"functionCall": {"name": "Read", "args": [1]}}]),
                )]),
                "functionCall args is not an object",
            ),
        ];
        for (input, expected) in tests {
            let error = both_fail(&input);
            assert_eq!(error, StreamError::Malformed(expected), "{input}");
        }
    }

    #[test]
    fn generated_ids_and_signature_envelopes_count_toward_the_content_limit() {
        for keep_content in [false, true] {
            let mut out = Vec::new();
            let mut state = TranslationState::new(config(true), keep_content);
            state.content_bytes = MAX_CONTENT_BYTES - BLOCK_CHARGE - "Read{}".len();
            assert!(matches!(
                state.function_call(String::new(), "Read".into(), json!({}), None, &mut out),
                Err(StreamError::Malformed(
                    "the response content exceeds the 16 MiB limit"
                ))
            ));

            let mut state = TranslationState::new(config(true), keep_content);
            state.content_bytes = MAX_CONTENT_BYTES - BLOCK_CHARGE - 4;
            state
                .thinking_signature("YWJj", &mut out)
                .expect("raw signature fits");
            assert!(matches!(
                state.close_open(&mut out),
                Err(StreamError::Malformed(
                    "the response content exceeds the 16 MiB limit"
                ))
            ));
        }
    }

    #[test]
    fn upstream_call_ids_pass_through_and_unusable_ones_are_replaced() {
        let tests = [
            ("call_1", "call_1"),
            ("", "toolu_msg_1_0"),
            ("has space", "toolu_msg_1_0"),
            ("ünïcode", "toolu_msg_1_0"),
        ];
        for (upstream, expected) in tests {
            let input = frames(&[ccpa_finish(
                json!([{"functionCall": {"id": upstream, "name": "Read", "args": {}}}]),
                "STOP",
                None,
            )]);
            let message = both_succeed(&input, true);
            assert_eq!(message["content"][0]["id"], json!(expected), "{upstream:?}");
        }
    }

    #[test]
    fn tool_use_ids_are_unique_within_a_response() {
        let calls = |ids: &[&str]| {
            let parts: Vec<Value> = ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    json!({"functionCall": {"id": id, "name": "Read", "args": {"n": index}}})
                })
                .collect();
            frames(&[ccpa_finish(Value::Array(parts), "STOP", None)])
        };
        let tests = [
            (
                "a repeated upstream ID",
                vec!["dup", "dup", "dup"],
                vec!["dup", "toolu_msg_1_1", "toolu_msg_1_2"],
            ),
            (
                "a passed-through ID equal to the next generated one",
                vec!["toolu_msg_1_1", ""],
                vec!["toolu_msg_1_1", "toolu_msg_1_2"],
            ),
            (
                "a passed-through ID equal to an earlier generated one",
                vec!["", "toolu_msg_1_0"],
                vec!["toolu_msg_1_0", "toolu_msg_1_1"],
            ),
            ("distinct upstream IDs", vec!["a", "b"], vec!["a", "b"]),
        ];
        for (case, upstream, expected) in tests {
            let message = both_succeed(&calls(&upstream), true);
            let ids: Vec<&str> = message["content"]
                .as_array()
                .expect("content is an array")
                .iter()
                .map(|block| block["id"].as_str().expect("tool_use id"))
                .collect();
            assert_eq!(ids, expected, "{case}");
        }
    }

    #[test]
    fn content_past_the_limit_fails_in_both_modes() {
        let chunk = "x".repeat(1024 * 1024);
        let mut events: Vec<Value> = (0..17)
            .map(|_| ccpa_parts(json!([{"text": chunk}])))
            .collect();
        events.push(ccpa_finish(json!([]), "STOP", None));
        let error = both_fail(&frames(&events));
        assert_eq!(
            error,
            StreamError::Malformed("the response content exceeds the 16 MiB limit")
        );
    }

    #[test]
    fn oversized_signature_falls_back_to_the_empty_envelope() {
        let input = frames(&[ccpa_finish(
            json!([{"thought": true, "thoughtSignature": "A".repeat(1024 * 1024 + 1)}]),
            "STOP",
            None,
        )]);
        let message = both_succeed(&input, false);
        assert_eq!(message["content"][0]["signature"], json!(envelope("")));
        assert!(
            message["content"][0]["signature"]
                .as_str()
                .is_some_and(|value| value.starts_with(ENVELOPE_PREFIX))
        );
    }

    #[test]
    fn streaming_translator_keeps_no_content() {
        let mut translator = LiveStreamTranslator::streaming(config(true));
        translator
            .push(recorded_tool_turn().as_bytes())
            .expect("translates");
        assert!(translator.is_finished());
        assert_eq!(translator.tool_calls(), 1);
        assert_eq!(translator.model(), "google-ai-plan/gemini-4-hydrogen");
        assert!(matches!(
            translator.response(),
            Err(StreamError::Malformed(_))
        ));
    }
}
