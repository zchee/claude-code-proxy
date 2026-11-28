//! Interactions API SSE step events → Anthropic Messages.
//!
//! One [`LiveStreamTranslator`] drives both the streaming response and the
//! non-streaming one ([`accumulate_response`]), so the two decide content
//! and stop reason the same way.
//!
//! The upstream sends `step.start`, `step.delta` and `step.stop` events keyed
//! by a step `index`, closed by `interaction.completed`. Steps may
//! interleave, but Anthropic content blocks cannot overlap, so blocks follow
//! the order in which steps started: only the oldest open step streams, and
//! the content of later steps is held until every step before it has
//! stopped. A step is never split across two blocks. A function call is
//! emitted whole when its step stops, once its arguments are known to be one
//! JSON object.
//!
//! | `interaction.completed` status | stop reason |
//! |---|---|
//! | `completed` | `end_turn` |
//! | `requires_action` | `tool_use` |
//! | `incomplete` | `max_tokens` |
//! | anything else | error |
//!
//! A stream that closes before `interaction.completed` is an error, never a
//! short answer, and so is an `error` event. The response is complete at
//! `interaction.completed`; anything after it is not read.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Map, Value, json};

use super::thought_signature::encode_envelope;
use crate::anthropic::sse::encode_sse_event;
use crate::providers::grok::translate::stream::SseDecoder;
pub use crate::providers::vertex::translate::stream::{
    BLOCK_CHARGE, MAX_CONTENT_BYTES, MAX_SSE_FRAME_BYTES, TranslatorConfig,
};

/// Longest upstream token (a status or an error code) kept in error text.
const MAX_TOKEN_CHARS: usize = 64;

/// Placeholder for an upstream token that is not a plain identifier.
const UNRECOGNIZED_TOKEN: &str = "unrecognized";

/// Longest upstream call ID passed through as the `tool_use` ID.
const MAX_CALL_ID_BYTES: usize = 256;

/// Why a stream could not be translated into a successful response.
///
/// `Display` contains only proxy-authored words and identifier tokens.
/// Upstream prose is kept in `upstream_message` and is added, sanitised,
/// only by [`StreamError::message_with`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    /// The stream carried an `error` event.
    #[error("Google AI Ultra plan stream reported an error{}", code_suffix(.code))]
    Upstream {
        /// The error `code` token, such as `invalid_request`.
        code: Option<String>,
        /// The upstream `error.message`. Unsanitised.
        upstream_message: Option<String>,
    },
    /// `interaction.completed` carried a status that is not a successful
    /// end of the turn.
    #[error("Google AI Ultra plan interaction ended with status {0}")]
    Status(String),
    /// The upstream closed before `interaction.completed`.
    #[error("Google AI Ultra plan stream ended before interaction.completed")]
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
            } => upstream_message.as_deref(),
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
                "too_many_requests" | "resource_exhausted" | "429" => "rate_limit_error",
                "invalid_request" | "invalid_argument" | "400" => "invalid_request_error",
                "unauthenticated" | "401" => "authentication_error",
                "permission_denied" | "403" => "permission_error",
                _ => "api_error",
            },
            _ => "api_error",
        }
    }

    /// A short label for logs and traffic-capture records.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Upstream { .. } => "upstream_error",
            Self::Status(_) => "interaction_status",
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

/// `interaction.usage`, as far as the translation uses it.
#[derive(Debug, Clone, Copy, Default)]
struct Usage {
    input: u64,
    cached: u64,
    output: u64,
    thought: u64,
}

impl Usage {
    fn from_value(value: &Value) -> Self {
        let field = |name: &str| value.get(name).and_then(Value::as_u64).unwrap_or(0);
        Self {
            input: field("total_input_tokens"),
            cached: field("total_cached_tokens"),
            output: field("total_output_tokens"),
            thought: field("total_thought_tokens"),
        }
    }

    /// `(input, cache_read, output)`: cached tokens are part of the input
    /// total, and thought tokens are output.
    fn anthropic(self) -> (u64, u64, u64) {
        (
            self.input.saturating_sub(self.cached),
            self.cached,
            self.output.saturating_add(self.thought),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Completed,
    RequiresAction,
    Incomplete,
}

impl Status {
    fn stop_reason(self) -> &'static str {
        match self {
            Self::Completed => "end_turn",
            Self::RequiresAction => "tool_use",
            Self::Incomplete => "max_tokens",
        }
    }
}

/// What a tracked step is.
#[derive(Debug)]
enum StepKind {
    Thought {
        /// Summary text not emitted yet.
        pending: String,
        signature: String,
    },
    Text {
        /// Text not emitted yet.
        pending: String,
    },
    Call {
        id: String,
        name: String,
        /// The `arguments` of `step.start`, used when no delta arrives.
        start_arguments: Value,
        /// The concatenated `arguments_delta` strings.
        arguments: String,
        streamed_arguments: bool,
    },
}

#[derive(Debug)]
struct TrackedStep {
    kind: StepKind,
    stopped: bool,
    /// The Anthropic block index, once the block has started.
    block: Option<usize>,
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

struct TranslationState {
    config: TranslatorConfig,
    keep_content: bool,
    content_bytes: usize,
    message_started: bool,
    finished: bool,
    blocks: Vec<Block>,
    block_count: usize,
    tool_count: usize,
    /// Every `tool_use` ID emitted so far, so no two blocks share one. Each
    /// entry is at most [`MAX_CALL_ID_BYTES`] or a proxy-made ID, and there
    /// is one per block, each of which already paid [`BLOCK_CHARGE`], so the
    /// set stays within a fixed multiple of the content budget.
    tool_ids: HashSet<String>,
    /// Steps in start order; the front one is the only one that streams.
    queue: VecDeque<TrackedStep>,
    /// Sequence number of the front of `queue`.
    head_seq: u64,
    /// Sequence number of every step that has started and not stopped, by
    /// its wire index.
    open: HashMap<u64, u64>,
    status: Option<Status>,
    usage: Option<Usage>,
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
            tool_count: 0,
            tool_ids: HashSet::new(),
            queue: VecDeque::new(),
            head_seq: 0,
            open: HashMap::new(),
            status: None,
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

    fn apply_event(&mut self, event: &Map<String, Value>) -> Result<Vec<u8>, StreamError> {
        let mut out = Vec::new();
        let event_type = event.get("event_type").and_then(Value::as_str);
        match event_type {
            Some("step.start") => {
                let index = wire_index(event)?;
                self.start(index, event.get("step").unwrap_or(&Value::Null))?;
            }
            Some("step.delta") => {
                let index = wire_index(event)?;
                self.delta(index, event.get("delta").unwrap_or(&Value::Null))?;
            }
            Some("step.stop") => {
                let index = wire_index(event)?;
                if let Some(seq) = self.open.remove(&index)
                    && let Some(step) = self.step_mut(seq)
                {
                    step.stopped = true;
                }
            }
            Some("interaction.completed") => {
                let interaction = event.get("interaction").unwrap_or(&Value::Null);
                if let Some(usage) = interaction.get("usage").filter(|usage| usage.is_object()) {
                    self.usage = Some(Usage::from_value(usage));
                }
                let status = interaction.get("status").and_then(Value::as_str).ok_or(
                    StreamError::Malformed("interaction.completed has no status"),
                )?;
                self.status = Some(match status {
                    "completed" => Status::Completed,
                    "requires_action" => Status::RequiresAction,
                    "incomplete" => Status::Incomplete,
                    other => return Err(StreamError::Status(upstream_token(other))),
                });
            }
            Some("error") => return Err(upstream_error(event.get("error"))),
            None if event.contains_key("error") => {
                return Err(upstream_error(event.get("error")));
            }
            // Other event types carry nothing the translation uses.
            _ => return Ok(out),
        }
        self.drain(&mut out)?;
        if self.status.is_some() {
            self.finalize(&mut out)?;
        }
        Ok(out)
    }

    fn step_mut(&mut self, seq: u64) -> Option<&mut TrackedStep> {
        let position = usize::try_from(seq.checked_sub(self.head_seq)?).ok()?;
        self.queue.get_mut(position)
    }

    fn start(&mut self, index: u64, step: &Value) -> Result<(), StreamError> {
        // Checked before the type, so a step the translator ignores cannot
        // start over an open one and have its deltas and stop applied there.
        if self.open.contains_key(&index) {
            return Err(StreamError::Malformed(
                "a step started at an index that is still open",
            ));
        }
        let kind = match step.get("type").and_then(Value::as_str) {
            Some("thought") => {
                let signature = str_field(step, "signature").to_string();
                self.count_content(signature.len())?;
                StepKind::Thought {
                    pending: String::new(),
                    signature,
                }
            }
            Some("model_output") => StepKind::Text {
                pending: String::new(),
            },
            Some("function_call") => {
                let name = str_field(step, "name");
                if name.is_empty() {
                    return Err(StreamError::Malformed("a function_call step has no name"));
                }
                let id = str_field(step, "id");
                let arguments = step.get("arguments");
                // The call is held until its step and every step before it
                // stop, so what it holds is charged now; otherwise many
                // calls queued behind an open step could pass the budget.
                // A missing `arguments` is held as `{}`.
                let arguments_len = arguments.map_or(2, compact_len);
                self.count_content(
                    id.len()
                        .saturating_add(name.len())
                        .saturating_add(arguments_len),
                )?;
                let start_arguments = arguments.cloned().unwrap_or_else(|| json!({}));
                StepKind::Call {
                    id: id.to_string(),
                    name: name.to_string(),
                    start_arguments,
                    arguments: String::new(),
                    streamed_arguments: false,
                }
            }
            // Server-side tool steps (code execution, URL context, search)
            // are not requested and not shown.
            _ => return Ok(()),
        };
        self.count_content(BLOCK_CHARGE)?;
        let seq = self.head_seq + self.queue.len() as u64;
        self.open.insert(index, seq);
        self.queue.push_back(TrackedStep {
            kind,
            stopped: false,
            block: None,
        });
        Ok(())
    }

    fn delta(&mut self, index: u64, delta: &Value) -> Result<(), StreamError> {
        let Some(&seq) = self.open.get(&index) else {
            return Ok(());
        };
        let emit_thinking = self.config.emit_thinking;
        let delta_type = delta.get("type").and_then(Value::as_str);
        let appended = {
            let Some(step) = self.step_mut(seq) else {
                return Ok(());
            };
            match (&mut step.kind, delta_type) {
                (StepKind::Text { pending }, Some("text")) => {
                    let text = delta
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    pending.push_str(text);
                    text.len()
                }
                (StepKind::Thought { pending, .. }, Some("thought_summary")) => {
                    let content = delta.get("content");
                    let text = content
                        .filter(|content| {
                            content.get("type").and_then(Value::as_str) == Some("text")
                        })
                        .and_then(|content| content.get("text"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if emit_thinking {
                        pending.push_str(text);
                        text.len()
                    } else {
                        0
                    }
                }
                (StepKind::Thought { signature, .. }, Some("thought_signature")) => {
                    let part = delta
                        .get("signature")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    signature.push_str(part);
                    part.len()
                }
                (
                    StepKind::Call {
                        arguments,
                        streamed_arguments,
                        ..
                    },
                    Some("arguments_delta"),
                ) => {
                    let part = delta
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    arguments.push_str(part);
                    *streamed_arguments |= !part.is_empty();
                    part.len()
                }
                _ => 0,
            }
        };
        self.count_content(appended)
    }

    /// Emits what the front steps allow: the new content of the front step,
    /// then, while the front step has stopped, its end and the same for the
    /// step after it.
    fn drain(&mut self, out: &mut Vec<u8>) -> Result<(), StreamError> {
        while let Some(mut step) = self.queue.pop_front() {
            self.emit_pending(&mut step, out);
            if !step.stopped {
                self.queue.push_front(step);
                return Ok(());
            }
            self.head_seq += 1;
            self.end_step(step, out)?;
        }
        Ok(())
    }

    fn emit_pending(&mut self, step: &mut TrackedStep, out: &mut Vec<u8>) {
        match &mut step.kind {
            StepKind::Thought { pending, .. } if !pending.is_empty() => {
                let text = std::mem::take(pending);
                let index = *step.block.get_or_insert_with(|| {
                    self.start_block(
                        out,
                        Block::Thinking {
                            text: String::new(),
                            signature: String::new(),
                        },
                        json!({"type":"thinking","thinking":"","signature":""}),
                    )
                });
                if self.keep_content
                    && let Some(Block::Thinking { text: stored, .. }) = self.blocks.get_mut(index)
                {
                    stored.push_str(&text);
                }
                emit(
                    out,
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":text}}),
                );
            }
            StepKind::Text { pending } if !pending.is_empty() => {
                let text = std::mem::take(pending);
                let index = *step.block.get_or_insert_with(|| {
                    self.start_block(
                        out,
                        Block::Text {
                            text: String::new(),
                        },
                        json!({"type":"text","text":""}),
                    )
                });
                if self.keep_content
                    && let Some(Block::Text { text: stored }) = self.blocks.get_mut(index)
                {
                    stored.push_str(&text);
                }
                emit(
                    out,
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
                );
            }
            _ => {}
        }
    }

    fn end_step(&mut self, step: TrackedStep, out: &mut Vec<u8>) -> Result<(), StreamError> {
        match step.kind {
            StepKind::Thought { signature, .. } => {
                // The block carries the signature back with the history; a
                // thought without one and without forwarded text leaves no
                // block. An oversized signature is dropped, so the next
                // request falls back to the skip sentinel.
                let index = match step.block {
                    Some(index) => index,
                    None if !signature.is_empty() => self.start_block(
                        out,
                        Block::Thinking {
                            text: String::new(),
                            signature: String::new(),
                        },
                        json!({"type":"thinking","thinking":"","signature":""}),
                    ),
                    None => return Ok(()),
                };
                let envelope = encode_envelope(&self.config.signature_model, &signature)
                    .or_else(|| encode_envelope(&self.config.signature_model, ""))
                    .unwrap_or_default();
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
                    && let Some(Block::Thinking { signature, .. }) = self.blocks.get_mut(index)
                {
                    *signature = envelope;
                }
            }
            StepKind::Text { .. } => {
                if let Some(index) = step.block {
                    emit(
                        out,
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":index}),
                    );
                }
            }
            StepKind::Call {
                id,
                name,
                start_arguments,
                arguments,
                streamed_arguments,
            } => {
                let input = if streamed_arguments {
                    serde_json::from_str::<Value>(&arguments).map_err(|_| {
                        StreamError::Malformed("function_call arguments are not valid JSON")
                    })?
                } else {
                    start_arguments
                };
                if !input.is_object() {
                    return Err(StreamError::Malformed(
                        "function_call arguments are not a JSON object",
                    ));
                }
                // The upstream ID and name were charged when the step
                // started; a replacement ID is proxy-made and short, and the
                // block already paid `BLOCK_CHARGE`.
                let id = if is_valid_call_id(&id) && !self.tool_ids.contains(&id) {
                    id
                } else {
                    self.unused_call_id()
                };
                let compact = input.to_string();
                self.tool_ids.insert(id.clone());
                self.tool_count += 1;
                let content_block = json!({"type":"tool_use","id":id,"name":name,"input":{}});
                let index = self.start_block(out, Block::Tool { id, name, input }, content_block);
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
            }
        }
        Ok(())
    }

    /// A `toolu_<message id>_<n>` ID no block of this response has, with the
    /// lowest `n` from the number of calls emitted so far. The set has one
    /// entry per emitted call, so a free ID is found within that many tries
    /// plus one.
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

    /// Starts the next block; `block` is kept only by the non-streaming
    /// translator.
    fn start_block(&mut self, out: &mut Vec<u8>, block: Block, content_block: Value) -> usize {
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
        index
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

    /// Ends every step still open and completes the message.
    fn finalize(&mut self, out: &mut Vec<u8>) -> Result<(), StreamError> {
        let Some(status) = self.status else {
            return Err(StreamError::Incomplete);
        };
        // An output limit can cut a function call short; a call that never
        // stopped is dropped rather than sent with arguments the model did
        // not finish.
        if status == Status::Incomplete {
            self.queue
                .retain(|step| step.stopped || !matches!(step.kind, StepKind::Call { .. }));
        }
        for step in &mut self.queue {
            step.stopped = true;
        }
        self.open.clear();
        self.drain(out)?;
        self.ensure_message_start(out);
        // Input usage is published only when the upstream reported it, so the
        // client keeps the seeded estimate otherwise.
        let usage = match self.usage.map(Usage::anthropic) {
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
            json!({"type":"message_delta","delta":{"stop_reason":status.stop_reason(),"stop_sequence":null},"usage":usage}),
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
        let Some(status) = self.status.filter(|_| self.finished) else {
            return Err(StreamError::Incomplete);
        };
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
        let (input, cache_read, output) =
            self.usage
                .map(Usage::anthropic)
                .unwrap_or((self.config.estimated_input_tokens, 0, 0));
        Ok(json!({
            "id":self.config.message_id,
            "type":"message",
            "role":"assistant",
            "model":self.config.model,
            "content":content,
            "stop_reason":status.stop_reason(),
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

fn str_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn wire_index(event: &Map<String, Value>) -> Result<u64, StreamError> {
    event
        .get("index")
        .and_then(Value::as_u64)
        .ok_or(StreamError::Malformed("a step event has no index"))
}

/// The length of `value` serialised as compact JSON, without building the
/// string.
fn compact_len(value: &Value) -> usize {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(buf.len());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    // A `Value` always serialises and this writer never fails, so there is
    // no error to report.
    let _ = serde_json::to_writer(&mut counter, value);
    counter.0
}

fn is_valid_call_id(id: &str) -> bool {
    (1..=MAX_CALL_ID_BYTES).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Keeps an upstream token (a status or an error code) only when it looks
/// like one.
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
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(|code| match code {
            Value::String(code) => Some(upstream_token(code)),
            Value::Number(code) => Some(upstream_token(&code.to_string())),
            _ => None,
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

fn emit(out: &mut Vec<u8>, event: &str, value: Value) {
    out.extend(encode_sse_event(Some(event), &value.to_string()));
}

/// Incremental translator from Interactions SSE bytes to Anthropic SSE
/// bytes.
///
/// Feed every upstream chunk to [`Self::push`] and send what it returns.
/// Once [`Self::is_finished`] is true the response is complete and the
/// upstream need not be read further. When the upstream closes first, call
/// [`Self::finish`], which fails with [`StreamError::Incomplete`]. On any
/// error, send [`StreamError::sse_event`] as the last event (or answer the
/// non-streaming request with the error) and drop the translator.
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
    ///
    /// # Errors
    ///
    /// [`StreamError::Upstream`] for an `error` event,
    /// [`StreamError::Status`] for an unsuccessful final status, and
    /// [`StreamError::Malformed`] for a stream that breaks the documented
    /// shape or the content limit.
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
            // Google sends `[DONE]` after `interaction.completed`.
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let value: Value = serde_json::from_str(data)
                .map_err(|_| StreamError::Malformed("an event is not valid JSON"))?;
            let Value::Object(object) = value else {
                return Err(StreamError::Malformed("an event is not a JSON object"));
            };
            out.extend(self.state.apply_event(&object)?);
            if self.state.finished {
                break;
            }
        }
        Ok(out)
    }

    /// Completes the response after the upstream closed.
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] when `interaction.completed` never
    /// arrived, whatever was buffered.
    pub fn finish(&mut self) -> Result<Vec<u8>, StreamError> {
        if self.state.finished {
            Ok(Vec::new())
        } else {
            Err(StreamError::Incomplete)
        }
    }

    /// Whether `interaction.completed` arrived and the message is complete.
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
    ///
    /// # Errors
    ///
    /// [`StreamError::Incomplete`] before completion, and
    /// [`StreamError::Malformed`] for a translator made with
    /// [`Self::streaming`].
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

    /// One SSE frame in the shape Google sends (`event:` line plus `data:`
    /// JSON).
    fn frame(data: Value) -> String {
        let name = data["event_type"].as_str().unwrap_or("message").to_string();
        format!("event: {name}\ndata: {data}\n\n")
    }

    fn frames(events: &[Value]) -> String {
        events.iter().cloned().map(frame).collect()
    }

    fn start(index: u64, step: Value) -> Value {
        json!({"event_type":"step.start","index":index,"step":step})
    }

    fn delta(index: u64, delta: Value) -> Value {
        json!({"event_type":"step.delta","index":index,"delta":delta})
    }

    fn stop(index: u64) -> Value {
        json!({"event_type":"step.stop","index":index})
    }

    fn text(index: u64, text: &str) -> Value {
        delta(index, json!({"type":"text","text":text}))
    }

    fn summary(index: u64, text: &str) -> Value {
        delta(
            index,
            json!({"type":"thought_summary","content":{"type":"text","text":text}}),
        )
    }

    fn completed(status: &str) -> Value {
        json!({"event_type":"interaction.completed","interaction":{"status":status}})
    }

    /// The recorded gemini-4-hydrogen tool turn of the reference test.
    fn recorded_tool_turn() -> String {
        frames(&[
            start(0, json!({"type":"thought"})),
            summary(0, "Need the file."),
            delta(0, json!({"type":"thought_signature","signature":"c2ln"})),
            stop(0),
            start(1, json!({"type":"model_output"})),
            text(1, "Reading it."),
            stop(1),
            start(
                2,
                json!({"type":"function_call","id":"call_1","name":"read_file","arguments":{}}),
            ),
            delta(
                2,
                json!({"type":"arguments_delta","arguments":"{\"path\":"}),
            ),
            delta(2, json!({"type":"arguments_delta","arguments":"\"a.py\"}"})),
            stop(2),
            json!({"event_type":"interaction.completed","interaction":{
                "status":"requires_action",
                "usage":{"total_tokens":184,"total_input_tokens":114,"total_cached_tokens":10,"total_output_tokens":26,"total_thought_tokens":44}
            }}),
        ]) + "event: done\ndata: [DONE]\n\n"
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

    /// Rebuilds the content, stop reason and usage a client would assemble
    /// from the live events, checking that blocks never overlap.
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

    /// Asserts that the live and accumulated translations agree and returns
    /// the accumulated message.
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
            // Without upstream usage the stream keeps the seeded estimate
            // and the message reports it.
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
    fn thought_text_and_function_call_steps_become_one_message() {
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
            start(0, json!({"type":"thought"})),
            summary(0, "hidden"),
            stop(0),
            start(1, json!({"type":"model_output"})),
            text(1, "answer"),
            stop(1),
            completed("completed"),
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
    fn a_signature_on_step_start_is_carried() {
        let input = frames(&[
            start(0, json!({"type":"thought","signature":"c3RhcnQ="})),
            stop(0),
            completed("completed"),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(
            message["content"],
            json!([{"type":"thinking","thinking":"","signature":envelope("c3RhcnQ=")}])
        );
    }

    #[test]
    fn statuses_map_to_stop_reasons_or_errors() {
        let tests = [
            ("completed", Some("end_turn")),
            ("requires_action", Some("tool_use")),
            ("incomplete", Some("max_tokens")),
            ("failed", None),
            ("cancelled", None),
        ];
        for (status, expected) in tests {
            let input = frames(&[
                start(0, json!({"type":"model_output"})),
                text(0, "hi"),
                stop(0),
                completed(status),
            ]);
            match expected {
                Some(reason) => {
                    let message = both_succeed(&input, true);
                    assert_eq!(message["stop_reason"], json!(reason), "{status}");
                    assert_eq!(
                        message["usage"]["input_tokens"],
                        json!(42),
                        "{status}: estimate kept without usage"
                    );
                }
                None => assert_eq!(
                    both_fail(&input),
                    StreamError::Status(status.to_string()),
                    "{status}"
                ),
            }
        }
        let weird = frames(&[completed("Not A Token!")]);
        assert_eq!(
            both_fail(&weird),
            StreamError::Status("unrecognized".to_string())
        );
    }

    #[test]
    fn a_stream_that_closes_before_completion_fails() {
        let partial = frames(&[start(0, json!({"type":"model_output"})), text(0, "Partial")]);
        let error = both_fail(&partial);
        assert_eq!(error, StreamError::Incomplete);
        assert_eq!(
            error.to_string(),
            "Google AI Ultra plan stream ended before interaction.completed"
        );
        let cut_frame = format!("{partial}event: step.delta\ndata: {{\"event_type\"");
        assert_eq!(
            both_fail(&cut_frame),
            StreamError::Incomplete,
            "a buffered partial frame changes nothing"
        );
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
    fn interleaved_steps_never_overlap_and_keep_start_order() {
        let input = frames(&[
            start(0, json!({"type":"thought"})),
            summary(0, "think "),
            start(1, json!({"type":"model_output"})),
            text(1, "early text "),
            start(
                2,
                json!({"type":"function_call","id":"c1","name":"Read","arguments":{}}),
            ),
            delta(2, json!({"type":"arguments_delta","arguments":"{\"a\":1}"})),
            summary(0, "more"),
            stop(2),
            text(1, "late text"),
            delta(0, json!({"type":"thought_signature","signature":"czA="})),
            stop(0),
            stop(1),
            completed("requires_action"),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(
            message["content"],
            json!([
                {"type":"thinking","thinking":"think more","signature":envelope("czA=")},
                {"type":"text","text":"early text late text"},
                {"type":"tool_use","id":"c1","name":"Read","input":{"a":1}}
            ])
        );
        let events = live(&input, true).expect("live");
        let first_text = events
            .iter()
            .position(|(_, data)| data["delta"]["type"] == "text_delta")
            .expect("text delta");
        let thinking_stop = events
            .iter()
            .position(|(name, data)| name == "content_block_stop" && data["index"] == 0)
            .expect("thinking stop");
        assert!(
            thinking_stop < first_text,
            "held text follows the thinking block"
        );
    }

    #[test]
    fn steps_still_open_at_completion_are_closed_and_indexes_may_be_reused() {
        let input = frames(&[
            start(0, json!({"type":"model_output"})),
            text(0, "one"),
            stop(0),
            start(0, json!({"type":"model_output"})),
            text(0, "two"),
            start(
                1,
                json!({"type":"function_call","id":"c","name":"Read","arguments":{"path":"x"}}),
            ),
            completed("requires_action"),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(
            message["content"],
            json!([
                {"type":"text","text":"one"},
                {"type":"text","text":"two"},
                {"type":"tool_use","id":"c","name":"Read","input":{"path":"x"}}
            ]),
            "a call without deltas uses the arguments of step.start"
        );
    }

    #[test]
    fn a_call_cut_short_by_the_output_limit_is_dropped_and_the_turn_ends_at_max_tokens() {
        let input = frames(&[
            start(0, json!({"type":"model_output"})),
            text(0, "Writing the file"),
            start(
                1,
                json!({"type":"function_call","id":"c","name":"Write","arguments":{}}),
            ),
            delta(
                1,
                json!({"type":"arguments_delta","arguments":"{\"content\":\"half"}),
            ),
            start(
                2,
                json!({"type":"function_call","id":"d","name":"Read","arguments":{}}),
            ),
            completed("incomplete"),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(message["stop_reason"], json!("max_tokens"));
        assert_eq!(
            message["content"],
            json!([{"type":"text","text":"Writing the file"}]),
            "no tool_use is sent for a call that never stopped"
        );
        let stopped = frames(&[
            start(
                0,
                json!({"type":"function_call","id":"c","name":"Read","arguments":{}}),
            ),
            delta(0, json!({"type":"arguments_delta","arguments":"{\"a\":1}"})),
            stop(0),
            completed("incomplete"),
        ]);
        let message = both_succeed(&stopped, true);
        assert_eq!(
            message["content"],
            json!([{"type":"tool_use","id":"c","name":"Read","input":{"a":1}}]),
            "a call that stopped before the limit is kept"
        );
    }

    #[test]
    fn unknown_events_and_server_side_steps_are_ignored() {
        let input = frames(&[
            json!({"event_type":"interaction.start","interaction":{"id":"i1"}}),
            start(0, json!({"type":"code_execution_call","code":"print(1)"})),
            delta(0, json!({"type":"text","text":"ignored"})),
            stop(0),
            start(1, json!({"type":"model_output"})),
            delta(1, json!({"type":"something_new"})),
            text(1, "kept"),
            stop(1),
            json!({"event_type":"interaction.status_update"}),
            completed("completed"),
        ]);
        let message = both_succeed(&input, true);
        assert_eq!(message["content"], json!([{"type":"text","text":"kept"}]));
    }

    #[test]
    fn an_ignored_step_at_a_free_index_leaves_the_output_unchanged() {
        let without = frames(&[
            start(0, json!({"type":"model_output"})),
            text(0, "one "),
            text(0, "two"),
            stop(0),
            completed("completed"),
        ]);
        let with = frames(&[
            start(0, json!({"type":"model_output"})),
            text(0, "one "),
            start(5, json!({"type":"code_execution_call","code":"print(1)"})),
            delta(5, json!({"type":"text","text":"ignored"})),
            text(0, "two"),
            stop(5),
            stop(0),
            completed("completed"),
        ]);
        let message = both_succeed(&with, true);
        assert_eq!(
            message["content"],
            json!([{"type":"text","text":"one two"}])
        );
        assert_eq!(message, both_succeed(&without, true));
        assert_eq!(
            live(&with, true).expect("live with the ignored step"),
            live(&without, true).expect("live without it"),
            "the ignored step adds no event"
        );
    }

    #[test]
    fn events_after_completion_are_not_read() {
        let input = frames(&[completed("completed")])
            + "event: done\ndata: [DONE]\n\n"
            + &frames(&[start(0, json!({"type":"model_output"})), text(0, "late")]);
        let message = both_succeed(&input, true);
        assert_eq!(message["content"], json!([]));
        let mut translator = LiveStreamTranslator::streaming(config(true));
        translator
            .push(frames(&[completed("completed")]).as_bytes())
            .expect("completes");
        assert!(translator.is_finished());
        assert_eq!(
            translator
                .push(b"garbage that is not SSE JSON\n\n")
                .expect("ignored"),
            Vec::<u8>::new()
        );
        assert_eq!(translator.finish().expect("finished"), Vec::<u8>::new());
    }

    #[test]
    fn an_error_event_mid_stream_fails_with_the_mapped_type() {
        let error_frame = "event: error\ndata: {\"error\":{\"message\":\"Function call is missing a thought_signature.\",\"code\":\"invalid_request\"},\"event_type\":\"error\"}\n\n";
        let input =
            frames(&[start(0, json!({"type":"model_output"})), text(0, "partial")]) + error_frame;
        let error = both_fail(&input);
        assert_eq!(
            error,
            StreamError::Upstream {
                code: Some("invalid_request".to_string()),
                upstream_message: Some("Function call is missing a thought_signature.".to_string()),
            }
        );
        assert_eq!(error.error_type(), "invalid_request_error");
        let event = parse_sse_events(
            &error.sse_event(&|text: &str| text.replace("thought_signature", "[x]")),
        );
        assert_eq!(event.len(), 1);
        assert_eq!(event[0].event.as_deref(), Some("error"));
        assert_eq!(
            serde_json::from_str::<Value>(&event[0].data).expect("JSON"),
            json!({"type":"error","error":{"type":"invalid_request_error","message":"Google AI Ultra plan stream reported an error (invalid_request): Function call is missing a [x]."}}),
            "the upstream message goes through the sanitiser"
        );
        let tests = [
            (
                json!({"event_type":"error","error":{"code":"too_many_requests","message":"slow down"}}),
                "rate_limit_error",
            ),
            (
                json!({"event_type":"error","error":{"code":"unauthenticated"}}),
                "authentication_error",
            ),
            (
                json!({"event_type":"error","error":{"code":"permission_denied"}}),
                "permission_error",
            ),
            (
                json!({"event_type":"error","error":{"code":"internal"}}),
                "api_error",
            ),
            (json!({"event_type":"error"}), "api_error"),
            (
                json!({"error":{"code":429,"message":"no event_type"}}),
                "rate_limit_error",
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
                frames(&[json!({"event_type":"step.start","step":{"type":"model_output"}})]),
                "a step event has no index",
            ),
            (
                frames(&[start(
                    0,
                    json!({"type":"function_call","id":"c","arguments":{}}),
                )]),
                "a function_call step has no name",
            ),
            (
                frames(&[
                    start(0, json!({"type":"model_output"})),
                    start(0, json!({"type":"model_output"})),
                ]),
                "a step started at an index that is still open",
            ),
            (
                frames(&[
                    start(0, json!({"type":"model_output"})),
                    start(0, json!({"type":"code_execution_call","code":"print(1)"})),
                ]),
                "a step started at an index that is still open",
            ),
            (
                frames(&[
                    start(0, json!({"type":"function_call","id":"c","name":"Read"})),
                    start(0, json!({"type":"url_context_call"})),
                    stop(0),
                    completed("requires_action"),
                ]),
                "a step started at an index that is still open",
            ),
            (
                frames(&[
                    start(0, json!({"type":"function_call","id":"c","name":"Read"})),
                    delta(0, json!({"type":"arguments_delta","arguments":"{\"a\":"})),
                    stop(0),
                    completed("requires_action"),
                ]),
                "function_call arguments are not valid JSON",
            ),
            (
                frames(&[
                    start(
                        0,
                        json!({"type":"function_call","id":"c","name":"Read","arguments":[1]}),
                    ),
                    stop(0),
                    completed("requires_action"),
                ]),
                "function_call arguments are not a JSON object",
            ),
            (
                frames(&[json!({"event_type":"interaction.completed","interaction":{}})]),
                "interaction.completed has no status",
            ),
        ];
        for (input, expected) in tests {
            let error = both_fail(&input);
            assert_eq!(error, StreamError::Malformed(expected), "{input}");
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
            let input = frames(&[
                start(
                    0,
                    json!({"type":"function_call","id":upstream,"name":"Read","arguments":{}}),
                ),
                stop(0),
                completed("requires_action"),
            ]);
            let message = both_succeed(&input, true);
            assert_eq!(message["content"][0]["id"], json!(expected), "{upstream:?}");
        }
    }

    #[test]
    fn tool_use_ids_are_unique_within_a_response() {
        let calls = |ids: &[&str]| {
            let mut events: Vec<Value> = Vec::new();
            for (index, id) in (0u64..).zip(ids) {
                events.push(start(
                    index,
                    json!({"type":"function_call","id":id,"name":"Read","arguments":{"n":index}}),
                ));
                events.push(stop(index));
            }
            events.push(completed("requires_action"));
            frames(&events)
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
            let inputs: Vec<Value> = message["content"]
                .as_array()
                .expect("content is an array")
                .iter()
                .map(|block| block["input"].clone())
                .collect();
            let calls: Vec<Value> = (0..upstream.len()).map(|n| json!({"n":n})).collect();
            assert_eq!(inputs, calls, "{case}: each block keeps its own call");
        }
    }

    #[test]
    fn function_calls_are_charged_when_they_start() {
        let limit = StreamError::Malformed("the response content exceeds the 16 MiB limit");
        let chunk = "x".repeat(1024 * 1024);
        let call = |index: u64| {
            start(
                index,
                json!({"type":"function_call","id":format!("c{index}"),"name":"Write","arguments":{"content":chunk}}),
            )
        };
        let mut queued = vec![start(0, json!({"type":"model_output"})), text(0, "open")];
        for index in 1..=17 {
            queued.push(call(index));
            queued.push(stop(index));
        }
        assert_eq!(
            both_fail(&frames(&queued)),
            limit,
            "calls held behind an open text step fail before the stream ends"
        );
        let mut emitted = Vec::new();
        for index in 0..17 {
            emitted.push(call(index));
            emitted.push(stop(index));
        }
        emitted.push(completed("requires_action"));
        assert_eq!(
            both_fail(&frames(&emitted)),
            limit,
            "calls emitted as they stop count against the same limit"
        );
        let large = frames(&[
            start(
                0,
                json!({"type":"function_call","id":"c","name":"Write","arguments":{"content":"y".repeat(15 * 1024 * 1024)}}),
            ),
            stop(0),
            completed("requires_action"),
        ]);
        let message = both_succeed(&large, true);
        assert_eq!(message["content"][0]["id"], json!("c"));
        assert_eq!(
            message["content"][0]["input"]["content"]
                .as_str()
                .map(str::len),
            Some(15 * 1024 * 1024),
            "a call within the limit is charged once"
        );
    }

    #[test]
    fn content_past_the_limit_fails_in_both_modes() {
        let chunk = "x".repeat(1024 * 1024);
        let mut events = vec![start(0, json!({"type":"model_output"}))];
        events.extend((0..17).map(|_| text(0, &chunk)));
        events.push(stop(0));
        events.push(completed("completed"));
        let error = both_fail(&frames(&events));
        assert_eq!(
            error,
            StreamError::Malformed("the response content exceeds the 16 MiB limit")
        );
    }

    #[test]
    fn oversized_signature_falls_back_to_the_empty_envelope() {
        let input = frames(&[
            start(0, json!({"type":"thought"})),
            delta(
                0,
                json!({"type":"thought_signature","signature":"A".repeat(1024 * 1024 + 1)}),
            ),
            stop(0),
            completed("completed"),
        ]);
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
