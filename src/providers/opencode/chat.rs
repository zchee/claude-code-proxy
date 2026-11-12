use std::borrow::Cow;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use base64::Engine;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde::Serialize;
use serde_json::{Value, json};

use super::client::{OpenCodeError, OpenCodeResponse};
use crate::anthropic::{
    schema::MessagesRequest,
    sse::{encode_sse_event, parse_sse_events},
};
use crate::monitor::{MonitorHandle, usage_from_anthropic_sse};
use crate::providers::{
    grok::translate::stream::SseDecoder,
    translate_shared::{
        ContentBlock, flatten_system_text, image_source_to_url, normalize_content, read_effort,
    },
};
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

const DEFAULT_MAX_TOKENS: u32 = 32_000;

/// How long the upstream may stay open after the terminal event (`[DONE]`,
/// `message_stop`, `response.completed`) under [`TerminalPolicy::Strict`]
/// before the response is completed without waiting for it to close.
pub const POST_TERMINAL_READ_LIMIT: Duration = Duration::from_secs(2);

/// Most bytes read after the terminal event under
/// [`TerminalPolicy::Strict`]; more is an error.
pub const POST_TERMINAL_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Cleans upstream prose before it reaches a traffic capture or an error
/// returned to the client. It is the same type as the Vertex provider's
/// sanitiser, so one closure serves both.
pub type Sanitizer = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// The names the stream wiring of this module, [`super::messages`] and
/// [`super::responses`] writes: the provider name in error messages and the
/// stem of traffic-capture file names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamLabels {
    /// Provider name in error messages, as in `OpenCode Go stream is
    /// invalid`.
    pub provider: &'static str,
    /// Stem of traffic-capture file names, as in
    /// `061-opencode-stream-summary`.
    pub capture_stem: &'static str,
}

impl StreamLabels {
    /// The labels OpenCode Go uses.
    pub const OPENCODE_GO: Self = Self {
        provider: "OpenCode Go",
        capture_stem: "opencode",
    };
}

impl Default for StreamLabels {
    fn default() -> Self {
        Self::OPENCODE_GO
    }
}

/// What happens after the terminal event of an upstream stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalPolicy {
    /// OpenCode Go's rules. Chat Completions settles at `[DONE]` (or at EOF
    /// after a `finish_reason`) and accepts OpenCode's cost metadata after
    /// `[DONE]`; Messages settles at `message_stop`; Responses waits for EOF
    /// after the completion and accepts `ping` events in between.
    #[default]
    OpenCodeGo,
    /// After the terminal event, keep reading until the upstream closes, for
    /// at most [`POST_TERMINAL_READ_LIMIT`] and [`POST_TERMINAL_MAX_BYTES`].
    /// Any event in that window, a second `[DONE]` included, is an error,
    /// and the terminal output reaches the client only after the window.
    /// When the upstream stays open, the response completes at the read
    /// limit. A partial frame still buffered then is discarded by the Chat
    /// Completions and Responses translators; the Messages pass-through,
    /// which forwards only complete frames exactly as received, fails
    /// instead. Bytes are counted from the end of the terminal frame,
    /// whichever chunk they arrive in. Responses
    /// frames must also carry a `type` listed in
    /// [`super::responses::RESPONSES_HANDLED_EVENT_TYPES`] or
    /// [`super::responses::RESPONSES_HARMLESS_EVENT_TYPES`], a snapshot
    /// must not carry text that no delta delivered. When the Responses
    /// translator closes the message by itself (it repairs a `Read` call
    /// whose arguments stalled on whitespace), the stream ends there: the
    /// closing is sent at once, the upstream is not read again and is
    /// dropped, and what it would have sent afterwards is neither waited for
    /// nor checked. The read window and the byte limit apply only to a
    /// completion the upstream sends (see
    /// [`super::responses::stream_body_with`]).
    Strict,
}

/// How a translator receives the upstream response.
#[derive(Clone, Default)]
pub struct ReceivePolicy {
    /// The rules after the terminal event.
    pub terminal: TerminalPolicy,
    /// Applied to every string under a `message` key of an upstream event
    /// before the event is captured, and to the error a buffered response
    /// returns. `None` keeps both as the upstream sent them.
    pub sanitize: Option<Sanitizer>,
}

impl ReceivePolicy {
    fn is_strict(&self) -> bool {
        self.terminal == TerminalPolicy::Strict
    }

    pub(super) fn upstream_value<'a>(&self, value: &'a Value) -> Cow<'a, Value> {
        match self.sanitize.as_ref() {
            Some(sanitize) => {
                let mut value = value.clone();
                sanitize_messages(&mut value, &**sanitize);
                Cow::Owned(value)
            }
            None => Cow::Borrowed(value),
        }
    }

    pub(super) fn upstream_error(&self, error: anyhow::Error) -> anyhow::Error {
        match self.sanitize.as_ref() {
            Some(sanitize) => anyhow::anyhow!(sanitize(&format!("{error:#}"))),
            None => error,
        }
    }
}

/// A copy of `value` with every string under a `message` key, at any depth,
/// passed through `sanitize`: the cleaning [`ReceivePolicy::sanitize`]
/// applies to an upstream event before it is captured. Other strings are
/// kept as they are.
pub fn sanitize_upstream_messages(value: &Value, sanitize: &dyn Fn(&str) -> String) -> Value {
    let mut value = value.clone();
    sanitize_messages(&mut value, sanitize);
    value
}

fn sanitize_messages(value: &mut Value, sanitize: &dyn Fn(&str) -> String) {
    match value {
        Value::Object(object) => {
            for (key, item) in object.iter_mut() {
                match item {
                    Value::String(text) if key == "message" => *text = sanitize(text),
                    _ => sanitize_messages(item, sanitize),
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                sanitize_messages(item, sanitize);
            }
        }
        _ => {}
    }
}

/// Labels and receive policy for one upstream stream. [`Default`] is OpenCode
/// Go's.
#[derive(Clone, Default)]
pub struct StreamWiring {
    /// Names in error messages and traffic-capture files.
    pub labels: StreamLabels,
    /// How the upstream response is received.
    pub receive: ReceivePolicy,
}

pub(super) type UpstreamStream = Pin<Box<dyn Stream<Item = Result<Bytes, OpenCodeError>> + Send>>;

/// Boxes any upstream byte stream for the stream states, which only note
/// that a transport error happened and never read it.
pub(super) fn box_upstream<S, E>(upstream: S) -> UpstreamStream
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
{
    Box::pin(upstream.map(|item| {
        item.map_err(|_| OpenCodeError {
            status: http::StatusCode::BAD_GATEWAY,
            retry_after: None,
            message: "upstream stream failed".to_string(),
        })
    }))
}

/// The wiring of one stream plus what it has to remember between reads.
#[derive(Default)]
pub(super) struct ReceiveContext {
    pub(super) wiring: StreamWiring,
    read_deadline: Option<tokio::time::Instant>,
    post_terminal_bytes: usize,
    frames: FrameScanner,
    /// The terminal frame ended at the CR of a CRLF that was the last byte
    /// of its chunk, so the LF may still arrive as the next byte.
    terminal_lf_pending: bool,
}

impl ReceiveContext {
    pub(super) fn new(wiring: StreamWiring) -> Self {
        Self {
            wiring,
            read_deadline: None,
            post_terminal_bytes: 0,
            frames: FrameScanner::default(),
            terminal_lf_pending: false,
        }
    }

    pub(super) fn is_strict(&self) -> bool {
        self.wiring.receive.is_strict()
    }

    pub(super) fn labels(&self) -> &StreamLabels {
        &self.wiring.labels
    }

    /// The next upstream item after the terminal event, or `None` when
    /// [`POST_TERMINAL_READ_LIMIT`] passed first. The limit runs from
    /// [`Self::start_read_window`], or from the first call when the window
    /// was not started. When it has already passed, the upstream is not
    /// polled, even if an item is waiting there.
    pub(super) async fn read_after_terminal<S>(
        &mut self,
        upstream: &mut S,
    ) -> Option<Option<S::Item>>
    where
        S: Stream + Unpin,
    {
        let deadline = *self
            .read_deadline
            .get_or_insert_with(|| tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT);
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        tokio::time::timeout_at(deadline, upstream.next())
            .await
            .ok()
    }

    /// Starts the read window now, when a terminal frame has been parsed,
    /// and starts it again for a terminal event that follows an earlier one
    /// (`[DONE]` after a `finish_reason`). The window runs while the client
    /// takes the output released before it, so a slow client does not give
    /// the upstream more time.
    pub(super) fn start_read_window(&mut self) {
        self.read_deadline = Some(tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT);
    }

    /// Counts bytes read after the terminal event; `true` once they exceed
    /// [`POST_TERMINAL_MAX_BYTES`].
    pub(super) fn exceeds_post_terminal_bytes(&mut self, len: usize) -> bool {
        self.post_terminal_bytes = self.post_terminal_bytes.saturating_add(len);
        self.post_terminal_bytes > POST_TERMINAL_MAX_BYTES
    }

    /// [`FrameScanner::raw_frame_ends`] carried across the chunks of one
    /// stream. A strict stream feeds every chunk through here and through
    /// its decoder.
    pub(super) fn raw_frame_ends(&mut self, chunk: &[u8]) -> RawFrameEnds {
        self.frames.raw_frame_ends(chunk)
    }

    /// Charges `chunk`, read after the chunk that held the terminal frame,
    /// and returns `true` once more than [`POST_TERMINAL_MAX_BYTES`] have
    /// been read after that frame. `continued` is
    /// [`RawFrameEnds::continued`] of the chunk: an LF that completes the
    /// terminal frame's CRLF is part of that frame and is not charged.
    pub(super) fn charge_after_terminal(&mut self, chunk: &[u8], continued: usize) -> bool {
        let own = self.terminal_lf(chunk, continued);
        self.exceeds_post_terminal_bytes(chunk.len() - own)
    }

    /// Notes where the terminal frame ended in `chunk` and charges the bytes
    /// after it, as [`Self::charge_after_terminal`].
    pub(super) fn terminal_frame_ended(&mut self, chunk: &[u8], end: usize) -> bool {
        self.terminal_lf_pending = end == chunk.len() && chunk.last() == Some(&b'\r');
        self.exceeds_post_terminal_bytes(chunk.len() - end)
    }

    /// How many leading bytes of `chunk` complete the terminal frame: 1 when
    /// it ended at a CR that closed the previous chunk and `chunk` opens
    /// with the LF of that CRLF, otherwise 0. Only the first chunk after
    /// the terminal frame is asked.
    pub(super) fn terminal_lf(&mut self, chunk: &[u8], continued: usize) -> usize {
        if chunk.is_empty() || !std::mem::take(&mut self.terminal_lf_pending) {
            return 0;
        }
        continued
    }
}

/// Where frames end in one chunk, the raw bytes of each frame counted
/// whole.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct RawFrameEnds {
    /// 1 when the chunk opens with the LF of a CRLF whose CR ended a frame
    /// at the end of the previous chunk: that byte belongs to that frame.
    /// Otherwise 0.
    pub(super) continued: usize,
    /// The offset just after the last byte of each frame that ends in the
    /// chunk: after the LF of a CRLF when that LF is in the chunk.
    pub(super) ends: Vec<usize>,
}

/// Finds where SSE frames end the way [`SseDecoder`] does: a line ends at
/// LF, CR or CRLF, and an empty line ends the frame.
#[derive(Debug, Clone, Copy)]
pub(super) struct FrameScanner {
    line_empty: bool,
    skip_lf: bool,
    /// The last byte scanned was a CR that ended a frame.
    frame_cr: bool,
}

impl Default for FrameScanner {
    fn default() -> Self {
        Self {
            line_empty: true,
            skip_lf: false,
            frame_cr: false,
        }
    }
}

impl FrameScanner {
    /// The offset just after each byte of `chunk` that ends a frame: the
    /// line ending of an empty line. The LF of a CRLF that ends a frame
    /// comes after that offset, as the decoder ends the line at the CR.
    pub(super) fn frame_ends(&mut self, chunk: &[u8]) -> Vec<usize> {
        let mut ends = Vec::new();
        for (index, &byte) in chunk.iter().enumerate() {
            if std::mem::take(&mut self.skip_lf) && byte == b'\n' {
                continue;
            }
            match byte {
                b'\n' | b'\r' => {
                    if self.line_empty {
                        ends.push(index + 1);
                    }
                    self.line_empty = true;
                    self.skip_lf = byte == b'\r';
                }
                _ => self.line_empty = false,
            }
        }
        ends
    }

    /// The frames that end in `chunk`, each up to its last raw byte: unlike
    /// [`Self::frame_ends`], a frame that ends with a CRLF ends after the
    /// LF when the LF is in `chunk`, and an LF that opens `chunk` after a
    /// frame-ending CR at the end of the previous chunk is reported in
    /// [`RawFrameEnds::continued`]. Feeding the decoder up to these ends
    /// dispatches the same events as up to [`Self::frame_ends`].
    pub(super) fn raw_frame_ends(&mut self, chunk: &[u8]) -> RawFrameEnds {
        if chunk.is_empty() {
            return RawFrameEnds::default();
        }
        let continued = usize::from(std::mem::take(&mut self.frame_cr) && chunk[0] == b'\n');
        let mut ends = self.frame_ends(chunk);
        for end in &mut ends {
            if chunk[*end - 1] == b'\r' && chunk.get(*end) == Some(&b'\n') {
                *end += 1;
            }
        }
        self.frame_cr = ends.last() == Some(&chunk.len()) && chunk[chunk.len() - 1] == b'\r';
        RawFrameEnds { continued, ends }
    }
}

/// The part of `input` up to the end of its last complete SSE frame, the
/// empty line that ends it included (both bytes of a CRLF). What follows,
/// a frame or a line that has not ended, is left out; input that ends where
/// a frame ends comes back whole, and input with no complete frame gives an
/// empty slice.
///
/// A caller that reads a buffered body itself and stops reading at
/// [`POST_TERMINAL_READ_LIMIT`] passes what it read through this before
/// [`accumulate_response_with`] or
/// [`super::responses::accumulate_response_with`], so a frame the limit
/// cut off is dropped as the streaming path drops it. A body that ended
/// cleanly is passed whole: a partial frame at its end is an error.
pub fn complete_frames_prefix(input: &[u8]) -> &[u8] {
    let Some(&end) = FrameScanner::default().frame_ends(input).last() else {
        return &input[..0];
    };
    let end = if input[end - 1] == b'\r' && input.get(end) == Some(&b'\n') {
        end + 1
    } else {
        end
    };
    &input[..end]
}

/// How [`prepare_request_with`] treats reasoning. [`Default`] is OpenCode
/// Go's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RequestPolicy {
    /// How `reasoning_effort` is set.
    pub effort: EffortPolicy,
    /// When an assistant turn carries its reasoning back as
    /// `reasoning_content`.
    pub reasoning_replay: ReasoningReplay,
}

/// How `reasoning_effort` is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffortPolicy {
    /// Map the requested effort by OpenCode Go model name and reject the
    /// values a model does not accept.
    #[default]
    OpenCodeGo,
    /// Leave `reasoning_effort` unset and the requested effort unread; the
    /// caller sets the field itself.
    Untouched,
}

/// When an assistant turn carries its reasoning back as `reasoning_content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningReplay {
    /// Always for a model whose name contains `deepseek`, otherwise as
    /// [`Self::WhenPresent`].
    #[default]
    OpenCodeGo,
    /// When the turn has reasoning text.
    WhenPresent,
    /// On every assistant turn, empty when the turn has no reasoning.
    Always,
    /// Never; thinking blocks are dropped.
    Never,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    pub stream: bool,
    pub stream_options: StreamOptions,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// Translate the Anthropic Messages wire model to the OpenAI-compatible wire
/// model accepted by OpenCode Go's `/chat/completions` endpoint.
pub fn prepare_request(req: &MessagesRequest, model: &str) -> anyhow::Result<ChatRequest> {
    prepare_request_with(req, model, RequestPolicy::default())
}

/// [`prepare_request`] with the reasoning rules of `policy`, for a provider
/// whose models do not follow OpenCode Go's effort names.
///
/// # Errors
///
/// The request cannot be expressed as Chat Completions, or, under
/// [`EffortPolicy::OpenCodeGo`], the model does not accept the requested
/// effort.
pub fn prepare_request_with(
    req: &MessagesRequest,
    model: &str,
    policy: RequestPolicy,
) -> anyhow::Result<ChatRequest> {
    let messages = build_messages(req, model, policy.reasoning_replay)?;
    let tools = read_tools(req)?;
    let (tool_choice, parallel_tool_calls) = read_tool_choice(req)?;
    if tools.is_empty()
        && tool_choice
            .as_ref()
            .is_some_and(|choice| choice.as_str() != Some("none"))
    {
        anyhow::bail!("tool_choice requires at least one tool");
    }

    Ok(ChatRequest {
        model: model.to_string(),
        messages,
        tools: (!tools.is_empty()).then_some(tools),
        tool_choice,
        parallel_tool_calls,
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
        max_tokens: req
            .max_tokens
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MAX_TOKENS),
        reasoning_effort: match policy.effort {
            EffortPolicy::OpenCodeGo => map_reasoning_effort(req, model)?,
            EffortPolicy::Untouched => None,
        },
    })
}

fn build_messages(
    req: &MessagesRequest,
    model: &str,
    replay: ReasoningReplay,
) -> anyhow::Result<Vec<Value>> {
    let replay = match replay {
        ReasoningReplay::OpenCodeGo if model.to_ascii_lowercase().contains("deepseek") => {
            ReasoningReplay::Always
        }
        ReasoningReplay::OpenCodeGo => ReasoningReplay::WhenPresent,
        other => other,
    };
    let mut system = Vec::new();
    if let Some(text) = flatten_system_text(req.extra.get("system")) {
        system.push(text);
    }

    let mut messages = Vec::new();
    for message in &req.messages {
        let blocks = normalize_content(&message.content, json!({}));
        match message.role.as_str() {
            "system" | "developer" => {
                let text = blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    system.push(text);
                }
            }
            "user" => push_user_messages(&mut messages, &blocks),
            "assistant" => {
                if let Some(message) = assistant_message(&blocks, replay)? {
                    messages.push(message);
                }
            }
            other => anyhow::bail!("unexpected message role: {other}"),
        }
    }

    if !system.is_empty() {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": system.join("\n\n"),
            }),
        );
    }
    Ok(messages)
}

fn push_user_messages(messages: &mut Vec<Value>, blocks: &[ContentBlock]) {
    let mut content = Vec::new();
    let mut tool_messages = Vec::new();
    let flush = |messages: &mut Vec<Value>, content: &mut Vec<Value>| {
        if content.is_empty() {
            return;
        }
        let value = if content
            .iter()
            .all(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        {
            Value::String(
                content
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect(),
            )
        } else {
            Value::Array(std::mem::take(content))
        };
        messages.push(json!({"role":"user", "content":value}));
        content.clear();
    };

    for block in blocks {
        match block {
            ContentBlock::Text { text } => content.push(json!({"type":"text", "text":text})),
            ContentBlock::Image { source } => content.push(json!({
                "type":"image_url",
                "image_url":{"url":image_source_to_url(source)},
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content: result,
                is_error,
            } => {
                let rendered = render_tool_result(result, is_error.unwrap_or(false));
                tool_messages.push(json!({
                    "role":"tool",
                    "tool_call_id":tool_use_id,
                    "content":rendered.text,
                }));
                content.extend(rendered.images);
            }
            ContentBlock::Thinking { .. } | ContentBlock::ToolUse { .. } => {}
        }
    }
    // A parallel assistant tool-call turn requires every matching tool
    // response before any subsequent user message. Keep tool results in their
    // original order, then reattach vision parts and ordinary user content in
    // one protocol-compatible user message.
    messages.extend(tool_messages);
    flush(messages, &mut content);
}

fn assistant_message(
    blocks: &[ContentBlock],
    replay: ReasoningReplay,
) -> anyhow::Result<Option<Value>> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text: value } => text.push_str(value),
            ContentBlock::Thinking { .. } if replay == ReasoningReplay::Never => {}
            ContentBlock::Thinking {
                thinking,
                signature: _,
            } => {
                if !reasoning.is_empty() && !thinking.is_empty() {
                    reasoning.push_str("\n\n");
                }
                reasoning.push_str(thinking);
            }
            ContentBlock::ToolUse { id, name, input } => {
                if id.is_empty() || name.is_empty() {
                    anyhow::bail!("assistant tool_use requires non-empty id and name");
                }
                tool_calls.push(json!({
                    "id":id,
                    "type":"function",
                    "function":{
                        "name":name,
                        "arguments":serde_json::to_string(input)?,
                    },
                }));
            }
            ContentBlock::Image { .. } | ContentBlock::ToolResult { .. } => {}
        }
    }
    if text.is_empty() && reasoning.is_empty() && tool_calls.is_empty() {
        return Ok(None);
    }

    let mut message = serde_json::Map::new();
    message.insert("role".into(), Value::String("assistant".into()));
    message.insert("content".into(), Value::String(text));
    // DeepSeek requires the field on every replayed assistant message, even
    // when that turn did not contain visible reasoning.
    if replay == ReasoningReplay::Always || !reasoning.is_empty() {
        message.insert("reasoning_content".into(), Value::String(reasoning));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    Ok(Some(Value::Object(message)))
}

struct RenderedToolResult {
    text: String,
    images: Vec<Value>,
}

fn render_tool_result(value: &Value, is_error: bool) -> RenderedToolResult {
    let mut text = String::new();
    let mut images = Vec::new();
    if is_error {
        text.push_str("[tool execution error]\n");
    }
    match value {
        Value::String(value) => text.push_str(value),
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(value) = part.get("text").and_then(Value::as_str) {
                            text.push_str(value);
                        }
                    }
                    Some("image") => match normalized_tool_result_image(part) {
                        Some(image_url) => {
                            images.push(json!({
                                "type":"image_url",
                                "image_url":{"url":image_url},
                            }));
                        }
                        None => text.push_str("[unsupported tool result block omitted: image]"),
                    },
                    Some(kind) => {
                        text.push_str(&format!("[unsupported tool result block omitted: {kind}]"))
                    }
                    None => text.push_str("[unsupported tool result block omitted]"),
                }
            }
        }
        other => text.push_str(&other.to_string()),
    }
    RenderedToolResult { text, images }
}

fn normalized_tool_result_image(part: &Value) -> Option<String> {
    let block = normalize_content(&Value::Array(vec![part.clone()]), json!({}))
        .into_iter()
        .next()?;
    let ContentBlock::Image { source } = block else {
        return None;
    };
    Some(image_source_to_url(&source))
}

fn read_tools(req: &MessagesRequest) -> anyhow::Result<Vec<Value>> {
    let Some(value) = req.extra.get("tools") else {
        return Ok(Vec::new());
    };
    let tools = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tools must be an array"))?;
    tools
        .iter()
        .map(|tool| {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool name must be a non-empty string"))?;
            let description = tool.get("description").cloned();
            let parameters = tool
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let mut function = serde_json::Map::new();
            function.insert("name".into(), Value::String(name.to_string()));
            if let Some(description) = description.filter(|value| !value.is_null()) {
                function.insert("description".into(), description);
            }
            function.insert("parameters".into(), parameters);
            Ok(json!({"type":"function", "function":function}))
        })
        .collect()
}

fn read_tool_choice(req: &MessagesRequest) -> anyhow::Result<(Option<Value>, Option<bool>)> {
    let Some(value) = req.extra.get("tool_choice") else {
        return Ok((None, None));
    };
    if value.is_null() {
        return Ok((None, None));
    }
    let choice = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("tool_choice must be an object"))?;
    let parallel = match choice.get("disable_parallel_tool_use") {
        Some(Value::Bool(disabled)) => Some(!disabled),
        Some(_) => anyhow::bail!("tool_choice.disable_parallel_tool_use must be a boolean"),
        None => None,
    };
    let translated = match choice.get("type").and_then(Value::as_str) {
        Some("auto") => Some(Value::String("auto".into())),
        Some("none") => Some(Value::String("none".into())),
        Some("any") => Some(Value::String("required".into())),
        Some("tool") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool_choice.name must be a non-empty string"))?;
            Some(json!({"type":"function", "function":{"name":name}}))
        }
        Some(kind) => anyhow::bail!("unsupported tool_choice type: {kind}"),
        None => anyhow::bail!("tool_choice.type must be a string"),
    };
    Ok((translated, parallel))
}

fn map_reasoning_effort(req: &MessagesRequest, model: &str) -> anyhow::Result<Option<String>> {
    let Some(effort) = read_effort(req)? else {
        return Ok(None);
    };
    let id = model.to_ascii_lowercase();
    if [
        "glm-5.2", "glm-5-2", "glm-5p2", "glm-5.3", "glm-5-3", "glm-5p3",
    ]
    .iter()
    .any(|needle| id.contains(needle))
    {
        return match effort {
            "high" => Ok(Some("high".into())),
            "xhigh" | "max" => Ok(Some("max".into())),
            other => anyhow::bail!(
                "OpenCode Go model {model} does not support reasoning effort {other}; use high, xhigh, or max"
            ),
        };
    }
    if id.contains("deepseek-v4") {
        return match effort {
            "low" | "medium" | "high" | "max" => Ok(Some(effort.into())),
            "xhigh" => Ok(Some("max".into())),
            _ => unreachable!("read_effort validates the effort"),
        };
    }
    if id.contains("mimo") {
        return match effort {
            "low" | "medium" | "high" => Ok(Some(effort.into())),
            other => anyhow::bail!(
                "OpenCode Go model {model} does not support reasoning effort {other}; use low, medium, or high"
            ),
        };
    }
    // OpenCode exposes no selectable effort variants for the remaining chat
    // models. Their native/default reasoning behavior remains in effect.
    Ok(None)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

impl StopReason {
    fn anthropic(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
        }
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<u64>,
}

impl Usage {
    fn anthropic(&self) -> Value {
        let cached = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .or(self.cached_tokens)
            .unwrap_or(0);
        json!({
            "input_tokens":self.prompt_tokens.unwrap_or(0).saturating_sub(cached),
            "output_tokens":self.completion_tokens.unwrap_or(0),
            "cache_creation_input_tokens":0,
            "cache_read_input_tokens":cached,
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Option<Vec<Choice>>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<UpstreamError>,
}

#[derive(Debug, serde::Deserialize)]
struct UpstreamError {
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, serde::Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ToolFunctionDelta>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct ToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug)]
struct Block {
    index: usize,
    kind: BlockKind,
}

#[derive(Debug)]
enum BlockKind {
    Thinking {
        text: String,
    },
    Text {
        text: String,
    },
    Tool {
        id: String,
        name: String,
        args: String,
    },
}

#[derive(Debug)]
struct ToolSlot {
    upstream_index: usize,
    block_index: usize,
    upstream_id: String,
    downstream_id: String,
    name: String,
    args: String,
    state: ToolBlockState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolBlockState {
    /// The upstream index is known, but no downstream block has been emitted.
    Pending,
    /// The downstream block has started and can still receive argument deltas.
    Open,
    /// The downstream block was emitted and stopped; it must not be reopened.
    Closed,
}

struct TranslationState {
    message_id: String,
    model: String,
    provider: &'static str,
    message_started: bool,
    finished: bool,
    next_block_index: usize,
    thinking_index: Option<usize>,
    text_index: Option<usize>,
    blocks: Vec<Block>,
    tools: Vec<ToolSlot>,
    pending_stop: Option<StopReason>,
    usage: Usage,
}

impl TranslationState {
    fn new(message_id: String, model: String, provider: &'static str) -> Self {
        Self {
            message_id,
            model,
            provider,
            message_started: false,
            finished: false,
            next_block_index: 0,
            thinking_index: None,
            text_index: None,
            blocks: Vec::new(),
            tools: Vec::new(),
            pending_stop: None,
            usage: Usage::default(),
        }
    }

    fn apply_chunk(&mut self, chunk: StreamChunk) -> anyhow::Result<Vec<u8>> {
        if self.finished {
            anyhow::bail!("{} event after terminal completion", self.provider);
        }
        if let Some(error) = chunk.error {
            anyhow::bail!(
                "{} upstream error: {}",
                self.provider,
                error.message.unwrap_or_else(|| "unknown error".into())
            );
        }
        if let Some(usage) = chunk.usage {
            self.usage = usage;
        }
        let Some(choices) = chunk.choices else {
            return Ok(Vec::new());
        };
        if choices.is_empty() {
            return Ok(Vec::new());
        }
        if choices.len() != 1 {
            anyhow::bail!("{} stream returned multiple choices", self.provider);
        }
        let choice = choices.into_iter().next().expect("one choice");
        let mut out = Vec::new();
        if let Some(delta) = choice.delta {
            self.apply_delta(delta, &mut out)?;
        }
        if let Some(reason) = choice.finish_reason {
            let reason = parse_finish_reason(&reason, self.provider)?;
            if self.pending_stop.is_some_and(|current| current != reason) {
                anyhow::bail!("{} stream changed finish_reason", self.provider);
            }
            self.pending_stop = Some(reason);
        }
        Ok(out)
    }

    fn apply_delta(&mut self, delta: Delta, out: &mut Vec<u8>) -> anyhow::Result<()> {
        if let Some(reasoning) = delta.reasoning_content.filter(|value| !value.is_empty()) {
            self.close_text(out);
            self.close_tools(out);
            let index = match self.thinking_index {
                Some(index) => index,
                None => {
                    let index = self.allocate_block();
                    self.ensure_message_start(out);
                    emit(
                        out,
                        "content_block_start",
                        json!({"type":"content_block_start","index":index,"content_block":{"type":"thinking","thinking":"","signature":""}}),
                    );
                    self.blocks.push(Block {
                        index,
                        kind: BlockKind::Thinking {
                            text: String::new(),
                        },
                    });
                    self.thinking_index = Some(index);
                    index
                }
            };
            if let Some(Block {
                kind: BlockKind::Thinking { text },
                ..
            }) = self.blocks.iter_mut().find(|block| block.index == index)
            {
                text.push_str(&reasoning);
            }
            emit(
                out,
                "content_block_delta",
                json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":reasoning}}),
            );
        }

        if let Some(content) = delta.content.filter(|value| !value.is_empty()) {
            self.close_thinking(out);
            self.close_tools(out);
            let index = match self.text_index {
                Some(index) => index,
                None => {
                    let index = self.allocate_block();
                    self.ensure_message_start(out);
                    emit(
                        out,
                        "content_block_start",
                        json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}),
                    );
                    self.blocks.push(Block {
                        index,
                        kind: BlockKind::Text {
                            text: String::new(),
                        },
                    });
                    self.text_index = Some(index);
                    index
                }
            };
            if let Some(Block {
                kind: BlockKind::Text { text },
                ..
            }) = self.blocks.iter_mut().find(|block| block.index == index)
            {
                text.push_str(&content);
            }
            emit(
                out,
                "content_block_delta",
                json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":content}}),
            );
        }

        if let Some(tool_calls) = delta.tool_calls.filter(|value| !value.is_empty()) {
            self.close_thinking(out);
            self.close_text(out);
            for call in tool_calls {
                self.apply_tool_delta(call, out)?;
            }
        }
        Ok(())
    }

    fn apply_tool_delta(&mut self, delta: ToolCallDelta, out: &mut Vec<u8>) -> anyhow::Result<()> {
        let ToolCallDelta {
            index: upstream_index,
            id,
            function,
        } = delta;
        let position = match self
            .tools
            .iter()
            .position(|slot| slot.upstream_index == upstream_index)
        {
            Some(position) => position,
            None => {
                let block_index = self.allocate_block();
                self.tools.push(ToolSlot {
                    upstream_index,
                    block_index,
                    upstream_id: String::new(),
                    downstream_id: make_tool_use_id(&self.message_id, upstream_index),
                    name: String::new(),
                    args: String::new(),
                    state: ToolBlockState::Pending,
                });
                self.tools.len() - 1
            }
        };
        if self.tools[position].state == ToolBlockState::Closed {
            anyhow::bail!(
                "{} tool call {} continued after its content block was closed",
                self.provider,
                upstream_index
            );
        }
        let mut new_arguments = String::new();
        {
            let slot = &mut self.tools[position];
            if let Some(id) = id {
                slot.upstream_id.push_str(&id);
            }
            if let Some(function) = function {
                if let Some(name) = function.name {
                    slot.name.push_str(&name);
                }
                if let Some(arguments) = function.arguments {
                    new_arguments = arguments;
                    slot.args.push_str(&new_arguments);
                }
            }
        }
        let should_start = {
            let slot = &self.tools[position];
            slot.state == ToolBlockState::Pending
                && !slot.upstream_id.is_empty()
                && !slot.name.is_empty()
        };
        if should_start {
            self.ensure_message_start(out);
            let (block_index, id, name, args) = {
                let slot = &mut self.tools[position];
                slot.state = ToolBlockState::Open;
                (
                    slot.block_index,
                    slot.downstream_id.clone(),
                    slot.name.clone(),
                    slot.args.clone(),
                )
            };
            emit(
                out,
                "content_block_start",
                json!({"type":"content_block_start","index":block_index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}),
            );
            self.blocks.push(Block {
                index: block_index,
                kind: BlockKind::Tool {
                    id,
                    name,
                    args: String::new(),
                },
            });
            if !args.is_empty() {
                emit(
                    out,
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":block_index,"delta":{"type":"input_json_delta","partial_json":args}}),
                );
                if let Some(Block {
                    kind: BlockKind::Tool { args: stored, .. },
                    ..
                }) = self
                    .blocks
                    .iter_mut()
                    .find(|block| block.index == block_index)
                {
                    stored.push_str(&args);
                }
            }
        } else if self.tools[position].state == ToolBlockState::Open && !new_arguments.is_empty() {
            let block_index = self.tools[position].block_index;
            emit(
                out,
                "content_block_delta",
                json!({"type":"content_block_delta","index":block_index,"delta":{"type":"input_json_delta","partial_json":new_arguments}}),
            );
            if let Some(Block {
                kind: BlockKind::Tool { args, .. },
                ..
            }) = self
                .blocks
                .iter_mut()
                .find(|block| block.index == block_index)
            {
                args.push_str(&new_arguments);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> anyhow::Result<Vec<u8>> {
        if self.finished {
            return Ok(Vec::new());
        }
        for slot in &self.tools {
            if slot.state == ToolBlockState::Pending {
                anyhow::bail!(
                    "{} tool call {} ended without id or function name",
                    self.provider,
                    slot.upstream_index
                );
            }
        }
        for block in &self.blocks {
            if let BlockKind::Tool { args, .. } = &block.kind {
                let value: Value = serde_json::from_str(if args.is_empty() { "{}" } else { args })
                    .map_err(|_| {
                        anyhow::anyhow!("{} tool arguments are invalid JSON", self.provider)
                    })?;
                if !value.is_object() {
                    anyhow::bail!("{} tool arguments must be a JSON object", self.provider);
                }
            }
        }
        let mut out = Vec::new();
        self.close_thinking(&mut out);
        self.close_text(&mut out);
        self.close_tools(&mut out);
        self.ensure_message_start(&mut out);
        let stop = self.pending_stop.unwrap_or({
            if self.tools.is_empty() {
                StopReason::EndTurn
            } else {
                StopReason::ToolUse
            }
        });
        emit(
            &mut out,
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":stop.anthropic(),"stop_sequence":null},"usage":self.usage.anthropic()}),
        );
        emit(&mut out, "message_stop", json!({"type":"message_stop"}));
        self.finished = true;
        Ok(out)
    }

    fn response(&self) -> anyhow::Result<Value> {
        if !self.finished {
            anyhow::bail!("{} response is not complete", self.provider);
        }
        let mut content = Vec::new();
        for block in &self.blocks {
            match &block.kind {
                BlockKind::Thinking { text } => content.push(json!({
                    "type":"thinking",
                    "thinking":text,
                    "signature":make_thinking_signature(&self.message_id, block.index),
                })),
                BlockKind::Text { text } => {
                    content.push(json!({"type":"text", "text":text}));
                }
                BlockKind::Tool { id, name, args } => {
                    let input: Value =
                        serde_json::from_str(if args.is_empty() { "{}" } else { args })?;
                    content.push(json!({
                        "type":"tool_use",
                        "id":id,
                        "name":name,
                        "input":input,
                    }));
                }
            }
        }
        let stop = self.pending_stop.unwrap_or({
            if self.tools.is_empty() {
                StopReason::EndTurn
            } else {
                StopReason::ToolUse
            }
        });
        Ok(json!({
            "id":self.message_id,
            "type":"message",
            "role":"assistant",
            "model":self.model,
            "content":content,
            "stop_reason":stop.anthropic(),
            "stop_sequence":null,
            "usage":self.usage.anthropic(),
        }))
    }

    fn ensure_message_start(&mut self, out: &mut Vec<u8>) {
        if self.message_started {
            return;
        }
        emit(
            out,
            "message_start",
            json!({"type":"message_start","message":{"id":self.message_id,"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}),
        );
        self.message_started = true;
    }

    fn allocate_block(&mut self) -> usize {
        let index = self.next_block_index;
        self.next_block_index += 1;
        index
    }

    fn close_thinking(&mut self, out: &mut Vec<u8>) {
        let Some(index) = self.thinking_index.take() else {
            return;
        };
        emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"signature_delta","signature":make_thinking_signature(&self.message_id, index)}}),
        );
        emit(
            out,
            "content_block_stop",
            json!({"type":"content_block_stop","index":index}),
        );
    }

    fn close_text(&mut self, out: &mut Vec<u8>) {
        let Some(index) = self.text_index.take() else {
            return;
        };
        emit(
            out,
            "content_block_stop",
            json!({"type":"content_block_stop","index":index}),
        );
    }

    fn close_tools(&mut self, out: &mut Vec<u8>) {
        for slot in &mut self.tools {
            if slot.state == ToolBlockState::Open {
                emit(
                    out,
                    "content_block_stop",
                    json!({"type":"content_block_stop","index":slot.block_index}),
                );
                slot.state = ToolBlockState::Closed;
            }
        }
    }
}

fn parse_finish_reason(reason: &str, provider: &str) -> anyhow::Result<StopReason> {
    match reason {
        "stop" => Ok(StopReason::EndTurn),
        "tool_calls" | "function_call" => Ok(StopReason::ToolUse),
        "length" => Ok(StopReason::MaxTokens),
        "content_filter" => anyhow::bail!("{provider} response was blocked by a content filter"),
        other => anyhow::bail!("unsupported {provider} finish_reason: {other}"),
    }
}

fn make_thinking_signature(message_id: &str, index: usize) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(format!("ccp:opencode:v1:{message_id}:{index}"))
}

fn make_tool_use_id(message_id: &str, upstream_index: usize) -> String {
    // Several Chat Completions providers restart tool-call IDs (for example,
    // `Agent_0`) in every independent completion. Nested Claude Code agents
    // are flattened into one downstream stream, so forwarding those IDs can
    // make a child tool call appear to reference itself as its parent. The
    // response message ID is generated uniquely by CCP; namespacing the slot
    // with it keeps split deltas stable and independent completions distinct.
    format!("toolu_{message_id}_{upstream_index}")
}

fn emit(out: &mut Vec<u8>, event: &str, value: Value) {
    out.extend(encode_sse_event(Some(event), &value.to_string()));
}

pub struct LiveStreamTranslator {
    decoder: SseDecoder,
    state: TranslationState,
    terminal: TerminalPolicy,
    done_seen: bool,
    /// Whether SSE decoding errors name the provider. OpenCode Go keeps the
    /// shared decoder's own wording.
    labelled_decoder_errors: bool,
}

impl LiveStreamTranslator {
    pub fn new(message_id: String, model: String) -> Self {
        Self::with_wiring(
            message_id,
            model,
            StreamLabels::OPENCODE_GO,
            TerminalPolicy::OpenCodeGo,
        )
    }

    /// A translator whose errors name `labels.provider`. Under
    /// [`TerminalPolicy::Strict`], `[DONE]` does not complete the response:
    /// a second `[DONE]` or any later event is an error, and the closing
    /// events come from [`Self::finish`] or [`Self::finish_after_read_limit`].
    /// Between a `finish_reason` and `[DONE]` only chunks without content
    /// are accepted (usage, or a choice with an empty delta and the same
    /// finish reason); text, reasoning, a tool-call delta or a different
    /// finish reason there is an error. When `labels` are not
    /// [`StreamLabels::OPENCODE_GO`], an SSE framing error also names
    /// `labels.provider`, as in `<provider> stream ended with an incomplete
    /// frame`.
    pub fn with_wiring(
        message_id: String,
        model: String,
        labels: StreamLabels,
        terminal: TerminalPolicy,
    ) -> Self {
        Self {
            decoder: SseDecoder::default(),
            state: TranslationState::new(message_id, model, labels.provider),
            terminal,
            done_seen: false,
            labelled_decoder_errors: labels != StreamLabels::OPENCODE_GO,
        }
    }

    pub fn push(&mut self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut out = Vec::new();
        let events = match self.decoder.push(input) {
            Ok(events) => events,
            Err(error) => {
                return Err(self.decoder_error(
                    error,
                    "stream sent an SSE frame that is too large or not valid UTF-8",
                ));
            }
        };
        for event in events {
            let data = event.data.trim();
            if self.terminal == TerminalPolicy::Strict {
                if data == "[DONE]" {
                    if self.done_seen {
                        anyhow::bail!("{} stream sent [DONE] twice", self.state.provider);
                    }
                    self.done_seen = true;
                    continue;
                }
                if self.done_seen {
                    anyhow::bail!("{} event after [DONE]", self.state.provider);
                }
            }
            if data == "[DONE]" {
                out.extend(self.state.finalize()?);
                continue;
            }
            if self.state.finished {
                validate_post_done_metadata(data, self.state.provider)?;
                continue;
            }
            let chunk: StreamChunk = serde_json::from_str(data)
                .map_err(|_| anyhow::anyhow!("malformed {} SSE event", self.state.provider))?;
            if self.terminal == TerminalPolicy::Strict && self.state.pending_stop.is_some() {
                check_after_finish_reason(&chunk, self.state.provider)?;
            }
            out.extend(self.state.apply_chunk(chunk)?);
        }
        Ok(out)
    }

    pub fn finish(&mut self) -> anyhow::Result<Vec<u8>> {
        if let Err(error) = self.decoder.finish() {
            return Err(self.decoder_error(error, "stream ended with an incomplete frame"));
        }
        if self.state.finished {
            return Ok(Vec::new());
        }
        if self.state.pending_stop.is_none() && !self.done_seen {
            anyhow::bail!(
                "{} stream ended without [DONE] or finish_reason",
                self.state.provider
            );
        }
        self.state.finalize()
    }

    /// Completes the response when the upstream stayed open for
    /// [`POST_TERMINAL_READ_LIMIT`] after the terminal event. A partial frame
    /// still buffered is discarded.
    ///
    /// # Errors
    ///
    /// No `[DONE]` or `finish_reason` has arrived, or the response content
    /// is invalid.
    pub fn finish_after_read_limit(&mut self) -> anyhow::Result<Vec<u8>> {
        if !self.is_terminal() {
            anyhow::bail!(
                "{} stream ended without [DONE] or finish_reason",
                self.state.provider
            );
        }
        self.state.finalize()
    }

    pub fn is_finished(&self) -> bool {
        self.state.finished
    }

    /// Whether `[DONE]` or a `finish_reason` has arrived; from then on a
    /// strict caller reads for at most [`POST_TERMINAL_READ_LIMIT`], counted
    /// again from `[DONE]` when it follows the `finish_reason`.
    pub fn is_terminal(&self) -> bool {
        self.state.finished || self.done_seen || self.state.pending_stop.is_some()
    }

    /// `error` from the shared SSE decoder, or `<provider> <labelled>` when
    /// the labels are not OpenCode Go's.
    fn decoder_error(&self, error: anyhow::Error, labelled: &str) -> anyhow::Error {
        if self.labelled_decoder_errors {
            anyhow::anyhow!("{} {labelled}", self.state.provider)
        } else {
            error
        }
    }

    fn has_seen_done(&self) -> bool {
        self.done_seen
    }
}

/// Under [`TerminalPolicy::Strict`], rejects a chunk after a `finish_reason`
/// that carries text, reasoning or a tool-call delta. A different finish
/// reason is rejected by [`TranslationState::apply_chunk`] in every policy.
fn check_after_finish_reason(chunk: &StreamChunk, provider: &str) -> anyhow::Result<()> {
    let has_text = |text: &Option<String>| text.as_deref().is_some_and(|text| !text.is_empty());
    let carries_content = chunk
        .choices
        .iter()
        .flatten()
        .filter_map(|choice| choice.delta.as_ref())
        .any(|delta| {
            has_text(&delta.content)
                || has_text(&delta.reasoning_content)
                || delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
        });
    if carries_content {
        anyhow::bail!("{provider} stream sent content after finish_reason");
    }
    Ok(())
}

fn validate_post_done_metadata(data: &str, provider: &str) -> anyhow::Result<()> {
    let value: Value = serde_json::from_str(data)
        .map_err(|_| anyhow::anyhow!("malformed {provider} SSE event after [DONE]"))?;
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{provider} event after terminal completion"))?;
    let has_empty_choices = object
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    let has_cost = object
        .get("cost")
        .is_some_and(|cost| cost.is_string() || cost.is_number());
    let known_keys = object.keys().all(|key| {
        matches!(
            key.as_str(),
            "choices" | "cost" | "x-opencode-type" | "normalizedUsage"
        )
    });
    if has_empty_choices && has_cost && known_keys {
        return Ok(());
    }
    anyhow::bail!("{provider} event after terminal completion")
}

pub fn accumulate_response(input: &[u8], message_id: &str, model: &str) -> anyhow::Result<Value> {
    accumulate_response_with(input, message_id, model, &StreamWiring::default())
}

/// [`accumulate_response`] with the labels and receive policy of `wiring`.
/// The error goes through [`ReceivePolicy::sanitize`].
///
/// `input` is the whole body, already read, so the caller that read it owns
/// [`POST_TERMINAL_READ_LIMIT`]. When that limit ended the read, the caller
/// passes [`complete_frames_prefix`] of what it read, dropping a frame the
/// limit cut off; when the upstream closed the body, the caller passes it
/// whole, and a partial frame at its end is an error.
///
/// Under [`TerminalPolicy::Strict`] the same rules hold as on the streaming
/// path: any event after `[DONE]` is an error, and so is more than
/// [`POST_TERMINAL_MAX_BYTES`] after the end of the frame that carried the
/// first `finish_reason` or `[DONE]`.
///
/// # Errors
///
/// The body is not a complete, valid Chat Completions stream.
pub fn accumulate_response_with(
    input: &[u8],
    message_id: &str,
    model: &str,
    wiring: &StreamWiring,
) -> anyhow::Result<Value> {
    let mut translator = LiveStreamTranslator::with_wiring(
        message_id.into(),
        model.into(),
        wiring.labels,
        wiring.receive.terminal,
    );
    let pushed = if wiring.receive.is_strict() {
        push_buffered_strict(&mut translator, input)
    } else {
        translator.push(input).map(drop)
    };
    let result = pushed
        .and_then(|()| translator.finish())
        .and_then(|_| translator.state.response());
    result.map_err(|error| wiring.receive.upstream_error(error))
}

/// Pushes a strict buffered body one frame at a time, so the bytes after
/// the terminal frame are counted from where it ends.
fn push_buffered_strict(translator: &mut LiveStreamTranslator, input: &[u8]) -> anyhow::Result<()> {
    let mut frames = FrameScanner::default();
    let mut terminal = false;
    let mut start = 0;
    for end in frames
        .raw_frame_ends(input)
        .ends
        .into_iter()
        .chain([input.len()])
    {
        if end == start {
            continue;
        }
        translator.push(&input[start..end])?;
        start = end;
        if !terminal && translator.is_terminal() {
            terminal = true;
            if input.len() - end > POST_TERMINAL_MAX_BYTES {
                anyhow::bail!(
                    "{} stream sent more than {POST_TERMINAL_MAX_BYTES} bytes after its terminal event",
                    translator.state.provider
                );
            }
        }
    }
    Ok(())
}

pub fn stream_error(message: &str) -> Vec<u8> {
    encode_sse_event(
        Some("error"),
        &json!({"type":"error","error":{"type":"api_error","message":message}}).to_string(),
    )
}

pub fn stream_body(
    upstream: OpenCodeResponse,
    message_id: String,
    model: String,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
) -> Body {
    stream_body_with(
        upstream.into_stream(),
        message_id,
        model,
        monitor,
        req_id,
        traffic,
        StreamWiring::default(),
    )
}

/// The Anthropic SSE body translated from any Chat Completions upstream
/// byte stream, with the labels and receive policy of `wiring`.
pub fn stream_body_with<S, E>(
    upstream: S,
    message_id: String,
    model: String,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
    wiring: StreamWiring,
) -> Body
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let stream = wired_stream(
        upstream, message_id, model, monitor, req_id, traffic, wiring,
    );
    let stream = futures_util::stream::unfold(stream, |mut stream| async move {
        stream
            .state
            .next_output_with(&mut stream.context)
            .await
            .map(|bytes| (Ok::<Bytes, Infallible>(Bytes::from(bytes)), stream))
    });
    Body::from_stream(stream)
}

fn wired_stream<S, E>(
    upstream: S,
    message_id: String,
    model: String,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
    wiring: StreamWiring,
) -> WiredChatStream
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
{
    let upstream = box_upstream(upstream);
    let state = OpenCodeChatStreamState {
        upstream,
        translator: LiveStreamTranslator::with_wiring(
            message_id,
            model,
            wiring.labels,
            wiring.receive.terminal,
        ),
        capture_decoder: SseDecoder::default(),
        terminal: false,
        error_sent: false,
        monitor,
        req_id,
        bytes: 0,
        chunks: 0,
        stream_capture: traffic.as_ref().map(|traffic| traffic.stream_capture()),
        traffic,
    };
    WiredChatStream {
        state,
        context: ReceiveContext::new(wiring),
    }
}

/// One stream and its wiring. Its `Drop` runs before the state's, so an
/// abandoned stream is summarised under the wiring's labels.
struct WiredChatStream {
    state: OpenCodeChatStreamState<UpstreamStream>,
    context: ReceiveContext,
}

impl Drop for WiredChatStream {
    fn drop(&mut self) {
        self.state.abandon(self.context.labels());
    }
}

struct OpenCodeChatStreamState<S> {
    upstream: S,
    translator: LiveStreamTranslator,
    capture_decoder: SseDecoder,
    terminal: bool,
    error_sent: bool,
    monitor: Option<MonitorHandle>,
    req_id: String,
    bytes: u64,
    chunks: u64,
    stream_capture: Option<StreamTrafficCapture>,
    traffic: Option<Arc<TrafficCapture>>,
}

impl<S> OpenCodeChatStreamState<S>
where
    S: futures_util::Stream<Item = Result<Bytes, OpenCodeError>> + Unpin,
{
    #[cfg(test)]
    async fn next_output(&mut self) -> Option<Vec<u8>> {
        self.next_output_with(&mut ReceiveContext::default()).await
    }

    async fn next_output_with(&mut self, context: &mut ReceiveContext) -> Option<Vec<u8>> {
        if self.terminal {
            return None;
        }
        if self.error_sent {
            self.terminal = true;
            return None;
        }
        loop {
            let after_terminal = context.is_strict() && self.translator.is_terminal();
            let next = if after_terminal {
                match context.read_after_terminal(&mut self.upstream).await {
                    Some(next) => next,
                    None => {
                        let result = self.translator.finish_after_read_limit();
                        return Some(self.settle_strict(result, context));
                    }
                }
            } else {
                self.upstream.next().await
            };
            let chunk = match next {
                Some(Ok(chunk)) => chunk,
                Some(Err(_)) => return Some(self.fail_at("transport", "upstream_stream", context)),
                None if context.is_strict() => {
                    let result = self.translator.finish();
                    if result.is_ok() && self.capture_decoder.finish().is_err() {
                        return Some(self.fail_at("capture", "incomplete_stream", context));
                    }
                    return Some(self.settle_strict(result, context));
                }
                None => {
                    let output = match self.translator.finish() {
                        Ok(output) => output,
                        Err(_) => {
                            return Some(self.fail_at("decoder", "incomplete_stream", context));
                        }
                    };
                    if self.capture_decoder.finish().is_err() {
                        return Some(self.fail_at("capture", "incomplete_stream", context));
                    }
                    self.terminal = true;
                    self.capture_downstream(&output);
                    self.finish_capture(true, context);
                    return (!output.is_empty()).then_some(output);
                }
            };
            if self.bytes == 0
                && let Some(monitor) = self.monitor.as_ref()
            {
                monitor.generation_started(&self.req_id);
            }
            self.bytes = self.bytes.saturating_add(chunk.len() as u64);
            self.chunks = self.chunks.saturating_add(1);
            self.capture_upstream(&chunk, context);
            let output = if context.is_strict() {
                match self.push_strict(&chunk, context) {
                    Ok(output) => output,
                    Err((stage, kind)) => return Some(self.fail_at(stage, kind, context)),
                }
            } else {
                match self.translator.push(&chunk) {
                    Ok(output) => output,
                    Err(_) => return Some(self.fail_at("translation", "invalid_event", context)),
                }
            };
            if !output.is_empty() {
                self.report_progress(&output);
                self.capture_downstream(&output);
            }
            if self.translator.is_finished() {
                if self.translator.finish().is_err() || self.capture_decoder.finish().is_err() {
                    return Some(self.fail_at("decoder", "trailing_incomplete_frame", context));
                }
                self.terminal = true;
                self.finish_capture(true, context);
                return (!output.is_empty()).then_some(output);
            }
            if !output.is_empty() {
                return Some(output);
            }
        }
    }

    /// Pushes a chunk of a strict stream one frame at a time. The bytes after
    /// the frame that made the stream terminal are charged to
    /// [`POST_TERMINAL_MAX_BYTES`] by position, so a tail counts the same in
    /// the terminal frame's chunk as in later ones.
    fn push_strict(
        &mut self,
        chunk: &[u8],
        context: &mut ReceiveContext,
    ) -> Result<Vec<u8>, (&'static str, &'static str)> {
        let RawFrameEnds { continued, ends } = context.raw_frame_ends(chunk);
        let mut terminal = self.translator.is_terminal();
        if terminal && context.charge_after_terminal(chunk, continued) {
            return Err(("protocol", "post_terminal_limit"));
        }
        let mut output = Vec::new();
        let mut start = 0;
        for end in ends.into_iter().chain([chunk.len()]) {
            if end == start {
                continue;
            }
            let done_before = self.translator.has_seen_done();
            match self.translator.push(&chunk[start..end]) {
                Ok(translated) => output.extend(translated),
                Err(_) => return Err(("translation", "invalid_event")),
            }
            start = end;
            if !done_before && self.translator.has_seen_done() {
                context.start_read_window();
            }
            if !terminal && self.translator.is_terminal() {
                terminal = true;
                context.start_read_window();
                if context.terminal_frame_ended(chunk, end) {
                    return Err(("protocol", "post_terminal_limit"));
                }
            }
        }
        Ok(output)
    }

    /// Ends a strict stream with the closing events, or with an error event
    /// when the response cannot be completed.
    fn settle_strict(
        &mut self,
        result: anyhow::Result<Vec<u8>>,
        context: &ReceiveContext,
    ) -> Vec<u8> {
        let output = match result {
            Ok(output) => output,
            Err(_) => return self.fail_at("decoder", "incomplete_stream", context),
        };
        self.terminal = true;
        if !output.is_empty() {
            self.report_progress(&output);
            self.capture_downstream(&output);
        }
        self.finish_capture(true, context);
        output
    }

    fn report_progress(&self, output: &[u8]) {
        let (input_tokens, output_tokens) = usage_from_anthropic_sse(output);
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.stream_progress(
                &self.req_id,
                output.len() as u64,
                count_sse_events(output),
                input_tokens,
                output_tokens,
            );
        }
    }

    fn capture_upstream(&mut self, bytes: &[u8], context: &ReceiveContext) {
        let events = match self.capture_decoder.push(bytes) {
            Ok(events) => events,
            Err(_) => {
                if let Some(capture) = self.stream_capture.as_mut() {
                    capture.malformed("decoder", "malformed_sse");
                }
                return;
            }
        };
        let Some(capture) = self.stream_capture.as_mut() else {
            return;
        };
        for event in events {
            match serde_json::from_str(&event.data) {
                Ok(value) => capture.upstream_event(
                    event.event.as_deref(),
                    &context.wiring.receive.upstream_value(&value),
                ),
                Err(_) if event.data.trim() == "[DONE]" => {
                    capture.upstream_event(event.event.as_deref(), &json!("[DONE]"));
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

    fn fail_at(&mut self, stage: &str, kind: &str, context: &ReceiveContext) -> Vec<u8> {
        self.error_sent = true;
        let output = stream_error(&format!("{} stream is invalid", context.labels().provider));
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.malformed(stage, kind);
        }
        self.capture_downstream(&output);
        self.finish_capture(false, context);
        output
    }

    fn finish_capture(&mut self, completed: bool, context: &ReceiveContext) {
        if let (Some(capture), Some(traffic)) = (self.stream_capture.take(), self.traffic.as_ref())
        {
            capture.finish_named(
                traffic,
                json!({
                    "kind":if completed { "stream_completion" } else { "stream_error" },
                    "bytes":self.bytes,
                    "chunks":self.chunks,
                }),
                &summary_name(context.labels()),
            );
        }
    }
}

fn summary_name(labels: &StreamLabels) -> String {
    format!("061-{}-stream-summary", labels.capture_stem)
}

impl<S> OpenCodeChatStreamState<S> {
    fn abandon(&mut self, labels: &StreamLabels) {
        if self.terminal || self.stream_capture.is_none() {
            return;
        }
        if let (Some(capture), Some(traffic)) = (self.stream_capture.take(), self.traffic.as_ref())
        {
            capture.finish_named(
                traffic,
                json!({
                    "kind":"stream_abandoned",
                    "reason":"downstream_body_dropped",
                    "bytes":self.bytes,
                    "chunks":self.chunks,
                }),
                &summary_name(labels),
            );
        }
    }
}

impl<S> Drop for OpenCodeChatStreamState<S> {
    fn drop(&mut self) {
        self.abandon(&StreamLabels::OPENCODE_GO);
    }
}

fn count_sse_events(bytes: &[u8]) -> u64 {
    parse_sse_events(bytes).len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn request_maps_glm_effort_tools_parallelism_and_replay() {
        let req = request(json!({
            "model":"opencode-go/glm-5.2",
            "max_tokens":123,
            "system":"system",
            "messages":[
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"reason","signature":"opaque"},
                    {"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"rust"}}
                ]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}
            ],
            "tools":[{"name":"lookup","description":"Lookup","input_schema":{"type":"object"}}],
            "tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true},
            "output_config":{"effort":"xhigh"}
        }));
        let wire = serde_json::to_value(prepare_request(&req, "glm-5.2").unwrap()).unwrap();
        assert_eq!(wire["model"], "glm-5.2");
        assert_eq!(wire["max_tokens"], 123);
        assert_eq!(wire["reasoning_effort"], "max");
        assert_eq!(wire["parallel_tool_calls"], false);
        assert_eq!(wire["messages"][1]["reasoning_content"], "reason");
        assert_eq!(wire["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(wire["messages"][2]["tool_call_id"], "call_1");
        assert_eq!(
            wire["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn tool_result_images_follow_the_required_tool_response_in_image_order() {
        let req = request(json!({
            "messages":[
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"call_image","name":"Read","input":{"file_path":"image.png"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"call_image","is_error":true,"content":[
                        {"type":"text","text":"before"},
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"YWJj"}},
                        {"type":"text","text":"between"},
                        {"type":"image","source":{"type":"url","url":"https://example.invalid/image.webp"}},
                        {"type":"text","text":"after"}
                    ]}
                ]}
            ]
        }));

        let wire = serde_json::to_value(prepare_request(&req, "glm-5.2").unwrap()).unwrap();
        let messages = wire["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_image");
        assert_eq!(
            messages[1]["content"],
            concat!("[tool execution error]\n", "before", "between", "after")
        );
        assert!(
            !messages[1]["content"]
                .as_str()
                .unwrap()
                .contains("unsupported tool result block omitted: image")
        );

        assert_eq!(messages[2]["role"], "user");
        assert_eq!(
            messages[2]["content"],
            json!([
                {
                    "type":"image_url",
                    "image_url":{"url":"data:image/png;base64,YWJj"}
                },
                {
                    "type":"image_url",
                    "image_url":{"url":"https://example.invalid/image.webp"}
                }
            ])
        );
    }

    #[test]
    fn tool_result_image_reattachment_preserves_surrounding_user_message_order() {
        let req = request(json!({
            "messages":[{"role":"user","content":[
                {"type":"text","text":"before tool"},
                {"type":"tool_result","tool_use_id":"call_image","content":[
                    {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"YWJj"}}
                ]},
                {"type":"text","text":"after tool"}
            ]}]
        }));

        let wire = serde_json::to_value(prepare_request(&req, "glm-5.2").unwrap()).unwrap();
        let messages = wire["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_image");
        assert_eq!(messages[0]["content"], "");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(
            messages[1]["content"],
            json!([
                {"type":"text","text":"before tool"},
                {
                    "type":"image_url",
                    "image_url":{"url":"data:image/jpeg;base64,YWJj"}
                },
                {"type":"text","text":"after tool"}
            ])
        );
    }

    #[test]
    fn parallel_tool_results_stay_contiguous_before_reattached_images() {
        let req = request(json!({
            "messages":[
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"call_input","name":"Read","input":{"file_path":"input.txt"}},
                    {"type":"tool_use","id":"call_image","name":"Read","input":{"file_path":"image.png"}},
                    {"type":"tool_use","id":"call_bash","name":"Bash","input":{"command":"pwd"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"call_input","content":"input ok"},
                    {"type":"tool_result","tool_use_id":"call_image","content":[
                        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"YWJj"}}
                    ]},
                    {"type":"tool_result","tool_use_id":"call_bash","content":"bash ok"}
                ]}
            ]
        }));

        let wire = serde_json::to_value(prepare_request(&req, "deepseek-v4-pro").unwrap()).unwrap();
        let messages = wire["messages"].as_array().unwrap();
        assert_eq!(
            messages
                .iter()
                .map(|message| message["role"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["assistant", "tool", "tool", "tool", "user"]
        );
        assert_eq!(messages[1]["tool_call_id"], "call_input");
        assert_eq!(messages[2]["tool_call_id"], "call_image");
        assert_eq!(messages[2]["content"], "");
        assert_eq!(messages[3]["tool_call_id"], "call_bash");
        assert_eq!(
            messages[4]["content"],
            json!([{
                "type":"image_url",
                "image_url":{"url":"data:image/png;base64,YWJj"}
            }])
        );
    }

    #[test]
    fn deepseek_replay_always_has_reasoning_content() {
        let req = request(json!({
            "messages":[
                {"role":"assistant","content":[{"type":"text","text":"answer"}]},
                {"role":"user","content":"next"}
            ]
        }));
        let wire = serde_json::to_value(prepare_request(&req, "deepseek-v4-pro").unwrap()).unwrap();
        assert_eq!(wire["messages"][0]["reasoning_content"], "");
    }

    #[test]
    fn live_stream_preserves_fragmented_reasoning_text_tools_usage_and_indices() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":7,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":7,\"function\":{\"arguments\":\":\\\"rust\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
            "data: [DONE]\n\n"
        );
        for split in 0..=upstream.len() {
            let mut translator = LiveStreamTranslator::new("msg_1".into(), "glm-5.2".into());
            let mut output = translator.push(&upstream.as_bytes()[..split]).unwrap();
            output.extend(translator.push(&upstream.as_bytes()[split..]).unwrap());
            output.extend(translator.finish().unwrap());
            let rendered = String::from_utf8(output).unwrap();
            assert!(rendered.contains("thinking_delta"), "split {split}");
            assert!(rendered.contains("signature_delta"), "split {split}");
            assert!(rendered.contains("text_delta"), "split {split}");
            assert!(rendered.contains("input_json_delta"), "split {split}");
            assert!(rendered.contains("tool_use"), "split {split}");
            assert_eq!(
                rendered.matches("\"id\":\"toolu_msg_1_7\"").count(),
                1,
                "split {split}"
            );
            assert!(
                rendered.contains("cache_read_input_tokens"),
                "split {split}"
            );
            assert!(rendered.contains("message_stop"), "split {split}");
        }
    }

    #[test]
    fn live_stream_accepts_text_and_reasoning_after_a_tool_call() {
        for (upstream, expected_type, expected_value) in [
            (
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"after tool\"}}]}\n\n",
                    "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
                    "data: [DONE]\n\n"
                ),
                "text",
                "after tool",
            ),
            (
                concat!(
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"after tool\"}}]}\n\n",
                    "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
                    "data: [DONE]\n\n"
                ),
                "thinking",
                "after tool",
            ),
        ] {
            for split in 0..=upstream.len() {
                let mut translator =
                    LiveStreamTranslator::new("msg_after_tool".into(), "glm-5.2".into());
                let mut output = translator.push(&upstream.as_bytes()[..split]).unwrap();
                output.extend(translator.push(&upstream.as_bytes()[split..]).unwrap());
                output.extend(translator.finish().unwrap());
                let rendered = String::from_utf8(output).unwrap();
                let events = crate::anthropic::sse::parse_sse_events(rendered.as_bytes())
                    .into_iter()
                    .map(|event| serde_json::from_str::<Value>(&event.data).unwrap())
                    .collect::<Vec<_>>();
                let tool_starts = events
                    .iter()
                    .enumerate()
                    .filter(|(_, event)| {
                        event["type"] == "content_block_start"
                            && event["content_block"]["type"] == "tool_use"
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                let tool_stops = events
                    .iter()
                    .enumerate()
                    .filter(|(_, event)| {
                        event["type"] == "content_block_stop" && event["index"] == 0
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                let following_start = events
                    .iter()
                    .position(|event| {
                        event["type"] == "content_block_start"
                            && event["content_block"]["type"] == expected_type
                    })
                    .unwrap();
                assert_eq!(tool_starts.len(), 1, "split {split}");
                assert_eq!(tool_stops.len(), 1, "split {split}");
                assert!(tool_starts[0] < tool_stops[0], "split {split}");
                assert!(tool_stops[0] < following_start, "split {split}");
                assert_eq!(
                    rendered
                        .matches("\"id\":\"toolu_msg_after_tool_0\"")
                        .count(),
                    1,
                    "split {split}"
                );
                assert!(rendered.contains("message_stop"), "split {split}");
            }

            let response =
                accumulate_response(upstream.as_bytes(), "msg_after_tool_buffered", "glm-5.2")
                    .unwrap();
            assert_eq!(response["content"][0]["type"], "tool_use");
            assert_eq!(response["content"][1]["type"], expected_type);
            let value_field = if expected_type == "text" {
                "text"
            } else {
                "thinking"
            };
            assert_eq!(response["content"][1][value_field], expected_value);
            assert_eq!(response["stop_reason"], "tool_use");
        }
    }

    #[test]
    fn tool_call_without_identity_still_fails_at_finalize() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let error = accumulate_response(upstream.as_bytes(), "msg_missing", "glm-5.2")
            .unwrap_err()
            .to_string();
        assert!(error.contains("ended without id or function name"));
    }

    #[test]
    fn tool_call_cannot_resume_after_its_content_block_closed() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"after tool\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" \"}}]}}]}\n\n"
        );
        let error = accumulate_response(upstream.as_bytes(), "msg_resumed", "glm-5.2")
            .unwrap_err()
            .to_string();
        assert!(error.contains("continued after its content block was closed"));
    }

    #[test]
    fn new_tool_index_can_start_after_text() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"first\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"between tools\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"second\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = accumulate_response(upstream.as_bytes(), "msg_new_tool", "glm-5.2").unwrap();
        assert_eq!(response["content"][0]["type"], "tool_use");
        assert_eq!(response["content"][1]["type"], "text");
        assert_eq!(response["content"][2]["type"], "tool_use");
        assert_eq!(response["content"][2]["name"], "second");
    }

    #[test]
    fn chat_tool_ids_are_namespaced_per_response() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"Agent_0\",\"function\":{\"name\":\"Agent\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let main = accumulate_response(upstream.as_bytes(), "msg_main", "kimi-k3").unwrap();
        let child = accumulate_response(upstream.as_bytes(), "msg_child", "kimi-k3").unwrap();
        let main_id = main["content"][0]["id"].as_str().unwrap();
        let child_id = child["content"][0]["id"].as_str().unwrap();

        assert_eq!(main_id, "toolu_msg_main_0");
        assert_eq!(child_id, "toolu_msg_child_0");
        assert_ne!(main_id, child_id);
        assert_ne!(main_id, "Agent_0");
        assert_ne!(child_id, "Agent_0");
    }

    #[test]
    fn buffered_response_is_strict_and_complete() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n"
        );
        let response = accumulate_response(upstream.as_bytes(), "msg_2", "glm-5.2").unwrap();
        assert_eq!(response["content"][0]["text"], "hello");
        assert_eq!(response["stop_reason"], "end_turn");
        assert_eq!(response["usage"]["output_tokens"], 1);
    }

    #[test]
    fn malformed_incomplete_and_unrecognized_terminal_streams_fail() {
        let malformed = b"data: not-json\n\n";
        let incomplete = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n";
        let partial_frame = b"data: {\"choices\":[]}";
        let filtered = b"data: {\"choices\":[{\"finish_reason\":\"content_filter\"}]}\n\n";
        let unknown = b"data: {\"choices\":[{\"finish_reason\":\"mystery\"}]}\n\n";

        assert!(accumulate_response(malformed, "m", "model").is_err());
        assert!(accumulate_response(incomplete, "m", "model").is_err());
        assert!(accumulate_response(partial_frame, "m", "model").is_err());
        assert!(accumulate_response(filtered, "m", "model").is_err());
        assert!(accumulate_response(unknown, "m", "model").is_err());
    }

    #[test]
    fn done_is_a_real_terminal_even_without_finish_reason() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = accumulate_response(upstream.as_bytes(), "msg_3", "model").unwrap();
        assert_eq!(response["stop_reason"], "end_turn");
    }

    #[test]
    fn accepts_opencode_cost_metadata_after_done() {
        let upstream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[],\"cost\":\"0\"}\n\n"
        );
        let response = accumulate_response(upstream.as_bytes(), "msg_4", "glm-5.2").unwrap();
        assert_eq!(response["content"][0]["text"], "hello");
        assert_eq!(response["usage"]["output_tokens"], 1);
    }

    #[test]
    fn rejects_content_after_done() {
        let upstream = concat!(
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n"
        );
        assert!(accumulate_response(upstream.as_bytes(), "msg_5", "glm-5.2").is_err());
    }

    #[tokio::test]
    async fn live_stream_rejects_an_incomplete_frame_after_done() {
        let upstream = futures_util::stream::iter([Ok::<Bytes, OpenCodeError>(
            Bytes::from_static(b"data: [DONE]\n\ndata: {"),
        )]);
        let mut state = OpenCodeChatStreamState {
            upstream,
            translator: LiveStreamTranslator::new("msg_4".into(), "model".into()),
            capture_decoder: SseDecoder::default(),
            terminal: false,
            error_sent: false,
            monitor: None,
            req_id: "req".into(),
            bytes: 0,
            chunks: 0,
            stream_capture: None,
            traffic: None,
        };
        let output = state.next_output().await.expect("error event");
        assert!(String::from_utf8_lossy(&output).contains("OpenCode Go stream is invalid"));
    }

    const SYNTHETIC_LABELS: StreamLabels = StreamLabels {
        provider: "Synthetic Gateway",
        capture_stem: "synthetic",
    };
    const TEXT_THEN_STOP: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
        "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n"
    );

    fn strict_wiring() -> StreamWiring {
        StreamWiring {
            labels: SYNTHETIC_LABELS,
            receive: ReceivePolicy {
                terminal: TerminalPolicy::Strict,
                sanitize: None,
            },
        }
    }

    fn strict_body<S>(upstream: S) -> Body
    where
        S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    {
        stream_body_with(
            upstream,
            "msg_strict".into(),
            "synthetic-model".into(),
            None,
            "req".into(),
            None,
            strict_wiring(),
        )
    }

    async fn body_text(body: Body) -> String {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn chunks(parts: &[&str]) -> Vec<Result<Bytes, std::io::Error>> {
        parts
            .iter()
            .map(|part| Ok(Bytes::copy_from_slice(part.as_bytes())))
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_an_event_in_a_later_chunk_after_done() {
        for late in [
            "data: {\"choices\":[],\"cost\":\"0\"}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n",
            "data: not-json\n\n",
        ] {
            let done = format!("{TEXT_THEN_STOP}data: [DONE]\n\n");
            let upstream = futures_util::stream::iter(chunks(&[&done, late]));
            let text = body_text(strict_body(upstream)).await;
            assert!(
                text.contains("Synthetic Gateway stream is invalid"),
                "late {late}: {text}"
            );
            assert!(!text.contains("OpenCode Go"), "late {late}: {text}");
            assert!(!text.contains("message_stop"), "late {late}: {text}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_a_second_done_in_a_later_chunk() {
        let done = format!("{TEXT_THEN_STOP}data: [DONE]\n\n");
        let upstream = futures_util::stream::iter(chunks(&[&done, "data: [DONE]\n\n"]));
        let text = body_text(strict_body(upstream)).await;
        assert!(
            text.contains("Synthetic Gateway stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_completes_without_waiting_when_upstream_closes() {
        let done = format!("{TEXT_THEN_STOP}data: [DONE]\n\n");
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(futures_util::stream::iter(chunks(&[&done])))).await;
        assert!(started.elapsed() < POST_TERMINAL_READ_LIMIT);
        assert!(text.contains("\"text\":\"hello\""), "{text}");
        assert!(text.contains("\"stop_reason\":\"end_turn\""), "{text}");
        assert!(
            text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            "{text}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_completes_at_the_read_limit_when_upstream_stays_open() {
        for wire in [
            format!("{TEXT_THEN_STOP}data: [DONE]\n\n"),
            TEXT_THEN_STOP.to_string(),
            format!("{TEXT_THEN_STOP}data: [DONE]\n\ndata: {{\"choi"),
        ] {
            let upstream =
                futures_util::stream::iter(chunks(&[&wire])).chain(futures_util::stream::pending());
            let started = tokio::time::Instant::now();
            let text = body_text(strict_body(upstream)).await;
            assert!(started.elapsed() >= POST_TERMINAL_READ_LIMIT, "{wire}");
            assert!(
                text.contains("\"stop_reason\":\"end_turn\""),
                "{wire}: {text}"
            );
            assert!(
                text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
                "{wire}: {text}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_sends_message_stop_only_after_the_read_window() {
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let upstream = futures_util::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|chunk| (chunk, receiver))
        });
        let mut stream = wired_stream(
            upstream,
            "msg_held".into(),
            "synthetic-model".into(),
            None,
            "req".into(),
            None,
            strict_wiring(),
        );
        sender
            .send(Ok(Bytes::from(format!("{TEXT_THEN_STOP}data: [DONE]\n\n"))))
            .await
            .unwrap();
        let first = stream
            .state
            .next_output_with(&mut stream.context)
            .await
            .unwrap();
        let first = String::from_utf8(first).unwrap();
        assert!(first.contains("\"text\":\"hello\""), "{first}");
        assert!(!first.contains("message_stop"), "{first}");
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(20),
                stream.state.next_output_with(&mut stream.context)
            )
            .await
            .is_err()
        );
        drop(sender);
        let last = stream
            .state
            .next_output_with(&mut stream.context)
            .await
            .unwrap();
        assert!(String::from_utf8(last).unwrap().contains("message_stop"));
        assert!(
            stream
                .state
                .next_output_with(&mut stream.context)
                .await
                .is_none()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_more_than_the_post_terminal_byte_limit() {
        let done = format!("{TEXT_THEN_STOP}data: [DONE]\n\n");
        // Comment-only frames carry no event, so only the byte limit stops them.
        let comment = format!(":{}\n\n", "x".repeat(512 * 1024));
        let mut parts = vec![done.as_str()];
        parts.extend(std::iter::repeat_n(
            comment.as_str(),
            POST_TERMINAL_MAX_BYTES / comment.len() + 1,
        ));
        let upstream =
            futures_util::stream::iter(chunks(&parts)).chain(futures_util::stream::pending());
        let text = body_text(strict_body(upstream)).await;
        assert!(
            text.contains("Synthetic Gateway stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[test]
    fn synthetic_strict_buffered_response_rejects_events_after_done() {
        let wiring = strict_wiring();
        for tail in [
            "data: {\"choices\":[],\"cost\":\"0\"}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n",
        ] {
            let upstream = format!("{TEXT_THEN_STOP}data: [DONE]\n\n{tail}");
            let error = accumulate_response_with(upstream.as_bytes(), "msg", "model", &wiring)
                .unwrap_err()
                .to_string();
            assert!(error.starts_with("Synthetic Gateway"), "{tail}: {error}");
        }
        let upstream = format!("{TEXT_THEN_STOP}data: [DONE]\n\n");
        let response =
            accumulate_response_with(upstream.as_bytes(), "msg", "model", &wiring).unwrap();
        assert_eq!(response["content"][0]["text"], "hello");
        assert_eq!(response["stop_reason"], "end_turn");
    }

    #[test]
    fn synthetic_labels_name_the_provider_in_translation_errors() {
        let wiring = StreamWiring {
            labels: SYNTHETIC_LABELS,
            receive: ReceivePolicy::default(),
        };
        for (upstream, expected) in [
            (
                "data: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n",
                "Synthetic Gateway event after terminal completion",
            ),
            (
                "data: {\"choices\":[{\"finish_reason\":\"mystery\"}]}\n\n",
                "unsupported Synthetic Gateway finish_reason: mystery",
            ),
            (
                "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
                "Synthetic Gateway stream ended without [DONE] or finish_reason",
            ),
            (
                "data: not-json\n\n",
                "malformed Synthetic Gateway SSE event",
            ),
        ] {
            let error = accumulate_response_with(upstream.as_bytes(), "msg", "model", &wiring)
                .unwrap_err()
                .to_string();
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn synthetic_sanitizer_cleans_the_upstream_error_of_a_buffered_response() {
        let sanitize: Sanitizer = Arc::new(|text: &str| text.replace("SECRET-123", "[redacted]"));
        let wiring = StreamWiring {
            labels: SYNTHETIC_LABELS,
            receive: ReceivePolicy {
                terminal: TerminalPolicy::Strict,
                sanitize: Some(sanitize),
            },
        };
        let upstream = b"data: {\"error\":{\"message\":\"quota for SECRET-123 exhausted\"}}\n\n";
        let error = accumulate_response_with(upstream, "msg", "model", &wiring)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "Synthetic Gateway upstream error: quota for [redacted] exhausted"
        );
        let unsanitized = accumulate_response(upstream, "msg", "model")
            .unwrap_err()
            .to_string();
        assert!(unsanitized.contains("SECRET-123"), "{unsanitized}");
    }

    #[test]
    fn sanitizer_reaches_every_message_string_of_a_captured_event() {
        let policy = ReceivePolicy {
            terminal: TerminalPolicy::Strict,
            sanitize: Some(Arc::new(|text: &str| text.replace("SECRET", "[redacted]"))),
        };
        let value = json!({
            "error":{"message":"SECRET in error"},
            "response":{"error":{"message":"SECRET in response"}},
            "items":[{"message":"SECRET in array"}],
            "message":{"content":"SECRET is not a message string"}
        });
        let captured = policy.upstream_value(&value);
        assert_eq!(captured["error"]["message"], "[redacted] in error");
        assert_eq!(
            captured["response"]["error"]["message"],
            "[redacted] in response"
        );
        assert_eq!(captured["items"][0]["message"], "[redacted] in array");
        assert_eq!(
            captured["message"]["content"],
            "SECRET is not a message string"
        );
        assert!(matches!(
            ReceivePolicy::default().upstream_value(&value),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn untouched_effort_policy_leaves_effort_to_the_caller_for_a_model_the_default_rejects() {
        let req = request(json!({
            "model":"vertex/zai-org/glm-5.2-maas",
            "messages":[{"role":"user","content":"hello"}],
            "output_config":{"effort":"low"}
        }));
        let error = prepare_request(&req, "zai-org/glm-5.2-maas")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not support reasoning effort low"),
            "{error}"
        );

        let policy = RequestPolicy {
            effort: EffortPolicy::Untouched,
            reasoning_replay: ReasoningReplay::OpenCodeGo,
        };
        let wire = serde_json::to_value(
            prepare_request_with(&req, "zai-org/glm-5.2-maas", policy).unwrap(),
        )
        .unwrap();
        assert_eq!(wire["model"], "zai-org/glm-5.2-maas");
        assert!(wire.get("reasoning_effort").is_none(), "{wire}");

        let req = request(json!({
            "messages":[{"role":"user","content":"hello"}],
            "output_config":{"effort":"bogus"}
        }));
        assert!(prepare_request_with(&req, "zai-org/glm-5.2-maas", policy).is_ok());
    }

    #[test]
    fn reasoning_replay_policy_controls_reasoning_content() {
        let req = request(json!({
            "messages":[
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"reason","signature":"opaque"},
                    {"type":"text","text":"first"}
                ]},
                {"role":"user","content":"next"},
                {"role":"assistant","content":[{"type":"text","text":"second"}]},
                {"role":"user","content":"again"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"only reasoning","signature":"opaque"}
                ]},
                {"role":"user","content":"last"}
            ]
        }));
        let wire = |model: &str, reasoning_replay| {
            let policy = RequestPolicy {
                effort: EffortPolicy::OpenCodeGo,
                reasoning_replay,
            };
            serde_json::to_value(prepare_request_with(&req, model, policy).unwrap()).unwrap()
        };

        let never = wire("deepseek-v4-pro", ReasoningReplay::Never);
        let roles = never["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(roles, ["assistant", "user", "assistant", "user", "user"]);
        assert!(never["messages"][0].get("reasoning_content").is_none());
        assert!(never["messages"][2].get("reasoning_content").is_none());

        let always = wire("zai-org/glm-5.2-maas", ReasoningReplay::Always);
        assert_eq!(always["messages"][0]["reasoning_content"], "reason");
        assert_eq!(always["messages"][2]["reasoning_content"], "");
        assert_eq!(always["messages"][4]["reasoning_content"], "only reasoning");

        let present = wire("deepseek-v4-pro", ReasoningReplay::WhenPresent);
        assert_eq!(present["messages"][0]["reasoning_content"], "reason");
        assert!(present["messages"][2].get("reasoning_content").is_none());

        let default = wire("deepseek-v4-pro", ReasoningReplay::OpenCodeGo);
        assert_eq!(default["messages"][2]["reasoning_content"], "");
        assert_eq!(
            serde_json::to_value(prepare_request(&req, "deepseek-v4-pro").unwrap()).unwrap(),
            default
        );
    }

    fn paced_upstream(
        steps: Vec<(u64, &'static str)>,
    ) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
        futures_util::stream::unfold(steps.into_iter(), |mut steps| async move {
            let (delay_ms, part) = steps.next()?;
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            Some((Ok(Bytes::from_static(part.as_bytes())), steps))
        })
        .chain(futures_util::stream::pending())
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_counts_the_read_window_again_from_a_late_done() {
        let late = "data: {\"choices\":[],\"cost\":\"0\"}\n\n";
        let text = body_text(strict_body(paced_upstream(vec![
            (0, TEXT_THEN_STOP),
            (1_500, "data: [DONE]\n\n"),
            (1_500, late),
        ])))
        .await;
        assert!(
            text.contains("Synthetic Gateway stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("message_stop"), "{text}");

        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(paced_upstream(vec![
            (0, TEXT_THEN_STOP),
            (1_500, "data: [DONE]\n\n"),
            (2_500, late),
        ])))
        .await;
        assert!(
            started.elapsed() >= std::time::Duration::from_millis(1_500) + POST_TERMINAL_READ_LIMIT
        );
        assert!(
            text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            "{text}"
        );
    }

    #[test]
    fn synthetic_post_terminal_limits_are_two_seconds_and_sixteen_mib() {
        assert_eq!(POST_TERMINAL_READ_LIMIT, std::time::Duration::from_secs(2));
        assert_eq!(POST_TERMINAL_MAX_BYTES, 16 * 1024 * 1024);
    }

    #[test]
    fn synthetic_frame_scanner_ends_frames_where_the_decoder_does_in_any_split() {
        let tests: [(&str, &str, &[usize]); 6] = [
            ("lf", "data: a\n\n: c\n\n", &[9, 14]),
            ("cr", "data: a\r\rdata: b\r\r", &[9, 18]),
            ("crlf", "data: a\r\n\r\ndata: b\r\n\r\n", &[10, 21]),
            ("leading blank lines", "\n\ndata: a\n\n", &[1, 2, 11]),
            (
                "trailing line without blank line",
                "data: a\n\ndata: b\n",
                &[9],
            ),
            ("no frame end", "data: a", &[]),
        ];
        for (name, input, expected) in tests {
            let input = input.as_bytes();
            assert_eq!(
                FrameScanner::default().frame_ends(input),
                expected,
                "{name}: whole"
            );
            for split in 0..=input.len() {
                let mut scanner = FrameScanner::default();
                let mut ends = scanner.frame_ends(&input[..split]);
                ends.extend(
                    scanner
                        .frame_ends(&input[split..])
                        .into_iter()
                        .map(|end| end + split),
                );
                assert_eq!(ends, expected, "{name}: split at {split}");

                let prefix = complete_frames_prefix(&input[..split]);
                let mut whole = SseDecoder::default();
                let events = whole.push(&input[..split]).unwrap();
                let mut trimmed = SseDecoder::default();
                let kept = trimmed.push(prefix).unwrap();
                assert!(
                    trimmed.finish().is_ok(),
                    "{name}: prefix of {split} ends mid-frame"
                );
                assert_eq!(
                    kept.len(),
                    events.len(),
                    "{name}: prefix of {split} lost an event"
                );
            }
        }
    }

    #[test]
    fn synthetic_complete_frames_prefix_keeps_whole_frames_only() {
        let tests = [
            ("empty input", "", ""),
            ("one whole frame", "data: a\n\n", "data: a\n\n"),
            (
                "partial line after a frame",
                "data: a\n\ndata: b",
                "data: a\n\n",
            ),
            (
                "line without its blank line",
                "data: a\n\ndata: b\n",
                "data: a\n\n",
            ),
            (
                "crlf frame keeps both bytes",
                "data: a\r\n\r\ndata: b",
                "data: a\r\n\r\n",
            ),
            ("cr frame", "data: a\r\rdata", "data: a\r\r"),
            (
                "partial comment after a comment frame",
                ": keep\n\n: par",
                ": keep\n\n",
            ),
            ("no complete frame", "data: a", ""),
            ("frame then blank line", "data: a\n\n\n", "data: a\n\n\n"),
        ];
        for (name, input, expected) in tests {
            assert_eq!(
                complete_frames_prefix(input.as_bytes()),
                expected.as_bytes(),
                "{name}"
            );
        }
    }

    /// Exactly `16 * 1024 * 1024` bytes of 1024-byte comment frames. They
    /// carry no event, so only the byte limit can stop them.
    fn synthetic_comment_tail_of_sixteen_mib() -> String {
        let frame = format!(": {}\n\n", "x".repeat(1020));
        assert_eq!(frame.len(), 1024);
        let tail = frame.repeat(16 * 1024);
        assert_eq!(tail.len(), 16 * 1024 * 1024);
        tail
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_charges_the_tail_after_finish_reason_by_position() {
        let at_limit = synthetic_comment_tail_of_sixteen_mib();
        let over_limit = format!("{at_limit}\n");
        for (tail, accepted) in [(at_limit.as_str(), true), (over_limit.as_str(), false)] {
            let wire = format!("{TEXT_THEN_STOP}{tail}");
            let terminal_end = TEXT_THEN_STOP.len();
            let inside_tail = terminal_end + 4096 + 7;
            for parts in [
                vec![wire.as_str()],
                vec![&wire[..terminal_end], &wire[terminal_end..]],
                vec![&wire[..inside_tail], &wire[inside_tail..]],
            ] {
                let upstream = futures_util::stream::iter(chunks(&parts));
                let text = body_text(strict_body(upstream)).await;
                let case = format!("{} tail bytes in {} items", tail.len(), parts.len());
                assert_eq!(
                    text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
                    accepted,
                    "{case}"
                );
                assert_eq!(
                    text.contains("Synthetic Gateway stream is invalid"),
                    !accepted,
                    "{case}"
                );
            }
            let buffered =
                accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring());
            assert_eq!(
                buffered.is_ok(),
                accepted,
                "{} tail bytes: {buffered:?}",
                tail.len()
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_waits_exactly_two_seconds_after_the_terminal_event() {
        for wire in [
            TEXT_THEN_STOP.to_string(),
            format!("{TEXT_THEN_STOP}data: [DONE]\n\n"),
        ] {
            let upstream =
                futures_util::stream::iter(chunks(&[&wire])).chain(futures_util::stream::pending());
            let started = tokio::time::Instant::now();
            let text = body_text(strict_body(upstream)).await;
            assert_eq!(
                started.elapsed(),
                std::time::Duration::from_secs(2),
                "{wire}"
            );
            assert!(
                text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
                "{wire}: {text}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_content_after_finish_reason() {
        let tests = [
            (
                "text",
                "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n",
            ),
            (
                "reasoning",
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"late\"}}]}\n\n",
            ),
            (
                "tool call delta",
                "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{}\"}}]}}]}\n\n",
            ),
            (
                "different finish reason",
                "data: {\"choices\":[{\"finish_reason\":\"length\"}]}\n\n",
            ),
        ];
        for (name, late) in tests {
            let tail = format!("{late}data: [DONE]\n\n");
            let wire = format!("{TEXT_THEN_STOP}{tail}");
            for parts in [vec![wire.as_str()], vec![TEXT_THEN_STOP, tail.as_str()]] {
                let text = body_text(strict_body(futures_util::stream::iter(chunks(&parts)))).await;
                let case = format!("{name} in {} items", parts.len());
                assert!(
                    text.contains("Synthetic Gateway stream is invalid"),
                    "{case}: {text}"
                );
                assert!(!text.contains("message_stop"), "{case}: {text}");
                assert!(!text.contains("late"), "{case}: {text}");
            }
            let error = accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring())
                .unwrap_err()
                .to_string();
            assert!(error.starts_with("Synthetic Gateway"), "{name}: {error}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_accepts_usage_and_one_done_after_finish_reason() {
        let tests = [
            (
                "usage only",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7}}\n\n",
            ),
            (
                "empty delta with the same finish reason",
                "data: {\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7}}\n\n",
            ),
        ];
        for (name, after) in tests {
            let wire = format!("{TEXT_THEN_STOP}{after}data: [DONE]\n\n");
            let text = body_text(strict_body(futures_util::stream::iter(chunks(&[&wire])))).await;
            assert!(text.contains("\"text\":\"hello\""), "{name}: {text}");
            assert!(text.contains("\"output_tokens\":7"), "{name}: {text}");
            assert!(
                text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
                "{name}: {text}"
            );
            let response =
                accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring())
                    .unwrap();
            assert_eq!(response["content"][0]["text"], "hello", "{name}");
            assert_eq!(response["usage"]["output_tokens"], 7, "{name}");
        }
    }

    #[test]
    fn synthetic_default_policy_still_accepts_content_after_finish_reason() {
        let upstream = format!(
            "{TEXT_THEN_STOP}data: {{\"choices\":[{{\"delta\":{{\"content\":\"late\"}}}}]}}\n\ndata: [DONE]\n\n"
        );
        let response = accumulate_response(upstream.as_bytes(), "msg", "glm-5.2").unwrap();
        assert_eq!(response["content"][0]["text"], "hellolate");
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_labels_name_the_provider_in_sse_framing_errors() {
        let oversized = format!("data: {}\n\n", "x".repeat(1024 * 1024));
        let tests: [(&str, &[u8], &str); 3] = [
            (
                "incomplete frame at the end",
                b"data: {\"choices\":[]}",
                "Synthetic Gateway stream ended with an incomplete frame",
            ),
            (
                "frame over the decoder's size limit",
                oversized.as_bytes(),
                "Synthetic Gateway stream sent an SSE frame that is too large or not valid UTF-8",
            ),
            (
                "invalid UTF-8",
                b"data: \xff\n\n",
                "Synthetic Gateway stream sent an SSE frame that is too large or not valid UTF-8",
            ),
        ];
        for (name, upstream, expected) in tests {
            for receive in [ReceivePolicy::default(), strict_wiring().receive] {
                let wiring = StreamWiring {
                    labels: SYNTHETIC_LABELS,
                    receive,
                };
                let error = accumulate_response_with(upstream, "msg", "model", &wiring)
                    .unwrap_err()
                    .to_string();
                assert_eq!(error, expected, "{name}, {:?}", wiring.receive.terminal);

                let body = stream_body_with(
                    futures_util::stream::iter([Ok::<Bytes, std::io::Error>(
                        Bytes::copy_from_slice(upstream),
                    )]),
                    "msg".into(),
                    "model".into(),
                    None,
                    "req".into(),
                    None,
                    wiring,
                );
                let text = body_text(body).await;
                assert!(
                    text.contains("Synthetic Gateway stream is invalid"),
                    "{name}: {text}"
                );
                assert!(!text.contains("Grok"), "{name}: {text}");
            }
        }
        let error = accumulate_response(b"data: {\"choices\":[]}", "msg", "model")
            .unwrap_err()
            .to_string();
        assert_eq!(error, "Grok SSE stream ended with an incomplete frame");
    }

    #[test]
    fn synthetic_raw_frame_ends_keep_both_bytes_of_a_crlf_in_any_split() {
        let tests: [(&str, &str, &[usize]); 5] = [
            ("lf", "data: a\n\n: c\n\n", &[9, 14]),
            ("cr", "data: a\r\rdata: b\r\r", &[9, 18]),
            ("crlf", "data: a\r\n\r\ndata: b\r\n\r\n", &[11, 22]),
            ("cr then crlf", "data: a\r\rdata: b\r\n\r\n", &[9, 20]),
            (
                "crlf then partial frame",
                "data: a\r\n\r\ndata: b\r\n",
                &[11],
            ),
        ];
        for (name, input, expected) in tests {
            let input = input.as_bytes();
            let whole = FrameScanner::default().raw_frame_ends(input);
            assert_eq!(whole.continued, 0, "{name}: whole");
            assert_eq!(whole.ends, expected, "{name}: whole");
            for split in 0..=input.len() {
                let mut scanner = FrameScanner::default();
                let first = scanner.raw_frame_ends(&input[..split]);
                let second = scanner.raw_frame_ends(&input[split..]);
                assert_eq!(first.continued, 0, "{name}: split at {split}");
                let mut ends = first.ends;
                if second.continued == 1 {
                    // The LF that opens the second part completes the frame
                    // that ended at the CR closing the first.
                    assert_eq!(ends.pop(), Some(split), "{name}: split at {split}");
                    ends.push(split + 1);
                }
                ends.extend(second.ends.into_iter().map(|end| end + split));
                assert_eq!(ends, expected, "{name}: split at {split}");
            }
        }
        // An empty chunk between the CR and the LF changes nothing.
        let mut scanner = FrameScanner::default();
        assert_eq!(scanner.raw_frame_ends(b"data: a\r\n\r").ends, [10]);
        assert_eq!(scanner.raw_frame_ends(b""), RawFrameEnds::default());
        let rest = scanner.raw_frame_ends(b"\ndata: b\r\n\r\n");
        assert_eq!(rest.continued, 1);
        assert_eq!(rest.ends, [12]);
        // An LF after a CR that only ended a line is not a frame's.
        let mut scanner = FrameScanner::default();
        assert_eq!(
            scanner.raw_frame_ends(b"data: a\r").ends,
            Vec::<usize>::new()
        );
        assert_eq!(scanner.raw_frame_ends(b"\n\r\n").continued, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_charges_no_byte_of_a_crlf_terminal_frame_to_the_tail() {
        let message = TEXT_THEN_STOP.replace('\n', "\r\n");
        let at_limit = synthetic_comment_tail_of_sixteen_mib();
        let over_limit = format!("{at_limit}\n");
        for (tail, accepted) in [(at_limit.as_str(), true), (over_limit.as_str(), false)] {
            let wire = format!("{message}{tail}");
            let terminal_end = message.len();
            let between_cr_and_lf = terminal_end - 1;
            let inside_tail = terminal_end + 4096 + 7;
            for (split, parts) in [
                ("whole", vec![wire.as_str()]),
                (
                    "after the frame",
                    vec![&wire[..terminal_end], &wire[terminal_end..]],
                ),
                (
                    "between its CR and LF",
                    vec![&wire[..between_cr_and_lf], &wire[between_cr_and_lf..]],
                ),
                (
                    "inside the tail",
                    vec![&wire[..inside_tail], &wire[inside_tail..]],
                ),
            ] {
                let upstream = futures_util::stream::iter(chunks(&parts));
                let text = body_text(strict_body(upstream)).await;
                let case = format!("{} tail bytes, split {split}", tail.len());
                assert_eq!(
                    text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
                    accepted,
                    "{case}"
                );
                assert_eq!(
                    text.contains("Synthetic Gateway stream is invalid"),
                    !accepted,
                    "{case}"
                );
            }
            let buffered =
                accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring());
            assert_eq!(
                buffered.is_ok(),
                accepted,
                "{} tail bytes: {buffered:?}",
                tail.len()
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_starts_the_read_window_at_the_finish_reason_not_at_the_next_poll()
     {
        let late = "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n";
        for (delay_ms, settles) in [(3_000, true), (1_000, false)] {
            let pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = Arc::clone(&pulled);
            // The late chunk is ready as soon as it is polled for.
            let upstream = futures_util::stream::iter(chunks(&[TEXT_THEN_STOP, late]))
                .chain(futures_util::stream::pending())
                .inspect(move |_| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                });
            let started = tokio::time::Instant::now();
            let mut body = strict_body(upstream).into_data_stream();
            let first = body.next().await.unwrap().unwrap();
            assert!(
                String::from_utf8_lossy(&first).contains("hello"),
                "{delay_ms} ms: {first:?}"
            );
            // The client takes its time before it asks for more.
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            let mut rest = String::new();
            while let Some(item) = body.next().await {
                rest.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
            }
            let case = format!("next poll after {delay_ms} ms: {rest}");
            assert_eq!(
                rest.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
                settles,
                "{case}"
            );
            assert_eq!(
                rest.contains("Synthetic Gateway stream is invalid"),
                !settles,
                "{case}"
            );
            if settles {
                // The window had passed: the waiting chunk was not read.
                assert_eq!(
                    pulled.load(std::sync::atomic::Ordering::SeqCst),
                    1,
                    "{case}"
                );
                assert_eq!(
                    started.elapsed(),
                    std::time::Duration::from_millis(3_000),
                    "{case}"
                );
            }
        }
    }
}
