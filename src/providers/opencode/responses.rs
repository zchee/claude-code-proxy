use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::anthropic::schema::MessagesRequest;
use crate::anthropic::sse::{SseEvent, encode_sse_event};
use crate::monitor::{MonitorHandle, usage_from_anthropic_sse};
use crate::providers::codex::translate::{
    IncompleteResponsePolicy, accumulate::accumulate_response_with_policy,
    live_stream::LiveStreamTranslator, request::translate_openai_compatible_request,
};
use crate::providers::grok::translate::stream::SseDecoder;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use super::chat::{
    RawFrameEnds, ReceiveContext, RedactedWiring, StreamLabels, StreamWiring, TerminalPolicy,
    UpstreamStream, box_upstream,
};
use super::client::{OpenCodeError, OpenCodeResponse};

/// Responses event types the Codex translator acts on: the arms of
/// `LiveStreamTranslator::accept` and the failure events its classifier
/// turns into an error with the upstream's status and message. Under
/// [`TerminalPolicy::Strict`] a frame whose `type` is in neither this list
/// nor [`RESPONSES_HARMLESS_EVENT_TYPES`] is an error.
pub const RESPONSES_HANDLED_EVENT_TYPES: &[&str] = &[
    "codex.rate_limits",
    "keepalive",
    "response.created",
    "response.in_progress",
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
    "response.output_item.added",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.output_text.delta",
    "response.output_text.annotation.added",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.output_item.done",
    "response.completed",
    "response.incomplete",
    "response.done",
    "response.failed",
    "response.error",
    "error",
];

/// Responses event types the Codex translator ignores and that carry
/// nothing it would need: progress notifications and the `.done` events
/// that repeat what the deltas already delivered. Under
/// [`TerminalPolicy::Strict`] a snapshot here, or a `response.output_item.done`,
/// that carries output text or reasoning summary text when no delta of that
/// kind has arrived in the stream is an error: the translator would drop
/// that text. So is a content part, or a message item's part, that carries
/// a non-empty `refusal`, which the translator has no output for.
pub const RESPONSES_HARMLESS_EVENT_TYPES: &[&str] = &[
    "response.queued",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.done",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.done",
];

const TERMINAL_EVENT_TYPES: [&str; 3] =
    ["response.completed", "response.incomplete", "response.done"];

/// Which kinds of text a strict stream has received as deltas, so that a
/// snapshot repeating them can be told from one that carries text no delta
/// delivered.
#[derive(Debug, Default)]
struct DeltasSeen {
    output_text: bool,
    reasoning_summary: bool,
}

impl DeltasSeen {
    /// Notes the delta `value` carries, or returns why a strict stream
    /// rejects it: a snapshot with non-empty text of a kind no delta has
    /// delivered.
    fn check(&mut self, value: &serde_json::Value) -> Option<&'static str> {
        let has_text = |text: Option<&serde_json::Value>| {
            text.and_then(serde_json::Value::as_str)
                .is_some_and(|text| !text.is_empty())
        };
        let event_type = value.get("type")?.as_str()?;
        if carries_refusal(event_type, value) {
            return Some("refusal_in_snapshot");
        }
        let (carries_text, delivered) = match event_type {
            "response.output_text.delta" => {
                self.output_text |= has_text(value.get("delta"));
                return None;
            }
            "response.reasoning_summary_text.delta" => {
                self.reasoning_summary |= has_text(value.get("delta"));
                return None;
            }
            "response.output_text.done" => (has_text(value.get("text")), self.output_text),
            "response.content_part.added" | "response.content_part.done" => {
                (has_text(value.pointer("/part/text")), self.output_text)
            }
            "response.reasoning_summary_text.done" => {
                (has_text(value.get("text")), self.reasoning_summary)
            }
            "response.reasoning_summary_part.done" => (
                has_text(value.pointer("/part/text")),
                self.reasoning_summary,
            ),
            "response.output_item.done" => {
                let item = value.get("item")?;
                let (parts, delivered) = match item.get("type")?.as_str()? {
                    "message" => (item.get("content"), self.output_text),
                    "reasoning" => (item.get("summary"), self.reasoning_summary),
                    _ => return None,
                };
                let carries_text = parts
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|parts| parts.iter().any(|part| has_text(part.get("text"))));
                (carries_text, delivered)
            }
            _ => return None,
        };
        (carries_text && !delivered).then_some("snapshot_without_delta")
    }
}

/// Whether a snapshot the translator ignores carries a refusal: a
/// `response.content_part.*` part, or a part of a message
/// `response.output_item.done`, with a non-empty `refusal`. The translator
/// has no refusal output, so the refusal would be lost.
fn carries_refusal(event_type: &str, value: &serde_json::Value) -> bool {
    let refuses = |part: &serde_json::Value| {
        part.get("refusal")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|refusal| !refusal.is_empty())
    };
    match event_type {
        "response.content_part.added" | "response.content_part.done" => {
            value.get("part").is_some_and(refuses)
        }
        "response.output_item.done" => value.get("item").is_some_and(|item| {
            item.get("type").and_then(serde_json::Value::as_str) == Some("message")
                && item
                    .get("content")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|parts| parts.iter().any(refuses))
        }),
        _ => false,
    }
}

/// Why a strict stream rejects a parsed frame, or `None` when its `type`
/// is listed.
fn unlisted_frame(value: &serde_json::Value) -> Option<&'static str> {
    let Some(event_type) = value.get("type").and_then(serde_json::Value::as_str) else {
        return Some("missing_event_type");
    };
    let listed = RESPONSES_HANDLED_EVENT_TYPES.contains(&event_type)
        || RESPONSES_HARMLESS_EVENT_TYPES.contains(&event_type);
    (!listed).then_some("unlisted_event_type")
}

pub fn prepare_request(
    body: &MessagesRequest,
    model: &str,
    session_id: Option<String>,
) -> anyhow::Result<serde_json::Value> {
    let translated = translate_openai_compatible_request(body, model.to_string(), session_id)?;
    let mut value = serde_json::to_value(translated)?;
    if let Some(max_tokens) = body.max_tokens.filter(|value| *value > 0) {
        value["max_output_tokens"] = serde_json::json!(max_tokens);
    }
    Ok(value)
}

pub fn accumulate_response(
    upstream: &[u8],
    message_id: &str,
    model: &str,
) -> anyhow::Result<serde_json::Value> {
    accumulate_response_with(upstream, message_id, model, &StreamWiring::default())
}

/// [`accumulate_response`] with the labels and receive policy of `wiring`.
/// The error goes through [`super::chat::ReceivePolicy::sanitize`].
///
/// `upstream` is the whole body, already read, so the caller that read it
/// owns [`super::chat::POST_TERMINAL_READ_LIMIT`]. When that limit ended the
/// read, the caller passes [`super::chat::complete_frames_prefix`] of what
/// it read, dropping a frame the limit cut off; when the upstream closed the
/// body, the caller passes it whole, and a partial frame at its end is an
/// error.
///
/// Under [`TerminalPolicy::Strict`] every frame is checked as the
/// streaming path checks it before the body is translated: any event after
/// the completion, `ping` included, is an error, and so is more than
/// [`super::chat::POST_TERMINAL_MAX_BYTES`] after the end of the
/// completion's frame.
///
/// # Errors
///
/// The body is not a complete, valid Responses stream, or it reports an
/// upstream failure.
pub fn accumulate_response_with(
    upstream: &[u8],
    message_id: &str,
    model: &str,
    wiring: &StreamWiring,
) -> anyhow::Result<serde_json::Value> {
    let result = if wiring.receive.terminal == TerminalPolicy::Strict {
        check_buffered_frames(upstream, &wiring.labels)
    } else {
        Ok(())
    };
    result
        .and_then(|()| {
            accumulate_response_with_policy(
                upstream,
                message_id,
                model,
                IncompleteResponsePolicy::AllowMaxOutputTokens,
            )
        })
        .map_err(|error| wiring.receive.upstream_error(error))
}

/// The strict frame rules over a whole buffered body. The frames are also
/// run through the live translator: when it closes the message by itself,
/// the body is settled there, as on the streaming path, and what follows is
/// not checked. The accumulator stops at the same point.
fn check_buffered_frames(upstream: &[u8], labels: &StreamLabels) -> anyhow::Result<()> {
    let mut observer = StrictResponsesObserver::new(*labels);
    match observer.push(upstream)? {
        ObservedChunk::Settled { .. } => Ok(()),
        ObservedChunk::Reading => observer.finish(),
    }
}

/// What [`StrictResponsesObserver::push`] found in one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedChunk {
    /// No frame so far ends the body; keep reading.
    Reading,
    /// The body is settled `at` bytes into the chunk, at the end of the
    /// frame that settled it: the translator closed the message by itself
    /// (it repaired a `Read` call whose arguments stalled on whitespace), or
    /// the upstream reported a failure. The upstream is not read again;
    /// [`accumulate_response_with`] over the body up to there returns the
    /// repaired message or the failure.
    Settled {
        /// Bytes of the chunk that belong to the settled body.
        at: usize,
    },
}

/// The strict frame rules of [`accumulate_response_with`], applied chunk by
/// chunk while a buffered body is still being read, so that the reader
/// stops where the streaming path stops instead of waiting for the
/// upstream to close.
///
/// [`Self::push`] fails at the first frame the strict policy rejects: an
/// SSE frame the decoder cannot read, an event that is not JSON, a `type`
/// in neither [`RESPONSES_HANDLED_EVENT_TYPES`] nor
/// [`RESPONSES_HARMLESS_EVENT_TYPES`], a snapshot carrying text no delta
/// delivered or a refusal, `[DONE]` before the completion or twice, any
/// other event after the completion, or more than
/// [`super::chat::POST_TERMINAL_MAX_BYTES`] after the completion's frame.
/// The error says `<provider> Responses stream is invalid: <kind>` and
/// carries no upstream text. Under the strict policy
/// [`accumulate_response_with`] over the bytes read so far fails with the
/// same error, so a reader may stop reading at the error and let the
/// accumulator report it. After an error or [`ObservedChunk::Settled`] the
/// observer is not fed again.
pub struct StrictResponsesObserver {
    labels: StreamLabels,
    translator: LiveStreamTranslator,
    decoder: SseDecoder,
    context: ReceiveContext,
    deltas: DeltasSeen,
    completed: bool,
    done_seen: bool,
}

impl StrictResponsesObserver {
    /// An observer whose errors name `labels.provider`.
    pub fn new(labels: StreamLabels) -> Self {
        Self {
            labels,
            translator: LiveStreamTranslator::new("", "")
                .with_incomplete_response_policy(IncompleteResponsePolicy::AllowMaxOutputTokens),
            decoder: SseDecoder::default(),
            context: ReceiveContext::new(StreamWiring {
                labels,
                receive: super::chat::ReceivePolicy {
                    terminal: TerminalPolicy::Strict,
                    sanitize: None,
                },
            }),
            deltas: DeltasSeen::default(),
            completed: false,
            done_seen: false,
        }
    }

    /// Checks the next chunk of the body, one frame at a time.
    ///
    /// # Errors
    ///
    /// The first frame the strict policy rejects, as listed on the type.
    pub fn push(&mut self, chunk: &[u8]) -> anyhow::Result<ObservedChunk> {
        let RawFrameEnds { continued, ends } = self.context.raw_frame_ends(chunk);
        if self.completed && self.context.charge_after_terminal(chunk, continued) {
            return Err(self.invalid("post_terminal_limit"));
        }
        let mut start = 0;
        for end in ends.into_iter().chain([chunk.len()]) {
            if end == start {
                continue;
            }
            let events = self
                .decoder
                .push(&chunk[start..end])
                .map_err(|_| self.invalid("malformed_sse"))?;
            start = end;
            let completed_before = self.completed;
            for event in events {
                if self.settles(&event)? {
                    return Ok(ObservedChunk::Settled { at: end });
                }
            }
            if !completed_before && self.completed && self.context.terminal_frame_ended(chunk, end)
            {
                return Err(self.invalid("post_terminal_limit"));
            }
        }
        Ok(ObservedChunk::Reading)
    }

    /// Ends the body where the upstream closed it.
    ///
    /// # Errors
    ///
    /// The body ends inside a frame.
    pub fn finish(&mut self) -> anyhow::Result<()> {
        self.decoder
            .finish()
            .map_err(|_| self.invalid("incomplete_stream"))
    }

    fn invalid(&self, kind: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "{} Responses stream is invalid: {kind}",
            self.labels.provider
        )
    }

    /// Checks one event; `true` when the body is settled at it.
    fn settles(&mut self, event: &SseEvent) -> anyhow::Result<bool> {
        let data = event.data.trim();
        if data == "[DONE]" {
            if !self.completed || self.done_seen {
                return Err(self.invalid("premature_done"));
            }
            self.done_seen = true;
            return Ok(false);
        }
        if self.completed {
            return Err(self.invalid("event_after_completion"));
        }
        let value: serde_json::Value =
            serde_json::from_str(data).map_err(|_| self.invalid("malformed_event"))?;
        if let Some(kind) = unlisted_frame(&value).or_else(|| self.deltas.check(&value)) {
            return Err(self.invalid(kind));
        }
        self.completed = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|event_type| TERMINAL_EVENT_TYPES.contains(&event_type));
        // An upstream failure is reported by the accumulator, so a
        // translation error settles the body without an error of its own.
        Ok(!self.completed
            && self
                .translator
                .accept(&value, None)
                .map_or(true, |_| self.translator.is_finished()))
    }
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

/// The Anthropic SSE body translated from any Responses upstream byte
/// stream, with the labels and receive policy of `wiring`.
///
/// Under [`TerminalPolicy::Strict`], `response.completed` (or
/// `response.incomplete` or `response.done`) is the only terminal event the
/// upstream sends: it is held, and the read window of
/// [`super::chat::POST_TERMINAL_READ_LIMIT`] starts after it. One `[DONE]`
/// after it is accepted and does not start the window again.
///
/// The translator can also close the message by itself, before the
/// upstream's completion: it repairs a `Read` call whose arguments stalled
/// on whitespace. The stream is settled there, as the Codex provider settles
/// it: the closing goes to the client at once, the upstream is not read
/// again and is dropped, and whatever it would have sent afterwards is
/// neither waited for nor checked. The read window and the byte limit apply
/// only to a completion the upstream sends.
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
    wired_body(stream)
}

/// [`stream_body_with`] for a strict stream whose traffic capture applies
/// [`RedactedWiring::redact`] to every string of every event it records,
/// the downstream event files included.
pub fn stream_body_redacted<S, E>(
    upstream: S,
    message_id: String,
    model: String,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
    wiring: RedactedWiring,
) -> Body
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let mut stream = wired_stream(
        upstream,
        message_id,
        model,
        monitor,
        req_id,
        traffic,
        wiring.wiring.clone(),
    );
    stream.context = ReceiveContext::redacted(wiring);
    wired_body(stream)
}

fn wired_body(stream: WiredResponsesStream) -> Body {
    let stream = futures_util::stream::unfold(stream, |mut stream| async move {
        let output = stream
            .state
            .next_output_with(&mut stream.context, &mut stream.strict)
            .await;
        if stream.context.is_strict() && stream.state.terminal {
            // A settled strict stream reads nothing more, so the upstream
            // connection is let go now rather than with the body.
            stream.state.upstream = Box::pin(futures_util::stream::empty());
        }
        output.map(|bytes| (Ok::<Bytes, Infallible>(Bytes::from(bytes)), stream))
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
) -> WiredResponsesStream
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
{
    let upstream = box_upstream(upstream);
    let state = ResponsesStreamState {
        upstream,
        decoder: SseDecoder::default(),
        translator: LiveStreamTranslator::new(message_id, model)
            .with_incomplete_response_policy(IncompleteResponsePolicy::AllowMaxOutputTokens),
        pending_completion: None,
        done_seen: false,
        terminal: false,
        error_sent: false,
        monitor,
        req_id,
        bytes: 0,
        chunks: 0,
        stream_capture: traffic.as_ref().map(|traffic| traffic.stream_capture()),
        traffic,
    };
    WiredResponsesStream {
        state,
        context: ReceiveContext::new(wiring),
        strict: StrictResponses::default(),
    }
}

/// One stream and its wiring. Its `Drop` runs before the state's, so an
/// abandoned stream is summarised under the wiring's labels.
struct WiredResponsesStream {
    state: ResponsesStreamState<UpstreamStream>,
    context: ReceiveContext,
    strict: StrictResponses,
}

/// What a strict stream remembers besides its receive context.
#[derive(Debug, Default)]
struct StrictResponses {
    deltas: DeltasSeen,
    /// The client has received a `message_start`.
    message_started: bool,
    /// The blocks the client has received a `content_block_start` for and
    /// no `content_block_stop` yet, by index.
    open_blocks: BTreeSet<usize>,
}

impl StrictResponses {
    /// Notes which message and block events of `output` the client now
    /// has. Called with every output a strict stream releases.
    fn note_released(&mut self, output: &[u8]) {
        for event in generated_events(output) {
            match event.event.as_deref() {
                Some("message_start") => self.message_started = true,
                Some("content_block_start") => {
                    if let Some(index) = block_index(&event) {
                        self.open_blocks.insert(index);
                    }
                }
                Some("content_block_stop") => {
                    if let Some(index) = block_index(&event) {
                        self.open_blocks.remove(&index);
                    }
                }
                _ => {}
            }
        }
    }

    /// The events that end a strict stream with an error after the
    /// translator's output for the last chunk, `discarded`, was withheld
    /// from the client. The translator counts what that output started as
    /// sent, so its closing events, `closing`, may stop a block the client
    /// never saw start, or presuppose a `message_start` the client never
    /// got. The result is a sequence the client can read: the
    /// `message_start` from `discarded` when the client has none yet, the
    /// events of `closing` except its block events, and a
    /// `content_block_stop` for each block the client still has open, in
    /// index order, before the final `error` event. The translator closes
    /// its blocks in no fixed order, so its own stops are not used.
    fn error_sequence(&self, closing: &[u8], discarded: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.message_started
            && let Some(start) = generated_events(discarded)
                .into_iter()
                .find(|event| event.event.as_deref() == Some("message_start"))
        {
            out.extend(encode_sse_event(Some("message_start"), &start.data));
        }
        for event in generated_events(closing) {
            let name = event.event.as_deref();
            match name {
                Some("content_block_start" | "content_block_delta" | "content_block_stop") => {
                    continue;
                }
                Some("error") => {
                    for index in &self.open_blocks {
                        out.extend(encode_sse_event(
                            Some("content_block_stop"),
                            &serde_json::json!({"type": "content_block_stop", "index": index})
                                .to_string(),
                        ));
                    }
                }
                _ => {}
            }
            out.extend(encode_sse_event(name, &event.data));
        }
        out
    }
}

/// The events of output the translator generated. Its frames are not held
/// to the upstream frame limit: a delta just under that limit becomes a
/// larger Anthropic frame, and the output is already bounded by what the
/// upstream decoder accepted.
fn generated_events(output: &[u8]) -> Vec<SseEvent> {
    SseDecoder::with_max_frame_bytes(usize::MAX)
        .push(output)
        .unwrap_or_default()
}

/// The `index` of an Anthropic block event.
fn block_index(event: &SseEvent) -> Option<usize> {
    serde_json::from_str::<serde_json::Value>(&event.data)
        .ok()?
        .get("index")?
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
}

impl Drop for WiredResponsesStream {
    fn drop(&mut self) {
        self.state.abandon(self.context.labels());
    }
}

struct ResponsesStreamState<S> {
    upstream: S,
    decoder: SseDecoder,
    translator: LiveStreamTranslator,
    pending_completion: Option<serde_json::Value>,
    done_seen: bool,
    terminal: bool,
    error_sent: bool,
    monitor: Option<MonitorHandle>,
    req_id: String,
    bytes: u64,
    chunks: u64,
    stream_capture: Option<StreamTrafficCapture>,
    traffic: Option<Arc<TrafficCapture>>,
}

impl<S> ResponsesStreamState<S>
where
    S: futures_util::Stream<Item = Result<Bytes, OpenCodeError>> + Unpin,
{
    #[cfg(test)]
    async fn next_output(&mut self) -> Option<Vec<u8>> {
        self.next_output_with(
            &mut ReceiveContext::default(),
            &mut StrictResponses::default(),
        )
        .await
    }

    async fn next_output_with(
        &mut self,
        context: &mut ReceiveContext,
        strict: &mut StrictResponses,
    ) -> Option<Vec<u8>> {
        if self.terminal {
            return None;
        }
        if self.error_sent {
            self.terminal = true;
            return None;
        }

        loop {
            let after_terminal = context.is_strict() && self.pending_completion.is_some();
            let next = if after_terminal {
                match context.read_after_terminal(&mut self.upstream).await {
                    Some(next) => next,
                    None => return self.complete(context),
                }
            } else {
                self.upstream.next().await
            };
            let chunk = match next {
                Some(Ok(chunk)) => chunk,
                Some(Err(_)) => return Some(self.fail_at("transport", "upstream_stream", context)),
                None => {
                    if self.decoder.finish().is_err() {
                        return Some(self.fail_at("decoder", "incomplete_stream", context));
                    }
                    return self.complete(context);
                }
            };

            if self.bytes == 0
                && let Some(monitor) = self.monitor.as_ref()
            {
                monitor.generation_started(&self.req_id);
            }
            self.bytes = self.bytes.saturating_add(chunk.len() as u64);
            self.chunks = self.chunks.saturating_add(1);
            if context.is_strict() {
                let mut output = Vec::new();
                match self.strict_chunk(&chunk, context, strict, &mut output) {
                    Ok(()) if self.translator.is_finished() => {
                        self.terminal = true;
                        self.write_released_events(&output, context);
                        self.record_progress(&output);
                        self.capture_downstream(&output, context);
                        self.finish_capture(true, context);
                        return Some(output);
                    }
                    Ok(()) if output.is_empty() => continue,
                    Ok(()) => {
                        strict.note_released(&output);
                        self.write_released_events(&output, context);
                        self.record_progress(&output);
                        self.capture_downstream(&output, context);
                        return Some(output);
                    }
                    Err((stage, kind)) => {
                        return Some(self.fail_discarding(stage, kind, context, strict, &output));
                    }
                }
            }

            let events = match self.decoder.push(&chunk) {
                Ok(events) => events,
                Err(_) => return Some(self.fail_at("decoder", "malformed_sse", context)),
            };
            let mut output = Vec::new();
            for event in events {
                let data = event.data.trim();
                if data == "[DONE]" {
                    if self.pending_completion.is_none() || self.done_seen {
                        return Some(self.fail_at("protocol", "premature_done", context));
                    }
                    self.done_seen = true;
                    if let Some(capture) = self.stream_capture.as_mut() {
                        capture
                            .upstream_event(event.event.as_deref(), &serde_json::json!("[DONE]"));
                    }
                    continue;
                }
                let value: serde_json::Value = match serde_json::from_str(data) {
                    Ok(value) => value,
                    Err(_) => return Some(self.fail_at("json", "malformed_event", context)),
                };
                if let Some(capture) = self.stream_capture.as_mut() {
                    capture.upstream_event(
                        event.event.as_deref(),
                        &context.wiring.receive.upstream_value(&value),
                    );
                }
                let event_type = value.get("type").and_then(serde_json::Value::as_str);
                if self.pending_completion.is_some() {
                    if event_type == Some("ping") {
                        continue;
                    }
                    return Some(self.fail_at("protocol", "event_after_completion", context));
                }
                if matches!(
                    event_type,
                    Some("response.completed" | "response.incomplete" | "response.done")
                ) {
                    // Keep the translator open until EOF so invalid tails can emit an error.
                    self.pending_completion = Some(value);
                    continue;
                }
                let translated = match self.translator.accept(&value, self.traffic.as_deref()) {
                    Ok(translated) => translated,
                    Err(_) => return Some(self.fail_at("translation", "invalid_event", context)),
                };
                output.extend(translated);
            }
            if !output.is_empty() {
                self.record_progress(&output);
                self.capture_downstream(&output, context);
                return Some(output);
            }
        }
    }

    /// Handles one chunk of a strict stream a frame at a time. The bytes after
    /// the completion's frame are charged to
    /// [`super::chat::POST_TERMINAL_MAX_BYTES`] by position, so a tail counts
    /// the same in the completion's chunk as in later ones. When the
    /// translator closes the message by itself, the rest of the chunk is
    /// left unread and the output ends with that closing. The translator's
    /// output is added to `output`; on an error it holds what the chunk
    /// produced before the rejected frame, which the client never gets.
    fn strict_chunk(
        &mut self,
        chunk: &[u8],
        context: &mut ReceiveContext,
        strict: &mut StrictResponses,
        output: &mut Vec<u8>,
    ) -> Result<(), (&'static str, &'static str)> {
        let RawFrameEnds { continued, ends } = context.raw_frame_ends(chunk);
        let mut terminal = self.pending_completion.is_some();
        if terminal && context.charge_after_terminal(chunk, continued) {
            return Err(("protocol", "post_terminal_limit"));
        }
        let mut start = 0;
        for end in ends.into_iter().chain([chunk.len()]) {
            if end == start {
                continue;
            }
            let events = self
                .decoder
                .push(&chunk[start..end])
                .map_err(|_| ("decoder", "malformed_sse"))?;
            start = end;
            for event in events {
                self.strict_event(&event, context, strict, output)?;
            }
            if self.translator.is_finished() {
                return Ok(());
            }
            if !terminal && self.pending_completion.is_some() {
                terminal = true;
                context.start_read_window();
                if context.terminal_frame_ended(chunk, end) {
                    return Err(("protocol", "post_terminal_limit"));
                }
            }
        }
        Ok(())
    }

    /// One frame of a strict stream: a `type` from neither list is an error,
    /// so is a snapshot whose text no delta delivered, and after the
    /// completion only one `[DONE]` is accepted.
    fn strict_event(
        &mut self,
        event: &SseEvent,
        context: &ReceiveContext,
        strict: &mut StrictResponses,
        output: &mut Vec<u8>,
    ) -> Result<(), (&'static str, &'static str)> {
        let data = event.data.trim();
        if data == "[DONE]" {
            if self.pending_completion.is_none() || self.done_seen {
                return Err(("protocol", "premature_done"));
            }
            self.done_seen = true;
            if let Some(capture) = self.stream_capture.as_mut() {
                capture.upstream_event(
                    context
                        .captured_event_name(event.event.as_deref())
                        .as_deref(),
                    &serde_json::json!("[DONE]"),
                );
            }
            return Ok(());
        }
        let value: serde_json::Value =
            serde_json::from_str(data).map_err(|_| ("json", "malformed_event"))?;
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.upstream_event(
                context
                    .captured_event_name(event.event.as_deref())
                    .as_deref(),
                &context.captured_upstream(&value),
            );
        }
        if self.pending_completion.is_some() {
            return Err(("protocol", "event_after_completion"));
        }
        if let Some(kind) = unlisted_frame(&value).or_else(|| strict.deltas.check(&value)) {
            return Err(("protocol", kind));
        }
        let event_type = value.get("type").and_then(serde_json::Value::as_str);
        if event_type.is_some_and(|event_type| TERMINAL_EVENT_TYPES.contains(&event_type)) {
            self.pending_completion = Some(value);
            return Ok(());
        }
        // Nothing is written to the traffic capture here: a later frame of
        // the same chunk can still discard this output, so its downstream
        // events are written when it is released.
        let translated = self
            .translator
            .accept(&value, None)
            .map_err(|_| ("translation", "invalid_event"))?;
        output.extend(translated);
        Ok(())
    }

    /// Writes the downstream event files for output a strict stream
    /// releases, in the form the translator writes them on the default
    /// path.
    fn write_released_events(&self, output: &[u8], context: &ReceiveContext) {
        let Some(traffic) = self.traffic.as_ref() else {
            return;
        };
        let Ok(events) = SseDecoder::default().push(output) else {
            return;
        };
        for event in events {
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&event.data) {
                traffic.write_json_event(
                    "050-downstream-event",
                    &serde_json::json!({
                        "event": event.event.as_deref().unwrap_or("message"),
                        "data": context.captured_downstream(data),
                    }),
                );
            }
        }
    }

    /// Translates the held completion once the upstream closed or, for a
    /// strict stream, the read window after it ended.
    fn complete(&mut self, context: &ReceiveContext) -> Option<Vec<u8>> {
        let Some(completion) = self.pending_completion.take() else {
            return Some(self.fail_at("decoder", "incomplete_stream", context));
        };
        // With a redactor the translator writes no event file; the files
        // are written here, redacted.
        let traffic = if context.redacts() {
            None
        } else {
            self.traffic.as_deref()
        };
        let output = match self.translator.accept(&completion, traffic) {
            Ok(output) => output,
            Err(_) => return Some(self.fail_at("translation", "invalid_event", context)),
        };
        if context.redacts() {
            self.write_released_events(&output, context);
        }
        self.terminal = true;
        self.record_progress(&output);
        self.capture_downstream(&output, context);
        self.finish_capture(true, context);
        (!output.is_empty()).then_some(output)
    }

    fn record_progress(&self, output: &[u8]) {
        let Some(monitor) = self.monitor.as_ref() else {
            return;
        };
        let (input_tokens, output_tokens) = usage_from_anthropic_sse(output);
        monitor.stream_progress(
            &self.req_id,
            output.len() as u64,
            count_sse_events(output),
            input_tokens,
            output_tokens,
        );
    }

    fn fail_at(&mut self, stage: &str, kind: &str, context: &ReceiveContext) -> Vec<u8> {
        let labels = context.labels();
        self.error_sent = true;
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.malformed(stage, kind);
        }
        if let Some(traffic) = self.traffic.as_ref() {
            traffic.write_json(
                &format!("060-{}-responses-stream-error", labels.capture_stem),
                &serde_json::json!({
                    "stage": stage,
                    "kind": kind,
                    "bytes": self.bytes,
                    "chunks": self.chunks,
                }),
            );
        }
        let output = self.translator.error_chunk(
            &format!("{} Responses stream is invalid", labels.provider),
            "api_error",
            self.traffic.as_deref(),
        );
        self.capture_downstream(&output, context);
        self.finish_capture(false, context);
        output
    }

    /// [`Self::fail_at`] for a strict stream whose last chunk was rejected
    /// after the translator had produced output for its earlier frames,
    /// `discarded`, which the client never gets. The client receives
    /// [`StrictResponses::error_sequence`] instead of the translator's
    /// error events, and the downstream event files are written for those
    /// events.
    fn fail_discarding(
        &mut self,
        stage: &str,
        kind: &str,
        context: &ReceiveContext,
        strict: &StrictResponses,
        discarded: &[u8],
    ) -> Vec<u8> {
        let labels = context.labels();
        self.error_sent = true;
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.malformed(stage, kind);
        }
        if let Some(traffic) = self.traffic.as_ref() {
            traffic.write_json(
                &format!("060-{}-responses-stream-error", labels.capture_stem),
                &serde_json::json!({
                    "stage": stage,
                    "kind": kind,
                    "bytes": self.bytes,
                    "chunks": self.chunks,
                }),
            );
        }
        let closing = self.translator.error_chunk(
            &format!("{} Responses stream is invalid", labels.provider),
            "api_error",
            None,
        );
        let output = strict.error_sequence(&closing, discarded);
        self.write_released_events(&output, context);
        self.capture_downstream(&output, context);
        self.finish_capture(false, context);
        output
    }

    fn capture_downstream(&mut self, bytes: &[u8], context: &ReceiveContext) {
        let Some(capture) = self.stream_capture.as_mut() else {
            return;
        };
        let mut decoder = SseDecoder::default();
        if let Ok(events) = decoder.push(bytes) {
            for event in events {
                if let Ok(value) = serde_json::from_str(&event.data) {
                    capture.downstream_event(
                        event.event.as_deref().unwrap_or("message"),
                        context.captured_downstream(value),
                    );
                }
            }
        }
    }

    fn finish_capture(&mut self, completed: bool, context: &ReceiveContext) {
        if let (Some(capture), Some(traffic)) = (self.stream_capture.take(), self.traffic.as_ref())
        {
            capture.finish_named(
                traffic,
                serde_json::json!({
                    "kind": if completed { "stream_completion" } else { "stream_error" },
                    "bytes": self.bytes,
                    "chunks": self.chunks,
                }),
                &summary_name(context.labels()),
            );
        }
    }
}

fn summary_name(labels: &StreamLabels) -> String {
    format!("061-{}-responses-stream-summary", labels.capture_stem)
}

impl<S> ResponsesStreamState<S> {
    fn abandon(&mut self, labels: &StreamLabels) {
        if self.terminal || self.stream_capture.is_none() {
            return;
        }
        if let (Some(capture), Some(traffic)) = (self.stream_capture.take(), self.traffic.as_ref())
        {
            capture.finish_named(
                traffic,
                serde_json::json!({
                    "kind": "stream_abandoned",
                    "reason": "downstream_body_dropped",
                    "bytes": self.bytes,
                    "chunks": self.chunks,
                }),
                &summary_name(labels),
            );
        }
    }
}

impl<S> Drop for ResponsesStreamState<S> {
    fn drop(&mut self) {
        self.abandon(&StreamLabels::OPENCODE_GO);
    }
}

fn count_sse_events(bytes: &[u8]) -> u64 {
    String::from_utf8_lossy(bytes).matches("event:").count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::grok::translate::stream::MAX_SSE_FRAME_BYTES;
    use serde_json::json;

    const COMPLETION: &str = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    const PING: &str = "event: ping\ndata: {\"type\":\"ping\",\"cost\":\"0\"}\n\n";

    fn test_state<S>(upstream: S) -> ResponsesStreamState<S> {
        ResponsesStreamState {
            upstream,
            decoder: SseDecoder::default(),
            translator: LiveStreamTranslator::new("msg_1", "gpt-5.6-luna"),
            pending_completion: None,
            done_seen: false,
            terminal: false,
            error_sent: false,
            monitor: None,
            req_id: "req".into(),
            bytes: 0,
            chunks: 0,
            stream_capture: None,
            traffic: None,
        }
    }

    #[tokio::test]
    async fn completion_tail_is_independent_of_chunk_boundaries() {
        let wire = format!("{COMPLETION}{PING}data: [DONE]\n\n{PING}");
        for split in 0..=wire.len() {
            let upstream = futures_util::stream::iter([
                Ok(Bytes::copy_from_slice(&wire.as_bytes()[..split])),
                Ok(Bytes::copy_from_slice(&wire.as_bytes()[split..])),
            ]);
            let mut state = test_state(upstream);
            let output = state.next_output().await.unwrap();
            let text = String::from_utf8_lossy(&output);
            assert!(text.contains("message_stop"), "split {split}: {text}");
            assert!(!text.contains("api_error"), "split {split}: {text}");
            assert_eq!(state.chunks, 2);
            assert!(state.next_output().await.is_none());
        }
    }

    #[tokio::test]
    async fn invalid_tails_are_rejected_across_chunks() {
        for tail in [
            COMPLETION,
            "data: {\"type\":\"response.done\"}\n\n",
            "data: {\"type\":\"response.incomplete\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"late\"}\n\n",
            "data: not-json\n\n",
            "event: ping\ndata: {\"",
            "data: [DONE]\n\ndata: [DONE]\n\n",
            "data: [DONE]\n\ndata: {\"type\":\"response.completed\"}\n\n",
        ] {
            let wire = format!("{COMPLETION}{tail}");
            for split in 0..=wire.len() {
                let mut state = test_state(futures_util::stream::iter([
                    Ok(Bytes::copy_from_slice(&wire.as_bytes()[..split])),
                    Ok(Bytes::copy_from_slice(&wire.as_bytes()[split..])),
                ]));
                let output = state.next_output().await.unwrap();
                let text = String::from_utf8_lossy(&output);
                assert!(
                    text.contains("api_error"),
                    "tail {tail}, split {split}: {text}"
                );
                assert!(!text.contains("message_stop"));
                assert!(state.next_output().await.is_none());
            }
        }
    }

    #[tokio::test]
    async fn rejects_premature_done_and_transport_error_after_completion() {
        for chunks in [
            vec![Ok(Bytes::from_static(b"data: [DONE]\n\n"))],
            vec![
                Ok(Bytes::from_static(COMPLETION.as_bytes())),
                Err(OpenCodeError {
                    status: http::StatusCode::BAD_GATEWAY,
                    retry_after: None,
                    message: "disconnected".into(),
                }),
            ],
        ] {
            let mut state = test_state(futures_util::stream::iter(chunks));
            let output = state.next_output().await.unwrap();
            assert!(String::from_utf8_lossy(&output).contains("api_error"));
            assert!(state.next_output().await.is_none());
        }
    }

    #[tokio::test]
    async fn streams_text_before_completion_and_waits_for_eof_to_finish() {
        let (sender, receiver) = tokio::sync::mpsc::channel(4);
        let upstream = futures_util::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|chunk| (chunk, receiver))
        });
        let mut state = test_state(Box::pin(upstream));
        sender.send(Ok(Bytes::from_static(
            b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"early\"}\n\n",
        ))).await.unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(1), state.next_output())
            .await
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&output).contains("early"));
        sender
            .send(Ok(Bytes::from_static(COMPLETION.as_bytes())))
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), state.next_output())
                .await
                .is_err()
        );
        sender
            .send(Ok(Bytes::from_static(PING.as_bytes())))
            .await
            .unwrap();
        drop(sender);
        let output = state.next_output().await.unwrap();
        assert!(String::from_utf8_lossy(&output).contains("message_stop"));
        assert!(state.next_output().await.is_none());
    }

    #[test]
    fn request_uses_standard_responses_fields_without_codex_lane_metadata() {
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "opencode-go/gpt-5.6-luna",
            "max_tokens": 2048,
            "output_config": {"effort": "xhigh"},
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let translated = prepare_request(&body, "gpt-5.6-luna", Some("session-1".into()))
            .expect("translate Responses request");

        assert_eq!(translated["model"], "gpt-5.6-luna");
        assert_eq!(translated["stream"], true);
        assert_eq!(translated["max_output_tokens"], 2048);
        assert_eq!(translated["reasoning"]["effort"], "xhigh");
        assert_eq!(translated["reasoning"]["summary"], "auto");
        assert_eq!(translated["prompt_cache_key"], "session-1");
        assert!(translated.get("service_tier").is_none());
        assert!(translated.get("client_metadata").is_none());
    }

    #[tokio::test]
    async fn live_stream_rejects_an_incomplete_frame_after_completion() {
        let upstream = futures_util::stream::iter([Ok::<Bytes, OpenCodeError>(
            Bytes::from_static(
                b"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\ndata: {",
            ),
        )]);
        let mut state = ResponsesStreamState {
            upstream,
            decoder: SseDecoder::default(),
            translator: LiveStreamTranslator::new("msg_1", "gpt-5.6-luna"),
            pending_completion: None,
            done_seen: false,
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
        assert!(
            String::from_utf8_lossy(&output).contains("OpenCode Go Responses stream is invalid")
        );
    }

    #[tokio::test]
    async fn live_stream_accepts_ping_after_completion() {
        let upstream = futures_util::stream::iter([Ok::<Bytes, OpenCodeError>(
            Bytes::from_static(
                b"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\nevent: ping\ndata: {\"type\":\"ping\",\"cost\":\"0\"}\n\n",
            ),
        )]);
        let mut state = ResponsesStreamState {
            upstream,
            decoder: SseDecoder::default(),
            translator: LiveStreamTranslator::new("msg_1", "grok-4.6"),
            pending_completion: None,
            done_seen: false,
            terminal: false,
            error_sent: false,
            monitor: None,
            req_id: "req".into(),
            bytes: 0,
            chunks: 0,
            stream_capture: None,
            traffic: None,
        };

        let output = state.next_output().await.expect("completion event");
        let text = String::from_utf8_lossy(&output);
        assert!(text.contains("message_stop"));
        assert!(!text.contains("OpenCode Go Responses stream is invalid"));
    }

    use crate::providers::opencode::chat::{POST_TERMINAL_READ_LIMIT, ReceivePolicy, Sanitizer};

    const INVALID: &str = "Synthetic Gateway Responses stream is invalid";

    fn strict_wiring() -> StreamWiring {
        StreamWiring {
            labels: StreamLabels {
                provider: "Synthetic Gateway",
                capture_stem: "synthetic",
            },
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

    #[test]
    fn listed_event_types_are_distinct_and_cover_every_terminal_event() {
        let mut all = RESPONSES_HANDLED_EVENT_TYPES
            .iter()
            .chain(RESPONSES_HARMLESS_EVENT_TYPES)
            .collect::<Vec<_>>();
        let listed = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), listed);
        for terminal in TERMINAL_EVENT_TYPES {
            assert!(RESPONSES_HANDLED_EVENT_TYPES.contains(&terminal));
        }
        assert!(!all.contains(&&"ping"));
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_unreadable_or_unlisted_frames() {
        for frame in [
            "data: not-json\n\n",
            "data: {\"delta\":\"no type\"}\n\n",
            "data: {\"type\":7}\n\n",
            "data: {\"type\":\"response.refusal.delta\",\"delta\":\"no\"}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        ] {
            let upstream = futures_util::stream::iter(chunks(&[frame, COMPLETION]));
            let text = body_text(strict_body(upstream)).await;
            assert!(text.contains(INVALID), "frame {frame}: {text}");
            assert!(!text.contains("message_stop"), "frame {frame}: {text}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_passes_harmless_progress_events() {
        let wire = concat!(
            "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
            "data: {\"type\":\"response.in_progress\",\"response\":{}}\n\n",
            "data: {\"type\":\"response.queued\"}\n\n",
            "data: {\"type\":\"response.content_part.added\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"early\"}\n\n",
            "data: {\"type\":\"response.output_text.done\",\"output_index\":0,\"text\":\"early\"}\n\n",
            "data: {\"type\":\"response.content_part.done\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "data: [DONE]\n\n"
        );
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(futures_util::stream::iter(chunks(&[wire])))).await;
        assert!(started.elapsed() < POST_TERMINAL_READ_LIMIT);
        assert!(text.contains("early"), "{text}");
        assert!(text.contains("message_stop"), "{text}");
        assert!(!text.contains("api_error"), "{text}");
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_any_tail_after_a_valid_completion() {
        for tail in [
            "data: not-json\n\n",
            PING,
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"late\"}\n\n",
            "data: [DONE]\n\ndata: [DONE]\n\n",
        ] {
            let wire = format!("{COMPLETION}{tail}");
            for split in [COMPLETION.len(), wire.len()] {
                let upstream =
                    futures_util::stream::iter(chunks(&[&wire[..split], &wire[split..]]));
                let text = body_text(strict_body(upstream)).await;
                assert!(text.contains(INVALID), "tail {tail}, split {split}: {text}");
                assert!(
                    !text.contains("message_stop"),
                    "tail {tail}, split {split}: {text}"
                );
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_completes_at_the_read_limit_when_upstream_stays_open() {
        for wire in [
            COMPLETION.to_string(),
            format!("{COMPLETION}data: [DONE]\n\n"),
            format!("{COMPLETION}data: {{\"ty"),
        ] {
            let upstream =
                futures_util::stream::iter(chunks(&[&wire])).chain(futures_util::stream::pending());
            let started = tokio::time::Instant::now();
            let text = body_text(strict_body(upstream)).await;
            assert!(started.elapsed() >= POST_TERMINAL_READ_LIMIT, "{wire}");
            assert!(text.contains("message_stop"), "{wire}: {text}");
            assert!(!text.contains("api_error"), "{wire}: {text}");
        }
    }

    #[test]
    fn synthetic_strict_buffered_response_checks_every_frame() {
        let wiring = strict_wiring();
        for wire in [
            format!("data: not-json\n\n{COMPLETION}"),
            format!("data: {{\"delta\":\"no type\"}}\n\n{COMPLETION}"),
            format!("data: {{\"type\":\"response.refusal.delta\"}}\n\n{COMPLETION}"),
            format!("{COMPLETION}{PING}"),
            format!("{COMPLETION}data: [DONE]\n\ndata: [DONE]\n\n"),
            format!("data: [DONE]\n\n{COMPLETION}"),
            format!("{COMPLETION}data: {{\"ty"),
        ] {
            let error = accumulate_response_with(wire.as_bytes(), "msg", "model", &wiring)
                .unwrap_err()
                .to_string();
            assert!(error.starts_with(INVALID), "{wire}: {error}");
        }
        let wire = format!(
            "data: {{\"type\":\"response.created\"}}\n\ndata: {{\"type\":\"response.content_part.added\"}}\n\n{COMPLETION}data: [DONE]\n\n"
        );
        assert!(accumulate_response_with(wire.as_bytes(), "msg", "model", &wiring).is_ok());
        let wire = format!("{COMPLETION}{PING}");
        assert!(accumulate_response(wire.as_bytes(), "msg", "model").is_ok());
    }

    #[test]
    fn synthetic_sanitizer_cleans_the_upstream_failure_of_a_buffered_response() {
        let sanitize: Sanitizer = Arc::new(|text: &str| text.replace("SECRET-123", "[redacted]"));
        let mut wiring = strict_wiring();
        wiring.receive.sanitize = Some(sanitize);
        let wire = concat!(
            "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
            "\"error\":{\"code\":\"server_error\",\"message\":\"model saw SECRET-123\"}}}\n\n"
        );
        let error = accumulate_response_with(wire.as_bytes(), "msg", "model", &wiring)
            .unwrap_err()
            .to_string();
        assert!(error.contains("[redacted]"), "{error}");
        assert!(!error.contains("SECRET-123"), "{error}");
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
    async fn synthetic_strict_stream_charges_the_tail_after_the_completion_by_position() {
        let at_limit = synthetic_comment_tail_of_sixteen_mib();
        let over_limit = format!("{at_limit}\n");
        for (tail, accepted) in [(at_limit.as_str(), true), (over_limit.as_str(), false)] {
            let wire = format!("{COMPLETION}{tail}");
            let terminal_end = COMPLETION.len();
            let inside_tail = terminal_end + 4096 + 7;
            for parts in [
                vec![wire.as_str()],
                vec![&wire[..terminal_end], &wire[terminal_end..]],
                vec![&wire[..inside_tail], &wire[inside_tail..]],
            ] {
                let upstream = futures_util::stream::iter(chunks(&parts));
                let text = body_text(strict_body(upstream)).await;
                let case = format!("{} tail bytes in {} items", tail.len(), parts.len());
                assert_eq!(text.contains("message_stop"), accepted, "{case}");
                assert_eq!(text.contains(INVALID), !accepted, "{case}");
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
    async fn synthetic_strict_stream_waits_exactly_two_seconds_after_the_completion() {
        let upstream = futures_util::stream::iter(chunks(&[COMPLETION]))
            .chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(upstream)).await;
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(2));
        assert!(text.contains("message_stop"), "{text}");
        assert!(!text.contains("api_error"), "{text}");
    }

    const READ_CALL_ADDED: &str = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"Read\"}}\n\n";

    /// A `Read` argument delta that ends in 1,024 spaces, which the Codex
    /// translator repairs into a finished tool call with its own
    /// `message_delta` and `message_stop`.
    fn synthetic_stalled_read_delta() -> String {
        let delta = format!("{{\"file_path\":\"/tmp/a\"}}{}", " ".repeat(1024));
        let frame = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": delta,
        });
        format!("data: {frame}\n\n")
    }

    /// The events of a client body, with the error event last when there is
    /// one.
    fn events_of(text: &str) -> Vec<(Option<String>, serde_json::Value)> {
        let mut decoder = SseDecoder::with_max_frame_bytes(usize::MAX);
        let events = decoder.push(text.as_bytes()).unwrap();
        assert!(decoder.finish().is_ok(), "{text}");
        events
            .into_iter()
            .map(|event| {
                let value = serde_json::from_str(&event.data)
                    .unwrap_or_else(|_| panic!("unparseable event in {text}"));
                (event.event, value)
            })
            .collect()
    }

    /// `items`, then `forever` without end when it is set (otherwise an
    /// upstream that stays open), and the count of items handed out.
    fn synthetic_counted_upstream(
        items: Vec<Result<Bytes, std::io::Error>>,
        forever: Option<&'static str>,
    ) -> (
        impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&pulled);
        let tail: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
            match forever {
                Some(frame) => Box::pin(futures_util::stream::repeat_with(move || {
                    Ok(Bytes::from_static(frame.as_bytes()))
                })),
                None => Box::pin(futures_util::stream::pending()),
            };
        let upstream = futures_util::stream::iter(items)
            .chain(tail)
            .inspect(move |_| {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            });
        (upstream, pulled)
    }

    /// The body ends with the repaired `Read` call, delivered once, and no
    /// error.
    fn assert_repaired_read_delivered_once(text: &str, case: &str) {
        let events = events_of(text);
        assert!(
            events
                .iter()
                .all(|(name, _)| name.as_deref() != Some("error")),
            "{case}: {text}"
        );
        assert_eq!(
            text.matches(r#""partial_json":"{\"file_path\":\"/tmp/a\"}""#)
                .count(),
            1,
            "{case}: {text}"
        );
        assert_eq!(
            text.matches(r#""stop_reason":"tool_use""#).count(),
            1,
            "{case}: {text}"
        );
        assert_eq!(
            text.matches("event: message_stop").count(),
            1,
            "{case}: {text}"
        );
        assert!(
            text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            "{case}: {text}"
        );
    }

    const WHITESPACE_DELTA: &str = "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"    \"}\n\n";

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_settles_at_a_repaired_read_call_while_the_upstream_keeps_sending()
     {
        let stalled = synthetic_stalled_read_delta();
        let repaired = format!("{READ_CALL_ADDED}{stalled}");
        for (case, parts) in [
            ("repair in one item", vec![repaired.as_str()]),
            (
                "repair over two items",
                vec![READ_CALL_ADDED, stalled.as_str()],
            ),
        ] {
            let items = chunks(&parts);
            let (upstream, pulled) = synthetic_counted_upstream(items, Some(WHITESPACE_DELTA));
            let started = tokio::time::Instant::now();
            let text = body_text(strict_body(upstream)).await;
            assert_eq!(started.elapsed(), std::time::Duration::ZERO, "{case}");
            assert_repaired_read_delivered_once(&text, case);
            assert_eq!(
                pulled.load(std::sync::atomic::Ordering::SeqCst),
                parts.len(),
                "{case}: the upstream was read after the repair"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_ignores_a_completion_and_late_delta_after_a_repaired_read_call()
     {
        let repaired = format!("{READ_CALL_ADDED}{}", synthetic_stalled_read_delta());
        let late = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"late\"}\n\n";
        let tail = format!("{COMPLETION}{late}");
        let wire = format!("{repaired}{tail}");
        for (case, parts) in [
            ("tail in the repair's item", vec![wire.as_str()]),
            (
                "tail in a later item",
                vec![repaired.as_str(), tail.as_str()],
            ),
        ] {
            let (upstream, pulled) = synthetic_counted_upstream(chunks(&parts), None);
            let started = tokio::time::Instant::now();
            let text = body_text(strict_body(upstream)).await;
            assert_eq!(started.elapsed(), std::time::Duration::ZERO, "{case}");
            assert_repaired_read_delivered_once(&text, case);
            assert!(!text.contains("late"), "{case}: {text}");
            assert_eq!(
                pulled.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "{case}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_ignores_a_transport_error_after_a_repaired_read_call() {
        let wire = format!("{READ_CALL_ADDED}{}", synthetic_stalled_read_delta());
        let items = vec![
            Ok(Bytes::from(wire)),
            Err(std::io::Error::other("connection reset")),
        ];
        let (upstream, pulled) = synthetic_counted_upstream(items, None);
        let text = body_text(strict_body(upstream)).await;
        assert_repaired_read_delivered_once(&text, "transport error after the repair");
        assert_eq!(pulled.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn synthetic_strict_buffered_response_settles_at_a_repaired_read_call() {
        let repaired = format!("{READ_CALL_ADDED}{}", synthetic_stalled_read_delta());
        let late = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"late\"}\n\n";
        for (case, wire) in [
            (
                "whitespace keeps coming",
                format!("{repaired}{}", WHITESPACE_DELTA.repeat(3)),
            ),
            (
                "completion and a late delta",
                format!("{repaired}{COMPLETION}{late}"),
            ),
        ] {
            let strict =
                accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring())
                    .unwrap_or_else(|error| panic!("{case}: {error:#}"));
            assert_eq!(strict["stop_reason"], "tool_use", "{case}: {strict}");
            let tool_use = strict["content"]
                .as_array()
                .and_then(|blocks| blocks.iter().find(|block| block["type"] == "tool_use"))
                .unwrap_or_else(|| panic!("{case}: {strict}"));
            assert_eq!(
                tool_use["input"],
                serde_json::json!({"file_path": "/tmp/a"}),
                "{case}"
            );
            let default = accumulate_response(wire.as_bytes(), "msg", "model").unwrap();
            assert_eq!(strict, default, "{case}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_a_snapshot_whose_text_no_delta_delivered() {
        let text_delta = "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"seen\"}\n\n";
        let summary_delta = "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"seen\"}\n\n";
        let tests = [
            (
                "output text done",
                "",
                "{\"type\":\"response.output_text.done\",\"output_index\":0,\"text\":\"lost\"}",
            ),
            (
                "content part added",
                "",
                "{\"type\":\"response.content_part.added\",\"output_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"lost\"}}",
            ),
            (
                "content part done",
                "",
                "{\"type\":\"response.content_part.done\",\"output_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"lost\"}}",
            ),
            (
                "message item done",
                "",
                "{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"lost\"}]}}",
            ),
            (
                "summary text done",
                "",
                "{\"type\":\"response.reasoning_summary_text.done\",\"output_index\":0,\"text\":\"lost\"}",
            ),
            (
                "summary part done",
                "",
                "{\"type\":\"response.reasoning_summary_part.done\",\"output_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"lost\"}}",
            ),
            (
                "reasoning item done",
                "",
                "{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"lost\"}]}}",
            ),
            (
                "summary snapshot after a text delta only",
                text_delta,
                "{\"type\":\"response.reasoning_summary_text.done\",\"output_index\":0,\"text\":\"lost\"}",
            ),
            (
                "text snapshot after a summary delta only",
                summary_delta,
                "{\"type\":\"response.output_text.done\",\"output_index\":0,\"text\":\"lost\"}",
            ),
        ];
        for (name, before, snapshot) in tests {
            let wire = format!("{before}data: {snapshot}\n\n{COMPLETION}");
            let text = body_text(strict_body(futures_util::stream::iter(chunks(&[&wire])))).await;
            assert!(text.contains(INVALID), "{name}: {text}");
            assert!(!text.contains("message_stop"), "{name}: {text}");
            let error = accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring())
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                format!("{INVALID}: snapshot_without_delta"),
                "{name}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_accepts_snapshots_that_repeat_their_deltas() {
        let wire = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[]}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.added\",\"output_index\":0,\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"think\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.done\",\"output_index\":0,\"summary_index\":0,\"text\":\"think\"}\n\n",
            "data: {\"type\":\"response.reasoning_summary_part.done\",\"output_index\":0,\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"think\"}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"think\"}]}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n",
            "data: {\"type\":\"response.content_part.added\",\"output_index\":1,\"item_id\":\"msg_a\",\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"item_id\":\"msg_a\",\"delta\":\"hello\"}\n\n",
            "data: {\"type\":\"response.output_text.done\",\"output_index\":1,\"item_id\":\"msg_a\",\"text\":\"hello\"}\n\n",
            "data: {\"type\":\"response.content_part.done\",\"output_index\":1,\"item_id\":\"msg_a\",\"part\":{\"type\":\"output_text\",\"text\":\"hello\"}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_a\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
            "data: [DONE]\n\n"
        );
        let text = body_text(strict_body(futures_util::stream::iter(chunks(&[wire])))).await;
        assert!(text.contains("think"), "{text}");
        assert!(text.contains("hello"), "{text}");
        assert!(text.contains("message_stop"), "{text}");
        assert!(!text.contains("api_error"), "{text}");
        let wiring = strict_wiring();
        assert!(
            check_buffered_frames(wire.as_bytes(), &wiring.labels).is_ok(),
            "{wire}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_does_not_restart_the_read_window_at_done() {
        let upstream = futures_util::stream::unfold(
            vec![(0, COMPLETION), (1_500, "data: [DONE]\n\n")].into_iter(),
            |mut steps| async move {
                let (delay_ms, part) = steps.next()?;
                if delay_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
                Some((Ok(Bytes::from_static(part.as_bytes())), steps))
            },
        )
        .chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(upstream)).await;
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(2));
        assert!(text.contains("message_stop"), "{text}");
        assert!(!text.contains("api_error"), "{text}");
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_charges_no_byte_of_a_crlf_completion_frame_to_the_tail() {
        let completion = COMPLETION.replace('\n', "\r\n");
        let at_limit = synthetic_comment_tail_of_sixteen_mib();
        let over_limit = format!("{at_limit}\n");
        for (tail, accepted) in [(at_limit.as_str(), true), (over_limit.as_str(), false)] {
            let wire = format!("{completion}{tail}");
            let terminal_end = completion.len();
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
                assert_eq!(text.contains("message_stop"), accepted, "{case}");
                assert_eq!(text.contains(INVALID), !accepted, "{case}");
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
    async fn synthetic_strict_stream_starts_the_read_window_at_the_completion_not_at_the_next_poll()
    {
        let text_then_completion = format!(
            "{}{}{COMPLETION}",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"hello\"}\n\n",
        );
        let late = "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"late\"}\n\n";
        for (delay_ms, settles) in [(3_000, true), (1_000, false)] {
            let items = chunks(&[&text_then_completion, late]);
            let (upstream, pulled) = synthetic_counted_upstream(items, None);
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
            assert_eq!(rest.contains("message_stop"), settles, "{case}");
            assert_eq!(rest.contains(INVALID), !settles, "{case}");
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

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_a_snapshot_that_carries_a_refusal() {
        let message_added = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n";
        let tests = [
            (
                "content part added",
                "{\"type\":\"response.content_part.added\",\"output_index\":0,\"part\":{\"type\":\"refusal\",\"refusal\":\"I cannot help\"}}",
            ),
            (
                "content part done",
                "{\"type\":\"response.content_part.done\",\"output_index\":0,\"part\":{\"type\":\"refusal\",\"refusal\":\"I cannot help\"}}",
            ),
            (
                "message item done",
                "{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"I cannot help\"}]}}",
            ),
        ];
        for (name, snapshot) in tests {
            let wire = format!("{message_added}data: {snapshot}\n\n{COMPLETION}");
            let text = body_text(strict_body(futures_util::stream::iter(chunks(&[&wire])))).await;
            assert!(text.contains(INVALID), "{name}: {text}");
            assert!(!text.contains("message_stop"), "{name}: {text}");
            let error = accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring())
                .unwrap_err()
                .to_string();
            assert_eq!(error, format!("{INVALID}: refusal_in_snapshot"), "{name}");
        }

        // A refusal part that is still empty carries nothing to lose.
        let wire = format!(
            "{message_added}data: {}\n\n{COMPLETION}",
            "{\"type\":\"response.content_part.added\",\"output_index\":0,\"part\":{\"type\":\"refusal\",\"refusal\":\"\"}}"
        );
        let text = body_text(strict_body(futures_util::stream::iter(chunks(&[&wire])))).await;
        assert!(text.contains("message_stop"), "{text}");
        assert!(!text.contains(INVALID), "{text}");
        assert!(
            accumulate_response_with(wire.as_bytes(), "msg", "model", &strict_wiring()).is_ok(),
            "{wire}"
        );
    }

    /// The names of the downstream event files a traffic capture wrote
    /// under `root`, in order, and the text of each.
    fn synthetic_downstream_event_files(root: &std::path::Path) -> (Vec<String>, Vec<String>) {
        let mut paths = std::fs::read_dir(root.join("events"))
            .map(|dir| dir.map(|entry| entry.unwrap().path()).collect::<Vec<_>>())
            .unwrap_or_default();
        paths.sort();
        let mut names = Vec::new();
        let mut texts = Vec::new();
        for path in paths {
            if !path
                .to_string_lossy()
                .ends_with("-050-downstream-event.json")
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            names.push(value["event"].as_str().unwrap().to_string());
            texts.push(text);
        }
        (names, texts)
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_capture_writes_downstream_events_only_for_released_output() {
        let head = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"early\"}\n\n",
        );
        let rejected =
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"no\"}\n\n";
        for (case, parts, accepted) in [
            (
                "a rejected frame in the same chunk as the delta",
                vec![format!("{head}{rejected}")],
                false,
            ),
            (
                "the delta released, then a completion",
                vec![head.to_string(), COMPLETION.to_string()],
                true,
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("traffic");
            let parts = parts.iter().map(String::as_str).collect::<Vec<_>>();
            let body = stream_body_with(
                futures_util::stream::iter(chunks(&parts)),
                "msg_strict".into(),
                "synthetic-model".into(),
                None,
                "req".into(),
                Some(Arc::new(crate::traffic::test_capture(root.clone()))),
                strict_wiring(),
            );
            let text = body_text(body).await;
            assert_eq!(text.contains("message_stop"), accepted, "{case}: {text}");
            assert_eq!(text.contains(INVALID), !accepted, "{case}: {text}");
            assert_eq!(text.contains("early"), accepted, "{case}: {text}");
            let (names, texts) = synthetic_downstream_event_files(&root);
            let sent: Vec<String> = events_of(&text)
                .into_iter()
                .map(|(name, _)| name.unwrap_or_else(|| "message".to_string()))
                .collect();
            // The translator's file for each event the client received,
            // written when it was released, then the stream capture's copy
            // of the same events when the capture finishes; nothing for
            // output the client never got.
            assert_eq!(names, [sent.clone(), sent].concat(), "{case}: {text}");
            assert_eq!(
                texts.iter().any(|text| text.contains("early")),
                accepted,
                "{case}: {texts:?}"
            );
        }
    }

    #[test]
    fn synthetic_observer_settles_and_fails_where_the_whole_body_check_does_in_any_split() {
        let repaired_prefix = format!("{READ_CALL_ADDED}{}", synthetic_stalled_read_delta());
        let repaired = format!("{repaired_prefix}{WHITESPACE_DELTA}");
        let failed_prefix = concat!(
            "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
            "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"synthetic failure\"}}}\n\n",
        );
        let failed = format!(
            "{failed_prefix}data: {{\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"x\"}}\n\n"
        );
        let unlisted = concat!(
            "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
            "data: {\"type\":\"response.audio.delta\",\"delta\":\"AA==\"}\n\n",
            "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
        );
        let tests = [
            (
                "repaired read",
                repaired.as_str(),
                Ok(repaired_prefix.len()),
            ),
            ("upstream failure", failed.as_str(), Ok(failed_prefix.len())),
            (
                "unlisted event type",
                unlisted,
                Err(format!("{INVALID}: unlisted_event_type")),
            ),
        ];
        for (name, wire, expected) in tests {
            let bytes = wire.as_bytes();
            for split in 0..=bytes.len() {
                let mut observer = StrictResponsesObserver::new(strict_wiring().labels);
                let outcome = match observer.push(&bytes[..split]) {
                    Ok(ObservedChunk::Settled { at }) => Ok(at),
                    Ok(ObservedChunk::Reading) => match observer.push(&bytes[split..]) {
                        Ok(ObservedChunk::Settled { at }) => Ok(split + at),
                        Ok(ObservedChunk::Reading) => {
                            panic!("{name}: split at {split} did not settle")
                        }
                        Err(error) => Err(error.to_string()),
                    },
                    Err(error) => Err(error.to_string()),
                };
                assert_eq!(outcome, expected, "{name}: split at {split}");
            }
            match expected {
                Ok(at) => {
                    // The whole-body check settles too, and the accumulator
                    // gives the same answer over the body up to where the
                    // observer settled as over all of it.
                    assert!(
                        check_buffered_frames(bytes, &strict_wiring().labels).is_ok(),
                        "{name}"
                    );
                    let prefix =
                        accumulate_response_with(&bytes[..at], "msg", "model", &strict_wiring());
                    let whole = accumulate_response_with(bytes, "msg", "model", &strict_wiring());
                    assert_eq!(
                        prefix.map_err(|error| error.to_string()),
                        whole.map_err(|error| error.to_string()),
                        "{name}"
                    );
                }
                Err(message) => {
                    let error = check_buffered_frames(bytes, &strict_wiring().labels)
                        .unwrap_err()
                        .to_string();
                    assert_eq!(error, message, "{name}");
                }
            }
        }
    }

    /// The texts of every file a traffic capture wrote under `root`.
    fn synthetic_capture_texts(root: &std::path::Path) -> Vec<String> {
        let mut texts = Vec::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    texts.push(std::fs::read_to_string(&path).unwrap());
                }
            }
        }
        texts
    }

    const SYNTHETIC_SECRET: &str = "synthetic-secret-0123";

    fn synthetic_redactor() -> RedactedWiring {
        RedactedWiring {
            wiring: strict_wiring(),
            redact: Arc::new(|text: &str| text.replace(SYNTHETIC_SECRET, "[gone]")),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_redacted_capture_keeps_the_secret_out_of_every_file() {
        let wire = format!(
            "{}data: {{\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"echo {SYNTHETIC_SECRET}\"}}\n\n{}",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n",
            COMPLETION,
        );
        for redacted in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("traffic");
            let traffic = Some(Arc::new(crate::traffic::test_capture(root.clone())));
            let upstream = futures_util::stream::iter(chunks(&[&wire]));
            let body = if redacted {
                stream_body_redacted(
                    upstream,
                    "msg_strict".into(),
                    "synthetic-model".into(),
                    None,
                    "req".into(),
                    traffic,
                    synthetic_redactor(),
                )
            } else {
                stream_body_with(
                    upstream,
                    "msg_strict".into(),
                    "synthetic-model".into(),
                    None,
                    "req".into(),
                    traffic,
                    strict_wiring(),
                )
            };
            let text = body_text(body).await;
            assert!(text.contains(SYNTHETIC_SECRET), "{text}");
            assert!(text.contains("message_stop"), "{text}");
            let texts = synthetic_capture_texts(&root);
            assert_eq!(
                texts.iter().any(|text| text.contains(SYNTHETIC_SECRET)),
                !redacted,
                "redacted {redacted}: {texts:?}"
            );
            assert_eq!(
                texts.iter().any(|text| text.contains("echo [gone]")),
                redacted,
                "redacted {redacted}: {texts:?}"
            );
            // Upstream and downstream event files, the translator's included.
            let (names, _) = synthetic_downstream_event_files(&root);
            let sent = events_of(&text).len();
            assert_eq!(names.len(), 2 * sent, "redacted {redacted}: {names:?}");
        }
    }

    /// The name and block index of every event the client received.
    fn synthetic_event_sequence(text: &str) -> Vec<(String, Option<u64>)> {
        events_of(text)
            .into_iter()
            .map(|(name, value)| {
                (
                    name.unwrap_or_else(|| "message".to_string()),
                    value.get("index").and_then(serde_json::Value::as_u64),
                )
            })
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_discarded_output_leaves_a_well_formed_sequence_ending_in_the_error() {
        let rejected =
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"no\"}\n\n";
        let first_delta = "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hi\"}\n\n";
        let released = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"early\"}\n\n",
        );
        let second_block = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_b\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"late\"}\n\n",
        );
        let first_block_done = "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_a\"}}\n\n";
        let event = |name: &str, index: Option<u64>| (name.to_string(), index);
        let tests = [
            (
                "a delta and a rejected frame in the first chunk",
                vec![format!("{first_delta}{rejected}")],
                vec![event("message_start", None), event("error", None)],
            ),
            (
                "a rejected frame alone in the first chunk",
                vec![rejected.to_string()],
                vec![event("message_start", None), event("error", None)],
            ),
            (
                "a new block and a rejected frame after released output",
                vec![released.to_string(), format!("{second_block}{rejected}")],
                vec![
                    event("message_start", None),
                    event("content_block_start", Some(0)),
                    event("content_block_delta", Some(0)),
                    event("content_block_stop", Some(0)),
                    event("error", None),
                ],
            ),
            (
                "the close of a released block and a rejected frame",
                vec![
                    released.to_string(),
                    format!("{first_block_done}{rejected}"),
                ],
                vec![
                    event("message_start", None),
                    event("content_block_start", Some(0)),
                    event("content_block_delta", Some(0)),
                    event("content_block_stop", Some(0)),
                    event("error", None),
                ],
            ),
        ];
        for (name, parts, expected) in tests {
            let parts = parts.iter().map(String::as_str).collect::<Vec<_>>();
            let text = body_text(strict_body(futures_util::stream::iter(chunks(&parts)))).await;
            assert_eq!(synthetic_event_sequence(&text), expected, "{name}: {text}");
            let events = events_of(&text);
            assert_eq!(events[0].1["message"]["id"], "msg_strict", "{name}: {text}");
            assert_eq!(
                events[events.len() - 1].1,
                json!({"type":"error","error":{"type":"api_error","message":INVALID}}),
                "{name}: {text}"
            );
            assert!(!text.contains("late"), "{name}: {text}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_discarded_output_capture_records_the_events_the_client_received() {
        let chunk = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hi\"}\n\n",
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"no\"}\n\n",
        );
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("traffic");
        let body = stream_body_with(
            futures_util::stream::iter(chunks(&[chunk])),
            "msg_strict".into(),
            "synthetic-model".into(),
            None,
            "req".into(),
            Some(Arc::new(crate::traffic::test_capture(root.clone()))),
            strict_wiring(),
        );
        let text = body_text(body).await;
        let (names, _) = synthetic_downstream_event_files(&root);
        // Written when the error was sent, then the stream capture's copy.
        assert_eq!(
            names,
            ["message_start", "error", "message_start", "error"],
            "{text}"
        );
    }

    /// One upstream frame carrying a text delta of `len` ASCII `x` bytes
    /// for `output_index`.
    fn synthetic_text_delta(output_index: u64, len: usize) -> String {
        format!(
            "data: {{\"type\":\"response.output_text.delta\",\"output_index\":{output_index},\"delta\":\"{}\"}}\n\n",
            "x".repeat(len)
        )
    }

    /// Whether the client event `value` named `name`, re-encoded the way
    /// the translator writes it, is larger than the upstream frame limit.
    fn synthetic_exceeds_frame_limit(name: &str, value: &serde_json::Value) -> bool {
        encode_sse_event(Some(name), &value.to_string()).len() - 1 > MAX_SSE_FRAME_BYTES
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_large_released_delta_keeps_every_open_block_for_the_error() {
        let rejected =
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"no\"}\n\n";
        let first = "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"early\"}\n\n";
        let added = "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"msg_b\"}}\n\n";
        let event = |name: &str, index: Option<u64>| (name.to_string(), index);
        let expected = vec![
            event("message_start", None),
            event("content_block_start", Some(0)),
            event("content_block_delta", Some(0)),
            event("content_block_start", Some(1)),
            event("content_block_delta", Some(1)),
            event("content_block_stop", Some(0)),
            event("content_block_stop", Some(1)),
            event("error", None),
        ];
        let tests = [
            (
                "a delta whose client frame exceeds the upstream limit",
                1_048_480,
                true,
            ),
            (
                "a delta whose client frame stays under the upstream limit",
                1_048_400,
                false,
            ),
        ];
        for (name, len, exceeds) in tests {
            let large = synthetic_text_delta(1, len);
            // The upstream frame itself is within the limit, without its
            // final blank line.
            assert!(large.len() - 1 <= MAX_SSE_FRAME_BYTES, "{name}");
            let second = format!("{added}{large}");
            let parts = [first, second.as_str(), rejected];
            let text = body_text(strict_body(futures_util::stream::iter(chunks(&parts)))).await;
            assert_eq!(synthetic_event_sequence(&text), expected, "{name}");
            let events = events_of(&text);
            assert_eq!(events[0].1["message"]["id"], "msg_strict", "{name}");
            assert_eq!(events[2].1["delta"]["text"], "early", "{name}");
            let delta = &events[4].1;
            assert_eq!(
                delta["delta"]["text"].as_str().map(str::len),
                Some(len),
                "{name}"
            );
            assert_eq!(
                synthetic_exceeds_frame_limit("content_block_delta", delta),
                exceeds,
                "{name}"
            );
            assert_eq!(
                events[events.len() - 1].1,
                json!({"type":"error","error":{"type":"api_error","message":INVALID}}),
                "{name}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_large_discarded_delta_still_sends_the_message_start() {
        let rejected =
            "data: {\"type\":\"response.refusal.delta\",\"output_index\":0,\"delta\":\"no\"}\n\n";
        let large = synthetic_text_delta(0, 1_048_480);
        assert!(large.len() - 1 <= MAX_SSE_FRAME_BYTES);
        let chunk = format!("{large}{rejected}");
        let text = body_text(strict_body(futures_util::stream::iter(chunks(&[
            chunk.as_str()
        ]))))
        .await;
        assert_eq!(
            synthetic_event_sequence(&text),
            [
                ("message_start".to_string(), None),
                ("error".to_string(), None),
            ],
            "{text}"
        );
        let events = events_of(&text);
        assert_eq!(events[0].1["message"]["id"], "msg_strict", "{text}");
        assert_eq!(
            events[1].1,
            json!({"type":"error","error":{"type":"api_error","message":INVALID}}),
            "{text}"
        );
        assert!(!text.contains("xxxx"), "{text}");
    }
}
