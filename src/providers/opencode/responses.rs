use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::anthropic::schema::MessagesRequest;
use crate::monitor::{MonitorHandle, usage_from_anthropic_sse};
use crate::providers::codex::translate::{
    IncompleteResponsePolicy, accumulate::accumulate_response_with_policy,
    live_stream::LiveStreamTranslator, request::translate_openai_compatible_request,
};
use crate::providers::grok::translate::stream::SseDecoder;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use super::chat::{
    ReceiveContext, StreamLabels, StreamWiring, TerminalPolicy, UpstreamStream, box_upstream,
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
/// that repeat what the deltas already delivered.
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
/// The whole body is already read, so
/// [`super::chat::POST_TERMINAL_READ_LIMIT`] is the reader's to apply.
/// Under [`TerminalPolicy::Strict`] every frame is checked as the
/// streaming path checks it before the body is translated, and any event
/// after the completion, `ping` included, is an error. The error goes
/// through [`super::chat::ReceivePolicy::sanitize`].
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

/// The strict frame rules over a whole buffered body.
fn check_buffered_frames(upstream: &[u8], labels: &StreamLabels) -> anyhow::Result<()> {
    let invalid =
        |kind: &str| anyhow::anyhow!("{} Responses stream is invalid: {kind}", labels.provider);
    let mut decoder = SseDecoder::default();
    let events = decoder
        .push(upstream)
        .map_err(|_| invalid("malformed_sse"))?;
    decoder.finish().map_err(|_| invalid("incomplete_stream"))?;
    let mut completed = false;
    let mut done_seen = false;
    for event in events {
        let data = event.data.trim();
        if data == "[DONE]" {
            if !completed || done_seen {
                return Err(invalid("premature_done"));
            }
            done_seen = true;
            continue;
        }
        if completed {
            return Err(invalid("event_after_completion"));
        }
        let value: serde_json::Value =
            serde_json::from_str(data).map_err(|_| invalid("malformed_event"))?;
        if let Some(kind) = unlisted_frame(&value) {
            return Err(invalid(kind));
        }
        completed = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|event_type| TERMINAL_EVENT_TYPES.contains(&event_type));
    }
    Ok(())
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
    }
}

/// One stream and its wiring. Its `Drop` runs before the state's, so an
/// abandoned stream is summarised under the wiring's labels.
struct WiredResponsesStream {
    state: ResponsesStreamState<UpstreamStream>,
    context: ReceiveContext,
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
            if after_terminal && context.exceeds_post_terminal_bytes(chunk.len()) {
                return Some(self.fail_at("protocol", "post_terminal_limit", context));
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
                    if event_type == Some("ping") && !context.is_strict() {
                        continue;
                    }
                    return Some(self.fail_at("protocol", "event_after_completion", context));
                }
                if context.is_strict()
                    && let Some(kind) = unlisted_frame(&value)
                {
                    return Some(self.fail_at("protocol", kind, context));
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
                self.capture_downstream(&output);
                return Some(output);
            }
        }
    }

    /// Translates the held completion once the upstream closed or, for a
    /// strict stream, the read window after it ended.
    fn complete(&mut self, context: &ReceiveContext) -> Option<Vec<u8>> {
        let Some(completion) = self.pending_completion.take() else {
            return Some(self.fail_at("decoder", "incomplete_stream", context));
        };
        let output = match self.translator.accept(&completion, self.traffic.as_deref()) {
            Ok(output) => output,
            Err(_) => return Some(self.fail_at("translation", "invalid_event", context)),
        };
        self.terminal = true;
        self.record_progress(&output);
        self.capture_downstream(&output);
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
        self.capture_downstream(&output);
        self.finish_capture(false, context);
        output
    }

    fn capture_downstream(&mut self, bytes: &[u8]) {
        let Some(capture) = self.stream_capture.as_mut() else {
            return;
        };
        let mut decoder = SseDecoder::default();
        if let Ok(events) = decoder.push(bytes) {
            for event in events {
                if let Ok(value) = serde_json::from_str(&event.data) {
                    capture.downstream_event(event.event.as_deref().unwrap_or("message"), value);
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
}
