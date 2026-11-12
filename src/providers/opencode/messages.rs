use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::anthropic::schema::MessagesRequest;
use crate::anthropic::sse::encode_sse_event;
use crate::monitor::MonitorHandle;
use crate::providers::grok::translate::stream::SseDecoder;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use super::chat::{ReceiveContext, StreamLabels, StreamWiring, UpstreamStream, box_upstream};
use super::client::{OpenCodeError, OpenCodeResponse};

const DEFAULT_MAX_TOKENS: u32 = 32_000;

pub fn prepare_request(
    body: &MessagesRequest,
    model: &str,
) -> Result<serde_json::Value, serde_json::Error> {
    let mut translated = serde_json::to_value(body)?;
    translated["model"] = serde_json::Value::String(model.to_string());
    translated["max_tokens"] = serde_json::json!(
        body.max_tokens
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MAX_TOKENS)
    );
    if model == "minimax-m3" && translated.get("thinking").is_none() {
        translated["thinking"] = serde_json::json!({"type": "adaptive"});
    }
    Ok(translated)
}

pub fn stream_body(
    upstream: OpenCodeResponse,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
) -> Body {
    stream_body_with(
        upstream.into_stream(),
        monitor,
        req_id,
        traffic,
        StreamWiring::default(),
    )
}

/// The Anthropic SSE body passed through from any Messages upstream byte
/// stream, with the labels and receive policy of `wiring`. Under
/// [`super::chat::TerminalPolicy::Strict`] the chunk that carries
/// `message_stop` is held until the read window after it has passed.
pub fn stream_body_with<S, E>(
    upstream: S,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
    wiring: StreamWiring,
) -> Body
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let stream = wired_stream(upstream, monitor, req_id, traffic, wiring);
    let stream = futures_util::stream::unfold(stream, |mut stream| async move {
        stream
            .state
            .next_output_with(&mut stream.context, &mut stream.held)
            .await
            .map(|bytes| (Ok::<Bytes, Infallible>(bytes), stream))
    });
    Body::from_stream(stream)
}

fn wired_stream<S, E>(
    upstream: S,
    monitor: Option<MonitorHandle>,
    req_id: String,
    traffic: Option<Arc<TrafficCapture>>,
    wiring: StreamWiring,
) -> WiredMessagesStream
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
{
    let upstream = box_upstream(upstream);
    let state = MessagesStreamState {
        upstream,
        decoder: SseDecoder::default(),
        terminal: false,
        error_sent: false,
        monitor,
        req_id,
        bytes: 0,
        chunks: 0,
        stream_capture: traffic.as_ref().map(|traffic| traffic.stream_capture()),
        traffic,
    };
    WiredMessagesStream {
        state,
        context: ReceiveContext::new(wiring),
        held: None,
    }
}

/// One stream and its wiring. Its `Drop` runs before the state's, so an
/// abandoned stream is summarised under the wiring's labels.
struct WiredMessagesStream {
    state: MessagesStreamState<UpstreamStream>,
    context: ReceiveContext,
    /// The chunk that carried `message_stop`, held by a strict stream until
    /// the read window after it has passed.
    held: Option<Bytes>,
}

impl Drop for WiredMessagesStream {
    fn drop(&mut self) {
        self.state.abandon(self.context.labels());
    }
}

struct MessagesStreamState<S> {
    upstream: S,
    decoder: SseDecoder,
    terminal: bool,
    error_sent: bool,
    monitor: Option<MonitorHandle>,
    req_id: String,
    bytes: u64,
    chunks: u64,
    stream_capture: Option<StreamTrafficCapture>,
    traffic: Option<Arc<TrafficCapture>>,
}

impl<S> MessagesStreamState<S>
where
    S: futures_util::Stream<Item = Result<Bytes, OpenCodeError>> + Unpin,
{
    #[cfg(test)]
    async fn next_output(&mut self) -> Option<Bytes> {
        self.next_output_with(&mut ReceiveContext::default(), &mut None)
            .await
    }

    async fn next_output_with(
        &mut self,
        context: &mut ReceiveContext,
        held: &mut Option<Bytes>,
    ) -> Option<Bytes> {
        if self.terminal {
            return None;
        }
        if self.error_sent {
            self.terminal = true;
            return None;
        }
        if held.is_some() {
            return self.settle_held(context, held).await;
        }

        let chunk = match self.upstream.next().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(_)) => return Some(self.fail_at("transport", "upstream_stream", context)),
            None => {
                if self.decoder.finish().is_err() {
                    return Some(self.fail_at("decoder", "incomplete_stream", context));
                }
                return Some(self.fail_at("protocol", "missing_message_stop", context));
            }
        };
        if self.bytes == 0
            && let Some(monitor) = self.monitor.as_ref()
        {
            monitor.generation_started(&self.req_id);
        }
        self.bytes = self.bytes.saturating_add(chunk.len() as u64);
        self.chunks = self.chunks.saturating_add(1);

        let events = match self.decoder.push(&chunk) {
            Ok(events) => events,
            Err(_) => return Some(self.fail_at("decoder", "malformed_sse", context)),
        };
        let mut input_tokens = None;
        let mut output_tokens = None;
        let mut terminal = false;
        for event in events {
            if terminal {
                return Some(self.fail_at("protocol", "event_after_message_stop", context));
            }
            let value: serde_json::Value = match serde_json::from_str(&event.data) {
                Ok(value) => value,
                Err(_) => return Some(self.fail_at("json", "malformed_event", context)),
            };
            if let Some(capture) = self.stream_capture.as_mut() {
                let captured = context.wiring.receive.upstream_value(&value);
                capture.upstream_event(event.event.as_deref(), &captured);
                capture.downstream_event(
                    event.event.as_deref().unwrap_or("message"),
                    captured.into_owned(),
                );
            }
            input_tokens = value
                .pointer("/message/usage/input_tokens")
                .or_else(|| value.pointer("/usage/input_tokens"))
                .and_then(serde_json::Value::as_u64)
                .or(input_tokens);
            output_tokens = value
                .pointer("/message/usage/output_tokens")
                .or_else(|| value.pointer("/usage/output_tokens"))
                .and_then(serde_json::Value::as_u64)
                .or(output_tokens);
            let kind = value.get("type").and_then(serde_json::Value::as_str);
            if event.event.as_deref() == Some("error") || kind == Some("error") {
                return Some(self.fail_at("upstream", "error_event", context));
            }
            if event.event.as_deref() == Some("message_stop") || kind == Some("message_stop") {
                terminal = true;
            }
        }
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.stream_progress(
                &self.req_id,
                chunk.len() as u64,
                1,
                input_tokens,
                output_tokens,
            );
        }
        if terminal && context.is_strict() {
            *held = Some(chunk);
            return self.settle_held(context, held).await;
        }
        if terminal {
            if self.decoder.finish().is_err() {
                return Some(self.fail_at("decoder", "trailing_incomplete_frame", context));
            }
            self.terminal = true;
            self.finish_capture(true, context);
        }
        Some(chunk)
    }

    /// Reads a strict stream after `message_stop` until the upstream closes
    /// or the read window ends, then releases the held chunk. Any event in
    /// that window is an error.
    async fn settle_held(
        &mut self,
        context: &mut ReceiveContext,
        held: &mut Option<Bytes>,
    ) -> Option<Bytes> {
        loop {
            let chunk = match context.read_after_terminal(&mut self.upstream).await {
                Some(Some(Ok(chunk))) => chunk,
                Some(Some(Err(_))) => {
                    return Some(self.fail_at("transport", "upstream_stream", context));
                }
                // The upstream closed or the read window ended. The held chunk
                // reaches the client as received, so a partial frame still
                // buffered is an error in both cases.
                Some(None) | None => {
                    if self.decoder.finish().is_err() {
                        return Some(self.fail_at("decoder", "trailing_incomplete_frame", context));
                    }
                    break;
                }
            };
            self.bytes = self.bytes.saturating_add(chunk.len() as u64);
            self.chunks = self.chunks.saturating_add(1);
            if context.exceeds_post_terminal_bytes(chunk.len()) {
                return Some(self.fail_at("protocol", "post_terminal_limit", context));
            }
            let events = match self.decoder.push(&chunk) {
                Ok(events) => events,
                Err(_) => return Some(self.fail_at("decoder", "malformed_sse", context)),
            };
            if let Some(event) = events.first() {
                if let (Some(capture), Ok(value)) = (
                    self.stream_capture.as_mut(),
                    serde_json::from_str::<serde_json::Value>(&event.data),
                ) {
                    capture.upstream_event(
                        event.event.as_deref(),
                        &context.wiring.receive.upstream_value(&value),
                    );
                }
                return Some(self.fail_at("protocol", "event_after_message_stop", context));
            }
        }
        self.terminal = true;
        self.finish_capture(true, context);
        held.take()
    }

    fn fail_at(&mut self, stage: &str, kind: &str, context: &ReceiveContext) -> Bytes {
        let labels = context.labels();
        self.error_sent = true;
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.malformed(stage, kind);
        }
        if let Some(traffic) = self.traffic.as_ref() {
            traffic.write_json(
                &format!("060-{}-messages-stream-error", labels.capture_stem),
                &serde_json::json!({
                    "stage": stage,
                    "kind": kind,
                    "bytes": self.bytes,
                    "chunks": self.chunks,
                }),
            );
        }
        let value = serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": format!("{} Messages stream is invalid", labels.provider)
            }
        });
        if let Some(capture) = self.stream_capture.as_mut() {
            capture.downstream_event("error", value.clone());
        }
        self.finish_capture(false, context);
        Bytes::from(encode_sse_event(Some("error"), &value.to_string()))
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
    format!("061-{}-messages-stream-summary", labels.capture_stem)
}

impl<S> MessagesStreamState<S> {
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

impl<S> Drop for MessagesStreamState<S> {
    fn drop(&mut self) {
        self.abandon(&StreamLabels::OPENCODE_GO);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn minimax_m3_defaults_to_adaptive_thinking_without_overriding_the_caller() {
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "minimax-m3",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let translated = prepare_request(&body, "minimax-m3").unwrap();
        assert_eq!(translated["thinking"], json!({"type": "adaptive"}));
        assert_eq!(translated["max_tokens"], DEFAULT_MAX_TOKENS);

        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "minimax-m3",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        }))
        .unwrap();
        let translated = prepare_request(&body, "minimax-m3").unwrap();
        assert_eq!(
            translated["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048})
        );
        assert_eq!(translated["max_tokens"], 2048);
    }

    #[tokio::test]
    async fn live_stream_rejects_an_incomplete_frame_after_message_stop() {
        let upstream =
            futures_util::stream::iter([Ok::<Bytes, OpenCodeError>(Bytes::from_static(
                b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\ndata: {",
            ))]);
        let mut state = MessagesStreamState {
            upstream,
            decoder: SseDecoder::default(),
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
            String::from_utf8_lossy(&output).contains("OpenCode Go Messages stream is invalid")
        );
    }

    #[tokio::test]
    async fn live_stream_rejects_a_complete_event_after_message_stop() {
        let upstream =
            futures_util::stream::iter([Ok::<Bytes, OpenCodeError>(Bytes::from_static(
                concat!(
                    "event: message_stop\n",
                    "data: {\"type\":\"message_stop\"}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\n"
                )
                .as_bytes(),
            ))]);
        let mut state = MessagesStreamState {
            upstream,
            decoder: SseDecoder::default(),
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
            String::from_utf8_lossy(&output).contains("OpenCode Go Messages stream is invalid")
        );
    }

    use crate::providers::opencode::chat::{
        POST_TERMINAL_MAX_BYTES, POST_TERMINAL_READ_LIMIT, ReceivePolicy, TerminalPolicy,
    };

    const MESSAGE: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    fn strict_body<S>(upstream: S) -> Body
    where
        S: futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    {
        stream_body_with(
            upstream,
            None,
            "req".into(),
            None,
            StreamWiring {
                labels: StreamLabels {
                    provider: "Synthetic Gateway",
                    capture_stem: "synthetic",
                },
                receive: ReceivePolicy {
                    terminal: TerminalPolicy::Strict,
                    sanitize: None,
                },
            },
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
    async fn synthetic_strict_stream_rejects_an_event_in_a_later_chunk_after_message_stop() {
        for late in [
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
            "data: not-json\n\n",
        ] {
            let upstream = futures_util::stream::iter(chunks(&[MESSAGE, late]));
            let text = body_text(strict_body(upstream)).await;
            assert!(
                text.contains("Synthetic Gateway Messages stream is invalid"),
                "late {late}: {text}"
            );
            assert!(!text.contains("message_stop"), "late {late}: {text}");
            assert!(!text.contains("\"hi\""), "late {late}: {text}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_a_partial_frame_left_when_upstream_closes() {
        let upstream = futures_util::stream::iter(chunks(&[MESSAGE, "data: {"]));
        let text = body_text(strict_body(upstream)).await;
        assert!(
            text.contains("Synthetic Gateway Messages stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_passes_the_message_through_when_upstream_closes() {
        let upstream = futures_util::stream::iter(chunks(&[MESSAGE, ": keepalive\n\n"]));
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(upstream)).await;
        assert!(started.elapsed() < POST_TERMINAL_READ_LIMIT);
        assert_eq!(text, MESSAGE);
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_completes_at_the_read_limit_when_upstream_stays_open() {
        let upstream =
            futures_util::stream::iter(chunks(&[MESSAGE])).chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(upstream)).await;
        assert!(started.elapsed() >= POST_TERMINAL_READ_LIMIT);
        assert_eq!(text, MESSAGE);
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_a_partial_frame_left_at_the_read_limit() {
        for parts in [
            vec![format!("{MESSAGE}data: {{\"ty")],
            vec![MESSAGE.to_string(), "data: {\"ty".to_string()],
        ] {
            let parts = parts.iter().map(String::as_str).collect::<Vec<_>>();
            let upstream =
                futures_util::stream::iter(chunks(&parts)).chain(futures_util::stream::pending());
            let started = tokio::time::Instant::now();
            let text = body_text(strict_body(upstream)).await;
            assert!(started.elapsed() >= POST_TERMINAL_READ_LIMIT, "{parts:?}");
            assert!(
                text.contains("Synthetic Gateway Messages stream is invalid"),
                "{parts:?}: {text}"
            );
            assert!(!text.contains("message_stop"), "{parts:?}: {text}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_holds_the_message_stop_chunk_until_upstream_closes() {
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let upstream = futures_util::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|chunk| (chunk, receiver))
        });
        let mut body = strict_body(upstream).into_data_stream();
        sender
            .send(Ok(Bytes::from_static(MESSAGE.as_bytes())))
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), body.next())
                .await
                .is_err()
        );
        drop(sender);
        let chunk = body.next().await.unwrap().unwrap();
        assert_eq!(chunk, MESSAGE.as_bytes());
        assert!(body.next().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_rejects_more_than_the_post_terminal_byte_limit() {
        // Comment-only frames carry no event, so only the byte limit stops them.
        let comment = format!(":{}\n\n", "x".repeat(512 * 1024));
        let mut parts = vec![MESSAGE];
        parts.extend(std::iter::repeat_n(
            comment.as_str(),
            POST_TERMINAL_MAX_BYTES / comment.len() + 1,
        ));
        let upstream =
            futures_util::stream::iter(chunks(&parts)).chain(futures_util::stream::pending());
        let text = body_text(strict_body(upstream)).await;
        assert!(
            text.contains("Synthetic Gateway Messages stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("message_stop"), "{text}");
    }
}
