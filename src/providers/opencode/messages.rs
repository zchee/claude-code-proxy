use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::anthropic::schema::MessagesRequest;
use crate::anthropic::sse::{SseEvent, encode_sse_event};
use crate::monitor::MonitorHandle;
use crate::providers::grok::translate::stream::SseDecoder;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use super::chat::{
    RawFrameEnds, ReceiveContext, StreamLabels, StreamWiring, UpstreamStream, box_upstream,
};
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
/// stream, with the labels and receive policy of `wiring`.
///
/// Under [`super::chat::TerminalPolicy::Strict`] only complete SSE frames
/// reach the client, as the upstream sent them: the bytes of a frame that
/// has not ended wait for the rest of it. What arrived with `message_stop`,
/// up to the end of its frame, is held until the read window after it has
/// passed; bytes after that frame are read and checked, never forwarded.
/// Traffic capture records a downstream event, and the monitor its
/// progress, only when its bytes are released; upstream events are
/// captured, sanitised, as they arrive.
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
            .next_output_with(&mut stream.context, &mut stream.strict)
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
        strict: StrictPassThrough::default(),
    }
}

/// One stream and its wiring. Its `Drop` runs before the state's, so an
/// abandoned stream is summarised under the wiring's labels.
struct WiredMessagesStream {
    state: MessagesStreamState<UpstreamStream>,
    context: ReceiveContext,
    strict: StrictPassThrough,
}

/// The bytes a strict stream has read and not yet forwarded.
#[derive(Debug, Default)]
struct StrictPassThrough {
    /// The start of a frame that has not ended yet.
    partial: Vec<u8>,
    /// Everything not yet forwarded up to the end of the `message_stop`
    /// frame, held until the read window after it has passed. When that
    /// frame ends with a CRLF whose LF arrives in a later chunk, the LF is
    /// added here.
    held: Option<Vec<u8>>,
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
        self.next_output_with(
            &mut ReceiveContext::default(),
            &mut StrictPassThrough::default(),
        )
        .await
    }

    async fn next_output_with(
        &mut self,
        context: &mut ReceiveContext,
        strict: &mut StrictPassThrough,
    ) -> Option<Bytes> {
        if self.terminal {
            return None;
        }
        if self.error_sent {
            self.terminal = true;
            return None;
        }
        if context.is_strict() {
            return self.next_output_strict(context, strict).await;
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
        if terminal {
            if self.decoder.finish().is_err() {
                return Some(self.fail_at("decoder", "trailing_incomplete_frame", context));
            }
            self.terminal = true;
            self.finish_capture(true, context);
        }
        Some(chunk)
    }

    /// The next bytes a strict stream releases: complete frames before
    /// `message_stop`, then what [`Self::settle_held`] releases.
    async fn next_output_strict(
        &mut self,
        context: &mut ReceiveContext,
        strict: &mut StrictPassThrough,
    ) -> Option<Bytes> {
        loop {
            if strict.held.is_some() {
                return self.settle_held(context, strict).await;
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
            match self.strict_chunk(&chunk, context, strict) {
                Ok(Some(released)) => {
                    self.record_released(&released, context);
                    return Some(released);
                }
                Ok(None) => {}
                Err((stage, kind)) => return Some(self.fail_at(stage, kind, context)),
            }
        }
    }

    /// Reads one chunk of a strict stream a frame at a time. Returns the
    /// complete frames to release, or holds everything up to the end of the
    /// `message_stop` frame in `strict.held`. The bytes after that frame are
    /// charged to [`super::chat::POST_TERMINAL_MAX_BYTES`] and decoded, and
    /// any event among them is an error.
    fn strict_chunk(
        &mut self,
        chunk: &Bytes,
        context: &mut ReceiveContext,
        strict: &mut StrictPassThrough,
    ) -> Result<Option<Bytes>, (&'static str, &'static str)> {
        let RawFrameEnds { continued, ends } = context.raw_frame_ends(chunk);
        // An LF that completes the CRLF of a frame released with the
        // previous chunk is released now, so every frame reaches the client
        // with both bytes of its line ending.
        let last_end = ends.last().copied().unwrap_or(0).max(continued);
        let mut start = 0;
        for end in ends {
            let events = self
                .decoder
                .push(&chunk[start..end])
                .map_err(|_| ("decoder", "malformed_sse"))?;
            start = end;
            let mut terminal = false;
            for event in events {
                let value: serde_json::Value =
                    serde_json::from_str(&event.data).map_err(|_| ("json", "malformed_event"))?;
                if let Some(capture) = self.stream_capture.as_mut() {
                    capture.upstream_event(
                        event.event.as_deref(),
                        &context.wiring.receive.upstream_value(&value),
                    );
                }
                let kind = value.get("type").and_then(serde_json::Value::as_str);
                if event.event.as_deref() == Some("error") || kind == Some("error") {
                    return Err(("upstream", "error_event"));
                }
                terminal |=
                    event.event.as_deref() == Some("message_stop") || kind == Some("message_stop");
            }
            if !terminal {
                continue;
            }
            if context.terminal_frame_ended(chunk, end) {
                return Err(("protocol", "post_terminal_limit"));
            }
            let events = self
                .decoder
                .push(&chunk[end..])
                .map_err(|_| ("decoder", "malformed_sse"))?;
            if let Some(event) = events.first() {
                self.capture_late_event(event, context);
                return Err(("protocol", "event_after_message_stop"));
            }
            let mut held = std::mem::take(&mut strict.partial);
            held.extend_from_slice(&chunk[..end]);
            strict.held = Some(held);
            return Ok(None);
        }
        self.decoder
            .push(&chunk[start..])
            .map_err(|_| ("decoder", "malformed_sse"))?;
        if last_end == 0 {
            strict.partial.extend_from_slice(chunk);
            return Ok(None);
        }
        let mut released = std::mem::take(&mut strict.partial);
        released.extend_from_slice(&chunk[..last_end]);
        strict.partial.extend_from_slice(&chunk[last_end..]);
        Ok(Some(Bytes::from(released)))
    }

    /// Records bytes a strict stream releases to the client: their events
    /// as downstream capture, sanitised, and their size and usage as
    /// progress.
    fn record_released(&mut self, released: &[u8], context: &ReceiveContext) {
        let mut input_tokens = None;
        let mut output_tokens = None;
        let events = SseDecoder::default().push(released).unwrap_or_default();
        for event in events {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&event.data) else {
                continue;
            };
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
            if let Some(capture) = self.stream_capture.as_mut() {
                capture.downstream_event(
                    event.event.as_deref().unwrap_or("message"),
                    context.wiring.receive.upstream_value(&value).into_owned(),
                );
            }
        }
        if let Some(monitor) = self.monitor.as_ref() {
            monitor.stream_progress(
                &self.req_id,
                released.len() as u64,
                1,
                input_tokens,
                output_tokens,
            );
        }
    }

    /// Captures the first event read after `message_stop`, which ends a
    /// strict stream with an error.
    fn capture_late_event(&mut self, event: &SseEvent, context: &ReceiveContext) {
        if let (Some(capture), Ok(value)) = (
            self.stream_capture.as_mut(),
            serde_json::from_str::<serde_json::Value>(&event.data),
        ) {
            capture.upstream_event(
                event.event.as_deref(),
                &context.wiring.receive.upstream_value(&value),
            );
        }
    }

    /// Reads a strict stream after `message_stop` until the upstream closes
    /// or the read window ends, then releases the held bytes. Any event in
    /// that window is an error.
    async fn settle_held(
        &mut self,
        context: &mut ReceiveContext,
        strict: &mut StrictPassThrough,
    ) -> Option<Bytes> {
        loop {
            let chunk = match context.read_after_terminal(&mut self.upstream).await {
                Some(Some(Ok(chunk))) => chunk,
                Some(Some(Err(_))) => {
                    return Some(self.fail_at("transport", "upstream_stream", context));
                }
                // The upstream closed or the read window ended. The held bytes
                // reach the client as received, so a partial frame still
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
            let continued = context.raw_frame_ends(&chunk).continued;
            let own = context.terminal_lf(&chunk, continued);
            if let Some(held) = strict.held.as_mut() {
                held.extend_from_slice(&chunk[..own]);
            }
            if context.exceeds_post_terminal_bytes(chunk.len() - own) {
                return Some(self.fail_at("protocol", "post_terminal_limit", context));
            }
            let events = match self.decoder.push(&chunk) {
                Ok(events) => events,
                Err(_) => return Some(self.fail_at("decoder", "malformed_sse", context)),
            };
            if let Some(event) = events.first() {
                self.capture_late_event(event, context);
                return Some(self.fail_at("protocol", "event_after_message_stop", context));
            }
        }
        let held = Bytes::from(strict.held.take()?);
        self.record_released(&held, context);
        self.terminal = true;
        self.finish_capture(true, context);
        Some(held)
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

    const INVALID: &str = "Synthetic Gateway Messages stream is invalid";

    fn synthetic_strict_wiring() -> StreamWiring {
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

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_never_forwards_part_of_a_frame() {
        let late = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
        for split in 0..=MESSAGE.len() {
            for follow in ["late event", "transport error", "clean close"] {
                let mut items = vec![
                    Ok(Bytes::copy_from_slice(&MESSAGE.as_bytes()[..split])),
                    Ok(Bytes::copy_from_slice(&MESSAGE.as_bytes()[split..])),
                ];
                match follow {
                    "late event" => items.push(Ok(Bytes::from_static(late.as_bytes()))),
                    "transport error" => items.push(Err(std::io::Error::other("connection reset"))),
                    _ => {}
                }
                let case = format!("split at {split}, then {follow}");
                let mut body = strict_body(futures_util::stream::iter(items)).into_data_stream();
                let mut text = String::new();
                while let Some(item) = body.next().await {
                    let item = item.unwrap();
                    let mut decoder = SseDecoder::default();
                    decoder.push(&item).unwrap();
                    assert!(
                        decoder.finish().is_ok(),
                        "{case}: {item:?} ends inside a frame"
                    );
                    text.push_str(std::str::from_utf8(&item).unwrap());
                }
                if follow == "clean close" {
                    assert_eq!(text, MESSAGE, "{case}");
                    continue;
                }
                let events = SseDecoder::default().push(text.as_bytes()).unwrap();
                for event in &events {
                    assert!(
                        serde_json::from_str::<serde_json::Value>(&event.data).is_ok(),
                        "{case}: {text}"
                    );
                    assert_ne!(
                        event.event.as_deref(),
                        Some("message_stop"),
                        "{case}: {text}"
                    );
                }
                let last = events.last().unwrap_or_else(|| panic!("{case}: {text}"));
                assert_eq!(last.event.as_deref(), Some("error"), "{case}: {text}");
                let error: serde_json::Value = serde_json::from_str(&last.data).unwrap();
                assert_eq!(error["error"]["message"], INVALID, "{case}: {text}");
            }
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
    async fn synthetic_strict_stream_charges_the_tail_after_message_stop_by_position() {
        let at_limit = synthetic_comment_tail_of_sixteen_mib();
        let over_limit = format!("{at_limit}\n");
        for (tail, accepted) in [(at_limit.as_str(), true), (over_limit.as_str(), false)] {
            let wire = format!("{MESSAGE}{tail}");
            let inside_tail = MESSAGE.len() + 4096 + 7;
            for parts in [
                vec![wire.as_str()],
                vec![MESSAGE, tail],
                vec![&wire[..inside_tail], &wire[inside_tail..]],
            ] {
                let upstream = futures_util::stream::iter(chunks(&parts));
                let text = body_text(strict_body(upstream)).await;
                let case = format!("{} tail bytes in {} items", tail.len(), parts.len());
                if accepted {
                    assert_eq!(text, MESSAGE, "{case}");
                } else {
                    assert!(text.contains(INVALID), "{case}");
                    assert!(!text.contains("message_stop"), "{case}");
                }
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_waits_exactly_two_seconds_after_message_stop() {
        let upstream =
            futures_util::stream::iter(chunks(&[MESSAGE])).chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let text = body_text(strict_body(upstream)).await;
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(2));
        assert_eq!(text, MESSAGE);
    }

    /// The events a finished traffic capture wrote under `root`, upstream
    /// and downstream, by event name, and the stream summary.
    fn synthetic_captured(root: &std::path::Path) -> (Vec<String>, Vec<String>, serde_json::Value) {
        let read = |path: &std::path::Path| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
        };
        let mut events = std::fs::read_dir(root.join("events"))
            .map(|dir| dir.map(|entry| entry.unwrap().path()).collect::<Vec<_>>())
            .unwrap_or_default();
        events.sort();
        let mut upstream = Vec::new();
        let mut downstream = Vec::new();
        for path in events {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let event = read(&path)["event"].as_str().unwrap().to_string();
            if name.ends_with("-040-upstream-event.json") {
                upstream.push(event);
            } else if name.ends_with("-050-downstream-event.json") {
                downstream.push(event);
            }
        }
        let summary = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.to_string_lossy()
                    .ends_with("-061-synthetic-messages-stream-summary.json")
            })
            .map(|path| read(&path))
            .unwrap_or_else(|| panic!("no stream summary under {}", root.display()));
        (upstream, downstream, summary)
    }

    fn synthetic_traffic_body<S>(upstream: S, root: &std::path::Path) -> Body
    where
        S: futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    {
        stream_body_with(
            upstream,
            None,
            "req".into(),
            Some(Arc::new(crate::traffic::test_capture(root.to_path_buf()))),
            synthetic_strict_wiring(),
        )
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_capture_records_only_released_output() {
        let late = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\n";
        let whole = format!("{MESSAGE}{late}");
        for parts in [vec![whole.as_str()], vec![MESSAGE, late]] {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().join("traffic");
            let upstream = futures_util::stream::iter(chunks(&parts));
            let text = body_text(synthetic_traffic_body(upstream, &root)).await;
            assert!(text.contains(INVALID), "{parts:?}: {text}");
            let (upstream, downstream, summary) = synthetic_captured(&root);
            assert_eq!(
                upstream,
                [
                    "message_start",
                    "content_block_delta",
                    "message_stop",
                    "content_block_delta"
                ],
                "{parts:?}"
            );
            assert_eq!(downstream, ["error"], "{parts:?}");
            assert_eq!(summary["completion"]["kind"], "stream_error", "{parts:?}");
            assert_eq!(summary["downstream_events"]["captured"], 1, "{parts:?}");
        }

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("traffic");
        let upstream = futures_util::stream::iter(chunks(&[MESSAGE]));
        let text = body_text(synthetic_traffic_body(upstream, &root)).await;
        assert_eq!(text, MESSAGE);
        let (upstream, downstream, summary) = synthetic_captured(&root);
        let expected = ["message_start", "content_block_delta", "message_stop"];
        assert_eq!(upstream, expected);
        assert_eq!(downstream, expected);
        assert_eq!(summary["completion"]["kind"], "stream_completion");
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_capture_records_no_held_output_when_the_body_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("traffic");
        let head = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n";
        let rest = &MESSAGE[head.len()..];
        let upstream = futures_util::stream::iter(chunks(&[head, rest]))
            .chain(futures_util::stream::pending());
        let mut body = synthetic_traffic_body(upstream, &root).into_data_stream();
        let first = body.next().await.unwrap().unwrap();
        assert_eq!(first, head.as_bytes());
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(500), body.next())
                .await
                .is_err()
        );
        drop(body);
        let (upstream, downstream, summary) = synthetic_captured(&root);
        assert_eq!(
            upstream,
            ["message_start", "content_block_delta", "message_stop"]
        );
        assert_eq!(downstream, ["message_start"]);
        assert_eq!(summary["completion"]["kind"], "stream_abandoned");
        assert_eq!(summary["downstream_events"]["captured"], 1);
    }

    /// [`MESSAGE`] with every line ended by a CRLF.
    fn synthetic_crlf_message() -> String {
        MESSAGE.replace('\n', "\r\n")
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_passes_a_crlf_message_through_byte_for_byte_in_any_split() {
        let message = synthetic_crlf_message();
        for split in 0..=message.len() {
            for open in [false, true] {
                let items = chunks(&[&message[..split], &message[split..]]);
                let case = format!(
                    "split at {split}, upstream {}",
                    if open { "left open" } else { "closed" }
                );
                let text = if open {
                    let upstream =
                        futures_util::stream::iter(items).chain(futures_util::stream::pending());
                    body_text(strict_body(upstream)).await
                } else {
                    body_text(strict_body(futures_util::stream::iter(items))).await
                };
                assert_eq!(text, message, "{case}");
            }
        }
        // The LF of the final CRLF alone in the last item, after the CR.
        let between_cr_and_lf = message.len() - 1;
        let upstream = futures_util::stream::iter(chunks(&[
            &message[..between_cr_and_lf],
            "\n",
            ": keepalive\r\n\r\n",
        ]));
        assert_eq!(body_text(strict_body(upstream)).await, message);
    }

    #[tokio::test(start_paused = true)]
    async fn synthetic_strict_stream_charges_no_byte_of_a_crlf_message_stop_frame_to_the_tail() {
        let message = synthetic_crlf_message();
        let at_limit = synthetic_comment_tail_of_sixteen_mib();
        let over_limit = format!("{at_limit}\n");
        for (tail, accepted) in [(at_limit.as_str(), true), (over_limit.as_str(), false)] {
            let wire = format!("{message}{tail}");
            let between_cr_and_lf = message.len() - 1;
            let inside_tail = message.len() + 4096 + 7;
            for (split, parts) in [
                ("whole", vec![wire.as_str()]),
                ("after the frame", vec![message.as_str(), tail]),
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
                if accepted {
                    assert_eq!(text, message, "{case}");
                } else {
                    assert!(text.contains(INVALID), "{case}");
                    assert!(!text.contains("message_stop"), "{case}");
                }
            }
        }
    }
}
