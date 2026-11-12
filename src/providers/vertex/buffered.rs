//! Reading a whole upstream response for a client that did not ask for a
//! stream. Every wire is requested as a stream; this reader applies the
//! same limits to all of them: at most [`MAX_BUFFERED_RESPONSE_BYTES`], and
//! at most [`POST_TERMINAL_READ_LIMIT`] after a terminal event when the
//! upstream stays open.

use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::providers::grok::translate::stream::SseDecoder;
use crate::providers::opencode::chat::complete_frames_prefix;

use super::translate::stream::{POST_TERMINAL_READ_LIMIT, Sanitizer, StreamError};
use super::{BufferedCapture, MAX_BUFFERED_RESPONSE_BYTES};

/// The Responses events that end a response. The OpenCode Go wiring keeps
/// the same list privately; it is the set its strict reader holds the
/// completion for.
const RESPONSES_TERMINAL_EVENT_TYPES: [&str; 3] =
    ["response.completed", "response.incomplete", "response.done"];

/// What [`read_upstream`] feeds the body to, chunk by chunk.
pub(super) trait BufferedSink {
    /// Takes the next chunk of the body. An error ends the read with it.
    fn push(&mut self, chunk: &[u8]) -> Result<(), StreamError>;

    /// How many terminal events have arrived. The read window of
    /// [`POST_TERMINAL_READ_LIMIT`] starts at the first one and starts again
    /// whenever the count grows.
    fn terminal_events(&self) -> usize;
}

/// Why a non-streaming response could not be produced.
#[derive(Debug)]
pub(super) enum BufferedFailure {
    /// Reading the upstream failed, or the Gemini translator rejected it.
    Read(StreamError),
    /// A translator that takes the whole body rejected it. The message has
    /// already been through the sanitiser.
    Translate(anyhow::Error),
}

impl BufferedFailure {
    /// The kind recorded in the traffic capture.
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Read(error) => error.kind(),
            Self::Translate(_) => "invalid_response",
        }
    }
}

/// How a read that did not fail ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReadEnd {
    /// The upstream closed the body.
    Closed,
    /// The upstream stayed open for [`POST_TERMINAL_READ_LIMIT`] after the
    /// last terminal event.
    ReadLimit,
}

/// Feeds the upstream body to `sink` until the upstream closes, or until
/// [`POST_TERMINAL_READ_LIMIT`] has passed since the sink's last terminal
/// event. Before any terminal event there is no deadline: a model may think
/// for a long time before it sends anything. Each chunk is recorded in
/// `capture` before the sink sees it.
///
/// # Errors
///
/// A broken connection gives [`StreamError::Transport`]; more than
/// [`MAX_BUFFERED_RESPONSE_BYTES`] gives [`StreamError::Malformed`]; a
/// sink error is returned as it is.
pub(super) async fn read_upstream<S, E, K>(
    upstream: S,
    sink: &mut K,
    sanitize: &Sanitizer,
    mut capture: Option<&mut BufferedCapture>,
) -> Result<ReadEnd, StreamError>
where
    S: Stream<Item = Result<Bytes, E>>,
    K: BufferedSink,
{
    let mut upstream = std::pin::pin!(upstream);
    // The terminal count the current deadline was set for, and the deadline.
    let mut window: Option<(usize, tokio::time::Instant)> = None;
    let mut read = 0usize;
    loop {
        let terminal_events = sink.terminal_events();
        let next = if terminal_events == 0 {
            upstream.next().await
        } else {
            let deadline = match window {
                Some((counted, deadline)) if counted == terminal_events => deadline,
                _ => {
                    let deadline = tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT;
                    window = Some((terminal_events, deadline));
                    deadline
                }
            };
            match tokio::time::timeout_at(deadline, upstream.next()).await {
                Ok(next) => next,
                Err(_) => return Ok(ReadEnd::ReadLimit),
            }
        };
        let chunk = match next {
            Some(Ok(chunk)) => chunk,
            Some(Err(_)) => return Err(StreamError::Transport),
            None => return Ok(ReadEnd::Closed),
        };
        read = read.saturating_add(chunk.len());
        if read > MAX_BUFFERED_RESPONSE_BYTES {
            return Err(StreamError::Malformed("the response exceeds 16 MiB"));
        }
        if let Some(capture) = capture.as_deref_mut() {
            capture.upstream(&chunk, sanitize);
        }
        sink.push(&chunk)?;
    }
}

/// Which events of an event stream are terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TerminalRule {
    /// OpenAI Responses: `response.completed`, `response.incomplete` or
    /// `response.done`. A `[DONE]` after it does not start the window again,
    /// as in the streaming path.
    Responses,
    /// Chat Completions: a `finish_reason` or `[DONE]`. A `[DONE]` after the
    /// `finish_reason` starts the window again, as in the streaming path.
    ChatCompletions,
}

/// Keeps an event-stream body whole for a translator that takes the whole
/// body at once, and counts its terminal events for [`read_upstream`].
/// Validation is the translator's: this sink only reads what it needs to
/// find the terminal events.
pub(super) struct EventStreamSink {
    bytes: Vec<u8>,
    decoder: SseDecoder,
    rule: TerminalRule,
    /// The completion event, or the first `finish_reason`.
    finished: bool,
    /// `[DONE]`, counted for [`TerminalRule::ChatCompletions`] only.
    done: bool,
}

impl EventStreamSink {
    pub(super) fn new(rule: TerminalRule) -> Self {
        Self {
            bytes: Vec::new(),
            // The translators decode with the default frame limit; the same
            // limit here means both fail on the same frame.
            decoder: SseDecoder::default(),
            rule,
            finished: false,
            done: false,
        }
    }

    /// The body to translate. After [`ReadEnd::ReadLimit`] a frame still
    /// being received is left out, as the streaming path discards it; the
    /// translators would otherwise reject the body as cut off. After
    /// [`ReadEnd::Closed`] the body is passed whole, so a frame the upstream
    /// cut off stays an error.
    pub(super) fn into_body(self, end: ReadEnd) -> Vec<u8> {
        let mut bytes = self.bytes;
        if end == ReadEnd::ReadLimit {
            let complete = complete_frames_prefix(&bytes).len();
            bytes.truncate(complete);
        }
        bytes
    }
}

impl BufferedSink for EventStreamSink {
    fn push(&mut self, chunk: &[u8]) -> Result<(), StreamError> {
        self.bytes.extend_from_slice(chunk);
        let events = self
            .decoder
            .push(chunk)
            .map_err(|_| StreamError::Malformed("the response is not a readable event stream"))?;
        for event in events {
            let data = event.data.trim();
            if data == "[DONE]" {
                self.done = true;
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            self.finished |= match self.rule {
                TerminalRule::Responses => value
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|kind| RESPONSES_TERMINAL_EVENT_TYPES.contains(&kind)),
                TerminalRule::ChatCompletions => value
                    .get("choices")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|choices| {
                        choices.iter().any(|choice| {
                            choice
                                .get("finish_reason")
                                .is_some_and(|reason| reason.is_string())
                        })
                    }),
            };
        }
        Ok(())
    }

    fn terminal_events(&self) -> usize {
        match self.rule {
            TerminalRule::Responses => usize::from(self.finished),
            TerminalRule::ChatCompletions => usize::from(self.finished) + usize::from(self.done),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures_util::stream;

    use super::*;

    fn sanitizer() -> Sanitizer {
        Arc::new(|text: &str| text.to_string())
    }

    fn chunk(text: &str) -> Result<Bytes, std::io::Error> {
        Ok(Bytes::from(text.to_string()))
    }

    const DELTA: &str = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n";
    const COMPLETED: &str = "data: {\"type\":\"response.completed\",\"response\":{}}\n\n";

    #[tokio::test]
    async fn responses_sink_synthetic_reads_to_the_close_and_keeps_every_byte() {
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        let end = read_upstream(
            stream::iter([chunk(DELTA), chunk(COMPLETED)]),
            &mut sink,
            &sanitizer(),
            None,
        )
        .await
        .expect("a closed stream is read whole");
        assert_eq!(end, ReadEnd::Closed);
        assert_eq!(sink.terminal_events(), 1);
        assert_eq!(
            String::from_utf8(sink.into_body(end)).unwrap(),
            format!("{DELTA}{COMPLETED}")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn responses_sink_synthetic_completion_then_open_upstream_ends_at_the_read_limit() {
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        let started = tokio::time::Instant::now();
        let upstream = stream::iter([chunk(DELTA), chunk(COMPLETED), chunk("data: {\"ty")])
            .chain(stream::pending());
        let end = tokio::time::timeout(
            Duration::from_secs(60),
            read_upstream(upstream, &mut sink, &sanitizer(), None),
        )
        .await
        .expect("an upstream left open after the completion must not hold the read")
        .expect("the read ends without an error");
        assert_eq!(end, ReadEnd::ReadLimit);
        assert_eq!(started.elapsed(), Duration::from_secs(2));
        // The half frame that arrived after the completion is left out.
        assert_eq!(
            String::from_utf8(sink.into_body(end)).unwrap(),
            format!("{DELTA}{COMPLETED}")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn responses_sink_synthetic_silence_before_any_terminal_event_is_not_cut() {
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        let upstream = stream::iter([chunk(DELTA)]).chain(stream::pending());
        let pending = tokio::time::timeout(
            Duration::from_secs(600),
            read_upstream(upstream, &mut sink, &sanitizer(), None),
        )
        .await;
        assert!(pending.is_err(), "the read ended without a terminal event");
    }

    #[tokio::test]
    async fn responses_sink_synthetic_a_broken_connection_is_a_transport_error() {
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        let upstream = stream::iter([chunk(DELTA), Err(std::io::Error::other("synthetic reset"))]);
        let error = read_upstream(upstream, &mut sink, &sanitizer(), None)
            .await
            .expect_err("a broken connection is not a body");
        assert_eq!(error, StreamError::Transport);
    }

    #[tokio::test]
    async fn responses_sink_synthetic_a_body_over_the_cap_is_rejected() {
        // Complete comment frames: no frame nears the decoder's limit, so
        // only the byte cap can end the read.
        let mut filler = vec![b' '; 1024 * 1024];
        filler[0] = b':';
        filler[1024 * 1024 - 2] = b'\n';
        filler[1024 * 1024 - 1] = b'\n';
        let filler = Bytes::from(filler);
        let chunks = (0..=16).map(move |_| Ok::<Bytes, std::io::Error>(filler.clone()));
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        let error = read_upstream(stream::iter(chunks), &mut sink, &sanitizer(), None)
            .await
            .expect_err("more than 16 MiB is not a body");
        assert_eq!(error, StreamError::Malformed("the response exceeds 16 MiB"));
    }

    #[tokio::test]
    async fn responses_sink_synthetic_an_unreadable_event_stream_ends_the_read() {
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        let upstream = stream::iter([Ok::<Bytes, std::io::Error>(Bytes::from_static(
            b"data: \xff\xfe\n\n",
        ))])
        .chain(stream::pending());
        let error = read_upstream(upstream, &mut sink, &sanitizer(), None)
            .await
            .expect_err("bytes that are not an event stream end the read");
        assert_eq!(
            error,
            StreamError::Malformed("the response is not a readable event stream")
        );
    }

    #[test]
    fn responses_terminal_types_synthetic_are_the_three_completion_events() {
        for kind in ["response.completed", "response.incomplete", "response.done"] {
            let mut sink = EventStreamSink::new(TerminalRule::Responses);
            sink.push(format!("data: {{\"type\":\"{kind}\"}}\n\n").as_bytes())
                .unwrap();
            assert_eq!(sink.terminal_events(), 1, "{kind}");
        }
        for kind in [
            "response.output_text.delta",
            "response.output_item.done",
            "response.failed",
            "error",
        ] {
            let mut sink = EventStreamSink::new(TerminalRule::Responses);
            sink.push(format!("data: {{\"type\":\"{kind}\"}}\n\n").as_bytes())
                .unwrap();
            assert_eq!(sink.terminal_events(), 0, "{kind}");
        }
        // Every terminal type is one the strict wiring lets through.
        for kind in RESPONSES_TERMINAL_EVENT_TYPES {
            assert!(
                crate::providers::opencode::responses::RESPONSES_HANDLED_EVENT_TYPES
                    .contains(&kind),
                "{kind}"
            );
        }
    }

    const CHAT_DELTA: &str =
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n";
    const CHAT_FINISH: &str =
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";

    #[test]
    fn chat_terminal_events_synthetic_count_the_finish_reason_and_done() {
        let mut sink = EventStreamSink::new(TerminalRule::ChatCompletions);
        sink.push(CHAT_DELTA.as_bytes()).unwrap();
        assert_eq!(sink.terminal_events(), 0);
        // A null finish_reason is not a terminal event.
        sink.push(b"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\n")
            .unwrap();
        assert_eq!(sink.terminal_events(), 0);
        sink.push(CHAT_FINISH.as_bytes()).unwrap();
        assert_eq!(sink.terminal_events(), 1);
        sink.push(b"data: [DONE]\n\n").unwrap();
        assert_eq!(sink.terminal_events(), 2);

        let mut sink = EventStreamSink::new(TerminalRule::ChatCompletions);
        sink.push(b"data: [DONE]\n\n").unwrap();
        assert_eq!(sink.terminal_events(), 1);

        // For Responses, [DONE] is not counted.
        let mut sink = EventStreamSink::new(TerminalRule::Responses);
        sink.push(format!("{COMPLETED}data: [DONE]\n\n").as_bytes())
            .unwrap();
        assert_eq!(sink.terminal_events(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn chat_sink_synthetic_a_late_done_starts_the_read_window_again() {
        let mut sink = EventStreamSink::new(TerminalRule::ChatCompletions);
        let started = tokio::time::Instant::now();
        let upstream = stream::iter([chunk(CHAT_DELTA), chunk(CHAT_FINISH)])
            .chain(stream::once(async {
                tokio::time::sleep(Duration::from_millis(1_500)).await;
                chunk("data: [DONE]\n\n")
            }))
            .chain(stream::pending());
        let end = tokio::time::timeout(
            Duration::from_secs(60),
            read_upstream(upstream, &mut sink, &sanitizer(), None),
        )
        .await
        .expect("an upstream left open after [DONE] must not hold the read")
        .expect("the read ends without an error");
        assert_eq!(end, ReadEnd::ReadLimit);
        assert_eq!(started.elapsed(), Duration::from_millis(3_500));
        assert_eq!(
            String::from_utf8(sink.into_body(end)).unwrap(),
            format!("{CHAT_DELTA}{CHAT_FINISH}data: [DONE]\n\n")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn chat_sink_synthetic_finish_reason_then_open_upstream_ends_at_the_read_limit() {
        let mut sink = EventStreamSink::new(TerminalRule::ChatCompletions);
        let started = tokio::time::Instant::now();
        let upstream =
            stream::iter([chunk(CHAT_DELTA), chunk(CHAT_FINISH)]).chain(stream::pending());
        let end = tokio::time::timeout(
            Duration::from_secs(60),
            read_upstream(upstream, &mut sink, &sanitizer(), None),
        )
        .await
        .expect("an upstream left open after the finish_reason must not hold the read")
        .expect("the read ends without an error");
        assert_eq!(end, ReadEnd::ReadLimit);
        assert_eq!(started.elapsed(), Duration::from_secs(2));
    }

    #[tokio::test(start_paused = true)]
    async fn chat_sink_synthetic_silence_before_any_terminal_event_is_not_cut() {
        let mut sink = EventStreamSink::new(TerminalRule::ChatCompletions);
        let upstream = stream::iter([chunk(CHAT_DELTA)]).chain(stream::pending());
        let pending = tokio::time::timeout(
            Duration::from_secs(600),
            read_upstream(upstream, &mut sink, &sanitizer(), None),
        )
        .await;
        assert!(pending.is_err(), "the read ended without a terminal event");
    }
}
