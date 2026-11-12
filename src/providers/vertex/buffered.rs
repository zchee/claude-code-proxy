//! Reading a whole upstream response for a client that did not ask for a
//! stream. Every wire is requested as a stream; this reader applies the
//! same limits to all of them: at most [`MAX_BUFFERED_RESPONSE_BYTES`], and
//! at most [`POST_TERMINAL_READ_LIMIT`] after a terminal event when the
//! upstream stays open.

use bytes::Bytes;
use futures_util::{Stream, StreamExt};

use crate::providers::grok::translate::stream::SseDecoder;

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
}

/// Keeps an event-stream body whole for a translator that takes the whole
/// body at once, and counts its terminal events for [`read_upstream`].
/// Validation is the translator's: this sink only reads what it needs to
/// find the terminal events.
pub(super) struct EventStreamSink {
    bytes: Vec<u8>,
    decoder: SseDecoder,
    rule: TerminalRule,
    completed: bool,
}

impl EventStreamSink {
    pub(super) fn new(rule: TerminalRule) -> Self {
        Self {
            bytes: Vec::new(),
            // The translators decode with the default frame limit; the same
            // limit here means both fail on the same frame.
            decoder: SseDecoder::default(),
            rule,
            completed: false,
        }
    }

    /// The body to translate. After [`ReadEnd::ReadLimit`] a frame still
    /// being received is left out, as the streaming path discards it; the
    /// translators would otherwise reject the body as cut off.
    pub(super) fn into_body(self, end: ReadEnd) -> Vec<u8> {
        let mut bytes = self.bytes;
        if end == ReadEnd::ReadLimit {
            bytes.truncate(complete_frames_len(&bytes));
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
            match self.rule {
                TerminalRule::Responses => {
                    let terminal = serde_json::from_str::<serde_json::Value>(event.data.trim())
                        .ok()
                        .and_then(|value| {
                            value
                                .get("type")
                                .and_then(serde_json::Value::as_str)
                                .map(|kind| RESPONSES_TERMINAL_EVENT_TYPES.contains(&kind))
                        })
                        .unwrap_or(false);
                    self.completed |= terminal;
                }
            }
        }
        Ok(())
    }

    fn terminal_events(&self) -> usize {
        usize::from(self.completed)
    }
}

/// Length of the prefix of `bytes` that ends with a complete event-stream
/// frame: what follows the last blank line is a frame still being received.
/// Lines end with `\n`, `\r` or `\r\n`, as the decoder reads them.
fn complete_frames_len(bytes: &[u8]) -> usize {
    let mut end = 0;
    let mut line_empty = true;
    let mut after_cr = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if after_cr {
            after_cr = false;
            if byte == b'\n' {
                // The second half of a `\r\n` already counted.
                if end == index {
                    end = index + 1;
                }
                continue;
            }
        }
        match byte {
            b'\n' | b'\r' => {
                if line_empty {
                    end = index + 1;
                }
                line_empty = true;
                after_cr = byte == b'\r';
            }
            _ => line_empty = false,
        }
    }
    end
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

    #[test]
    fn complete_frames_len_synthetic_stops_after_the_last_blank_line_for_every_line_ending() {
        let cases: [(&str, usize); 9] = [
            ("", 0),
            ("data: a", 0),
            ("data: a\n", 0),
            ("data: a\n\n", 9),
            ("data: a\n\ndata: b", 9),
            ("data: a\r\n\r\ndata: b\r\n", 11),
            ("data: a\r\rdata: b", 9),
            ("data: a\n\r\n", 10),
            (": comment\n\ndata: a\n\n: partial", 20),
        ];
        for (input, expected) in cases {
            assert_eq!(complete_frames_len(input.as_bytes()), expected, "{input:?}");
        }
    }

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
}
