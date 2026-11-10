use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::accumulate::accumulate_anthropic_sse;
use super::reducer::{Reducer, ReducerEvent};
use super::stream::{
    DecodedFrame, INVALID_STREAM_MESSAGE, SseDecoder, StreamRenderer, error_event,
};
use crate::monitor::MonitorHandle;
use crate::providers::codex::native::NativeResponseOutcome;
use crate::providers::muse::client::MuseError;
use crate::providers::muse::usage::{LoginId, UsageSink};
use crate::traffic::{StreamTrafficCapture, TrafficCapture, sanitize_muse_capture};

/// Rendered SSE chunks buffered towards the client. A slow client makes the
/// reader wait, which in turn stops reading the upstream.
pub const DOWNSTREAM_CAPACITY: usize = 16;

/// How long the reader keeps reading after the content-terminal event, for
/// the allowance frame that follows it. Covers reading and recording.
pub const POST_TERMINAL_READ_LIMIT: Duration = Duration::from_secs(2);

/// How much the reader takes in after the content-terminal event. The
/// allowance frame is a few hundred bytes and is followed only by `[DONE]`;
/// an upstream that keeps sending after it is cut off here even when it
/// never lets the timer run. The limits count bytes and decoded frames, not
/// transport reads, so the frame is kept however finely it is split. Reads
/// that deliver no bytes make no progress and are counted on their own.
const POST_TERMINAL_BYTE_LIMIT: usize = 64 * 1024;
const POST_TERMINAL_FRAME_LIMIT: usize = 256;
const POST_TERMINAL_EMPTY_READ_LIMIT: usize = 64;

const SUMMARY_NAME: &str = "061-muse-stream-summary";

/// Everything the reader task needs besides the upstream body.
pub struct ReaderConfig {
    pub req_id: String,
    pub message_id: String,
    /// The model name echoed to the client.
    pub model: String,
    pub estimated_input_tokens: u64,
    /// The login the request ran under; allowance updates are tagged with it.
    pub login: LoginId,
    pub usage_sink: Arc<dyn UsageSink>,
    /// Credential values the request used (access token, API key). They are
    /// removed from upstream error text and from capture data.
    pub secrets: Vec<String>,
    pub traffic: Option<Arc<TrafficCapture>>,
    pub monitor: Option<MonitorHandle>,
}

/// The single task that owns the upstream response, the decoder, the
/// reducer and the capture of one request, and the receiving side it feeds.
///
/// Dropping the receiver (the client went away) makes the task drop the
/// upstream response. The task is never aborted by this handle: after the
/// client's response has ended it may still be reading the allowance frame.
pub struct MuseReader {
    receiver: mpsc::Receiver<Bytes>,
    traffic: Option<Arc<TrafficCapture>>,
    /// The failure the task reported to the client, if any; set before the
    /// error event is sent, so it is visible once the channel has closed.
    failure: Arc<Mutex<Option<MuseError>>>,
    #[cfg(test)]
    task: JoinHandle<()>,
}

/// Starts the reader task for one upstream response body.
pub fn spawn_reader<S, E>(upstream: S, config: ReaderConfig) -> MuseReader
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let (sender, receiver) = mpsc::channel(DOWNSTREAM_CAPACITY);
    let traffic = config.traffic.clone();
    let failure = Arc::new(Mutex::new(None));
    let task: JoinHandle<()> =
        tokio::spawn(run(Box::pin(upstream), sender, config, failure.clone()));
    #[cfg(not(test))]
    drop(task);
    MuseReader {
        receiver,
        traffic,
        failure,
        #[cfg(test)]
        task,
    }
}

impl MuseReader {
    /// The Anthropic SSE body for a streaming client.
    pub fn into_body(self) -> Body {
        self.body_reporting(None)
    }

    /// [`Self::into_body`] with the SSE response headers. A stream that ends
    /// in an error event is reported through the response's outcome, which
    /// the server reads when the body ends, so the request is recorded as
    /// failed with the same sanitised message the client saw.
    pub fn into_response(self) -> Response {
        let outcome = NativeResponseOutcome::default();
        let mut response = (
            [
                (http::header::CONTENT_TYPE, "text/event-stream"),
                (http::header::CACHE_CONTROL, "no-cache"),
            ],
            self.body_reporting(Some(outcome.clone())),
        )
            .into_response();
        response.extensions_mut().insert(outcome);
        response
    }

    fn body_reporting(self, outcome: Option<NativeResponseOutcome>) -> Body {
        let failure = self.failure;
        let stream = futures_util::stream::unfold(self.receiver, move |mut receiver| {
            let failure = failure.clone();
            let outcome = outcome.clone();
            async move {
                if let Some(bytes) = receiver.recv().await {
                    return Some((Ok::<Bytes, Infallible>(bytes), receiver));
                }
                // The task records its failure before sending the error
                // event, so it is visible once the channel has closed.
                let reported = failure.lock().ok().and_then(|slot| slot.clone());
                if let (Some(outcome), Some(error)) = (outcome, reported) {
                    outcome.fail(error.message_with_status());
                }
                None
            }
        });
        Body::from_stream(stream)
    }

    /// The non-streaming Anthropic message. Returns as soon as the reader
    /// closes the client side, without waiting for the allowance frame. A
    /// stream failure comes back as the same error the streaming client saw.
    pub async fn into_message(mut self) -> Result<Value, MuseError> {
        let mut sse = Vec::new();
        while let Some(chunk) = self.receiver.recv().await {
            sse.extend_from_slice(&chunk);
        }
        let reported = self.failure.lock().ok().and_then(|mut slot| slot.take());
        if let Some(error) = reported {
            return Err(error);
        }
        let message = accumulate_anthropic_sse(&sse)?;
        if let Some(traffic) = self.traffic.as_ref() {
            traffic.write_json("051-downstream-response", &message);
        }
        Ok(message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Completed,
    Failed {
        stage: &'static str,
        kind: &'static str,
    },
    Abandoned(&'static str),
}

/// Finalises the capture exactly once, from `Drop`, whatever ends the task:
/// normal completion, an upstream failure, a client disconnect, or the task
/// being dropped by the runtime. Anything but completion is recorded as such.
struct CaptureGuard {
    capture: Option<StreamTrafficCapture>,
    traffic: Option<Arc<TrafficCapture>>,
    outcome: Outcome,
    bytes: u64,
    chunks: u64,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        let (Some(capture), Some(traffic)) = (self.capture.take(), self.traffic.as_ref()) else {
            return;
        };
        let completion = match self.outcome {
            Outcome::Completed => json!({
                "kind": "stream_completion",
                "bytes": self.bytes,
                "chunks": self.chunks,
            }),
            Outcome::Failed { stage, kind } => {
                traffic.write_json(
                    "060-muse-stream-error",
                    &json!({"stage": stage, "kind": kind, "bytes": self.bytes, "chunks": self.chunks}),
                );
                json!({"kind": "stream_error", "stage": stage, "error": kind, "bytes": self.bytes, "chunks": self.chunks})
            }
            Outcome::Abandoned(reason) => {
                traffic.write_json(
                    "060-muse-stream-abandoned",
                    &json!({"stage": "downstream", "reason": reason, "bytes": self.bytes, "chunks": self.chunks}),
                );
                json!({"kind": "stream_abandoned", "reason": reason, "bytes": self.bytes, "chunks": self.chunks})
            }
        };
        capture.finish_named(traffic, completion, SUMMARY_NAME);
    }
}

/// State shared by the phases before and after the content-terminal event.
struct ReadState<'a> {
    reducer: Reducer,
    renderer: StreamRenderer,
    guard: CaptureGuard,
    secrets: &'a [&'a str],
    login: LoginId,
    usage_sink: &'a dyn UsageSink,
    monitor: Option<&'a MonitorHandle>,
    req_id: &'a str,
    failure: &'a Mutex<Option<MuseError>>,
}

/// What one decoded frame did to the stream before its terminal event.
enum Step {
    Continue,
    Finished,
    /// A `response.failed` or `error` event, with its sanitised message.
    Failed(String),
    Invalid,
    Done,
}

impl ReadState<'_> {
    fn capture_upstream(&mut self, event: Option<&str>, value: &Value) {
        if let Some(capture) = self.guard.capture.as_mut() {
            capture.upstream_event(event, &sanitize_muse_capture(value, self.secrets));
        }
    }

    fn capture_downstream(&mut self, rendered: &[u8]) {
        let Some(capture) = self.guard.capture.as_mut() else {
            return;
        };
        for event in crate::anthropic::sse::parse_sse_events(rendered) {
            if let Ok(data) = serde_json::from_str::<Value>(&event.data) {
                capture.downstream_event(
                    event.event.as_deref().unwrap_or("message"),
                    sanitize_muse_capture(&data, self.secrets),
                );
            }
        }
    }

    fn malformed(&mut self, stage: &str, kind: &str) {
        if let Some(capture) = self.guard.capture.as_mut() {
            capture.malformed(stage, kind);
        }
    }

    fn record_usage(&self, events: &[ReducerEvent]) {
        for event in events {
            if let ReducerEvent::SubscriptionUsage(usage) = event {
                self.usage_sink.record_usage(self.login, usage.clone());
            }
        }
    }

    /// Handles one frame before the terminal event, appending client output
    /// to `out`.
    fn step(&mut self, frame: DecodedFrame, out: &mut Vec<u8>) -> Step {
        let event = match frame {
            DecodedFrame::Event(event) => event,
            DecodedFrame::Malformed(kind) => {
                self.malformed("decoder", kind);
                return Step::Continue;
            }
        };
        if event.event.is_none() && event.data == "[DONE]" {
            return Step::Done;
        }
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            self.malformed("json", "malformed_event");
            return Step::Continue;
        };
        self.capture_upstream(event.event.as_deref(), &value);
        let events = match self.reducer.push(&value) {
            Ok(events) => events,
            Err(_) => return Step::Invalid,
        };
        self.record_usage(&events);
        if let Some(ReducerEvent::Finish {
            input_tokens,
            output_tokens,
            unexpected_incomplete_reason,
            ..
        }) = events
            .iter()
            .find(|event| matches!(event, ReducerEvent::Finish { .. }))
        {
            if let Some(monitor) = self.monitor {
                monitor.usage_updated(self.req_id, *input_tokens, Some(*output_tokens));
            }
            if let Some(reason) = unexpected_incomplete_reason {
                let mut fields = Map::new();
                fields.insert("reqId".into(), json!(self.req_id));
                fields.insert("reason".into(), json!(reason));
                crate::logging::create_logger("muse").warn("muse_stream_incomplete", Some(fields));
            }
        }
        let rendered = self.renderer.render(&events);
        self.capture_downstream(&rendered);
        out.extend(rendered);
        if self.reducer.finished() {
            return Step::Finished;
        }
        match events.into_iter().find_map(|event| match event {
            ReducerEvent::Failed(message) => Some(message),
            _ => None,
        }) {
            Some(message) => Step::Failed(message),
            None => Step::Continue,
        }
    }

    /// Handles one frame after the terminal event. Returns true at `[DONE]`.
    fn step_after_terminal(&mut self, frame: DecodedFrame) -> bool {
        let event = match frame {
            DecodedFrame::Event(event) => event,
            DecodedFrame::Malformed(kind) => {
                self.malformed("decoder", kind);
                return false;
            }
        };
        if event.event.is_none() && event.data == "[DONE]" {
            return true;
        }
        let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
            self.malformed("json", "malformed_event");
            return false;
        };
        self.capture_upstream(event.event.as_deref(), &value);
        if let Ok(events) = self.reducer.push(&value) {
            self.record_usage(&events);
        }
        false
    }

    /// Ends the stream with `error`: records it for the non-streaming path and
    /// appends the client's error event to `out`.
    fn stream_error(
        &mut self,
        stage: &'static str,
        kind: &'static str,
        error: MuseError,
        out: &mut Vec<u8>,
    ) {
        self.fail(stage, kind);
        let rendered = error_event(&error);
        self.capture_downstream(&rendered);
        out.extend(rendered);
        if let Ok(mut slot) = self.failure.lock() {
            *slot = Some(error);
        }
    }

    fn fail(&mut self, stage: &'static str, kind: &'static str) {
        self.guard.outcome = Outcome::Failed { stage, kind };
        self.malformed(stage, kind);
        let mut fields = Map::new();
        fields.insert("reqId".into(), json!(self.req_id));
        fields.insert("stage".into(), json!(stage));
        fields.insert("kind".into(), json!(kind));
        fields.insert("bytes".into(), json!(self.guard.bytes));
        fields.insert("chunks".into(), json!(self.guard.chunks));
        crate::logging::create_logger("muse").warn("muse_stream_failed", Some(fields));
    }
}

async fn run<S, E>(
    mut upstream: Pin<Box<S>>,
    sender: mpsc::Sender<Bytes>,
    config: ReaderConfig,
    failure: Arc<Mutex<Option<MuseError>>>,
) where
    S: Stream<Item = Result<Bytes, E>> + Send,
{
    let ReaderConfig {
        req_id,
        message_id,
        model,
        estimated_input_tokens,
        login,
        usage_sink,
        secrets,
        traffic,
        monitor,
    } = config;
    let secret_refs: Vec<&str> = secrets.iter().map(String::as_str).collect();
    let mut state = ReadState {
        reducer: Reducer::new(secrets.clone()),
        renderer: StreamRenderer::new(message_id, model, estimated_input_tokens),
        guard: CaptureGuard {
            capture: traffic.as_ref().map(|traffic| traffic.stream_capture()),
            traffic,
            outcome: Outcome::Abandoned("reader_dropped"),
            bytes: 0,
            chunks: 0,
        },
        secrets: &secret_refs,
        login,
        usage_sink: usage_sink.as_ref(),
        monitor: monitor.as_ref(),
        req_id: &req_id,
        failure: &failure,
    };
    let mut decoder = SseDecoder::default();
    loop {
        // An upstream read has no timeout, so a client that leaves while the
        // upstream is silent must end the wait here.
        let next = tokio::select! {
            biased;
            () = sender.closed() => {
                state.guard.outcome = Outcome::Abandoned("client_disconnect");
                return;
            }
            next = upstream.next() => next,
        };
        let chunk = match next {
            Some(Ok(chunk)) => chunk,
            Some(Err(_)) | None => {
                let (stage, kind) = if next.is_some() {
                    ("transport", "upstream_stream")
                } else {
                    ("decoder", "incomplete_stream")
                };
                let mut out = Vec::new();
                state.stream_error(stage, kind, MuseError::stream_ended(), &mut out);
                let _ = sender.send(out.into()).await;
                return;
            }
        };
        if state.guard.bytes == 0
            && let Some(monitor) = state.monitor
        {
            monitor.generation_started(state.req_id);
        }
        state.guard.bytes = state.guard.bytes.saturating_add(chunk.len() as u64);
        state.guard.chunks = state.guard.chunks.saturating_add(1);
        if let Some(monitor) = state.monitor {
            monitor.stream_progress(state.req_id, chunk.len() as u64, 1, None, None);
        }
        let mut frames = decoder.push(&chunk).into_iter();
        let mut out = Vec::new();
        let mut step = Step::Continue;
        for frame in frames.by_ref() {
            step = state.step(frame, &mut out);
            if !matches!(step, Step::Continue) {
                break;
            }
        }
        match &step {
            Step::Continue | Step::Finished => {}
            Step::Failed(message) => state.stream_error(
                "upstream",
                "response_failed",
                MuseError::stream_failed(message.clone()),
                &mut out,
            ),
            Step::Invalid => state.stream_error(
                "reducer",
                "invalid_event",
                MuseError::stream_failed(INVALID_STREAM_MESSAGE),
                &mut out,
            ),
            Step::Done => state.stream_error(
                "decoder",
                "incomplete_stream",
                MuseError::stream_ended(),
                &mut out,
            ),
        }
        if !out.is_empty() && sender.send(out.into()).await.is_err() {
            if matches!(step, Step::Continue | Step::Finished) {
                state.guard.outcome = Outcome::Abandoned("downstream_send_failed");
            }
            return;
        }
        match step {
            Step::Continue => continue,
            Step::Finished => {}
            Step::Failed(_) | Step::Invalid | Step::Done => return,
        }
        // The client's response ends here; the allowance frame may still be
        // on its way, in this chunk or a later one.
        drop(sender);
        state.guard.outcome = Outcome::Completed;
        // `timeout_at` alone is not enough: it returns a ready read without
        // looking at the clock, so an upstream that is always ready would
        // never be stopped by it. The deadline is also checked between
        // frames, and the reader yields after every chunk.
        let deadline = tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT;
        let expired = || tokio::time::Instant::now() >= deadline;
        let mut frames_left = POST_TERMINAL_FRAME_LIMIT;
        for frame in frames {
            if expired() || frames_left == 0 || state.step_after_terminal(frame) {
                return;
            }
            frames_left -= 1;
        }
        let mut bytes = 0usize;
        let mut empty_reads = 0usize;
        loop {
            if expired() {
                return;
            }
            let Ok(Some(Ok(chunk))) = tokio::time::timeout_at(deadline, upstream.next()).await
            else {
                return;
            };
            if chunk.is_empty() {
                empty_reads += 1;
                if empty_reads > POST_TERMINAL_EMPTY_READ_LIMIT {
                    return;
                }
            }
            bytes = bytes.saturating_add(chunk.len());
            if bytes > POST_TERMINAL_BYTE_LIMIT {
                return;
            }
            for frame in decoder.push(&chunk) {
                if expired() || frames_left == 0 || state.step_after_terminal(frame) {
                    return;
                }
                frames_left -= 1;
            }
            tokio::task::yield_now().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::*;
    use super::*;
    use crate::anthropic::sse::parse_sse_events;
    use crate::providers::muse::usage::{SubscriptionUsage, UsageWindow};
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<(LoginId, SubscriptionUsage)>>);

    impl UsageSink for RecordingSink {
        fn record_usage(&self, login: LoginId, usage: SubscriptionUsage) {
            self.0.lock().expect("sink lock").push((login, usage));
        }
    }

    impl RecordingSink {
        fn recorded(&self) -> Vec<(LoginId, SubscriptionUsage)> {
            self.0.lock().expect("sink lock").clone()
        }
    }

    const LOGIN: LoginId = LoginId([7; 32]);
    const SECRET: &str = "synthetic-api-key-0123456789abcdefghijklmnopqrstuv";

    fn observed_usage() -> SubscriptionUsage {
        SubscriptionUsage {
            window: Some(UsageWindow {
                used_percent: 1,
                resets_at: 1_794_291_326,
                window_duration_mins: Some(300),
            }),
            weekly: Some(UsageWindow {
                used_percent: 0,
                resets_at: 1_794_787_200,
                window_duration_mins: None,
            }),
        }
    }

    struct Harness {
        sink: Arc<RecordingSink>,
        traffic: Arc<TrafficCapture>,
        _dir: tempfile::TempDir,
    }

    impl Harness {
        fn new() -> Self {
            let dir = tempfile::TempDir::new().expect("tempdir");
            Self {
                sink: Arc::new(RecordingSink::default()),
                traffic: Arc::new(crate::traffic::test_capture(dir.path().join("traffic"))),
                _dir: dir,
            }
        }

        fn spawn<S>(&self, upstream: S) -> MuseReader
        where
            S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
        {
            spawn_reader(
                upstream,
                ReaderConfig {
                    req_id: "req_1".into(),
                    message_id: "msg_1".into(),
                    model: "muse-spark-1.3".into(),
                    estimated_input_tokens: 9,
                    login: LOGIN,
                    usage_sink: self.sink.clone(),
                    secrets: vec![SECRET.into()],
                    traffic: Some(self.traffic.clone()),
                    monitor: None,
                },
            )
        }

        /// Every captured file's contents, concatenated.
        fn captured(&self) -> String {
            let mut out = String::new();
            let root = self.traffic.root().to_path_buf();
            for dir in [root.clone(), root.join("events")] {
                let Ok(entries) = std::fs::read_dir(dir) else {
                    continue;
                };
                for entry in entries {
                    let path = entry.expect("entry").path();
                    if path.is_file() {
                        out.push_str(&std::fs::read_to_string(path).unwrap_or_default());
                        out.push('\n');
                    }
                }
            }
            out
        }

        fn summaries(&self) -> Vec<Value> {
            std::fs::read_dir(self.traffic.root())
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.to_string_lossy().contains(SUMMARY_NAME))
                        .map(|path| {
                            serde_json::from_str(&std::fs::read_to_string(path).expect("read"))
                                .expect("summary JSON")
                        })
                        .collect()
                })
                .unwrap_or_default()
        }

        fn completion_kind(&self) -> String {
            let summaries = self.summaries();
            assert_eq!(summaries.len(), 1, "capture must finish exactly once");
            summaries[0]["completion"]["kind"]
                .as_str()
                .expect("kind")
                .to_string()
        }
    }

    fn chunk(text: impl Into<String>) -> Result<Bytes, std::io::Error> {
        Ok(Bytes::from(text.into()))
    }

    /// An upstream the test feeds chunk by chunk; it stays silent until the
    /// test sends or drops the sender.
    fn controlled() -> (
        mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
        impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    ) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let stream = futures_util::stream::unfold(receiver, |mut receiver| async move {
            receiver.recv().await.map(|item| (item, receiver))
        });
        (sender, stream)
    }

    async fn drain(reader: &mut MuseReader) -> Vec<Value> {
        let mut bytes = Vec::new();
        while let Some(chunk) = reader.receiver.recv().await {
            bytes.extend_from_slice(&chunk);
        }
        events(&bytes)
    }

    fn events(bytes: &[u8]) -> Vec<Value> {
        parse_sse_events(bytes)
            .into_iter()
            .map(|event| serde_json::from_str(&event.data).expect("rendered JSON"))
            .collect()
    }

    fn types(events: &[Value]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| event["type"].as_str())
            .collect()
    }

    #[tokio::test]
    async fn allowance_frame_in_the_terminal_chunk_is_recorded_and_hidden() {
        let harness = Harness::new();
        let stream = join(TEXT_ONLY) + &usage_frame() + DONE;
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(stream)]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(
            types(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert!(
            !serde_json::to_string(&events)
                .expect("json")
                .contains("used_percent")
        );
        assert_eq!(harness.sink.recorded(), vec![(LOGIN, observed_usage())]);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test]
    async fn client_side_ends_before_a_later_allowance_frame_arrives() {
        let harness = Harness::new();
        let (upstream, stream) = controlled();
        let mut reader = harness.spawn(stream);
        upstream.send(chunk(join(TEXT_ONLY))).expect("send");
        let events = drain(&mut reader).await;
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        assert!(harness.sink.recorded().is_empty(), "frame not sent yet");
        upstream.send(chunk(usage_frame())).expect("send");
        upstream.send(chunk(DONE)).expect("send");
        reader.task.await.expect("task");
        assert_eq!(harness.sink.recorded(), vec![(LOGIN, observed_usage())]);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test(start_paused = true)]
    async fn a_missing_allowance_frame_stops_the_reader_after_the_limit() {
        let harness = Harness::new();
        let upstream = futures_util::stream::iter([chunk(join(TEXT_ONLY))])
            .chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let mut reader = harness.spawn(upstream);
        let events = drain(&mut reader).await;
        assert_eq!(started.elapsed(), Duration::ZERO, "client side is not held");
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        reader.task.await.expect("task");
        assert_eq!(started.elapsed(), POST_TERMINAL_READ_LIMIT);
        assert!(harness.sink.recorded().is_empty());
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    /// Runs the reader on its own thread so that, should it never yield, the
    /// bounded wait fails the test instead of hanging it.
    fn reader_ends_on_its_own<S>(upstream: impl FnOnce() -> S + Send + 'static) -> Vec<String>
    where
        S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            let delivered = runtime.block_on(async move {
                let harness = Harness::new();
                let mut reader = harness.spawn(upstream());
                let events = drain(&mut reader).await;
                reader.task.await.expect("task");
                assert_eq!(harness.completion_kind(), "stream_completion");
                types(&events).iter().map(|kind| kind.to_string()).collect()
            });
            let _ = sender.send(delivered);
        });
        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("reader ends within its post-terminal bound")
    }

    #[test]
    fn synthetic_always_ready_trailing_frames_end_the_reader() {
        let delivered = reader_ends_on_its_own(|| {
            futures_util::stream::iter([chunk(join(TEXT_ONLY))]).chain(
                futures_util::stream::repeat_with(|| {
                    chunk(frame(
                        "response.output_text.delta",
                        r#"{"type": "response.output_text.delta", "delta": "late"}"#,
                    ))
                }),
            )
        });
        assert_eq!(delivered.last().map(String::as_str), Some("message_stop"));
    }

    #[test]
    fn synthetic_always_ready_empty_chunks_end_the_reader() {
        let delivered = reader_ends_on_its_own(|| {
            futures_util::stream::iter([chunk(join(TEXT_ONLY))])
                .chain(futures_util::stream::repeat_with(|| Ok(Bytes::new())))
        });
        assert_eq!(delivered.last().map(String::as_str), Some("message_stop"));
    }

    #[tokio::test]
    async fn synthetic_allowance_frame_split_into_single_bytes_is_recorded() {
        let harness = Harness::new();
        let trailing: Vec<_> = (usage_frame() + DONE)
            .bytes()
            .map(|byte| Ok(Bytes::copy_from_slice(&[byte])))
            .collect();
        assert!(
            trailing.len() > 64,
            "more reads than a per-read limit allows"
        );
        let upstream = futures_util::stream::iter([chunk(join(TEXT_ONLY))])
            .chain(futures_util::stream::iter(trailing));
        let mut reader = harness.spawn(upstream);
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        assert_eq!(harness.sink.recorded(), vec![(LOGIN, observed_usage())]);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test]
    async fn synthetic_many_harmless_frames_before_the_allowance_frame_keep_it() {
        let harness = Harness::new();
        let harmless = (0..100).map(|_| {
            chunk(frame(
                "response.output_text.delta",
                r#"{"type": "response.output_text.delta", "delta": "late"}"#,
            ))
        });
        let upstream = futures_util::stream::iter([chunk(join(TEXT_ONLY))])
            .chain(futures_util::stream::iter(harmless))
            .chain(futures_util::stream::iter([
                chunk(usage_frame()),
                chunk(DONE),
            ]));
        let mut reader = harness.spawn(upstream);
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        assert!(!types(&events).contains(&"error"));
        assert_eq!(harness.sink.recorded(), vec![(LOGIN, observed_usage())]);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test(start_paused = true)]
    async fn done_after_the_terminal_ends_the_reader_without_waiting() {
        let harness = Harness::new();
        let upstream = futures_util::stream::iter([chunk(join(TEXT_ONLY)), chunk(DONE)])
            .chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let mut reader = harness.spawn(upstream);
        drain(&mut reader).await;
        reader.task.await.expect("task");
        assert!(started.elapsed() < POST_TERMINAL_READ_LIMIT);
        assert!(harness.sink.recorded().is_empty());
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test]
    async fn synthetic_broken_frames_after_the_terminal_keep_the_response_successful() {
        let harness = Harness::new();
        let after = "data: {not json\n\n".to_string()
            + "data: \u{0}\u{1}\n\n"
            + &frame(
                "response.output_text.delta",
                r#"{"type": "response.output_text.delta"}"#,
            )
            + &usage_frame()
            + DONE;
        let upstream = futures_util::stream::iter([chunk(join(TEXT_ONLY)), Ok(Bytes::from(after))]);
        let mut reader = harness.spawn(upstream);
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        assert!(!types(&events).contains(&"error"));
        assert_eq!(harness.sink.recorded(), vec![(LOGIN, observed_usage())]);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test]
    async fn eof_after_the_terminal_is_success() {
        let harness = Harness::new();
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(join(TEXT_ONLY))]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test]
    async fn eof_in_the_middle_of_text_is_an_error_event() {
        let harness = Harness::new();
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(join(&TEXT_ONLY[..7]))]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(
            types(&events),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "error"
            ]
        );
        assert_eq!(
            events.last().expect("error")["error"],
            json!({"type": "api_error", "message": "Muse stream ended before completion (upstream HTTP 200)"})
        );
        assert_eq!(harness.completion_kind(), "stream_error");
    }

    #[tokio::test]
    async fn eof_in_the_middle_of_tool_arguments_is_an_error_event() {
        let harness = Harness::new();
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(join(
            &FUNCTION_CALL[..2],
        ))]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"error"));
        assert_eq!(harness.completion_kind(), "stream_error");
    }

    #[tokio::test]
    async fn synthetic_transport_error_before_the_terminal_is_an_error_event() {
        let harness = Harness::new();
        let upstream = futures_util::stream::iter([
            chunk(join(&TEXT_ONLY[..7])),
            Err(std::io::Error::other("reset")),
        ]);
        let mut reader = harness.spawn(upstream);
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"error"));
        assert_eq!(harness.completion_kind(), "stream_error");
    }

    #[tokio::test]
    async fn synthetic_done_before_the_terminal_is_an_error_event() {
        let harness = Harness::new();
        let upstream = futures_util::stream::iter([chunk(join(&TEXT_ONLY[..7]) + DONE)])
            .chain(futures_util::stream::pending());
        let mut reader = harness.spawn(upstream);
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"error"));
        assert_eq!(harness.completion_kind(), "stream_error");
    }

    #[tokio::test]
    async fn synthetic_unparseable_data_lines_before_the_terminal_are_ignored() {
        let harness = Harness::new();
        let stream = join(&TEXT_ONLY[..6]) + "data: {\"truncated\n\n" + &join(&TEXT_ONLY[6..]);
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(stream)]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(types(&events).last(), Some(&"message_stop"));
        assert_eq!(harness.completion_kind(), "stream_completion");
        assert!(harness.captured().contains("malformed_event"));
    }

    #[tokio::test]
    async fn synthetic_response_failed_with_a_secret_is_sanitised_everywhere() {
        let harness = Harness::new();
        let failed = frame(
            "response.failed",
            &json!({
                "type": "response.failed",
                "response": {"status": "failed", "error": {
                    "code": format!("bad {SECRET}"),
                    "param": "owner@example.com",
                    "message": format!("key {SECRET} of owner@example.com was refused")
                }}
            })
            .to_string(),
        );
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(
            join(&TEXT_ONLY[..2]) + &failed,
        )]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        assert_eq!(
            events,
            vec![json!({"type": "error", "error": {"type": "api_error",
                "message": "Muse stream failed: key [redacted] of [redacted] was refused (upstream HTTP 200)"}})]
        );
        let captured = harness.captured();
        assert!(!captured.contains(SECRET), "secret leaked to capture");
        assert!(
            !captured.contains("owner@example.com"),
            "e-mail leaked to capture"
        );
        assert!(captured.contains("Muse stream failed: key [redacted]"));
        assert_eq!(harness.completion_kind(), "stream_error");
    }

    #[tokio::test]
    async fn non_streaming_message_is_assembled_before_the_allowance_frame() {
        let harness = Harness::new();
        let (upstream, stream) = controlled();
        let reader = harness.spawn(stream);
        upstream.send(chunk(join(TEXT_ONLY))).expect("send");
        let task = tokio::spawn(reader.into_message());
        let message = task.await.expect("join").expect("message");
        assert_eq!(message["content"], json!([{"type": "text", "text": "ok"}]));
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(
            message["usage"],
            json!({"input_tokens": 14, "output_tokens": 307})
        );
        assert!(harness.sink.recorded().is_empty());
        upstream.send(chunk(usage_frame())).expect("send");
        drop(upstream);
        tokio::time::timeout(Duration::from_secs(5), async {
            while harness.sink.recorded().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("allowance recorded");
        assert!(harness.captured().contains("\"stop_reason\": \"end_turn\""));
    }

    #[tokio::test]
    async fn non_streaming_with_the_allowance_frame_in_the_terminal_chunk() {
        let harness = Harness::new();
        let stream = join(TEXT_ONLY) + &usage_frame() + DONE;
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(stream)]));
        let task = std::mem::replace(&mut reader.task, tokio::spawn(async {}));
        let message = reader.into_message().await.expect("message");
        task.await.expect("task");
        assert_eq!(message["content"], json!([{"type": "text", "text": "ok"}]));
        assert_eq!(harness.sink.recorded(), vec![(LOGIN, observed_usage())]);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test(start_paused = true)]
    async fn non_streaming_reply_does_not_wait_for_a_missing_allowance_frame() {
        let harness = Harness::new();
        let upstream = futures_util::stream::iter([chunk(join(TEXT_ONLY))])
            .chain(futures_util::stream::pending());
        let started = tokio::time::Instant::now();
        let mut reader = harness.spawn(upstream);
        let task = std::mem::replace(&mut reader.task, tokio::spawn(async {}));
        let message = reader.into_message().await.expect("message");
        assert_eq!(started.elapsed(), Duration::ZERO);
        assert_eq!(message["stop_reason"], "end_turn");
        task.await.expect("task");
        assert_eq!(started.elapsed(), POST_TERMINAL_READ_LIMIT);
        assert_eq!(harness.completion_kind(), "stream_completion");
    }

    #[tokio::test]
    async fn synthetic_signed_reasoning_reaches_the_client_but_not_the_capture() {
        let harness = Harness::new();
        let blob = "gAAAA-synthetic-encrypted-reasoning-blob";
        let stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0)
            + &join(SUMMARY_PART)
            + &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, blob)
            + &synthetic_completed(1, 1);
        for streaming in [true, false] {
            let mut reader = harness.spawn(futures_util::stream::iter([chunk(stream.clone())]));
            let task = std::mem::replace(&mut reader.task, tokio::spawn(async {}));
            let delivered = if streaming {
                serde_json::to_string(&drain(&mut reader).await).expect("json")
            } else {
                reader.into_message().await.expect("message").to_string()
            };
            task.await.expect("task");
            assert!(delivered.contains("ccp:muse:v1:"), "{delivered}");
            assert!(delivered.contains(blob), "{delivered}");
        }
        let captured = harness.captured();
        assert!(!captured.contains("ccp:muse:v1:"), "{captured}");
        assert!(!captured.contains(blob), "{captured}");
        assert!(captured.contains(SUMMARY_TEXT));
    }

    #[tokio::test]
    async fn non_streaming_failure_carries_the_error_message() {
        let harness = Harness::new();
        let reader = harness.spawn(futures_util::stream::iter([chunk(join(&TEXT_ONLY[..7]))]));
        assert_eq!(reader.into_message().await, Err(MuseError::stream_ended()));

        let failed = frame(
            "response.failed",
            &json!({"type": "response.failed", "response": {"error": {
                "message": format!("key {SECRET} refused")
            }}})
            .to_string(),
        );
        let reader = harness.spawn(futures_util::stream::iter([chunk(failed)]));
        let error = reader.into_message().await.expect_err("failed stream");
        assert_eq!(
            error,
            MuseError::stream_failed("Muse stream failed: key [redacted] refused")
        );
        let provider = error.into_provider_error();
        assert_eq!(provider.status, http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            provider.message,
            "Muse stream failed: key [redacted] refused (upstream HTTP 200)"
        );
    }

    #[tokio::test]
    async fn synthetic_half_written_call_under_another_incomplete_reason_fails_both_modes() {
        let harness = Harness::new();
        let stream = join(&FUNCTION_CALL[..1])
            + &frame(
                "response.function_call_arguments.delta",
                r#"{"delta": "{\"city\":", "item_id": "fc_01a1c315c9c87310a50529eaae77c51e", "output_index": 2, "type": "response.function_call_arguments.delta"}"#,
            )
            + &frame(
                "response.incomplete",
                &INCOMPLETE.replace("max_output_tokens\"}", "content_filter\"}"),
            );
        let mut reader = harness.spawn(futures_util::stream::iter([chunk(stream.clone())]));
        let events = drain(&mut reader).await;
        reader.task.await.expect("task");
        let kinds = types(&events);
        assert!(!kinds.contains(&"content_block_stop"), "{kinds:?}");
        assert!(!kinds.contains(&"message_stop"), "{kinds:?}");
        assert_eq!(
            events.last().expect("error")["error"],
            json!({"type": "api_error", "message": "Muse stream is invalid (upstream HTTP 200)"})
        );

        let reader = harness.spawn(futures_util::stream::iter([chunk(stream)]));
        assert_eq!(
            reader.into_message().await,
            Err(MuseError::stream_failed(INVALID_STREAM_MESSAGE))
        );
    }

    #[tokio::test]
    async fn streaming_response_outcome_carries_the_sanitised_failure_only_on_failure() {
        use http_body_util::BodyExt;
        let harness = Harness::new();
        let failed = frame(
            "response.failed",
            &json!({"type": "response.failed", "response": {"error": {
                "message": format!("key {SECRET} of owner@example.com was refused")
            }}})
            .to_string(),
        );
        for (stream, expected) in [
            (join(TEXT_ONLY) + DONE, None),
            (
                join(&TEXT_ONLY[..2]) + &failed,
                Some(
                    "Muse stream failed: key [redacted] of [redacted] was refused (upstream HTTP 200)",
                ),
            ),
            (
                join(&TEXT_ONLY[..7]),
                Some("Muse stream ended before completion (upstream HTTP 200)"),
            ),
        ] {
            let response = harness
                .spawn(futures_util::stream::iter([chunk(stream)]))
                .into_response();
            let outcome = response
                .extensions()
                .get::<NativeResponseOutcome>()
                .cloned()
                .expect("outcome");
            response.into_body().collect().await.expect("body");
            assert_eq!(outcome.failure().as_deref(), expected);
        }
    }

    #[tokio::test]
    async fn client_disconnect_while_the_upstream_is_silent_ends_the_reader() {
        let harness = Harness::new();
        let (upstream, stream) = controlled();
        let mut reader = harness.spawn(stream);
        upstream.send(chunk(join(&TEXT_ONLY[..7]))).expect("send");
        assert!(reader.receiver.recv().await.is_some());
        let MuseReader { receiver, task, .. } = reader;
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("reader ends without upstream activity")
            .expect("task");
        assert!(upstream.is_closed(), "upstream response was dropped");
        assert_eq!(harness.completion_kind(), "stream_abandoned");
    }

    #[tokio::test]
    async fn client_disconnect_noticed_by_a_failed_send_ends_the_reader() {
        let harness = Harness::new();
        let (upstream, stream) = controlled();
        let reader = harness.spawn(stream);
        upstream.send(chunk(join(&TEXT_ONLY[4..6]))).expect("send");
        // More output chunks than the channel holds, so the reader ends up
        // waiting in a send that the disconnect then fails.
        let delta = TEXT_ONLY[6].1;
        for _ in 0..DOWNSTREAM_CAPACITY + 4 {
            upstream
                .send(chunk(frame("response.output_text.delta", delta)))
                .expect("send");
        }
        // Once the channel is full the reader's next action is a send.
        tokio::time::timeout(Duration::from_secs(5), async {
            while reader.receiver.len() < DOWNSTREAM_CAPACITY {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("channel fills");
        let MuseReader { receiver, task, .. } = reader;
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("reader ends")
            .expect("task");
        let summaries = harness.summaries();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0]["completion"]["kind"], "stream_abandoned");
        assert_eq!(
            summaries[0]["completion"]["reason"],
            "downstream_send_failed"
        );
    }

    #[tokio::test]
    async fn an_aborted_reader_still_finishes_the_capture_once() {
        let harness = Harness::new();
        let (upstream, stream) = controlled();
        let mut reader = harness.spawn(stream);
        upstream.send(chunk(join(&TEXT_ONLY[..7]))).expect("send");
        assert!(reader.receiver.recv().await.is_some());
        reader.task.abort();
        assert!(reader.task.await.expect_err("aborted").is_cancelled());
        assert_eq!(harness.completion_kind(), "stream_abandoned");
    }
}
