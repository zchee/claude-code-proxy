//! Mistral on Vertex AI: Anthropic Messages requests sent in Mistral's
//! Chat-Completions-shaped format to
//! `publishers/mistralai/models/<id>:streamRawPredict`.
//!
//! The request is built by the OpenCode Go Chat Completions translator and
//! then edited for Mistral; the response stream goes through the same
//! translator's wiring with Vertex labels, the strict terminal policy and
//! the Vertex sanitiser. The upstream is always asked for a stream; a
//! non-streaming client gets the stream accumulated into one message.

use axum::body::Body;
use bytes::Bytes;
use futures_util::Stream;
use serde_json::Value;

use crate::anthropic::schema::MessagesRequest;
use crate::provider::RequestContext;
use crate::providers::opencode::chat::{
    self as translator, EffortPolicy, ReasoningReplay, ReceivePolicy, RedactedWiring,
    RequestPolicy, StreamLabels, StreamWiring, TerminalPolicy,
};

use super::BufferedCapture;
use super::buffered::{BufferedFailure, EventStreamSink, read_upstream};
use super::client::{VertexError, VertexTarget};
use super::models::ResolvedModel;
use super::translate::stream::Sanitizer;

/// Names in error messages and capture files: `Vertex AI Mistral stream is
/// invalid` and `061-vertex-mistral-stream-summary`, which does not collide
/// with Gemini's `061-vertex-stream-summary`.
const LABELS: StreamLabels = StreamLabels {
    provider: "Vertex AI Mistral",
    capture_stem: "vertex-mistral",
};

/// The translator neither sets `reasoning_effort` nor sends reasoning back.
///
/// No effort control is documented for Mistral on Vertex AI. The reasoning
/// of earlier turns is not replayed as `reasoning_content`: Google documents
/// that field only for the managed open models on `endpoints/openapi`, not
/// for Mistral's payload on `rawPredict`; the statement that unsupported
/// parameters are ignored belongs to Google's OpenAI-compatible endpoint,
/// not to this route, so an unknown message field may be refused; and the
/// thinking text in a history may have been written by another provider.
const REQUEST_POLICY: RequestPolicy = RequestPolicy {
    effort: EffortPolicy::Untouched,
    reasoning_replay: ReasoningReplay::Never,
};

/// The Mistral request for `body`: the Chat Completions translation without
/// `stream_options` and `reasoning_effort`, which Google does not document
/// for Mistral, with `stream: true` and the bare model ID. Google's Mistral
/// page says to leave a `@` version out of the body's `model`; the URL
/// keeps it.
///
/// # Errors
///
/// The request cannot be expressed as Chat Completions.
pub(super) fn prepare_request(
    body: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<Value> {
    let bare_id = model
        .upstream_model
        .split_once('@')
        .map_or(model.upstream_model.as_str(), |(id, _)| id);
    let translated = translator::prepare_request_with(body, bare_id, REQUEST_POLICY)?;
    let mut request = serde_json::to_value(translated)?;
    let Some(object) = request.as_object_mut() else {
        anyhow::bail!("the Mistral request is not a JSON object");
    };
    object.remove("stream_options");
    object.remove("reasoning_effort");
    object.insert("model".to_string(), Value::from(bare_id));
    object.insert("stream".to_string(), Value::Bool(true));
    Ok(request)
}

/// `{base}/v1/projects/{project}/locations/{location}/publishers/mistralai/models/{model}:streamRawPredict`.
///
/// # Errors
///
/// A project, location or model that cannot be placed in the path.
pub(super) fn url(
    target: &VertexTarget,
    model: &ResolvedModel,
) -> Result<reqwest::Url, VertexError> {
    target.model_method_url(model, "streamRawPredict")
}

/// Vertex labels, the strict terminal policy and `sanitize` for upstream
/// prose in captures and buffered errors.
fn wiring(sanitize: Sanitizer) -> StreamWiring {
    StreamWiring {
        labels: LABELS,
        receive: ReceivePolicy {
            terminal: TerminalPolicy::Strict,
            sanitize: Some(sanitize),
        },
    }
}

/// The Anthropic SSE body translated from the upstream Mistral stream.
/// `model` is the name the client asked for. `sanitize` cleans upstream
/// prose; `redact` is applied to every string of every event the traffic
/// capture records.
pub(super) fn stream_body_redacted<S, E>(
    upstream: S,
    message_id: String,
    model: String,
    ctx: &RequestContext,
    sanitize: Sanitizer,
    redact: Sanitizer,
) -> Body
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    translator::stream_body_redacted(
        upstream,
        message_id,
        model,
        ctx.monitor.clone(),
        ctx.req_id.clone(),
        ctx.traffic.clone(),
        RedactedWiring {
            wiring: wiring(sanitize),
            redact,
        },
    )
}

/// [`stream_body_redacted`] with a redactor that changes nothing, for tests
/// whose context has no traffic capture.
#[cfg(test)]
pub(super) fn stream_body<S, E>(
    upstream: S,
    message_id: String,
    model: String,
    ctx: &RequestContext,
    sanitize: Sanitizer,
) -> Body
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let redact: Sanitizer = std::sync::Arc::new(str::to_string);
    stream_body_redacted(upstream, message_id, model, ctx, sanitize, redact)
}

/// Reads the whole upstream Mistral stream with the shared buffered limits
/// and accumulates it into one Anthropic message. A `finish_reason` without
/// `[DONE]` completes the message. The chunks are checked as they arrive
/// ([`translator::StrictChatObserver`]): a frame the streaming path
/// rejects, such as an upstream error event, ends the read at once, even
/// while the upstream stays open. Silence before the terminal event is not
/// cut.
///
/// # Errors
///
/// The read fails ([`BufferedFailure::Read`]), or the body is not a
/// complete, valid Chat Completions stream ([`BufferedFailure::Translate`],
/// already sanitised).
pub(super) async fn read_response<S, E>(
    upstream: S,
    message_id: &str,
    model: &str,
    sanitize: &Sanitizer,
    capture: Option<&mut BufferedCapture>,
) -> Result<Value, BufferedFailure>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    let mut sink = EventStreamSink::observed(translator::StrictChatObserver::new(LABELS));
    let end = read_upstream(upstream, &mut sink, sanitize, capture)
        .await
        .map_err(BufferedFailure::Read)?;
    translator::accumulate_response_with(
        &sink.into_body(end),
        message_id,
        model,
        &wiring(sanitize.clone()),
    )
    .map_err(BufferedFailure::Translate)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures_util::{StreamExt, stream};
    use serde_json::json;

    use super::*;
    use crate::providers::vertex::models::resolve;
    use crate::providers::vertex::translate::stream::StreamError;

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn synthetic_full_request(model: &str) -> MessagesRequest {
        request(json!({
            "model": model,
            "max_tokens": 1_024,
            "stream": false,
            "system": "Be brief.",
            "output_config": {"effort": "low"},
            "tools": [{
                "name": "lookup",
                "description": "Looks a word up.",
                "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}}
            }],
            "messages": [
                {"role": "user", "content": "Define proxy."},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Earlier reasoning", "signature": "ccp:codex:v1:cnNfc3ludGhldGlj:gAAAAABsynthetic"},
                    {"type": "text", "text": "A go-between."}
                ]},
                {"role": "user", "content": "Shorter."}
            ]
        }))
    }

    #[test]
    fn request_synthetic_drops_stream_options_and_effort_and_sends_the_bare_id() {
        for (raw, id) in [
            ("vertex/mistralai/mistral-medium-3", "mistral-medium-3"),
            ("vertex/mistralai/mistral-small-2503", "mistral-small-2503"),
            ("vertex/mistralai/codestral-2", "codestral-2"),
        ] {
            let model = resolve(raw).unwrap();
            let body = prepare_request(&synthetic_full_request(raw), &model).unwrap();
            assert_eq!(body["model"], id, "{body}");
            assert_eq!(body["stream"], json!(true), "{body}");
            assert_eq!(body["max_tokens"], json!(1_024), "{body}");
            assert!(body.get("stream_options").is_none(), "{body}");
            assert!(body.get("reasoning_effort").is_none(), "{body}");
            assert_eq!(body["tools"][0]["function"]["name"], "lookup", "{body}");
            let text = body.to_string();
            // Reasoning from an earlier turn is not sent back.
            assert!(!text.contains("reasoning_content"), "{body}");
            assert!(!text.contains("Earlier reasoning"), "{body}");
            assert!(!text.contains("ccp:codex:v1:"), "{body}");
            assert!(text.contains("A go-between."), "{body}");
            let mut keys: Vec<&str> = body
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                ["max_tokens", "messages", "model", "stream", "tools"],
                "{body}"
            );
        }
    }

    #[test]
    fn request_synthetic_an_effort_the_opencode_mapping_rejects_is_not_an_error() {
        let model = resolve("vertex/mistralai/mistral-medium-3").unwrap();
        let body = request(json!({
            "model": "vertex/mistralai/mistral-medium-3",
            "output_config": {"effort": "max"},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let prepared = prepare_request(&body, &model).unwrap();
        assert!(prepared.get("reasoning_effort").is_none(), "{prepared}");
        // No max_tokens downstream: the translator's default.
        assert_eq!(prepared["max_tokens"], json!(32_000), "{prepared}");
    }

    #[test]
    fn request_synthetic_a_versioned_id_keeps_the_version_in_the_url_only() {
        let model = resolve("vertex/mistralai/mistral-large-2411@001").unwrap();
        let body = prepare_request(&synthetic_full_request("x"), &model).unwrap();
        assert_eq!(body["model"], "mistral-large-2411", "{body}");
        let target = VertexTarget::new("synthetic-project", "us-central1", None).unwrap();
        assert_eq!(
            url(&target, &model).unwrap().as_str(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/publishers/mistralai/models/mistral-large-2411@001:streamRawPredict"
        );
    }

    #[test]
    fn url_synthetic_uses_stream_raw_predict_in_each_documented_location() {
        for (raw, location, expected) in [
            (
                "vertex/mistralai/mistral-medium-3",
                "us-central1",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/publishers/mistralai/models/mistral-medium-3:streamRawPredict",
            ),
            (
                "vertex/mistralai/codestral-2",
                "europe-west4",
                "https://europe-west4-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/europe-west4/publishers/mistralai/models/codestral-2:streamRawPredict",
            ),
        ] {
            let model = resolve(raw).unwrap();
            let target = VertexTarget::new("synthetic-project", location, None).unwrap();
            assert_eq!(url(&target, &model).unwrap().as_str(), expected, "{raw}");
        }
    }

    // Streams and buffered reads.

    const SYNTHETIC_TOKEN: &str = "ya29.synthetic-mistral-token-0123456789abcdefghijklmnopqrstuvwx";

    fn sanitizer() -> Sanitizer {
        Arc::new(|text: &str| {
            crate::providers::vertex::sanitize::sanitize_vertex_text(text, &[SYNTHETIC_TOKEN])
        })
    }

    fn context() -> RequestContext {
        RequestContext {
            req_id: "req_synthetic".to_string(),
            session_id: None,
            session_seq: None,
            provider: "vertex".to_string(),
            traffic: None,
            monitor: None,
        }
    }

    fn frame(value: Value) -> String {
        format!("data: {value}\n\n")
    }

    /// Shaped like Google's Mistral example: `chat.completion.chunk`
    /// objects with `choices[].delta.content` and a final `usage`.
    fn text_wire() -> Vec<String> {
        vec![
            frame(json!({
                "id": "synthetic", "object": "chat.completion.chunk", "model": "mistral-medium-3",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]
            })),
            frame(json!({
                "id": "synthetic", "object": "chat.completion.chunk", "model": "mistral-medium-3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "total_tokens": 9, "completion_tokens": 2}
            })),
        ]
    }

    fn chunks(parts: Vec<String>) -> Vec<Result<Bytes, std::io::Error>> {
        parts
            .into_iter()
            .map(|part| Ok(Bytes::from(part)))
            .collect()
    }

    async fn body_text(body: Body) -> String {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn stream_synthetic_translates_a_stream_that_ends_without_done() {
        let body = stream_body(
            stream::iter(chunks(text_wire())),
            "msg_synthetic".to_string(),
            "vertex/mistralai/mistral-medium-3".to_string(),
            &context(),
            sanitizer(),
        );
        let text = body_text(body).await;
        assert!(text.contains("Hello"), "{text}");
        assert!(text.contains("message_stop"), "{text}");
        assert!(!text.contains("event: error"), "{text}");
    }

    #[tokio::test]
    async fn stream_synthetic_an_event_after_done_reaches_the_client_as_an_error() {
        let mut wire = text_wire();
        wire.push("data: [DONE]\n\n".to_string());
        wire.push(frame(
            json!({"choices": [{"index": 0, "delta": {"content": "late"}}]}),
        ));
        let body = stream_body(
            stream::iter(chunks(wire)),
            "msg_synthetic".to_string(),
            "vertex/mistralai/mistral-medium-3".to_string(),
            &context(),
            sanitizer(),
        );
        let text = body_text(body).await;
        assert!(text.contains("event: error"), "{text}");
        assert!(text.contains("Vertex AI Mistral"), "{text}");
        assert!(!text.contains("OpenCode Go"), "{text}");
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test]
    async fn buffered_synthetic_completes_when_the_upstream_closes() {
        let message = read_response(
            stream::iter(chunks(text_wire())),
            "msg_synthetic",
            "vertex/mistralai/mistral-medium-3",
            &sanitizer(),
            None,
        )
        .await
        .unwrap_or_else(|failure| panic!("a finished stream is a message: {}", failure.kind()));
        assert_eq!(message["model"], "vertex/mistralai/mistral-medium-3");
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(message["content"][0]["text"], "Hello");
        assert_eq!(message["usage"]["input_tokens"], 7);
        assert_eq!(message["usage"]["output_tokens"], 2);
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_finish_then_open_upstream_completes_at_the_read_limit() {
        let mut wire = text_wire();
        // Half a frame after the finish_reason, then silence.
        wire.push("data: {\"cho".to_string());
        let upstream = stream::iter(chunks(wire)).chain(stream::pending());
        let started = tokio::time::Instant::now();
        let message = tokio::time::timeout(
            Duration::from_secs(60),
            read_response(
                upstream,
                "msg_synthetic",
                "vertex/mistralai/mistral-medium-3",
                &sanitizer(),
                None,
            ),
        )
        .await
        .expect("an upstream left open must not hold the response")
        .unwrap_or_else(|failure| {
            panic!(
                "a finish followed by silence is a message: {}",
                failure.kind()
            )
        });
        assert_eq!(started.elapsed(), Duration::from_secs(2));
        assert_eq!(message["content"][0]["text"], "Hello");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_silence_before_the_finish_reason_is_not_cut() {
        let upstream = stream::iter(chunks(text_wire()[..1].to_vec())).chain(stream::pending());
        let pending = tokio::time::timeout(
            Duration::from_secs(600),
            read_response(
                upstream,
                "msg_synthetic",
                "vertex/mistralai/mistral-medium-3",
                &sanitizer(),
                None,
            ),
        )
        .await;
        assert!(pending.is_err(), "the read ended without a finish_reason");
    }

    /// `items`, then `after` (an endless repetition of one chunk, or an
    /// upstream left open when `None`), and the count of items handed out.
    fn synthetic_counted_upstream(
        items: Vec<Result<Bytes, std::io::Error>>,
        after: Option<String>,
    ) -> (
        impl Stream<Item = Result<Bytes, std::io::Error>>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&pulled);
        let tail: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
            match after {
                Some(part) => Box::pin(stream::repeat_with(move || Ok(Bytes::from(part.clone())))),
                None => Box::pin(stream::pending()),
            };
        let upstream = stream::iter(items).chain(tail).inspect(move |_| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        (upstream, pulled)
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_a_rejected_frame_ends_the_read_while_the_upstream_stays_open() {
        let keep_alive = ": keep-alive\n\n".to_string();
        let overloaded = frame(json!({
            "error": {"message": format!("overloaded, saw {SYNTHETIC_TOKEN}")}
        }));
        let malformed = "data: {\"choices\":[\n\n".to_string();
        let late = frame(json!({"choices": [{"index": 0, "delta": {"content": "late"}}]}));
        let upstream_error = "Vertex AI Mistral upstream error: overloaded, saw [redacted]";
        let not_json = "malformed Vertex AI Mistral SSE event";
        let mut after_done = text_wire();
        after_done.push("data: [DONE]\n\n".to_string());
        after_done.push(late.clone());
        let mut after_finish = text_wire();
        after_finish.push(late);
        // The streaming path fails at these frames as soon as it reads them,
        // within the read window after a terminal event too, so the
        // buffered read must not wait for that window to pass.
        let tests = [
            (
                "upstream error before any finish_reason, upstream open",
                vec![text_wire()[0].clone(), overloaded.clone()],
                None,
                upstream_error,
            ),
            (
                "upstream error alone, then keep-alives",
                vec![keep_alive.clone(), overloaded.clone()],
                Some(keep_alive.clone()),
                upstream_error,
            ),
            (
                "malformed event, upstream open",
                vec![text_wire()[0].clone(), malformed.clone()],
                None,
                not_json,
            ),
            (
                "malformed event, then keep-alives",
                vec![malformed],
                Some(keep_alive.clone()),
                not_json,
            ),
            (
                "content after finish_reason, upstream open",
                after_finish,
                None,
                "Vertex AI Mistral stream sent content after finish_reason",
            ),
            (
                "an event after [DONE], upstream open",
                after_done,
                None,
                "Vertex AI Mistral event after [DONE]",
            ),
        ];
        for (case, parts, after, expected) in tests {
            let count = parts.len();
            let (upstream, pulled) = synthetic_counted_upstream(chunks(parts), after);
            let started = tokio::time::Instant::now();
            let failure = tokio::time::timeout(
                Duration::from_secs(60),
                read_response(
                    upstream,
                    "msg_synthetic",
                    "vertex/mistralai/mistral-medium-3",
                    &sanitizer(),
                    None,
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("{case}: the read did not end at the rejected frame"))
            .expect_err("a rejected frame is not a message");
            // A clock that moved means the read waited for a deadline;
            // more items pulled means it read past the rejected frame.
            assert_eq!(started.elapsed(), Duration::ZERO, "{case}");
            assert_eq!(
                pulled.load(std::sync::atomic::Ordering::SeqCst),
                count,
                "{case}: the upstream was read after the rejected frame"
            );
            let BufferedFailure::Translate(error) = failure else {
                panic!(
                    "{case}: expected a translation failure, got {}",
                    failure.kind()
                );
            };
            assert_eq!(error.to_string(), expected, "{case}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_keep_alives_before_the_finish_reason_do_not_end_the_read() {
        let parts = chunks(text_wire()[..1].to_vec());
        let (upstream, _) = synthetic_counted_upstream(parts, Some(": keep-alive\n\n".to_string()));
        // Keep-alives that never stop would keep the read busy without the
        // clock moving; a small pause between them lets it move.
        let upstream = upstream.then(|item| async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            item
        });
        let pending = tokio::time::timeout(
            Duration::from_secs(600),
            read_response(
                upstream,
                "msg_synthetic",
                "vertex/mistralai/mistral-medium-3",
                &sanitizer(),
                None,
            ),
        )
        .await;
        assert!(pending.is_err(), "the read ended without a finish_reason");
    }

    #[tokio::test]
    async fn buffered_synthetic_a_broken_connection_is_a_transport_error() {
        let mut parts = chunks(text_wire()[..1].to_vec());
        parts.push(Err(std::io::Error::other("synthetic reset")));
        let failure = read_response(
            stream::iter(parts),
            "msg_synthetic",
            "vertex/mistralai/mistral-medium-3",
            &sanitizer(),
            None,
        )
        .await
        .expect_err("a broken connection is not a message");
        assert!(
            matches!(failure, BufferedFailure::Read(StreamError::Transport)),
            "{}",
            failure.kind()
        );
    }

    #[tokio::test]
    async fn buffered_synthetic_a_body_over_the_cap_is_rejected() {
        let mut filler = vec![b' '; 1024 * 1024];
        filler[0] = b':';
        filler[1024 * 1024 - 2] = b'\n';
        filler[1024 * 1024 - 1] = b'\n';
        let filler = Bytes::from(filler);
        let chunks = (0..=16).map(move |_| Ok::<Bytes, std::io::Error>(filler.clone()));
        let failure = read_response(
            stream::iter(chunks),
            "msg_synthetic",
            "vertex/mistralai/mistral-medium-3",
            &sanitizer(),
            None,
        )
        .await
        .expect_err("more than 16 MiB is not a message");
        assert!(
            matches!(
                failure,
                BufferedFailure::Read(StreamError::Malformed("the response exceeds 16 MiB"))
            ),
            "{}",
            failure.kind()
        );
    }

    #[tokio::test]
    async fn buffered_synthetic_an_event_after_done_is_an_error() {
        let mut wire = text_wire();
        wire.push("data: [DONE]\n\n".to_string());
        wire.push(frame(
            json!({"choices": [{"index": 0, "delta": {"content": "late"}}]}),
        ));
        let failure = read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            "vertex/mistralai/mistral-medium-3",
            &sanitizer(),
            None,
        )
        .await
        .expect_err("an event after [DONE] is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        assert_eq!(error.to_string(), "Vertex AI Mistral event after [DONE]");
    }

    #[tokio::test]
    async fn buffered_synthetic_a_partial_frame_at_a_clean_close_stays_an_error() {
        // Only the time limit drops a frame still arriving; an upstream that
        // closes in the middle of a frame sent a cut-off body.
        let mut wire = text_wire();
        wire.push("data: {\"cho".to_string());
        let failure = read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            "vertex/mistralai/mistral-medium-3",
            &sanitizer(),
            None,
        )
        .await
        .expect_err("a body cut off in a frame is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        assert_eq!(
            error.to_string(),
            "Vertex AI Mistral stream ended with an incomplete frame"
        );
    }
}
