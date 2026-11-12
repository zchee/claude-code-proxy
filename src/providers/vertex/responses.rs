//! xAI Grok, OpenAI and Meta Muse on Vertex AI: Anthropic Messages requests
//! sent as OpenAI Responses requests.
//!
//! The request is built by the OpenCode Go Responses translator and then
//! edited for what Google documents for each publisher. The response stream
//! goes through the same translator's wiring with Vertex labels, the strict
//! terminal policy (every frame checked, a 2-second limit after the
//! completion) and the Vertex sanitiser.

use axum::body::Body;
use bytes::Bytes;
use futures_util::Stream;
use serde_json::{Map, Value};

use crate::anthropic::schema::MessagesRequest;
use crate::provider::RequestContext;
use crate::providers::opencode::chat::{
    ReceivePolicy, RedactedWiring, StreamLabels, StreamWiring, TerminalPolicy,
};
use crate::providers::opencode::responses as translator;

use super::BufferedCapture;
use super::buffered::{BufferedFailure, EventStreamSink, read_upstream};
use super::client::{OpenApiPath, VertexError, VertexTarget};
use super::models::{Family, ResolvedModel, ResponsesRoute};
use super::translate::stream::Sanitizer;

/// Names in error messages and capture files: `Vertex AI Responses stream
/// is invalid` and `061-vertex-responses-stream-summary`.
const LABELS: StreamLabels = StreamLabels {
    provider: "Vertex AI",
    capture_stem: "vertex",
};

/// Largest `max_output_tokens` Google documents for the OpenAI models on
/// Vertex AI.
const OPENAI_MAX_OUTPUT_TOKENS: u64 = 128_000;

/// The Responses request for `body`, sent as `model.upstream_model`
/// (`xai/<id>` for Grok, the bare ID for OpenAI and Meta).
///
/// Edits after translation, for every publisher: `store` stays `false`
/// (Meta does not support `true`, Grok's stored responses are a Preview
/// behind an organisation policy); no `reasoning.summary` (OpenAI documents
/// summaries as unsupported); no `include`, `text.verbosity` or
/// `prompt_cache_key` (no publisher documents them). `reasoning.effort` is
/// sent to OpenAI only, the one publisher documented to take it, and
/// OpenAI's `max_output_tokens` is capped at 128,000.
///
/// # Errors
///
/// The request cannot be translated, or it carries a hosted tool such as
/// web search, which OpenAI documents as unsupported and the others do not
/// document.
pub(super) fn prepare_request(
    body: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<Value> {
    let history = without_thinking(body);
    let mut request = translator::prepare_request(&history, &model.upstream_model, None)?;
    let Some(object) = request.as_object_mut() else {
        anyhow::bail!("the Responses request is not a JSON object");
    };
    reject_hosted_tools(object, model.family)?;
    object.insert("store".to_string(), Value::Bool(false));
    object.remove("include");
    object.remove("prompt_cache_key");
    if let Some(reasoning) = object.get_mut("reasoning").and_then(Value::as_object_mut) {
        reasoning.remove("summary");
        if model.family != Family::OpenAi {
            reasoning.remove("effort");
        }
    }
    remove_if_empty(object, "reasoning");
    if let Some(text) = object.get_mut("text").and_then(Value::as_object_mut) {
        text.remove("verbosity");
    }
    remove_if_empty(object, "text");
    if model.family == Family::OpenAi
        && let Some(tokens) = object.get("max_output_tokens").and_then(Value::as_u64)
    {
        object.insert(
            "max_output_tokens".to_string(),
            Value::from(tokens.min(OPENAI_MAX_OUTPUT_TOKENS)),
        );
    }
    Ok(request)
}

/// A copy of `body` without thinking blocks.
///
/// The translator sends back every thinking block whose signature starts
/// with `ccp:codex:v1:` as `encrypted_content`, whatever produced it: Codex,
/// another Vertex publisher, another model or another project. Opaque
/// reasoning made elsewhere must not be sent to a publisher that did not
/// make it, and nothing in the signature says where it came from, so no
/// thinking block is carried over at all. Reasoning does not continue
/// across turns as a result; the visible text and tool calls do.
fn without_thinking(body: &MessagesRequest) -> MessagesRequest {
    let mut copy = body.clone();
    for message in &mut copy.messages {
        if let Value::Array(blocks) = &mut message.content {
            blocks.retain(|block| {
                !matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("thinking" | "redacted_thinking")
                )
            });
        }
    }
    copy
}

/// The translator turns Anthropic's `web_search_20250305` into the hosted
/// `web_search` tool and every other tool into a `function`; anything but a
/// function would run on the publisher's side.
fn reject_hosted_tools(request: &Map<String, Value>, family: Family) -> anyhow::Result<()> {
    let hosted = request
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("type").and_then(Value::as_str))
        .find(|kind| *kind != "function");
    match hosted {
        Some(kind) => anyhow::bail!(
            "Vertex AI does not serve the hosted {kind} tool for {} models; remove the web search \
             tool from the request",
            family.label()
        ),
        None => Ok(()),
    }
}

fn remove_if_empty(object: &mut Map<String, Value>, key: &str) {
    if object
        .get(key)
        .and_then(Value::as_object)
        .is_some_and(Map::is_empty)
    {
        object.remove(key);
    }
}

/// Where the request goes: `endpoints/openapi/responses` for Grok,
/// `publishers/<publisher>/v1/responses` for OpenAI and Meta.
///
/// # Errors
///
/// A project, location or publisher that cannot be a path segment.
pub(super) fn url(
    target: &VertexTarget,
    model: &ResolvedModel,
    route: ResponsesRoute,
) -> Result<reqwest::Url, VertexError> {
    match route {
        ResponsesRoute::OpenApi => target.openapi_url(model, OpenApiPath::Responses),
        ResponsesRoute::Publisher => target.publisher_responses_url(model),
    }
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

/// The Anthropic SSE body translated from the upstream Responses stream.
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

/// Reads the whole upstream Responses stream with the shared buffered
/// limits and translates it into one Anthropic message.
///
/// Every chunk is checked as it arrives, with the strict frame rules. The
/// read stops at once, without polling the upstream again, where the
/// streaming path stops: when the translator closes the message by itself
/// (it repairs a `Read` call whose arguments stalled on whitespace), the
/// message ends there, whatever the upstream sends afterwards, a broken
/// connection included; an upstream failure event, a frame that is not
/// readable or not JSON, an unlisted event type or an event after the
/// completion is the error at once, even while the upstream stays open.
///
/// # Errors
///
/// The read fails ([`BufferedFailure::Read`]), or the body is not a
/// complete, valid Responses stream or reports an upstream failure
/// ([`BufferedFailure::Translate`], already sanitised).
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
    let mut sink = EventStreamSink::observed(translator::StrictResponsesObserver::new(LABELS));
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
    use crate::providers::vertex::models::{WireFormat, resolve};
    use crate::providers::vertex::translate::stream::StreamError;

    /// A thinking block signed by the Codex translator: `ccp:codex:v1:`,
    /// the base64url of the reasoning item ID `rs_synthetic`, then the
    /// encrypted content.
    const CODEX_SIGNATURE: &str =
        "ccp:codex:v1:cnNfc3ludGhldGlj:gAAAAABsyntheticEncryptedReasoning";
    /// Shaped like an Anthropic thinking signature.
    const ANTHROPIC_SIGNATURE: &str = "EqQBCkgIBRABGAIiQLsyntheticAnthropicSignature0123456789";

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    /// A request that exercises every row of the edit table: an effort, a
    /// large output budget, a session-like history with thinking blocks and
    /// a function tool.
    fn synthetic_full_request(model: &str) -> MessagesRequest {
        request(json!({
            "model": model,
            "max_tokens": 200_000,
            "stream": true,
            "system": "Be brief.",
            "output_config": {"effort": "high"},
            "tools": [{
                "name": "lookup",
                "description": "Looks a word up.",
                "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}}
            }],
            "messages": [
                {"role": "user", "content": "Define proxy."},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Codex reasoning", "signature": CODEX_SIGNATURE},
                    {"type": "thinking", "thinking": "Claude reasoning", "signature": ANTHROPIC_SIGNATURE},
                    {"type": "redacted_thinking", "data": "synthetic-redacted"},
                    {"type": "text", "text": "A go-between."}
                ]},
                {"role": "user", "content": "Shorter."}
            ]
        }))
    }

    fn prepared(model: &str) -> Value {
        let resolved = resolve(model).expect("registered model");
        prepare_request(&synthetic_full_request(model), &resolved).expect("translatable request")
    }

    fn keys(value: &Value) -> Vec<String> {
        let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn request_synthetic_follows_the_edit_table_for_xai_grok() {
        let body = prepared("vertex/xai/grok-4.7");
        assert_eq!(body["model"], "xai/grok-4.7", "{body}");
        assert_eq!(body["store"], json!(false), "{body}");
        assert_eq!(body["stream"], json!(true), "{body}");
        assert_eq!(body["max_output_tokens"], json!(200_000), "{body}");
        for absent in ["reasoning", "include", "text", "prompt_cache_key"] {
            assert!(body.get(absent).is_none(), "{absent} in {body}");
        }
        assert_eq!(
            keys(&body),
            [
                "input",
                "instructions",
                "max_output_tokens",
                "model",
                "parallel_tool_calls",
                "store",
                "stream",
                "tools"
            ],
            "{body}"
        );
    }

    #[test]
    fn request_synthetic_follows_the_edit_table_for_openai() {
        let body = prepared("vertex/openai/gpt-6.1-sol");
        assert_eq!(body["model"], "gpt-6.1-sol", "{body}");
        assert_eq!(body["store"], json!(false), "{body}");
        assert_eq!(body["stream"], json!(true), "{body}");
        // Effort is sent; the summary is not.
        assert_eq!(body["reasoning"], json!({"effort": "high"}), "{body}");
        // 200,000 requested, 128,000 documented.
        assert_eq!(body["max_output_tokens"], json!(128_000), "{body}");
        for absent in ["include", "text", "prompt_cache_key"] {
            assert!(body.get(absent).is_none(), "{absent} in {body}");
        }
        assert_eq!(
            keys(&body),
            [
                "input",
                "instructions",
                "max_output_tokens",
                "model",
                "parallel_tool_calls",
                "reasoning",
                "store",
                "stream",
                "tools"
            ],
            "{body}"
        );
    }

    #[test]
    fn request_synthetic_follows_the_edit_table_for_meta_muse() {
        let body = prepared("vertex/meta/muse-spark-1.3");
        assert_eq!(body["model"], "muse-spark-1.3", "{body}");
        assert_eq!(body["store"], json!(false), "{body}");
        assert_eq!(body["stream"], json!(true), "{body}");
        assert_eq!(body["max_output_tokens"], json!(200_000), "{body}");
        for absent in ["reasoning", "include", "text", "prompt_cache_key"] {
            assert!(body.get(absent).is_none(), "{absent} in {body}");
        }
    }

    #[test]
    fn request_synthetic_openai_under_the_cap_keeps_the_downstream_value() {
        let resolved = resolve("vertex/openai/gpt-6-luna").unwrap();
        let mut body = synthetic_full_request("vertex/openai/gpt-6-luna");
        body.max_tokens = Some(4_096);
        let prepared = prepare_request(&body, &resolved).unwrap();
        assert_eq!(prepared["max_output_tokens"], json!(4_096));
        body.max_tokens = None;
        let prepared = prepare_request(&body, &resolved).unwrap();
        assert!(prepared.get("max_output_tokens").is_none(), "{prepared}");
    }

    #[test]
    fn request_synthetic_keeps_a_structured_output_format_without_verbosity() {
        for model in [
            "vertex/xai/grok-4.6",
            "vertex/openai/gpt-6-astra",
            "vertex/meta/muse-wonder-1.0",
        ] {
            let resolved = resolve(model).unwrap();
            let body = request(json!({
                "model": model,
                "max_tokens": 64,
                "output_config": {"format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
                }},
                "messages": [{"role": "user", "content": "json please"}]
            }));
            let prepared = prepare_request(&body, &resolved).unwrap();
            assert_eq!(
                prepared["text"]["format"]["type"], "json_schema",
                "{model}: {prepared}"
            );
            assert!(
                prepared["text"].get("verbosity").is_none(),
                "{model}: {prepared}"
            );
        }
    }

    #[test]
    fn request_synthetic_without_effort_sends_no_reasoning_for_any_publisher() {
        for model in [
            "vertex/xai/grok-4.20-reasoning",
            "vertex/openai/gpt-6.1-terra",
            "vertex/meta/muse-ember-1.0",
        ] {
            let resolved = resolve(model).unwrap();
            let body = request(json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}]
            }));
            let prepared = prepare_request(&body, &resolved).unwrap();
            assert!(prepared.get("reasoning").is_none(), "{model}: {prepared}");
            assert!(prepared.get("include").is_none(), "{model}: {prepared}");
            assert_eq!(prepared["store"], json!(false), "{model}: {prepared}");
        }
    }

    #[test]
    fn thinking_blocks_synthetic_are_removed_before_translation() {
        let original = synthetic_full_request("vertex/openai/gpt-6.1-sol");
        // Without the removal the translator replays the Codex-signed block
        // as an encrypted reasoning item.
        let unedited = translator::prepare_request(&original, "gpt-6.1-sol", None).unwrap();
        assert!(
            unedited["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "reasoning"),
            "{unedited}"
        );
        for model in [
            "vertex/xai/grok-4.7",
            "vertex/openai/gpt-6.1-sol",
            "vertex/meta/muse-spark-1.3",
        ] {
            let body = prepared(model);
            let text = body.to_string();
            let input = body["input"].as_array().unwrap();
            assert!(
                input.iter().all(|item| item["type"] != "reasoning"),
                "{model}: {body}"
            );
            for leaked in [
                "encrypted_content",
                "gAAAAABsyntheticEncryptedReasoning",
                "rs_synthetic",
                ANTHROPIC_SIGNATURE,
                "Codex reasoning",
                "Claude reasoning",
                "synthetic-redacted",
            ] {
                assert!(!text.contains(leaked), "{model} sent {leaked}: {body}");
            }
            // The visible text of the same turn is still there.
            assert!(text.contains("A go-between."), "{model}: {body}");
        }
        // The caller's request is not changed.
        let blocks = original.messages[1].content.as_array().unwrap();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0]["signature"], CODEX_SIGNATURE);
    }

    #[test]
    fn hosted_web_search_synthetic_is_refused_for_every_publisher() {
        for (model, family) in [
            ("vertex/xai/grok-4.7", "xAI Grok"),
            ("vertex/openai/gpt-6.1-sol", "OpenAI"),
            ("vertex/meta/muse-spark-1.3", "Meta Muse"),
        ] {
            let resolved = resolve(model).unwrap();
            let body = request(json!({
                "model": model,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": "news?"}]
            }));
            let error = prepare_request(&body, &resolved).unwrap_err().to_string();
            assert_eq!(
                error,
                format!(
                    "Vertex AI does not serve the hosted web_search tool for {family} models; \
                     remove the web search tool from the request"
                ),
                "{model}"
            );
        }
    }

    #[test]
    fn url_synthetic_uses_the_documented_path_for_each_route_and_location_kind() {
        let cases = [
            (
                "vertex/xai/grok-4.7",
                "global",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/endpoints/openapi/responses",
            ),
            (
                "vertex/xai/grok-4.7",
                "us",
                "https://aiplatform.us.rep.googleapis.com/v1/projects/synthetic-project/locations/us/endpoints/openapi/responses",
            ),
            (
                "vertex/xai/grok-4.20-reasoning",
                "us-central1",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/endpoints/openapi/responses",
            ),
            (
                "vertex/openai/gpt-6.1-sol",
                "global",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/openai/v1/responses",
            ),
            (
                "vertex/openai/gpt-6-luna",
                "us",
                "https://aiplatform.us.rep.googleapis.com/v1/projects/synthetic-project/locations/us/publishers/openai/v1/responses",
            ),
            (
                "vertex/meta/muse-spark-1.3",
                "global",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/meta/v1/responses",
            ),
        ];
        for (model, location, expected) in cases {
            let resolved = resolve(model).unwrap();
            let WireFormat::Responses { route } = resolved.wire else {
                panic!("{model} is not a Responses model");
            };
            let target = VertexTarget::new("synthetic-project", location, None).unwrap();
            assert_eq!(
                url(&target, &resolved, route).unwrap().as_str(),
                expected,
                "{model} in {location}"
            );
        }
    }

    // Streams and buffered reads.

    const SYNTHETIC_TOKEN: &str = "ya29.synthetic-responses-token-0123456789abcdefghijklmnopqrstuv";

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

    fn text_wire() -> Vec<String> {
        vec![
            frame(json!({"type":"response.created","response":{}})),
            frame(
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_up"}}),
            ),
            frame(json!({"type":"response.output_text.delta","output_index":0,"delta":"Hello"})),
            frame(
                json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}),
            ),
        ]
    }

    fn completed() -> String {
        frame(json!({
            "type":"response.completed",
            "response":{"id":"resp_1","status":"completed","incomplete_details":null,"usage":{"input_tokens":5,"output_tokens":1}}
        }))
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
    async fn stream_synthetic_translates_a_complete_responses_stream() {
        let mut wire = text_wire();
        wire.push(completed());
        let body = stream_body(
            stream::iter(chunks(wire)),
            "msg_synthetic".to_string(),
            "vertex/xai/grok-4.7".to_string(),
            &context(),
            sanitizer(),
        );
        let text = body_text(body).await;
        assert!(text.contains("Hello"), "{text}");
        assert!(text.contains("message_stop"), "{text}");
        assert!(text.contains("vertex/xai/grok-4.7"), "{text}");
        assert!(!text.contains("event: error"), "{text}");
    }

    #[tokio::test]
    async fn stream_synthetic_an_unlisted_event_type_reaches_the_client_as_an_error() {
        let mut wire = text_wire();
        wire.insert(
            2,
            frame(json!({"type":"response.refusal.delta","output_index":0,"delta":"no"})),
        );
        wire.push(completed());
        let body = stream_body(
            stream::iter(chunks(wire)),
            "msg_synthetic".to_string(),
            "vertex/openai/gpt-6.1-sol".to_string(),
            &context(),
            sanitizer(),
        );
        let text = body_text(body).await;
        assert!(text.contains("event: error"), "{text}");
        assert!(
            text.contains("Vertex AI Responses stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("OpenCode Go"), "{text}");
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test]
    async fn stream_synthetic_an_event_after_the_completion_is_an_error() {
        let mut wire = text_wire();
        wire.push(completed());
        wire.push(frame(
            json!({"type":"response.output_text.delta","output_index":0,"delta":"late"}),
        ));
        let body = stream_body(
            stream::iter(chunks(wire)),
            "msg_synthetic".to_string(),
            "vertex/meta/muse-spark-1.3".to_string(),
            &context(),
            sanitizer(),
        );
        let text = body_text(body).await;
        assert!(
            text.contains("Vertex AI Responses stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test]
    async fn buffered_synthetic_completes_when_the_upstream_closes() {
        let mut wire = text_wire();
        wire.push(completed());
        let message = read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            "vertex/xai/grok-4.7",
            &sanitizer(),
            None,
        )
        .await
        .unwrap_or_else(|_| panic!("a complete stream is a message"));
        assert_eq!(message["model"], "vertex/xai/grok-4.7");
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(message["content"][0]["text"], "Hello");
        assert_eq!(message["usage"]["input_tokens"], 5);
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_completion_then_open_upstream_completes_at_the_read_limit() {
        let mut wire = text_wire();
        wire.push(completed());
        // Half a frame after the completion, then silence.
        wire.push("data: {\"ty".to_string());
        let upstream = stream::iter(chunks(wire)).chain(stream::pending());
        let started = tokio::time::Instant::now();
        let message = tokio::time::timeout(
            Duration::from_secs(60),
            read_response(
                upstream,
                "msg_synthetic",
                "vertex/openai/gpt-6.1-sol",
                &sanitizer(),
                None,
            ),
        )
        .await
        .expect("an upstream left open must not hold the response")
        .unwrap_or_else(|_| panic!("a completion followed by silence is a message"));
        assert_eq!(started.elapsed(), Duration::from_secs(2));
        assert_eq!(message["content"][0]["text"], "Hello");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_silence_before_the_completion_is_not_cut() {
        let upstream = stream::iter(chunks(text_wire())).chain(stream::pending());
        let pending = tokio::time::timeout(
            Duration::from_secs(600),
            read_response(
                upstream,
                "msg_synthetic",
                "vertex/xai/grok-4.7",
                &sanitizer(),
                None,
            ),
        )
        .await;
        assert!(pending.is_err(), "the read ended without a completion");
    }

    #[tokio::test]
    async fn buffered_synthetic_a_broken_connection_is_a_transport_error() {
        let mut parts = chunks(text_wire());
        parts.push(Err(std::io::Error::other("synthetic reset")));
        let failure = read_response(
            stream::iter(parts),
            "msg_synthetic",
            "vertex/xai/grok-4.7",
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
            "vertex/xai/grok-4.7",
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
    async fn buffered_synthetic_an_unlisted_event_type_is_an_error() {
        let mut wire = text_wire();
        wire.insert(
            2,
            frame(json!({"type":"response.audio.delta","delta":"AA=="})),
        );
        wire.push(completed());
        let failure = read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            "vertex/meta/muse-spark-1.3",
            &sanitizer(),
            None,
        )
        .await
        .expect_err("an unlisted event type is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        assert_eq!(
            error.to_string(),
            "Vertex AI Responses stream is invalid: unlisted_event_type"
        );
    }

    #[tokio::test]
    async fn buffered_synthetic_an_upstream_failure_is_sanitised() {
        let wire = vec![frame(json!({
            "type":"response.failed",
            "response":{"status":"failed","error":{"code":"server_error","message":format!("saw {SYNTHETIC_TOKEN}")}}
        }))];
        let failure = read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            "vertex/openai/gpt-6.1-sol",
            &sanitizer(),
            None,
        )
        .await
        .expect_err("a failed response is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        let text = error.to_string();
        assert!(!text.contains(SYNTHETIC_TOKEN), "{text}");
        assert!(text.contains("[redacted]"), "{text}");
    }

    #[tokio::test]
    async fn buffered_synthetic_a_partial_frame_at_a_clean_close_stays_an_error() {
        // Only the time limit drops a frame still arriving; an upstream that
        // closes in the middle of a frame sent a cut-off body.
        let mut wire = text_wire();
        wire.push(completed());
        wire.push("data: {\"ty".to_string());
        let failure = read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            "vertex/xai/grok-4.7",
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
            "Vertex AI Responses stream is invalid: incomplete_stream"
        );
    }

    fn read_call_added() -> String {
        frame(json!({
            "type":"response.output_item.added",
            "output_index":0,
            "item":{"type":"function_call","call_id":"call_1","name":"Read"}
        }))
    }

    /// A `Read` argument delta that ends in 1,024 spaces, which the
    /// translator repairs into a finished tool call.
    fn stalled_read_delta() -> String {
        frame(json!({
            "type":"response.function_call_arguments.delta",
            "output_index":0,
            "delta":format!("{{\"file_path\":\"/tmp/a\"}}{}", " ".repeat(1024)),
        }))
    }

    /// `items`, then `after` (an endless repetition of one chunk, one
    /// second of paused time apart, or an upstream left open when `None`),
    /// and the count of items handed out. The pause lets the clock move, so
    /// a read that wrongly keeps pulling the repetition reaches the test's
    /// timeout instead of spinning on chunks that are always ready.
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
                Some(part) => Box::pin(stream::unfold(part, |part| async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    Some((Ok(Bytes::from(part.clone())), part))
                })),
                None => Box::pin(stream::pending()),
            };
        let upstream = stream::iter(items).chain(tail).inspect(move |_| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        (upstream, pulled)
    }

    fn assert_repaired_read(message: &Value, case: &str) {
        assert_eq!(message["stop_reason"], "tool_use", "{case}: {message}");
        let tool_use = message["content"]
            .as_array()
            .and_then(|blocks| blocks.iter().find(|block| block["type"] == "tool_use"))
            .unwrap_or_else(|| panic!("{case}: {message}"));
        assert_eq!(tool_use["name"], "Read", "{case}");
        assert_eq!(tool_use["input"], json!({"file_path": "/tmp/a"}), "{case}");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_settles_at_a_repaired_read_call_without_reading_on() {
        let whitespace = frame(json!({
            "type":"response.function_call_arguments.delta",
            "output_index":0,
            "delta":"    ",
        }));
        let repaired = format!("{}{}", read_call_added(), stalled_read_delta());
        for (case, parts, after) in [
            (
                "repair in one item, whitespace keeps coming",
                vec![repaired.clone()],
                Some(whitespace.clone()),
            ),
            (
                "repair over two items, whitespace keeps coming",
                vec![read_call_added(), stalled_read_delta()],
                Some(whitespace.clone()),
            ),
            (
                "repair, then the upstream stays open",
                vec![repaired.clone()],
                None,
            ),
        ] {
            let count = parts.len();
            let (upstream, pulled) = synthetic_counted_upstream(chunks(parts), after);
            let started = tokio::time::Instant::now();
            let message = tokio::time::timeout(
                Duration::from_secs(60),
                read_response(
                    upstream,
                    "msg_synthetic",
                    "vertex/openai/gpt-6.1-sol",
                    &sanitizer(),
                    None,
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("{case}: the read did not settle at the repair"))
            .unwrap_or_else(|failure| panic!("{case}: {}", failure.kind()));
            assert_eq!(started.elapsed(), Duration::ZERO, "{case}");
            assert_repaired_read(&message, case);
            assert_eq!(
                pulled.load(std::sync::atomic::Ordering::SeqCst),
                count,
                "{case}: the upstream was read after the repair"
            );
        }
    }

    #[tokio::test]
    async fn buffered_synthetic_a_broken_connection_after_a_repaired_read_call_is_not_read() {
        let mut parts = chunks(vec![read_call_added(), stalled_read_delta()]);
        parts.push(Err(std::io::Error::other("synthetic reset")));
        let message = read_response(
            stream::iter(parts),
            "msg_synthetic",
            "vertex/xai/grok-4.7",
            &sanitizer(),
            None,
        )
        .await
        .unwrap_or_else(|failure| panic!("the repaired call is the message: {}", failure.kind()));
        assert_repaired_read(&message, "repair, then a broken connection");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_a_failure_or_rejected_frame_ends_the_read_while_the_upstream_stays_open()
     {
        let failed = frame(json!({
            "type":"response.failed",
            "response":{"status":"failed","error":{"code":"server_error","message":format!("saw {SYNTHETIC_TOKEN}")}}
        }));
        let late =
            frame(json!({"type":"response.output_text.delta","output_index":0,"delta":"late"}));
        let mut after_completion = text_wire();
        after_completion.push(completed());
        after_completion.push(late);
        let mut unlisted = text_wire();
        unlisted.insert(
            2,
            frame(json!({"type":"response.audio.delta","delta":"AA=="})),
        );
        let mut malformed = text_wire();
        malformed.push("data: {not json\n\n".to_string());
        let cases = [
            ("upstream failure", vec![failed], None),
            (
                "unlisted event type",
                unlisted,
                Some("Vertex AI Responses stream is invalid: unlisted_event_type"),
            ),
            (
                "event after the completion",
                after_completion,
                Some("Vertex AI Responses stream is invalid: event_after_completion"),
            ),
            (
                "malformed event",
                malformed,
                Some("Vertex AI Responses stream is invalid: malformed_event"),
            ),
        ];
        for (case, parts, expected) in cases {
            let count = parts.len();
            let (upstream, pulled) = synthetic_counted_upstream(chunks(parts), None);
            let started = tokio::time::Instant::now();
            let failure = tokio::time::timeout(
                Duration::from_secs(60),
                read_response(
                    upstream,
                    "msg_synthetic",
                    "vertex/openai/gpt-6.1-sol",
                    &sanitizer(),
                    None,
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("{case}: the read waited for the upstream to close"))
            .expect_err(case);
            assert_eq!(started.elapsed(), Duration::ZERO, "{case}");
            assert_eq!(
                pulled.load(std::sync::atomic::Ordering::SeqCst),
                count,
                "{case}"
            );
            let BufferedFailure::Translate(error) = failure else {
                panic!(
                    "{case}: expected a translation failure, got {}",
                    failure.kind()
                );
            };
            let text = error.to_string();
            match expected {
                Some(expected) => assert_eq!(text, expected, "{case}"),
                None => {
                    assert!(text.contains("[redacted]"), "{case}: {text}");
                    assert!(!text.contains(SYNTHETIC_TOKEN), "{case}: {text}");
                }
            }
        }
    }

    /// What may follow a settling frame in the same upstream item: bytes
    /// that are not UTF-8, a frame over the decoder's 1 MiB frame limit,
    /// and comment frames that take the item over the 16 MiB cap.
    fn synthetic_suffixes() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("bytes that are not UTF-8", b"data: \xff\xfe\n\n".to_vec()),
            (
                "a frame over the frame limit",
                format!("data: {}\n\n", "x".repeat(1024 * 1024 + 1)).into_bytes(),
            ),
            (
                "comment frames past the 16 MiB cap",
                ":\n\n".repeat(6 * 1024 * 1024).into_bytes(),
            ),
        ]
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_a_repaired_read_call_ignores_what_follows_in_any_split() {
        let repaired = format!("{}{}", read_call_added(), stalled_read_delta()).into_bytes();
        for (suffix_name, suffix) in synthetic_suffixes() {
            let same_item = [repaired.as_slice(), suffix.as_slice()].concat();
            for (split, items) in [
                ("in the same item", vec![same_item]),
                ("in the next item", vec![repaired.clone(), suffix.clone()]),
            ] {
                let case = format!("{suffix_name} {split}");
                let items = items
                    .into_iter()
                    .map(|item| Ok(Bytes::from(item)))
                    .collect();
                let (upstream, pulled) = synthetic_counted_upstream(items, None);
                let started = tokio::time::Instant::now();
                let message = tokio::time::timeout(
                    Duration::from_secs(60),
                    read_response(
                        upstream,
                        "msg_synthetic",
                        "vertex/openai/gpt-6.1-sol",
                        &sanitizer(),
                        None,
                    ),
                )
                .await
                .unwrap_or_else(|_| panic!("{case}: the read did not settle at the repair"))
                .unwrap_or_else(|failure| panic!("{case}: {}", failure.kind()));
                assert_eq!(started.elapsed(), Duration::ZERO, "{case}");
                assert_repaired_read(&message, &case);
                assert_eq!(
                    pulled.load(std::sync::atomic::Ordering::SeqCst),
                    1,
                    "{case}: the upstream was read after the repair"
                );
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_a_rejected_frame_stays_the_error_whatever_follows_in_its_item() {
        let mut unlisted = text_wire();
        unlisted.insert(
            2,
            frame(json!({"type":"response.audio.delta","delta":"AA=="})),
        );
        let rejected = unlisted.concat().into_bytes();
        for (suffix_name, suffix) in synthetic_suffixes() {
            let over_cap = suffix.len() > 16 * 1024 * 1024;
            let item = [rejected.as_slice(), suffix.as_slice()].concat();
            let (upstream, pulled) = synthetic_counted_upstream(vec![Ok(Bytes::from(item))], None);
            let failure = tokio::time::timeout(
                Duration::from_secs(60),
                read_response(
                    upstream,
                    "msg_synthetic",
                    "vertex/openai/gpt-6.1-sol",
                    &sanitizer(),
                    None,
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("{suffix_name}: the read waited for the upstream"))
            .expect_err(suffix_name);
            assert_eq!(
                pulled.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "{suffix_name}"
            );
            if over_cap {
                // A rejected item is kept whole, so the cap counts all of it.
                assert!(
                    matches!(
                        failure,
                        BufferedFailure::Read(StreamError::Malformed(
                            "the response exceeds 16 MiB"
                        ))
                    ),
                    "{suffix_name}: {}",
                    failure.kind()
                );
                continue;
            }
            let BufferedFailure::Translate(error) = failure else {
                panic!(
                    "{suffix_name}: expected a translation failure, got {}",
                    failure.kind()
                );
            };
            assert_eq!(
                error.to_string(),
                "Vertex AI Responses stream is invalid: unlisted_event_type",
                "{suffix_name}"
            );
        }
    }
}
