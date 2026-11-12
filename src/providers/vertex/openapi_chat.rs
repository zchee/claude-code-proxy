//! Model Garden's managed open models on Vertex AI
//! (`<publisher>/<model-id>-maas`): Anthropic Messages requests sent as
//! OpenAI Chat Completions to `endpoints/openapi/chat/completions`.
//!
//! The request is built by the OpenCode Go Chat Completions translator and
//! then edited by the model's controls ([`OpenModelControls`]); the response
//! stream goes through the same translator's wiring with Vertex labels, the
//! strict terminal policy and the Vertex sanitiser. The upstream is always
//! asked for a stream; a non-streaming client gets the stream accumulated
//! into one message. Upstream `delta.reasoning_content` becomes a thinking
//! block, as the translator does for every Chat Completions provider.

use axum::body::Body;
use bytes::Bytes;
use futures_util::Stream;
use serde_json::{Map, Value, json};

use crate::anthropic::schema::MessagesRequest;
use crate::provider::RequestContext;
use crate::providers::opencode::chat::{
    self as translator, EffortPolicy, ReasoningReplay, ReceivePolicy, RedactedWiring,
    RequestPolicy, StreamLabels, StreamWiring, TerminalPolicy,
};
use crate::providers::translate_shared::read_effort;

use super::BufferedCapture;
use super::buffered::{BufferedFailure, EventStreamSink, read_upstream};
use super::client::{OpenApiPath, VertexError, VertexTarget};
use super::models::{OpenModelControls, ResolvedModel, ThinkingControl, ToolChoiceRule};
use super::translate::stream::Sanitizer;

/// Names in error messages and capture files: `Vertex AI open model stream
/// is invalid` and `061-vertex-open-model-stream-summary`, which collides
/// with neither Gemini's `061-vertex-stream-summary` nor Mistral's.
const LABELS: StreamLabels = StreamLabels {
    provider: "Vertex AI open model",
    capture_stem: "vertex-open-model",
};

/// The translator neither sets `reasoning_effort` nor sends reasoning back.
///
/// The effort is set here by the model's [`ThinkingControl`]: the
/// translator's own mapping follows OpenCode Go's model names and would
/// refuse efforts that Vertex AI's models take. The reasoning of earlier
/// turns is not replayed as `reasoning_content`: Google documents that
/// field in responses only, and the thinking text in a history may have
/// been written by another provider.
const REQUEST_POLICY: RequestPolicy = RequestPolicy {
    effort: EffortPolicy::Untouched,
    reasoning_replay: ReasoningReplay::Never,
};

/// The Chat Completions request for `body`, sent as `model.upstream_model`
/// (`<publisher>/<model-id>`) with `stream: true` and
/// `stream_options.include_usage`, edited by the model's controls:
///
/// - [`ThinkingControl::ReasoningEffort`]: Claude Code's effort becomes
///   `reasoning_effort` `low` or `medium`, and `high` for `high`, `xhigh`
///   and `max`; no effort sends none.
/// - [`ThinkingControl::EnableThinking`]: `chat_template_kwargs` is
///   `{"enable_thinking": true}` when thinking is on: `thinking` is present
///   with any `type` but `disabled`, or `thinking` is absent and the request
///   carries an effort. Otherwise it is `false`.
/// - [`ThinkingControl::NoSwitch`]: neither field.
///
/// # Errors
///
/// The request cannot be expressed as Chat Completions, it carries an
/// invalid effort or a `thinking` that is not an object, or it forces a
/// tool (`tool_choice` `any` or `tool`) on a model whose rule is
/// [`ToolChoiceRule::AutoOrNone`]: the choice is refused rather than sent
/// as one the model would read differently.
pub(super) fn prepare_request(
    body: &MessagesRequest,
    model: &ResolvedModel,
) -> anyhow::Result<Value> {
    let controls = model.open_model_controls();
    let translated = translator::prepare_request_with(body, &model.upstream_model, REQUEST_POLICY)?;
    let mut request = serde_json::to_value(translated)?;
    let Some(object) = request.as_object_mut() else {
        anyhow::bail!("the open-model request is not a JSON object");
    };
    check_tool_choice(object, controls, &model.upstream_model)?;
    match controls.thinking {
        ThinkingControl::ReasoningEffort => {
            if let Some(effort) = read_effort(body)? {
                let effort = match effort {
                    "low" | "medium" => effort,
                    _ => "high",
                };
                object.insert("reasoning_effort".to_string(), Value::from(effort));
            }
        }
        ThinkingControl::EnableThinking => {
            object.insert(
                "chat_template_kwargs".to_string(),
                json!({"enable_thinking": thinking_on(body)?}),
            );
        }
        ThinkingControl::NoSwitch => {}
    }
    object.insert(
        "model".to_string(),
        Value::from(model.upstream_model.as_str()),
    );
    object.insert("stream".to_string(), Value::Bool(true));
    Ok(request)
}

/// Refuses a forced tool choice for a model that only takes `auto` and
/// `none`. The translator writes Anthropic's `any` as `"required"` and
/// `tool` as an object naming the function.
fn check_tool_choice(
    request: &Map<String, Value>,
    controls: OpenModelControls,
    upstream_model: &str,
) -> anyhow::Result<()> {
    let forced = match request.get("tool_choice") {
        Some(Value::String(choice)) => choice == "required",
        Some(Value::Object(_)) => true,
        _ => false,
    };
    if forced && controls.tool_choice == ToolChoiceRule::AutoOrNone {
        anyhow::bail!(
            "Vertex AI model {upstream_model} does not support a forced tool choice \
             (tool_choice any or tool); use tool_choice auto or none"
        );
    }
    Ok(())
}

/// Whether thinking is on: a `thinking` object whose `type` is not
/// `disabled`, or, without `thinking`, an `output_config.effort`. The
/// OpenAI-compatible routes carry a client's `reasoning_effort` as the
/// effort and never set `thinking`, so the effort alone must be able to
/// turn thinking on; an explicit `thinking` always decides.
///
/// # Errors
///
/// `thinking` is not an object, or, without `thinking`, the effort is not
/// one of the valid values.
fn thinking_on(body: &MessagesRequest) -> anyhow::Result<bool> {
    match body.extra.get("thinking") {
        None | Some(Value::Null) => Ok(read_effort(body)?.is_some()),
        Some(Value::Object(thinking)) => {
            Ok(thinking.get("type").and_then(Value::as_str) != Some("disabled"))
        }
        Some(_) => anyhow::bail!("thinking must be an object"),
    }
}

/// `{base}/{version}/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`,
/// in the model's API version.
///
/// # Errors
///
/// A project or location that cannot be placed in the path.
pub(super) fn url(
    target: &VertexTarget,
    model: &ResolvedModel,
) -> Result<reqwest::Url, VertexError> {
    target.openapi_url(model, OpenApiPath::ChatCompletions)
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

/// The Anthropic SSE body translated from the upstream Chat Completions
/// stream. `model` is the name the client asked for. `sanitize` cleans
/// upstream prose; `redact` is applied to every string of every event the
/// traffic capture records.
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

/// Reads the whole upstream stream with the shared buffered limits and
/// accumulates it into one Anthropic message. The chunks are checked as
/// they arrive ([`translator::StrictChatObserver`]): a frame the streaming
/// path rejects, such as an upstream error event, ends the read at once,
/// even while the upstream stays open. Silence before the terminal event
/// is not cut.
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

    /// System prompt, effort, thinking, one tool and a history with a
    /// thinking block, a redacted one and text.
    fn synthetic_full_request(model: &str, effort: &str, thinking: Value) -> MessagesRequest {
        request(json!({
            "model": model,
            "max_tokens": 1_024,
            "stream": false,
            "system": "Be brief.",
            "output_config": {"effort": effort},
            "thinking": thinking,
            "tools": [{
                "name": "lookup",
                "description": "Looks a word up.",
                "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}}
            }],
            "messages": [
                {"role": "user", "content": "Define proxy."},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Earlier reasoning", "signature": "ccp:codex:v1:cnNfc3ludGhldGlj:gAAAAABsynthetic"},
                    {"type": "redacted_thinking", "data": "c3ludGhldGljLXJlZGFjdGVk"},
                    {"type": "text", "text": "A go-between."}
                ]},
                {"role": "user", "content": "Shorter."}
            ]
        }))
    }

    fn prepared(raw: &str, body: &MessagesRequest) -> Value {
        let model = resolve(raw).unwrap_or_else(|error| panic!("{raw}: {error}"));
        prepare_request(body, &model).unwrap_or_else(|error| panic!("{raw}: {error}"))
    }

    /// The body every model gets for [`synthetic_full_request`], before the
    /// per-model thinking field.
    fn expected_common(upstream_model: &str) -> Value {
        json!({
            "model": upstream_model,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Define proxy."},
                {"role": "assistant", "content": "A go-between."},
                {"role": "user", "content": "Shorter."}
            ],
            "tools": [{"type": "function", "function": {
                "name": "lookup",
                "description": "Looks a word up.",
                "parameters": {"type": "object", "properties": {"word": {"type": "string"}}}
            }}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 1_024
        })
    }

    #[test]
    fn request_synthetic_gpt_oss_gets_reasoning_effort_and_nothing_else() {
        let raw = "vertex/openai/gpt-oss-120b-maas";
        for (effort, sent) in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ] {
            let body = prepared(
                raw,
                &synthetic_full_request(raw, effort, json!({"type": "adaptive"})),
            );
            let mut expected = expected_common("openai/gpt-oss-120b-maas");
            expected["reasoning_effort"] = json!(sent);
            assert_eq!(body, expected, "{effort}");
        }
        // No effort, no field.
        let body = prepared(
            raw,
            &request(json!({"model": raw, "messages": [{"role": "user", "content": "hi"}]})),
        );
        assert!(body.get("reasoning_effort").is_none(), "{body}");
        assert!(body.get("chat_template_kwargs").is_none(), "{body}");
    }

    #[test]
    fn request_synthetic_glm_and_gemma_get_enable_thinking_from_the_thinking_setting() {
        for (raw, upstream) in [
            ("vertex/zai-org/glm-5.2-maas", "zai-org/glm-5.2-maas"),
            (
                "vertex/google/gemma-4-26b-a4b-it-maas",
                "google/gemma-4-26b-a4b-it-maas",
            ),
        ] {
            for (thinking, enabled) in [
                (json!({"type": "adaptive"}), true),
                (json!({"type": "enabled", "budget_tokens": 2_048}), true),
                (json!({"type": "disabled"}), false),
                // No thinking setting: the request's effort turns it on.
                (Value::Null, true),
            ] {
                let body = prepared(raw, &synthetic_full_request(raw, "low", thinking.clone()));
                let mut expected = expected_common(upstream);
                expected["chat_template_kwargs"] = json!({"enable_thinking": enabled});
                // The effort is not sent: these models document no effort.
                assert_eq!(body, expected, "{raw} {thinking}");
            }
            // No `thinking` at all is thinking off.
            let body = prepared(
                raw,
                &request(json!({"model": raw, "messages": [{"role": "user", "content": "hi"}]})),
            );
            assert_eq!(
                body["chat_template_kwargs"],
                json!({"enable_thinking": false}),
                "{body}"
            );
            let error = prepare_request(
                &request(json!({"model": raw, "thinking": true, "messages": []})),
                &resolve(raw).unwrap(),
            )
            .unwrap_err();
            assert_eq!(error.to_string(), "thinking must be an object");
        }
    }

    #[test]
    fn request_synthetic_glm_and_gemma_without_thinking_follow_the_effort() {
        for raw in [
            "vertex/zai-org/glm-5.2-maas",
            "vertex/google/gemma-4-26b-a4b-it-maas",
        ] {
            for (case, thinking, effort, enabled) in [
                ("thinking absent, effort present", None, Some("high"), true),
                ("thinking absent, no effort", None, None, false),
                (
                    "thinking disabled, effort present",
                    Some(json!({"type": "disabled"})),
                    Some("high"),
                    false,
                ),
                (
                    "thinking enabled, no effort",
                    Some(json!({"type": "enabled", "budget_tokens": 2_048})),
                    None,
                    true,
                ),
                (
                    "thinking adaptive, no effort",
                    Some(json!({"type": "adaptive"})),
                    None,
                    true,
                ),
                (
                    "thinking adaptive, effort present",
                    Some(json!({"type": "adaptive"})),
                    Some("max"),
                    true,
                ),
            ] {
                let mut value =
                    json!({"model": raw, "messages": [{"role": "user", "content": "hi"}]});
                if let Some(thinking) = thinking {
                    value["thinking"] = thinking;
                }
                if let Some(effort) = effort {
                    value["output_config"] = json!({"effort": effort});
                }
                let body = prepared(raw, &request(value));
                assert_eq!(
                    body["chat_template_kwargs"],
                    json!({"enable_thinking": enabled}),
                    "{raw}: {case}"
                );
                assert!(body.get("reasoning_effort").is_none(), "{raw}: {case}");
            }
            let error = prepare_request(
                &request(json!({
                    "model": raw,
                    "output_config": {"effort": "huge"},
                    "messages": [{"role": "user", "content": "hi"}]
                })),
                &resolve(raw).unwrap(),
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "Invalid output_config.effort: huge",
                "{raw}"
            );
        }
    }

    #[test]
    fn request_synthetic_llama_and_deepseek_get_no_thinking_field() {
        for (raw, upstream) in [
            (
                "vertex/meta/llama-4-maverick-17b-128e-instruct-maas",
                "meta/llama-4-maverick-17b-128e-instruct-maas",
            ),
            (
                "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
                "meta/llama-4-scout-17b-16e-instruct-maas",
            ),
            (
                "vertex/deepseek-ai/deepseek-v4.1-flash-maas",
                "deepseek-ai/deepseek-v4.1-flash-maas",
            ),
            (
                "vertex/qwen/qwen-synthetic-maas",
                "qwen/qwen-synthetic-maas",
            ),
        ] {
            let body = prepared(
                raw,
                &synthetic_full_request(raw, "max", json!({"type": "adaptive"})),
            );
            assert_eq!(body, expected_common(upstream), "{raw}");
        }
    }

    #[test]
    fn request_synthetic_reasoning_from_the_history_is_not_sent_back() {
        for raw in [
            "vertex/openai/gpt-oss-120b-maas",
            "vertex/zai-org/glm-5.2-maas",
            "vertex/deepseek-ai/deepseek-v4.1-flash-maas",
        ] {
            let body = prepared(
                raw,
                &synthetic_full_request(raw, "high", json!({"type": "adaptive"})),
            );
            let text = body.to_string();
            assert!(!text.contains("reasoning_content"), "{body}");
            assert!(!text.contains("Earlier reasoning"), "{body}");
            assert!(!text.contains("ccp:codex:v1:"), "{body}");
            assert!(!text.contains("c3ludGhldGljLXJlZGFjdGVk"), "{body}");
            assert!(text.contains("A go-between."), "{body}");
        }
    }

    fn with_tool_choice(raw: &str, tool_choice: Value) -> MessagesRequest {
        request(json!({
            "model": raw,
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "tool_choice": tool_choice,
            "messages": [{"role": "user", "content": "look it up"}]
        }))
    }

    #[test]
    fn tool_choice_synthetic_gpt_oss_refuses_any_and_a_named_tool() {
        let raw = "vertex/openai/gpt-oss-120b-maas";
        let model = resolve(raw).unwrap();
        for tool_choice in [
            json!({"type": "any"}),
            json!({"type": "any", "disable_parallel_tool_use": true}),
            json!({"type": "tool", "name": "lookup"}),
        ] {
            let error = prepare_request(&with_tool_choice(raw, tool_choice.clone()), &model)
                .expect_err("a forced tool choice is refused");
            assert_eq!(
                error.to_string(),
                "Vertex AI model openai/gpt-oss-120b-maas does not support a forced tool choice \
                 (tool_choice any or tool); use tool_choice auto or none",
                "{tool_choice}"
            );
        }
        for (tool_choice, sent) in [
            (json!({"type": "auto"}), json!("auto")),
            (json!({"type": "none"}), json!("none")),
        ] {
            let body = prepare_request(&with_tool_choice(raw, tool_choice), &model).unwrap();
            assert_eq!(body["tool_choice"], sent, "{body}");
        }
    }

    #[test]
    fn tool_choice_synthetic_other_models_send_every_choice() {
        for raw in [
            "vertex/zai-org/glm-5.2-maas",
            "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
            "vertex/openai/gpt-oss-20b-maas",
        ] {
            let model = resolve(raw).unwrap();
            for (tool_choice, sent) in [
                (json!({"type": "any"}), json!("required")),
                (
                    json!({"type": "tool", "name": "lookup"}),
                    json!({"type": "function", "function": {"name": "lookup"}}),
                ),
                (json!({"type": "auto"}), json!("auto")),
            ] {
                let body = prepare_request(&with_tool_choice(raw, tool_choice), &model)
                    .unwrap_or_else(|error| panic!("{raw}: {error}"));
                assert_eq!(body["tool_choice"], sent, "{raw}: {body}");
            }
        }
    }

    #[test]
    fn url_synthetic_uses_chat_completions_in_the_model_version_and_location() {
        for (raw, location, expected) in [
            (
                "vertex/openai/gpt-oss-120b-maas",
                "global",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/endpoints/openapi/chat/completions",
            ),
            (
                "vertex/zai-org/glm-5.2-maas",
                "us-central1",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/endpoints/openapi/chat/completions",
            ),
            (
                "vertex/meta/llama-4-maverick-17b-128e-instruct-maas",
                "us-east5",
                "https://us-east5-aiplatform.googleapis.com/v1beta1/projects/synthetic-project/locations/us-east5/endpoints/openapi/chat/completions",
            ),
            (
                "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
                "us-east5",
                "https://us-east5-aiplatform.googleapis.com/v1beta1/projects/synthetic-project/locations/us-east5/endpoints/openapi/chat/completions",
            ),
        ] {
            let model = resolve(raw).unwrap();
            let target = VertexTarget::new("synthetic-project", location, None).unwrap();
            assert_eq!(url(&target, &model).unwrap().as_str(), expected, "{raw}");
        }
    }

    // Streams and buffered reads.

    const SYNTHETIC_TOKEN: &str =
        "ya29.synthetic-open-model-token-0123456789abcdefghijklmnopqrstuvwx";

    const MODEL: &str = "vertex/zai-org/glm-5.2-maas";

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

    fn chunk(delta: Value, finish_reason: Option<&str>) -> String {
        frame(json!({
            "id": "synthetic", "object": "chat.completion.chunk", "model": "zai-org/glm-5.2-maas",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }))
    }

    /// Shaped like Google's open-model streaming example:
    /// `chat.completion.chunk` objects, the thinking in `reasoning_content`,
    /// a final chunk with `usage` and no choices, and `data: [DONE]`.
    fn thinking_wire() -> Vec<String> {
        vec![
            chunk(
                json!({"role": "assistant", "reasoning_content": "Thinking it over"}),
                None,
            ),
            chunk(json!({"content": "Hello"}), None),
            chunk(json!({}), Some("stop")),
            frame(json!({
                "id": "synthetic", "object": "chat.completion.chunk", "model": "zai-org/glm-5.2-maas",
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
            })),
            "data: [DONE]\n\n".to_string(),
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

    fn stream_of(wire: Vec<String>) -> Body {
        stream_body(
            stream::iter(chunks(wire)),
            "msg_synthetic".to_string(),
            MODEL.to_string(),
            &context(),
            sanitizer(),
        )
    }

    async fn buffered_of(wire: Vec<String>) -> Result<Value, BufferedFailure> {
        read_response(
            stream::iter(chunks(wire)),
            "msg_synthetic",
            MODEL,
            &sanitizer(),
            None,
        )
        .await
    }

    #[tokio::test]
    async fn stream_synthetic_reasoning_becomes_thinking_and_the_stream_ends_at_done() {
        let text = body_text(stream_of(thinking_wire())).await;
        assert!(!text.contains("event: error"), "{text}");
        assert!(text.contains("\"type\":\"thinking\""), "{text}");
        assert!(text.contains("\"thinking\":\"Thinking it over\""), "{text}");
        assert!(text.contains("\"text\":\"Hello\""), "{text}");
        assert!(text.contains("\"stop_reason\":\"end_turn\""), "{text}");
        assert!(text.contains("\"output_tokens\":2"), "{text}");
        assert!(
            text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn stream_synthetic_usage_on_the_finish_chunk_also_completes() {
        let mut finish: Value = serde_json::from_str(
            chunk(json!({}), Some("stop"))
                .trim_start_matches("data: ")
                .trim_end(),
        )
        .unwrap();
        finish["usage"] = json!({"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9});
        let wire = vec![
            chunk(json!({"role": "assistant", "content": "Hello"}), None),
            frame(finish),
            "data: [DONE]\n\n".to_string(),
        ];
        let text = body_text(stream_of(wire.clone())).await;
        assert!(!text.contains("event: error"), "{text}");
        assert!(text.contains("message_stop"), "{text}");
        let message = buffered_of(wire)
            .await
            .unwrap_or_else(|failure| panic!("{}", failure.kind()));
        assert_eq!(message["usage"]["input_tokens"], 7);
        assert_eq!(message["usage"]["output_tokens"], 2);
    }

    #[tokio::test]
    async fn stream_synthetic_content_after_the_finish_reason_is_an_error() {
        let mut wire = thinking_wire()[..3].to_vec();
        wire.push(chunk(json!({"content": "late"}), None));
        wire.push("data: [DONE]\n\n".to_string());
        let text = body_text(stream_of(wire)).await;
        assert!(text.contains("event: error"), "{text}");
        assert!(
            text.contains("Vertex AI open model stream is invalid"),
            "{text}"
        );
        assert!(!text.contains("OpenCode Go"), "{text}");
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test]
    async fn stream_synthetic_an_event_after_done_is_an_error() {
        let mut wire = thinking_wire();
        wire.push(chunk(json!({"content": "late"}), None));
        let text = body_text(stream_of(wire)).await;
        assert!(text.contains("event: error"), "{text}");
        assert!(text.contains("Vertex AI open model"), "{text}");
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test]
    async fn buffered_synthetic_completes_with_thinking_text_and_usage() {
        let message = buffered_of(thinking_wire())
            .await
            .unwrap_or_else(|failure| panic!("a finished stream is a message: {}", failure.kind()));
        assert_eq!(message["model"], MODEL);
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(message["content"][0]["type"], "thinking");
        assert_eq!(message["content"][0]["thinking"], "Thinking it over");
        assert_eq!(
            message["content"][1],
            json!({"type": "text", "text": "Hello"})
        );
        assert_eq!(message["usage"]["input_tokens"], 7);
        assert_eq!(message["usage"]["output_tokens"], 2);
    }

    #[tokio::test]
    async fn buffered_synthetic_content_after_the_finish_reason_is_an_error() {
        let mut wire = thinking_wire()[..3].to_vec();
        wire.push(chunk(json!({"content": "late"}), None));
        let failure = buffered_of(wire)
            .await
            .expect_err("content after finish_reason is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        assert_eq!(
            error.to_string(),
            "Vertex AI open model stream sent content after finish_reason"
        );
    }

    #[tokio::test]
    async fn buffered_synthetic_an_event_after_done_is_an_error() {
        let mut wire = thinking_wire();
        wire.push(chunk(json!({"content": "late"}), None));
        let failure = buffered_of(wire)
            .await
            .expect_err("an event after [DONE] is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        assert_eq!(error.to_string(), "Vertex AI open model event after [DONE]");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_done_then_open_upstream_completes_at_the_read_limit() {
        let upstream = stream::iter(chunks(thinking_wire())).chain(stream::pending());
        let started = tokio::time::Instant::now();
        let message = tokio::time::timeout(
            Duration::from_secs(60),
            read_response(upstream, "msg_synthetic", MODEL, &sanitizer(), None),
        )
        .await
        .expect("an upstream left open must not hold the response")
        .unwrap_or_else(|failure| {
            panic!(
                "[DONE] followed by silence is a message: {}",
                failure.kind()
            )
        });
        assert_eq!(started.elapsed(), Duration::from_secs(2));
        assert_eq!(message["content"][1]["text"], "Hello");
    }

    #[tokio::test]
    async fn buffered_synthetic_an_upstream_error_is_sanitised() {
        let wire = vec![frame(json!({
            "error": {"message": format!("saw {SYNTHETIC_TOKEN} in the request")}
        }))];
        let failure = buffered_of(wire)
            .await
            .expect_err("an error event is not a message");
        let BufferedFailure::Translate(error) = failure else {
            panic!("expected a translation failure, got {}", failure.kind());
        };
        let message = error.to_string();
        assert!(!message.contains(SYNTHETIC_TOKEN), "{message}");
        assert!(
            message.starts_with("Vertex AI open model upstream error: saw [redacted]"),
            "{message}"
        );
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

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_a_rejected_frame_ends_the_read_while_the_upstream_stays_open() {
        let keep_alive = ": keep-alive\n\n".to_string();
        let overloaded = frame(json!({
            "error": {"message": format!("overloaded, saw {SYNTHETIC_TOKEN}")}
        }));
        let malformed = "data: {\"choices\":[\n\n".to_string();
        let late = chunk(json!({"content": "late"}), None);
        let upstream_error = "Vertex AI open model upstream error: overloaded, saw [redacted]";
        let not_json = "malformed Vertex AI open model SSE event";
        let mut after_done = thinking_wire();
        after_done.push(late.clone());
        // The streaming path fails at these frames as soon as it reads them,
        // within the read window after a terminal event too, so the
        // buffered read must not wait for that window to pass.
        let tests = [
            (
                "upstream error before any finish_reason, upstream open",
                vec![thinking_wire()[0].clone(), overloaded.clone()],
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
                vec![thinking_wire()[0].clone(), malformed.clone()],
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
                vec![
                    thinking_wire()[0].clone(),
                    thinking_wire()[2].clone(),
                    late.clone(),
                ],
                None,
                "Vertex AI open model stream sent content after finish_reason",
            ),
            (
                "an event after [DONE], upstream open",
                after_done,
                None,
                "Vertex AI open model event after [DONE]",
            ),
        ];
        for (case, parts, after, expected) in tests {
            let count = parts.len();
            let (upstream, pulled) = synthetic_counted_upstream(chunks(parts), after);
            let started = tokio::time::Instant::now();
            let failure = tokio::time::timeout(
                Duration::from_secs(60),
                read_response(upstream, "msg_synthetic", MODEL, &sanitizer(), None),
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
    async fn buffered_synthetic_silence_before_the_finish_reason_is_not_cut() {
        for (case, after) in [
            ("reasoning, then silence", None),
            (
                "reasoning, then keep-alives",
                Some(": keep-alive\n\n".to_string()),
            ),
        ] {
            let parts = chunks(thinking_wire()[..2].to_vec());
            let (upstream, _) = synthetic_counted_upstream(parts, after);
            // Keep-alives that never stop would keep the read busy without
            // the clock moving; a small pause between them lets it move.
            let upstream = upstream.then(|item| async move {
                tokio::time::sleep(Duration::from_secs(1)).await;
                item
            });
            let pending = tokio::time::timeout(
                Duration::from_secs(600),
                read_response(upstream, "msg_synthetic", MODEL, &sanitizer(), None),
            )
            .await;
            assert!(
                pending.is_err(),
                "{case}: the read ended without a finish_reason"
            );
        }
    }

    #[tokio::test]
    async fn buffered_synthetic_a_broken_connection_is_a_transport_error() {
        let mut parts = chunks(thinking_wire()[..1].to_vec());
        parts.push(Err(std::io::Error::other("synthetic reset")));
        let failure = read_response(
            stream::iter(parts),
            "msg_synthetic",
            MODEL,
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
}
