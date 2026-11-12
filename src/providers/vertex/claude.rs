//! Anthropic Claude on Vertex AI: the Messages request is passed through to
//! `publishers/anthropic/models/{model}:streamRawPredict` (or `:rawPredict`
//! when the client did not ask for a stream).
//!
//! Vertex AI takes the Anthropic Messages body without `model` (the URL
//! names the model) and with `anthropic_version` in the body. It reads one
//! `anthropic-beta` header only, so every beta the client sent is filtered
//! and joined into one. The response is the Anthropic one: a stream goes to
//! the client as received, and a JSON answer is returned as it is.

use std::sync::Arc;

use axum::{
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::Engine;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use http::{HeaderMap, HeaderValue};
use serde_json::{Map, Value, json};

use crate::anthropic::schema::MessagesRequest;
use crate::provider::{
    Generation, GenerationBody, ProviderError, ProviderErrorKind, RequestContext,
};
use crate::providers::codex::native::NativeResponseOutcome;
use crate::providers::opencode::chat::{
    ReceivePolicy, StreamLabels, StreamWiring, TerminalPolicy, sanitize_upstream_messages,
};
use crate::providers::opencode::messages;

use super::client::{EVENT_STREAM, UpstreamCall, VertexError, VertexResponse, VertexTarget};
use super::models::ResolvedModel;
use super::sanitize::sanitize_vertex_text;
use super::translate::stream::Sanitizer;
use super::{
    MAX_BUFFERED_RESPONSE_BYTES, VertexProvider, error_response, invalid_request,
    report_error_events, sse_response, status_for_error_type,
};

/// The `anthropic_version` put in a body that has none. Vertex AI also
/// documents `2026-10-21`; a client that sends either keeps its own.
pub const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// `max_tokens` of a request that sets none. Vertex AI requires the field.
pub const DEFAULT_MAX_TOKENS: u32 = 32_000;

/// The beta values sent to Vertex AI; any other value the client sent is
/// dropped.
///
/// This is the list Claude Code itself sends when it talks to Vertex AI,
/// Bedrock or Mantle directly: Claude Code 2.1.322 filters its beta header
/// against exactly these values for those providers. Behind this proxy
/// Claude Code believes it talks to Anthropic and sends its whole list, so
/// the proxy applies the same filter. Refresh it from a newer Claude Code
/// release when Claude Code changes the set.
pub const VERTEX_BETAS: [&str; 14] = [
    "claude-code-20250219",
    "interleaved-thinking-2025-05-14",
    "context-1m-2025-08-07",
    "context-management-2025-06-27",
    "structured-outputs-2025-12-15",
    "web-search-2025-03-05",
    "effort-2025-11-24",
    "tool-search-tool-2025-10-19",
    "afk-mode-2026-01-31",
    "dangerous-tool-use-2026-09-03",
    "fallback-credit-2026-06-01",
    "thinking-token-count-2026-05-13",
    "thinking-binding-controls-2026-08-01",
    "thinking-display-updates-2026-08-18",
];

/// Names in stream error messages and traffic-capture files:
/// `Vertex AI Messages stream is invalid`,
/// `060-vertex-claude-messages-stream-error` and
/// `061-vertex-claude-messages-stream-summary`.
pub const STREAM_LABELS: StreamLabels = StreamLabels {
    provider: "Vertex AI",
    capture_stem: "vertex-claude",
};

/// The `accept` value of a request answered with one JSON message.
const JSON_MEDIA_TYPE: &str = "application/json";

/// Every thinking signature this proxy makes starts with this, either as
/// it is or after base64url decoding.
const PROXY_SIGNATURE_PREFIX: &str = "ccp:";

/// base64url with or without padding.
const BASE64URL: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::URL_SAFE,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// A Claude request that Vertex AI accepted: the body is still unread.
struct Started {
    upstream: VertexResponse,
    sanitize: Sanitizer,
}

/// The upstream method for a request: `streamRawPredict` for a stream,
/// `rawPredict` for one JSON answer.
pub const fn method(stream: bool) -> &'static str {
    if stream {
        "streamRawPredict"
    } else {
        "rawPredict"
    }
}

/// The upstream body: the client's request without `model`, with
/// [`ANTHROPIC_VERSION`] when it has no `anthropic_version`, with
/// [`DEFAULT_MAX_TOKENS`] when it has no `max_tokens`, and without the
/// thinking blocks Claude cannot verify: those this proxy signed (see
/// [`is_proxy_signature`]) and those with no signature, an empty or
/// whitespace-only one, or one that is not a string. Other providers behind
/// this proxy emit thinking blocks with an empty signature. A thinking block
/// with any other signature, and every `redacted_thinking` block, is sent
/// unchanged. Every other field, `stream` included, is the client's.
///
/// An assistant message whose content was only such thinking blocks is left
/// out. The Messages API combines consecutive turns of the same role, so the
/// history stays valid, whereas an empty content array would be rejected.
///
/// # Errors
///
/// Fails when the request cannot be serialised.
pub fn prepare_request(body: &MessagesRequest) -> Result<Value, serde_json::Error> {
    let mut translated = serde_json::to_value(body)?;
    if let Value::Object(object) = &mut translated {
        object.remove("model");
        if object.get("anthropic_version").is_none_or(Value::is_null) {
            object.insert("anthropic_version".to_string(), ANTHROPIC_VERSION.into());
        }
        if object.get("max_tokens").is_none_or(Value::is_null) {
            object.insert("max_tokens".to_string(), DEFAULT_MAX_TOKENS.into());
        }
        if let Some(Value::Array(messages)) = object.get_mut("messages") {
            drop_unverifiable_thinking(messages);
        }
    }
    Ok(translated)
}

/// Removes every thinking block Claude cannot verify, and every message
/// that this removal left without content.
fn drop_unverifiable_thinking(messages: &mut Vec<Value>) {
    messages.retain_mut(|message| {
        let Some(Value::Array(content)) = message.get_mut("content") else {
            return true;
        };
        let before = content.len();
        content.retain(|block| !is_unverifiable_thinking(block));
        !(content.is_empty() && before > 0)
    });
}

/// A thinking block whose signature is missing, not a string, blank, or
/// made by this proxy.
fn is_unverifiable_thinking(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("thinking")
        && block
            .get("signature")
            .and_then(Value::as_str)
            .is_none_or(|signature| signature.trim().is_empty() || is_proxy_signature(signature))
}

/// Whether this proxy made a thinking signature: it starts with `ccp:`
/// (the Codex, Muse and Vertex Gemini form), or decodes from base64url to
/// bytes that start with `ccp:` (the Kimi and OpenCode form). Claude cannot
/// verify such a signature, so the block is not sent. Anthropic's own
/// signatures are standard base64 of bytes that never start with `ccp:`.
pub fn is_proxy_signature(signature: &str) -> bool {
    signature.starts_with(PROXY_SIGNATURE_PREFIX)
        || BASE64URL
            .decode(signature)
            .is_ok_and(|bytes| bytes.starts_with(PROXY_SIGNATURE_PREFIX.as_bytes()))
}

/// Splits the client's beta values into those sent (members of
/// [`VERTEX_BETAS`]) and those dropped. Each list keeps the first
/// occurrence of a value, in the order received.
pub fn select_betas(values: &[String]) -> (Vec<&'static str>, Vec<&str>) {
    let mut sent: Vec<&'static str> = Vec::new();
    let mut dropped: Vec<&str> = Vec::new();
    for value in values {
        match VERTEX_BETAS.iter().find(|beta| **beta == value.as_str()) {
            Some(beta) if !sent.contains(beta) => sent.push(beta),
            Some(_) => {}
            None if !dropped.contains(&value.as_str()) => dropped.push(value),
            None => {}
        }
    }
    (sent, dropped)
}

/// The extra request headers: one `anthropic-beta` with the sent values
/// joined by commas, or none when no value is left. The dropped values are
/// logged in one debug line; dropping one is never an error.
fn beta_headers(values: &[String], req_id: &str) -> HeaderMap {
    let (sent, dropped) = select_betas(values);
    if !dropped.is_empty() {
        let fields = Map::from_iter([
            ("reqId".to_string(), json!(req_id)),
            ("dropped".to_string(), json!(dropped)),
        ]);
        crate::logging::create_logger("vertex").debug(
            "Vertex AI Claude request: beta values outside the Vertex list were not sent",
            Some(fields),
        );
    }
    let mut headers = HeaderMap::new();
    // The values are the list's own literals, so the header is valid.
    if !sent.is_empty()
        && let Ok(value) = HeaderValue::from_str(&sent.join(","))
    {
        headers.insert("anthropic-beta", value);
    }
    headers
}

/// The sanitiser for upstream prose in this response: the access token the
/// request carried is one of the secrets.
fn token_sanitizer(upstream: &VertexResponse) -> Sanitizer {
    let secrets: Vec<String> = upstream.access_token.iter().cloned().collect();
    Arc::new(move |text: &str| {
        let secrets: Vec<&str> = secrets.iter().map(String::as_str).collect();
        sanitize_vertex_text(text, &secrets)
    })
}

/// Builds and sends the request. The location comes from the model's rule;
/// a setting problem is reported before credentials are looked up. A 403 or
/// 404 carries [`VertexError::with_partner_guidance`].
async fn start(
    provider: &VertexProvider,
    body: &MessagesRequest,
    model: &ResolvedModel,
    ctx: &RequestContext,
) -> Result<Started, ProviderError> {
    if let Some(monitor) = ctx.monitor.as_ref() {
        monitor.model_resolved(&ctx.req_id, &model.upstream_model);
    }
    let translated = prepare_request(body).map_err(invalid_request)?;
    let ready = provider.ready()?;
    let target =
        VertexTarget::for_model(&ready.config, model).map_err(VertexError::into_provider_error)?;
    let url = target
        .model_method_url(model, method(body.stream))
        .map_err(VertexError::into_provider_error)?;
    let call = UpstreamCall {
        url,
        accept: if body.stream {
            EVENT_STREAM
        } else {
            JSON_MEDIA_TYPE
        },
        headers: beta_headers(&body.anthropic_beta, &ctx.req_id),
    };
    if let Some(monitor) = ctx.monitor.as_ref() {
        monitor.upstream_started(&ctx.req_id);
    }
    let credentials = VertexProvider::credentials(ready).await?;
    let upstream = ready
        .client
        .post(
            &target,
            &model.upstream_model,
            call,
            &translated,
            credentials,
            ctx.traffic.as_deref(),
        )
        .await
        .map_err(|error| {
            error
                .with_partner_guidance(&target, &model.upstream_model)
                .into_provider_error()
        })?;
    let sanitize = token_sanitizer(&upstream);
    Ok(Started { upstream, sanitize })
}

/// Answers a Messages request for a Claude model: the upstream stream as
/// received, or the upstream JSON message as it is.
pub(super) async fn handle_messages(
    provider: &VertexProvider,
    body: MessagesRequest,
    model: ResolvedModel,
    ctx: RequestContext,
) -> Response {
    let started = match start(provider, &body, &model, &ctx).await {
        Ok(started) => started,
        Err(error) => return error_response(error),
    };
    if body.stream {
        // A stream that ends in an error event is reported through the
        // outcome, which the server reads when the body ends, so the request
        // is recorded as failed. The wiring never forwards an upstream error
        // event: it answers with its own fixed message, so that is the text
        // recorded.
        let outcome = NativeResponseOutcome::default();
        let body = report_error_events(stream_body(started, &ctx), Some(outcome.clone()));
        let mut response = sse_response(body);
        response.extensions_mut().insert(outcome);
        response
    } else {
        buffered_response(started, &ctx).await
    }
}

/// The live Anthropic stream for the OpenAI-compatible routes.
pub(super) async fn generate_anthropic_stream(
    provider: &VertexProvider,
    mut body: MessagesRequest,
    model: ResolvedModel,
    ctx: RequestContext,
) -> Result<Generation, ProviderError> {
    body.stream = true;
    let started = start(provider, &body, &model, &ctx).await?;
    Ok(Generation {
        body: GenerationBody::LiveSse(stream_body(started, &ctx)),
        resolved_model: model.upstream_model,
    })
}

/// The stream wiring of Claude: Vertex labels, the strict rule after
/// `message_stop`, and the token sanitiser for captured upstream events.
fn wiring(sanitize: Sanitizer) -> StreamWiring {
    StreamWiring {
        labels: STREAM_LABELS,
        receive: ReceivePolicy {
            terminal: TerminalPolicy::Strict,
            sanitize: Some(sanitize),
        },
    }
}

fn stream_body(started: Started, ctx: &RequestContext) -> Body {
    messages::stream_body_with(
        started.upstream.into_byte_stream(),
        ctx.monitor.clone(),
        ctx.req_id.clone(),
        ctx.traffic.clone(),
        wiring(started.sanitize),
    )
}

/// Returns the upstream JSON message unchanged once it is read whole and
/// parses as JSON. The capture records it after the sanitiser, which
/// cleans every string under a `message` key, at any depth, as the
/// streaming path cleans the events it captures.
///
/// A body that is an Anthropic error object (`{"type": "error", "error":
/// {"type", "message"}}`) is not a message, whatever the status: it is
/// answered with the status [`status_for_error_type`] gives for its type
/// and its message, sanitised.
async fn buffered_response(started: Started, ctx: &RequestContext) -> Response {
    let Started { upstream, sanitize } = started;
    let bytes = match read_body(upstream.into_byte_stream()).await {
        Ok(bytes) => bytes,
        Err(error) => return error_response(error),
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return error_response(ProviderError::new(
            StatusCode::BAD_GATEWAY,
            ProviderErrorKind::Api,
            "Vertex AI returned a Claude response that is not JSON",
        ));
    };
    if value.get("type").and_then(Value::as_str) == Some("error") {
        let error = error_in_message_body(&value, &sanitize);
        if let Some(traffic) = ctx.traffic.as_ref() {
            traffic.write_json(
                "032-upstream-response-body",
                &sanitize_upstream_messages(&value, &*sanitize),
            );
            traffic.write_json(
                "051-downstream-response",
                &json!({
                    "type": "error",
                    "error": {"type": error.error_type(), "message": &error.message},
                }),
            );
        }
        return error_response(error);
    }
    if let Some(traffic) = ctx.traffic.as_ref() {
        let captured = sanitize_upstream_messages(&value, &*sanitize);
        traffic.write_json("032-upstream-response-body", &captured);
        traffic.write_json("051-downstream-response", &captured);
    }
    if let Some(monitor) = ctx.monitor.as_ref() {
        monitor.usage_updated(
            &ctx.req_id,
            value.pointer("/usage/input_tokens").and_then(Value::as_u64),
            value
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64),
        );
    }
    (
        StatusCode::OK,
        [(http::header::CONTENT_TYPE, JSON_MEDIA_TYPE)],
        Bytes::from(bytes),
    )
        .into_response()
}

/// The error a successful response reports in an Anthropic error body:
/// the status and kind of its `error.type`, and a message naming that type
/// and carrying its `error.message`, both sanitised.
fn error_in_message_body(value: &Value, sanitize: &Sanitizer) -> ProviderError {
    let error_type = value
        .pointer("/error/type")
        .and_then(Value::as_str)
        .unwrap_or("api_error");
    let (status, kind) = status_for_error_type(error_type);
    let detail = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(|message| sanitize(message))
        .filter(|message| !message.trim().is_empty());
    let named = sanitize(error_type);
    let message = match detail {
        Some(detail) => format!("Vertex AI returned a Claude {named}: {detail}"),
        None => format!("Vertex AI returned a Claude {named}"),
    };
    ProviderError::new(status, kind, message)
}

/// Reads the whole body. More than [`MAX_BUFFERED_RESPONSE_BYTES`] or a
/// broken connection is a 502.
async fn read_body<S>(upstream: S) -> Result<Vec<u8>, ProviderError>
where
    S: Stream<Item = Result<Bytes, VertexError>>,
{
    let mut upstream = std::pin::pin!(upstream);
    let mut body = Vec::new();
    while let Some(chunk) = upstream.next().await {
        let chunk = chunk.map_err(VertexError::into_provider_error)?;
        if body.len().saturating_add(chunk.len()) > MAX_BUFFERED_RESPONSE_BYTES {
            return Err(ProviderError::new(
                StatusCode::BAD_GATEWAY,
                ProviderErrorKind::Api,
                "Vertex AI Claude response exceeds 16 MiB",
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{
        Router,
        extract::{OriginalUri, State},
        routing::post,
    };
    use futures_util::stream;

    use super::*;
    use crate::config::{ResolvedVertexConfig, VertexSetting, VertexSettingSource};
    use crate::provider::Provider;
    use crate::providers::vertex::auth::VertexCredentials;
    use crate::providers::vertex::client::{VertexClient, client_builder};
    use crate::providers::vertex::models::resolve;

    const SYNTHETIC_TOKEN: &str =
        "ya29.a0Synthetic0Vertex0Claude0Token0Body0With0Mixed0Case0And0Digits0Qz7";

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn request_synthetic_drops_model_adds_the_version_and_keeps_every_other_field() {
        let body = request(json!({
            "model": "vertex/anthropic/claude-opus-5-5",
            "max_tokens": 1024,
            "stream": true,
            "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_synthetic", "content": "ok"}]}
            ],
            "tools": [{"name": "Read", "description": "read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "metadata": {"user_id": "synthetic-user"},
            "synthetic_unknown_field": {"nested": [1, 2.5, null, true, "x"]}
        }));
        assert_eq!(
            prepare_request(&body).expect("serialisable"),
            json!({
                "anthropic_version": "vertex-2023-10-16",
                "max_tokens": 1024,
                "stream": true,
                "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_synthetic", "content": "ok"}]}
                ],
                "tools": [{"name": "Read", "description": "read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "metadata": {"user_id": "synthetic-user"},
                "synthetic_unknown_field": {"nested": [1, 2.5, null, true, "x"]}
            })
        );
    }

    #[test]
    fn request_synthetic_keeps_the_client_version_and_stream_flag() {
        let body = request(json!({
            "model": "vertex/anthropic/claude-sonnet-5",
            "anthropic_version": "2026-10-21",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}]
        }));
        let translated = prepare_request(&body).expect("serialisable");
        assert_eq!(translated["anthropic_version"], "2026-10-21");
        assert_eq!(translated["stream"], false);
        assert!(translated.get("model").is_none(), "{translated}");
    }

    #[test]
    fn request_synthetic_without_max_tokens_gets_32000() {
        let body = request(json!({
            "model": "vertex/anthropic/claude-haiku-5-5",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }));
        let translated = prepare_request(&body).expect("serialisable");
        assert_eq!(translated["max_tokens"], 32000);
        assert_eq!(translated["stream"], true);
        assert_eq!(translated["anthropic_version"], "vertex-2023-10-16");
    }

    #[test]
    fn method_follows_the_stream_flag_on_the_documented_urls() {
        let url = |location: &str, raw: &str, stream: bool| {
            let model = resolve(raw).expect("a Claude model");
            VertexTarget::new("PROJECT_ID", location, None)
                .expect("valid target")
                .model_method_url(&model, method(stream))
                .expect("valid URL")
                .to_string()
        };
        assert_eq!(
            url("global", "vertex/anthropic/claude-opus-5-5", true),
            "https://aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/global/publishers/anthropic/models/claude-opus-5-5:streamRawPredict"
        );
        assert_eq!(
            url("global", "vertex/anthropic/claude-opus-5-5", false),
            "https://aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/global/publishers/anthropic/models/claude-opus-5-5:rawPredict"
        );
        assert_eq!(
            url("us", "vertex/anthropic/claude-sonnet-5-5", false),
            "https://aiplatform.us.rep.googleapis.com/v1/projects/PROJECT_ID/locations/us/publishers/anthropic/models/claude-sonnet-5-5:rawPredict"
        );
        assert_eq!(
            url("eu", "vertex/anthropic/claude-fable-5-2", true),
            "https://aiplatform.eu.rep.googleapis.com/v1/projects/PROJECT_ID/locations/eu/publishers/anthropic/models/claude-fable-5-2:streamRawPredict"
        );
        // The snapshot separator stays literal in the path.
        assert_eq!(
            url(
                "us-east5",
                "vertex/anthropic/claude-opus-4-5@20251101",
                false
            ),
            "https://us-east5-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-east5/publishers/anthropic/models/claude-opus-4-5@20251101:rawPredict"
        );
    }

    #[test]
    fn vertex_betas_are_the_list_claude_code_sends_to_vertex() {
        assert_eq!(
            VERTEX_BETAS,
            [
                "claude-code-20250219",
                "interleaved-thinking-2025-05-14",
                "context-1m-2025-08-07",
                "context-management-2025-06-27",
                "structured-outputs-2025-12-15",
                "web-search-2025-03-05",
                "effort-2025-11-24",
                "tool-search-tool-2025-10-19",
                "afk-mode-2026-01-31",
                "dangerous-tool-use-2026-09-03",
                "fallback-credit-2026-06-01",
                "thinking-token-count-2026-05-13",
                "thinking-binding-controls-2026-08-01",
                "thinking-display-updates-2026-08-18",
            ]
        );
    }

    #[test]
    fn betas_synthetic_keep_listed_values_once_and_drop_the_rest() {
        // The values of two headers, `a, b` and `c` style, as the server
        // splits them.
        let values = strings(&[
            "claude-code-20250219",
            "oauth-2025-04-20",
            "effort-2025-11-24",
            "claude-code-20250219",
            "synthetic-unknown-2099-01-01",
            "oauth-2025-04-20",
            "context-1m-2025-08-07",
        ]);
        let (sent, dropped) = select_betas(&values);
        assert_eq!(
            sent,
            vec![
                "claude-code-20250219",
                "effort-2025-11-24",
                "context-1m-2025-08-07"
            ]
        );
        assert_eq!(
            dropped,
            vec!["oauth-2025-04-20", "synthetic-unknown-2099-01-01"]
        );
        let headers = beta_headers(&values, "req_synthetic");
        let all: Vec<&HeaderValue> = headers.get_all("anthropic-beta").iter().collect();
        assert_eq!(
            all,
            vec![&HeaderValue::from_static(
                "claude-code-20250219,effort-2025-11-24,context-1m-2025-08-07"
            )]
        );
        assert_eq!(headers.len(), 1);
    }

    #[test]
    fn betas_synthetic_with_nothing_listed_send_no_header() {
        assert!(beta_headers(&[], "req_synthetic").is_empty());
        let values = strings(&["oauth-2025-04-20", "synthetic-unknown-2099-01-01"]);
        assert!(beta_headers(&values, "req_synthetic").is_empty());
        let (sent, dropped) = select_betas(&values);
        assert!(sent.is_empty());
        assert_eq!(
            dropped,
            vec!["oauth-2025-04-20", "synthetic-unknown-2099-01-01"]
        );
    }

    #[test]
    fn proxy_signatures_synthetic_are_recognised_in_both_forms() {
        for signature in [
            "ccp:codex:v1:synthetic-opaque",
            "ccp:muse:v1:synthetic-opaque",
            "ccp:vertex:v1:gemini-3.5-flash:c3ludGhldGlj",
            // base64url of `ccp:kimi:v1:msg_1:2`, without and with padding.
            "Y2NwOmtpbWk6djE6bXNnXzE6Mg",
            "Y2NwOmtpbWk6djE6bXNnXzE6Mg==",
            // base64url of `ccp:opencode:v1:msg_synthetic:0`.
            "Y2NwOm9wZW5jb2RlOnYxOm1zZ19zeW50aGV0aWM6MA",
        ] {
            assert!(is_proxy_signature(signature), "{signature}");
        }
        for signature in [
            // Shaped like Anthropic's: standard base64 with `+` and `/`.
            "ErUBCkYIBRgCIkA+synthetic/AnthropicSignature0123456789==",
            // Standard base64 of `not-a-proxy-signature-synthetic`.
            "bm90LWEtcHJveHktc2lnbmF0dXJlLXN5bnRoZXRpYw==",
            "cccp:synthetic",
        ] {
            assert!(!is_proxy_signature(signature), "{signature}");
        }
    }

    #[test]
    fn proxy_thinking_synthetic_is_removed_and_anthropic_thinking_is_kept() {
        let anthropic = "ErUBCkYIBRgCIkA+synthetic/AnthropicSignature0123456789==";
        let body = request(json!({
            "model": "vertex/anthropic/claude-opus-5-5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "codex reasoning", "signature": "ccp:codex:v1:synthetic-opaque"},
                    {"type": "text", "text": "first"}
                ]},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "kimi reasoning", "signature": "Y2NwOmtpbWk6djE6bXNnXzE6Mg"},
                    {"type": "tool_use", "id": "toolu_synthetic", "name": "Read", "input": {"path": "a"}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_synthetic", "content": "ok"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "claude reasoning", "signature": anthropic},
                    {"type": "redacted_thinking", "data": "synthetic-redacted"},
                    {"type": "text", "text": "done"}
                ]}
            ]
        }));
        let translated = prepare_request(&body).expect("serialisable");
        assert_eq!(
            translated["messages"],
            json!([
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "first"}]},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_synthetic", "name": "Read", "input": {"path": "a"}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_synthetic", "content": "ok"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "claude reasoning", "signature": anthropic},
                    {"type": "redacted_thinking", "data": "synthetic-redacted"},
                    {"type": "text", "text": "done"}
                ]}
            ])
        );
    }

    #[test]
    fn unsigned_thinking_synthetic_is_removed_and_signed_thinking_is_kept() {
        let anthropic = "ErUBCkYIBRgCIkA+synthetic/AnthropicSignature0123456789==";
        let body = request(json!({
            "model": "vertex/anthropic/claude-opus-5-5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "grok reasoning", "signature": ""},
                    {"type": "thinking", "thinking": "no signature field"},
                    {"type": "thinking", "thinking": "blank signature", "signature": " \t\n"},
                    {"type": "thinking", "thinking": "null signature", "signature": null},
                    {"type": "thinking", "thinking": "numeric signature", "signature": 7},
                    {"type": "text", "text": "first"}
                ]},
                {"role": "user", "content": "next"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "only unsigned reasoning", "signature": ""}
                ]},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "claude reasoning", "signature": anthropic},
                    {"type": "redacted_thinking", "data": "synthetic-redacted"},
                    {"type": "redacted_thinking", "data": ""},
                    {"type": "text", "text": "done"}
                ]}
            ]
        }));
        let translated = prepare_request(&body).expect("serialisable");
        assert_eq!(
            translated["messages"],
            json!([
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "first"}]},
                {"role": "user", "content": "next"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "claude reasoning", "signature": anthropic},
                    {"type": "redacted_thinking", "data": "synthetic-redacted"},
                    {"type": "redacted_thinking", "data": ""},
                    {"type": "text", "text": "done"}
                ]}
            ])
        );
    }

    #[test]
    fn message_left_empty_synthetic_is_dropped_and_one_sent_empty_is_kept() {
        let body = request(json!({
            "model": "vertex/anthropic/claude-opus-5-5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "only reasoning", "signature": "ccp:opencode:v1:synthetic"}
                ]},
                {"role": "user", "content": "go on"},
                {"role": "user", "content": []}
            ]
        }));
        let translated = prepare_request(&body).expect("serialisable");
        assert_eq!(
            translated["messages"],
            json!([
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "go on"},
                {"role": "user", "content": []}
            ])
        );
    }

    #[tokio::test]
    async fn buffered_body_synthetic_over_the_cap_is_a_502() {
        let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
        let chunks = (0..=MAX_BUFFERED_RESPONSE_BYTES / chunk.len())
            .map(move |_| Ok::<Bytes, VertexError>(chunk.clone()));
        let error = read_body(stream::iter(chunks))
            .await
            .expect_err("more than the cap is not a response");
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert_eq!(error.error_type(), "api_error");
        assert_eq!(error.message, "Vertex AI Claude response exceeds 16 MiB");
    }

    #[tokio::test]
    async fn buffered_body_synthetic_broken_connection_is_a_502() {
        let upstream = stream::iter([
            Ok(Bytes::from_static(b"{\"type\":")),
            Err(VertexError::stream_failed("Vertex AI stream broke off")),
        ]);
        let error = read_body(upstream)
            .await
            .expect_err("a broken connection is not a response");
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert_eq!(error.error_type(), "api_error");
    }

    // Against a local upstream.

    #[derive(Debug, Clone)]
    struct Seen {
        path: String,
        betas: Vec<String>,
        accept: Vec<String>,
        body: Value,
    }

    type Shared = Arc<Mutex<Vec<Seen>>>;

    const SYNTHETIC_SSE: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_vrtx_synthetic\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    const SYNTHETIC_MESSAGE: &str = "{\"id\":\"msg_vrtx_synthetic\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}],\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":727,\"output_tokens\":308}}";

    async fn spawn_upstream() -> (String, Shared) {
        async fn handler(
            State(seen): State<Shared>,
            OriginalUri(uri): OriginalUri,
            headers: http::HeaderMap,
            body: Bytes,
        ) -> Response {
            let values = |name: &str| {
                headers
                    .get_all(name)
                    .iter()
                    .map(|value| value.to_str().unwrap_or_default().to_string())
                    .collect::<Vec<_>>()
            };
            let path = uri.path().to_string();
            seen.lock().unwrap().push(Seen {
                path: path.clone(),
                betas: values("anthropic-beta"),
                accept: values("accept"),
                body: serde_json::from_slice(&body).unwrap_or(Value::Null),
            });
            if path.ends_with(":streamRawPredict") {
                (
                    [(http::header::CONTENT_TYPE, "text/event-stream")],
                    SYNTHETIC_SSE,
                )
                    .into_response()
            } else {
                (
                    [(http::header::CONTENT_TYPE, "application/json")],
                    SYNTHETIC_MESSAGE,
                )
                    .into_response()
            }
        }
        let seen: Shared = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/{*path}", post(handler))
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}"), seen)
    }

    fn local_provider(base_url: &str) -> VertexProvider {
        let setting = |value: &str, source| VertexSetting {
            value: value.to_string(),
            source,
        };
        let config = ResolvedVertexConfig {
            project_setting: Some(setting(
                "synthetic-project",
                VertexSettingSource::Env("CCP_VERTEX_PROJECT"),
            )),
            location_setting: setting("global", VertexSettingSource::Default),
            base_url_setting: Some(setting(
                base_url,
                VertexSettingSource::Env("CCP_VERTEX_BASE_URL"),
            )),
            location_overrides_setting: None,
        };
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            format!("Bearer {SYNTHETIC_TOKEN}").parse().unwrap(),
        );
        VertexProvider::with_client(
            VertexClient::with_client(client_builder().no_proxy().build().unwrap()),
            VertexCredentials::fixed(headers),
            config,
        )
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

    fn claude_request(stream: bool) -> MessagesRequest {
        let mut body = request(json!({
            "model": "vertex/anthropic/claude-opus-5-5",
            "max_tokens": 128,
            "stream": stream,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "gemini reasoning", "signature": "ccp:vertex:v1:gemini-3.5-flash:c3ludGhldGlj"},
                    {"type": "text", "text": "hi"}
                ]},
                {"role": "user", "content": "again"}
            ]
        }));
        // As the server splits `anthropic-beta: claude-code-20250219,
        // oauth-2025-04-20` and `anthropic-beta: effort-2025-11-24`.
        body.anthropic_beta = strings(&[
            "claude-code-20250219",
            "oauth-2025-04-20",
            "effort-2025-11-24",
        ]);
        body
    }

    fn expected_upstream_body(stream: bool) -> Value {
        json!({
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 128,
            "stream": stream,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": "again"}
            ]
        })
    }

    #[tokio::test]
    async fn streaming_synthetic_request_reaches_stream_raw_predict_and_passes_the_events_through()
    {
        let (base, seen) = spawn_upstream().await;
        let provider = local_provider(&base);
        let response = provider
            .handle_messages(claude_request(true), context())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(http::header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), SYNTHETIC_SSE);

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].path,
            "/v1/projects/synthetic-project/locations/global/publishers/anthropic/models/claude-opus-5-5:streamRawPredict"
        );
        assert_eq!(
            seen[0].betas,
            vec!["claude-code-20250219,effort-2025-11-24"]
        );
        assert_eq!(seen[0].accept, vec!["text/event-stream"]);
        assert_eq!(seen[0].body, expected_upstream_body(true));
    }

    #[tokio::test]
    async fn non_streaming_synthetic_request_reaches_raw_predict_and_returns_the_json_as_is() {
        let (base, seen) = spawn_upstream().await;
        let provider = local_provider(&base);
        let response = provider
            .handle_messages(claude_request(false), context())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(http::header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), SYNTHETIC_MESSAGE);

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].path,
            "/v1/projects/synthetic-project/locations/global/publishers/anthropic/models/claude-opus-5-5:rawPredict"
        );
        assert_eq!(
            seen[0].betas,
            vec!["claude-code-20250219,effort-2025-11-24"]
        );
        assert_eq!(seen[0].accept, vec!["application/json"]);
        assert_eq!(seen[0].body, expected_upstream_body(false));
    }

    #[tokio::test]
    async fn openai_route_synthetic_gets_the_live_stream_from_stream_raw_predict() {
        let (base, seen) = spawn_upstream().await;
        let provider = local_provider(&base);
        let mut body = claude_request(false);
        body.anthropic_beta.clear();
        let generation = provider
            .generate_anthropic_stream(body, context())
            .await
            .expect("the stream starts");
        assert_eq!(generation.resolved_model, "claude-opus-5-5");
        let GenerationBody::LiveSse(body) = generation.body else {
            panic!("a Claude stream is live");
        };
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), SYNTHETIC_SSE);
        let seen = seen.lock().unwrap().clone();
        assert!(seen[0].path.ends_with("/claude-opus-5-5:streamRawPredict"));
        // No listed beta: no header at all.
        assert!(seen[0].betas.is_empty(), "{:?}", seen[0].betas);
        assert_eq!(seen[0].body, expected_upstream_body(true));
    }

    #[tokio::test]
    async fn count_tokens_synthetic_for_claude_is_the_local_estimate() {
        let provider = VertexProvider::new();
        let body = request(json!({
            "model": "vertex/anthropic/claude-opus-5-5",
            "messages": [{"role": "user", "content": "hello world"}]
        }));
        let expected = crate::providers::kimi::count_tokens::count_tokens(&body);
        let response = provider.handle_count_tokens(body, context()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value, json!({"input_tokens": expected}));
        // Counting needs no settings, credentials or upstream.
        assert!(provider.ready.get().is_none());
    }

    const SYNTHETIC_ERROR_SSE: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_vrtx_synthetic\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
    );

    /// A local upstream that answers every request with `sse`.
    async fn spawn_sse_upstream(sse: &'static str) -> String {
        let app = Router::new().route(
            "/{*path}",
            post(move || async move {
                ([(http::header::CONTENT_TYPE, "text/event-stream")], sse).into_response()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn streaming_synthetic_error_event_is_recorded_as_a_failure_and_a_clean_stream_is_not() {
        let base = spawn_sse_upstream(SYNTHETIC_ERROR_SSE).await;
        let response = local_provider(&base)
            .handle_messages(claude_request(true), context())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let outcome = response
            .extensions()
            .get::<NativeResponseOutcome>()
            .cloned()
            .expect("a streaming answer carries an outcome");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("event: error"), "{text}");
        assert!(!text.contains("Overloaded"), "{text}");
        assert_eq!(
            outcome.failure().as_deref(),
            Some("Vertex AI Messages stream is invalid")
        );

        let base = spawn_sse_upstream(SYNTHETIC_SSE).await;
        let response = local_provider(&base)
            .handle_messages(claude_request(true), context())
            .await;
        let outcome = response
            .extensions()
            .get::<NativeResponseOutcome>()
            .cloned()
            .expect("a streaming answer carries an outcome");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), SYNTHETIC_SSE);
        assert_eq!(outcome.failure(), None);
    }

    /// A local upstream that answers every request with `body` as JSON and
    /// status 200.
    async fn spawn_json_upstream(body: String) -> String {
        let app = Router::new().route(
            "/{*path}",
            post(move || {
                let body = body.clone();
                async move { ([(http::header::CONTENT_TYPE, "application/json")], body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn non_streaming_synthetic_error_body_with_status_200_is_an_error_response() {
        let echoed = format!("saw {SYNTHETIC_TOKEN}");
        let tests = [
            (
                "rate_limit_error",
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
            (
                "invalid_request_error",
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            ("overloaded_error", StatusCode::BAD_GATEWAY, "api_error"),
            ("api_error", StatusCode::BAD_GATEWAY, "api_error"),
        ];
        for (upstream_type, status, error_type) in tests {
            let body = json!({
                "type": "error",
                "error": {"type": upstream_type, "message": echoed},
            })
            .to_string();
            let base = spawn_json_upstream(body).await;
            let provider = local_provider(&base);
            let response = provider
                .handle_messages(claude_request(false), context())
                .await;
            assert_eq!(response.status(), status, "{upstream_type}");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["type"], "error", "{upstream_type}: {value}");
            assert_eq!(
                value["error"]["type"], error_type,
                "{upstream_type}: {value}"
            );
            assert_eq!(
                value["error"]["message"],
                format!("Vertex AI returned a Claude {upstream_type}: saw [redacted]"),
                "{upstream_type}"
            );
            assert!(
                !String::from_utf8_lossy(&bytes).contains(SYNTHETIC_TOKEN),
                "{upstream_type}"
            );
        }

        // Without a type or a message the error is still not a message.
        let base = spawn_json_upstream(json!({"type": "error"}).to_string()).await;
        let response = local_provider(&base)
            .handle_messages(claude_request(false), context())
            .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["error"]["message"],
            "Vertex AI returned a Claude api_error"
        );
    }
}
