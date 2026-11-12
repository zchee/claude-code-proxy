//! Models the user deployed from Model Garden, Hugging Face models included,
//! to a Vertex AI endpoint: Anthropic Messages requests sent as OpenAI Chat
//! Completions to the endpoint's
//! `{host}/v1/projects/{project}/locations/{location}/endpoints/{endpoint}/chat/completions`.
//!
//! An endpoint is named in `vertex.endpoints` in `config.json` and requested
//! as `vertex/endpoint/<name>`. The proxy only calls it: it never deploys a
//! model, creates, changes or deletes an endpoint, or enables a model.
//!
//! The request goes to the endpoint's dedicated host: the configured
//! `host`, or else the `dedicatedEndpointDns` that `endpoints.get` reports,
//! read once per endpoint and kept for the life of the process. A shared
//! endpoint is refused, because it does not stream (no SSE) and limits a
//! request to 60 seconds and 1.5 MB, and the upstream is always asked for a
//! stream. The access token goes only to the configured Vertex base URL or
//! the regional Google API host (for `endpoints.get`) and to a validated
//! dedicated host (for the request).
//!
//! Whether the serving container accepts Chat Completions and tool calls
//! cannot be read from the endpoint, so tools are sent as the translator
//! writes them and an upstream refusal is returned as it comes.

use std::collections::HashMap;
use std::sync::Mutex;

use axum::body::Body;
use bytes::Bytes;
use futures_util::Stream;
use http::StatusCode;
use serde_json::{Value, json};

use crate::anthropic::schema::MessagesRequest;
use crate::config::{
    ResolvedVertexConfig, VERTEX_ENDPOINTS_KEY, VertexEndpointConfig, VertexEndpoints,
    normalize_vertex_dedicated_host,
};
use crate::provider::{ProviderErrorKind, RequestContext};
use crate::providers::opencode::chat::{
    self as translator, EffortPolicy, ReasoningReplay, ReceivePolicy, RedactedWiring,
    RequestPolicy, StreamLabels, StreamWiring, TerminalPolicy,
};
use crate::traffic::TrafficCapture;

use super::BufferedCapture;
use super::auth::VertexCredentials;
use super::buffered::{BufferedFailure, EventStreamSink, read_upstream};
use super::client::{
    EVENT_STREAM, UpstreamCall, UpstreamErrorDetail, VertexClient, VertexError, VertexResponse,
    VertexTarget, redact_for_capture,
};
use super::models::{ApiVersion, MODEL_PREFIX};
use super::translate::stream::Sanitizer;

/// Names in error messages and capture files: `Vertex AI endpoint stream is
/// invalid` and `061-vertex-endpoint-stream-summary`, which collides with no
/// other family's.
const LABELS: StreamLabels = StreamLabels {
    provider: "Vertex AI endpoint",
    capture_stem: "vertex-endpoint",
};

/// The translator neither sets `reasoning_effort` nor sends reasoning back.
/// A serving container's controls are unknown, so no effort is sent, and
/// the thinking text in a history may have been written by another
/// provider.
const REQUEST_POLICY: RequestPolicy = RequestPolicy {
    effort: EffortPolicy::Untouched,
    reasoning_replay: ReasoningReplay::Never,
};

/// Capture names of the endpoint lookup. They sort before the request's own
/// `020-upstream-request` files.
const LOOKUP_CAPTURE_STEM: &str = "018-upstream-endpoint-lookup";

/// What Google documents a scale-to-zero endpoint answering, with HTTP 429,
/// while no replica is ready. It is a start-up, not a rate limit.
const NOT_READY_TEXT: &str = "model is not yet ready for inference";

/// One configured endpoint as a request uses it.
#[derive(Debug, Clone, Copy)]
pub(super) struct EndpointCall<'a> {
    /// The name in `vertex.endpoints` and in `vertex/endpoint/<name>`.
    pub(super) name: &'a str,
    /// The endpoint's validated settings.
    pub(super) config: &'a VertexEndpointConfig,
    /// The endpoint's project: its own setting, else the Vertex project.
    pub(super) project: &'a str,
}

impl EndpointCall<'_> {
    /// `vertex/endpoint/<name>`.
    fn model_name(&self) -> String {
        format!("{MODEL_PREFIX}endpoint/{}", self.name)
    }

    /// Where the endpoint is, for messages.
    fn whereabouts(&self) -> String {
        format!(
            "endpoint {} in project {}, location {} ({VERTEX_ENDPOINTS_KEY}.{} in config.json)",
            self.config.endpoint, self.project, self.config.location, self.name
        )
    }
}

/// The dedicated hosts `endpoints.get` reported, by endpoint name. Only a
/// successful lookup is kept, so a lookup that failed is tried again by the
/// next request.
#[derive(Debug, Default)]
pub(super) struct DedicatedHosts {
    hosts: Mutex<HashMap<String, reqwest::Url>>,
}

impl DedicatedHosts {
    fn get(&self, name: &str) -> Option<reqwest::Url> {
        self.hosts.lock().ok()?.get(name).cloned()
    }

    /// Keeps `host` for `name`. Concurrent first requests may each look the
    /// endpoint up; the first answer kept is the one every later request
    /// uses.
    fn insert(&self, name: &str, host: reqwest::Url) -> reqwest::Url {
        match self.hosts.lock() {
            Ok(mut hosts) => hosts.entry(name.to_string()).or_insert(host).clone(),
            Err(_) => host,
        }
    }
}

/// The settings of the endpoint named `name`.
///
/// # Errors
///
/// No entry has the name: 400, listing the configured names. The entry or
/// the whole setting is invalid: [`VertexError::configuration`] with the
/// message that names the setting and the entry.
pub(super) fn find<'a>(
    endpoints: &'a VertexEndpoints,
    name: &str,
) -> Result<&'a VertexEndpointConfig, VertexError> {
    if endpoints.unreadable
        && let Some(error) = endpoints.first_error()
    {
        return Err(VertexError::configuration(error));
    }
    match endpoints.get(name) {
        Some(entry) => entry.config.as_ref().map_err(VertexError::configuration),
        None => {
            let configured: Vec<&str> = endpoints.usable_names().collect();
            let configured = if configured.is_empty() {
                "none".to_string()
            } else {
                configured.join(", ")
            };
            Err(VertexError::invalid_request(format!(
                "Vertex AI endpoint {name:?} is not configured: add \"{name}\": {{\"endpoint\": \"<endpoint-id>\", \"location\": \"<region>\", \"model\": \"<model the container serves>\"}} to {VERTEX_ENDPOINTS_KEY} in config.json and restart the proxy; configured endpoints: {configured}"
            )))
        }
    }
}

/// The Chat Completions request for `body`, sent with the configured
/// `model`, `stream: true` and `stream_options.include_usage`, and without
/// `reasoning_effort`. Tools are sent as the translator writes them.
///
/// # Errors
///
/// The request cannot be expressed as Chat Completions, or it carries tools
/// and the endpoint is configured with `"tools": false`.
pub(super) fn prepare_request(
    body: &MessagesRequest,
    call: &EndpointCall,
) -> anyhow::Result<Value> {
    let model = call.config.model.as_str();
    let translated = translator::prepare_request_with(body, model, REQUEST_POLICY)?;
    let mut request = serde_json::to_value(translated)?;
    let Some(object) = request.as_object_mut() else {
        anyhow::bail!("the endpoint request is not a JSON object");
    };
    let carries_tools = object
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if carries_tools && !call.config.tools {
        anyhow::bail!(
            "Vertex AI endpoint {} is configured with \"tools\": false in {VERTEX_ENDPOINTS_KEY}, so a request that carries tools is not sent to it; send the request without tools, or set \"tools\": true once the endpoint's serving container supports tool calling",
            call.name
        );
    }
    object.remove("reasoning_effort");
    object.insert("model".to_string(), Value::from(model));
    object.insert("stream".to_string(), Value::Bool(true));
    Ok(request)
}

/// `{host}/v1/projects/{project}/locations/{location}/endpoints/{endpoint}/chat/completions`,
/// with the path appended to the host once.
///
/// # Errors
///
/// A value that cannot be placed in the path.
pub(super) fn chat_url(
    host: &reqwest::Url,
    call: &EndpointCall,
) -> Result<(VertexTarget, reqwest::Url), VertexError> {
    let target = VertexTarget::new(call.project, &call.config.location, Some(host.as_str()))?;
    let url = target.versioned_location_url(
        ApiVersion::V1,
        ["endpoints", &call.config.endpoint, "chat", "completions"],
    )?;
    Ok((target, url))
}

/// `{base}/v1/projects/{project}/locations/{location}/endpoints/{endpoint}`,
/// where `{base}` is the configured Vertex base URL or the regional host of
/// the endpoint's location.
///
/// # Errors
///
/// An invalid base URL, or a value that cannot be placed in the path.
pub(super) fn lookup_url(
    vertex: &ResolvedVertexConfig,
    call: &EndpointCall,
) -> Result<(VertexTarget, reqwest::Url), VertexError> {
    let base_url = vertex.base_url().map_err(VertexError::configuration)?;
    let target = VertexTarget::new(call.project, &call.config.location, base_url)?;
    let url =
        target.versioned_location_url(ApiVersion::V1, ["endpoints", &call.config.endpoint])?;
    Ok((target, url))
}

/// Sends `body` to the endpoint and asks for an event stream. The dedicated
/// host is found first ([`dedicated_host`]); nothing is sent to the
/// endpoint's inference URL unless that succeeds.
///
/// # Errors
///
/// The host cannot be found, or the request fails ([`inference_error`]).
pub(super) async fn send(
    client: &VertexClient,
    vertex: &ResolvedVertexConfig,
    hosts: &DedicatedHosts,
    call: &EndpointCall<'_>,
    body: &Value,
    credentials: &VertexCredentials,
    traffic: Option<&TrafficCapture>,
) -> Result<VertexResponse, VertexError> {
    let host = dedicated_host(client, vertex, hosts, call, credentials, traffic).await?;
    let (target, url) = chat_url(&host, call)?;
    let upstream_call = UpstreamCall {
        url,
        accept: EVENT_STREAM,
        headers: http::HeaderMap::new(),
    };
    client
        .post_mapped(
            upstream_call,
            body,
            credentials,
            traffic,
            |status, detail| inference_error(status, detail, &target, call),
        )
        .await
}

/// The endpoint's dedicated host: the configured `host`, one reported
/// earlier, or the one `endpoints.get` reports now.
///
/// # Errors
///
/// The lookup fails ([`lookup_error`]), or its answer does not name a
/// dedicated host the token may be sent to ([`host_from_metadata`]).
async fn dedicated_host(
    client: &VertexClient,
    vertex: &ResolvedVertexConfig,
    hosts: &DedicatedHosts,
    call: &EndpointCall<'_>,
    credentials: &VertexCredentials,
    traffic: Option<&TrafficCapture>,
) -> Result<reqwest::Url, VertexError> {
    if let Some(host) = &call.config.host {
        return reqwest::Url::parse(host).map_err(|_| {
            VertexError::configuration(format!(
                "the host of {VERTEX_ENDPOINTS_KEY}.{} is not a valid URL",
                call.name
            ))
        });
    }
    if let Some(host) = hosts.get(call.name) {
        return Ok(host);
    }
    let (target, url) = lookup_url(vertex, call)?;
    let (metadata, token) = client
        .get_json_and_token(
            url,
            credentials,
            traffic,
            LOOKUP_CAPTURE_STEM,
            |status, detail| lookup_error(status, detail, &target, call),
        )
        .await?;
    if let Some(capture) = traffic {
        // Only the two fields the proxy reads: the rest of the metadata
        // describes the user's deployment and is not needed to debug a
        // request. They are written before they are checked, so a value
        // that is refused is recorded too, and the answer may echo the
        // token anywhere in them, keys included.
        capture.write_json(
            &format!("{LOOKUP_CAPTURE_STEM}-fields"),
            &redact_for_capture(
                &json!({
                    "dedicatedEndpointEnabled": metadata.get("dedicatedEndpointEnabled"),
                    "dedicatedEndpointDns": metadata.get("dedicatedEndpointDns"),
                }),
                token.as_deref(),
            ),
        );
    }
    let host = host_from_metadata(&metadata, call)?;
    Ok(hosts.insert(call.name, host))
}

/// Reads `dedicatedEndpointEnabled` and `dedicatedEndpointDns` from an
/// `endpoints.get` answer. The API reference describes the DNS value as a
/// full `https://{endpoint_id}.{region}-{uid}.prediction.vertexai.goog`
/// URL, while Google's client library puts `https://` in front of it, so
/// both forms are taken by the rule a configured `host` follows
/// ([`normalize_vertex_dedicated_host`]): a URL as it is, a DNS name alone
/// with `https://`.
///
/// # Errors
///
/// The endpoint does not say it is dedicated: 400, saying a dedicated
/// endpoint is required. It does, but the DNS value is missing or is not a
/// URL the token may be sent to: 502. The shared host is never used
/// instead: a dedicated endpoint does not answer there.
fn host_from_metadata(metadata: &Value, call: &EndpointCall) -> Result<reqwest::Url, VertexError> {
    let name = call.name;
    if metadata
        .get("dedicatedEndpointEnabled")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(VertexError::invalid_request(format!(
            "Vertex AI endpoint {name} is not a dedicated endpoint ({}): endpoints.get did not report dedicatedEndpointEnabled: true. A shared endpoint does not stream (no SSE) and limits a request to 60 seconds and 1.5 MB, so claude-code-proxy needs a dedicated endpoint: deploy the model to a dedicated endpoint, or, if this endpoint is one, set \"host\" in {VERTEX_ENDPOINTS_KEY}.{name} to its dedicated URL",
            call.whereabouts()
        )));
    }
    let unusable = |reason: &str| VertexError {
        kind: ProviderErrorKind::Api,
        status: StatusCode::BAD_GATEWAY,
        upstream_status: Some(StatusCode::OK),
        message: format!(
            "Vertex AI endpoint {name} is a dedicated endpoint, but endpoints.get returned {reason} ({}). The proxy sends the access token only to https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog and never to the shared host, which a dedicated endpoint does not answer on; set \"host\" in {VERTEX_ENDPOINTS_KEY}.{name} to the endpoint's dedicated URL",
            call.whereabouts()
        ),
        retry_after: None,
    };
    let Some(dns) = metadata
        .get("dedicatedEndpointDns")
        .and_then(Value::as_str)
        .filter(|dns| !dns.trim().is_empty())
    else {
        return Err(unusable("no dedicatedEndpointDns"));
    };
    let origin = normalize_vertex_dedicated_host(dns.trim())
        .map_err(|reason| unusable(&format!("a dedicatedEndpointDns that {reason}")))?;
    reqwest::Url::parse(&origin).map_err(|_| unusable("a dedicatedEndpointDns that is not a URL"))
}

/// `text`, or the bare status when the error body had nothing to say.
fn upstream_text(status: StatusCode, detail: &UpstreamErrorDetail) -> String {
    detail
        .text()
        .unwrap_or_else(|| format!("Vertex AI returned HTTP {}", status.as_u16()))
}

fn mapped(
    kind: ProviderErrorKind,
    status: StatusCode,
    upstream: StatusCode,
    message: String,
) -> VertexError {
    VertexError {
        kind,
        status,
        upstream_status: Some(upstream),
        message,
        retry_after: None,
    }
}

/// A failed `endpoints.get`. A 403 says the permission the read needs and
/// that setting `host` makes the read unnecessary; a 404 names the
/// endpoint, project and location. Every other status is mapped as for any
/// Vertex AI request, with where the endpoint is appended.
fn lookup_error(
    upstream: StatusCode,
    detail: UpstreamErrorDetail,
    target: &VertexTarget,
    call: &EndpointCall,
) -> VertexError {
    let text = upstream_text(upstream, &detail);
    let name = call.name;
    let whereabouts = call.whereabouts();
    match upstream.as_u16() {
        403 => mapped(
            ProviderErrorKind::Permission,
            StatusCode::FORBIDDEN,
            upstream,
            format!(
                "{text}. Reading Vertex AI {whereabouts} with endpoints.get needs the permission aiplatform.endpoints.get (included in roles/aiplatform.user): grant it, or set \"host\" in {VERTEX_ENDPOINTS_KEY}.{name} to the endpoint's dedicated URL so the proxy does not read the endpoint"
            ),
        ),
        404 => mapped(
            ProviderErrorKind::InvalidRequest,
            StatusCode::NOT_FOUND,
            upstream,
            format!(
                "Vertex AI {whereabouts} was not found: {text}. Check endpoint, location and project in {VERTEX_ENDPOINTS_KEY}.{name}"
            ),
        ),
        _ => with_whereabouts(
            VertexError::from_upstream(upstream, detail, target, &call.model_name()),
            &whereabouts,
        ),
    }
}

/// A failed request to the endpoint's inference URL.
///
/// - 403: the permission the request needs.
/// - 404: the endpoint, project and location, and the configured host.
/// - 429 whose message says the model is not yet ready for inference: a
///   503 `api_error` saying the endpoint is starting, as a scale-to-zero
///   endpoint answers while no replica is ready.
/// - Any other 429: `rate_limit_error`, naming the endpoint.
/// - Every other status: mapped as for any Vertex AI request, with where
///   the endpoint is appended.
fn inference_error(
    upstream: StatusCode,
    detail: UpstreamErrorDetail,
    target: &VertexTarget,
    call: &EndpointCall,
) -> VertexError {
    let text = upstream_text(upstream, &detail);
    let name = call.name;
    let whereabouts = call.whereabouts();
    let retry_after = detail.retry_after_secs.map(|secs| secs.to_string());
    match upstream.as_u16() {
        403 => mapped(
            ProviderErrorKind::Permission,
            StatusCode::FORBIDDEN,
            upstream,
            format!(
                "{text}. Check that the account has the permission aiplatform.endpoints.predict (included in roles/aiplatform.user) for Vertex AI {whereabouts}"
            ),
        ),
        404 => mapped(
            ProviderErrorKind::InvalidRequest,
            StatusCode::NOT_FOUND,
            upstream,
            format!(
                "Vertex AI {whereabouts} was not found at its dedicated host: {text}. Check endpoint, location and project in {VERTEX_ENDPOINTS_KEY}.{name}, and, when \"host\" is set there, that it is this endpoint's dedicated URL"
            ),
        ),
        429 if text.to_ascii_lowercase().contains(NOT_READY_TEXT) => VertexError {
            retry_after,
            ..mapped(
                ProviderErrorKind::Api,
                StatusCode::SERVICE_UNAVAILABLE,
                upstream,
                format!(
                    "Vertex AI endpoint {name} is starting: {text}. An endpoint that scales to zero answers this until a replica is ready; it is not a rate limit. Retry in a few minutes"
                ),
            )
        },
        429 => VertexError {
            retry_after,
            ..mapped(
                ProviderErrorKind::RateLimit,
                StatusCode::TOO_MANY_REQUESTS,
                upstream,
                format!(
                    "{text}. Vertex AI {whereabouts} is rate limited or its serving container is overloaded; retry later"
                ),
            )
        },
        _ => with_whereabouts(
            VertexError::from_upstream(upstream, detail, target, &call.model_name()),
            &whereabouts,
        ),
    }
}

fn with_whereabouts(mut error: VertexError, whereabouts: &str) -> VertexError {
    error
        .message
        .push_str(&format!(" (Vertex AI {whereabouts})"));
    error
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

/// The Anthropic SSE body translated from the endpoint's Chat Completions
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

/// Reads the whole endpoint stream with the shared buffered limits and
/// accumulates it into one Anthropic message. The chunks are checked as
/// they arrive ([`translator::StrictChatObserver`]): a frame the streaming
/// path rejects, such as an upstream error event, ends the read at once,
/// even while the upstream stays open. Silence before the terminal event is
/// not cut.
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

    use super::*;
    use crate::providers::vertex::translate::stream::StreamError;

    const SYNTHETIC_TOKEN: &str =
        "ya29.synthetic-endpoint-token-0123456789abcdefghijklmnopqrstuvwxyz";

    /// The documented dedicated DNS form,
    /// `https://{endpoint_id}.{region}-{uid}.prediction.vertexai.goog`.
    const SYNTHETIC_DEDICATED: &str =
        "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog";

    fn synthetic_config(host: Option<&str>, tools: bool) -> VertexEndpointConfig {
        VertexEndpointConfig {
            endpoint: "1234567890123456789".to_string(),
            location: "us-central1".to_string(),
            model: "gemma-2-9b-it".to_string(),
            project: None,
            host: host.map(str::to_string),
            tools,
        }
    }

    fn call(config: &VertexEndpointConfig) -> EndpointCall<'_> {
        EndpointCall {
            name: "my-llm",
            config,
            project: "synthetic-project",
        }
    }

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    /// System prompt, effort, thinking, one tool and a history with a
    /// thinking block.
    fn synthetic_full_request() -> MessagesRequest {
        request(json!({
            "model": "vertex/endpoint/my-llm",
            "max_tokens": 1_024,
            "system": "Be brief.",
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
            "tools": [{
                "name": "lookup",
                "description": "Looks a word up.",
                "input_schema": {"type": "object", "properties": {"word": {"type": "string"}}}
            }],
            "messages": [
                {"role": "user", "content": "Define proxy."},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Earlier reasoning", "signature": "synthetic-signature"},
                    {"type": "text", "text": "A go-between."}
                ]},
                {"role": "user", "content": "Shorter."}
            ]
        }))
    }

    #[test]
    fn request_synthetic_uses_the_configured_model_and_sends_no_effort() {
        let config = synthetic_config(None, true);
        let body = prepare_request(&synthetic_full_request(), &call(&config)).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "gemma-2-9b-it",
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
        );
    }

    #[test]
    fn request_synthetic_tools_off_refuses_a_request_with_tools_and_sends_one_without() {
        let config = synthetic_config(None, false);
        let error = prepare_request(&synthetic_full_request(), &call(&config)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vertex AI endpoint my-llm is configured with \"tools\": false in vertex.endpoints, so a \
             request that carries tools is not sent to it; send the request without tools, or set \
             \"tools\": true once the endpoint's serving container supports tool calling"
        );
        let plain = request(json!({
            "model": "vertex/endpoint/my-llm",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let body = prepare_request(&plain, &call(&config)).unwrap();
        assert_eq!(body["model"], "gemma-2-9b-it");
        assert!(body.get("tools").is_none(), "{body}");
    }

    #[test]
    fn urls_synthetic_put_the_path_after_the_host_once() {
        let config = synthetic_config(None, true);
        for host in [
            SYNTHETIC_DEDICATED.to_string(),
            format!("{SYNTHETIC_DEDICATED}/"),
        ] {
            let host = reqwest::Url::parse(&host).unwrap();
            let (_, url) = chat_url(&host, &call(&config)).unwrap();
            assert_eq!(
                url.as_str(),
                "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog/v1/projects/synthetic-project/locations/us-central1/endpoints/1234567890123456789/chat/completions"
            );
        }

        let setting = |value: &str| crate::config::VertexSetting {
            value: value.to_string(),
            source: crate::config::VertexSettingSource::Env("CCP_VERTEX_PROJECT"),
        };
        let mut vertex = ResolvedVertexConfig {
            project_setting: Some(setting("synthetic-project")),
            location_setting: setting("global"),
            base_url_setting: None,
            location_overrides_setting: None,
        };
        // The endpoint's own region, not the Vertex location.
        let (_, url) = lookup_url(&vertex, &call(&config)).unwrap();
        assert_eq!(
            url.as_str(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/endpoints/1234567890123456789"
        );
        vertex.base_url_setting = Some(setting("http://127.0.0.1:9/relay"));
        let (_, url) = lookup_url(&vertex, &call(&config)).unwrap();
        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:9/relay/v1/projects/synthetic-project/locations/us-central1/endpoints/1234567890123456789"
        );
    }

    #[test]
    fn metadata_synthetic_the_dedicated_flag_decides_and_a_dedicated_endpoint_needs_its_dns() {
        let config = synthetic_config(None, true);
        let host = host_from_metadata(
            &json!({
                "name": "projects/123/locations/us-central1/endpoints/1234567890123456789",
                "dedicatedEndpointEnabled": true,
                "dedicatedEndpointDns": SYNTHETIC_DEDICATED,
            }),
            &call(&config),
        )
        .unwrap();
        assert_eq!(host.as_str(), format!("{SYNTHETIC_DEDICATED}/"));

        let shared = "Vertex AI endpoint my-llm is not a dedicated endpoint (endpoint \
                      1234567890123456789 in project synthetic-project, location us-central1 \
                      (vertex.endpoints.my-llm in config.json)): endpoints.get did not report \
                      dedicatedEndpointEnabled: true. A shared endpoint does not stream (no SSE) \
                      and limits a request to 60 seconds and 1.5 MB, so claude-code-proxy needs a \
                      dedicated endpoint: deploy the model to a dedicated endpoint, or, if this \
                      endpoint is one, set \"host\" in vertex.endpoints.my-llm to its dedicated URL";
        let tests = [
            (
                "shared endpoint",
                json!({"dedicatedEndpointEnabled": false}),
                StatusCode::BAD_REQUEST,
                Some(shared),
                None,
            ),
            (
                "no dedicated flag",
                json!({"dedicatedEndpointDns": SYNTHETIC_DEDICATED}),
                StatusCode::BAD_REQUEST,
                Some(shared),
                None,
            ),
            (
                "flag that is not a boolean",
                json!({"dedicatedEndpointEnabled": "true", "dedicatedEndpointDns": SYNTHETIC_DEDICATED}),
                StatusCode::BAD_REQUEST,
                Some(shared),
                None,
            ),
            (
                "dedicated without a DNS name",
                json!({"dedicatedEndpointEnabled": true}),
                StatusCode::BAD_GATEWAY,
                None,
                Some("returned no dedicatedEndpointDns"),
            ),
            (
                "dedicated with an empty DNS name",
                json!({"dedicatedEndpointEnabled": true, "dedicatedEndpointDns": " "}),
                StatusCode::BAD_GATEWAY,
                None,
                Some("returned no dedicatedEndpointDns"),
            ),
            (
                "plain http",
                json!({"dedicatedEndpointEnabled": true,
                       "dedicatedEndpointDns": "http://1234567890123456789.us-central1-987654321.prediction.vertexai.goog"}),
                StatusCode::BAD_GATEWAY,
                None,
                Some("must use https (plain http only for a loopback host)"),
            ),
            (
                "another host",
                json!({"dedicatedEndpointEnabled": true,
                       "dedicatedEndpointDns": "https://us-central1-aiplatform.googleapis.com"}),
                StatusCode::BAD_GATEWAY,
                None,
                Some("must be a dedicated endpoint host"),
            ),
        ];
        for (case, metadata, status, exact, contains) in tests {
            let error = host_from_metadata(&metadata, &call(&config)).unwrap_err();
            assert_eq!(error.status, status, "{case}");
            if let Some(exact) = exact {
                assert_eq!(error.message, exact, "{case}");
                assert_eq!(error.kind, ProviderErrorKind::InvalidRequest, "{case}");
            }
            if let Some(contains) = contains {
                assert!(
                    error.message.contains(contains),
                    "{case}: {}",
                    error.message
                );
                assert!(
                    error.message.contains(
                        "never to the shared host, which a dedicated endpoint does not answer on"
                    ),
                    "{case}: {}",
                    error.message
                );
                assert_eq!(error.kind, ProviderErrorKind::Api, "{case}");
            }
        }
    }

    #[test]
    fn metadata_synthetic_dns_as_a_url_or_a_bare_name_gives_one_host_and_others_are_refused() {
        let config = synthetic_config(None, true);
        let unusable = |reason: &str| {
            format!(
                "Vertex AI endpoint my-llm is a dedicated endpoint, but endpoints.get returned \
                 {reason} (endpoint 1234567890123456789 in project synthetic-project, location \
                 us-central1 (vertex.endpoints.my-llm in config.json)). The proxy sends the \
                 access token only to https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog \
                 and never to the shared host, which a dedicated endpoint does not answer on; set \
                 \"host\" in vertex.endpoints.my-llm to the endpoint's dedicated URL"
            )
        };
        let that = |reason: &str| unusable(&format!("a dedicatedEndpointDns that {reason}"));
        let bare = "1234567890123456789.us-central1-987654321.prediction.vertexai.goog";
        let base = "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog/";
        let tests: [(&str, Option<String>, Result<&str, String>); 10] = [
            ("full URL", Some(SYNTHETIC_DEDICATED.to_string()), Ok(base)),
            ("bare DNS name", Some(bare.to_string()), Ok(base)),
            (
                "plain http",
                Some(format!("http://{bare}")),
                Err(that("must use https (plain http only for a loopback host)")),
            ),
            (
                "user info",
                Some(format!("https://user:synthetic-pass@{bare}")),
                Err(that("must not contain a user name or password")),
            ),
            (
                "a path",
                Some(format!(
                    "{SYNTHETIC_DEDICATED}/v1/projects/synthetic-project"
                )),
                Err(that(
                    "must be a scheme and a host only, without a path, query or fragment",
                )),
            ),
            (
                "a port",
                Some(format!("{SYNTHETIC_DEDICATED}:8443")),
                Err(that(
                    "must not name a port: a dedicated endpoint host takes the default https port",
                )),
            ),
            (
                "a foreign host",
                Some("https://synthetic.example".to_string()),
                Err(that(
                    "must be a dedicated endpoint host, \
                     https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a \
                     loopback host",
                )),
            ),
            (
                "a bare foreign host",
                Some("synthetic.example".to_string()),
                Err(that(
                    "must be a dedicated endpoint host, \
                     https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a \
                     loopback host",
                )),
            ),
            (
                "an empty string",
                Some(String::new()),
                Err(unusable("no dedicatedEndpointDns")),
            ),
            ("no field", None, Err(unusable("no dedicatedEndpointDns"))),
        ];
        for (case, dns, expected) in tests {
            let mut metadata = json!({"dedicatedEndpointEnabled": true});
            if let Some(dns) = dns {
                metadata["dedicatedEndpointDns"] = json!(dns);
            }
            match (host_from_metadata(&metadata, &call(&config)), expected) {
                (Ok(host), Ok(expected)) => {
                    assert_eq!(host.as_str(), expected, "{case}");
                    assert_eq!(host.as_str().matches("://").count(), 1, "{case}");
                    let (_, url) = chat_url(&host, &call(&config)).unwrap();
                    assert_eq!(
                        url.as_str(),
                        "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog/v1/projects/synthetic-project/locations/us-central1/endpoints/1234567890123456789/chat/completions",
                        "{case}"
                    );
                }
                (Err(error), Err(expected)) => {
                    assert_eq!(error.status, StatusCode::BAD_GATEWAY, "{case}");
                    assert_eq!(error.kind, ProviderErrorKind::Api, "{case}");
                    assert_eq!(error.message, expected, "{case}");
                    assert!(!error.message.contains("synthetic-pass"), "{case}");
                }
                (got, expected) => panic!("{case}: got {got:?}, expected {expected:?}"),
            }
        }
        // A value that already carries a scheme never gets a second one.
        let doubled = json!({
            "dedicatedEndpointEnabled": true,
            "dedicatedEndpointDns": format!("https://{SYNTHETIC_DEDICATED}"),
        });
        assert!(host_from_metadata(&doubled, &call(&config)).is_err());
    }

    fn google_error(code: u16, status: &str, message: &str) -> UpstreamErrorDetail {
        UpstreamErrorDetail::read(
            json!({"error": {"code": code, "message": message, "status": status}})
                .to_string()
                .as_bytes(),
            None,
            &[SYNTHETIC_TOKEN],
        )
    }

    fn target() -> VertexTarget {
        VertexTarget::new(
            "synthetic-project",
            "us-central1",
            Some(SYNTHETIC_DEDICATED),
        )
        .unwrap()
    }

    #[test]
    fn inference_errors_synthetic_a_cold_start_429_is_a_503_and_others_keep_their_class() {
        let config = synthetic_config(None, true);
        let whereabouts = "endpoint 1234567890123456789 in project synthetic-project, location \
                           us-central1 (vertex.endpoints.my-llm in config.json)";
        // The autoscaling guide's wording for a scale-to-zero endpoint
        // without a ready replica.
        let not_ready = google_error(
            429,
            "RESOURCE_EXHAUSTED",
            "Model is not yet ready for inference. Please wait and then try your request again",
        );
        let error = inference_error(
            StatusCode::TOO_MANY_REQUESTS,
            not_ready,
            &target(),
            &call(&config),
        );
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.kind, ProviderErrorKind::Api);
        assert_eq!(
            error.message,
            "Vertex AI endpoint my-llm is starting: RESOURCE_EXHAUSTED: Model is not yet ready for \
             inference. Please wait and then try your request again. An endpoint that scales to \
             zero answers this until a replica is ready; it is not a rate limit. Retry in a few \
             minutes"
        );
        // A plain-text body is read the same way.
        let plain = UpstreamErrorDetail::read(
            b"Model is not yet ready for inference. Please wait and then try your request again",
            Some("30"),
            &[],
        );
        let error = inference_error(
            StatusCode::TOO_MANY_REQUESTS,
            plain,
            &target(),
            &call(&config),
        );
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.retry_after.as_deref(), Some("30"));

        let tests = [
            (
                StatusCode::TOO_MANY_REQUESTS,
                google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded."),
                StatusCode::TOO_MANY_REQUESTS,
                ProviderErrorKind::RateLimit,
                format!(
                    "RESOURCE_EXHAUSTED: Quota exceeded.. Vertex AI {whereabouts} is rate limited or its serving container is overloaded; retry later"
                ),
            ),
            (
                StatusCode::FORBIDDEN,
                google_error(403, "PERMISSION_DENIED", "Permission denied."),
                StatusCode::FORBIDDEN,
                ProviderErrorKind::Permission,
                format!(
                    "PERMISSION_DENIED: Permission denied.. Check that the account has the permission aiplatform.endpoints.predict (included in roles/aiplatform.user) for Vertex AI {whereabouts}"
                ),
            ),
            (
                StatusCode::NOT_FOUND,
                google_error(404, "NOT_FOUND", "Endpoint not found."),
                StatusCode::NOT_FOUND,
                ProviderErrorKind::InvalidRequest,
                format!(
                    "Vertex AI {whereabouts} was not found at its dedicated host: NOT_FOUND: Endpoint not found.. Check endpoint, location and project in vertex.endpoints.my-llm, and, when \"host\" is set there, that it is this endpoint's dedicated URL"
                ),
            ),
            (
                StatusCode::BAD_REQUEST,
                UpstreamErrorDetail::read(
                    br#"{"object":"error","message":"\"auto\" tool choice requires --enable-auto-tool-choice","type":"BadRequestError"}"#,
                    None,
                    &[],
                ),
                StatusCode::BAD_REQUEST,
                ProviderErrorKind::InvalidRequest,
                format!(
                    "\"auto\" tool choice requires --enable-auto-tool-choice (Vertex AI {whereabouts})"
                ),
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                UpstreamErrorDetail::default(),
                StatusCode::BAD_GATEWAY,
                ProviderErrorKind::Api,
                format!("Vertex AI returned HTTP 503 (Vertex AI {whereabouts})"),
            ),
        ];
        for (upstream, detail, status, kind, message) in tests {
            let error = inference_error(upstream, detail, &target(), &call(&config));
            assert_eq!(error.status, status, "{upstream}");
            assert_eq!(error.kind, kind, "{upstream}");
            assert_eq!(error.upstream_status, Some(upstream), "{upstream}");
            assert_eq!(error.message, message, "{upstream}");
        }
    }

    #[test]
    fn lookup_errors_synthetic_403_offers_the_host_setting_and_404_names_the_endpoint() {
        let config = synthetic_config(None, true);
        let whereabouts = "endpoint 1234567890123456789 in project synthetic-project, location \
                           us-central1 (vertex.endpoints.my-llm in config.json)";
        let error = lookup_error(
            StatusCode::FORBIDDEN,
            google_error(
                403,
                "PERMISSION_DENIED",
                "Permission 'aiplatform.endpoints.get' denied.",
            ),
            &target(),
            &call(&config),
        );
        assert_eq!(error.status, StatusCode::FORBIDDEN);
        assert_eq!(
            error.message,
            format!(
                "PERMISSION_DENIED: Permission 'aiplatform.endpoints.get' denied.. Reading Vertex AI {whereabouts} with endpoints.get needs the permission aiplatform.endpoints.get (included in roles/aiplatform.user): grant it, or set \"host\" in vertex.endpoints.my-llm to the endpoint's dedicated URL so the proxy does not read the endpoint"
            )
        );
        let error = lookup_error(
            StatusCode::NOT_FOUND,
            google_error(404, "NOT_FOUND", "Endpoint not found."),
            &target(),
            &call(&config),
        );
        assert_eq!(error.status, StatusCode::NOT_FOUND);
        assert_eq!(
            error.message,
            format!(
                "Vertex AI {whereabouts} was not found: NOT_FOUND: Endpoint not found.. Check endpoint, location and project in vertex.endpoints.my-llm"
            )
        );
        let error = lookup_error(
            StatusCode::UNAUTHORIZED,
            google_error(
                401,
                "UNAUTHENTICATED",
                "Request had invalid authentication credentials.",
            ),
            &target(),
            &call(&config),
        );
        assert_eq!(error.status, StatusCode::UNAUTHORIZED);
        assert_eq!(error.kind, ProviderErrorKind::Authentication);
    }

    #[test]
    fn find_synthetic_names_the_configured_endpoints_and_the_broken_setting() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"vertex":{"endpoints":{
                "my-llm":{"endpoint":"1","location":"us-central1","model":"m"},
                "broken":{"endpoint":"2","location":"global","model":"m"}
            }}}"#,
        )
        .unwrap();
        let endpoints = crate::config::resolve_vertex_endpoints(dir.path());
        assert_eq!(find(&endpoints, "my-llm").unwrap().model, "m");
        let error = find(&endpoints, "other").unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(
            error.message,
            "Vertex AI endpoint \"other\" is not configured: add \"other\": {\"endpoint\": \
             \"<endpoint-id>\", \"location\": \"<region>\", \"model\": \"<model the container \
             serves>\"} to vertex.endpoints in config.json and restart the proxy; configured \
             endpoints: my-llm"
        );
        let error = find(&endpoints, "broken").unwrap_err();
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            error.message,
            "Invalid Vertex configuration: Invalid Vertex setting vertex.endpoints in config.json: \
             endpoint \"broken\": location must be the endpoint's region, such as us-central1; an \
             endpoint is never in global"
        );
        let error = find(&VertexEndpoints::default(), "my-llm").unwrap_err();
        assert!(
            error.message.ends_with("configured endpoints: none"),
            "{}",
            error.message
        );
    }

    #[test]
    fn dedicated_hosts_synthetic_keep_the_first_answer() {
        let hosts = DedicatedHosts::default();
        assert_eq!(hosts.get("my-llm"), None);
        let first = reqwest::Url::parse(SYNTHETIC_DEDICATED).unwrap();
        let second = reqwest::Url::parse("http://127.0.0.1:9").unwrap();
        assert_eq!(hosts.insert("my-llm", first.clone()), first);
        assert_eq!(hosts.insert("my-llm", second), first);
        assert_eq!(hosts.get("my-llm"), Some(first));
        assert_eq!(hosts.get("other"), None);
    }

    // Streams and buffered reads.

    const MODEL: &str = "vertex/endpoint/my-llm";

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
            "id": "synthetic", "object": "chat.completion.chunk", "model": "gemma-2-9b-it",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        }))
    }

    /// Shaped like the Chat Completions stream of TGI and vLLM:
    /// `chat.completion.chunk` deltas, a finishing chunk, a usage chunk
    /// without choices and `data: [DONE]`.
    fn text_wire() -> Vec<String> {
        vec![
            chunk(json!({"role": "assistant", "content": "Hel"}), None),
            chunk(json!({"content": "lo"}), None),
            chunk(json!({}), Some("stop")),
            frame(json!({
                "id": "synthetic", "object": "chat.completion.chunk", "model": "gemma-2-9b-it",
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

    #[tokio::test]
    async fn stream_synthetic_text_ends_at_done_with_endpoint_labels_on_errors() {
        let redact: Sanitizer = Arc::new(str::to_string);
        let body = stream_body_redacted(
            stream::iter(chunks(text_wire())),
            "msg_synthetic".to_string(),
            MODEL.to_string(),
            &context(),
            sanitizer(),
            redact.clone(),
        );
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(!text.contains("event: error"), "{text}");
        assert!(text.contains("\"text\":\"Hel\""), "{text}");
        assert!(text.contains("\"stop_reason\":\"end_turn\""), "{text}");
        assert!(text.contains("\"output_tokens\":2"), "{text}");
        assert!(
            text.ends_with("data: {\"type\":\"message_stop\"}\n\n"),
            "{text}"
        );

        let mut late = text_wire();
        late.push(chunk(json!({"content": "late"}), None));
        let body = stream_body_redacted(
            stream::iter(chunks(late)),
            "msg_synthetic".to_string(),
            MODEL.to_string(),
            &context(),
            sanitizer(),
            redact,
        );
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("event: error"), "{text}");
        assert!(text.contains("Vertex AI endpoint"), "{text}");
        assert!(!text.contains("OpenCode Go"), "{text}");
        assert!(!text.contains("message_stop"), "{text}");
    }

    #[tokio::test]
    async fn buffered_synthetic_completes_with_text_and_usage() {
        let message = read_response(
            stream::iter(chunks(text_wire())),
            "msg_synthetic",
            MODEL,
            &sanitizer(),
            None,
        )
        .await
        .unwrap_or_else(|failure| panic!("a finished stream is a message: {}", failure.kind()));
        assert_eq!(message["model"], MODEL);
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(
            message["content"][0],
            json!({"type": "text", "text": "Hello"})
        );
        assert_eq!(message["usage"]["input_tokens"], 7);
        assert_eq!(message["usage"]["output_tokens"], 2);
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_synthetic_an_upstream_error_ends_the_read_while_the_upstream_stays_open() {
        let overloaded = frame(json!({
            "error": {"type": "overloaded", "message": format!("overloaded, saw {SYNTHETIC_TOKEN}")}
        }));
        let keep_alive = ": keep-alive\n\n".to_string();
        for (case, tail_keeps_sending) in [("upstream open", false), ("keep-alives follow", true)] {
            let pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = Arc::clone(&pulled);
            let keep_alive = keep_alive.clone();
            // The tail never ends. When it sends, it pauses one second of
            // paused time before each comment, so a read that wrongly keeps
            // pulling it reaches the timeout instead of spinning.
            let tail: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> =
                if tail_keeps_sending {
                    Box::pin(stream::unfold(keep_alive, |part| async move {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        Some((Ok(Bytes::from(part.clone())), part))
                    }))
                } else {
                    Box::pin(stream::pending())
                };
            let parts = vec![text_wire()[0].clone(), overloaded.clone()];
            let count = parts.len();
            let upstream = stream::iter(chunks(parts)).chain(tail).inspect(move |_| {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            });
            let started = tokio::time::Instant::now();
            let failure = tokio::time::timeout(
                Duration::from_secs(60),
                read_response(upstream, "msg_synthetic", MODEL, &sanitizer(), None),
            )
            .await
            .unwrap_or_else(|_| panic!("{case}: the read did not end at the error"))
            .expect_err("an error event is not a message");
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
            let message = error.to_string();
            assert!(!message.contains(SYNTHETIC_TOKEN), "{case}: {message}");
            assert!(
                message
                    .starts_with("Vertex AI endpoint upstream error: overloaded, saw [redacted]"),
                "{case}: {message}"
            );
        }
    }

    #[tokio::test]
    async fn buffered_synthetic_a_broken_connection_is_a_transport_error() {
        let mut parts = chunks(text_wire()[..1].to_vec());
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
