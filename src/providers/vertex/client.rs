//! HTTP client for Vertex AI. It builds the request URL, attaches the Google
//! credentials, waits a bounded time for the response headers and turns a
//! non-success answer into an Anthropic-style error. A successful body is not
//! parsed here: the caller gets the byte stream.
//!
//! 429 and 5xx are not retried; Claude Code retries on its own. Only the wait
//! for response headers is bounded. Once they arrive there is no deadline
//! between chunks, because a model may stay silent for a long time while it
//! thinks.

use std::time::Duration;

use futures_util::StreamExt;
use http::StatusCode;
use serde::Serialize;

use super::auth::{LOGIN_COMMAND, VertexAuthError, VertexCredentials, access_token};
use super::sanitize::sanitize_vertex_text;
use crate::config::{ResolvedVertexConfig, vertex_host};
use crate::provider::{ProviderError, ProviderErrorKind};
use crate::traffic::TrafficCapture;

/// Path prefix of the Vertex AI REST API version the proxy speaks.
pub const API_VERSION: &str = "v1";

/// How long to wait for the response headers. The same default as the Codex
/// provider's; a request still queued upstream after this is answered 504.
pub const RESPONSE_HEADER_TIMEOUT: Duration = Duration::from_secs(300);

/// How long an error body may take to arrive. The status is already known,
/// so an upstream that keeps an error body open must not hold the request.
pub const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest error body read. A longer one is cut and not parsed; the status
/// alone then decides the error.
pub const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Largest retry delay passed on as `retry-after`: one day, in seconds.
const MAX_RETRY_AFTER_SECS: u64 = 86_400;

const USER_AGENT: &str = concat!("claude-code-proxy/", env!("CARGO_PKG_VERSION"));

/// The `@type` of the `google.rpc.RetryInfo` entry in an error's `details`.
const RETRY_INFO_TYPE: &str = "type.googleapis.com/google.rpc.RetryInfo";

/// Response headers that may be copied into a traffic capture.
const CAPTURED_RESPONSE_HEADERS: [&str; 3] = ["content-type", "content-length", "retry-after"];

/// A failed Vertex request, already sanitised. `status` is what the proxy
/// answers with; `upstream_status` is what Vertex AI answered, when it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexError {
    /// The Anthropic error type.
    pub kind: ProviderErrorKind,
    /// The status the proxy answers with.
    pub status: StatusCode,
    /// The status Vertex AI answered with, if it answered.
    pub upstream_status: Option<StatusCode>,
    /// The message without the upstream status.
    pub message: String,
    /// Whole seconds for `retry-after`, from 0 to one day.
    pub retry_after: Option<String>,
}

impl VertexError {
    fn new(
        kind: ProviderErrorKind,
        status: StatusCode,
        upstream_status: Option<StatusCode>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            status,
            upstream_status,
            message: message.into(),
            retry_after: None,
        }
    }

    /// A Vertex setting is missing or invalid: 500 `api_error` with the
    /// message, which names the setting.
    pub fn configuration(error: impl std::fmt::Display) -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::INTERNAL_SERVER_ERROR,
            None,
            format!("Invalid Vertex configuration: {error}"),
        )
    }

    /// The request cannot be sent as asked: 400 `invalid_request_error`.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(
            ProviderErrorKind::InvalidRequest,
            StatusCode::BAD_REQUEST,
            None,
            message,
        )
    }

    /// Vertex AI could not be reached, or the connection failed before a
    /// status arrived: 502 `api_error`.
    pub fn unreachable() -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::BAD_GATEWAY,
            None,
            "Vertex AI is unreachable",
        )
    }

    /// No response headers within `timeout`: 504 `api_error`.
    pub fn response_header_timeout(timeout: Duration) -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::GATEWAY_TIMEOUT,
            None,
            format!(
                "Vertex AI sent no response headers within {} seconds",
                timeout.as_secs()
            ),
        )
    }

    /// The 200 stream broke off or reported a failure: 502 `api_error`.
    /// `message` must already have passed through [`sanitize_vertex_text`].
    pub fn stream_failed(message: impl Into<String>) -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::BAD_GATEWAY,
            Some(StatusCode::OK),
            message,
        )
    }

    /// Maps a non-success answer. The class comes from the HTTP status alone;
    /// the body only adds its message and retry delay. 401, 403, 404 and 429
    /// say what to check, with the model, project and location of the
    /// request.
    pub fn from_upstream(
        upstream: StatusCode,
        detail: UpstreamErrorDetail,
        target: &VertexTarget,
        model: &str,
    ) -> Self {
        let text = detail
            .text()
            .unwrap_or_else(|| format!("Vertex AI returned HTTP {}", upstream.as_u16()));
        let project = &target.project;
        let location = &target.location;
        let (kind, status, message) = match upstream.as_u16() {
            401 => (
                ProviderErrorKind::Authentication,
                StatusCode::UNAUTHORIZED,
                format!(
                    "Vertex AI rejected the Google access token: {text}. Run `{LOGIN_COMMAND}`, or check the key file named by GOOGLE_APPLICATION_CREDENTIALS"
                ),
            ),
            403 => (
                ProviderErrorKind::Permission,
                StatusCode::FORBIDDEN,
                format!(
                    "{text}. Check that the account has the role roles/aiplatform.user on project {project}, that the Vertex AI API (aiplatform.googleapis.com) is enabled there, and, for user credentials, that a quota project is set: gcloud auth application-default set-quota-project {project}"
                ),
            ),
            404 => (
                ProviderErrorKind::InvalidRequest,
                StatusCode::NOT_FOUND,
                format!(
                    "Model {model} is not available in project {project} / location {location}: {text}. Check the model ID and that the model is offered in that location (CCP_VERTEX_LOCATION, GOOGLE_CLOUD_LOCATION or vertex.location)"
                ),
            ),
            429 => (
                ProviderErrorKind::RateLimit,
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "{text}. The Vertex AI quota or shared capacity for model {model} in location {location} is exhausted; retry later or use another location"
                ),
            ),
            400..=499 => (ProviderErrorKind::InvalidRequest, upstream, text),
            _ => (ProviderErrorKind::Api, StatusCode::BAD_GATEWAY, text),
        };
        let mut error = Self::new(kind, status, Some(upstream), message);
        error.retry_after = detail
            .retry_after_secs
            .filter(|_| upstream == StatusCode::TOO_MANY_REQUESTS)
            .map(|secs| secs.to_string());
        error
    }

    /// The message as clients see it, ending with `(upstream HTTP <status>)`
    /// whenever Vertex AI answered.
    pub fn message_with_status(&self) -> String {
        match self.upstream_status {
            Some(status) => format!("{} (upstream HTTP {})", self.message, status.as_u16()),
            None => self.message.clone(),
        }
    }

    /// The error in the form the server answers with.
    pub fn into_provider_error(self) -> ProviderError {
        let mut error = ProviderError::new(self.status, self.kind, self.message_with_status());
        error.retry_after = self.retry_after;
        error
    }
}

impl From<VertexAuthError> for VertexError {
    fn from(error: VertexAuthError) -> Self {
        Self::new(
            ProviderErrorKind::Authentication,
            StatusCode::UNAUTHORIZED,
            None,
            error.to_string(),
        )
    }
}

impl From<VertexError> for ProviderError {
    fn from(error: VertexError) -> Self {
        error.into_provider_error()
    }
}

/// The sanitised parts of a Vertex AI error body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamErrorDetail {
    /// `error.message`, sanitised.
    pub message: Option<String>,
    /// `error.status`, the canonical code name such as `PERMISSION_DENIED`.
    pub status: Option<String>,
    /// `retryDelay` of a `google.rpc.RetryInfo` detail, or else the
    /// `retry-after` header, rounded up to whole seconds and only when it is
    /// between 0 and one day.
    pub retry_after_secs: Option<u64>,
}

impl UpstreamErrorDetail {
    /// Reads `{"error": {"code", "message", "status", "details"}}` or the same
    /// object as the only element of an array. Anything else gives an empty
    /// detail, so the status alone decides the error.
    pub fn parse(body: &[u8], retry_after_header: Option<&str>, secrets: &[&str]) -> Self {
        let value = serde_json::from_slice::<serde_json::Value>(body).ok();
        let error = value.as_ref().and_then(|value| match value {
            serde_json::Value::Array(items) => items.first()?.get("error"),
            other => other.get("error"),
        });
        let text = |name: &str| {
            error
                .and_then(|error| error.get(name))
                .and_then(serde_json::Value::as_str)
        };
        let retry_delay = error
            .and_then(|error| error.get("details"))
            .and_then(serde_json::Value::as_array)
            .and_then(|details| {
                details.iter().find_map(|detail| {
                    (detail.get("@type")?.as_str()? == RETRY_INFO_TYPE)
                        .then(|| detail.get("retryDelay")?.as_str())
                        .flatten()
                })
            })
            .and_then(retry_delay_secs);
        Self {
            message: text("message")
                .filter(|message| !message.trim().is_empty())
                .map(|message| sanitize_vertex_text(message, secrets)),
            status: text("status")
                .filter(|status| is_status_name(status))
                .map(str::to_string),
            retry_after_secs: retry_delay
                .or_else(|| retry_after_header.and_then(retry_after_header_secs)),
        }
    }

    /// `STATUS: message`, the message alone, or `None` when the body had
    /// neither.
    fn text(&self) -> Option<String> {
        match (&self.status, &self.message) {
            (Some(status), Some(message)) => Some(format!("{status}: {message}")),
            (None, Some(message)) => Some(message.clone()),
            (Some(status), None) => Some(status.clone()),
            (None, None) => None,
        }
    }
}

/// A canonical code name: upper-case letters and `_`, at most 64 of them.
fn is_status_name(value: &str) -> bool {
    (1..=64).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'_')
}

/// Converts a protobuf JSON duration (`"2.540s"`, `"30s"`) to whole seconds,
/// rounding up. `None` for any other shape or a value above one day.
pub fn retry_delay_secs(value: &str) -> Option<u64> {
    let number = value.trim().strip_suffix('s')?;
    let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
    if whole.is_empty() || !whole.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if fraction.len() > 9 || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let whole: u64 = whole.parse().ok()?;
    let round_up = u64::from(fraction.bytes().any(|byte| byte != b'0'));
    let secs = whole.checked_add(round_up)?;
    (secs <= MAX_RETRY_AFTER_SECS).then_some(secs)
}

fn retry_after_header_secs(value: &str) -> Option<u64> {
    let secs: u64 = value.trim().parse().ok()?;
    (secs <= MAX_RETRY_AFTER_SECS).then_some(secs)
}

/// Where requests go: the validated project and location, and the URL that
/// replaces `https://{host}` (the `baseUrl` setting, or the host derived from
/// the location).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexTarget {
    /// Google Cloud project ID.
    pub project: String,
    /// Vertex AI location.
    pub location: String,
    /// Scheme, host and optional path prefix; `/v1/projects/...` is appended.
    pub base_url: reqwest::Url,
}

impl VertexTarget {
    /// Validates the resolved settings.
    ///
    /// # Errors
    ///
    /// A missing or invalid setting gives [`VertexError::configuration`].
    pub fn from_config(config: &ResolvedVertexConfig) -> Result<Self, VertexError> {
        let project = config.project().map_err(VertexError::configuration)?;
        let location = config.location().map_err(VertexError::configuration)?;
        let base_url = config.base_url().map_err(VertexError::configuration)?;
        Self::new(project, location, base_url)
    }

    /// A target from values that are already validated; `base_url` replaces
    /// the host derived from `location`. Its query and fragment are dropped.
    ///
    /// # Errors
    ///
    /// Fails when the base URL cannot be parsed or cannot take a path.
    pub fn new(project: &str, location: &str, base_url: Option<&str>) -> Result<Self, VertexError> {
        let raw = match base_url {
            Some(url) => url.to_string(),
            None => format!("https://{}", vertex_host(location)),
        };
        let mut base_url = reqwest::Url::parse(&raw)
            .ok()
            .filter(|url| !url.cannot_be_a_base())
            .ok_or_else(|| VertexError::configuration("the Vertex base URL is not a valid URL"))?;
        base_url.set_query(None);
        base_url.set_fragment(None);
        Ok(Self {
            project: project.to_string(),
            location: location.to_string(),
            base_url,
        })
    }

    /// `{base}/v1/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:{method}`.
    ///
    /// Every value goes in as one path segment: `/`, `?`, `#`, `%` and spaces
    /// are percent-encoded.
    ///
    /// # Errors
    ///
    /// An empty segment, `.` or `..` (which a URL path would collapse rather
    /// than encode), or a `:` in the model (which would change the method),
    /// gives [`VertexError::invalid_request`].
    pub fn publisher_model_url(
        &self,
        publisher: &str,
        model: &str,
        method: &str,
    ) -> Result<reqwest::Url, VertexError> {
        for (name, value) in [
            ("publisher", publisher),
            ("model", model),
            ("method", method),
        ] {
            check_segment(name, value)?;
        }
        if model.contains(':') {
            return Err(VertexError::invalid_request(format!(
                "Vertex model ID must not contain ':': {model}"
            )));
        }
        self.location_url([
            "publishers",
            publisher,
            "models",
            &format!("{model}:{method}"),
        ])
    }

    /// The Gemini streaming call:
    /// `.../publishers/google/models/{model}:streamGenerateContent?alt=sse`.
    ///
    /// # Errors
    ///
    /// As for [`Self::publisher_model_url`].
    pub fn gemini_stream_url(&self, model: &str) -> Result<reqwest::Url, VertexError> {
        let mut url = self.publisher_model_url("google", model, "streamGenerateContent")?;
        url.set_query(Some("alt=sse"));
        Ok(url)
    }

    /// `{base}/v1/projects/{project}/locations/{location}/` followed by
    /// `segments`, each percent-encoded as one segment.
    ///
    /// # Errors
    ///
    /// Fails for an empty, `.` or `..` segment, including the project and
    /// location.
    pub fn location_url<'a>(
        &self,
        segments: impl IntoIterator<Item = &'a str>,
    ) -> Result<reqwest::Url, VertexError> {
        check_segment("project", &self.project)?;
        check_segment("location", &self.location)?;
        let segments: Vec<&str> = segments.into_iter().collect();
        for segment in &segments {
            check_segment("path", segment)?;
        }
        let mut url = self.base_url.clone();
        {
            let mut path = url.path_segments_mut().map_err(|()| {
                VertexError::configuration("the Vertex base URL cannot take a path")
            })?;
            path.pop_if_empty();
            path.extend([
                API_VERSION,
                "projects",
                &self.project,
                "locations",
                &self.location,
            ]);
            path.extend(segments);
        }
        Ok(url)
    }
}

fn check_segment(name: &str, value: &str) -> Result<(), VertexError> {
    if value.is_empty() || value == "." || value == ".." {
        return Err(VertexError::invalid_request(format!(
            "Vertex {name} must not be empty, '.' or '..'"
        )));
    }
    Ok(())
}

/// A successful response: the caller reads the SSE body. The access token it
/// was sent with is kept for scrubbing upstream prose inside the stream.
pub struct VertexResponse {
    /// The 2xx response, body unread.
    pub response: reqwest::Response,
    /// The bearer token the request carried, if any.
    pub access_token: Option<String>,
}

// Hand-written so the access token cannot reach a log line through `{:?}`.
impl std::fmt::Debug for VertexResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexResponse")
            .field("status", &self.response.status())
            .finish_non_exhaustive()
    }
}

impl VertexResponse {
    /// Secrets to pass to [`sanitize_vertex_text`] for text from this
    /// response.
    pub fn secrets(&self) -> Vec<&str> {
        self.access_token.as_deref().into_iter().collect()
    }

    /// The body as it arrives. A broken connection ends the stream with
    /// [`VertexError::stream_failed`].
    pub fn into_byte_stream(
        self,
    ) -> impl futures_util::Stream<Item = Result<bytes::Bytes, VertexError>> + Send + 'static {
        self.response.bytes_stream().map(|chunk| {
            chunk.map_err(|_| VertexError::stream_failed("Vertex AI stream broke off"))
        })
    }
}

/// Connection settings: no redirects (the request carries a bearer token), a
/// 10-second connect timeout, and no total or read timeout.
pub fn client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
}

/// Sends Vertex AI requests. Build it on the first request: building loads
/// the system root certificates, which takes long enough to matter for every
/// CLI command and server start.
pub struct VertexClient {
    client: reqwest::Client,
    response_header_timeout: Duration,
    error_body_timeout: Duration,
}

impl VertexClient {
    /// A client with [`client_builder`] settings.
    ///
    /// # Errors
    ///
    /// Fails when the TLS configuration cannot be built.
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self::with_client(client_builder().build()?))
    }

    /// A client around `client`, for tests against a local upstream.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client,
            response_header_timeout: RESPONSE_HEADER_TIMEOUT,
            error_body_timeout: ERROR_BODY_TIMEOUT,
        }
    }

    /// Sends one Gemini `streamGenerateContent?alt=sse` request for `model`
    /// with the resolved settings.
    ///
    /// # Errors
    ///
    /// See [`Self::post_sse`]; an invalid setting gives
    /// [`VertexError::configuration`].
    pub async fn stream_generate_content<B: Serialize + ?Sized>(
        &self,
        config: &ResolvedVertexConfig,
        model: &str,
        body: &B,
        credentials: &VertexCredentials,
        traffic: Option<&TrafficCapture>,
    ) -> Result<VertexResponse, VertexError> {
        let target = VertexTarget::from_config(config)?;
        let url = target.gemini_stream_url(model)?;
        self.post_sse(&target, model, url, body, credentials, traffic)
            .await
    }

    /// POSTs `body` as JSON to `url` and asks for an event stream. `target`
    /// and `model` only shape error messages.
    ///
    /// # Errors
    ///
    /// No credentials: 401. No connection: 502. No response headers within
    /// [`RESPONSE_HEADER_TIMEOUT`]: 504. A non-success status: mapped by
    /// [`VertexError::from_upstream`].
    pub async fn post_sse<B: Serialize + ?Sized>(
        &self,
        target: &VertexTarget,
        model: &str,
        url: reqwest::Url,
        body: &B,
        credentials: &VertexCredentials,
        traffic: Option<&TrafficCapture>,
    ) -> Result<VertexResponse, VertexError> {
        let bytes = serde_json::to_vec(body).map_err(|_| {
            VertexError::new(
                ProviderErrorKind::Api,
                StatusCode::INTERNAL_SERVER_ERROR,
                None,
                "Vertex request could not be encoded",
            )
        })?;
        if let Some(capture) = traffic {
            let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            capture.write_json("020-upstream-request", &value);
        }
        let auth_headers = credentials.headers().await?;
        let token = access_token(&auth_headers).map(str::to_string);
        if let Some(capture) = traffic {
            capture.write_json(
                "021-upstream-request-metadata",
                &serde_json::json!({
                    "method": "POST",
                    "url": url.as_str(),
                    "provider": "vertex",
                    "transport": "http",
                    "headers": request_headers_for_capture(&auth_headers),
                    "body_bytes": bytes.len(),
                }),
            );
        }

        let request = self
            .client
            .post(url)
            .headers(auth_headers)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::ACCEPT, "text/event-stream")
            .header(http::header::USER_AGENT, USER_AGENT)
            .body(bytes);
        let response =
            match tokio::time::timeout(self.response_header_timeout, request.send()).await {
                Err(_) => {
                    return Err(VertexError::response_header_timeout(
                        self.response_header_timeout,
                    ));
                }
                Ok(Err(_)) => return Err(VertexError::unreachable()),
                Ok(Ok(response)) => response,
            };

        let status = response.status();
        if let Some(capture) = traffic {
            capture.write_json(
                "030-upstream-response-headers",
                &serde_json::json!({
                    "status": status.as_u16(),
                    "headers": safe_response_headers(response.headers()),
                }),
            );
        }
        if status.is_success() {
            return Ok(VertexResponse {
                response,
                access_token: token,
            });
        }
        let secrets: Vec<&str> = token.as_deref().into_iter().collect();
        let detail = self.error_detail(response, &secrets).await;
        Err(VertexError::from_upstream(status, detail, target, model))
    }

    /// Reads at most [`MAX_ERROR_BODY_BYTES`] for at most the error-body
    /// timeout. A body that is cut off or late is not parsed.
    async fn error_detail(
        &self,
        response: reqwest::Response,
        secrets: &[&str],
    ) -> UpstreamErrorDetail {
        let retry_after = response
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let complete = tokio::time::timeout(
            self.error_body_timeout,
            read_bounded(response, MAX_ERROR_BODY_BYTES),
        )
        .await
        .ok()
        .and_then(|(body, cut)| (!cut).then_some(body));
        UpstreamErrorDetail::parse(
            complete.as_deref().unwrap_or_default(),
            retry_after.as_deref(),
            secrets,
        )
    }
}

/// Reads at most `limit` bytes. The flag is true when the body reached the
/// limit or broke off; reaching the limit ends the read without waiting for
/// more.
async fn read_bounded(response: reqwest::Response, limit: usize) -> (Vec<u8>, bool) {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else {
            return (body, true);
        };
        let remaining = limit.saturating_sub(body.len());
        if chunk.len() >= remaining {
            body.extend_from_slice(&chunk[..remaining]);
            return (body, true);
        }
        body.extend_from_slice(&chunk);
    }
    (body, false)
}

/// The request headers as a capture shows them: every credential header is
/// `[redacted]` except the quota project, which is a project ID.
fn request_headers_for_capture(auth_headers: &http::HeaderMap) -> serde_json::Value {
    let mut headers = serde_json::Map::new();
    for (name, value) in auth_headers {
        let shown = if name.as_str() == "x-goog-user-project" {
            value.to_str().unwrap_or("[invalid]").to_string()
        } else {
            "[redacted]".to_string()
        };
        headers.insert(name.as_str().to_string(), shown.into());
    }
    headers.insert("accept".to_string(), "text/event-stream".into());
    headers.insert("content-type".to_string(), "application/json".into());
    headers.insert("user-agent".to_string(), USER_AGENT.into());
    serde_json::Value::Object(headers)
}

fn safe_response_headers(headers: &reqwest::header::HeaderMap) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for name in CAPTURED_RESPONSE_HEADERS {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            result.insert(name.to_string(), value.into());
        }
    }
    serde_json::Value::Object(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{VertexSetting, VertexSettingSource};
    use axum::{
        Router,
        body::{Body, Bytes},
        extract::{OriginalUri, State},
        http::HeaderMap,
        response::{IntoResponse, Response},
        routing::post,
    };
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    const SYNTHETIC_TOKEN: &str =
        "ya29.a0Synthetic0Vertex0Client0Token0Body0With0Mixed0Case0And0Digits0Qz7";
    const MODEL: &str = "gemini-3.5-flash";

    fn setting(value: &str, source: VertexSettingSource) -> VertexSetting {
        VertexSetting {
            value: value.to_string(),
            source,
        }
    }

    fn target(location: &str) -> VertexTarget {
        VertexTarget::new("synthetic-project", location, None).unwrap()
    }

    fn fixed_credentials() -> VertexCredentials {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            format!("Bearer {SYNTHETIC_TOKEN}").parse().unwrap(),
        );
        headers.insert("x-goog-user-project", "synthetic-quota".parse().unwrap());
        VertexCredentials::fixed(headers)
    }

    // URLs: the path shape is `v1` `projects.locations.publishers.models.
    // streamGenerateContent` from the Vertex AI discovery document; the hosts
    // are the ones Google documents for global, the us/eu multi-regions and a
    // region.

    #[test]
    fn gemini_stream_url_uses_the_documented_path_and_host_for_each_location_kind() {
        for (location, expected) in [
            (
                "global",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
            ),
            (
                "us",
                "https://aiplatform.us.rep.googleapis.com/v1/projects/synthetic-project/locations/us/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
            ),
            (
                "eu",
                "https://aiplatform.eu.rep.googleapis.com/v1/projects/synthetic-project/locations/eu/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
            ),
            (
                "asia-northeast1",
                "https://asia-northeast1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/asia-northeast1/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
            ),
        ] {
            assert_eq!(
                target(location).gemini_stream_url(MODEL).unwrap().as_str(),
                expected
            );
        }
    }

    #[test]
    fn gemini_stream_url_synthetic_base_url_replaces_the_host_and_keeps_its_prefix() {
        for base in [
            "http://127.0.0.1:8080/relay",
            "http://127.0.0.1:8080/relay/",
            "http://127.0.0.1:8080/relay?ignored=1#frag",
        ] {
            let target = VertexTarget::new("synthetic-project", "global", Some(base)).unwrap();
            assert_eq!(
                target.gemini_stream_url(MODEL).unwrap().as_str(),
                "http://127.0.0.1:8080/relay/v1/projects/synthetic-project/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
                "{base}"
            );
        }
        let bare =
            VertexTarget::new("synthetic-project", "us", Some("http://localhost:9")).unwrap();
        assert_eq!(
            bare.gemini_stream_url(MODEL).unwrap().as_str(),
            "http://localhost:9/v1/projects/synthetic-project/locations/us/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn publisher_model_url_synthetic_encodes_reserved_characters_in_one_segment() {
        let url = target("global")
            .publisher_model_url("google", "a/../b?c#d e%", "streamGenerateContent")
            .unwrap();
        assert_eq!(
            url.as_str(),
            "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/google/models/a%2F..%2Fb%3Fc%23d%20e%25:streamGenerateContent"
        );
        assert_eq!(url.query(), None);
    }

    #[test]
    fn publisher_model_url_synthetic_rejects_segments_a_path_would_collapse() {
        let target = target("global");
        for (publisher, model) in [
            ("google", ""),
            ("google", "."),
            ("google", ".."),
            ("..", MODEL),
            ("", MODEL),
        ] {
            let error = target
                .publisher_model_url(publisher, model, "streamGenerateContent")
                .unwrap_err();
            assert_eq!(
                error.status,
                StatusCode::BAD_REQUEST,
                "{publisher:?} {model:?}"
            );
            assert_eq!(error.kind, ProviderErrorKind::InvalidRequest);
        }
        let error = target
            .publisher_model_url("google", "gemini:countTokens", "streamGenerateContent")
            .unwrap_err();
        assert!(error.message.contains("':'"), "{}", error.message);
    }

    #[test]
    fn target_from_config_synthetic_reports_settings_as_500_configuration_errors() {
        let unset = ResolvedVertexConfig {
            project_setting: None,
            location_setting: setting("global", VertexSettingSource::Default),
            base_url_setting: None,
        };
        let error = VertexTarget::from_config(&unset).unwrap_err();
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.kind, ProviderErrorKind::Api);
        assert_eq!(
            error.message,
            "Invalid Vertex configuration: Vertex project is not set: set CCP_VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT or vertex.project in config.json"
        );
        let slash = ResolvedVertexConfig {
            project_setting: Some(setting(
                "synthetic/project",
                VertexSettingSource::Env("CCP_VERTEX_PROJECT"),
            )),
            ..unset
        };
        let error = VertexTarget::from_config(&slash).unwrap_err();
        assert!(
            error.message.contains("CCP_VERTEX_PROJECT"),
            "{}",
            error.message
        );
        assert!(
            !error.message.contains("synthetic/project"),
            "{}",
            error.message
        );
    }

    // Error bodies: Google APIs answer `{"error": {"code", "message",
    // "status", "details"}}` (google.rpc.Status); some front ends wrap it in a
    // one-element array. These bodies are synthetic: none has been observed
    // from Vertex AI yet.

    #[test]
    fn error_detail_synthetic_reads_the_object_and_the_array_form() {
        let object = br#"{"error":{"code":403,"message":"Permission 'aiplatform.endpoints.predict' denied on resource '//aiplatform.googleapis.com/projects/synthetic-project/locations/global/publishers/google/models/gemini-3.5-flash' (or it may not exist).","status":"PERMISSION_DENIED"}}"#;
        let array = [b"[".as_slice(), object.as_slice(), b"]".as_slice()].concat();
        for body in [object.as_slice(), array.as_slice()] {
            let detail = UpstreamErrorDetail::parse(body, None, &[]);
            assert_eq!(detail.status.as_deref(), Some("PERMISSION_DENIED"));
            assert_eq!(
                detail.message.as_deref(),
                Some(
                    "Permission 'aiplatform.endpoints.predict' denied on resource '//aiplatform.googleapis.com/projects/synthetic-project/locations/global/publishers/google/models/gemini-3.5-flash' (or it may not exist)."
                )
            );
        }
        for body in [
            b"".as_slice(),
            b"<html>busy</html>",
            b"[]",
            br#"{"message":"x"}"#,
        ] {
            assert_eq!(
                UpstreamErrorDetail::parse(body, None, &[]),
                UpstreamErrorDetail::default()
            );
        }
    }

    #[test]
    fn error_detail_synthetic_drops_a_status_that_is_not_a_code_name() {
        let body = br#"{"error":{"message":"m","status":"not a code; drop"}}"#;
        assert_eq!(UpstreamErrorDetail::parse(body, None, &[]).status, None);
    }

    #[test]
    fn retry_delay_rounds_protobuf_durations_up_within_one_day() {
        assert_eq!(retry_delay_secs("2.540s"), Some(3));
        assert_eq!(retry_delay_secs("30s"), Some(30));
        assert_eq!(retry_delay_secs("0s"), Some(0));
        assert_eq!(retry_delay_secs("0.000000001s"), Some(1));
        assert_eq!(retry_delay_secs("4.000s"), Some(4));
        assert_eq!(retry_delay_secs("86400s"), Some(86_400));
        for value in [
            "86400.5s",
            "86401s",
            "-1s",
            "1.5",
            "s",
            ".5s",
            "1e3s",
            "1.0000000001s",
            "abc",
        ] {
            assert_eq!(retry_delay_secs(value), None, "{value}");
        }
    }

    #[test]
    fn error_detail_synthetic_takes_retry_info_before_the_header() {
        let body = br#"{"error":{"code":429,"message":"Resource exhausted. Please try again later.","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"RATE_LIMIT_EXCEEDED"},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"2.540s"}]}}"#;
        assert_eq!(
            UpstreamErrorDetail::parse(body, Some("60"), &[]).retry_after_secs,
            Some(3)
        );
        let without = br#"{"error":{"code":429,"message":"m","status":"RESOURCE_EXHAUSTED"}}"#;
        assert_eq!(
            UpstreamErrorDetail::parse(without, Some("60"), &[]).retry_after_secs,
            Some(60)
        );
        assert_eq!(
            UpstreamErrorDetail::parse(without, Some("Wed, 21 Oct 2026 07:28:00 GMT"), &[])
                .retry_after_secs,
            None
        );
        assert_eq!(
            UpstreamErrorDetail::parse(without, Some("86401"), &[]).retry_after_secs,
            None
        );
    }

    fn detail(message: &str, status: &str) -> UpstreamErrorDetail {
        UpstreamErrorDetail {
            message: Some(message.to_string()),
            status: Some(status.to_string()),
            retry_after_secs: Some(7),
        }
    }

    #[test]
    fn upstream_statuses_synthetic_map_to_anthropic_error_types() {
        let target = target("global");
        for (upstream, status, error_type) in [
            (401, 401, "authentication_error"),
            (403, 403, "permission_error"),
            (404, 404, "invalid_request_error"),
            (429, 429, "rate_limit_error"),
            (400, 400, "invalid_request_error"),
            (409, 409, "invalid_request_error"),
            (500, 502, "api_error"),
            (503, 502, "api_error"),
            (504, 502, "api_error"),
            (302, 502, "api_error"),
        ] {
            let upstream = StatusCode::from_u16(upstream).unwrap();
            let mapped = VertexError::from_upstream(
                upstream,
                detail("synthetic", "UNKNOWN"),
                &target,
                MODEL,
            )
            .into_provider_error();
            assert_eq!(mapped.status.as_u16(), status, "{upstream}");
            assert_eq!(mapped.error_type(), error_type, "{upstream}");
            assert!(
                mapped
                    .message
                    .ends_with(&format!("(upstream HTTP {})", upstream.as_u16())),
                "{}",
                mapped.message
            );
            // Only a 429 passes a retry delay on.
            assert_eq!(
                mapped.retry_after.is_some(),
                upstream == StatusCode::TOO_MANY_REQUESTS
            );
        }
    }

    #[test]
    fn upstream_401_403_404_429_synthetic_messages_name_what_to_check() {
        let target = target("asia-northeast1");
        let message = |status: u16| {
            VertexError::from_upstream(
                StatusCode::from_u16(status).unwrap(),
                detail("synthetic upstream text", "SYNTHETIC"),
                &target,
                MODEL,
            )
            .message_with_status()
        };
        let unauthorized = message(401);
        assert!(
            unauthorized.contains("gcloud auth application-default login"),
            "{unauthorized}"
        );
        assert!(
            unauthorized.contains("SYNTHETIC: synthetic upstream text"),
            "{unauthorized}"
        );

        let forbidden = message(403);
        for needle in [
            "roles/aiplatform.user",
            "project synthetic-project",
            "aiplatform.googleapis.com",
            "gcloud auth application-default set-quota-project synthetic-project",
        ] {
            assert!(forbidden.contains(needle), "{needle}: {forbidden}");
        }

        let not_found = message(404);
        assert!(
            not_found.starts_with("Model gemini-3.5-flash is not available in project synthetic-project / location asia-northeast1: "),
            "{not_found}"
        );
        assert!(not_found.contains("CCP_VERTEX_LOCATION"), "{not_found}");

        let limited = message(429);
        assert!(limited.contains("location asia-northeast1"), "{limited}");
        assert!(limited.ends_with("(upstream HTTP 429)"), "{limited}");
    }

    #[test]
    fn upstream_error_synthetic_without_a_body_keeps_the_status() {
        let mapped = VertexError::from_upstream(
            StatusCode::SERVICE_UNAVAILABLE,
            UpstreamErrorDetail::default(),
            &target("global"),
            MODEL,
        )
        .into_provider_error();
        assert_eq!(mapped.status, StatusCode::BAD_GATEWAY);
        assert_eq!(
            mapped.message,
            "Vertex AI returned HTTP 503 (upstream HTTP 503)"
        );
    }

    #[test]
    fn auth_and_configuration_failures_synthetic_map_to_401_and_500() {
        let auth: ProviderError =
            VertexError::from(VertexAuthError::Token("synthetic".to_string())).into();
        assert_eq!(auth.status, StatusCode::UNAUTHORIZED);
        assert_eq!(auth.error_type(), "authentication_error");
        assert!(
            auth.message
                .contains("gcloud auth application-default login"),
            "{}",
            auth.message
        );
        let config: ProviderError = VertexError::configuration("synthetic").into();
        assert_eq!(config.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(config.error_type(), "api_error");
    }

    // Against a local upstream.

    #[derive(Debug, Default)]
    struct Seen {
        path_and_query: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    type Shared = Arc<Mutex<Option<Seen>>>;

    async fn spawn_upstream(respond: fn() -> Response) -> (String, Shared) {
        async fn handler(
            State((seen, respond)): State<(Shared, fn() -> Response)>,
            OriginalUri(uri): OriginalUri,
            headers: HeaderMap,
            body: Bytes,
        ) -> Response {
            *seen.lock().unwrap() = Some(Seen {
                path_and_query: uri
                    .path_and_query()
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                headers: headers
                    .iter()
                    .map(|(name, value)| {
                        (
                            name.to_string(),
                            value.to_str().unwrap_or_default().to_string(),
                        )
                    })
                    .collect(),
                body: body.to_vec(),
            });
            respond()
        }
        let seen: Shared = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/{*path}", post(handler))
            .with_state((seen.clone(), respond));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}"), seen)
    }

    fn local_config(base_url: &str) -> ResolvedVertexConfig {
        ResolvedVertexConfig {
            project_setting: Some(setting(
                "synthetic-project",
                VertexSettingSource::Env("CCP_VERTEX_PROJECT"),
            )),
            location_setting: setting("global", VertexSettingSource::Default),
            base_url_setting: Some(setting(
                base_url,
                VertexSettingSource::Env("CCP_VERTEX_BASE_URL"),
            )),
        }
    }

    fn local_client() -> VertexClient {
        VertexClient::with_client(client_builder().no_proxy().build().unwrap())
    }

    async fn send(client: &VertexClient, base_url: &str) -> Result<VertexResponse, VertexError> {
        client
            .stream_generate_content(
                &local_config(base_url),
                MODEL,
                &serde_json::json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}),
                &fixed_credentials(),
                None,
            )
            .await
    }

    const SYNTHETIC_SSE: &str = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hello\"}]},\"finishReason\":\"STOP\"}]}\r\n\r\n";

    #[tokio::test]
    async fn request_synthetic_carries_credentials_and_fixed_headers_and_streams_the_body() {
        let (base, seen) = spawn_upstream(|| {
            (
                [(http::header::CONTENT_TYPE, "text/event-stream")],
                SYNTHETIC_SSE,
            )
                .into_response()
        })
        .await;
        let response = send(&local_client(), &base).await.unwrap();
        assert_eq!(response.access_token.as_deref(), Some(SYNTHETIC_TOKEN));
        assert_eq!(response.secrets(), vec![SYNTHETIC_TOKEN]);
        let mut stream = std::pin::pin!(response.into_byte_stream());
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(body, SYNTHETIC_SSE.as_bytes());

        let seen = seen.lock().unwrap().take().unwrap();
        assert_eq!(
            seen.path_and_query,
            "/v1/projects/synthetic-project/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
        );
        let header = |name: &str| {
            seen.headers
                .iter()
                .filter(|(seen_name, _)| seen_name == name)
                .map(|(_, value)| value.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            header("authorization"),
            vec![format!("Bearer {SYNTHETIC_TOKEN}").as_str()]
        );
        assert_eq!(header("x-goog-user-project"), vec!["synthetic-quota"]);
        assert_eq!(header("content-type"), vec!["application/json"]);
        assert_eq!(header("accept"), vec!["text/event-stream"]);
        assert_eq!(header("user-agent"), vec![USER_AGENT]);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&seen.body).unwrap(),
            serde_json::json!({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
        );
    }

    #[tokio::test]
    async fn request_synthetic_error_body_echoing_secrets_comes_out_redacted() {
        fn echo() -> Response {
            let image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk".repeat(4);
            let signature = format!("CiQBjz1rX{}", "Ab3dEf9GhIjK0lMn".repeat(8));
            let body = serde_json::json!({"error": {
                "code": 400,
                "message": format!(
                    "Invalid request for token {SYNTHETIC_TOKEN}: part inline_data '{image}' and thought_signature 'ccp:vertex:v1:gemini-3.5-flash:{signature}' are invalid"
                ),
                "status": "INVALID_ARGUMENT"
            }});
            (StatusCode::BAD_REQUEST, body.to_string()).into_response()
        }
        let (base, _) = spawn_upstream(echo).await;
        let error = send(&local_client(), &base).await.unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        let message = error.message_with_status();
        assert!(!message.contains(SYNTHETIC_TOKEN), "{message}");
        assert!(!message.contains("iVBORw0KGgo"), "{message}");
        assert!(!message.contains("Ab3dEf9GhIjK0lMn"), "{message}");
        assert_eq!(
            message,
            "INVALID_ARGUMENT: Invalid request for token [redacted]: part inline_data '[redacted len=240]' and thought_signature 'ccp:vertex:v1:gemini-3.5-flash:[redacted len=137]' are invalid (upstream HTTP 400)"
        );
    }

    #[tokio::test]
    async fn request_synthetic_without_response_headers_in_time_is_a_504() {
        // A streamed body would still send headers, so the handler itself
        // stalls before answering.
        async fn stall() -> Response {
            tokio::time::sleep(Duration::from_secs(30)).await;
            StatusCode::OK.into_response()
        }
        let app = Router::new().route("/{*path}", post(stall));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut client = local_client();
        client.response_header_timeout = Duration::from_millis(200);
        let error = send(&client, &format!("http://{address}"))
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(error.kind, ProviderErrorKind::Api);
        assert_eq!(error.upstream_status, None);
    }

    #[tokio::test]
    async fn request_synthetic_oversized_or_stalled_error_bodies_keep_the_status() {
        fn oversized() -> Response {
            let message = "x".repeat(MAX_ERROR_BODY_BYTES);
            let body = serde_json::json!({"error": {"code": 403, "message": message, "status": "PERMISSION_DENIED"}});
            (StatusCode::FORBIDDEN, body.to_string()).into_response()
        }
        fn stalled() -> Response {
            let start = futures_util::stream::once(async {
                Ok::<_, Infallible>(Bytes::from_static(br#"{"error":{"code":403,"#))
            });
            let rest = futures_util::stream::pending::<Result<Bytes, Infallible>>();
            (StatusCode::FORBIDDEN, Body::from_stream(start.chain(rest))).into_response()
        }
        for respond in [oversized as fn() -> Response, stalled] {
            let (base, _) = spawn_upstream(respond).await;
            let mut client = local_client();
            client.error_body_timeout = Duration::from_millis(200);
            let error = send(&client, &base).await.unwrap_err();
            assert_eq!(error.status, StatusCode::FORBIDDEN);
            assert_eq!(error.kind, ProviderErrorKind::Permission);
            assert!(
                error
                    .message
                    .starts_with("Vertex AI returned HTTP 403. Check that"),
                "{}",
                error.message
            );
        }
    }

    #[tokio::test]
    async fn request_synthetic_to_a_closed_port_is_a_502() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let error = send(&local_client(), &format!("http://{address}"))
            .await
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert_eq!(error.kind, ProviderErrorKind::Api);
        assert_eq!(error.message_with_status(), "Vertex AI is unreachable");
    }
}
