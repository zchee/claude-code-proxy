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
use super::models::{ApiVersion, ResolvedModel};
use super::sanitize::sanitize_vertex_text;
use crate::config::{ResolvedVertexConfig, vertex_host};
use crate::provider::{ProviderError, ProviderErrorKind};
use crate::traffic::TrafficCapture;

/// Path prefix of the Vertex AI REST API version the proxy speaks unless a
/// model names another (see [`ApiVersion`]).
pub const API_VERSION: &str = ApiVersion::V1.as_str();

/// How long to wait for the response headers. The same default as the Codex
/// provider's; a request still queued upstream after this is answered 504.
pub const RESPONSE_HEADER_TIMEOUT: Duration = Duration::from_secs(300);

/// How long an error body may take to arrive. The status is already known,
/// so an upstream that keeps an error body open must not hold the request.
pub const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest error body read. A longer one is cut and not parsed; the status
/// alone then decides the error.
pub const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Largest body read from a metadata call such as `endpoints.get`. An
/// endpoint's metadata is a few kilobytes; a longer body is refused.
pub const MAX_METADATA_BODY_BYTES: usize = 1024 * 1024;

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

    /// The error of a request for a partner model (every family but
    /// Gemini). A 403 or 404 from Vertex AI also names the model, project
    /// and location and says to check that the model is enabled in Model
    /// Garden for the project and, for a preview model, that the project
    /// has been granted access: a partner model can be refused for that
    /// reason as well as for the ones the message already gives, and the
    /// status does not say which applies. Every other error is returned
    /// unchanged.
    pub fn with_partner_guidance(mut self, target: &VertexTarget, model: &str) -> Self {
        if matches!(
            self.upstream_status.map(|status| status.as_u16()),
            Some(403 | 404)
        ) {
            self.message.push_str(&format!(
                ". Also check that model {model} is enabled in Model Garden for project {} in location {} and, for a preview model, that the project has been granted access; the status alone does not say which of these is the cause",
                target.project, target.location
            ));
        }
        self
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

    /// Reads a complete error body. [`Self::parse`] finds the message at
    /// `/error/message`, which is where Google
    /// (`{"error": {"code", "message", "status"}}`), Anthropic
    /// (`{"type": "error", "error": {"type", "message"}}`) and OpenAI
    /// (`{"error": {"message", "type", "code"}}`) put it. Without one, a
    /// string `message` at the top level of a JSON object is the message:
    /// Mistral's own error body is `{"object": "error", "message", ...}`.
    /// When the body gave neither a message nor a status, the body itself
    /// becomes the message,
    /// passed through [`sanitize_vertex_text`], which also cuts it to 500
    /// characters. Only a complete body may be used that way: in one cut
    /// short, a secret may have lost the end the sanitiser recognises it by.
    pub fn read(body: &[u8], retry_after_header: Option<&str>, secrets: &[&str]) -> Self {
        let mut detail = Self::parse(body, retry_after_header, secrets);
        if detail.message.is_none() {
            detail.message = serde_json::from_slice::<serde_json::Value>(body)
                .ok()
                .as_ref()
                .and_then(|value| value.get("message"))
                .and_then(serde_json::Value::as_str)
                .filter(|message| !message.trim().is_empty())
                .map(|message| sanitize_vertex_text(message, secrets));
        }
        if detail.message.is_none() && detail.status.is_none() {
            let text = sanitize_vertex_text(&String::from_utf8_lossy(body), secrets);
            let text = text.trim();
            if !text.is_empty() {
                detail.message = Some(text.to_string());
            }
        }
        detail
    }

    /// `STATUS: message`, the message alone, or `None` when the body had
    /// neither.
    pub(super) fn text(&self) -> Option<String> {
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

    /// The target for one model's requests: the location comes from
    /// [`ResolvedModel::location`] with the configured overrides and
    /// location.
    ///
    /// # Errors
    ///
    /// A missing or invalid setting, including an invalid override, gives
    /// [`VertexError::configuration`]. A model served only in some
    /// locations, with neither an override nor a configured location among
    /// them, gives [`VertexError::invalid_request`] with a message naming
    /// the settings and the locations to choose from.
    pub fn for_model(
        config: &ResolvedVertexConfig,
        model: &ResolvedModel,
    ) -> Result<Self, VertexError> {
        let project = config.project().map_err(VertexError::configuration)?;
        let configured = config.location().map_err(VertexError::configuration)?;
        let overrides = config
            .location_overrides()
            .map_err(VertexError::configuration)?;
        let location = model
            .location(&overrides, configured)
            .map_err(|error| VertexError::invalid_request(error.to_string()))?;
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
        self.versioned_publisher_model_url(ApiVersion::V1, publisher, model, method)
    }

    fn versioned_publisher_model_url(
        &self,
        version: ApiVersion,
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
        self.versioned_location_url(
            version,
            [
                "publishers",
                publisher,
                "models",
                &format!("{model}:{method}"),
            ],
        )
    }

    /// A publisher model's method in the model's API version:
    /// `{base}/{version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{upstream_model}:{method}`.
    /// Claude and Mistral use `rawPredict` and `streamRawPredict`.
    ///
    /// # Errors
    ///
    /// As for [`Self::publisher_model_url`].
    pub fn model_method_url(
        &self,
        model: &ResolvedModel,
        method: &str,
    ) -> Result<reqwest::Url, VertexError> {
        self.versioned_publisher_model_url(
            model.api_version,
            &model.publisher,
            &model.upstream_model,
            method,
        )
    }

    /// An OpenAI-compatible API under the shared `openapi` endpoint, in the
    /// model's API version:
    /// `{base}/{version}/projects/{project}/locations/{location}/endpoints/openapi/{api}`.
    /// The model goes in the request body, not the URL.
    ///
    /// # Errors
    ///
    /// Fails for an empty, `.` or `..` project or location.
    pub fn openapi_url(
        &self,
        model: &ResolvedModel,
        api: OpenApiPath,
    ) -> Result<reqwest::Url, VertexError> {
        let segments = ["endpoints", "openapi"]
            .into_iter()
            .chain(api.segments().iter().copied());
        self.versioned_location_url(model.api_version, segments)
    }

    /// The Responses API under a publisher, in the model's API version:
    /// `{base}/{version}/projects/{project}/locations/{location}/publishers/{publisher}/v1/responses`.
    /// The `v1` after the publisher is part of that API's path and does not
    /// follow the model's API version. The model goes in the request body.
    ///
    /// # Errors
    ///
    /// Fails for an empty, `.` or `..` project, location or publisher.
    pub fn publisher_responses_url(
        &self,
        model: &ResolvedModel,
    ) -> Result<reqwest::Url, VertexError> {
        check_segment("publisher", &model.publisher)?;
        self.versioned_location_url(
            model.api_version,
            ["publishers", &model.publisher, "v1", "responses"],
        )
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
        self.versioned_location_url(ApiVersion::V1, segments)
    }

    /// [`Self::location_url`] in API version `version`.
    ///
    /// # Errors
    ///
    /// As for [`Self::location_url`].
    pub fn versioned_location_url<'a>(
        &self,
        version: ApiVersion,
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
                version.as_str(),
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

/// An API under `endpoints/openapi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenApiPath {
    /// `responses`.
    Responses,
    /// `chat/completions`.
    ChatCompletions,
}

impl OpenApiPath {
    /// The path segments after `endpoints/openapi`.
    pub const fn segments(self) -> &'static [&'static str] {
        match self {
            OpenApiPath::Responses => &["responses"],
            OpenApiPath::ChatCompletions => &["chat", "completions"],
        }
    }
}

/// The `accept` value of a streaming request.
pub const EVENT_STREAM: &str = "text/event-stream";

/// The parts of a request that differ between model families.
#[derive(Debug, Clone)]
pub struct UpstreamCall {
    /// Where the request goes.
    pub url: reqwest::Url,
    /// The `accept` header: [`EVENT_STREAM`] for a stream,
    /// `application/json` for one JSON answer.
    pub accept: &'static str,
    /// Headers sent besides the credential headers, `content-type`,
    /// `accept` and `user-agent`, and never one of those names. A capture
    /// records them as sent, so they must not carry a secret.
    pub headers: http::HeaderMap,
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
    /// with the resolved settings, in the location
    /// [`VertexTarget::for_model`] picks for it.
    ///
    /// # Errors
    ///
    /// See [`Self::post_sse`] and [`VertexTarget::for_model`].
    pub async fn stream_generate_content<B: Serialize + ?Sized>(
        &self,
        config: &ResolvedVertexConfig,
        model: &ResolvedModel,
        body: &B,
        credentials: &VertexCredentials,
        traffic: Option<&TrafficCapture>,
    ) -> Result<VertexResponse, VertexError> {
        let target = VertexTarget::for_model(config, model)?;
        let url = target.gemini_stream_url(&model.upstream_model)?;
        self.post_sse(
            &target,
            &model.upstream_model,
            url,
            body,
            credentials,
            traffic,
        )
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
        let call = UpstreamCall {
            url,
            accept: EVENT_STREAM,
            headers: http::HeaderMap::new(),
        };
        self.post(target, model, call, body, credentials, traffic)
            .await
    }

    /// POSTs `body` as JSON as `call` describes. `target` and `model` only
    /// shape error messages.
    ///
    /// # Errors
    ///
    /// As for [`Self::post_sse`].
    pub async fn post<B: Serialize + ?Sized>(
        &self,
        target: &VertexTarget,
        model: &str,
        call: UpstreamCall,
        body: &B,
        credentials: &VertexCredentials,
        traffic: Option<&TrafficCapture>,
    ) -> Result<VertexResponse, VertexError> {
        self.post_mapped(call, body, credentials, traffic, |status, detail| {
            VertexError::from_upstream(status, detail, target, model)
        })
        .await
    }

    /// [`Self::post`], with a non-success answer turned into an error by
    /// `map_error` from the upstream status and the sanitised error body.
    ///
    /// # Errors
    ///
    /// As for [`Self::post_sse`], except that `map_error` decides the error
    /// for a non-success status.
    pub async fn post_mapped<B: Serialize + ?Sized>(
        &self,
        call: UpstreamCall,
        body: &B,
        credentials: &VertexCredentials,
        traffic: Option<&TrafficCapture>,
        map_error: impl FnOnce(StatusCode, UpstreamErrorDetail) -> VertexError,
    ) -> Result<VertexResponse, VertexError> {
        let UpstreamCall {
            url,
            accept,
            headers: extra_headers,
        } = call;
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
                    "headers": request_headers_for_capture(&auth_headers, accept, &extra_headers),
                    "body_bytes": bytes.len(),
                }),
            );
        }

        let request = self
            .client
            .post(url)
            .headers(extra_headers)
            .headers(auth_headers)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::ACCEPT, accept)
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
        Err(map_error(status, detail))
    }

    /// GETs `url` with the credentials and reads the JSON body: at most
    /// [`MAX_METADATA_BODY_BYTES`], within the error-body timeout once the
    /// headers have arrived. A capture records the request metadata, with
    /// every credential header redacted, and the response status and safe
    /// headers, under names that start with `capture_stem`; the body is not
    /// recorded.
    ///
    /// # Errors
    ///
    /// No credentials: 401. No connection: 502. No response headers within
    /// [`RESPONSE_HEADER_TIMEOUT`]: 504. A non-success status: `map_error`.
    /// A body that is too long, late, cut off or not JSON: 502.
    pub async fn get_json(
        &self,
        url: reqwest::Url,
        credentials: &VertexCredentials,
        traffic: Option<&TrafficCapture>,
        capture_stem: &str,
        map_error: impl FnOnce(StatusCode, UpstreamErrorDetail) -> VertexError,
    ) -> Result<serde_json::Value, VertexError> {
        let accept = "application/json";
        let auth_headers = credentials.headers().await?;
        let token = access_token(&auth_headers).map(str::to_string);
        if let Some(capture) = traffic {
            capture.write_json(
                &format!("{capture_stem}-request-metadata"),
                &serde_json::json!({
                    "method": "GET",
                    "url": url.as_str(),
                    "provider": "vertex",
                    "transport": "http",
                    "headers": request_headers_for_capture(&auth_headers, accept, &http::HeaderMap::new()),
                }),
            );
        }
        let request = self
            .client
            .get(url)
            .headers(auth_headers)
            .header(http::header::ACCEPT, accept)
            .header(http::header::USER_AGENT, USER_AGENT);
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
                &format!("{capture_stem}-response-headers"),
                &serde_json::json!({
                    "status": status.as_u16(),
                    "headers": safe_response_headers(response.headers()),
                }),
            );
        }
        if !status.is_success() {
            let secrets: Vec<&str> = token.as_deref().into_iter().collect();
            let detail = self.error_detail(response, &secrets).await;
            return Err(map_error(status, detail));
        }
        let unreadable = || {
            VertexError::new(
                ProviderErrorKind::Api,
                StatusCode::BAD_GATEWAY,
                Some(status),
                "Vertex AI returned a metadata body that could not be read as JSON",
            )
        };
        let (body, cut) = tokio::time::timeout(
            self.error_body_timeout,
            read_bounded(response, MAX_METADATA_BODY_BYTES),
        )
        .await
        .map_err(|_| unreadable())?;
        if cut {
            return Err(unreadable());
        }
        serde_json::from_slice(&body).map_err(|_| unreadable())
    }

    /// Reads at most [`MAX_ERROR_BODY_BYTES`] for at most the error-body
    /// timeout and passes it to [`UpstreamErrorDetail::read`]. A body that is
    /// cut off or late is not used.
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
        UpstreamErrorDetail::read(
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
/// `[redacted]` except the quota project, which is a project ID. The extra
/// headers of an [`UpstreamCall`] carry no secret and are shown as sent.
fn request_headers_for_capture(
    auth_headers: &http::HeaderMap,
    accept: &str,
    extra_headers: &http::HeaderMap,
) -> serde_json::Value {
    let mut headers = serde_json::Map::new();
    for (name, value) in extra_headers {
        headers.insert(
            name.as_str().to_string(),
            value.to_str().unwrap_or("[invalid]").into(),
        );
    }
    for (name, value) in auth_headers {
        let shown = if name.as_str() == "x-goog-user-project" {
            value.to_str().unwrap_or("[invalid]").to_string()
        } else {
            "[redacted]".to_string()
        };
        headers.insert(name.as_str().to_string(), shown.into());
    }
    headers.insert("accept".to_string(), accept.into());
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
            location_overrides_setting: None,
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

    // Per-model URLs: the paths are the ones Google documents for each
    // family (partner-model guides and the `v1` discovery document), with
    // the project replaced by `synthetic-project`. The hosts follow the same
    // rule as Gemini's for global, the us/eu multi-regions and a region.

    fn model(raw: &str) -> ResolvedModel {
        crate::providers::vertex::models::resolve(raw).unwrap()
    }

    #[test]
    fn model_method_url_uses_the_documented_raw_predict_path_for_each_location_kind() {
        for (location, raw, method, expected) in [
            (
                "global",
                "vertex/anthropic/claude-opus-5-5",
                "rawPredict",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/anthropic/models/claude-opus-5-5:rawPredict",
            ),
            (
                "us",
                "vertex/anthropic/claude-opus-5-5",
                "streamRawPredict",
                "https://aiplatform.us.rep.googleapis.com/v1/projects/synthetic-project/locations/us/publishers/anthropic/models/claude-opus-5-5:streamRawPredict",
            ),
            (
                "eu",
                "vertex/anthropic/claude-sonnet-5",
                "streamRawPredict",
                "https://aiplatform.eu.rep.googleapis.com/v1/projects/synthetic-project/locations/eu/publishers/anthropic/models/claude-sonnet-5:streamRawPredict",
            ),
            (
                "us-east5",
                "vertex/anthropic/claude-opus-4-5@20251101",
                "rawPredict",
                "https://us-east5-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-east5/publishers/anthropic/models/claude-opus-4-5@20251101:rawPredict",
            ),
            (
                "us-central1",
                "vertex/mistralai/mistral-medium-3",
                "streamRawPredict",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/publishers/mistralai/models/mistral-medium-3:streamRawPredict",
            ),
            (
                "europe-west4",
                "vertex/mistralai/codestral-2",
                "rawPredict",
                "https://europe-west4-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/europe-west4/publishers/mistralai/models/codestral-2:rawPredict",
            ),
        ] {
            let url = target(location)
                .model_method_url(&model(raw), method)
                .unwrap();
            assert_eq!(url.as_str(), expected, "{raw} {location}");
            assert_eq!(url.query(), None);
        }
    }

    #[test]
    fn openapi_url_uses_the_documented_responses_and_chat_completions_paths() {
        let grok = model("vertex/xai/grok-4.7");
        for (location, api, expected) in [
            (
                "global",
                OpenApiPath::Responses,
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/endpoints/openapi/responses",
            ),
            (
                "us",
                OpenApiPath::Responses,
                "https://aiplatform.us.rep.googleapis.com/v1/projects/synthetic-project/locations/us/endpoints/openapi/responses",
            ),
            (
                "us-central1",
                OpenApiPath::ChatCompletions,
                "https://us-central1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/us-central1/endpoints/openapi/chat/completions",
            ),
            (
                "global",
                OpenApiPath::ChatCompletions,
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/endpoints/openapi/chat/completions",
            ),
        ] {
            assert_eq!(
                target(location).openapi_url(&grok, api).unwrap().as_str(),
                expected,
                "{location} {api:?}"
            );
        }
    }

    #[test]
    fn openapi_url_synthetic_follows_the_model_api_version() {
        // The Llama model page sends Chat Completions to `v1beta1` in
        // `us-east5`; a Grok model in that version is synthetic and shows
        // that the version follows the model, whatever its family.
        let beta = ResolvedModel {
            api_version: ApiVersion::V1Beta1,
            ..model("vertex/xai/grok-4.7")
        };
        assert_eq!(
            target("us-east5")
                .openapi_url(&beta, OpenApiPath::ChatCompletions)
                .unwrap()
                .as_str(),
            "https://us-east5-aiplatform.googleapis.com/v1beta1/projects/synthetic-project/locations/us-east5/endpoints/openapi/chat/completions"
        );
        assert_eq!(
            target("us-east5")
                .model_method_url(&beta, "rawPredict")
                .unwrap()
                .as_str(),
            "https://us-east5-aiplatform.googleapis.com/v1beta1/projects/synthetic-project/locations/us-east5/publishers/xai/models/xai%2Fgrok-4.7:rawPredict"
        );
    }

    #[test]
    fn publisher_responses_url_uses_the_documented_path_for_each_location_kind() {
        for (location, raw, expected) in [
            (
                "global",
                "vertex/meta/muse-spark-1.3",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/meta/v1/responses",
            ),
            (
                "global",
                "vertex/openai/gpt-6.1-sol",
                "https://aiplatform.googleapis.com/v1/projects/synthetic-project/locations/global/publishers/openai/v1/responses",
            ),
            (
                "us",
                "vertex/openai/gpt-6-luna",
                "https://aiplatform.us.rep.googleapis.com/v1/projects/synthetic-project/locations/us/publishers/openai/v1/responses",
            ),
            (
                "asia-northeast1",
                "vertex/openai/gpt-6-astra",
                "https://asia-northeast1-aiplatform.googleapis.com/v1/projects/synthetic-project/locations/asia-northeast1/publishers/openai/v1/responses",
            ),
        ] {
            assert_eq!(
                target(location)
                    .publisher_responses_url(&model(raw))
                    .unwrap()
                    .as_str(),
                expected,
                "{raw} {location}"
            );
        }
    }

    #[test]
    fn publisher_responses_url_synthetic_keeps_its_inner_v1_and_the_base_url_prefix() {
        let beta = ResolvedModel {
            api_version: ApiVersion::V1Beta1,
            ..model("vertex/openai/gpt-6.1-sol")
        };
        assert_eq!(
            target("global")
                .publisher_responses_url(&beta)
                .unwrap()
                .as_str(),
            "https://aiplatform.googleapis.com/v1beta1/projects/synthetic-project/locations/global/publishers/openai/v1/responses"
        );
        let relayed = VertexTarget::new(
            "synthetic-project",
            "global",
            Some("http://127.0.0.1:8080/relay/"),
        )
        .unwrap();
        assert_eq!(
            relayed
                .publisher_responses_url(&model("vertex/meta/muse-ember-1.0"))
                .unwrap()
                .as_str(),
            "http://127.0.0.1:8080/relay/v1/projects/synthetic-project/locations/global/publishers/meta/v1/responses"
        );
        let empty = ResolvedModel {
            publisher: String::new(),
            ..model("vertex/meta/muse-ember-1.0")
        };
        let error = target("global")
            .publisher_responses_url(&empty)
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }

    fn config_with(env: &[(&str, &str)]) -> ResolvedVertexConfig {
        let config_dir = tempfile::TempDir::new().unwrap();
        let env = env
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        crate::config::resolve_vertex_config(&env, config_dir.path())
    }

    #[test]
    fn target_for_model_synthetic_takes_the_location_from_the_model_rule() {
        let config = config_with(&[
            ("CCP_VERTEX_PROJECT", "synthetic-project"),
            ("CCP_VERTEX_LOCATION", "asia-northeast1"),
            (
                "CCP_VERTEX_LOCATION_OVERRIDES",
                "anthropic=eu,meta=us-east5,openai/gpt-6-luna=us",
            ),
        ]);
        for (raw, location) in [
            ("gemini-3.5-flash", "asia-northeast1"),
            ("vertex/anthropic/claude-opus-5-5", "eu"),
            ("vertex/meta/muse-spark-1.3", "global"),
            ("vertex/openai/gpt-6-luna", "us"),
            ("vertex/openai/gpt-6.1-sol", "asia-northeast1"),
        ] {
            let target = VertexTarget::for_model(&config, &model(raw)).unwrap();
            assert_eq!(target.location, location, "{raw}");
            assert_eq!(
                target.base_url.as_str(),
                format!("https://{}/", vertex_host(location)),
                "{raw}"
            );
        }

        let error = VertexTarget::for_model(&config, &model("vertex/mistralai/mistral-medium-3"))
            .unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.kind, ProviderErrorKind::InvalidRequest);
        for needle in [
            "CCP_VERTEX_LOCATION_OVERRIDES=mistralai=<location>",
            "vertex.locationOverrides",
            "us-central1 or europe-west4",
        ] {
            assert!(
                error.message.contains(needle),
                "{needle}: {}",
                error.message
            );
        }

        let regional = config_with(&[
            ("CCP_VERTEX_PROJECT", "synthetic-project"),
            ("CCP_VERTEX_LOCATION", "europe-west4"),
        ]);
        let target =
            VertexTarget::for_model(&regional, &model("vertex/mistralai/mistral-medium-3"))
                .unwrap();
        assert_eq!(target.location, "europe-west4");

        let invalid = config_with(&[
            ("CCP_VERTEX_PROJECT", "synthetic-project"),
            ("CCP_VERTEX_LOCATION_OVERRIDES", "anthropic=EU"),
        ]);
        // An invalid override is a configuration error for every model,
        // Gemini included, so it cannot go unnoticed.
        for raw in ["gemini-3.5-flash", "vertex/anthropic/claude-opus-5-5"] {
            let error = VertexTarget::for_model(&invalid, &model(raw)).unwrap_err();
            assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR, "{raw}");
            assert!(
                error
                    .message
                    .starts_with("Invalid Vertex configuration: Invalid Vertex setting CCP_VERTEX_LOCATION_OVERRIDES: "),
                "{}",
                error.message
            );
        }
    }

    // Error bodies from the partner families: neither Google's pages nor the
    // vendors' establish which shape Vertex AI answers with for them, so the
    // reader takes `/error/message` from each documented shape and falls back
    // to the body itself. These bodies are synthetic.

    #[test]
    fn error_detail_synthetic_reads_the_message_from_google_anthropic_and_openai_bodies() {
        for (body, status, message) in [
            (
                br#"{"error":{"code":400,"message":"Request contains an invalid argument.","status":"INVALID_ARGUMENT"}}"#.as_slice(),
                Some("INVALID_ARGUMENT"),
                "Request contains an invalid argument.",
            ),
            (
                br#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: Field required"}}"#.as_slice(),
                None,
                "max_tokens: Field required",
            ),
            (
                br#"{"error":{"message":"Unsupported parameter: 'reasoning.summary'","type":"invalid_request_error","param":"reasoning.summary","code":"unsupported_parameter"}}"#.as_slice(),
                None,
                "Unsupported parameter: 'reasoning.summary'",
            ),
        ] {
            for detail in [
                UpstreamErrorDetail::parse(body, None, &[]),
                UpstreamErrorDetail::read(body, None, &[]),
            ] {
                assert_eq!(detail.status.as_deref(), status, "{message}");
                assert_eq!(detail.message.as_deref(), Some(message));
            }
        }
    }

    #[test]
    fn error_detail_synthetic_falls_back_to_the_sanitised_body() {
        for (body, message) in [
            (
                b"<html><body>502 Bad Gateway</body></html>".as_slice(),
                Some("<html><body>502 Bad Gateway</body></html>"),
            ),
            (br#"{"message":"x"}"#.as_slice(), Some("x")),
            (
                br#"{"error":"overloaded"}"#.as_slice(),
                Some(r#"{"error":"overloaded"}"#),
            ),
            (b"  upstream reset \n".as_slice(), Some("upstream reset")),
            (b"".as_slice(), None),
            (b" \r\n\t".as_slice(), None),
        ] {
            let detail = UpstreamErrorDetail::read(body, Some("30"), &[]);
            assert_eq!(detail.message.as_deref(), message, "{body:?}");
            assert_eq!(detail.status, None, "{body:?}");
            assert_eq!(detail.retry_after_secs, Some(30), "{body:?}");
        }
        // A status alone is enough; the body is not used then.
        let status_only = br#"{"error":{"code":503,"status":"UNAVAILABLE"}}"#;
        assert_eq!(
            UpstreamErrorDetail::read(status_only, None, &[]),
            UpstreamErrorDetail {
                message: None,
                status: Some("UNAVAILABLE".to_string()),
                retry_after_secs: None,
            }
        );

        let secret_body =
            format!("upstream refused the token {SYNTHETIC_TOKEN} for synthetic@example.com");
        let detail = UpstreamErrorDetail::read(secret_body.as_bytes(), None, &[SYNTHETIC_TOKEN]);
        assert_eq!(
            detail.message.as_deref(),
            Some("upstream refused the token [redacted] for [redacted]")
        );

        // Cut to 500 characters after redaction, counted in characters.
        let long = "Überlastung ".repeat(100);
        let message = UpstreamErrorDetail::read(long.as_bytes(), None, &[])
            .message
            .unwrap();
        assert_eq!(message.chars().count(), 500);
        assert!(long.starts_with(&message));

        // Bytes that are not UTF-8 are replaced rather than dropping the body.
        let detail = UpstreamErrorDetail::read(b"bad \xff gateway", None, &[]);
        assert_eq!(detail.message.as_deref(), Some("bad \u{fffd} gateway"));
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
            location_overrides_setting: None,
        }
    }

    fn local_client() -> VertexClient {
        VertexClient::with_client(client_builder().no_proxy().build().unwrap())
    }

    async fn send(client: &VertexClient, base_url: &str) -> Result<VertexResponse, VertexError> {
        client
            .stream_generate_content(
                &local_config(base_url),
                &crate::providers::vertex::models::resolve(MODEL).unwrap(),
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
    async fn post_synthetic_sends_the_accept_and_extra_headers_of_the_call() {
        let (base, seen) = spawn_upstream(|| {
            (
                [(http::header::CONTENT_TYPE, "application/json")],
                "{\"type\":\"message\"}",
            )
                .into_response()
        })
        .await;
        let config = local_config(&base);
        let model =
            crate::providers::vertex::models::resolve("vertex/anthropic/claude-opus-5-5").unwrap();
        let target = VertexTarget::for_model(&config, &model).unwrap();
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            http::HeaderValue::from_static("claude-code-20250219,effort-2025-11-24"),
        );
        let call = UpstreamCall {
            url: target.model_method_url(&model, "rawPredict").unwrap(),
            accept: "application/json",
            headers,
        };
        local_client()
            .post(
                &target,
                &model.upstream_model,
                call,
                &serde_json::json!({"anthropic_version": "vertex-2023-10-16"}),
                &fixed_credentials(),
                None,
            )
            .await
            .unwrap();

        let seen = seen.lock().unwrap().take().unwrap();
        assert_eq!(
            seen.path_and_query,
            "/v1/projects/synthetic-project/locations/global/publishers/anthropic/models/claude-opus-5-5:rawPredict"
        );
        let header = |name: &str| {
            seen.headers
                .iter()
                .filter(|(seen_name, _)| seen_name == name)
                .map(|(_, value)| value.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            header("anthropic-beta"),
            vec!["claude-code-20250219,effort-2025-11-24"]
        );
        assert_eq!(header("accept"), vec!["application/json"]);
        assert_eq!(header("content-type"), vec!["application/json"]);
        assert_eq!(
            header("authorization"),
            vec![format!("Bearer {SYNTHETIC_TOKEN}").as_str()]
        );
    }

    #[test]
    fn capture_synthetic_shows_the_call_accept_and_extra_headers_but_no_credential() {
        let mut auth = http::HeaderMap::new();
        auth.insert(
            http::header::AUTHORIZATION,
            format!("Bearer {SYNTHETIC_TOKEN}").parse().unwrap(),
        );
        let mut extra = http::HeaderMap::new();
        extra.insert(
            "anthropic-beta",
            http::HeaderValue::from_static("claude-code-20250219"),
        );
        assert_eq!(
            request_headers_for_capture(&auth, "application/json", &extra),
            serde_json::json!({
                "anthropic-beta": "claude-code-20250219",
                "authorization": "[redacted]",
                "accept": "application/json",
                "content-type": "application/json",
                "user-agent": USER_AGENT,
            })
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
    #[tokio::test]
    async fn request_synthetic_plain_text_error_body_becomes_the_redacted_message() {
        fn plain() -> Response {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "upstream connect error or disconnect/reset before headers; token {SYNTHETIC_TOKEN}"
                ),
            )
                .into_response()
        }
        let (base, _) = spawn_upstream(plain).await;
        let error = send(&local_client(), &base).await.unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert_eq!(
            error.message_with_status(),
            "upstream connect error or disconnect/reset before headers; token [redacted] (upstream HTTP 503)"
        );
    }

    #[test]
    fn partner_403_404_synthetic_messages_also_name_model_garden() {
        let target = target("us-central1");
        let partner = |status: u16| {
            VertexError::from_upstream(
                StatusCode::from_u16(status).unwrap(),
                detail("synthetic upstream text", "SYNTHETIC"),
                &target,
                "xai/grok-4.7",
            )
            .with_partner_guidance(&target, "xai/grok-4.7")
            .message_with_status()
        };
        assert_eq!(
            partner(403),
            "SYNTHETIC: synthetic upstream text. Check that the account has the role roles/aiplatform.user on project synthetic-project, that the Vertex AI API (aiplatform.googleapis.com) is enabled there, and, for user credentials, that a quota project is set: gcloud auth application-default set-quota-project synthetic-project. Also check that model xai/grok-4.7 is enabled in Model Garden for project synthetic-project in location us-central1 and, for a preview model, that the project has been granted access; the status alone does not say which of these is the cause (upstream HTTP 403)"
        );
        assert_eq!(
            partner(404),
            "Model xai/grok-4.7 is not available in project synthetic-project / location us-central1: SYNTHETIC: synthetic upstream text. Check the model ID and that the model is offered in that location (CCP_VERTEX_LOCATION, GOOGLE_CLOUD_LOCATION or vertex.location). Also check that model xai/grok-4.7 is enabled in Model Garden for project synthetic-project in location us-central1 and, for a preview model, that the project has been granted access; the status alone does not say which of these is the cause (upstream HTTP 404)"
        );

        // Every other status, and an error with no upstream status, is left
        // as it was.
        for status in [400, 401, 429, 500] {
            let plain = VertexError::from_upstream(
                StatusCode::from_u16(status).unwrap(),
                detail("synthetic upstream text", "SYNTHETIC"),
                &target,
                "xai/grok-4.7",
            );
            assert_eq!(
                plain.clone().with_partner_guidance(&target, "xai/grok-4.7"),
                plain,
                "{status}"
            );
        }
        let unreachable = VertexError::unreachable();
        assert_eq!(
            unreachable
                .clone()
                .with_partner_guidance(&target, "xai/grok-4.7"),
            unreachable
        );
    }

    #[test]
    fn error_detail_synthetic_reads_a_top_level_message_such_as_mistrals() {
        let body = format!(
            r#"{{"object":"error","message":"saw {SYNTHETIC_TOKEN} in the request","type":"invalid_request_error","code":null}}"#
        );
        let detail = UpstreamErrorDetail::read(body.as_bytes(), None, &[SYNTHETIC_TOKEN]);
        assert_eq!(
            detail.message.as_deref(),
            Some("saw [redacted] in the request")
        );
        assert_eq!(detail.status, None);
        let mapped = VertexError::from_upstream(
            StatusCode::BAD_REQUEST,
            detail,
            &target("us-central1"),
            "mistral-medium-3",
        )
        .message_with_status();
        assert_eq!(mapped, "saw [redacted] in the request (upstream HTTP 400)");

        // `/error/message` comes first; a top-level message that is blank or
        // not a string is not one.
        let both = br#"{"error":{"message":"nested"},"message":"top"}"#;
        assert_eq!(
            UpstreamErrorDetail::read(both, None, &[])
                .message
                .as_deref(),
            Some("nested")
        );
        for body in [
            br#"{"message":"  "}"#.as_slice(),
            br#"{"message":{"text":"x"}}"#.as_slice(),
        ] {
            assert_eq!(
                UpstreamErrorDetail::read(body, None, &[])
                    .message
                    .as_deref(),
                Some(std::str::from_utf8(body).unwrap()),
                "{body:?}"
            );
        }
    }
}
