//! HTTP client for the Muse Model API (`POST {base_url}/responses`).
//!
//! A 401 gets exactly one recovery: the key is re-obtained from the manager
//! and the request replayed once. 429, 5xx and transport failures are not
//! retried here; Claude Code retries on its own. There is no total or
//! first-byte timeout, because a large high-effort request was observed to
//! send nothing, not even headers, for 78 seconds.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use http::StatusCode;
use serde::Serialize;

use super::auth::constants::{ACCOUNTS_CENTER_URL, LOGIN_COMMAND, MAX_AUTH_RESPONSE_BYTES};
use super::auth::manager::{ActiveKey, AuthFailure, MuseAuthManager};
use super::auth::mint::{MintError, with_upstream_status};
use super::sanitize::{sanitize_error_value, sanitize_upstream_text, sanitize_upstream_token};
use crate::anthropic::error::json_error;
use crate::provider::{ProviderError, ProviderErrorKind};
use crate::traffic::TrafficCapture;

const DEFAULT_BASE_URL: &str = "https://api.meta.ai/v1";
const MAX_ERROR_BODY_BYTES: usize = MAX_AUTH_RESPONSE_BYTES;
const FALLBACK_MESSAGE: &str = "Muse API request failed";
/// How long an error body may take to arrive. Only error bodies are bounded:
/// the status is already known, and an upstream that keeps one open must not
/// hold the request (or its 401 recovery) forever.
const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(5);
/// Largest `retry-after` passed on, one day in seconds, whether it came from
/// the upstream header or from the usage figures.
const MAX_RETRY_AFTER_SECS: u64 = 86_400;

/// Response headers that may be copied into a traffic capture. Everything
/// else is dropped; `Proxy-Status` in particular carries client-address data.
const CAPTURED_RESPONSE_HEADERS: [&str; 4] =
    ["x-request-id", "x-route", "retry-after", "content-type"];

/// A failed Muse request, already sanitised. `status` is what the proxy
/// answers with; `upstream_status` is what Muse answered, when it answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseError {
    pub kind: ProviderErrorKind,
    pub status: StatusCode,
    pub upstream_status: Option<StatusCode>,
    pub message: String,
    pub code: Option<String>,
    pub param: Option<String>,
    pub retry_after: Option<String>,
}

impl MuseError {
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
            code: None,
            param: None,
            retry_after: None,
        }
    }

    /// The Model API could not be reached or the connection failed before a
    /// status arrived.
    pub fn unreachable() -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::BAD_GATEWAY,
            None,
            "Muse API is unreachable",
        )
    }

    /// The 200 stream ended before its terminal event.
    pub fn stream_ended() -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::BAD_GATEWAY,
            Some(StatusCode::OK),
            "Muse stream ended before completion",
        )
    }

    /// The 200 stream reported a failure (`error` or `response.failed`).
    /// `message` must already be sanitised with `sanitize_upstream_text`.
    pub fn stream_failed(message: impl Into<String>) -> Self {
        Self::new(
            ProviderErrorKind::Api,
            StatusCode::BAD_GATEWAY,
            Some(StatusCode::OK),
            message,
        )
    }

    /// Maps a failure to obtain a key.
    pub fn from_auth(failure: AuthFailure) -> Self {
        let message = failure.message();
        let upstream = failure.upstream_status();
        let (kind, status) = match &failure {
            AuthFailure::NotAuthenticated | AuthFailure::Mint(MintError::Unauthorized) => {
                (ProviderErrorKind::Authentication, StatusCode::UNAUTHORIZED)
            }
            AuthFailure::Mint(MintError::Rejected { .. }) => {
                (ProviderErrorKind::Api, StatusCode::BAD_GATEWAY)
            }
            AuthFailure::Mint(MintError::Transient { .. }) => {
                (ProviderErrorKind::Api, StatusCode::SERVICE_UNAVAILABLE)
            }
        };
        Self::new(kind, status, upstream, message)
    }

    /// Maps a non-success Model API answer. The class comes from the HTTP
    /// status alone; the body's `error.type` and `error.code` are never used
    /// to decide it, only passed on.
    pub fn from_upstream(
        upstream: StatusCode,
        detail: UpstreamErrorDetail,
        auth: &MuseAuthManager,
    ) -> Self {
        let text = detail
            .message
            .clone()
            .unwrap_or_else(|| FALLBACK_MESSAGE.to_string());
        let (kind, status, message) = match upstream.as_u16() {
            401 => (
                ProviderErrorKind::Authentication,
                StatusCode::UNAUTHORIZED,
                match &detail.message {
                    Some(upstream_message) => format!(
                        "Muse rejected the API key. Run: {LOGIN_COMMAND}. Upstream message: {upstream_message}"
                    ),
                    None => format!("Muse rejected the API key. Run: {LOGIN_COMMAND}"),
                },
            ),
            400 | 404 => (ProviderErrorKind::InvalidRequest, upstream, text),
            402 | 403 => {
                let message = if auth.subscription_inactive() {
                    format!(
                        "{text}. This account has no active Muse Code subscription: {ACCOUNTS_CENTER_URL}"
                    )
                } else {
                    text
                };
                (
                    ProviderErrorKind::Permission,
                    StatusCode::FORBIDDEN,
                    message,
                )
            }
            429 => (ProviderErrorKind::RateLimit, upstream, text),
            529 => (
                ProviderErrorKind::Api,
                StatusCode::SERVICE_UNAVAILABLE,
                text,
            ),
            400..=599 => (ProviderErrorKind::Api, upstream, text),
            _ => (ProviderErrorKind::Api, StatusCode::BAD_GATEWAY, text),
        };
        let mut error = Self::new(kind, status, Some(upstream), message);
        error.code = detail.code;
        error.param = detail.param;
        error.retry_after = detail.retry_after.or_else(|| {
            (upstream == StatusCode::TOO_MANY_REQUESTS)
                .then(|| auth.retry_after_secs(now_unix()))
                .flatten()
                .map(|secs| secs.min(MAX_RETRY_AFTER_SECS).to_string())
        });
        error
    }

    /// Message as shown to clients, ending with `(upstream HTTP <status>)`
    /// whenever Muse answered with a status.
    pub fn message_with_status(&self) -> String {
        with_upstream_status(self.message.clone(), self.upstream_status)
    }

    pub fn into_provider_error(self) -> ProviderError {
        let mut error = ProviderError::new(self.status, self.kind, self.message_with_status());
        error.retry_after = self.retry_after;
        error.code = self.code;
        error.param = self.param;
        error
    }
}

impl From<MuseError> for ProviderError {
    fn from(error: MuseError) -> Self {
        error.into_provider_error()
    }
}

/// Anthropic-format error response for `/v1/messages`, with `retry-after`.
pub fn anthropic_error_response(error: ProviderError) -> Response {
    let response = json_error(error.status, error.error_type(), error.message);
    match error
        .retry_after
        .and_then(|value| http::HeaderValue::from_str(&value).ok())
    {
        Some(value) => ([(http::header::RETRY_AFTER, value)], response).into_response(),
        None => response,
    }
}

/// The sanitised parts of a Model API error body plus its `retry-after`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamErrorDetail {
    pub message: Option<String>,
    pub code: Option<String>,
    pub param: Option<String>,
    pub retry_after: Option<String>,
}

impl UpstreamErrorDetail {
    /// Reads `{"error":{"message","code","param"}}`, sanitising each field.
    pub fn parse(body: &[u8], retry_after: Option<String>, secrets: &[&str]) -> Self {
        let value = serde_json::from_slice::<serde_json::Value>(body).ok();
        let error = value.as_ref().and_then(|value| value.get("error"));
        let field = |name: &str| {
            error
                .and_then(|error| error.get(name))
                .and_then(|value| value.as_str())
        };
        Self {
            message: field("message")
                .filter(|message| !message.trim().is_empty())
                .map(|message| sanitize_upstream_text(message, secrets)),
            code: field("code").and_then(|code| sanitize_upstream_token(code, secrets)),
            param: field("param").and_then(|param| sanitize_upstream_token(param, secrets)),
            retry_after,
        }
    }
}

/// A successful Model API response and the key that produced it.
pub struct MuseResponse {
    pub response: reqwest::Response,
    pub key: ActiveKey,
}

/// Connection settings for the Model API: no redirects, a connect timeout,
/// HTTP/2 keep-alive, and deliberately no total or read timeout.
pub fn client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(15))
        .http2_keep_alive_timeout(Duration::from_secs(5))
        .http2_keep_alive_while_idle(true)
}

pub struct MuseResponsesClient {
    client: reqwest::Client,
    auth: Arc<MuseAuthManager>,
    url: String,
    user_agent: String,
    error_body_timeout: Duration,
}

impl MuseResponsesClient {
    pub fn new(
        base_url: String,
        user_agent: String,
        auth: Arc<MuseAuthManager>,
    ) -> anyhow::Result<Self> {
        Self::with_client(client_builder().build()?, base_url, user_agent, auth)
    }

    pub fn with_client(
        client: reqwest::Client,
        base_url: String,
        user_agent: String,
        auth: Arc<MuseAuthManager>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            client,
            auth,
            url: responses_url(&base_url)?,
            user_agent,
            error_body_timeout: ERROR_BODY_TIMEOUT,
        })
    }

    pub fn auth(&self) -> &Arc<MuseAuthManager> {
        &self.auth
    }

    /// Sends the request, recovering once from a 401. On success the caller
    /// owns the streaming response and the key it ran under (for usage
    /// attribution and for scrubbing upstream text).
    pub async fn post<B: Serialize + ?Sized>(
        &self,
        body: &B,
        traffic: Option<&TrafficCapture>,
    ) -> Result<MuseResponse, MuseError> {
        let bytes = serde_json::to_vec(body).map_err(|_| {
            MuseError::new(
                ProviderErrorKind::Api,
                StatusCode::INTERNAL_SERVER_ERROR,
                None,
                "Muse request could not be encoded",
            )
        })?;
        if let Some(capture) = traffic {
            let body_value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            capture.write_json("020-upstream-request", &body_value);
        }
        let key = self.auth.get_key().await.map_err(|failure| {
            capture_failure(traffic, "auth", 0, failure.upstream_status());
            MuseError::from_auth(failure)
        })?;
        if let Some(capture) = traffic {
            capture.write_json(
                "021-upstream-request-metadata",
                &serde_json::json!({
                    "method": "POST",
                    "url": safe_url(&self.url_for(&key)),
                    "provider": "muse",
                    "transport": "http",
                    "headers": {
                        "accept": "text/event-stream",
                        "content-type": "application/json",
                        "authorization": "[redacted]",
                        "user-agent": self.user_agent,
                    },
                    "body_bytes": bytes.len(),
                }),
            );
        }

        let response = self.attempt(&key, &bytes, 1, traffic).await?;
        if response.status() != StatusCode::UNAUTHORIZED {
            return self.finish(response, key, 1, traffic).await;
        }
        let _ = self.error_detail(response, &key, 1, traffic).await;
        let fresh = self
            .auth
            .recover(key.login, key.generation)
            .await
            .map_err(|failure| {
                capture_failure(traffic, "recover", 1, failure.upstream_status());
                MuseError::from_auth(failure)
            })?;
        let replay = self.attempt(&fresh, &bytes, 2, traffic).await?;
        self.finish(replay, fresh, 2, traffic).await
    }

    async fn finish(
        &self,
        response: reqwest::Response,
        key: ActiveKey,
        attempt: u8,
        traffic: Option<&TrafficCapture>,
    ) -> Result<MuseResponse, MuseError> {
        let status = response.status();
        if let Some(capture) = traffic {
            capture.write_json(
                "030-upstream-response-headers",
                &serde_json::json!({
                    "attempt": attempt,
                    "status": status.as_u16(),
                    "headers": safe_headers(response.headers()),
                }),
            );
        }
        if status.is_success() {
            return Ok(MuseResponse { response, key });
        }
        let detail = self.error_detail(response, &key, attempt, traffic).await;
        Err(MuseError::from_upstream(status, detail, &self.auth))
    }

    async fn attempt(
        &self,
        key: &ActiveKey,
        body: &[u8],
        attempt: u8,
        traffic: Option<&TrafficCapture>,
    ) -> Result<reqwest::Response, MuseError> {
        let started = Instant::now();
        let response = self
            .client
            .post(self.url_for(key))
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", key.api_key),
            )
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::ACCEPT, "text/event-stream")
            .header(http::header::USER_AGENT, &self.user_agent)
            .body(body.to_vec())
            .send()
            .await
            .map_err(|_| {
                capture_failure(traffic, "transport", attempt, None);
                MuseError::unreachable()
            })?;
        if let Some(capture) = traffic {
            capture.write_json(
                "022-upstream-attempt",
                &serde_json::json!({
                    "attempt": attempt,
                    "status": response.status().as_u16(),
                    "elapsed_ms": started.elapsed().as_millis(),
                    "headers": safe_headers(response.headers()),
                }),
            );
        }
        Ok(response)
    }

    /// Reads and sanitises an error body; the capture gets the sanitised
    /// copy only.
    async fn error_detail(
        &self,
        response: reqwest::Response,
        key: &ActiveKey,
        attempt: u8,
        traffic: Option<&TrafficCapture>,
    ) -> UpstreamErrorDetail {
        let status = response.status();
        let retry_after = response
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(retry_after_secs);
        let (body, complete) = match tokio::time::timeout(
            self.error_body_timeout,
            read_bounded(response, MAX_ERROR_BODY_BYTES),
        )
        .await
        {
            Ok((body, truncated)) => (body, !truncated),
            Err(_) => (Vec::new(), false),
        };
        let truncated = !complete;
        let secrets = key.secrets();
        if let Some(capture) = traffic {
            let captured = serde_json::from_slice::<serde_json::Value>(&body)
                .map(|value| sanitize_error_value(&value, &secrets))
                .unwrap_or_else(|_| serde_json::json!({ "body_bytes": body.len() }));
            capture.write_json(
                "031-upstream-error-body",
                &serde_json::json!({
                    "attempt": attempt,
                    "status": status.as_u16(),
                    "truncated": truncated,
                    "body": captured,
                }),
            );
        }
        if !complete {
            // A cut-off body is not parsed; the status decides the error.
            return UpstreamErrorDetail {
                retry_after,
                ..UpstreamErrorDetail::default()
            };
        }
        UpstreamErrorDetail::parse(&body, retry_after, &secrets)
    }

    fn url_for(&self, key: &ActiveKey) -> String {
        if key.base_url.trim().is_empty() {
            return self.url.clone();
        }
        responses_url(&key.base_url).unwrap_or_else(|_| self.url.clone())
    }
}

/// Reads at most `limit` bytes. The flag is true when the body was cut at the
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

/// An upstream `retry-after` is passed on only as whole seconds, at most one
/// day. Anything else (an HTTP date, a negative or fractional value) is
/// dropped so the usage-derived value can take its place.
fn retry_after_secs(value: &str) -> Option<String> {
    let secs: u64 = value.trim().parse().ok()?;
    Some(secs.min(MAX_RETRY_AFTER_SECS).to_string())
}

fn capture_failure(
    traffic: Option<&TrafficCapture>,
    stage: &str,
    attempt: u8,
    upstream_status: Option<StatusCode>,
) {
    if let Some(capture) = traffic {
        capture.write_json(
            "060-muse-request-error",
            &serde_json::json!({
                "stage": stage,
                "attempt": attempt,
                "upstream_status": upstream_status.map(|status| status.as_u16()),
            }),
        );
    }
}

/// Copies only the allowlisted response headers.
pub fn safe_headers(headers: &reqwest::header::HeaderMap) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for name in CAPTURED_RESPONSE_HEADERS {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            result.insert(name.to_string(), value.into());
        }
    }
    serde_json::Value::Object(result)
}

fn safe_url(raw: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(raw) else {
        return "[invalid-url]".into();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.to_string()
}

fn responses_url(base_url: &str) -> anyhow::Result<String> {
    let base_url = if base_url.trim().is_empty() {
        DEFAULT_BASE_URL
    } else {
        base_url.trim()
    };
    let mut url = reqwest::Url::parse(base_url)?;
    let path = url.path().trim_end_matches('/');
    if !path.ends_with("/responses") {
        url.set_path(&format!("{path}/responses"));
    }
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_compat::OpenAiError;
    use crate::providers::muse::auth::constants::MINT_TIMEOUT;
    use crate::providers::muse::auth::manager::{CALLER_DEADLINE, HttpMinter};
    use crate::providers::muse::auth::mint::{self, MintTarget};
    use crate::providers::muse::auth::token_store::{StoredAuth, store_at};
    use crate::providers::muse::usage::{SubscriptionUsage, UsageSink, UsageWindow};
    use axum::Router;
    use axum::routing::post;
    use http_body_util::BodyExt;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const ACCESS_TOKEN: &str = "synthetic-oauth-access-token-for-client-tests";
    const API_KEY: &str = "synthetic-minted-api-key-48-chars-0000000000000";
    const EMAIL: &str = "synthetic.person@example.com";

    const OBSERVED_401: &str = r#"{"error":{"code":"invalid_api_key","message":"Unauthorized","param":null,"type":"authentication_error"}}"#;
    const OBSERVED_404: &str = r#"{"error":{"code":"model_not_found","message":"The requested model was not found. It may not exist, or this API key may not have access to it. Check the model name; GET /v1/models lists the models this API key can use.","param":null,"type":"invalid_request_error"}}"#;
    const OBSERVED_400: &str = r#"{"error":{"code":null,"message":"reasoning_effort 'none' is not supported for model 'muse-spark-1.3'. Supported values: [minimal, low, medium, high, xhigh, max]","param":"reasoning.effort","type":"invalid_request_error"}}"#;
    const OBSERVED_405: &str = r#"{"error":{"code":"method_not_allowed","message":"Method not allowed.","param":null,"type":"invalid_request_error"}}"#;
    const OBSERVED_SSE: &str = "event: response.created\ndata: {\"type\":\"response.created\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n";

    #[derive(Clone)]
    struct Canned {
        status: StatusCode,
        headers: Vec<(&'static str, String)>,
        body: String,
        delay: Option<Duration>,
        /// Send the head and `body`, then keep the body open without sending
        /// anything more.
        silent_body: bool,
    }

    fn canned(status: StatusCode, body: impl Into<String>) -> Canned {
        Canned {
            status,
            headers: vec![("content-type", "application/json".into())],
            body: body.into(),
            delay: None,
            silent_body: false,
        }
    }

    fn sse_ok() -> Canned {
        Canned {
            status: StatusCode::OK,
            headers: vec![("content-type", "text/event-stream".into())],
            body: OBSERVED_SSE.into(),
            delay: None,
            silent_body: false,
        }
    }

    fn respond(canned: Canned) -> axum::response::Response {
        let body = if canned.silent_body {
            let sent = (!canned.body.is_empty())
                .then(|| Ok::<_, std::io::Error>(bytes::Bytes::from(canned.body)));
            axum::body::Body::from_stream(
                futures_util::stream::iter(sent).chain(futures_util::stream::pending()),
            )
        } else {
            axum::body::Body::from(canned.body)
        };
        let mut response = axum::response::Response::new(body);
        *response.status_mut() = canned.status;
        for (name, value) in canned.headers {
            response.headers_mut().insert(
                http::HeaderName::from_static(name),
                http::HeaderValue::from_str(&value).unwrap(),
            );
        }
        response
    }

    #[derive(Clone, Debug)]
    struct Seen {
        headers: http::HeaderMap,
        body: String,
    }

    /// Model API and key mint on one local server.
    struct Upstream {
        base: String,
        model_calls: Arc<Mutex<Vec<Seen>>>,
        mint_calls: Arc<AtomicUsize>,
    }

    impl Upstream {
        fn model_calls(&self) -> usize {
            self.model_calls.lock().unwrap().len()
        }

        fn mint_calls(&self) -> usize {
            self.mint_calls.load(Ordering::SeqCst)
        }
    }

    fn mint_answer() -> Canned {
        let mut body = mint::tests::observed_mint_body();
        body["api_key"] = API_KEY.into();
        canned(StatusCode::OK, body.to_string())
    }

    /// Model answers and mint answers are served in order; once a queue is
    /// empty the model streams the observed SSE and the mint mints.
    async fn spawn_upstream(model: Vec<Canned>, mint: Vec<Canned>) -> Upstream {
        let model_calls: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let mint_calls = Arc::new(AtomicUsize::new(0));
        let queue = Arc::new(Mutex::new(VecDeque::from(model)));
        let mint_queue = Arc::new(Mutex::new(VecDeque::from(mint)));
        let recorded = model_calls.clone();
        let minted = mint_calls.clone();
        let app = Router::new()
            .route(
                "/v1/responses",
                post(move |headers: http::HeaderMap, body: String| {
                    recorded.lock().unwrap().push(Seen { headers, body });
                    let next = queue.lock().unwrap().pop_front().unwrap_or_else(sse_ok);
                    async move {
                        if let Some(delay) = next.delay {
                            tokio::time::sleep(delay).await;
                        }
                        respond(next)
                    }
                }),
            )
            .route(
                "/muse-code/key",
                post(move || {
                    minted.fetch_add(1, Ordering::SeqCst);
                    let next = mint_queue
                        .lock()
                        .unwrap()
                        .pop_front()
                        .unwrap_or_else(mint_answer);
                    async move { respond(next) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Upstream {
            base,
            model_calls,
            mint_calls,
        }
    }

    struct Harness {
        _dir: tempfile::TempDir,
        path: String,
        upstream: Upstream,
        client: MuseResponsesClient,
    }

    fn save_login(path: &str, token: &str) {
        store_at(path.to_string())
            .save_auth(StoredAuth {
                access_token: token.to_string(),
                obtained_at_ms: 1_794_274_189_000,
                client_id: "1031625952748946".into(),
                oauth_host: "https://auth.meta.com".into(),
                user_full_name: None,
                is_subs_active: true,
                subs_tier_name: None,
            })
            .unwrap();
    }

    async fn harness_with(model: Vec<Canned>, mint: Vec<Canned>, logged_in: bool) -> Harness {
        let upstream = spawn_upstream(model, mint).await;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("auth.json").to_string_lossy().into_owned();
        if logged_in {
            save_login(&path, ACCESS_TOKEN);
        }
        let model_base = format!("{}/v1", upstream.base);
        let minter = Arc::new(HttpMinter::with_client(
            mint::async_client_builder(MINT_TIMEOUT)
                .no_proxy()
                .build()
                .unwrap(),
            MintTarget {
                mint_base_url: upstream.base.clone(),
                model_base_url: model_base.clone(),
                user_agent: "claude-code-proxy/test".into(),
            },
        ));
        let manager = Arc::new(MuseAuthManager::with_minter(
            store_at(path.clone()),
            minter,
            MINT_TIMEOUT,
            CALLER_DEADLINE,
        ));
        let client = MuseResponsesClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            model_base,
            "claude-code-proxy/test".into(),
            manager,
        )
        .unwrap();
        Harness {
            _dir: dir,
            path,
            upstream,
            client,
        }
    }

    async fn harness(model: Vec<Canned>) -> Harness {
        harness_with(model, vec![], true).await
    }

    fn request_body() -> serde_json::Value {
        serde_json::json!({"model": "muse-spark-1.3", "input": "Reply with the single word ok", "stream": true, "store": false})
    }

    async fn send(harness: &Harness) -> Result<MuseResponse, MuseError> {
        harness.client.post(&request_body(), None).await
    }

    async fn post_err(harness: &Harness) -> MuseError {
        send(harness).await.err().expect("request should fail")
    }

    #[tokio::test]
    async fn application_sets_exactly_four_request_headers() {
        let harness = harness(vec![]).await;
        let response = send(&harness).await.unwrap();
        assert_eq!(response.response.status(), StatusCode::OK);
        assert_eq!(response.key.api_key, API_KEY);
        let seen = harness.upstream.model_calls.lock().unwrap()[0].clone();
        let mut names: Vec<_> = seen
            .headers
            .keys()
            .map(|name| name.as_str().to_string())
            .collect();
        names.sort();
        // `host` and `content-length` are added by the HTTP library.
        assert_eq!(
            names,
            [
                "accept",
                "authorization",
                "content-length",
                "content-type",
                "host",
                "user-agent"
            ]
        );
        assert_eq!(
            seen.headers["authorization"],
            format!("Bearer {API_KEY}").as_str()
        );
        assert_eq!(seen.headers["content-type"], "application/json");
        assert_eq!(seen.headers["accept"], "text/event-stream");
        assert_eq!(seen.headers["user-agent"], "claude-code-proxy/test");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&seen.body).unwrap(),
            request_body()
        );
    }

    #[tokio::test]
    async fn model_401_recovers_once_and_replays_once() {
        let harness = harness(vec![canned(StatusCode::UNAUTHORIZED, OBSERVED_401)]).await;
        let warm = harness.client.auth().get_key().await.unwrap();
        assert_eq!(harness.upstream.mint_calls(), 1);
        let response = send(&harness).await.unwrap();
        assert_eq!(response.response.status(), StatusCode::OK);
        assert_eq!(response.key.generation, warm.generation + 1);
        // The mint hands back the same key string; only the generation moved.
        assert_eq!(response.key.api_key, warm.api_key);
        assert_eq!(harness.upstream.model_calls(), 2);
        assert_eq!(harness.upstream.mint_calls(), 2);
    }

    #[tokio::test]
    async fn replay_401_is_an_authentication_error_after_two_sends_and_one_re_mint() {
        let harness = harness(vec![
            canned(StatusCode::UNAUTHORIZED, OBSERVED_401),
            canned(StatusCode::UNAUTHORIZED, OBSERVED_401),
        ])
        .await;
        harness.client.auth().get_key().await.unwrap();
        let error = post_err(&harness).await;
        assert_eq!(harness.upstream.model_calls(), 2);
        assert_eq!(harness.upstream.mint_calls(), 2);
        assert_eq!(error.kind, ProviderErrorKind::Authentication);
        assert_eq!(error.status, StatusCode::UNAUTHORIZED);
        assert_eq!(error.code.as_deref(), Some("invalid_api_key"));
        assert_eq!(
            error.message_with_status(),
            "Muse rejected the API key. Run: claude-code-proxy muse auth login. Upstream message: Unauthorized (upstream HTTP 401)"
        );
    }

    #[tokio::test]
    async fn recover_whose_mint_fails_returns_the_mint_error_without_replaying() {
        let harness = harness_with(
            vec![canned(StatusCode::UNAUTHORIZED, OBSERVED_401)],
            vec![mint_answer(), canned(StatusCode::SERVICE_UNAVAILABLE, "{}")],
            true,
        )
        .await;
        harness.client.auth().get_key().await.unwrap();
        let error = post_err(&harness).await;
        assert_eq!(harness.upstream.model_calls(), 1);
        assert_eq!(harness.upstream.mint_calls(), 2);
        assert_eq!(error.kind, ProviderErrorKind::Api);
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error.message_with_status(),
            "Muse key service is unavailable (upstream HTTP 503)"
        );
    }

    fn silent(status: StatusCode, retry_after: Option<&str>) -> Canned {
        let mut answer = canned(status, "");
        answer.silent_body = true;
        if let Some(value) = retry_after {
            answer.headers.push(("retry-after", value.to_string()));
        }
        answer
    }

    /// Fails the test, instead of hanging it, when a request never returns.
    async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(10), future)
            .await
            .expect("the request returns")
    }

    #[tokio::test]
    async fn synthetic_401_with_a_silent_body_still_recovers_and_replays() {
        let mut harness = harness(vec![silent(StatusCode::UNAUTHORIZED, None)]).await;
        harness.client.error_body_timeout = Duration::from_millis(200);
        let warm = harness.client.auth().get_key().await.unwrap();
        let response = bounded(send(&harness)).await.unwrap();
        assert_eq!(response.response.status(), StatusCode::OK);
        assert_eq!(response.key.generation, warm.generation + 1);
        assert_eq!(harness.upstream.model_calls(), 2);
        assert_eq!(harness.upstream.mint_calls(), 2);
    }

    #[tokio::test]
    async fn synthetic_503_with_a_silent_body_keeps_its_status_and_retry_after() {
        let mut harness = harness(vec![silent(StatusCode::SERVICE_UNAVAILABLE, Some("7"))]).await;
        harness.client.error_body_timeout = Duration::from_millis(200);
        let dir = tempfile::TempDir::new().unwrap();
        let capture = crate::traffic::test_capture(dir.path().join("traffic"));
        let error = bounded(harness.client.post(&request_body(), Some(&capture)))
            .await
            .err()
            .unwrap();
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::SERVICE_UNAVAILABLE,
                Some(StatusCode::SERVICE_UNAVAILABLE)
            )
        );
        assert_eq!(error.retry_after.as_deref(), Some("7"));
        assert_eq!(
            error.message_with_status(),
            "Muse API request failed (upstream HTTP 503)"
        );
        assert!(capture_text(dir.path()).contains(r#""truncated": true"#));
    }

    #[tokio::test]
    async fn synthetic_error_body_that_fills_the_size_limit_is_not_waited_on() {
        let mut answer = silent(StatusCode::BAD_REQUEST, None);
        answer.body = "x".repeat(MAX_ERROR_BODY_BYTES);
        let mut harness = harness(vec![answer]).await;
        // Longer than `bounded`, so only the size limit can end the read.
        harness.client.error_body_timeout = Duration::from_secs(60);
        let error = bounded(post_err(&harness)).await;
        assert_eq!(error.upstream_status, Some(StatusCode::BAD_REQUEST));
        assert_eq!(
            error.message_with_status(),
            "Muse API request failed (upstream HTTP 400)"
        );
    }

    #[tokio::test]
    async fn synthetic_upstream_retry_after_is_passed_on_only_as_bounded_seconds() {
        const DERIVED: u64 = 300;
        for (header, passed_on) in [
            ("120", Some(120)),
            (" 45 ", Some(45)),
            ("0", Some(0)),
            ("86400", Some(86_400)),
            ("999999999", Some(86_400)),
            // Dropped, so the fresh usage figures take over.
            ("-5", None),
            ("1.5", None),
            ("Wed, 21 Oct 2026 07:28:00 GMT", None),
        ] {
            let harness = harness(vec![rate_limited(Some(header))]).await;
            let key = harness.client.auth().get_key().await.unwrap();
            harness
                .client
                .auth()
                .record_usage(key.login, exhausted_window(now_unix() + DERIVED as i64));
            let error = post_err(&harness).await;
            let secs: u64 = error
                .retry_after
                .as_deref()
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("{header:?}: {:?}", error.retry_after));
            match passed_on {
                Some(expected) => assert_eq!(secs, expected, "{header:?}"),
                // The derived value counts down from when it is computed.
                None => assert!(
                    (DERIVED - 5..=DERIVED).contains(&secs),
                    "{header:?}: {secs}"
                ),
            }
            let response = anthropic_error_response(error.into_provider_error());
            assert_eq!(
                response.headers()["retry-after"],
                secs.to_string().as_str(),
                "{header:?}"
            );
        }
    }

    #[tokio::test]
    async fn synthetic_429_retry_after_from_a_weekly_reset_is_capped_at_one_day() {
        let harness = harness(vec![rate_limited(None)]).await;
        let key = harness.client.auth().get_key().await.unwrap();
        // Only the weekly window is exhausted, six days before its reset.
        let mut usage = exhausted_window(now_unix() + 3_600);
        usage.window.as_mut().unwrap().used_percent = 3;
        let weekly = usage.weekly.as_mut().unwrap();
        weekly.used_percent = 100;
        weekly.resets_at = now_unix() + 6 * 86_400;
        harness.client.auth().record_usage(key.login, usage);
        let error = post_err(&harness).await;
        assert_eq!(error.retry_after.as_deref(), Some("86400"));
    }

    #[tokio::test]
    async fn rate_limit_and_unavailable_are_not_retried() {
        for status in [
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            let mut answer = canned(
                status,
                r#"{"error":{"code":"synthetic","message":"synthetic upstream failure","param":null,"type":"api_error"}}"#,
            );
            answer.headers.push(("retry-after", "10".into()));
            let harness = harness(vec![answer]).await;
            let error = post_err(&harness).await;
            assert_eq!(harness.upstream.model_calls(), 1, "{status}");
            assert_eq!(error.upstream_status, Some(status));
            // An upstream retry-after is passed through unchanged.
            assert_eq!(error.retry_after.as_deref(), Some("10"), "{status}");
        }
    }

    async fn error_for(status: StatusCode, body: &str) -> MuseError {
        post_err(&harness(vec![canned(status, body)]).await).await
    }

    fn synthetic_error(message: &str) -> String {
        serde_json::json!({"error": {"code": "synthetic_code", "message": message, "param": null, "type": "api_error"}}).to_string()
    }

    #[tokio::test]
    async fn observed_404_and_400_are_invalid_requests_with_code_and_param() {
        let error = error_for(StatusCode::NOT_FOUND, OBSERVED_404).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::InvalidRequest,
                StatusCode::NOT_FOUND,
                Some(StatusCode::NOT_FOUND)
            )
        );
        assert_eq!(error.code.as_deref(), Some("model_not_found"));
        assert_eq!(error.param, None);
        assert_eq!(
            error.message_with_status(),
            "The requested model was not found. It may not exist, or this API key may not have access to it. Check the model name; GET /v1/models lists the models this API key can use. (upstream HTTP 404)"
        );

        let error = error_for(StatusCode::BAD_REQUEST, OBSERVED_400).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::InvalidRequest,
                StatusCode::BAD_REQUEST,
                Some(StatusCode::BAD_REQUEST)
            )
        );
        assert_eq!(error.code, None);
        assert_eq!(error.param.as_deref(), Some("reasoning.effort"));
        assert!(
            error
                .message_with_status()
                .ends_with("xhigh, max] (upstream HTTP 400)")
        );
    }

    #[tokio::test]
    async fn observed_405_and_empty_404_keep_their_status() {
        let error = error_for(StatusCode::METHOD_NOT_ALLOWED, OBSERVED_405).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::METHOD_NOT_ALLOWED,
                Some(StatusCode::METHOD_NOT_ALLOWED)
            )
        );
        assert_eq!(error.code.as_deref(), Some("method_not_allowed"));
        assert_eq!(
            error.message_with_status(),
            "Method not allowed. (upstream HTTP 405)"
        );

        let error = error_for(StatusCode::NOT_FOUND, "").await;
        assert_eq!(error.kind, ProviderErrorKind::InvalidRequest);
        assert_eq!(
            error.message_with_status(),
            "Muse API request failed (upstream HTTP 404)"
        );
    }

    #[tokio::test]
    async fn synthetic_402_and_403_are_permission_errors() {
        for status in [StatusCode::PAYMENT_REQUIRED, StatusCode::FORBIDDEN] {
            let error = error_for(status, &synthetic_error("synthetic not subscribed")).await;
            assert_eq!(
                (error.kind, error.status, error.upstream_status),
                (
                    ProviderErrorKind::Permission,
                    StatusCode::FORBIDDEN,
                    Some(status)
                )
            );
            assert_eq!(
                error.message_with_status(),
                format!(
                    "synthetic not subscribed (upstream HTTP {})",
                    status.as_u16()
                )
            );
        }
    }

    #[tokio::test]
    async fn synthetic_403_points_an_unsubscribed_account_at_the_accounts_center() {
        let mut inactive = mint::tests::observed_mint_body();
        inactive["api_key"] = API_KEY.into();
        inactive["is_subs_active"] = false.into();
        let harness = harness_with(
            vec![canned(
                StatusCode::FORBIDDEN,
                synthetic_error("synthetic not subscribed"),
            )],
            vec![canned(StatusCode::OK, inactive.to_string())],
            true,
        )
        .await;
        let error = post_err(&harness).await;
        assert_eq!(
            error.message_with_status(),
            "synthetic not subscribed. This account has no active Muse Code subscription: https://accountscenter.meta.com/muse_code/ (upstream HTTP 403)"
        );
    }

    #[tokio::test]
    async fn synthetic_529_and_other_5xx_map_by_status() {
        let error = error_for(
            StatusCode::from_u16(529).unwrap(),
            &synthetic_error("synthetic overloaded"),
        )
        .await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::SERVICE_UNAVAILABLE,
                Some(StatusCode::from_u16(529).unwrap())
            )
        );
        assert_eq!(
            error.message_with_status(),
            "synthetic overloaded (upstream HTTP 529)"
        );

        let error = error_for(
            StatusCode::BAD_GATEWAY,
            &synthetic_error("synthetic bad gateway"),
        )
        .await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::BAD_GATEWAY,
                Some(StatusCode::BAD_GATEWAY)
            )
        );

        let error = error_for(StatusCode::INTERNAL_SERVER_ERROR, "<html>synthetic</html>").await;
        assert_eq!(
            (error.kind, error.status),
            (ProviderErrorKind::Api, StatusCode::INTERNAL_SERVER_ERROR)
        );
        assert_eq!(
            error.message_with_status(),
            "Muse API request failed (upstream HTTP 500)"
        );
    }

    fn rate_limited(retry_after: Option<&str>) -> Canned {
        let mut answer = canned(
            StatusCode::TOO_MANY_REQUESTS,
            synthetic_error("synthetic rate limit"),
        );
        if let Some(value) = retry_after {
            answer.headers.push(("retry-after", value.to_string()));
        }
        answer
    }

    fn exhausted_window(resets_at: i64) -> SubscriptionUsage {
        SubscriptionUsage {
            window: Some(UsageWindow {
                used_percent: 100,
                resets_at,
                window_duration_mins: Some(300),
            }),
            weekly: Some(UsageWindow {
                used_percent: 3,
                resets_at: resets_at + 86_400,
                window_duration_mins: None,
            }),
        }
    }

    #[tokio::test]
    async fn synthetic_429_retry_after_prefers_the_header_then_fresh_usage() {
        let harness = harness(vec![
            rate_limited(Some("30")),
            rate_limited(None),
            rate_limited(None),
        ])
        .await;
        let key = harness.client.auth().get_key().await.unwrap();

        harness
            .client
            .auth()
            .record_usage(key.login, exhausted_window(now_unix() + 120));
        let error = post_err(&harness).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::RateLimit,
                StatusCode::TOO_MANY_REQUESTS,
                Some(StatusCode::TOO_MANY_REQUESTS)
            )
        );
        assert_eq!(error.retry_after.as_deref(), Some("30"));
        assert_eq!(
            error.message_with_status(),
            "synthetic rate limit (upstream HTTP 429)"
        );

        let error = post_err(&harness).await;
        let derived: u64 = error.retry_after.as_deref().unwrap().parse().unwrap();
        assert!((115..=120).contains(&derived), "{derived}");

        // The observed figures (1 % and 0 %) are not exhausted.
        harness
            .client
            .auth()
            .record_usage(key.login, mint::tests::observed_usage());
        assert_eq!(post_err(&harness).await.retry_after, None);
        assert_eq!(harness.upstream.model_calls(), 3);
    }

    #[tokio::test]
    async fn synthetic_429_ignores_usage_older_than_ten_minutes() {
        let harness = harness(vec![sse_ok(), rate_limited(None), rate_limited(None)]).await;
        // A first request in real time leaves a pooled connection, so the
        // paused clock below never races a connect.
        send(&harness)
            .await
            .unwrap()
            .response
            .bytes()
            .await
            .unwrap();
        let key = harness.client.auth().get_key().await.unwrap();
        let resets_at = now_unix() + 3_600;
        harness
            .client
            .auth()
            .record_usage(key.login, exhausted_window(resets_at));
        tokio::time::pause();
        let error = post_err(&harness).await;
        assert!(error.retry_after.is_some());
        tokio::time::advance(Duration::from_secs(601)).await;
        assert_eq!(post_err(&harness).await.retry_after, None);
    }

    #[tokio::test]
    async fn ninety_seconds_to_the_first_byte_does_not_time_out() {
        let mut slow = sse_ok();
        slow.delay = Some(Duration::from_secs(90));
        let harness = harness(vec![sse_ok(), slow]).await;
        send(&harness)
            .await
            .unwrap()
            .response
            .bytes()
            .await
            .unwrap();
        tokio::time::pause();
        let started = tokio::time::Instant::now();
        let response = send(&harness).await.unwrap();
        assert_eq!(response.response.status(), StatusCode::OK);
        assert!(started.elapsed() >= Duration::from_secs(90));
        let body = response.response.text().await.unwrap();
        assert!(body.contains("response.completed"));
    }

    #[tokio::test]
    async fn mint_failures_map_to_the_table_rows() {
        let harness = harness_with(vec![], vec![], false).await;
        let error = post_err(&harness).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Authentication,
                StatusCode::UNAUTHORIZED,
                None
            )
        );
        assert_eq!(
            error.message_with_status(),
            "Not authenticated with Muse. Run: claude-code-proxy muse auth login"
        );
        assert_eq!(harness.upstream.mint_calls(), 0);

        let harness = harness_with(
            vec![],
            vec![canned(
                StatusCode::UNAUTHORIZED,
                r#"{"title":"Authentication Error","detail":"Authentication Error","status":401}"#,
            )],
            true,
        )
        .await;
        let error = post_err(&harness).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Authentication,
                StatusCode::UNAUTHORIZED,
                Some(StatusCode::UNAUTHORIZED)
            )
        );
        assert_eq!(
            error.message_with_status(),
            "Muse login is no longer valid. Run: claude-code-proxy muse auth login (upstream HTTP 401)"
        );
        // Five more requests reach neither the mint nor the Model API.
        for _ in 0..5 {
            assert_eq!(
                post_err(&harness).await.kind,
                ProviderErrorKind::Authentication
            );
        }
        assert_eq!(harness.upstream.mint_calls(), 1);
        assert_eq!(harness.upstream.model_calls(), 0);
        // A new stored token is tried once.
        save_login(&harness.path, "synthetic-new-login-token");
        send(&harness).await.unwrap();
        assert_eq!(harness.upstream.mint_calls(), 2);

        let harness = harness_with(
            vec![],
            vec![canned(
                StatusCode::BAD_REQUEST,
                r#"{"title":"JSON Schema Validation Error","detail":"Request body is required but was empty or null","status":400}"#,
            )],
            true,
        )
        .await;
        let error = post_err(&harness).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::BAD_GATEWAY,
                Some(StatusCode::BAD_REQUEST)
            )
        );
        assert_eq!(
            error.message_with_status(),
            "Request body is required but was empty or null (upstream HTTP 400)"
        );

        let harness = harness_with(
            vec![],
            vec![canned(StatusCode::INTERNAL_SERVER_ERROR, "{}")],
            true,
        )
        .await;
        let error = post_err(&harness).await;
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::SERVICE_UNAVAILABLE,
                Some(StatusCode::INTERNAL_SERVER_ERROR)
            )
        );
        assert_eq!(
            error.message_with_status(),
            "Muse key service is unavailable (upstream HTTP 500)"
        );
    }

    /// Mint and Model API at addresses chosen by the test.
    async fn split_harness(
        mint_base: String,
        model_base: String,
    ) -> (tempfile::TempDir, MuseResponsesClient) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("auth.json").to_string_lossy().into_owned();
        save_login(&path, ACCESS_TOKEN);
        let minter = Arc::new(HttpMinter::with_client(
            mint::async_client_builder(MINT_TIMEOUT)
                .no_proxy()
                .build()
                .unwrap(),
            MintTarget {
                mint_base_url: mint_base,
                model_base_url: model_base.clone(),
                user_agent: "claude-code-proxy/test".into(),
            },
        ));
        let manager = Arc::new(MuseAuthManager::with_minter(
            store_at(path),
            minter,
            MINT_TIMEOUT,
            CALLER_DEADLINE,
        ));
        let client = MuseResponsesClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            model_base,
            "claude-code-proxy/test".into(),
            manager,
        )
        .unwrap();
        (dir, client)
    }

    fn closed_port() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        base
    }

    #[tokio::test]
    async fn unreachable_services_report_no_upstream_status() {
        let closed = closed_port();
        let (_dir, client) = split_harness(closed.clone(), format!("{closed}/v1")).await;
        let error = client.post(&request_body(), None).await.err().unwrap();
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::SERVICE_UNAVAILABLE,
                None
            )
        );
        assert_eq!(
            error.message_with_status(),
            "Muse key service is unreachable"
        );

        let upstream = spawn_upstream(vec![], vec![]).await;
        let (_dir, client) =
            split_harness(upstream.base.clone(), format!("{}/v1", closed_port())).await;
        let error = client.post(&request_body(), None).await.err().unwrap();
        assert_eq!(
            (error.kind, error.status, error.upstream_status),
            (ProviderErrorKind::Api, StatusCode::BAD_GATEWAY, None)
        );
        assert_eq!(error.message_with_status(), "Muse API is unreachable");
        assert_eq!(
            MuseError::stream_ended().message_with_status(),
            "Muse stream ended before completion (upstream HTTP 200)"
        );
        let failed = MuseError::stream_failed("synthetic stream failure");
        assert_eq!(
            (failed.kind, failed.status, failed.upstream_status),
            (
                ProviderErrorKind::Api,
                StatusCode::BAD_GATEWAY,
                Some(StatusCode::OK)
            )
        );
        assert_eq!(
            failed.message_with_status(),
            "synthetic stream failure (upstream HTTP 200)"
        );
    }

    fn capture_text(dir: &std::path::Path) -> String {
        let mut text = String::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(path) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    text.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
                    text.push('\n');
                }
            }
        }
        text
    }

    #[tokio::test]
    async fn capture_keeps_only_allowlisted_response_headers() {
        let mut answer = sse_ok();
        answer.headers.extend([
            ("x-request-id", "synthetic-request-id".to_string()),
            ("x-route", "model-api-rust".to_string()),
            (
                "proxy-status",
                "edge; e_clientaddr=192.0.2.1; e_proxy=synthetic".to_string(),
            ),
            ("x-fb-debug", "synthetic-debug".to_string()),
        ]);
        let harness = harness(vec![answer]).await;
        let dir = tempfile::TempDir::new().unwrap();
        let capture = crate::traffic::test_capture(dir.path().join("traffic"));
        harness
            .client
            .post(&request_body(), Some(&capture))
            .await
            .unwrap();
        let text = capture_text(dir.path());
        assert!(text.contains("synthetic-request-id"), "{text}");
        assert!(text.contains("model-api-rust"), "{text}");
        assert!(text.contains("text/event-stream"), "{text}");
        assert!(
            !text.to_ascii_lowercase().contains("proxy-status"),
            "{text}"
        );
        assert!(!text.contains("e_clientaddr"), "{text}");
        assert!(!text.contains("synthetic-debug"), "{text}");
        assert!(!text.contains(API_KEY), "{text}");
        assert!(!text.contains(ACCESS_TOKEN), "{text}");
        assert!(text.contains(r#""authorization": "[redacted"#), "{text}");
    }

    fn malicious_body() -> String {
        serde_json::json!({
            "error": {
                "code": API_KEY,
                "message": format!("key {API_KEY} token {ACCESS_TOKEN} owner {EMAIL}"),
                "param": EMAIL,
                "type": "invalid_request_error"
            },
            "detail": format!("detail {ACCESS_TOKEN} {EMAIL}"),
            "title": format!("title {API_KEY}")
        })
        .to_string()
    }

    async fn rendered(error: &MuseError) -> (String, String) {
        let anthropic = anthropic_error_response(error.clone().into_provider_error());
        let anthropic = anthropic.into_body().collect().await.unwrap().to_bytes();
        let openai = OpenAiError::from(error.clone().into_provider_error()).response();
        let openai = openai.into_body().collect().await.unwrap().to_bytes();
        (
            String::from_utf8(anthropic.to_vec()).unwrap(),
            String::from_utf8(openai.to_vec()).unwrap(),
        )
    }

    #[tokio::test]
    async fn synthetic_malicious_upstream_leaks_nothing_downstream_or_into_the_capture() {
        let harness = harness(vec![
            canned(StatusCode::UNAUTHORIZED, malicious_body()),
            canned(StatusCode::BAD_REQUEST, malicious_body()),
        ])
        .await;
        let dir = tempfile::TempDir::new().unwrap();
        let capture = crate::traffic::test_capture(dir.path().join("traffic"));
        let error = harness
            .client
            .post(&request_body(), Some(&capture))
            .await
            .err()
            .unwrap();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, None);
        assert_eq!(error.param, None);
        assert_eq!(
            error.message_with_status(),
            "key [redacted] token [redacted] owner [redacted] (upstream HTTP 400)"
        );
        let (anthropic, openai) = rendered(&error).await;
        let captured = capture_text(dir.path());
        // Both error bodies were captured, scrubbed.
        assert_eq!(
            captured
                .matches("key [redacted] token [redacted] owner [redacted]")
                .count(),
            2,
            "{captured}"
        );
        for (place, text) in [
            ("anthropic", &anthropic),
            ("openai", &openai),
            ("capture", &captured),
        ] {
            for secret in [API_KEY, ACCESS_TOKEN, EMAIL, "@example.com"] {
                assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
            }
        }
        let openai: serde_json::Value = serde_json::from_str(&openai).unwrap();
        assert_eq!(openai["error"]["code"], serde_json::Value::Null);
        assert_eq!(openai["error"]["param"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn synthetic_malicious_mint_detail_is_scrubbed() {
        let body = serde_json::json!({"title": "Bad", "detail": format!("token {ACCESS_TOKEN} for {EMAIL}"), "status": 403});
        let harness = harness_with(
            vec![],
            vec![canned(StatusCode::FORBIDDEN, body.to_string())],
            true,
        )
        .await;
        let dir = tempfile::TempDir::new().unwrap();
        let capture = crate::traffic::test_capture(dir.path().join("traffic"));
        let error = harness
            .client
            .post(&request_body(), Some(&capture))
            .await
            .err()
            .unwrap();
        assert_eq!(
            error.message_with_status(),
            "token [redacted] for [redacted] (upstream HTTP 403)"
        );
        let captured = capture_text(dir.path());
        assert!(!captured.contains(ACCESS_TOKEN), "{captured}");
        assert!(!captured.contains(EMAIL), "{captured}");
        // The mint exchange itself is never captured.
        assert!(!captured.contains("Bad"), "{captured}");
    }

    #[tokio::test]
    async fn synthetic_long_messages_are_cut_after_redaction() {
        let long = "m".repeat(600);
        let error = error_for(StatusCode::BAD_REQUEST, &synthetic_error(&long)).await;
        assert_eq!(error.message.chars().count(), 500);
        assert_eq!(
            error.message_with_status(),
            format!("{} (upstream HTTP 400)", "m".repeat(500))
        );

        let straddling = format!("{}{API_KEY}{}", "s".repeat(490), "t".repeat(100));
        let error = error_for(StatusCode::BAD_REQUEST, &synthetic_error(&straddling)).await;
        assert!(!error.message.contains(&API_KEY[..10]), "{}", error.message);
        assert!(error.message.chars().count() <= 500);

        let multibyte = "\u{3042}".repeat(600);
        let error = error_for(StatusCode::BAD_REQUEST, &synthetic_error(&multibyte)).await;
        assert_eq!(error.message, "\u{3042}".repeat(500));
    }

    #[tokio::test]
    async fn synthetic_malformed_code_and_param_are_dropped() {
        let body = serde_json::json!({"error": {"code": "has space", "message": "synthetic", "param": "semi;colon", "type": "invalid_request_error"}});
        let error = error_for(StatusCode::BAD_REQUEST, &body.to_string()).await;
        assert_eq!((error.code, error.param), (None, None));
    }

    #[tokio::test]
    async fn rendered_errors_carry_status_type_and_retry_after() {
        let harness = harness(vec![rate_limited(Some("30"))]).await;
        let error = post_err(&harness).await;
        let response = anthropic_error_response(error.clone().into_provider_error());
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "30");
        let body: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(
            body["error"]["message"],
            "synthetic rate limit (upstream HTTP 429)"
        );

        let response = OpenAiError::from(error.into_provider_error()).response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "30");
        let body: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["error"]["code"], "synthetic_code");
    }

    #[test]
    fn responses_url_follows_the_base_path() {
        assert_eq!(
            responses_url("https://api.meta.ai/v1").unwrap(),
            "https://api.meta.ai/v1/responses"
        );
        assert_eq!(
            responses_url("http://127.0.0.1:8080/v1/").unwrap(),
            "http://127.0.0.1:8080/v1/responses"
        );
        assert_eq!(
            responses_url("").unwrap(),
            "https://api.meta.ai/v1/responses"
        );
        assert!(responses_url(":invalid").is_err());
    }
}
