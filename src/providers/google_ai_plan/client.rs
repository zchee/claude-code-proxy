//! HTTP client for the Cloud Code PA endpoint of the Google AI Ultra plan:
//! `POST <base URL>/v1internal:streamGenerateContent?alt=sse` with the access
//! token as a Bearer token, asking for an event stream, plus project discovery
//! via `v1internal:loadCodeAssist` and `v1internal:onboardUser`.
//!
//! The wait for the response headers is bounded, as on the Vertex AI Gemini
//! path, and a generation stream is not: a turn can stream for many minutes,
//! and a model may stay silent while it thinks. A successful discovery body
//! has its own read bound of the same length, and an error body a shorter
//! one. A 429 is never retried here. On the default base URL only, a 5xx or
//! a connection failure before any status is sent once more to the fallback
//! host; otherwise Claude Code retries by status.

use std::time::Duration;

use futures_util::{Stream, StreamExt};
use http::{HeaderMap, StatusCode};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::config::{
    GOOGLE_AI_PLAN_DEFAULT_BASE_URL, GOOGLE_AI_PLAN_FALLBACK_BASE_URL, GoogleAiPlanSettings,
};
use crate::providers::vertex::client::{
    ERROR_BODY_TIMEOUT, MAX_ERROR_BODY_BYTES, RESPONSE_HEADER_TIMEOUT, client_builder,
    read_bounded, redact_for_capture, safe_response_headers,
};
use crate::providers::vertex::sanitize::redact_secrets;
use crate::traffic::TrafficCapture;

/// The streaming generation endpoint path after the base URL.
pub const STREAM_GENERATE_CONTENT_PATH: &str = "v1internal:streamGenerateContent?alt=sse";

/// Alias kept for callers referencing the generation endpoint path constant.
pub const INTERACTIONS_PATH: &str = STREAM_GENERATE_CONTENT_PATH;

/// The Code Assist project discovery endpoint path after the base URL.
pub const LOAD_CODE_ASSIST_PATH: &str = "v1internal:loadCodeAssist";

/// The Code Assist user onboarding endpoint path after the base URL.
pub const ONBOARD_USER_PATH: &str = "v1internal:onboardUser";

const EVENT_STREAM: &str = "text/event-stream";

/// Builds a Cloud Code PA `v1internal:<method>` URL from `base_url` and
/// `method_and_query`.
pub fn endpoint_url(base_url: &str, method_and_query: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let method_and_query = method_and_query
        .strip_prefix("v1internal:")
        .unwrap_or(method_and_query);
    if base.ends_with("/v1internal") {
        format!("{base}:{method_and_query}")
    } else {
        format!("{base}/v1internal:{method_and_query}")
    }
}

/// The Cloud Code PA platform enum string for the current host OS and CPU
/// architecture, matching `agy`'s `ClientMetadata.Platform`.
pub fn platform_enum() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "DARWIN_ARM64",
        ("macos", "x86_64") => "DARWIN_AMD64",
        ("linux", "aarch64") => "LINUX_ARM64",
        ("linux", "x86_64") => "LINUX_AMD64",
        ("windows", "x86_64") => "WINDOWS_AMD64",
        _ => "PLATFORM_UNSPECIFIED",
    }
}

/// Why a model call produced no stream to read.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// The base URL and the endpoint path do not form a URL.
    #[error("Google AI Ultra plan base URL does not form a valid endpoint URL")]
    InvalidUrl,
    /// The request body could not be serialised.
    #[error("Google AI Ultra plan request could not be encoded")]
    Encode,
    /// No connection, or the connection failed before a status arrived.
    #[error("Google AI Ultra plan is unreachable")]
    Unreachable,
    /// No response headers within the timeout.
    #[error("Google AI Ultra plan sent no response headers within {} seconds", .0.as_secs())]
    HeaderTimeout(Duration),
    /// A non-2xx answer. `body` holds at most [`MAX_ERROR_BODY_BYTES`] of the
    /// error body, already without the request's access token; it is empty
    /// when the body was longer, broke off or came too late.
    #[error("Google AI Ultra plan answered HTTP {}", .status.as_u16())]
    Status {
        /// The upstream status.
        status: StatusCode,
        /// The upstream response headers.
        headers: HeaderMap,
        /// The error body, as described above.
        body: String,
    },
    /// A 2xx project-discovery response that cannot be used: a truncated or
    /// unparsable body, or an `onboardUser` result with no usable project.
    /// It is never a model call, so a generation stream is never started with
    /// an empty project.
    #[error("Google AI Ultra plan project discovery failed: {0}")]
    Discovery(&'static str),
}

/// Sends model calls and project-discovery calls to Cloud Code PA. Build it on
/// the first request: building loads the system root certificates.
pub struct InteractionsClient {
    client: reqwest::Client,
    base_url: String,
    fallback_base_url: Option<String>,
    url: String,
    user_agent: String,
    response_header_timeout: Duration,
    error_body_timeout: Duration,
    /// The discovered project ID, bound to the credential it was discovered
    /// for (`(credential_key, project_id)`), so switching accounts on a
    /// running client never returns one account's project to another.
    project_id: Mutex<Option<(String, String)>>,
}

fn should_fallback(error: &SendError) -> bool {
    match error {
        SendError::Unreachable => true,
        SendError::Status { status, .. } => status.is_server_error(),
        SendError::InvalidUrl
        | SendError::Encode
        | SendError::HeaderTimeout(_)
        | SendError::Discovery(_) => false,
    }
}

impl InteractionsClient {
    /// A client for `settings`, forcing HTTP/1.1 to match `agy`'s
    /// `newCloudCodeTransport` (`stripHTTP2NextProto`), with no redirects and
    /// the Vertex AI client's connect timeout.
    ///
    /// # Errors
    ///
    /// Fails when the TLS configuration cannot be built.
    pub fn new(settings: &GoogleAiPlanSettings) -> anyhow::Result<Self> {
        Ok(Self::with_client(
            client_builder().http1_only().build()?,
            settings,
        ))
    }

    /// A client around `client`, for tests against a local upstream (point
    /// the settings' base URL at it).
    pub fn with_client(client: reqwest::Client, settings: &GoogleAiPlanSettings) -> Self {
        let fallback_base_url = (settings.base_url == GOOGLE_AI_PLAN_DEFAULT_BASE_URL)
            .then(|| GOOGLE_AI_PLAN_FALLBACK_BASE_URL.to_string());
        Self {
            client,
            base_url: settings.base_url.clone(),
            fallback_base_url,
            url: endpoint_url(&settings.base_url, "streamGenerateContent?alt=sse"),
            user_agent: settings.user_agent.clone(),
            response_header_timeout: RESPONSE_HEADER_TIMEOUT,
            error_body_timeout: ERROR_BODY_TIMEOUT,
            project_id: Mutex::new(None),
        }
    }

    /// The same client with an explicit fallback base URL.
    pub fn with_fallback_base_url(mut self, fallback_base_url: Option<String>) -> Self {
        self.fallback_base_url = fallback_base_url;
        self
    }

    /// The same client with another wait for the response headers.
    pub fn with_response_header_timeout(mut self, timeout: Duration) -> Self {
        self.response_header_timeout = timeout;
        self
    }

    /// The streaming endpoint URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the Cloud AI Companion project ID (`cloudaicompanionProject`),
    /// using `stored_project` or the in-memory cache for `credential_key` when
    /// available, or discovering it via `v1internal:loadCodeAssist` (and
    /// `v1internal:onboardUser` when needed).
    ///
    /// The in-memory cache is keyed by `credential_key`, so switching accounts
    /// on a running client never returns one account's project to another. A
    /// discovery response that carries no usable project is a
    /// [`SendError::Discovery`] rather than an empty project, so a generation
    /// stream is never started without one.
    ///
    /// # Errors
    ///
    /// See [`SendError`].
    pub async fn ensure_project(
        &self,
        access_token: &str,
        credential_key: &str,
        stored_project: Option<&str>,
    ) -> Result<String, SendError> {
        // A stored ID is replayed into every generation request, so one that
        // fails the check a discovered ID passes (an older profile, a hand
        // edit) is treated as absent and discovered again.
        if let Some(project) = stored_project
            .map(str::trim)
            .filter(|project| is_valid_identifier(project))
        {
            return Ok(project.to_string());
        }
        let mut guard = self.project_id.lock().await;
        if let Some((key, project)) = guard.as_ref()
            && key == credential_key
            && !project.is_empty()
        {
            return Ok(project.clone());
        }

        let metadata = json!({
            "ideType": "ANTIGRAVITY",
            "platform": platform_enum(),
            "pluginType": "CLOUD_CODE",
        });
        let load_body = json!({
            "metadata": metadata,
            "mode": "FULL_ELIGIBILITY_CHECK",
        });
        let load_response = self
            .post_method_json("loadCodeAssist", &load_body, access_token)
            .await?;
        // Only an object is an eligibility answer: `null`, an array or a bare
        // value is a malformed body, not an account that has no project yet.
        if !load_response.is_object() {
            return Err(SendError::Discovery(
                "the eligibility response was not a JSON object",
            ));
        }
        let discovered = if let Some(project) = extract_project_id(&load_response) {
            project
        } else {
            if load_response
                .get("error")
                .is_some_and(|error| !error.is_null())
            {
                return Err(SendError::Discovery(
                    "the eligibility check reported an error",
                ));
            }
            let tier_id = extract_tier_id(&load_response);
            if !is_valid_identifier(&tier_id) {
                return Err(SendError::Discovery(
                    "the eligibility response carried no usable tier",
                ));
            }
            let onboard_body = json!({
                "tierId": tier_id,
                "metadata": metadata,
            });
            let onboard_response = self
                .post_method_json("onboardUser", &onboard_body, access_token)
                .await?;
            if onboard_response
                .get("error")
                .is_some_and(|error| !error.is_null())
            {
                return Err(SendError::Discovery("account onboarding reported an error"));
            }
            match onboard_response
                .get("response")
                .and_then(extract_project_id)
                .or_else(|| extract_project_id(&onboard_response))
            {
                Some(project) => project,
                None if onboard_response.get("done").and_then(Value::as_bool) == Some(false) => {
                    return Err(SendError::Discovery(
                        "account onboarding is still in progress; retry shortly",
                    ));
                }
                None => {
                    return Err(SendError::Discovery(
                        "account onboarding returned no project",
                    ));
                }
            }
        };

        if !is_valid_identifier(&discovered) {
            return Err(SendError::Discovery(
                "the discovered project id was malformed",
            ));
        }
        *guard = Some((credential_key.to_string(), discovered.clone()));
        Ok(discovered)
    }

    async fn post_method_json(
        &self,
        method_and_query: &str,
        body: &Value,
        access_token: &str,
    ) -> Result<Value, SendError> {
        let primary_url = endpoint_url(&self.base_url, method_and_query);
        match self.post_json(&primary_url, body, access_token).await {
            Err(err) if should_fallback(&err) => {
                if let Some(fallback_base) = self.fallback_base_url.as_deref() {
                    let fallback_url = endpoint_url(fallback_base, method_and_query);
                    return self.post_json(&fallback_url, body, access_token).await;
                }
                Err(err)
            }
            res => res,
        }
    }

    async fn post_json(
        &self,
        url_str: &str,
        body: &Value,
        access_token: &str,
    ) -> Result<Value, SendError> {
        let url = reqwest::Url::parse(url_str).map_err(|_| SendError::InvalidUrl)?;
        let bytes = serde_json::to_vec(body).map_err(|_| SendError::Encode)?;
        let request = self
            .client
            .post(url)
            .bearer_auth(access_token)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::USER_AGENT, &self.user_agent)
            .body(bytes);
        let response =
            match tokio::time::timeout(self.response_header_timeout, request.send()).await {
                Err(_) => return Err(SendError::HeaderTimeout(self.response_header_timeout)),
                Ok(Err(_)) => return Err(SendError::Unreachable),
                Ok(Ok(response)) => response,
            };
        let status = response.status();
        if status.is_success() {
            let (body, cut) = tokio::time::timeout(
                self.response_header_timeout,
                read_bounded(response, MAX_ERROR_BODY_BYTES),
            )
            .await
            .map_err(|_| SendError::HeaderTimeout(self.response_header_timeout))?;
            // A discovery body is a few kilobytes; a truncated or unparsable
            // one must fail closed, never fall back to a null that would be
            // read as "no project" and start onboarding or an empty stream.
            // The shared reader reports a body that fills the limit as cut,
            // so a body of 64 KiB (`MAX_ERROR_BODY_BYTES`) or more is rejected.
            if cut {
                return Err(SendError::Discovery(
                    "the discovery response was larger than the read limit",
                ));
            }
            let value = serde_json::from_slice(&body)
                .map_err(|_| SendError::Discovery("the discovery response was not valid JSON"))?;
            return Ok(value);
        }
        let headers = response.headers().clone();
        let body = tokio::time::timeout(
            self.error_body_timeout,
            read_bounded(response, MAX_ERROR_BODY_BYTES),
        )
        .await
        .ok()
        .and_then(|(body, cut)| (!cut).then_some(body))
        .map(|body| redact_secrets(&String::from_utf8_lossy(&body), &[access_token]))
        .unwrap_or_default();
        Err(SendError::Status {
            status,
            headers,
            body,
        })
    }

    /// POSTs `body` with `access_token` and returns the 2xx response, body
    /// unread. A capture records the request body, the request metadata with
    /// the `authorization` header redacted, and the response status and safe
    /// headers; the access token is redacted from every string of them.
    ///
    /// # Errors
    ///
    /// See [`SendError`].
    pub async fn send<B: Serialize + ?Sized>(
        &self,
        body: &B,
        access_token: &str,
        traffic: Option<&TrafficCapture>,
    ) -> Result<reqwest::Response, SendError> {
        let bytes = serde_json::to_vec(body).map_err(|_| SendError::Encode)?;
        match self
            .send_to_url(&self.url, &bytes, access_token, traffic)
            .await
        {
            Err(err) if should_fallback(&err) => {
                if let Some(fallback_base) = self.fallback_base_url.as_deref() {
                    let fallback_url = endpoint_url(fallback_base, "streamGenerateContent?alt=sse");
                    return self
                        .send_to_url(&fallback_url, &bytes, access_token, traffic)
                        .await;
                }
                Err(err)
            }
            res => res,
        }
    }

    async fn send_to_url(
        &self,
        url_str: &str,
        bytes: &[u8],
        access_token: &str,
        traffic: Option<&TrafficCapture>,
    ) -> Result<reqwest::Response, SendError> {
        let url = reqwest::Url::parse(url_str).map_err(|_| SendError::InvalidUrl)?;
        if let Some(capture) = traffic {
            let value = serde_json::from_slice(bytes).unwrap_or(serde_json::Value::Null);
            capture.write_json(
                "020-upstream-request",
                &redact_for_capture(&value, Some(access_token)),
            );
            capture.write_json(
                "021-upstream-request-metadata",
                &redact_for_capture(
                    &json!({
                        "method": "POST",
                        "url": url.as_str(),
                        "provider": "google-ai-plan",
                        "transport": "http",
                        "headers": {
                            "authorization": "[redacted]",
                            "accept": EVENT_STREAM,
                            "content-type": "application/json",
                            "user-agent": self.user_agent,
                        },
                        "body_bytes": bytes.len(),
                    }),
                    Some(access_token),
                ),
            );
        }

        let request = self
            .client
            .post(url)
            .bearer_auth(access_token)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::ACCEPT, EVENT_STREAM)
            .header(http::header::USER_AGENT, &self.user_agent)
            .body(bytes.to_vec());
        let response =
            match tokio::time::timeout(self.response_header_timeout, request.send()).await {
                Err(_) => return Err(SendError::HeaderTimeout(self.response_header_timeout)),
                Ok(Err(_)) => return Err(SendError::Unreachable),
                Ok(Ok(response)) => response,
            };

        let status = response.status();
        if let Some(capture) = traffic {
            capture.write_json(
                "030-upstream-response-headers",
                &redact_for_capture(
                    &json!({
                        "status": status.as_u16(),
                        "headers": safe_response_headers(response.headers()),
                    }),
                    Some(access_token),
                ),
            );
        }
        if status.is_success() {
            return Ok(response);
        }
        let headers = response.headers().clone();
        let body = tokio::time::timeout(
            self.error_body_timeout,
            read_bounded(response, MAX_ERROR_BODY_BYTES),
        )
        .await
        .ok()
        .and_then(|(body, cut)| (!cut).then_some(body))
        .map(|body| redact_secrets(&String::from_utf8_lossy(&body), &[access_token]))
        .unwrap_or_default();
        Err(SendError::Status {
            status,
            headers,
            body,
        })
    }
}

fn extract_project_id(value: &Value) -> Option<String> {
    match value.get("cloudaicompanionProject") {
        Some(Value::String(project)) => {
            let trimmed = project.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Some(Value::Object(object)) => object
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

fn extract_tier_id(response: &Value) -> String {
    let non_empty = |value: Option<&Value>| {
        value
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
    };
    if let Some(id) = non_empty(response.pointer("/paidTier/id")) {
        return id;
    }
    if let Some(id) = non_empty(response.pointer("/currentTier/id")) {
        return id;
    }
    if let Some(tiers) = response.get("allowedTiers").and_then(Value::as_array) {
        for tier in tiers {
            if tier.get("isDefault").and_then(Value::as_bool) == Some(true)
                && let Some(id) = non_empty(tier.get("id"))
            {
                return id;
            }
        }
    }
    "free-tier".to_string()
}

/// Whether a discovered project or tier ID is safe to cache, persist and
/// replay: non-empty after trimming, at most 256 bytes, and only the bytes a
/// Google resource ID uses, so a control character, whitespace or an oversized
/// value never reaches a later request.
fn is_valid_identifier(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && value.len() <= 256
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
}

/// The body of a 2xx response as it arrives. The error is a broken
/// connection.
pub fn byte_stream(
    response: reqwest::Response,
) -> impl Stream<Item = Result<bytes::Bytes, ()>> + Send + 'static {
    response.bytes_stream().map(|chunk| chunk.map_err(drop))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_url_handles_root_and_v1internal_base_urls() {
        assert_eq!(
            endpoint_url(
                "https://cloudcode-pa.googleapis.com",
                "streamGenerateContent?alt=sse"
            ),
            "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            endpoint_url("https://cloudcode-pa.googleapis.com/", "loadCodeAssist"),
            "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
        );
        assert_eq!(
            endpoint_url(
                "https://cloudcode-pa.googleapis.com/v1internal",
                "onboardUser"
            ),
            "https://cloudcode-pa.googleapis.com/v1internal:onboardUser"
        );
        assert_eq!(
            endpoint_url(
                "https://cloudcode-pa.googleapis.com/v1internal/",
                STREAM_GENERATE_CONTENT_PATH
            ),
            "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn project_and_tier_extraction_handle_string_and_object_shapes() {
        assert_eq!(
            extract_project_id(&json!({"cloudaicompanionProject": "proj-str"})),
            Some("proj-str".to_string())
        );
        assert_eq!(
            extract_project_id(&json!({"cloudaicompanionProject": {"id": "proj-obj"}})),
            Some("proj-obj".to_string())
        );
        assert_eq!(
            extract_project_id(&json!({"cloudaicompanionProject": ""})),
            None
        );
        assert_eq!(
            extract_tier_id(&json!({"paidTier": {"id": "g1-ultra-tier"}})),
            "g1-ultra-tier"
        );
        assert_eq!(
            extract_tier_id(&json!({"currentTier": {"id": "standard-tier"}})),
            "standard-tier"
        );
        assert_eq!(
            extract_tier_id(&json!({
                "allowedTiers": [
                    {"id": "other", "isDefault": false},
                    {"id": "default-tier", "isDefault": true}
                ]
            })),
            "default-tier"
        );
        assert_eq!(extract_tier_id(&json!({})), "free-tier");
    }

    #[test]
    fn default_base_url_enables_fallback_to_cloudcode_pa() {
        let default_settings = GoogleAiPlanSettings {
            base_url: GOOGLE_AI_PLAN_DEFAULT_BASE_URL.to_string(),
            client: crate::config::GoogleAiPlanClient::built_in(),
            user_agent: "antigravity/1.104.0".to_string(),
        };
        let client = InteractionsClient::with_client(reqwest::Client::new(), &default_settings);
        assert_eq!(
            client.fallback_base_url.as_deref(),
            Some(GOOGLE_AI_PLAN_FALLBACK_BASE_URL)
        );

        let custom_settings = GoogleAiPlanSettings {
            base_url: "https://custom-cloudcode.example".to_string(),
            client: crate::config::GoogleAiPlanClient::built_in(),
            user_agent: "antigravity/1.104.0".to_string(),
        };
        let custom_client =
            InteractionsClient::with_client(reqwest::Client::new(), &custom_settings);
        assert_eq!(custom_client.fallback_base_url, None);
    }

    #[tokio::test]
    async fn ensure_project_and_send_retry_fallback_on_unreachable_or_5xx() {
        use axum::{Router, body::Body, response::Response, routing::post};

        let fallback_app = Router::new().route(
            "/{*path}",
            post(|uri: http::Uri| async move {
                let body = if uri.path().ends_with(":loadCodeAssist") {
                    json!({"cloudaicompanionProject": "fallback-project"}).to_string()
                } else {
                    "data: [DONE]\n\n".to_string()
                };
                Response::builder()
                    .status(StatusCode::OK)
                    .header(http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap()
            }),
        );
        let fallback_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let fallback_addr = fallback_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(fallback_listener, fallback_app).await.unwrap();
        });
        let fallback_base = format!("http://{fallback_addr}");

        // 1. Primary returns 503 Service Unavailable -> falls back to fallback_base.
        let primary_503_app = Router::new().route(
            "/{*path}",
            post(|| async move {
                Response::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .body(Body::from("upstream 503"))
                    .unwrap()
            }),
        );
        let primary_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let primary_addr = primary_listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(primary_listener, primary_503_app)
                .await
                .unwrap();
        });
        let settings_503 = GoogleAiPlanSettings {
            base_url: format!("http://{primary_addr}"),
            client: crate::config::GoogleAiPlanClient::built_in(),
            user_agent: "antigravity/1.104.0".to_string(),
        };
        let client_503 = InteractionsClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            &settings_503,
        )
        .with_fallback_base_url(Some(fallback_base.clone()));

        let project = client_503
            .ensure_project("synthetic-token", "key-a", None)
            .await
            .expect("falls back on 503 in ensure_project");
        assert_eq!(project, "fallback-project");

        let response = client_503
            .send(&json!({"test": true}), "synthetic-token", None)
            .await
            .expect("falls back on 503 in send");
        assert_eq!(response.status(), StatusCode::OK);

        // 2. Primary is unreachable (closed local port) -> falls back to fallback_base.
        let dead_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_addr = dead_listener.local_addr().unwrap();
        drop(dead_listener);
        let settings_unreachable = GoogleAiPlanSettings {
            base_url: format!("http://{dead_addr}"),
            client: crate::config::GoogleAiPlanClient::built_in(),
            user_agent: "antigravity/1.104.0".to_string(),
        };
        let client_unreachable = InteractionsClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            &settings_unreachable,
        )
        .with_fallback_base_url(Some(fallback_base));

        let project = client_unreachable
            .ensure_project("synthetic-token", "key-a", None)
            .await
            .expect("falls back on Unreachable in ensure_project");
        assert_eq!(project, "fallback-project");

        let response = client_unreachable
            .send(&json!({"test": true}), "synthetic-token", None)
            .await
            .expect("falls back on Unreachable in send");
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// What one discovery upstream call received and how it was answered.
    struct DiscoveryState {
        responses: Vec<(StatusCode, String)>,
        seen: Vec<String>,
    }

    /// A local discovery upstream that answers each call with the next
    /// `responses` entry (the last one from then on) and records the method
    /// path of every call. Returns its base URL and the shared state.
    async fn discovery_upstream(
        responses: Vec<(StatusCode, String)>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<DiscoveryState>>) {
        use axum::{Router, body::Body, response::Response, routing::post};

        let state = std::sync::Arc::new(std::sync::Mutex::new(DiscoveryState {
            responses,
            seen: Vec::new(),
        }));
        let handler_state = state.clone();
        let app = Router::new().route(
            "/{*path}",
            post(move |uri: http::Uri| {
                let handler_state = handler_state.clone();
                async move {
                    let mut state = handler_state.lock().unwrap();
                    state.seen.push(uri.path().to_string());
                    let index = (state.seen.len() - 1).min(state.responses.len() - 1);
                    let (status, body) = state.responses[index].clone();
                    Response::builder()
                        .status(status)
                        .header(http::header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), state)
    }

    fn discovery_client(base_url: &str) -> InteractionsClient {
        let settings = GoogleAiPlanSettings {
            base_url: base_url.to_string(),
            client: crate::config::GoogleAiPlanClient::built_in(),
            user_agent: "antigravity/1.104.0".to_string(),
        };
        InteractionsClient::with_client(client_builder().no_proxy().build().unwrap(), &settings)
    }

    #[tokio::test]
    async fn a_cached_project_does_not_cross_credentials() {
        let (base, state) = discovery_upstream(vec![
            (
                StatusCode::OK,
                json!({"cloudaicompanionProject": "project-a"}).to_string(),
            ),
            (
                StatusCode::OK,
                json!({"cloudaicompanionProject": "project-b"}).to_string(),
            ),
        ])
        .await;
        let client = discovery_client(&base);

        let first = client.ensure_project("token", "key-a", None).await.unwrap();
        assert_eq!(first, "project-a");
        // A different credential must not receive the first one's project.
        let second = client.ensure_project("token", "key-b", None).await.unwrap();
        assert_eq!(second, "project-b");

        let seen = state.lock().unwrap().seen.clone();
        assert_eq!(seen.len(), 2, "each credential discovers its own project");
        assert!(seen.iter().all(|path| path.ends_with(":loadCodeAssist")));
    }

    #[tokio::test]
    async fn a_truncated_discovery_body_fails_without_onboarding() {
        let big = "x".repeat(MAX_ERROR_BODY_BYTES + 1);
        let (base, state) = discovery_upstream(vec![(
            StatusCode::OK,
            json!({"cloudaicompanionProject": big}).to_string(),
        )])
        .await;
        let client = discovery_client(&base);

        let error = client
            .ensure_project("token", "key", None)
            .await
            .expect_err("a truncated discovery body is rejected");
        assert!(matches!(error, SendError::Discovery(_)));
        let seen = state.lock().unwrap().seen.clone();
        assert_eq!(seen.len(), 1, "onboarding is never called: {seen:?}");
        assert!(seen[0].ends_with(":loadCodeAssist"), "{seen:?}");
    }

    #[tokio::test]
    async fn an_unfinished_onboarding_fails_without_a_stream() {
        let (base, state) = discovery_upstream(vec![
            (
                StatusCode::OK,
                json!({"currentTier": {"id": "ultra"}}).to_string(),
            ),
            (StatusCode::OK, json!({"done": false}).to_string()),
        ])
        .await;
        let client = discovery_client(&base);

        let error = client
            .ensure_project("token", "key", None)
            .await
            .expect_err("an unfinished onboarding is rejected");
        assert!(matches!(error, SendError::Discovery(_)));
        let seen = state.lock().unwrap().seen.clone();
        assert_eq!(seen.len(), 2, "a stream is never started: {seen:?}");
        assert!(seen[0].ends_with(":loadCodeAssist"), "{seen:?}");
        assert!(seen[1].ends_with(":onboardUser"), "{seen:?}");
    }

    #[tokio::test]
    async fn a_malformed_project_id_is_rejected() {
        let (base, _) = discovery_upstream(vec![(
            StatusCode::OK,
            json!({"cloudaicompanionProject": "has a space"}).to_string(),
        )])
        .await;
        let client = discovery_client(&base);
        let error = client
            .ensure_project("token", "key", None)
            .await
            .expect_err("a project id with whitespace is rejected");
        assert!(matches!(error, SendError::Discovery(_)));
    }

    #[tokio::test]
    async fn an_invalid_stored_project_id_is_rediscovered() {
        let oversized = "a".repeat(257);
        let tests = [
            ("whitespace inside", "has a space"),
            ("a control character", "project\u{7}"),
            ("oversized", oversized.as_str()),
        ];
        for (name, stored) in tests {
            let (base, state) = discovery_upstream(vec![(
                StatusCode::OK,
                json!({"cloudaicompanionProject": "project-a"}).to_string(),
            )])
            .await;
            let client = discovery_client(&base);

            let project = client
                .ensure_project("token", "key", Some(stored))
                .await
                .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            assert_eq!(project, "project-a", "{name}: the stored id is not used");
            let seen = state.lock().unwrap().seen.clone();
            assert_eq!(seen.len(), 1, "{name}: {seen:?}");
            assert!(seen[0].ends_with(":loadCodeAssist"), "{name}: {seen:?}");
        }

        // A valid stored id is still used without a discovery call.
        let (base, state) = discovery_upstream(vec![(StatusCode::OK, "{}".to_string())]).await;
        let client = discovery_client(&base);
        assert_eq!(
            client
                .ensure_project("token", "key", Some(" projects/stored-1 "))
                .await
                .unwrap(),
            "projects/stored-1"
        );
        assert!(state.lock().unwrap().seen.is_empty());
    }

    #[tokio::test]
    async fn an_eligibility_body_that_is_not_an_answer_fails_without_onboarding() {
        let tests = [
            ("null", "null".to_string()),
            (
                "an array",
                json!([{"cloudaicompanionProject": "p"}]).to_string(),
            ),
            ("a string", json!("project-a").to_string()),
            (
                "an error only",
                json!({"error": {"code": 403, "message": "denied"}}).to_string(),
            ),
        ];
        for (name, body) in tests {
            let (base, state) = discovery_upstream(vec![
                (StatusCode::OK, body),
                (
                    StatusCode::OK,
                    json!({"response": {"cloudaicompanionProject": "onboarded"}}).to_string(),
                ),
            ])
            .await;
            let client = discovery_client(&base);

            let result = client.ensure_project("token", "key", None).await;
            println!("{name}: {result:?}");
            assert!(
                matches!(result, Err(SendError::Discovery(_))),
                "{name}: {result:?}"
            );
            let seen = state.lock().unwrap().seen.clone();
            assert_eq!(
                seen.len(),
                1,
                "{name}: onboarding is never called: {seen:?}"
            );
            assert!(seen[0].ends_with(":loadCodeAssist"), "{name}: {seen:?}");
        }

        // An object without a project or an error still onboards.
        let (base, state) = discovery_upstream(vec![
            (StatusCode::OK, "{}".to_string()),
            (
                StatusCode::OK,
                json!({"response": {"cloudaicompanionProject": "onboarded"}}).to_string(),
            ),
        ])
        .await;
        let client = discovery_client(&base);
        assert_eq!(
            client.ensure_project("token", "key", None).await.unwrap(),
            "onboarded"
        );
        assert_eq!(state.lock().unwrap().seen.len(), 2);
    }

    #[test]
    fn identifier_validation_bounds_length_and_charset() {
        assert!(is_valid_identifier("projects/my-project_1.2:3"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("has space"));
        assert!(!is_valid_identifier("line\nbreak"));
        assert!(!is_valid_identifier(&"a".repeat(257)));
        assert!(is_valid_identifier(&"a".repeat(256)));
    }
}
