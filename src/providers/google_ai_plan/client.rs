//! HTTP client for the Interactions endpoint of the Google AI Ultra plan:
//! `POST <base URL>/interactions:createPerUserQuota` with the access token as
//! a Bearer token, asking for an event stream.
//!
//! Only the wait for the response headers is bounded, as on the Vertex AI
//! Gemini path: a turn can stream for many minutes, and a model may stay
//! silent while it thinks. 429 and 5xx are not retried here; Claude Code
//! retries by status.

use std::time::Duration;

use futures_util::{Stream, StreamExt};
use http::{HeaderMap, StatusCode};
use serde::Serialize;
use serde_json::json;

use crate::config::GoogleAiPlanSettings;
use crate::providers::vertex::client::{
    ERROR_BODY_TIMEOUT, MAX_ERROR_BODY_BYTES, RESPONSE_HEADER_TIMEOUT, client_builder,
    read_bounded, redact_for_capture, safe_response_headers,
};
use crate::providers::vertex::sanitize::redact_secrets;
use crate::traffic::TrafficCapture;

/// The endpoint path after the base URL.
pub const INTERACTIONS_PATH: &str = "interactions:createPerUserQuota";

const EVENT_STREAM: &str = "text/event-stream";

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
}

/// Sends model calls to the Interactions endpoint. Build it on the first
/// request: building loads the system root certificates.
pub struct InteractionsClient {
    client: reqwest::Client,
    url: String,
    user_agent: String,
    response_header_timeout: Duration,
    error_body_timeout: Duration,
}

impl InteractionsClient {
    /// A client for `settings`, with no redirects and the Vertex AI client's
    /// connect timeout.
    ///
    /// # Errors
    ///
    /// Fails when the TLS configuration cannot be built.
    pub fn new(settings: &GoogleAiPlanSettings) -> anyhow::Result<Self> {
        Ok(Self::with_client(client_builder().build()?, settings))
    }

    /// A client around `client`, for tests against a local upstream (point
    /// the settings' base URL at it).
    pub fn with_client(client: reqwest::Client, settings: &GoogleAiPlanSettings) -> Self {
        Self {
            client,
            url: format!("{}/{INTERACTIONS_PATH}", settings.base_url),
            user_agent: settings.user_agent.clone(),
            response_header_timeout: RESPONSE_HEADER_TIMEOUT,
            error_body_timeout: ERROR_BODY_TIMEOUT,
        }
    }

    /// The same client with another wait for the response headers.
    pub fn with_response_header_timeout(mut self, timeout: Duration) -> Self {
        self.response_header_timeout = timeout;
        self
    }

    /// The endpoint URL.
    pub fn url(&self) -> &str {
        &self.url
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
        let url = reqwest::Url::parse(&self.url).map_err(|_| SendError::InvalidUrl)?;
        let bytes = serde_json::to_vec(body).map_err(|_| SendError::Encode)?;
        if let Some(capture) = traffic {
            let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
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
            .body(bytes);
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

/// The body of a 2xx response as it arrives. The error is a broken
/// connection.
pub fn byte_stream(
    response: reqwest::Response,
) -> impl Stream<Item = Result<bytes::Bytes, ()>> + Send + 'static {
    response.bytes_stream().map(|chunk| chunk.map_err(drop))
}
