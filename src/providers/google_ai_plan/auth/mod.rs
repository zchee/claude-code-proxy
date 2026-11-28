//! Google sign-in for the Google AI Ultra plan: Google OAuth for a
//! `Desktop app` client (the built-in one in [`client`], or one from the
//! configuration) with a loopback redirect, PKCE and DPoP (RFC 9449).
//!
//! Google binds the refresh token to a P-256 key pair created on this machine
//! ([`dpop`]); every call to the token endpoint carries a proof signed with
//! that key. Model calls use the access token as a plain Bearer token, without
//! DPoP. [`login`] runs the browser sign-in, [`manager`] hands out access
//! tokens and refreshes them, and [`token_store`] keeps the credential.

pub mod client;
pub mod dpop;
pub mod login;
pub mod manager;
pub mod token_store;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::config::GoogleAiPlanClient;

/// Google's OAuth authorization endpoint.
pub const AUTHORIZE_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";

/// Google's OAuth token endpoint, for the code exchange and refreshes.
pub const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// The scope that grants the Google AI Ultra plan's per-user quota. A sign-in
/// that does not grant it cannot call the plan's models.
pub const QUOTA_SCOPE: &str = "https://www.googleapis.com/auth/generative-language.peruserquota";

/// Every scope the sign-in asks for.
pub const SCOPES: &str =
    "openid email profile https://www.googleapis.com/auth/generative-language.peruserquota";

/// The command that signs in, named in every message that asks the user to
/// sign in (again).
pub const LOGIN_COMMAND: &str = "claude-code-proxy google-ai-plan auth login";

const TOKEN_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How much of an error response that is not JSON at all (an HTML or text
/// page from an intermediary) goes into an error message.
const MAX_UNPARSED_BODY_CHARS: usize = 512;
/// The error text for a token response that is not a JSON object and whose
/// text may hold tokens, so none of it is shown.
const NOT_AN_OBJECT: &str = "The token response was not a JSON object.";

/// The fields of a token endpoint response this flow reads.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct TokenResponse {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_in: Option<u64>,
    pub scope: Option<String>,
    pub id_token: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
}

/// One token endpoint response: its status, body, the `DPoP-Nonce` header and
/// Google's clock from the `Date` header.
#[derive(Debug, Clone)]
pub(crate) struct TokenReply {
    pub status: u16,
    pub body: TokenResponse,
    pub nonce: Option<String>,
    /// Google's clock in Unix seconds; used when the local clock is off.
    pub server_time: Option<i64>,
}

impl TokenReply {
    /// Reads a response. A body that is not a JSON object becomes an
    /// `HTTP <status>` error, which is printed in the terminal and shown on
    /// the browser page.
    ///
    /// The start of the text goes into that error only when the status is
    /// not a success and the text is not JSON at all: an HTML or text page
    /// from a proxy or load balancer, which tells the user what went wrong.
    /// Any other body gets a fixed sentence instead. A success body is where
    /// Google puts the tokens, so a cut-short or otherwise malformed one may
    /// still hold them, and so may JSON that is not an object (a string, or
    /// an array around the token object) whatever the status.
    pub(crate) fn from_parts(status: u16, headers: &http::HeaderMap, text: &str) -> Self {
        let body = match serde_json::from_str::<serde_json::Value>(text) {
            Ok(serde_json::Value::Object(fields)) => {
                let string = |name: &str| {
                    fields
                        .get(name)
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                };
                let expires_in = fields.get("expires_in").and_then(|value| {
                    value
                        .as_u64()
                        .or_else(|| value.as_f64().filter(|v| *v >= 0.0).map(|v| v as u64))
                });
                TokenResponse {
                    access_token: string("access_token"),
                    refresh_token: string("refresh_token"),
                    expires_in,
                    scope: string("scope"),
                    id_token: string("id_token"),
                    error: string("error"),
                    error_description: string("error_description"),
                }
            }
            Err(_) if !(200..300).contains(&status) => TokenResponse {
                error: Some(format!("HTTP {status}")),
                error_description: Some(text.chars().take(MAX_UNPARSED_BODY_CHARS).collect()),
                ..TokenResponse::default()
            },
            _ => TokenResponse {
                error: Some(format!("HTTP {status}")),
                error_description: Some(NOT_AN_OBJECT.to_string()),
                ..TokenResponse::default()
            },
        };
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        };
        Self {
            status,
            body,
            nonce: header("dpop-nonce").filter(|nonce| !nonce.is_empty()),
            server_time: header("date").as_deref().and_then(parse_http_date),
        }
    }

    /// `<prefix>: <status> <error> <error_description>`, leaving out the
    /// parts Google did not send.
    pub(crate) fn describe(&self, prefix: &str) -> String {
        let mut out = format!("{prefix}: {}", self.status);
        for part in [&self.body.error, &self.body.error_description]
            .into_iter()
            .flatten()
            .filter(|part| !part.is_empty())
        {
            out.push(' ');
            out.push_str(part);
        }
        out
    }
}

/// The form of a token request: `params` plus the client's id and secret.
pub(crate) fn token_form<'a>(
    client: &'a GoogleAiPlanClient,
    params: &[(&'a str, &'a str)],
) -> Vec<(&'a str, &'a str)> {
    let mut form = params.to_vec();
    form.push(("client_id", client.client_id.as_str()));
    form.push(("client_secret", client.client_secret.as_str()));
    form
}

/// The access token's real expiry in Unix milliseconds. Callers refresh five
/// minutes before it, as Google's docs ask.
pub(crate) fn access_expiry_ms(body: &TokenResponse) -> u64 {
    now_ms().saturating_add(body.expires_in.unwrap_or(3599).saturating_mul(1000))
}

/// An HTTP date (`Wed, 25 Nov 2026 12:00:00 GMT`) in Unix seconds.
fn parse_http_date(value: &str) -> Option<i64> {
    jiff::fmt::rfc2822::DateTimeParser::new()
        .parse_timestamp(value.trim())
        .ok()
        .map(|timestamp| timestamp.as_second())
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// A client for the token endpoint: no redirects, since the request carries
/// the client secret and the refresh token.
pub(crate) fn token_http_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(TOKEN_TIMEOUT)
        .build()
}

/// The blocking form of [`token_http_client`], for the sign-in command.
pub(crate) fn blocking_token_http_client() -> reqwest::Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(TOKEN_TIMEOUT)
        .build()
}

#[cfg(test)]
pub(super) mod test_support {
    //! A local fake of Google's token endpoint, shared by the sign-in and
    //! refresh tests.

    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use bytes::Bytes;

    use crate::config::GoogleAiPlanClient;

    /// The date the reference tests use for Google's clock.
    pub const SERVER_DATE: &str = "Wed, 25 Nov 2026 12:00:00 GMT";
    /// [`SERVER_DATE`] in Unix seconds.
    pub const SERVER_TIME: i64 = 1_795_608_000;

    pub fn synthetic_client() -> GoogleAiPlanClient {
        GoogleAiPlanClient {
            client_id: "synthetic-client".to_string(),
            client_secret: "synthetic-secret".to_string(),
        }
    }

    /// One canned response of the fake endpoint.
    #[derive(Clone)]
    pub struct FakeReply {
        pub status: u16,
        pub headers: Vec<(&'static str, &'static str)>,
        pub body: String,
    }

    impl FakeReply {
        pub fn ok(body: serde_json::Value) -> Self {
            Self {
                status: 200,
                headers: Vec::new(),
                body: body.to_string(),
            }
        }

        pub fn error(error: &str, headers: &[(&'static str, &'static str)]) -> Self {
            Self {
                status: 400,
                headers: headers.to_vec(),
                body: serde_json::json!({ "error": error }).to_string(),
            }
        }

        pub fn with_headers(mut self, headers: &[(&'static str, &'static str)]) -> Self {
            self.headers = headers.to_vec();
            self
        }
    }

    /// One request the fake endpoint received.
    #[derive(Debug, Clone)]
    pub struct SeenTokenRequest {
        pub path: String,
        pub dpop: Option<String>,
        pub accept: Option<String>,
        pub content_type: Option<String>,
        pub form: HashMap<String, String>,
    }

    impl SeenTokenRequest {
        pub fn proof_header(&self) -> serde_json::Value {
            proof_part(
                self.dpop
                    .as_deref()
                    .expect("the request carries a DPoP proof"),
                0,
            )
        }

        pub fn proof_claims(&self) -> serde_json::Value {
            proof_part(
                self.dpop
                    .as_deref()
                    .expect("the request carries a DPoP proof"),
                1,
            )
        }
    }

    /// Decodes the header (`0`) or the claims (`1`) of a compact JWT.
    pub fn proof_part(jwt: &str, index: usize) -> serde_json::Value {
        let part = jwt.split('.').nth(index).expect("a compact JWT");
        let bytes = URL_SAFE_NO_PAD.decode(part).expect("base64url JWT part");
        serde_json::from_slice(&bytes).expect("JSON JWT part")
    }

    /// The fake endpoint: its URL and what it received.
    pub struct FakeTokenEndpoint {
        pub url: String,
        seen: Arc<Mutex<Vec<SeenTokenRequest>>>,
    }

    impl FakeTokenEndpoint {
        pub fn seen(&self) -> Vec<SeenTokenRequest> {
            self.seen.lock().unwrap().clone()
        }
    }

    /// Starts a fake token endpoint that answers request `n` with
    /// `replies[n]` (the last one from then on). It runs on its own thread
    /// and runtime, so blocking clients in plain tests and async clients in
    /// tokio tests can both call it.
    pub fn fake_token_endpoint(replies: Vec<FakeReply>) -> FakeTokenEndpoint {
        assert!(!replies.is_empty(), "the fake endpoint needs a reply");
        let seen: Arc<Mutex<Vec<SeenTokenRequest>>> = Arc::default();
        let recorded = seen.clone();
        let (url_tx, url_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let app = axum::Router::new().route(
                    "/{*path}",
                    axum::routing::post(
                        move |uri: http::Uri, headers: http::HeaderMap, body: Bytes| {
                            let recorded = recorded.clone();
                            let replies = replies.clone();
                            async move {
                                let header = |name: &str| {
                                    headers
                                        .get(name)
                                        .and_then(|value| value.to_str().ok())
                                        .map(str::to_string)
                                };
                                let mut seen = recorded.lock().unwrap();
                                let reply = replies[seen.len().min(replies.len() - 1)].clone();
                                seen.push(SeenTokenRequest {
                                    path: uri.path().to_string(),
                                    dpop: header("dpop"),
                                    accept: header("accept"),
                                    content_type: header("content-type"),
                                    form: url::form_urlencoded::parse(&body).into_owned().collect(),
                                });
                                let mut response = http::Response::builder()
                                    .status(reply.status)
                                    .header("content-type", "application/json");
                                for (name, value) in reply.headers {
                                    response = response.header(name, value);
                                }
                                response.body(axum::body::Body::from(reply.body)).unwrap()
                            }
                        },
                    ),
                );
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                url_tx
                    .send(format!("http://{}/token", listener.local_addr().unwrap()))
                    .unwrap();
                axum::serve(listener, app).await.unwrap();
            });
        });
        FakeTokenEndpoint {
            url: url_rx.recv().expect("the fake endpoint starts"),
            seen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{SERVER_DATE, SERVER_TIME};
    use super::*;

    fn headers(pairs: &[(&'static str, &'static str)]) -> http::HeaderMap {
        let mut map = http::HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, http::HeaderValue::from_static(value));
        }
        map
    }

    #[test]
    fn http_dates_parse_to_unix_seconds() {
        assert_eq!(parse_http_date(SERVER_DATE), Some(SERVER_TIME));
        assert_eq!(parse_http_date("not a date"), None);
    }

    #[test]
    fn token_reply_reads_the_body_nonce_and_server_clock() {
        let reply = TokenReply::from_parts(
            400,
            &headers(&[("dpop-nonce", "n1"), ("date", SERVER_DATE)]),
            r#"{"error":"use_dpop_nonce","error_description":"Use a nonce"}"#,
        );
        assert_eq!(reply.status, 400);
        assert_eq!(reply.body.error.as_deref(), Some("use_dpop_nonce"));
        assert_eq!(reply.nonce.as_deref(), Some("n1"));
        assert_eq!(reply.server_time, Some(SERVER_TIME));
        assert_eq!(
            reply.describe("Google token refresh failed"),
            "Google token refresh failed: 400 use_dpop_nonce Use a nonce"
        );
    }

    #[test]
    fn error_page_that_is_not_json_becomes_an_http_error_with_a_short_text() {
        let page = format!("<html>Bad gateway {}</html>", "x".repeat(4096));
        let reply = TokenReply::from_parts(502, &http::HeaderMap::new(), &page);
        assert_eq!(reply.body.error.as_deref(), Some("HTTP 502"));
        let description = reply.body.error_description.clone().unwrap();
        assert_eq!(
            description.chars().count(),
            MAX_UNPARSED_BODY_CHARS,
            "{description}"
        );
        let described = reply.describe("Google sign-in failed");
        println!("{described}");
        assert!(
            described.starts_with("Google sign-in failed: 502 HTTP 502 <html>Bad gateway xxx"),
            "{described}"
        );
        assert!(!described.contains(NOT_AN_OBJECT), "{described}");
        assert_eq!(reply.nonce, None);
        assert_eq!(reply.server_time, None);
    }

    #[test]
    fn token_reply_that_is_not_a_json_object_never_copies_the_body_into_the_error() {
        // Each body holds a synthetic token in its first characters, well
        // inside the length that would be copied from an error page.
        let cases = [
            (
                "a success whose body is a JSON string",
                200,
                r#""access_token=synthetic-leak-1""#.to_string(),
                "synthetic-leak-1",
            ),
            (
                "a success whose body is an array around the token object",
                200,
                r#"[{"access_token":"synthetic-leak-2","expires_in":3599}]"#.to_string(),
                "synthetic-leak-2",
            ),
            (
                "a success whose token object is cut short",
                200,
                r#"{"access_token":"synthetic-leak-3","refresh_token":"synth"#.to_string(),
                "synthetic-leak-3",
            ),
            (
                "a success whose body is plain text",
                201,
                "access_token=synthetic-leak-4".to_string(),
                "synthetic-leak-4",
            ),
            (
                "an error whose body is JSON but not an object",
                400,
                r#"["synthetic-leak-5"]"#.to_string(),
                "synthetic-leak-5",
            ),
        ];
        for (name, status, body, token) in cases {
            let reply = TokenReply::from_parts(status, &http::HeaderMap::new(), &body);
            let described = reply.describe("Google sign-in failed");
            println!("{name}: {described}");
            assert_eq!(
                described,
                format!("Google sign-in failed: {status} HTTP {status} {NOT_AN_OBJECT}"),
                "{name}"
            );
            assert!(!described.contains(token), "{name}: {described}");
            // Without an access token the sign-in and the refresh both fail.
            assert_eq!(reply.body.access_token, None, "{name}");
            assert_eq!(reply.body.refresh_token, None, "{name}");
        }
    }

    #[test]
    fn token_reply_with_mistyped_fields_never_copies_the_body_into_the_error() {
        // A JSON body whose fields have unexpected types must not fall back
        // to the raw text: that text holds the tokens.
        let reply = TokenReply::from_parts(
            200,
            &http::HeaderMap::new(),
            r#"{"access_token":"synthetic-access","expires_in":"soon","scope":7}"#,
        );
        assert_eq!(reply.body.access_token.as_deref(), Some("synthetic-access"));
        assert_eq!(reply.body.expires_in, None);
        assert_eq!(reply.body.scope, None);
        assert_eq!(reply.body.error, None);
        assert!(!reply.describe("x").contains("synthetic-access"));
    }

    #[test]
    fn access_expiry_defaults_to_one_hour() {
        let before = now_ms();
        let expiry = access_expiry_ms(&TokenResponse::default());
        assert!(expiry >= before + 3_599_000 && expiry <= now_ms() + 3_599_000);
        let short = access_expiry_ms(&TokenResponse {
            expires_in: Some(60),
            ..TokenResponse::default()
        });
        assert!(short <= now_ms() + 60_000, "{short}");
    }

    #[test]
    fn scopes_include_the_quota_scope() {
        assert!(SCOPES.split(' ').any(|scope| scope == QUOTA_SCOPE));
    }
}
