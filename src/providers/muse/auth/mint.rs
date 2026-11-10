//! API key mint: exchanges the stored OAuth access token for the Model API
//! key. The server path is async and the CLI path is blocking; both share the
//! request shape and the interpretation of the response.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use http::StatusCode;
use serde::Deserialize;

use super::constants::{
    LOGIN_COMMAND, MAX_AUTH_RESPONSE_BYTES, MINT_API_VERSION, MINT_CONNECT_TIMEOUT, MINT_PATH,
    MINT_TIMEOUT,
};
use crate::providers::muse::sanitize::sanitize_upstream_text;
use crate::providers::muse::usage::SubscriptionUsage;

const MINT_BODY: &str = r#"{"onboard":false}"#;

static FOREIGN_BASE_URL_WARNED: AtomicBool = AtomicBool::new(false);

/// Where to mint and what to compare the minted `base_url` against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintTarget {
    pub mint_base_url: String,
    pub model_base_url: String,
    pub user_agent: String,
}

impl MintTarget {
    pub fn from_config() -> anyhow::Result<Self> {
        Ok(Self {
            mint_base_url: crate::config::muse_mint_base_url()?,
            model_base_url: crate::config::muse_base_url()?,
            user_agent: crate::config::muse_user_agent()?,
        })
    }

    pub fn mint_url(&self) -> String {
        format!("{}{MINT_PATH}", self.mint_base_url.trim_end_matches('/'))
    }
}

/// A minted Model API key plus the account fields the proxy shows or acts on.
/// The account e-mail address is never read from the response.
#[derive(Clone, PartialEq, Eq)]
pub struct MintedKey {
    pub api_key: String,
    pub base_url: String,
    pub is_subs_active: bool,
    pub subs_tier_name: Option<String>,
    pub subs_usage: Option<SubscriptionUsage>,
    pub require_payment: Option<bool>,
    pub can_subscribe: Option<bool>,
    pub user_full_name: Option<String>,
}

impl std::fmt::Debug for MintedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MintedKey")
            .field("api_key", &"[redacted]")
            .field("base_url", &self.base_url)
            .field("is_subs_active", &self.is_subs_active)
            .field("subs_tier_name", &self.subs_tier_name)
            .field("subs_usage", &self.subs_usage)
            .field("require_payment", &self.require_payment)
            .field("can_subscribe", &self.can_subscribe)
            .finish_non_exhaustive()
    }
}

/// Why a mint failed. `Rejected::detail` is already sanitised against the
/// access token used for the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MintError {
    /// The mint answered 401: the stored login no longer works.
    Unauthorized,
    /// Any other answer that will not change by asking again.
    Rejected { status: StatusCode, detail: String },
    /// Connection failure, timeout or 5xx. `status` is set only for a 5xx.
    Transient { status: Option<StatusCode> },
}

impl MintError {
    /// The status the mint answered with, when it answered at all.
    pub fn upstream_status(&self) -> Option<StatusCode> {
        match self {
            MintError::Unauthorized => Some(StatusCode::UNAUTHORIZED),
            MintError::Rejected { status, .. } => Some(*status),
            MintError::Transient { status } => *status,
        }
    }

    /// User-facing message, without the upstream status suffix.
    pub fn message(&self) -> String {
        match self {
            MintError::Unauthorized => {
                format!("Muse login is no longer valid. Run: {LOGIN_COMMAND}")
            }
            MintError::Rejected { detail, .. } => detail.clone(),
            MintError::Transient { status: Some(_) } => "Muse key service is unavailable".into(),
            MintError::Transient { status: None } => "Muse key service is unreachable".into(),
        }
    }

    /// `message()` followed by `(upstream HTTP <status>)` when there is one.
    pub fn message_with_status(&self) -> String {
        with_upstream_status(self.message(), self.upstream_status())
    }
}

/// Appends the upstream status the way every Muse error reports it.
pub fn with_upstream_status(message: String, status: Option<StatusCode>) -> String {
    match status {
        Some(status) => format!("{message} (upstream HTTP {})", status.as_u16()),
        None => message,
    }
}

#[derive(Deserialize)]
struct MintResponse {
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    is_subs_active: bool,
    #[serde(default)]
    subs_tier_name: Option<String>,
    #[serde(default)]
    subs_usage: Option<SubscriptionUsage>,
    #[serde(default)]
    require_payment: Option<bool>,
    #[serde(default)]
    can_subscribe: Option<bool>,
    #[serde(default)]
    user_full_name: Option<String>,
}

#[derive(Deserialize)]
struct ProblemBody {
    #[serde(default)]
    detail: Option<String>,
}

/// Builder for the async mint client: no redirects, the mint's timeouts.
pub fn async_client_builder(timeout: Duration) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(MINT_CONNECT_TIMEOUT.min(timeout))
        .timeout(timeout)
}

/// Builder for the blocking client the CLI uses for login and mint.
pub fn blocking_client_builder(timeout: Duration) -> reqwest::blocking::ClientBuilder {
    reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(MINT_CONNECT_TIMEOUT.min(timeout))
        .timeout(timeout)
}

pub fn blocking_client() -> anyhow::Result<reqwest::blocking::Client> {
    Ok(blocking_client_builder(MINT_TIMEOUT).build()?)
}

pub async fn mint(
    client: &reqwest::Client,
    target: &MintTarget,
    access_token: &str,
) -> Result<MintedKey, MintError> {
    let response = client
        .post(target.mint_url())
        .bearer_auth(access_token)
        .header(http::header::CONTENT_TYPE, "application/json")
        .header(http::header::USER_AGENT, &target.user_agent)
        .header("x-api-version", MINT_API_VERSION)
        .body(MINT_BODY)
        .send()
        .await
        .map_err(|_| MintError::Transient { status: None })?;
    let status = response.status();
    classify_status(status)?;
    let body = read_limited(response)
        .await
        .map_err(|()| unreadable_body(status))?;
    interpret(status, body.as_deref(), target, access_token)
}

pub fn mint_blocking(
    client: &reqwest::blocking::Client,
    target: &MintTarget,
    access_token: &str,
) -> Result<MintedKey, MintError> {
    let response = client
        .post(target.mint_url())
        .bearer_auth(access_token)
        .header(http::header::CONTENT_TYPE, "application/json")
        .header(http::header::USER_AGENT, &target.user_agent)
        .header("x-api-version", MINT_API_VERSION)
        .body(MINT_BODY)
        .send()
        .map_err(|_| MintError::Transient { status: None })?;
    let status = response.status();
    classify_status(status)?;
    let body = read_limited_blocking(response).map_err(|()| unreadable_body(status))?;
    interpret(status, body.as_deref(), target, access_token)
}

/// The answers whose meaning is in the status alone. They are decided before
/// the body is read, so a body that stalls or breaks off cannot turn a 401
/// into a retryable failure.
fn classify_status(status: StatusCode) -> Result<(), MintError> {
    if status == StatusCode::UNAUTHORIZED {
        return Err(MintError::Unauthorized);
    }
    if status.is_server_error() {
        return Err(MintError::Transient {
            status: Some(status),
        });
    }
    if status.is_redirection() {
        return Err(MintError::Rejected {
            status,
            detail: "Muse key service redirected the request".into(),
        });
    }
    Ok(())
}

/// A body that could not be read in full. A rejection keeps its status; a
/// success whose body broke off is worth asking again.
fn unreadable_body(status: StatusCode) -> MintError {
    if status.is_success() {
        MintError::Transient { status: None }
    } else {
        MintError::Rejected {
            status,
            detail: "Muse key service rejected the request".into(),
        }
    }
}

/// Reads at most the auth response limit. `None` means the body was larger;
/// `Err` means it could not be read in full.
async fn read_limited(response: reqwest::Response) -> Result<Option<Vec<u8>>, ()> {
    if response
        .content_length()
        .is_some_and(|length| length as usize > MAX_AUTH_RESPONSE_BYTES)
    {
        return Ok(None);
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| ())?;
        if body.len().saturating_add(chunk.len()) > MAX_AUTH_RESPONSE_BYTES {
            return Ok(None);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Some(body))
}

fn read_limited_blocking(response: reqwest::blocking::Response) -> Result<Option<Vec<u8>>, ()> {
    let mut body = Vec::new();
    response
        .take(MAX_AUTH_RESPONSE_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|_| ())?;
    Ok((body.len() <= MAX_AUTH_RESPONSE_BYTES).then_some(body))
}

fn interpret(
    status: StatusCode,
    body: Option<&[u8]>,
    target: &MintTarget,
    access_token: &str,
) -> Result<MintedKey, MintError> {
    classify_status(status)?;
    let rejected = |detail: &str| MintError::Rejected {
        status,
        detail: detail.to_string(),
    };
    let Some(body) = body else {
        return Err(rejected("Muse key service response exceeds the size limit"));
    };
    if !status.is_success() {
        let detail = serde_json::from_slice::<ProblemBody>(body)
            .ok()
            .and_then(|problem| problem.detail)
            .filter(|detail| !detail.trim().is_empty())
            .map(|detail| sanitize_upstream_text(&detail, &[access_token]))
            .unwrap_or_else(|| "Muse key service rejected the request".into());
        return Err(MintError::Rejected { status, detail });
    }
    let Ok(response) = serde_json::from_slice::<MintResponse>(body) else {
        return Err(rejected("Muse key service returned an unreadable response"));
    };
    if response.api_key.trim().is_empty() {
        return Err(rejected("Muse key service returned no API key"));
    }
    let base_url = match choose_base_url(response.base_url.as_deref(), &target.model_base_url) {
        BaseUrlChoice::Minted(url) => url,
        BaseUrlChoice::Configured => target.model_base_url.clone(),
        BaseUrlChoice::ConfiguredOverForeign => {
            warn_foreign_base_url_once(response.base_url.as_deref());
            target.model_base_url.clone()
        }
    };
    Ok(MintedKey {
        api_key: response.api_key,
        base_url,
        is_subs_active: response.is_subs_active,
        subs_tier_name: response.subs_tier_name,
        subs_usage: response.subs_usage,
        require_payment: response.require_payment,
        can_subscribe: response.can_subscribe,
        user_full_name: response.user_full_name,
    })
}

#[derive(Debug, PartialEq, Eq)]
enum BaseUrlChoice {
    Minted(String),
    Configured,
    ConfiguredOverForeign,
}

/// The minted `base_url` is trusted only over https on the configured host,
/// so a compromised or confused mint cannot send the API key elsewhere. Only
/// its path is taken: scheme, host and port stay the configured ones, and
/// user info, query and fragment are dropped, so the mint can neither move
/// the key to another port nor attach credentials. A plain-http loopback
/// configuration is a local test upstream and always wins.
fn choose_base_url(minted: Option<&str>, configured: &str) -> BaseUrlChoice {
    let Ok(configured_url) = url::Url::parse(configured) else {
        return BaseUrlChoice::ConfiguredOverForeign;
    };
    if configured_url.scheme() == "http" && is_loopback(&configured_url) {
        return BaseUrlChoice::Configured;
    }
    if let Some(raw) = minted
        && configured_url.scheme() == "https"
        && let Ok(minted_url) = url::Url::parse(raw)
        && minted_url.scheme() == "https"
        && minted_url.host_str().is_some()
        && minted_url.host_str() == configured_url.host_str()
    {
        let mut adopted = configured_url;
        adopted.set_path(minted_url.path());
        adopted.set_query(None);
        adopted.set_fragment(None);
        // Neither call can fail on an https URL with a host.
        let _ = adopted.set_username("");
        let _ = adopted.set_password(None);
        return BaseUrlChoice::Minted(adopted.to_string());
    }
    BaseUrlChoice::ConfiguredOverForeign
}

pub(super) fn is_loopback(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        // `localhost.` is the same name written fully qualified.
        Some(url::Host::Domain(domain)) => domain
            .strip_suffix('.')
            .unwrap_or(domain)
            .eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

fn warn_foreign_base_url_once(minted: Option<&str>) {
    if FOREIGN_BASE_URL_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    let mut fields = serde_json::Map::new();
    let host = minted
        .and_then(|raw| url::Url::parse(raw).ok())
        .and_then(|url| url.host_str().map(str::to_string));
    fields.insert(
        "minted_host".into(),
        host.map_or(serde_json::Value::Null, serde_json::Value::String),
    );
    crate::logging::create_logger("muse").warn(
        "Muse key mint returned a base_url outside the configured Model API host; using muse.baseUrl",
        Some(fields),
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::Router;
    use axum::routing::post;
    use std::sync::{Arc, Mutex};

    pub(crate) const ACCESS_TOKEN: &str = "synthetic-oauth-access-token-274";
    pub(crate) const API_KEY: &str = "synthetic-minted-api-key-48-chars-0000000000000";

    /// The mint response recorded on the wire, with synthetic secrets.
    pub(crate) fn observed_mint_body() -> serde_json::Value {
        serde_json::json!({
            "api_key": API_KEY,
            "base_url": "https://api.meta.ai/v1",
            "has_payment_method": false,
            "require_payment": false,
            "is_subs_active": true,
            "can_subscribe": false,
            "show_subs_upsell": true,
            "user_full_name": "Synthetic Person",
            "user_email": "synthetic.person@example.com",
            "user_avatar_url": null,
            "payment_method": null,
            "action_url": null,
            "subs_tier_id": "12345678901234567",
            "subs_tier_name": "Muse Code Max Usage",
            "is_subs_upgrade_available": false,
            "subs_usage": {
                "window": {"used_percent": 1, "window_duration_mins": 300, "resets_at": 1794291326},
                "weekly": {"used_percent": 0, "resets_at": 1794787200},
                "tier": "12345678901234567"
            }
        })
    }

    #[derive(Debug, Clone)]
    pub(crate) struct Recorded {
        pub method: String,
        pub path: String,
        pub headers: http::HeaderMap,
        pub body: String,
    }

    pub(crate) type Log = Arc<Mutex<Vec<Recorded>>>;

    /// Serves `/muse-code/key` with a fixed answer and records each request.
    pub(crate) async fn spawn_mint(
        status: StatusCode,
        headers: Vec<(&'static str, &'static str)>,
        body: String,
    ) -> (String, Log) {
        let log: Log = Arc::default();
        let recorded = log.clone();
        let app = Router::new().route(
            "/muse-code/key",
            post(
                move |method: http::Method,
                      uri: http::Uri,
                      req_headers: http::HeaderMap,
                      req_body: String| {
                    let recorded = recorded.clone();
                    let headers = headers.clone();
                    let body = body.clone();
                    async move {
                        recorded.lock().unwrap().push(Recorded {
                            method: method.to_string(),
                            path: uri.path().to_string(),
                            headers: req_headers,
                            body: req_body,
                        });
                        let mut response =
                            axum::response::Response::new(axum::body::Body::from(body));
                        *response.status_mut() = status;
                        for (name, value) in headers {
                            response.headers_mut().insert(
                                http::HeaderName::from_static(name),
                                http::HeaderValue::from_static(value),
                            );
                        }
                        response
                    }
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, log)
    }

    pub(crate) fn target(mint_base_url: &str) -> MintTarget {
        MintTarget {
            mint_base_url: mint_base_url.to_string(),
            model_base_url: "http://127.0.0.1:9/v1".into(),
            user_agent: "claude-code-proxy/test".into(),
        }
    }

    fn test_client(timeout: Duration) -> reqwest::Client {
        async_client_builder(timeout).no_proxy().build().unwrap()
    }

    async fn mint_against(status: StatusCode, body: String) -> Result<MintedKey, MintError> {
        let (base, _) = spawn_mint(status, vec![], body).await;
        mint(&test_client(MINT_TIMEOUT), &target(&base), ACCESS_TOKEN).await
    }

    #[tokio::test]
    async fn mint_sends_the_observed_request_shape() {
        let (base, log) =
            spawn_mint(StatusCode::OK, vec![], observed_mint_body().to_string()).await;
        mint(&test_client(MINT_TIMEOUT), &target(&base), ACCESS_TOKEN)
            .await
            .unwrap();
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1);
        let request = &log[0];
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/muse-code/key");
        assert_eq!(request.body, r#"{"onboard":false}"#);
        assert_eq!(
            request.headers["authorization"],
            format!("Bearer {ACCESS_TOKEN}").as_str()
        );
        assert_eq!(request.headers["content-type"], "application/json");
        assert_eq!(request.headers["x-api-version"], "1.0.0");
        assert_eq!(request.headers["user-agent"], "claude-code-proxy/test");
    }

    #[tokio::test]
    async fn mint_reads_the_observed_response_and_ignores_unknown_fields() {
        let mut body = observed_mint_body();
        body["field_added_later"] = serde_json::json!({"nested": [1, 2, 3]});
        let minted = mint_against(StatusCode::OK, body.to_string())
            .await
            .unwrap();
        assert_eq!(minted.api_key, API_KEY);
        // The configured Model API base is a loopback test URL, which wins.
        assert_eq!(minted.base_url, "http://127.0.0.1:9/v1");
        assert!(minted.is_subs_active);
        assert_eq!(
            minted.subs_tier_name.as_deref(),
            Some("Muse Code Max Usage")
        );
        assert_eq!(minted.require_payment, Some(false));
        assert_eq!(minted.can_subscribe, Some(false));
        assert_eq!(minted.user_full_name.as_deref(), Some("Synthetic Person"));
        let usage = minted.subs_usage.unwrap();
        let window = usage.window.unwrap();
        assert_eq!(
            (
                window.used_percent,
                window.resets_at,
                window.window_duration_mins
            ),
            (1, 1794291326, Some(300))
        );
        let weekly = usage.weekly.unwrap();
        assert_eq!(
            (
                weekly.used_percent,
                weekly.resets_at,
                weekly.window_duration_mins
            ),
            (0, 1794787200, None)
        );
    }

    #[test]
    fn minted_key_debug_hides_the_key_and_has_no_email() {
        let minted = interpret(
            StatusCode::OK,
            Some(observed_mint_body().to_string().as_bytes()),
            &target("http://127.0.0.1:1"),
            ACCESS_TOKEN,
        )
        .unwrap();
        let rendered = format!("{minted:?}");
        assert!(!rendered.contains(API_KEY), "{rendered}");
        assert!(!rendered.contains("example.com"), "{rendered}");
    }

    #[tokio::test]
    async fn mint_synthetic_empty_api_key_is_rejected() {
        let mut body = observed_mint_body();
        body["api_key"] = serde_json::json!("");
        let error = mint_against(StatusCode::OK, body.to_string())
            .await
            .unwrap_err();
        assert_eq!(
            error,
            MintError::Rejected {
                status: StatusCode::OK,
                detail: "Muse key service returned no API key".into()
            }
        );
    }

    #[test]
    fn base_url_is_adopted_only_over_https_on_the_configured_host() {
        let configured = "https://api.meta.ai/v1";
        assert_eq!(
            choose_base_url(Some("https://api.meta.ai/v1"), configured),
            BaseUrlChoice::Minted("https://api.meta.ai/v1".into())
        );
        assert_eq!(
            choose_base_url(Some("https://api.meta.ai/v2"), configured),
            BaseUrlChoice::Minted("https://api.meta.ai/v2".into())
        );
        assert_eq!(
            choose_base_url(Some("http://api.meta.ai/v1"), configured),
            BaseUrlChoice::ConfiguredOverForeign
        );
        assert_eq!(
            choose_base_url(Some("https://attacker.example/v1"), configured),
            BaseUrlChoice::ConfiguredOverForeign
        );
        assert_eq!(
            choose_base_url(Some("not a url"), configured),
            BaseUrlChoice::ConfiguredOverForeign
        );
        assert_eq!(
            choose_base_url(None, configured),
            BaseUrlChoice::ConfiguredOverForeign
        );
        assert_eq!(
            choose_base_url(Some("https://api.meta.ai/v1"), "http://127.0.0.1:4000/v1"),
            BaseUrlChoice::Configured
        );
    }

    #[test]
    fn synthetic_fully_qualified_localhost_configuration_is_a_local_upstream() {
        for configured in ["http://localhost.:4000/v1", "http://LOCALHOST.:4000/v1"] {
            assert_eq!(
                choose_base_url(Some("https://api.meta.ai/v1"), configured),
                BaseUrlChoice::Configured,
                "{configured}"
            );
        }
        assert_eq!(
            choose_base_url(Some("https://api.meta.ai/v1"), "http://relay.example./v1"),
            BaseUrlChoice::ConfiguredOverForeign
        );
    }

    #[test]
    fn synthetic_minted_base_url_keeps_only_its_path() {
        let configured = "https://api.meta.ai/v1";
        for (minted, expected) in [
            (
                "https://user:secret@api.meta.ai/v1",
                "https://api.meta.ai/v1",
            ),
            ("https://api.meta.ai:8443/v1", "https://api.meta.ai/v1"),
            (
                "https://api.meta.ai/v2?key=synthetic#fragment",
                "https://api.meta.ai/v2",
            ),
            ("https://u@api.meta.ai:1/v3?q=1#f", "https://api.meta.ai/v3"),
        ] {
            assert_eq!(
                choose_base_url(Some(minted), configured),
                BaseUrlChoice::Minted(expected.into()),
                "{minted}"
            );
        }
        // A configured explicit port is kept; the minted one never is.
        assert_eq!(
            choose_base_url(
                Some("https://api.meta.ai:9999/v2"),
                "https://api.meta.ai:8443/v1"
            ),
            BaseUrlChoice::Minted("https://api.meta.ai:8443/v2".into())
        );
        assert_eq!(
            choose_base_url(
                Some("https://api.meta.ai/v2"),
                "https://api.meta.ai:8443/v1"
            ),
            BaseUrlChoice::Minted("https://api.meta.ai:8443/v2".into())
        );
    }

    /// Answers every connection with `status_line` and the first bytes of a
    /// body its Content-Length promises, then closes the connection.
    fn spawn_cut_off_mint(status_line: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                // Read the whole request first: closing with unread input
                // resets the connection before the client sees the head.
                let mut request = Vec::new();
                let mut buffer = [0u8; 4096];
                while !request.ends_with(MINT_BODY.as_bytes()) {
                    match std::io::Read::read(&mut stream, &mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => request.extend_from_slice(&buffer[..read]),
                    }
                }
                let answer = format!(
                    "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n{{\"title\":\"Authen"
                );
                let _ = std::io::Write::write_all(&mut stream, answer.as_bytes());
            }
        });
        base
    }

    #[tokio::test]
    async fn synthetic_mint_status_survives_a_body_that_breaks_off() {
        for (status_line, expected) in [
            ("401 Unauthorized", MintError::Unauthorized),
            (
                "503 Service Unavailable",
                MintError::Transient {
                    status: Some(StatusCode::SERVICE_UNAVAILABLE),
                },
            ),
            (
                "403 Forbidden",
                MintError::Rejected {
                    status: StatusCode::FORBIDDEN,
                    detail: "Muse key service rejected the request".into(),
                },
            ),
            ("200 OK", MintError::Transient { status: None }),
        ] {
            let base = spawn_cut_off_mint(status_line);
            let error = mint(&test_client(MINT_TIMEOUT), &target(&base), ACCESS_TOKEN)
                .await
                .unwrap_err();
            assert_eq!(error, expected, "{status_line}");
        }
    }

    #[test]
    fn synthetic_blocking_mint_status_survives_a_body_that_breaks_off() {
        let client = blocking_client_builder(MINT_TIMEOUT)
            .no_proxy()
            .build()
            .unwrap();
        for (status_line, expected) in [
            ("401 Unauthorized", MintError::Unauthorized),
            (
                "503 Service Unavailable",
                MintError::Transient {
                    status: Some(StatusCode::SERVICE_UNAVAILABLE),
                },
            ),
        ] {
            let base = spawn_cut_off_mint(status_line);
            let error = mint_blocking(&client, &target(&base), ACCESS_TOKEN).unwrap_err();
            assert_eq!(error, expected, "{status_line}");
        }
    }

    #[tokio::test]
    async fn mint_401_is_unauthorized() {
        let body =
            r#"{"title":"Authentication Error","detail":"Authentication Error","status":401}"#;
        let error = mint_against(StatusCode::UNAUTHORIZED, body.into())
            .await
            .unwrap_err();
        assert_eq!(error, MintError::Unauthorized);
        assert_eq!(
            error.message_with_status(),
            "Muse login is no longer valid. Run: claude-code-proxy muse auth login (upstream HTTP 401)"
        );
    }

    #[tokio::test]
    async fn mint_4xx_is_rejected_with_the_problem_detail() {
        let body = r#"{"title":"JSON Schema Validation Error","detail":"Request body is required but was empty or null","status":400}"#;
        let error = mint_against(StatusCode::BAD_REQUEST, body.into())
            .await
            .unwrap_err();
        assert_eq!(
            error,
            MintError::Rejected {
                status: StatusCode::BAD_REQUEST,
                detail: "Request body is required but was empty or null".into()
            }
        );
        let body = r#"{"title":"API version not found","detail":"The requested API version 9.9.9, does not exist. Please check the version and try again.","status":404}"#;
        let error = mint_against(StatusCode::NOT_FOUND, body.into())
            .await
            .unwrap_err();
        assert_eq!(error.upstream_status(), Some(StatusCode::NOT_FOUND));
        assert_eq!(
            error.message_with_status(),
            "The requested API version 9.9.9, does not exist. Please check the version and try again. (upstream HTTP 404)"
        );
    }

    #[tokio::test]
    async fn mint_synthetic_detail_is_sanitised_against_the_access_token() {
        let body = serde_json::json!({
            "title": "Bad",
            "detail": format!("token {ACCESS_TOKEN} for synthetic.person@example.com"),
            "status": 403
        });
        let error = mint_against(StatusCode::FORBIDDEN, body.to_string())
            .await
            .unwrap_err();
        let message = error.message();
        assert_eq!(message, "token [redacted] for [redacted]");
    }

    #[tokio::test]
    async fn mint_synthetic_5xx_is_transient_with_status() {
        let error = mint_against(StatusCode::SERVICE_UNAVAILABLE, "{}".into())
            .await
            .unwrap_err();
        assert_eq!(
            error,
            MintError::Transient {
                status: Some(StatusCode::SERVICE_UNAVAILABLE)
            }
        );
        assert_eq!(
            error.message_with_status(),
            "Muse key service is unavailable (upstream HTTP 503)"
        );
    }

    #[tokio::test]
    async fn mint_synthetic_unreadable_oversized_and_redirected_answers_are_rejected() {
        let error = mint_against(StatusCode::OK, "<html>not json</html>".into())
            .await
            .unwrap_err();
        assert!(matches!(error, MintError::Rejected { status, .. } if status == StatusCode::OK));

        let oversized = format!(
            r#"{{"api_key":"{API_KEY}","padding":"{}"}}"#,
            "x".repeat(MAX_AUTH_RESPONSE_BYTES)
        );
        let error = mint_against(StatusCode::OK, oversized).await.unwrap_err();
        assert_eq!(
            error,
            MintError::Rejected {
                status: StatusCode::OK,
                detail: "Muse key service response exceeds the size limit".into()
            }
        );

        let (base, log) = spawn_mint(
            StatusCode::FOUND,
            vec![("location", "https://www.facebook.com/unsupportedbrowser")],
            String::new(),
        )
        .await;
        let error = mint(&test_client(MINT_TIMEOUT), &target(&base), ACCESS_TOKEN)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            MintError::Rejected {
                status: StatusCode::FOUND,
                detail: "Muse key service redirected the request".into()
            }
        );
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn mint_unreachable_service_is_transient_without_status() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let error = mint(&test_client(MINT_TIMEOUT), &target(&base), ACCESS_TOKEN)
            .await
            .unwrap_err();
        assert_eq!(error, MintError::Transient { status: None });
        assert_eq!(
            error.message_with_status(),
            "Muse key service is unreachable"
        );
    }

    #[tokio::test]
    async fn mint_that_never_answers_times_out_after_20_seconds() {
        // The first mint answers so the connection is pooled; every later
        // one never answers. The clock is paused only after the warm-up, so
        // the 20 s timeout cannot race the connect. (Whether the stalled
        // request reaches the server before the paused clock jumps is not
        // deterministic, so the server's call count is not asserted.)
        let app = Router::new().route(
            "/muse-code/key",
            post({
                let answered = Arc::new(std::sync::atomic::AtomicBool::new(false));
                move || {
                    let first = !answered.swap(true, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        if !first {
                            std::future::pending::<()>().await;
                        }
                        observed_mint_body().to_string()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = test_client(MINT_TIMEOUT);
        mint(&client, &target(&base), ACCESS_TOKEN).await.unwrap();

        tokio::time::pause();
        let started = tokio::time::Instant::now();
        let error = mint(&client, &target(&base), ACCESS_TOKEN)
            .await
            .unwrap_err();
        assert_eq!(error, MintError::Transient { status: None });
        // The timer wheel rounds up to the next millisecond.
        let elapsed = started.elapsed();
        assert!(
            elapsed >= MINT_TIMEOUT && elapsed < MINT_TIMEOUT + Duration::from_secs(1),
            "{elapsed:?}"
        );
    }

    #[test]
    fn blocking_mint_matches_the_async_interpretation() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (base, log) = runtime.block_on(spawn_mint(
            StatusCode::OK,
            vec![],
            observed_mint_body().to_string(),
        ));
        let client = blocking_client_builder(MINT_TIMEOUT)
            .no_proxy()
            .build()
            .unwrap();
        let minted = mint_blocking(&client, &target(&base), ACCESS_TOKEN).unwrap();
        assert_eq!(minted.api_key, API_KEY);
        assert_eq!(
            minted.subs_tier_name.as_deref(),
            Some("Muse Code Max Usage")
        );
        let request = log.lock().unwrap()[0].clone();
        assert_eq!(request.body, r#"{"onboard":false}"#);
        assert_eq!(request.headers["x-api-version"], "1.0.0");
        assert_eq!(request.headers["user-agent"], "claude-code-proxy/test");
    }
}
