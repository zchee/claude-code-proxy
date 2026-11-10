//! Device-code login against the Meta OAuth host. Meta offers no browser
//! PKCE flow and no refresh grant for this client, so this is the only way to
//! obtain the stored access token.

use std::io::Read;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use http::StatusCode;
use serde::Deserialize;

use super::constants::{
    CLIENT_ID, DEVICE_AUTHORIZATION_PATH, DEVICE_TOKEN_PATH, GRANT_DEVICE_CODE, LOGIN_COMMAND,
    MAX_AUTH_RESPONSE_BYTES,
};
use super::mint::{self, MintError, MintTarget};
use super::token_store::{MuseTokenStore, StoredAuth};
use crate::auth::AuthStorage;
use crate::providers::muse::sanitize::sanitize_upstream_token;

const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Polling at exactly the advertised interval earned a `slow_down`, so the
/// first wait is padded.
const POLL_INTERVAL_PADDING: Duration = Duration::from_secs(2);
const SLOW_DOWN_INCREMENT: Duration = Duration::from_secs(5);
const DEFAULT_DEVICE_CODE_LIFETIME: Duration = Duration::from_secs(600);
/// Bounds on what the device authorisation response may ask for, so a
/// hostile answer can neither spin the poll loop nor park it for days.
const MIN_POLL_INTERVAL_SECS: u64 = 1;
const MAX_POLL_INTERVAL_SECS: u64 = 60;
const MAX_DEVICE_CODE_LIFETIME: Duration = Duration::from_secs(1800);
/// The verification page Meta was observed to send users to.
const META_VERIFICATION_HOST: &str = "auth.meta.com";
/// The observed user code is 9 characters (`XXXX-XXXX`).
const MAX_USER_CODE_LEN: usize = 32;

#[derive(Deserialize)]
struct DeviceAuthResponse {
    device_code: String,
    user_code: String,
    #[serde(default)]
    verification_uri: Option<String>,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    interval: Option<u64>,
}

#[derive(Deserialize)]
struct TokenResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    token_type: Option<String>,
}

#[derive(Deserialize)]
struct OAuthError {
    #[serde(default)]
    error: Option<String>,
}

enum DevicePoll {
    Token(String),
    Pending,
    SlowDown,
}

/// Clock and sleep, replaceable so tests do not wait in real time.
pub trait DeviceRuntime {
    fn monotonic_now(&self) -> Instant;
    fn unix_time_ms(&self) -> u64;
    fn sleep(&self, duration: Duration);
}

pub struct SystemRuntime;

impl DeviceRuntime for SystemRuntime {
    fn monotonic_now(&self) -> Instant {
        Instant::now()
    }

    fn unix_time_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLoginConfig {
    pub oauth_host: String,
    pub mint: MintTarget,
}

impl DeviceLoginConfig {
    pub fn from_config() -> anyhow::Result<Self> {
        Ok(Self {
            oauth_host: crate::config::muse_oauth_host()?,
            mint: MintTarget::from_config()?,
        })
    }

    fn oauth_url(&self, path: &str) -> String {
        format!("{}{path}", self.oauth_host.trim_end_matches('/'))
    }
}

/// Runs the device flow, mints a key once to fill the display fields, and
/// saves the login. Nothing is saved when the mint fails.
pub fn device_login<S: AuthStorage<StoredAuth>>(
    store: &MuseTokenStore<S>,
) -> anyhow::Result<StoredAuth> {
    let config = DeviceLoginConfig::from_config()?;
    let client = mint::blocking_client()?;
    device_login_with(store, &client, &config, &SystemRuntime)
}

pub fn device_login_with<S: AuthStorage<StoredAuth>>(
    store: &MuseTokenStore<S>,
    client: &reqwest::blocking::Client,
    config: &DeviceLoginConfig,
    runtime: &dyn DeviceRuntime,
) -> anyhow::Result<StoredAuth> {
    let access_token = run_device_flow(client, config, runtime)?;
    let minted =
        mint::mint_blocking(client, &config.mint, &access_token).map_err(|error| match error {
            MintError::Unauthorized => anyhow::anyhow!(
                "Meta approved the device login but the Muse key service refused it (upstream HTTP 401); nothing was saved"
            ),
            other => anyhow::anyhow!(
                "Muse key service did not mint a key for the new login, so nothing was saved: {}",
                other.message_with_status()
            ),
        })?;
    let auth = StoredAuth {
        access_token,
        obtained_at_ms: runtime.unix_time_ms(),
        client_id: CLIENT_ID.into(),
        oauth_host: config.oauth_host.trim_end_matches('/').to_string(),
        user_full_name: minted.user_full_name,
        is_subs_active: minted.is_subs_active,
        subs_tier_name: minted.subs_tier_name,
    };
    store.save_auth(auth.clone())?;
    Ok(auth)
}

fn run_device_flow(
    client: &reqwest::blocking::Client,
    config: &DeviceLoginConfig,
    runtime: &dyn DeviceRuntime,
) -> anyhow::Result<String> {
    let auth = request_device_code(client, config)?;
    let visit = verification_url(&auth, config)?;
    let user_code = display_user_code(&auth.user_code)?;
    println!(
        "\nOpen this URL on any device to authorize:\n\n  {visit}\n\nand enter the code:  {user_code}\n"
    );

    let mut interval = auth
        .interval
        .map(|secs| Duration::from_secs(secs.clamp(MIN_POLL_INTERVAL_SECS, MAX_POLL_INTERVAL_SECS)))
        .unwrap_or(DEFAULT_POLL_INTERVAL)
        .saturating_add(POLL_INTERVAL_PADDING);
    let lifetime = auth
        .expires_in
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_DEVICE_CODE_LIFETIME)
        .min(MAX_DEVICE_CODE_LIFETIME);
    let deadline = runtime
        .monotonic_now()
        .checked_add(lifetime)
        .ok_or_else(|| anyhow::anyhow!("Muse device code lifetime is too large"))?;
    let expired = || {
        anyhow::anyhow!(
            "Muse device login timed out after {}s. Run: {LOGIN_COMMAND}",
            lifetime.as_secs()
        )
    };

    loop {
        let remaining = deadline.saturating_duration_since(runtime.monotonic_now());
        if remaining.is_zero() {
            return Err(expired());
        }
        runtime.sleep(interval.min(remaining));
        if runtime.monotonic_now() >= deadline {
            return Err(expired());
        }
        match poll_token(client, config, &auth.device_code)? {
            DevicePoll::Token(token) => return Ok(token),
            DevicePoll::Pending => {}
            DevicePoll::SlowDown => interval = interval.saturating_add(SLOW_DOWN_INCREMENT),
        }
    }
}

/// The page the user is sent to. A URL from the response is shown only when
/// it is https (plain http only for a loopback host) on Meta's verification
/// host or on the configured OAuth host, and is printed in its parsed form so
/// control characters cannot reach the terminal. Anything else ends the login:
/// a response that points users elsewhere is not trusted.
fn verification_url(
    auth: &DeviceAuthResponse,
    config: &DeviceLoginConfig,
) -> anyhow::Result<String> {
    let Some(raw) = auth
        .verification_uri_complete
        .as_deref()
        .or(auth.verification_uri.as_deref())
    else {
        return Ok(config.oauth_url("/oauth/device/"));
    };
    let configured_host = url::Url::parse(&config.oauth_host)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase));
    let trusted = url::Url::parse(raw).ok().filter(|url| {
        let Some(host) = url.host_str().map(str::to_ascii_lowercase) else {
            return false;
        };
        let secure = match url.scheme() {
            "https" => true,
            "http" => mint::is_loopback(url),
            _ => false,
        };
        secure
            && url.username().is_empty()
            && url.password().is_none()
            && (host == META_VERIFICATION_HOST || configured_host.as_deref() == Some(&host))
    });
    match trusted {
        Some(url) => Ok(url.to_string()),
        None => anyhow::bail!(
            "Meta's login service sent a verification URL that is not https on {META_VERIFICATION_HOST} or on the configured OAuth host, so the login was stopped. Run: {LOGIN_COMMAND}"
        ),
    }
}

/// The user code with everything but ASCII letters, digits and hyphens
/// removed, so it cannot carry terminal control sequences, and bounded in
/// length, so it cannot flood the terminal either.
fn display_user_code(raw: &str) -> anyhow::Result<String> {
    let code: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    if code.is_empty() {
        anyhow::bail!("Meta's login service sent no usable device code. Run: {LOGIN_COMMAND}");
    }
    if code.len() > MAX_USER_CODE_LEN {
        anyhow::bail!(
            "Meta's login service sent a device code longer than {MAX_USER_CODE_LEN} characters, so the login was stopped. Run: {LOGIN_COMMAND}"
        );
    }
    Ok(code)
}

fn request_device_code(
    client: &reqwest::blocking::Client,
    config: &DeviceLoginConfig,
) -> anyhow::Result<DeviceAuthResponse> {
    let (status, body) = post_form(
        client,
        config,
        DEVICE_AUTHORIZATION_PATH,
        &[("client_id", CLIENT_ID)],
    )?;
    if status == StatusCode::NOT_FOUND && body.as_ref().is_some_and(|body| body.is_empty()) {
        anyhow::bail!(
            "Meta's login service rejected the Muse client id (HTTP 404 with an empty body). \
             Check that CCP_MUSE_OAUTH_HOST (muse.oauthHost in config.json) points at https://auth.meta.com"
        );
    }
    if !status.is_success() {
        anyhow::bail!(
            "Muse device authorization failed with status {}",
            status.as_u16()
        );
    }
    let body = body.ok_or_else(oversized)?;
    serde_json::from_slice(&body)
        .map_err(|_| anyhow::anyhow!("Muse device authorization response is not valid"))
}

fn poll_token(
    client: &reqwest::blocking::Client,
    config: &DeviceLoginConfig,
    device_code: &str,
) -> anyhow::Result<DevicePoll> {
    let (status, body) = post_form(
        client,
        config,
        DEVICE_TOKEN_PATH,
        &[
            ("grant_type", GRANT_DEVICE_CODE),
            ("device_code", device_code),
            ("client_id", CLIENT_ID),
        ],
    )?;
    let body = body.ok_or_else(oversized)?;
    if status.is_success() {
        let token: TokenResponse = serde_json::from_slice(&body)
            .map_err(|_| anyhow::anyhow!("Muse device token response is not valid JSON"))?;
        let bearer = token
            .token_type
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("bearer"));
        if token.access_token.is_empty() || !bearer {
            anyhow::bail!("Muse device token response is invalid");
        }
        return Ok(DevicePoll::Token(token.access_token));
    }
    let error = serde_json::from_slice::<OAuthError>(&body)
        .ok()
        .and_then(|body| body.error);
    match error.as_deref() {
        Some("authorization_pending") => Ok(DevicePoll::Pending),
        Some("slow_down") => Ok(DevicePoll::SlowDown),
        Some("expired_token") => {
            anyhow::bail!("Muse device code expired before it was approved. Run: {LOGIN_COMMAND}")
        }
        Some("access_denied") => anyhow::bail!("Muse device login was denied"),
        Some("invalid_grant") => anyhow::bail!(
            "Muse device login failed: invalid_grant (the device code was not accepted). Run: {LOGIN_COMMAND}"
        ),
        Some(other) => anyhow::bail!(
            "Muse device login failed: {} (HTTP {})",
            sanitize_upstream_token(other, &[device_code])
                .unwrap_or_else(|| "unknown error".into()),
            status.as_u16()
        ),
        None => anyhow::bail!("Muse device login failed with status {}", status.as_u16()),
    }
}

/// Posts a form to the OAuth host. The body is `None` when it exceeds the
/// size limit. A redirect means the host rejected the User-Agent.
fn post_form(
    client: &reqwest::blocking::Client,
    config: &DeviceLoginConfig,
    path: &str,
    form: &[(&str, &str)],
) -> anyhow::Result<(StatusCode, Option<Vec<u8>>)> {
    let response = client
        .post(config.oauth_url(path))
        .header(http::header::USER_AGENT, &config.mint.user_agent)
        .form(form)
        .send()
        .map_err(|error| {
            if error.is_timeout() {
                anyhow::anyhow!("Meta's login service did not answer in time")
            } else {
                anyhow::anyhow!("Meta's login service is unreachable")
            }
        })?;
    let status = response.status();
    if status.is_redirection() {
        anyhow::bail!(
            "Meta's login service redirected the request (HTTP {}), which it does when it rejects the User-Agent. \
             Set a non-empty User-Agent with CCP_MUSE_USER_AGENT (muse.userAgent in config.json)",
            status.as_u16()
        );
    }
    let mut body = Vec::new();
    response
        .take(MAX_AUTH_RESPONSE_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::TimedOut {
                anyhow::anyhow!("Meta's login service did not answer in time")
            } else {
                anyhow::anyhow!("Meta's login service response could not be read")
            }
        })?;
    Ok((
        status,
        (body.len() <= MAX_AUTH_RESPONSE_BYTES).then_some(body),
    ))
}

fn oversized() -> anyhow::Error {
    anyhow::anyhow!("Meta's login service response exceeds the size limit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::InMemoryAuthStore;
    use axum::Router;
    use axum::routing::post;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    const TEST_UNIX_TIME_MS: u64 = 1_794_274_189_000;
    const DEVICE_CODE: &str = "synthetic-device-code-43-chars-000000000000";
    const ACCESS_TOKEN: &str = "synthetic-oauth-access-token-274";

    struct TestRuntime {
        start: Instant,
        elapsed: Mutex<Duration>,
        sleeps: Mutex<Vec<Duration>>,
    }

    impl Default for TestRuntime {
        fn default() -> Self {
            Self {
                start: Instant::now(),
                elapsed: Mutex::new(Duration::ZERO),
                sleeps: Mutex::new(Vec::new()),
            }
        }
    }

    impl DeviceRuntime for TestRuntime {
        fn monotonic_now(&self) -> Instant {
            self.start + *self.elapsed.lock().unwrap()
        }

        fn unix_time_ms(&self) -> u64 {
            TEST_UNIX_TIME_MS
        }

        fn sleep(&self, duration: Duration) {
            self.sleeps.lock().unwrap().push(duration);
            *self.elapsed.lock().unwrap() += duration;
        }
    }

    #[derive(Clone)]
    struct Canned {
        status: StatusCode,
        location: Option<&'static str>,
        body: String,
    }

    fn json(status: StatusCode, body: impl Into<String>) -> Canned {
        Canned {
            status,
            location: None,
            body: body.into(),
        }
    }

    fn redirect() -> Canned {
        Canned {
            status: StatusCode::FOUND,
            location: Some("https://www.facebook.com/unsupportedbrowser"),
            body: String::new(),
        }
    }

    #[derive(Debug, Clone)]
    struct Seen {
        path: String,
        user_agent: Option<String>,
        form: String,
    }

    struct AuthServer {
        base: String,
        seen: Arc<Mutex<Vec<Seen>>>,
    }

    impl AuthServer {
        fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }
    }

    fn device_body(interval: u64, expires_in: u64) -> String {
        serde_json::json!({
            "device_code": DEVICE_CODE,
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://auth.meta.com/oauth/device/",
            "verification_uri_complete": "https://auth.meta.com/oauth/device/?code=ABCD-EFGH",
            "expires_in": expires_in,
            "interval": interval
        })
        .to_string()
    }

    fn pending() -> Canned {
        json(
            StatusCode::BAD_REQUEST,
            r#"{"error":"authorization_pending","error_description":"User has not yet completed authorization."}"#,
        )
    }

    fn slow_down() -> Canned {
        json(
            StatusCode::BAD_REQUEST,
            r#"{"error":"slow_down","error_description":"Polling interval exceeded."}"#,
        )
    }

    fn granted() -> Canned {
        json(
            StatusCode::OK,
            format!(r#"{{"access_token":"{ACCESS_TOKEN}","token_type":"Bearer"}}"#),
        )
    }

    fn respond(canned: Canned) -> axum::response::Response {
        let mut response = axum::response::Response::new(axum::body::Body::from(canned.body));
        *response.status_mut() = canned.status;
        if let Some(location) = canned.location {
            response.headers_mut().insert(
                http::header::LOCATION,
                http::HeaderValue::from_static(location),
            );
        }
        response
    }

    /// OAuth host and mint on one local axum server, running on its own
    /// thread because the blocking client cannot share a runtime with it.
    fn spawn_auth_server(device: Canned, polls: Vec<Canned>, mint_answer: Canned) -> AuthServer {
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let polls = Arc::new(Mutex::new(VecDeque::from(polls)));
        let record = |seen: Arc<Mutex<Vec<Seen>>>| {
            move |uri: http::Uri, headers: http::HeaderMap, body: String| {
                seen.lock().unwrap().push(Seen {
                    path: uri.path().to_string(),
                    user_agent: headers
                        .get(http::header::USER_AGENT)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_string),
                    form: body,
                });
            }
        };
        let on_device = record(seen.clone());
        let on_poll = record(seen.clone());
        let on_mint = record(seen.clone());
        let app = Router::new()
            .route(
                "/oidc/device/authorization/",
                post(
                    move |uri: http::Uri, headers: http::HeaderMap, body: String| {
                        on_device(uri, headers, body);
                        let device = device.clone();
                        async move { respond(device) }
                    },
                ),
            )
            .route(
                "/oidc/device/token/",
                post(
                    move |uri: http::Uri, headers: http::HeaderMap, body: String| {
                        on_poll(uri, headers, body);
                        let next = polls
                            .lock()
                            .unwrap()
                            .pop_front()
                            .unwrap_or_else(|| json(StatusCode::INTERNAL_SERVER_ERROR, "{}"));
                        async move { respond(next) }
                    },
                ),
            )
            .route(
                "/muse-code/key",
                post(
                    move |uri: http::Uri, headers: http::HeaderMap, body: String| {
                        on_mint(uri, headers, body);
                        let mint_answer = mint_answer.clone();
                        async move { respond(mint_answer) }
                    },
                ),
            );
        let base = serve_on_thread(app);
        AuthServer { base, seen }
    }

    fn serve_on_thread(app: Router) -> String {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                sender.send(listener.local_addr().unwrap()).unwrap();
                axum::serve(listener, app).await.unwrap();
            });
        });
        format!("http://{}", receiver.recv().unwrap())
    }

    fn config(base: &str) -> DeviceLoginConfig {
        DeviceLoginConfig {
            oauth_host: base.to_string(),
            mint: MintTarget {
                mint_base_url: base.to_string(),
                model_base_url: "http://127.0.0.1:9/v1".into(),
                user_agent: "claude-code-proxy/test".into(),
            },
        }
    }

    fn client(timeout: Duration) -> reqwest::blocking::Client {
        mint::blocking_client_builder(timeout)
            .no_proxy()
            .build()
            .unwrap()
    }

    fn mint_ok() -> Canned {
        json(
            StatusCode::OK,
            super::super::mint::tests::observed_mint_body().to_string(),
        )
    }

    fn login(
        server: &AuthServer,
        runtime: &TestRuntime,
    ) -> (
        anyhow::Result<StoredAuth>,
        MuseTokenStore<InMemoryAuthStore<StoredAuth>>,
    ) {
        let store = MuseTokenStore::new(InMemoryAuthStore::<StoredAuth>::default());
        let result = device_login_with(
            &store,
            &client(Duration::from_secs(20)),
            &config(&server.base),
            runtime,
        );
        (result, store)
    }

    #[test]
    fn device_login_follows_the_observed_exchange_and_saves_display_fields() {
        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 600)),
            vec![pending(), slow_down(), pending(), granted()],
            mint_ok(),
        );
        let runtime = TestRuntime::default();
        let (result, store) = login(&server, &runtime);
        let saved = result.unwrap();
        assert_eq!(store.load_auth().unwrap(), Some(saved.clone()));
        assert_eq!(saved.access_token, ACCESS_TOKEN);
        assert_eq!(saved.obtained_at_ms, TEST_UNIX_TIME_MS);
        assert_eq!(saved.client_id, "1031625952748946");
        assert_eq!(saved.oauth_host, server.base);
        assert_eq!(saved.user_full_name.as_deref(), Some("Synthetic Person"));
        assert!(saved.is_subs_active);
        assert_eq!(saved.subs_tier_name.as_deref(), Some("Muse Code Max Usage"));

        // Waiting starts at interval + 2 s and grows by 5 s on slow_down.
        assert_eq!(
            *runtime.sleeps.lock().unwrap(),
            [7, 7, 12, 12].map(Duration::from_secs)
        );

        let seen = server.seen();
        let paths: Vec<_> = seen.iter().map(|seen| seen.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "/oidc/device/authorization/",
                "/oidc/device/token/",
                "/oidc/device/token/",
                "/oidc/device/token/",
                "/oidc/device/token/",
                "/muse-code/key",
            ]
        );
        for request in &seen {
            assert_eq!(
                request.user_agent.as_deref(),
                Some("claude-code-proxy/test"),
                "{}",
                request.path
            );
        }
        assert_eq!(seen[0].form, "client_id=1031625952748946");
        assert_eq!(
            seen[1].form,
            format!(
                "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code&device_code={DEVICE_CODE}&client_id=1031625952748946"
            )
        );
    }

    #[test]
    fn device_login_stops_on_invalid_grant() {
        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 600)),
            vec![json(
                StatusCode::BAD_REQUEST,
                r#"{"error":"invalid_grant","error_description":"Unknown device_code."}"#,
            )],
            mint_ok(),
        );
        let (result, store) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("invalid_grant"), "{error}");
        assert!(store.load_auth().unwrap().is_none());
        assert_eq!(server.seen().len(), 2);
    }

    #[test]
    fn device_login_synthetic_stops_on_expired_token_and_access_denied() {
        for (code, expected) in [
            ("expired_token", "expired before it was approved"),
            ("access_denied", "was denied"),
        ] {
            let server = spawn_auth_server(
                json(StatusCode::OK, device_body(5, 600)),
                vec![
                    pending(),
                    json(StatusCode::BAD_REQUEST, format!(r#"{{"error":"{code}"}}"#)),
                ],
                mint_ok(),
            );
            let (result, store) = login(&server, &TestRuntime::default());
            let error = result.unwrap_err().to_string();
            assert!(error.contains(expected), "{code}: {error}");
            assert!(store.load_auth().unwrap().is_none());
            assert_eq!(server.seen().len(), 3, "{code}");
        }
    }

    #[test]
    fn device_login_synthetic_gives_up_when_the_code_expires() {
        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 10)),
            vec![pending(), pending(), pending()],
            mint_ok(),
        );
        let runtime = TestRuntime::default();
        let (result, _) = login(&server, &runtime);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("timed out after 10s"), "{error}");
        assert_eq!(
            *runtime.sleeps.lock().unwrap(),
            [7, 3].map(Duration::from_secs)
        );
        assert_eq!(server.seen().len(), 2);
    }

    #[test]
    fn device_login_synthetic_hostile_interval_and_lifetime_are_bounded() {
        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(100_000, 100_000)),
            vec![pending(); 40],
            mint_ok(),
        );
        let runtime = TestRuntime::default();
        let (result, _) = login(&server, &runtime);
        let error = result.unwrap_err().to_string();
        assert!(error.contains("timed out after 1800s"), "{error}");
        let sleeps = runtime.sleeps.lock().unwrap().clone();
        // At most 60 s plus the 2 s padding between polls.
        assert_eq!(sleeps[0], Duration::from_secs(62));
        assert_eq!(sleeps.iter().sum::<Duration>(), Duration::from_secs(1800));
        assert_eq!(server.seen().len(), 1 + 29);

        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(0, 600)),
            vec![granted()],
            mint_ok(),
        );
        let runtime = TestRuntime::default();
        let (result, _) = login(&server, &runtime);
        result.unwrap();
        assert_eq!(*runtime.sleeps.lock().unwrap(), [Duration::from_secs(3)]);
    }

    fn device_answer(
        uri: Option<&str>,
        complete: Option<&str>,
        user_code: &str,
    ) -> DeviceAuthResponse {
        DeviceAuthResponse {
            device_code: DEVICE_CODE.into(),
            user_code: user_code.into(),
            verification_uri: uri.map(str::to_string),
            verification_uri_complete: complete.map(str::to_string),
            expires_in: Some(600),
            interval: Some(5),
        }
    }

    #[test]
    fn verification_url_synthetic_is_shown_only_for_meta_or_the_configured_host() {
        let meta = config("https://auth.meta.com");
        let local = config("http://127.0.0.1:4000");
        for (config, uri, complete, expected) in [
            (
                &meta,
                Some("https://auth.meta.com/oauth/device/"),
                Some("https://auth.meta.com/oauth/device/?code=ABCD-EFGH"),
                "https://auth.meta.com/oauth/device/?code=ABCD-EFGH",
            ),
            (
                &meta,
                Some("https://AUTH.META.COM/oauth/device/"),
                None,
                "https://auth.meta.com/oauth/device/",
            ),
            (&meta, None, None, "https://auth.meta.com/oauth/device/"),
            (
                &local,
                Some("http://127.0.0.1:4000/oauth/device/"),
                None,
                "http://127.0.0.1:4000/oauth/device/",
            ),
            (
                &local,
                Some("https://auth.meta.com/oauth/device/"),
                None,
                "https://auth.meta.com/oauth/device/",
            ),
        ] {
            assert_eq!(
                verification_url(&device_answer(uri, complete, "ABCD-EFGH"), config).unwrap(),
                expected
            );
        }
        for hostile in [
            "https://attacker.example/oauth/device/",
            "https://auth.meta.com.attacker.example/oauth/device/",
            "http://auth.meta.com/oauth/device/",
            "https://user:pass@auth.meta.com/oauth/device/",
            "javascript:alert(1)",
            "not a url",
        ] {
            for answer in [
                device_answer(Some(hostile), None, "ABCD-EFGH"),
                device_answer(
                    Some("https://auth.meta.com/oauth/device/"),
                    Some(hostile),
                    "ABCD-EFGH",
                ),
            ] {
                let error = verification_url(&answer, &meta).unwrap_err().to_string();
                assert!(error.contains("verification URL"), "{hostile}: {error}");
            }
        }
    }

    #[test]
    fn user_code_synthetic_control_characters_are_removed() {
        assert_eq!(display_user_code("ABCD-EFGH").unwrap(), "ABCD-EFGH");
        assert_eq!(
            display_user_code("AB\u{1b}]8;;https://x\u{7}CD-EF\n").unwrap(),
            "AB8httpsxCD-EF"
        );
        assert!(display_user_code("\u{1b}\u{7} ;").is_err());
    }

    #[test]
    fn user_code_synthetic_longer_than_32_characters_stops_the_login() {
        let longest = "A".repeat(32);
        assert_eq!(display_user_code(&longest).unwrap(), longest);
        // Characters removed by the filter do not count towards the limit.
        assert_eq!(
            display_user_code(&format!("\u{1b}{longest}\u{7}")).unwrap(),
            longest
        );
        for flood in ["A".repeat(33), "ABCD-".repeat(13_000)] {
            let error = display_user_code(&flood).unwrap_err().to_string();
            assert_eq!(
                error,
                format!(
                    "Meta's login service sent a device code longer than 32 characters, so the login was stopped. Run: {LOGIN_COMMAND}"
                )
            );
        }
    }

    #[test]
    fn device_login_synthetic_flooding_user_code_stops_before_polling() {
        let body = serde_json::json!({
            "device_code": DEVICE_CODE,
            "user_code": "ABCD-".repeat(1000),
            "verification_uri": "https://auth.meta.com/oauth/device/",
            "expires_in": 600,
            "interval": 5
        });
        let server = spawn_auth_server(
            json(StatusCode::OK, body.to_string()),
            vec![granted()],
            mint_ok(),
        );
        let (result, store) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("longer than 32 characters"), "{error}");
        assert!(store.load_auth().unwrap().is_none());
        assert_eq!(server.seen().len(), 1);
    }

    #[test]
    fn verification_url_synthetic_accepts_a_fully_qualified_localhost() {
        let local = config("http://localhost.:4000");
        for (uri, expected) in [
            (
                "http://localhost.:4000/oauth/device/",
                "http://localhost.:4000/oauth/device/",
            ),
            (
                "http://LOCALHOST.:4000/oauth/device/?code=ABCD-EFGH",
                "http://localhost.:4000/oauth/device/?code=ABCD-EFGH",
            ),
        ] {
            assert_eq!(
                verification_url(&device_answer(Some(uri), None, "ABCD-EFGH"), &local).unwrap(),
                expected
            );
        }
        let error = verification_url(
            &device_answer(
                Some("http://auth.meta.com./oauth/device/"),
                None,
                "ABCD-EFGH",
            ),
            &local,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("verification URL"), "{error}");
    }

    #[test]
    fn device_login_synthetic_foreign_verification_url_stops_before_polling() {
        let body = serde_json::json!({
            "device_code": DEVICE_CODE,
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://attacker.example/oauth/device/",
            "expires_in": 600,
            "interval": 5
        });
        let server = spawn_auth_server(
            json(StatusCode::OK, body.to_string()),
            vec![granted()],
            mint_ok(),
        );
        let (result, store) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("verification URL"), "{error}");
        assert!(store.load_auth().unwrap().is_none());
        assert_eq!(server.seen().len(), 1);
    }

    #[test]
    fn device_authorization_empty_404_reports_a_rejected_client_id() {
        let server = spawn_auth_server(json(StatusCode::NOT_FOUND, ""), vec![], mint_ok());
        let (result, _) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("client id"), "{error}");
        assert!(error.contains("CCP_MUSE_OAUTH_HOST"), "{error}");
        assert!(error.contains("muse.oauthHost"), "{error}");
    }

    #[test]
    fn redirect_reports_a_rejected_user_agent_without_following_it() {
        let server = spawn_auth_server(redirect(), vec![], mint_ok());
        let (result, _) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("User-Agent"), "{error}");
        assert!(error.contains("CCP_MUSE_USER_AGENT"), "{error}");
        assert!(error.contains("muse.userAgent"), "{error}");
        assert_eq!(server.seen().len(), 1);

        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 600)),
            vec![redirect()],
            mint_ok(),
        );
        let (result, _) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("CCP_MUSE_USER_AGENT"), "{error}");
    }

    #[test]
    fn device_login_synthetic_rejects_bad_token_answers() {
        for (answer, expected) in [
            (
                json(
                    StatusCode::OK,
                    format!(r#"{{"access_token":"{ACCESS_TOKEN}","token_type":"mac"}}"#),
                ),
                "invalid",
            ),
            (
                json(
                    StatusCode::OK,
                    format!(r#"{{"access_token":"{ACCESS_TOKEN}"}}"#),
                ),
                "invalid",
            ),
            (
                json(StatusCode::OK, r#"{"token_type":"Bearer"}"#),
                "invalid",
            ),
            (json(StatusCode::OK, "<html>ok</html>"), "not valid JSON"),
            (
                json(StatusCode::OK, "x".repeat(MAX_AUTH_RESPONSE_BYTES + 1)),
                "size limit",
            ),
        ] {
            let server = spawn_auth_server(
                json(StatusCode::OK, device_body(5, 600)),
                vec![answer],
                mint_ok(),
            );
            let (result, store) = login(&server, &TestRuntime::default());
            let error = result.unwrap_err().to_string();
            assert!(error.contains(expected), "{expected}: {error}");
            assert!(store.load_auth().unwrap().is_none());
        }
    }

    #[test]
    fn device_login_accepts_a_lowercase_bearer_token_type() {
        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 600)),
            vec![json(
                StatusCode::OK,
                format!(r#"{{"access_token":"{ACCESS_TOKEN}","token_type":"bearer"}}"#),
            )],
            mint_ok(),
        );
        let (result, _) = login(&server, &TestRuntime::default());
        assert_eq!(result.unwrap().access_token, ACCESS_TOKEN);
    }

    #[test]
    fn device_login_saves_nothing_when_the_mint_fails() {
        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 600)),
            vec![granted()],
            json(
                StatusCode::UNAUTHORIZED,
                r#"{"title":"Authentication Error","detail":"Authentication Error","status":401}"#,
            ),
        );
        let (result, store) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("nothing was saved"), "{error}");
        assert!(error.contains("401"), "{error}");
        assert!(store.load_auth().unwrap().is_none());

        let server = spawn_auth_server(
            json(StatusCode::OK, device_body(5, 600)),
            vec![granted()],
            json(StatusCode::SERVICE_UNAVAILABLE, "{}"),
        );
        let (result, store) = login(&server, &TestRuntime::default());
        let error = result.unwrap_err().to_string();
        assert!(error.contains("nothing was saved"), "{error}");
        assert!(error.contains("upstream HTTP 503"), "{error}");
        assert!(store.load_auth().unwrap().is_none());
    }

    #[test]
    fn device_authorization_that_never_answers_fails_on_the_client_timeout() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                held.push(stream);
            }
        });
        let store = MuseTokenStore::new(InMemoryAuthStore::<StoredAuth>::default());
        let started = Instant::now();
        let error = device_login_with(
            &store,
            &client(Duration::from_secs(1)),
            &config(&base),
            &TestRuntime::default(),
        )
        .unwrap_err()
        .to_string();
        let elapsed = started.elapsed();
        assert!(error.contains("did not answer in time"), "{error}");
        assert!(
            elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(5),
            "{elapsed:?}"
        );
    }
}
