//! The browser sign-in: Google OAuth with a loopback redirect, PKCE (S256),
//! `state`, and an OIDC `nonce`, checked when Google returns an ID token.

use std::collections::HashMap;
use std::io::{BufRead, IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use url::Url;

use super::token_store::{GoogleAiPlanTokenStore, StoredAuth};
use super::{AUTHORIZE_URL, QUOTA_SCOPE, SCOPES, TOKEN_URL, TokenReply, access_expiry_ms, now_ms};
use crate::auth::AuthStorage;
use crate::config::GoogleAiPlanClient;
use crate::providers::grok::auth::pkce::{PkceCodes, generate_pkce, generate_state};
use crate::providers::vertex::sanitize::redact_secrets;

const CALLBACK_PATH: &str = "/auth/callback";
const LEGACY_CALLBACK_PATH: &str = "/oauth2callback";
/// Long enough to finish a plan upgrade checkout in the browser.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// How far the ID token's `exp` may lie in the past, for clock skew.
const ID_TOKEN_EXPIRY_LEEWAY_MS: u64 = 60_000;
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// The whole page for a callback with this sign-in's `state` that did not
/// sign in.
const CALLBACK_FAILED_PAGE: &str =
    "Sign-in did not complete. Return to the terminal for the reason.";

/// Signs in with Google in the browser and stores the credential.
///
/// The browser is sent back to a listener on `127.0.0.1`. When the browser
/// runs on another machine, the user can paste the URL it ended on into the
/// terminal instead.
pub fn login<S: AuthStorage<StoredAuth>>(
    client: &GoogleAiPlanClient,
    store: &GoogleAiPlanTokenStore<S>,
) -> anyhow::Result<()> {
    let session = SignInSession::start(client, TOKEN_URL)?;
    let auth_url = session.authorize_url(AUTHORIZE_URL)?;
    println!("Open this URL in your browser to sign in with Google:\n\n  {auth_url}\n");
    if let Err(err) = open_browser(
        &auth_url,
        std::env::consts::OS,
        std::process::Command::spawn,
    ) {
        println!("Could not open the browser automatically: {err}");
    }
    let manual = manual_input();
    if manual.interactive {
        println!(
            "Continue with Google in your browser. If the browser is on another machine, paste \
             the final URL from its address bar here:"
        );
    } else {
        println!("Continue with Google in your browser.");
    }
    let auth = session.wait(&manual.lines, CALLBACK_TIMEOUT)?;
    store.save_auth(auth)?;
    Ok(())
}

/// One sign-in attempt: the listener and the secrets the response must match.
struct SignInSession<'a> {
    client: &'a GoogleAiPlanClient,
    token_url: String,
    http: reqwest::blocking::Client,
    listener: TcpListener,
    redirect_uri: String,
    pkce: PkceCodes,
    state: String,
    nonce: String,
}

impl<'a> SignInSession<'a> {
    fn start(client: &'a GoogleAiPlanClient, token_url: &str) -> anyhow::Result<Self> {
        // Loopback only, on a free port: Desktop app clients accept any port,
        // and 0.0.0.0 is not allowed.
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        Ok(Self {
            client,
            token_url: token_url.to_string(),
            http: super::blocking_token_http_client()?,
            listener,
            redirect_uri: format!("http://127.0.0.1:{port}{CALLBACK_PATH}"),
            pkce: generate_pkce(),
            state: generate_state(),
            nonce: generate_state(),
        })
    }

    fn authorize_url(&self, authorize_endpoint: &str) -> anyhow::Result<String> {
        let mut url = Url::parse(authorize_endpoint)?;
        url.query_pairs_mut()
            .append_pair("client_id", &self.client.client_id)
            .append_pair("redirect_uri", &self.redirect_uri)
            .append_pair("response_type", "code")
            .append_pair("scope", SCOPES)
            .append_pair("access_type", "offline")
            .append_pair("prompt", "consent")
            .append_pair("code_challenge", &self.pkce.challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &self.state)
            .append_pair("nonce", &self.nonce);
        Ok(url.into())
    }

    /// Waits for the browser's callback or a pasted URL, whichever comes
    /// first, and finishes the sign-in with it. A request to the listener
    /// that does not carry this sign-in's `state` is answered and the wait
    /// goes on. `manual` yields the lines the user types; a closed channel
    /// only means there will be none.
    fn wait(&self, manual: &Receiver<String>, timeout: Duration) -> anyhow::Result<StoredAuth> {
        let deadline = Instant::now() + timeout;
        let mut manual_open = true;
        loop {
            if Instant::now() >= deadline {
                anyhow::bail!("Timed out waiting for the Google sign-in");
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if let Some(result) = self.answer_callback(stream) {
                        return result;
                    }
                    continue;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => return Err(error.into()),
            }
            if manual_open {
                match manual.try_recv() {
                    Ok(line) if line.trim().is_empty() => continue,
                    Ok(line) => return self.complete_pasted(&line),
                    Err(TryRecvError::Disconnected) => manual_open = false,
                    Err(TryRecvError::Empty) => {}
                }
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Handles one connection to the listener. `None` means it was not this
    /// sign-in's callback and the wait goes on: another path, not an HTTP
    /// request, or a callback without this sign-in's `state`.
    ///
    /// Any web page the user has open, and any local process, can send a
    /// request to the loopback port. If such a request could end the wait,
    /// it would abort the sign-in and the real redirect from Google would be
    /// ignored. So a callback with a missing or wrong `state` gets a
    /// `400 Bad Request` and nothing else in it (`error`, `code`) is read.
    fn answer_callback(&self, mut stream: TcpStream) -> Option<anyhow::Result<StoredAuth>> {
        // On macOS and the BSDs an accepted socket inherits the listener's
        // non-blocking mode, which would make the read fail at once.
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut buffer = [0_u8; 8192];
        let length = stream.read(&mut buffer).ok()?;
        let request = String::from_utf8_lossy(&buffer[..length]);
        let target = request.lines().next().and_then(|line| {
            let mut parts = line.split_whitespace();
            (parts.next() == Some("GET"))
                .then(|| parts.next())
                .flatten()
        })?;
        let url = Url::parse(&format!("http://127.0.0.1{target}")).ok()?;
        if !matches!(url.path(), CALLBACK_PATH | LEGACY_CALLBACK_PATH) {
            respond(&mut stream, "404 Not Found", "Not found");
            return None;
        }
        let params: HashMap<String, String> = url.query_pairs().into_owned().collect();
        if !self.carries_state(&params) {
            respond(
                &mut stream,
                "400 Bad Request",
                "This request does not belong to this sign-in.",
            );
            return None;
        }
        // Finishing before answering lets the page show a failed exchange or
        // check instead of "Signed in". The page never carries the error:
        // its text comes from Google's reply and the redirect, and any other
        // page in the browser could read it. The terminal prints it.
        let result = self.complete(&params);
        match &result {
            Ok(_) => respond(
                &mut stream,
                "200 OK",
                "Signed in to the Google AI Ultra plan. You can close this tab and return to the \
                 terminal.",
            ),
            Err(_) => respond(&mut stream, "400 Bad Request", CALLBACK_FAILED_PAGE),
        }
        Some(result)
    }

    /// Finishes the sign-in with the URL the browser was sent to after
    /// consent, pasted by the user.
    fn complete_pasted(&self, input: &str) -> anyhow::Result<StoredAuth> {
        let url = Url::parse(input.trim())
            .map_err(|_| anyhow::anyhow!("The pasted text is not a URL"))?;
        let params: HashMap<String, String> = url.query_pairs().into_owned().collect();
        self.complete(&params)
    }

    /// Whether the redirect's parameters carry this sign-in's `state`, which
    /// proves the redirect belongs to this sign-in. The comparison takes the
    /// same time wherever the first difference is.
    fn carries_state(&self, params: &HashMap<String, String>) -> bool {
        params
            .get("state")
            .is_some_and(|state| constant_time_eq(state, &self.state))
    }

    /// Checks the redirect's parameters and exchanges its code. The `state`
    /// is checked before anything else and before any token request; a
    /// missing or wrong one fails the sign-in. The browser path has already
    /// answered such a callback and kept waiting, so this failure is reached
    /// from a pasted URL, which the user chose to paste.
    fn complete(&self, params: &HashMap<String, String>) -> anyhow::Result<StoredAuth> {
        if !self.carries_state(params) {
            anyhow::bail!("Sign-in state mismatch");
        }
        match params.get("error").map(String::as_str) {
            Some("access_denied") => anyhow::bail!("Sign-in was canceled"),
            Some(error) if !error.is_empty() => anyhow::bail!("Google sign-in failed: {error}"),
            _ => {}
        }
        let code = params
            .get("code")
            .filter(|code| !code.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing authorization code"))?;
        self.finish(code)
    }

    /// Exchanges the authorization code for tokens, checks the result, and
    /// builds the credential.
    fn finish(&self, code: &str) -> anyhow::Result<StoredAuth> {
        println!("Finishing sign-in...");
        let form = super::token_form(
            self.client,
            &[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("code_verifier", &self.pkce.verifier),
                ("redirect_uri", &self.redirect_uri),
            ],
        );
        let response = self
            .http
            .post(&self.token_url)
            .header(http::header::ACCEPT, "application/json")
            .form(&form)
            .send()?;
        let status = response.status().as_u16();
        let text = response.text()?;
        // Google's text may quote the request. The code, the PKCE verifier
        // and the client secret must not reach the terminal, so they are
        // removed from an error body before `from_parts` cuts it to a length,
        // and again from the parsed fields, which also covers a JSON-escaped
        // quote and a success reply without the tokens. A success body is
        // left whole: it carries the tokens to store.
        let secrets = [
            code,
            self.pkce.verifier.as_str(),
            self.client.client_secret.as_str(),
        ];
        let text = if (200..300).contains(&status) {
            text
        } else {
            redact_secrets(&text, &secrets)
        };
        let mut reply = TokenReply::from_parts(status, &text);
        let non_empty = |value: &Option<String>| value.clone().filter(|value| !value.is_empty());
        let (200, Some(access), Some(refresh)) = (
            reply.status,
            non_empty(&reply.body.access_token),
            non_empty(&reply.body.refresh_token),
        ) else {
            for field in [&mut reply.body.error, &mut reply.body.error_description] {
                *field = field.as_deref().map(|text| redact_secrets(text, &secrets));
            }
            anyhow::bail!(reply.describe("Google sign-in failed"));
        };
        if !reply
            .body
            .scope
            .as_deref()
            .unwrap_or_default()
            .split(' ')
            .any(|scope| scope == QUOTA_SCOPE)
        {
            anyhow::bail!(
                "Google did not grant access to the Google AI Ultra plan. Try signing in again."
            );
        }
        if let Some(id_token) = non_empty(&reply.body.id_token) {
            check_id_token(
                Some(&id_token),
                &self.nonce,
                &self.client.client_id,
                now_ms(),
            )?;
        }
        Ok(StoredAuth {
            access,
            refresh,
            expires_at_ms: access_expiry_ms(&reply.body),
            client_id: self.client.client_id.clone(),
            dpop_key: String::new(),
            dpop_nonce: None,
            project_id: None,
        })
    }
}

/// Checks the ID token's issuer, audience, expiry and nonce, as Google's docs
/// require.
fn check_id_token(
    id_token: Option<&str>,
    nonce: &str,
    client_id: &str,
    now_ms: u64,
) -> anyhow::Result<()> {
    let Some(id_token) = id_token.filter(|token| !token.is_empty()) else {
        anyhow::bail!("Google sign-in returned no ID token");
    };
    // The token comes straight from Google's token endpoint over TLS, so the
    // claims are read without verifying the signature (OpenID Connect Core
    // 3.1.3.7).
    let claims = id_token
        .split('.')
        .nth(1)
        .and_then(|part| URL_SAFE_NO_PAD.decode(part.trim_end_matches('=')).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .unwrap_or_default();
    let issuer_ok = matches!(
        claims["iss"].as_str(),
        Some("https://accounts.google.com" | "accounts.google.com")
    );
    let not_expired = claims["exp"]
        .as_f64()
        .is_some_and(|exp| exp * 1000.0 > now_ms.saturating_sub(ID_TOKEN_EXPIRY_LEEWAY_MS) as f64);
    let audience_ok = claims["aud"].as_str() == Some(client_id);
    let nonce_ok = claims["nonce"].as_str() == Some(nonce);
    if !(issuer_ok && audience_ok && nonce_ok && not_expired) {
        anyhow::bail!("Google sign-in returned an ID token that failed validation");
    }
    Ok(())
}

/// The lines the user types, read on a helper thread so the wait can race
/// them against the browser's callback.
struct ManualInput {
    lines: Receiver<String>,
    interactive: bool,
}

fn manual_input() -> ManualInput {
    let (sender, lines) = mpsc::channel();
    // A stdin that is not a terminal (a pipe, /dev/null) gets no prompt and
    // is not read; dropping the sender tells the wait there will be no lines.
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        // The thread may stay blocked on stdin after the sign-in ends: std
        // cannot cancel a read, and the command exits right after.
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
    }
    ManualInput { lines, interactive }
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.bytes()
        .zip(right.bytes())
        .fold(0_u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}

fn open_browser(
    url: &str,
    platform: &str,
    execute: impl FnOnce(&mut std::process::Command) -> std::io::Result<std::process::Child>,
) -> anyhow::Result<()> {
    let command = match platform {
        "macos" => ("open", vec![url]),
        "linux" => ("xdg-open", vec![url]),
        "windows" => windows_browser_command(url),
        _ => return Ok(()),
    };
    execute(std::process::Command::new(command.0).args(command.1))?;
    Ok(())
}

fn windows_browser_command(url: &str) -> (&str, Vec<&str>) {
    // Keep URL metacharacters out of a command interpreter.
    ("rundll32.exe", vec!["url.dll,FileProtocolHandler", url])
}

#[cfg(test)]
use open_browser as open_login_url;

// Run the shared assertions against each provider's private launcher.
#[cfg(test)]
#[allow(clippy::duplicate_mod)]
#[path = "../../../../tests/support/oauth_browser_launch.rs"]
mod browser_launch_tests;

#[cfg(test)]
mod tests {
    use std::sync::mpsc::Sender;
    use std::thread::JoinHandle;

    use sha2::{Digest, Sha256};

    use super::super::test_support::{
        FakeReply, FakeTokenEndpoint, fake_token_endpoint, synthetic_client,
    };
    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(20);
    /// Shaped like a Google authorization code: `4/` and a long body.
    const SYNTHETIC_CODE: &str = "4/0ASynthetic0Authorization0Code0Body0xYz9_Kq-Lm3";

    fn b64_json(value: &serde_json::Value) -> String {
        URL_SAFE_NO_PAD.encode(value.to_string())
    }

    /// An unsigned ID token with `claims`, as the code reads it.
    fn id_token(claims: serde_json::Value) -> String {
        format!(
            "{}.{}.synthetic-signature",
            b64_json(&serde_json::json!({"alg": "RS256", "typ": "JWT"})),
            b64_json(&claims)
        )
    }

    fn good_claims(session: &SignInSession<'_>) -> serde_json::Value {
        serde_json::json!({
            "iss": "https://accounts.google.com",
            "aud": session.client.client_id,
            "exp": now_ms() / 1000 + 3600,
            "nonce": session.nonce,
        })
    }

    fn good_tokens(session: &SignInSession<'_>) -> serde_json::Value {
        serde_json::json!({
            "access_token": "synthetic-access",
            "refresh_token": "synthetic-refresh",
            "expires_in": 3599,
            "scope": format!("openid {QUOTA_SCOPE} https://www.googleapis.com/auth/userinfo.email"),
            "id_token": id_token(good_claims(session)),
        })
    }

    /// A pasted-input channel already holding `lines`, with its sender kept
    /// open so the wait does not see a closed terminal.
    fn pasted(lines: &[&str]) -> (Sender<String>, Receiver<String>) {
        let (sender, receiver) = mpsc::channel();
        for line in lines {
            sender.send((*line).to_string()).unwrap();
        }
        (sender, receiver)
    }

    /// Sends one browser request to `port` and returns the whole HTTP
    /// response. The request goes out in a single write, as a browser sends
    /// it, because the listener reads it once. The read gives up after
    /// [`TEST_TIMEOUT`], so a request nobody answers fails the test instead
    /// of hanging it.
    fn browser_request(port: u16, path_and_query: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.set_read_timeout(Some(TEST_TIMEOUT)).unwrap();
        let request = format!("GET {path_and_query} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n");
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    /// Plays the browser: requests each of `paths` from the session's
    /// listener in turn, each after the answer to the one before, and
    /// returns the responses.
    fn browser_get_all(session: &SignInSession<'_>, paths: Vec<String>) -> JoinHandle<Vec<String>> {
        let port = Url::parse(&session.redirect_uri).unwrap().port().unwrap();
        std::thread::spawn(move || {
            paths
                .iter()
                .map(|path| browser_request(port, path))
                .collect()
        })
    }

    /// Plays the browser for one request.
    fn browser_get(session: &SignInSession<'_>, path_and_query: String) -> JoinHandle<String> {
        let port = Url::parse(&session.redirect_uri).unwrap().port().unwrap();
        std::thread::spawn(move || browser_request(port, &path_and_query))
    }

    /// Asserts that `page` is the answer to a callback without this
    /// sign-in's `state`.
    fn assert_turned_away(name: &str, page: &str) {
        assert!(
            page.starts_with("HTTP/1.1 400 Bad Request"),
            "{name}: {page}"
        );
        assert!(
            page.ends_with("\r\n\r\nThis request does not belong to this sign-in."),
            "{name}: {page}"
        );
    }

    /// Asserts that `page` is the answer to a callback with this sign-in's
    /// `state` that did not sign in: the fixed page and nothing else.
    fn assert_failed_page(name: &str, page: &str) {
        assert!(
            page.starts_with("HTTP/1.1 400 Bad Request"),
            "{name}: {page}"
        );
        assert!(
            page.ends_with(&format!("\r\n\r\n{CALLBACK_FAILED_PAGE}")),
            "{name}: {page}"
        );
    }

    fn assert_no_token_request(endpoint: &FakeTokenEndpoint, name: &str) {
        let seen = endpoint.seen();
        assert!(
            seen.is_empty(),
            "{name}: token requests were sent: {seen:?}"
        );
    }

    #[test]
    fn authorize_url_carries_the_client_pkce_state_nonce_and_scopes() {
        let client = synthetic_client();
        let session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let url = Url::parse(&session.authorize_url(AUTHORIZE_URL).unwrap()).unwrap();
        println!("authorize URL: {url}");
        assert_eq!(
            url.as_str().split('?').next().unwrap(),
            "https://accounts.google.com/o/oauth2/v2/auth"
        );
        let params: HashMap<_, _> = url.query_pairs().into_owned().collect();
        let port = session.listener.local_addr().unwrap().port();
        let expected: HashMap<String, String> = [
            ("client_id", "synthetic-client".to_string()),
            (
                "redirect_uri",
                format!("http://127.0.0.1:{port}/auth/callback"),
            ),
            ("response_type", "code".to_string()),
            ("scope", SCOPES.to_string()),
            ("access_type", "offline".to_string()),
            ("prompt", "consent".to_string()),
            (
                "code_challenge",
                URL_SAFE_NO_PAD.encode(Sha256::digest(session.pkce.verifier.as_bytes())),
            ),
            ("code_challenge_method", "S256".to_string()),
            ("state", session.state.clone()),
            ("nonce", session.nonce.clone()),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
        assert_eq!(params, expected);
        assert_ne!(session.state, session.nonce);
        assert_eq!(session.state.len(), 43, "32 random bytes, base64url");
    }

    #[test]
    fn pasted_url_without_the_sign_ins_state_is_rejected_before_any_token_request() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(serde_json::json!({}))]);
        let client = synthetic_client();
        let session = SignInSession::start(&client, &endpoint.url).unwrap();
        let redirect = session.redirect_uri.clone();
        for (name, pasted_url) in [
            ("no state", format!("{redirect}?code=4%2F0Aabc")),
            (
                "another state",
                format!("{redirect}?code=4%2F0Aabc&state=someone-else"),
            ),
            (
                "an error with another state",
                format!("{redirect}?error=access_denied&state=someone-else"),
            ),
            (
                "the state with one character changed",
                format!(
                    "{redirect}?code=4%2F0Aabc&state={}x",
                    &session.state[..session.state.len() - 1]
                ),
            ),
        ] {
            let (_sender, lines) = pasted(&[&pasted_url]);
            let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
            println!("{name}: {error}");
            assert_eq!(error.to_string(), "Sign-in state mismatch", "{name}");
        }
        assert_no_token_request(&endpoint, "state mismatch");
    }

    #[test]
    fn pasted_url_with_the_state_exchanges_the_decoded_code_without_dpop() {
        let endpoint = fake_token_endpoint(vec![FakeReply::error("invalid_grant", &[])]);
        let client = synthetic_client();
        let session = SignInSession::start(&client, &endpoint.url).unwrap();
        let pasted_url = format!(
            "  {}?state={}&code=4%2F0Aabc&scope=x \n",
            session.redirect_uri, session.state
        );
        let (_sender, lines) = pasted(&[&pasted_url]);
        let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Google sign-in failed: 400 invalid_grant"
        );

        let seen = endpoint.seen();
        assert_eq!(seen.len(), 1);
        let request = &seen[0];
        println!("form: {:?}", request.form);
        let expected_form: HashMap<String, String> = [
            ("grant_type", "authorization_code"),
            ("code", "4/0Aabc"),
            ("code_verifier", session.pkce.verifier.as_str()),
            ("redirect_uri", session.redirect_uri.as_str()),
            ("client_id", "synthetic-client"),
            ("client_secret", "synthetic-secret"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
        assert_eq!(request.form, expected_form);
        assert_eq!(request.accept.as_deref(), Some("application/json"));
        assert_eq!(request.dpop, None);
    }

    #[test]
    fn pasted_errors_and_malformed_input_never_reach_the_token_endpoint() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(serde_json::json!({}))]);
        let client = synthetic_client();
        let session = SignInSession::start(&client, &endpoint.url).unwrap();
        let redirect = session.redirect_uri.clone();
        let state = session.state.clone();
        for (name, input, expected) in [
            (
                "consent denied",
                format!("{redirect}?state={state}&error=access_denied"),
                "Sign-in was canceled",
            ),
            (
                "another error",
                format!("{redirect}?state={state}&error=server_error"),
                "Google sign-in failed: server_error",
            ),
            (
                "no code",
                format!("{redirect}?state={state}"),
                "Missing authorization code",
            ),
            (
                "not a URL",
                "4/0Aabc".to_string(),
                "The pasted text is not a URL",
            ),
        ] {
            let (_sender, lines) = pasted(&[&input]);
            let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
            println!("{name}: {error}");
            assert_eq!(error.to_string(), expected, "{name}");
        }
        assert_no_token_request(&endpoint, "pasted errors");
    }

    #[test]
    fn callback_signs_in_and_tells_the_browser_after_the_exchange() {
        let client = synthetic_client();
        let mut session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(good_tokens(&session))]);
        session.token_url = endpoint.url.clone();
        // The terminal is closed: the wait must keep listening.
        let (sender, lines) = mpsc::channel::<String>();
        drop(sender);
        let browser = browser_get(
            &session,
            format!("/auth/callback?state={}&code=4%2F0Aabc", session.state),
        );
        let auth = session.wait(&lines, TEST_TIMEOUT).unwrap();
        let page = browser.join().unwrap();
        println!("browser page: {page}");
        println!("stored: {auth:?}");
        assert!(page.starts_with("HTTP/1.1 200 OK"), "{page}");
        assert!(
            page.contains("Signed in to the Google AI Ultra plan"),
            "{page}"
        );

        assert_eq!(auth.access, "synthetic-access");
        assert_eq!(auth.refresh, "synthetic-refresh");
        assert_eq!(auth.client_id, "synthetic-client");
        assert!(auth.dpop_key.is_empty());
        assert_eq!(auth.dpop_nonce, None);
        assert_eq!(auth.project_id, None);
        assert!(auth.expires_at_ms > now_ms() + 3_500_000);
        assert_eq!(endpoint.seen()[0].form["code"], "4/0Aabc");
        assert_eq!(endpoint.seen()[0].dpop, None);
    }

    #[test]
    fn callback_with_a_failed_exchange_tells_the_browser_that_sign_in_failed() {
        let endpoint = fake_token_endpoint(vec![FakeReply::error("invalid_grant", &[])]);
        let client = synthetic_client();
        let session = SignInSession::start(&client, &endpoint.url).unwrap();
        let (_sender, lines) = pasted(&[]);
        let browser = browser_get(
            &session,
            format!("/oauth2callback?code=c&state={}", session.state),
        );
        let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
        let page = browser.join().unwrap();
        println!("browser page: {page}");
        assert_eq!(
            error.to_string(),
            "Google sign-in failed: 400 invalid_grant"
        );
        assert_failed_page("failed exchange", &page);
    }

    #[test]
    fn callback_failure_page_never_carries_the_error_or_the_redirect_text() {
        let code = SYNTHETIC_CODE;
        let client = synthetic_client();
        let mut session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let verifier = session.pkce.verifier.clone();
        let echo = format!("code={code} code_verifier={verifier} client_secret=synthetic-secret");
        let endpoint = fake_token_endpoint(vec![FakeReply {
            status: 400,
            headers: Vec::new(),
            body: serde_json::json!({
                "error": "invalid_grant",
                "error_description": format!("Bad request: {echo}"),
            })
            .to_string(),
        }]);
        session.token_url = endpoint.url.clone();
        let (_sender, lines) = pasted(&[]);
        let browser = browser_get_all(
            &session,
            vec![
                // Google's redirect error, with prose that must stay off the
                // page.
                format!(
                    "/auth/callback?state={}&error=synthetic_error&error_description=synthetic-prose",
                    session.state
                ),
            ],
        );
        let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
        let [error_page]: [String; 1] = browser.join().unwrap().try_into().unwrap();
        println!("wait: {error}\nerror page: {error_page}");
        assert_eq!(error.to_string(), "Google sign-in failed: synthetic_error");
        assert_failed_page("redirect error", &error_page);
        for leaked in ["synthetic_error", "synthetic-prose"] {
            assert!(!error_page.contains(leaked), "{leaked} in {error_page}");
        }

        // A failed exchange whose reply quotes the request.
        let browser = browser_get(
            &session,
            format!(
                "/auth/callback?state={}&code=4%2F0ASynthetic",
                session.state
            ),
        );
        let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
        let exchange_page = browser.join().unwrap();
        println!("wait: {error}\nexchange page: {exchange_page}");
        assert!(
            error
                .to_string()
                .starts_with("Google sign-in failed: 400 invalid_grant"),
            "{error}"
        );
        assert_failed_page("failed exchange quoting the request", &exchange_page);
        for leaked in [
            "invalid_grant",
            "Bad request",
            "4/0ASynthetic",
            verifier.as_str(),
            "synthetic-secret",
        ] {
            assert!(
                !exchange_page.contains(leaked),
                "{leaked} in {exchange_page}"
            );
        }
    }

    #[test]
    fn exchange_errors_never_carry_the_code_the_verifier_or_the_client_secret() {
        let code = SYNTHETIC_CODE;
        let client = synthetic_client();
        let mut session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let verifier = session.pkce.verifier.clone();
        let secret = client.client_secret.clone();
        let echo = format!("code={code} code_verifier={verifier} client_secret={secret}");
        let form_echo = format!(
            "code={} code_verifier={verifier} client_secret={secret}",
            code.replace('/', "%2F")
        );
        // The verifier starts 7 characters before the 512-character cut of
        // an unparsed body, so only redaction before the cut leaves none of
        // it: `<html>` (6) + 485 + `code_verifier=` (14) is 505.
        let long_html = format!(
            "<html>{}code_verifier={verifier} client_secret={secret} code={code}</html>",
            "x".repeat(485)
        );
        let reply = |status: u16, body: String| FakeReply {
            status,
            headers: Vec::new(),
            body,
        };
        // (name, reply, text the error keeps)
        let tests = [
            (
                "JSON error quoting all three",
                reply(
                    400,
                    serde_json::json!({
                        "error": format!("invalid_request {code}"),
                        "error_description": format!("Bad request: {echo}"),
                    })
                    .to_string(),
                ),
                "Google sign-in failed: 400 invalid_request",
            ),
            (
                "JSON error quoting the code with escaped slashes",
                reply(
                    400,
                    format!(
                        r#"{{"error":"invalid_grant","error_description":"Bad code: {}"}}"#,
                        code.replace('/', r"\/")
                    ),
                ),
                "Google sign-in failed: 400 invalid_grant Bad code: [redacted]",
            ),
            (
                "text error quoting the form",
                reply(400, form_echo),
                "Google sign-in failed: 400 HTTP 400 code=[redacted]",
            ),
            (
                "page quoting all three across the cut",
                reply(502, long_html),
                "code_verifier=[redact",
            ),
            (
                "success without a refresh token quoting all three",
                reply(
                    200,
                    serde_json::json!({
                        "access_token": "synthetic-access",
                        "error_description": format!("Partial grant: {echo}"),
                    })
                    .to_string(),
                ),
                "Google sign-in failed: 200 Partial grant: code=[redacted]",
            ),
        ];
        for (name, reply, kept) in tests {
            let endpoint = fake_token_endpoint(vec![reply]);
            session.token_url = endpoint.url.clone();
            let error = session.finish(code).unwrap_err();
            let printed = format!("{error} / {error:?}");
            println!("{name}: {printed}");
            assert!(printed.contains(kept), "{name}: {printed}");
            for leaked in [
                code,
                "4%2F0ASynthetic",
                verifier.as_str(),
                secret.as_str(),
                "synthetic-access",
            ] {
                assert!(!printed.contains(leaked), "{name}: {leaked} in {printed}");
            }
        }
    }

    #[test]
    fn callback_without_the_sign_ins_state_is_turned_away_and_a_denial_with_it_ends_the_wait() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(serde_json::json!({}))]);
        let client = synthetic_client();
        let session = SignInSession::start(&client, &endpoint.url).unwrap();
        let (_sender, lines) = pasted(&[]);
        let turned_away = [
            ("another state", "code=c&state=someone-else"),
            ("no state", "code=c"),
            ("a denial with another state", "error=access_denied&state=x"),
            ("a denial without a state", "error=access_denied"),
        ];
        let mut paths: Vec<String> = turned_away
            .iter()
            .map(|(_, query)| format!("/oauth2callback?{query}"))
            .collect();
        paths.push(format!(
            "/oauth2callback?error=access_denied&state={}",
            session.state
        ));
        let browser = browser_get_all(&session, paths);
        let error = session.wait(&lines, TEST_TIMEOUT).unwrap_err();
        let pages = browser.join().unwrap();
        println!("wait: {error}\npages: {pages:#?}");
        assert_eq!(error.to_string(), "Sign-in was canceled");
        for ((name, _), page) in turned_away.iter().zip(&pages) {
            assert_turned_away(name, page);
        }
        assert_failed_page("denial with the state", &pages[turned_away.len()]);
        assert_no_token_request(&endpoint, "callback rejections");
    }

    #[test]
    fn callback_without_the_sign_ins_state_does_not_stop_the_real_callback() {
        let client = synthetic_client();
        let mut session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(good_tokens(&session))]);
        session.token_url = endpoint.url.clone();
        // The terminal is closed: only the listener can end the wait.
        let (sender, lines) = mpsc::channel::<String>();
        drop(sender);
        let turned_away = [
            (
                "a code with another state",
                "code=synthetic-other-code&state=someone-else",
            ),
            ("a denial with another state", "error=access_denied&state=x"),
            ("a code without a state", "code=synthetic-other-code"),
        ];
        let mut paths: Vec<String> = turned_away
            .iter()
            .map(|(_, query)| format!("/oauth2callback?{query}"))
            .collect();
        paths.push(format!(
            "/oauth2callback?state={}&code=4%2F0Aabc",
            session.state
        ));
        let browser = browser_get_all(&session, paths);
        let auth = session.wait(&lines, TEST_TIMEOUT).unwrap();
        let pages = browser.join().unwrap();
        println!("pages: {pages:#?}");
        for ((name, _), page) in turned_away.iter().zip(&pages) {
            assert_turned_away(name, page);
        }
        let callback_page = &pages[turned_away.len()];
        assert!(
            callback_page.starts_with("HTTP/1.1 200 OK"),
            "{callback_page}"
        );
        assert_eq!(auth.access, "synthetic-access");
        let seen = endpoint.seen();
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert_eq!(seen[0].form["code"], "4/0Aabc");
    }

    #[test]
    fn requests_for_other_paths_and_blank_lines_do_not_end_the_wait() {
        let client = synthetic_client();
        let mut session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(good_tokens(&session))]);
        session.token_url = endpoint.url.clone();
        let (_sender, lines) = pasted(&["", "   "]);
        // One browser thread, so the favicon request is answered before the
        // callback is sent.
        let browser = browser_get_all(
            &session,
            vec![
                "/favicon.ico".to_string(),
                format!("/oauth2callback?state={}&code=c", session.state),
            ],
        );
        let auth = session.wait(&lines, TEST_TIMEOUT).unwrap();
        let [favicon_page, callback_page]: [String; 2] =
            browser.join().unwrap().try_into().unwrap();
        println!("favicon: {favicon_page}\ncallback: {callback_page}");
        assert!(favicon_page.starts_with("HTTP/1.1 404"), "{favicon_page}");
        assert!(callback_page.starts_with("HTTP/1.1 200"), "{callback_page}");
        assert_eq!(auth.access, "synthetic-access");
        assert_eq!(endpoint.seen().len(), 1);
    }

    #[test]
    fn wait_times_out() {
        let client = synthetic_client();
        let session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let (_sender, lines) = pasted(&[]);
        let started = Instant::now();
        let error = session
            .wait(&lines, Duration::from_millis(200))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Timed out waiting for the Google sign-in"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn exchange_checks_the_tokens_and_the_quota_scope() {
        let client = synthetic_client();
        let mut session = SignInSession::start(&client, TOKEN_URL).unwrap();
        let tokens = good_tokens(&session);
        let with = |change: &dyn Fn(&mut serde_json::Value)| {
            let mut body = tokens.clone();
            change(&mut body);
            body
        };
        let cases = [
            (
                "no refresh token",
                with(&|body| {
                    body.as_object_mut().unwrap().remove("refresh_token");
                }),
                "Google sign-in failed: 200",
            ),
            (
                "empty access token",
                with(&|body| body["access_token"] = "".into()),
                "Google sign-in failed: 200",
            ),
            (
                "no quota scope",
                with(&|body| body["scope"] = "openid email profile".into()),
                "did not grant access to the Google AI Ultra plan",
            ),
            (
                "a scope that only starts like the quota scope",
                with(&|body| body["scope"] = format!("openid {QUOTA_SCOPE}.extra").into()),
                "did not grant access to the Google AI Ultra plan",
            ),
            (
                "no scope at all",
                with(&|body| {
                    body.as_object_mut().unwrap().remove("scope");
                }),
                "did not grant access to the Google AI Ultra plan",
            ),
            (
                "invalid ID token",
                with(&|body| {
                    body["id_token"] = "not-a-valid-jwt".into();
                }),
                "Google sign-in returned an ID token that failed validation",
            ),
        ];
        for (name, body, expected) in cases {
            let endpoint = fake_token_endpoint(vec![FakeReply::ok(body)]);
            session.token_url = endpoint.url.clone();
            let error = session.finish("c").unwrap_err();
            println!("{name}: {error}");
            assert!(error.to_string().contains(expected), "{name}: {error}");
            assert!(!error.to_string().contains("synthetic-access"), "{name}");
        }

        // Absence of an ID token succeeds (agy's oauth2 exchange does not require id_token).
        let without_id_token = with(&|body| {
            body.as_object_mut().unwrap().remove("id_token");
        });
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(without_id_token)]);
        session.token_url = endpoint.url.clone();
        let auth = session.finish("c").unwrap();
        assert_eq!(auth.access, "synthetic-access");
    }

    #[test]
    fn id_token_checks_issuer_audience_expiry_and_nonce() {
        let now = 1_795_608_000_000_u64;
        let valid = serde_json::json!({
            "iss": "https://accounts.google.com",
            "aud": "synthetic-client",
            "exp": now / 1000 + 3600,
            "nonce": "synthetic-nonce",
        });
        let with = |key: &str, value: serde_json::Value| {
            let mut claims = valid.clone();
            claims[key] = value;
            id_token(claims)
        };
        let accepted = [
            ("valid", id_token(valid.clone())),
            (
                "issuer without scheme",
                with("iss", "accounts.google.com".into()),
            ),
            (
                "expired 30 s ago, inside the leeway",
                with("exp", (now / 1000 - 30).into()),
            ),
        ];
        for (name, token) in accepted {
            check_id_token(Some(&token), "synthetic-nonce", "synthetic-client", now)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
        let rejected = [
            ("another issuer", with("iss", "https://evil.example".into())),
            ("another audience", with("aud", "other-client".into())),
            (
                "audience as a list",
                with("aud", serde_json::json!(["synthetic-client"])),
            ),
            ("another nonce", with("nonce", "replayed".into())),
            ("no nonce", with("nonce", serde_json::Value::Null)),
            ("expired", with("exp", (now / 1000 - 120).into())),
            ("no expiry", with("exp", serde_json::Value::Null)),
            ("not a JWT", "synthetic".to_string()),
            ("claims not JSON", "a.bm90IGpzb24.c".to_string()),
        ];
        for (name, token) in rejected {
            let error = check_id_token(Some(&token), "synthetic-nonce", "synthetic-client", now)
                .expect_err(name);
            println!("{name}: {error}");
            assert_eq!(
                error.to_string(),
                "Google sign-in returned an ID token that failed validation",
                "{name}"
            );
        }
        assert_eq!(
            check_id_token(None, "n", "c", now).unwrap_err().to_string(),
            "Google sign-in returned no ID token"
        );
    }

    #[test]
    fn state_comparison_is_exact() {
        assert!(constant_time_eq("state", "state"));
        assert!(!constant_time_eq("state", "statE"));
        assert!(!constant_time_eq("state", "state2"));
        assert!(!constant_time_eq("", "state"));
    }
}
