//! Access tokens for model calls, refreshed with DPoP before they expire.

use std::time::Duration;

use tokio::sync::Mutex;

use super::dpop::{DpopKey, ProofClaims};
use super::token_store::{GoogleAiPlanTokenStore, StoredAuth};
use super::{LOGIN_COMMAND, TOKEN_URL, TokenReply, access_expiry_ms, now_ms, token_form};
use crate::auth::AuthStorage;
use crate::config::GoogleAiPlanClient;
use crate::providers::vertex::sanitize::redact_secrets;

/// Refresh this long before the access token expires, as Google's docs ask.
const REFRESH_SKEW: Duration = Duration::from_secs(5 * 60);

/// Why no access token could be handed out.
#[derive(Debug, thiserror::Error)]
pub enum GoogleAiPlanAuthError {
    /// No sign-in is stored.
    #[error("Not signed in to the Google AI Ultra plan. Run `{LOGIN_COMMAND}` to sign in.")]
    NotSignedIn,
    /// The stored sign-in was issued to another client than the one in use,
    /// for example after switching between the built-in client and a
    /// configured one.
    #[error(
        "The stored Google AI Ultra plan sign-in belongs to a different OAuth client than the \
         one in use. Run `{LOGIN_COMMAND}` to sign in again."
    )]
    ClientChanged,
    /// Google says the grant is gone (`invalid_grant`, `unauthorized_client`),
    /// or the stored DPoP key cannot be read.
    #[error(
        "Your Google AI Ultra plan sign-in is no longer valid. Run `{LOGIN_COMMAND}` to sign in \
         again."
    )]
    SignInAgain,
    /// Any other refusal of the token endpoint.
    #[error("{}", describe_refresh_failure(*.status, .error.as_deref(), .description.as_deref()))]
    RefreshFailed {
        /// The HTTP status.
        status: u16,
        /// Google's `error`.
        error: Option<String>,
        /// Google's `error_description`.
        description: Option<String>,
    },
    /// The token endpoint could not be reached.
    #[error("Google token request failed: {0}")]
    Transport(#[source] reqwest::Error),
    /// The credential could not be read or written.
    #[error("Google AI Ultra plan credential store failed: {0:#}")]
    Store(anyhow::Error),
}

fn describe_refresh_failure(status: u16, error: Option<&str>, description: Option<&str>) -> String {
    let mut out = format!("Google token refresh failed: {status}");
    for part in [error, description].into_iter().flatten() {
        if !part.is_empty() {
            out.push(' ');
            out.push_str(part);
        }
    }
    out
}

/// Hands out the stored access token and refreshes it when it is about to
/// expire.
pub struct GoogleAiPlanAuthManager<S: AuthStorage<StoredAuth>> {
    store: GoogleAiPlanTokenStore<S>,
    client: GoogleAiPlanClient,
    token_url: String,
    http: reqwest::Client,
    refresh_lock: Mutex<()>,
}

impl<S: AuthStorage<StoredAuth>> GoogleAiPlanAuthManager<S> {
    /// A manager that refreshes at Google's token endpoint with `client`,
    /// the OAuth client in use.
    pub fn new(
        store: GoogleAiPlanTokenStore<S>,
        client: GoogleAiPlanClient,
    ) -> Result<Self, GoogleAiPlanAuthError> {
        Self::with_token_url(store, client, TOKEN_URL)
    }

    /// A manager that refreshes at `token_url`.
    pub fn with_token_url(
        store: GoogleAiPlanTokenStore<S>,
        client: GoogleAiPlanClient,
        token_url: impl Into<String>,
    ) -> Result<Self, GoogleAiPlanAuthError> {
        Ok(Self {
            store,
            client,
            token_url: token_url.into(),
            http: super::token_http_client().map_err(GoogleAiPlanAuthError::Transport)?,
            refresh_lock: Mutex::new(()),
        })
    }

    /// The credential store.
    pub fn store(&self) -> &GoogleAiPlanTokenStore<S> {
        &self.store
    }

    /// The access token for a model call: the stored one while it is more
    /// than five minutes from expiry, otherwise a refreshed one. Concurrent
    /// callers share one refresh.
    pub async fn get_access_token(&self) -> Result<String, GoogleAiPlanAuthError> {
        let (auth, _) = self.usable_auth()?;
        if is_fresh(&auth) {
            return Ok(auth.access);
        }
        let _guard = self.refresh_lock.lock().await;
        // Another caller may have refreshed while this one waited.
        let (auth, client) = self.usable_auth()?;
        if is_fresh(&auth) {
            return Ok(auth.access);
        }
        let updated = refresh(&self.http, &self.token_url, client, &auth).await?;
        self.store
            .save_auth(updated.clone())
            .map_err(GoogleAiPlanAuthError::Store)?;
        Ok(updated.access)
    }

    /// The stored sign-in and the client in use. The client is checked even
    /// while the token is fresh, so a configuration change shows at once
    /// rather than at the next refresh.
    fn usable_auth(&self) -> Result<(StoredAuth, &GoogleAiPlanClient), GoogleAiPlanAuthError> {
        let auth = self
            .store
            .load_auth()
            .map_err(GoogleAiPlanAuthError::Store)?
            .ok_or(GoogleAiPlanAuthError::NotSignedIn)?;
        if auth.client_id != self.client.client_id {
            return Err(GoogleAiPlanAuthError::ClientChanged);
        }
        Ok((auth, &self.client))
    }
}

fn is_fresh(auth: &StoredAuth) -> bool {
    auth.expires_at_ms > now_ms().saturating_add(REFRESH_SKEW.as_millis() as u64)
}

/// Refreshes `auth` at `token_url` and returns the updated credential.
///
/// When the reply carries a non-empty `refresh_token`, it replaces the one
/// that was sent; otherwise, as Google normally replies, the sent one stays.
/// OAuth (RFC 6749 section 6) lets the server issue a new refresh token on a
/// refresh and revoke the old one, and a client that kept the old one would
/// get `invalid_grant` at the next refresh and make the user sign in again.
/// The DPoP key and the client id stay as they are.
async fn refresh(
    http: &reqwest::Client,
    token_url: &str,
    client: &GoogleAiPlanClient,
    auth: &StoredAuth,
) -> Result<StoredAuth, GoogleAiPlanAuthError> {
    let key =
        DpopKey::from_stored(&auth.dpop_key).map_err(|_| GoogleAiPlanAuthError::SignInAgain)?;
    let mut claims = ProofClaims {
        nonce: auth.dpop_nonce.clone(),
        ..ProofClaims::random()
    };
    let mut reply = post_refresh(http, token_url, client, &auth.refresh, &key, &claims).await?;
    // Two rejections are fixed by retrying once each, in either order. A
    // stale nonce (and none at all after sign-in) gets `use_dpop_nonce` with
    // a fresh nonce. A clock more than 10 s ahead of Google's gets
    // `invalid_dpop_proof`; the retry takes the proof time from the
    // response's `Date` header.
    for _ in 0..2 {
        let error = reply.body.error.as_deref();
        if let (Some("use_dpop_nonce"), Some(nonce)) = (error, reply.nonce.as_ref())
            && claims.nonce.as_ref() != Some(nonce)
        {
            claims.nonce = Some(nonce.clone());
        } else if let (Some("invalid_dpop_proof"), None, Some(server_time)) =
            (error, claims.iat, reply.server_time)
        {
            claims.iat = Some(server_time);
        } else {
            break;
        }
        claims.jti = super::dpop::random_jti();
        reply = post_refresh(http, token_url, client, &auth.refresh, &key, &claims).await?;
    }
    let Some(access) = reply
        .body
        .access_token
        .clone()
        .filter(|token| reply.status == 200 && !token.is_empty())
    else {
        // Google's docs name these two as final: the grant expired or was
        // revoked, or the client is no longer allowed for this token.
        if matches!(
            reply.body.error.as_deref(),
            Some("invalid_grant" | "unauthorized_client")
        ) {
            return Err(GoogleAiPlanAuthError::SignInAgain);
        }
        // Google's text may quote what was sent; the refresh token and the
        // client secret must not reach the error, the log or the client.
        let secrets = [auth.refresh.as_str(), client.client_secret.as_str()];
        let clean = |text: Option<String>| text.map(|text| redact_secrets(&text, &secrets));
        return Err(GoogleAiPlanAuthError::RefreshFailed {
            status: reply.status,
            error: clean(reply.body.error),
            description: clean(reply.body.error_description),
        });
    };
    let refresh_token = reply
        .body
        .refresh_token
        .clone()
        .filter(|token| !token.is_empty())
        .unwrap_or_else(|| auth.refresh.clone());
    Ok(StoredAuth {
        access,
        refresh: refresh_token,
        expires_at_ms: access_expiry_ms(&reply.body),
        // Keep Google's newest nonce, else the one this request just used.
        dpop_nonce: reply.nonce.or(claims.nonce),
        ..auth.clone()
    })
}

async fn post_refresh(
    http: &reqwest::Client,
    token_url: &str,
    client: &GoogleAiPlanClient,
    refresh_token: &str,
    key: &DpopKey,
    claims: &ProofClaims,
) -> Result<TokenReply, GoogleAiPlanAuthError> {
    let proof = key
        .proof(token_url, claims)
        .map_err(|_| GoogleAiPlanAuthError::SignInAgain)?;
    let form = token_form(
        client,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
        ],
    );
    let response = http
        .post(token_url)
        .header(http::header::ACCEPT, "application/json")
        .header("DPoP", proof)
        .form(&form)
        .send()
        .await
        .map_err(GoogleAiPlanAuthError::Transport)?;
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let text = response
        .text()
        .await
        .map_err(GoogleAiPlanAuthError::Transport)?;
    // An error body may quote the request, and part of it can reach the
    // error text, cut to a length; the secrets are removed before that cut.
    // A success body is left whole: it carries the tokens to store.
    let text = if (200..300).contains(&status) {
        text
    } else {
        redact_secrets(&text, &[refresh_token, client.client_secret.as_str()])
    };
    Ok(TokenReply::from_parts(status, &headers, &text))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::test_support::{
        FakeReply, SERVER_DATE, SERVER_TIME, fake_token_endpoint, synthetic_client,
    };
    use super::*;
    use crate::auth::InMemoryAuthStore;

    fn tokens() -> serde_json::Value {
        serde_json::json!({ "access_token": "new", "expires_in": 3599 })
    }

    /// A signed-in credential whose access token has expired.
    fn expired_auth() -> StoredAuth {
        StoredAuth {
            access: "old".to_string(),
            refresh: "refresh-token".to_string(),
            expires_at_ms: 0,
            client_id: synthetic_client().client_id,
            dpop_key: DpopKey::generate().unwrap().to_stored(),
            dpop_nonce: None,
        }
    }

    fn manager(
        auth: Option<StoredAuth>,
        client: GoogleAiPlanClient,
        token_url: &str,
    ) -> GoogleAiPlanAuthManager<InMemoryAuthStore<StoredAuth>> {
        let store = GoogleAiPlanTokenStore::new(InMemoryAuthStore::new());
        if let Some(auth) = auth {
            store.save_auth(auth).unwrap();
        }
        GoogleAiPlanAuthManager::with_token_url(store, client, token_url).unwrap()
    }

    async fn refresh_with(
        auth: StoredAuth,
        replies: Vec<FakeReply>,
    ) -> (
        Result<StoredAuth, GoogleAiPlanAuthError>,
        Vec<super::super::test_support::SeenTokenRequest>,
    ) {
        let endpoint = fake_token_endpoint(replies);
        let http = super::super::token_http_client().unwrap();
        let result = refresh(&http, &endpoint.url, &synthetic_client(), &auth).await;
        let seen = endpoint.seen();
        for (n, request) in seen.iter().enumerate() {
            println!("request {n}: proof claims {}", request.proof_claims());
        }
        (result, seen)
    }

    #[tokio::test]
    async fn refresh_sends_the_client_the_refresh_token_and_a_proof_for_the_called_url() {
        let (result, seen) = refresh_with(
            expired_auth(),
            vec![FakeReply::ok(tokens()).with_headers(&[("dpop-nonce", "n1")])],
        )
        .await;
        let next = result.unwrap();
        assert_eq!(seen.len(), 1);
        let request = &seen[0];
        assert_eq!(request.path, "/token");
        assert_eq!(request.form["grant_type"], "refresh_token");
        assert_eq!(request.form["refresh_token"], "refresh-token");
        assert_eq!(request.form["client_id"], "synthetic-client");
        assert_eq!(request.form["client_secret"], "synthetic-secret");
        assert_eq!(request.accept.as_deref(), Some("application/json"));
        assert_eq!(
            request.content_type.as_deref(),
            Some("application/x-www-form-urlencoded")
        );
        let claims = request.proof_claims();
        assert_eq!(claims["htm"], "POST");
        assert!(
            claims["htu"].as_str().unwrap().ends_with("/token")
                && claims["htu"]
                    .as_str()
                    .unwrap()
                    .starts_with("http://127.0.0.1:"),
            "{claims}"
        );
        assert!(claims.get("nonce").is_none(), "{claims}");
        assert_eq!(next.access, "new");
        assert_eq!(
            next.refresh, "refresh-token",
            "the reply has no new refresh token, so the sent one stays"
        );
        assert_eq!(next.dpop_nonce.as_deref(), Some("n1"));
        assert!(next.expires_at_ms > now_ms() + 3_500_000);
    }

    #[tokio::test]
    async fn refresh_retries_once_with_googles_time_when_the_local_clock_is_ahead() {
        let (result, seen) = refresh_with(
            expired_auth(),
            vec![
                FakeReply::error("invalid_dpop_proof", &[("date", SERVER_DATE)]),
                FakeReply::ok(tokens()).with_headers(&[("dpop-nonce", "n1")]),
            ],
        )
        .await;
        let next = result.unwrap();
        assert_eq!(seen.len(), 2);
        let (first, second) = (seen[0].proof_claims(), seen[1].proof_claims());
        assert_eq!(second["iat"], SERVER_TIME);
        assert_ne!(first["jti"], second["jti"]);
        assert_eq!(next.access, "new");
        assert_eq!(next.dpop_nonce.as_deref(), Some("n1"));
    }

    #[tokio::test]
    async fn refresh_handles_a_nonce_rejection_and_a_clock_rejection_in_one_refresh() {
        let (result, seen) = refresh_with(
            expired_auth(),
            vec![
                FakeReply::error(
                    "use_dpop_nonce",
                    &[("dpop-nonce", "n1"), ("date", SERVER_DATE)],
                ),
                FakeReply::error("invalid_dpop_proof", &[("date", SERVER_DATE)]),
                FakeReply::ok(tokens()),
            ],
        )
        .await;
        let next = result.unwrap();
        let nonces: Vec<_> = seen
            .iter()
            .map(|request| request.proof_claims().get("nonce").cloned())
            .collect();
        assert_eq!(
            nonces,
            [None, Some("n1".into()), Some("n1".into())],
            "nonce per request"
        );
        assert_eq!(seen[2].proof_claims()["iat"], SERVER_TIME);
        assert_eq!(next.access, "new");
    }

    #[tokio::test]
    async fn refresh_keeps_the_nonce_that_worked_when_the_success_sends_none() {
        let auth = StoredAuth {
            dpop_nonce: Some("n1".to_string()),
            ..expired_auth()
        };
        let (result, seen) = refresh_with(
            auth,
            vec![
                FakeReply::error("use_dpop_nonce", &[("dpop-nonce", "n2")]),
                FakeReply::ok(tokens()),
            ],
        )
        .await;
        assert_eq!(seen[0].proof_claims()["nonce"], "n1", "stored nonce used");
        assert_eq!(result.unwrap().dpop_nonce.as_deref(), Some("n2"));
    }

    #[tokio::test]
    async fn refresh_does_not_retry_a_nonce_rejection_that_repeats_the_same_nonce() {
        let auth = StoredAuth {
            dpop_nonce: Some("n1".to_string()),
            ..expired_auth()
        };
        let (result, seen) = refresh_with(
            auth,
            vec![FakeReply::error("use_dpop_nonce", &[("dpop-nonce", "n1")])],
        )
        .await;
        assert_eq!(seen.len(), 1);
        assert_eq!(
            result.unwrap_err().to_string(),
            "Google token refresh failed: 400 use_dpop_nonce"
        );
    }

    #[tokio::test]
    async fn refresh_asks_to_sign_in_again_on_invalid_grant_and_unauthorized_client() {
        for error in ["invalid_grant", "unauthorized_client"] {
            let (result, seen) =
                refresh_with(expired_auth(), vec![FakeReply::error(error, &[])]).await;
            let error_value = result.unwrap_err();
            println!("{error}: {error_value}");
            assert!(
                matches!(error_value, GoogleAiPlanAuthError::SignInAgain),
                "{error}: {error_value:?}"
            );
            assert!(
                error_value.to_string().contains(
                    "Run `claude-code-proxy google-ai-plan auth login` to sign in again."
                ),
                "{error}: {error_value}"
            );
            assert_eq!(seen.len(), 1, "{error}");
        }
    }

    #[tokio::test]
    async fn refresh_gives_up_after_one_clock_retry() {
        let (result, seen) = refresh_with(
            expired_auth(),
            vec![FakeReply::error(
                "invalid_dpop_proof",
                &[("date", SERVER_DATE)],
            )],
        )
        .await;
        let error = result.unwrap_err();
        assert_eq!(seen.len(), 2);
        assert_eq!(
            error.to_string(),
            "Google token refresh failed: 400 invalid_dpop_proof"
        );
        assert!(matches!(
            error,
            GoogleAiPlanAuthError::RefreshFailed { status: 400, .. }
        ));
    }

    #[tokio::test]
    async fn refresh_makes_at_most_three_requests() {
        // A nonce rejection with a new nonce each time, after a clock fix.
        let (result, seen) = refresh_with(
            expired_auth(),
            vec![
                FakeReply::error("invalid_dpop_proof", &[("date", SERVER_DATE)]),
                FakeReply::error("use_dpop_nonce", &[("dpop-nonce", "n1")]),
                FakeReply::error("use_dpop_nonce", &[("dpop-nonce", "n2")]),
            ],
        )
        .await;
        assert_eq!(seen.len(), 3);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn refresh_failure_reports_the_status_and_googles_description() {
        let (result, _) = refresh_with(
            expired_auth(),
            vec![FakeReply {
                status: 500,
                headers: Vec::new(),
                body: r#"{"error":"internal_failure","error_description":"Try later"}"#.to_string(),
            }],
        )
        .await;
        assert_eq!(
            result.unwrap_err().to_string(),
            "Google token refresh failed: 500 internal_failure Try later"
        );
    }

    #[tokio::test]
    async fn refresh_with_an_unreadable_stored_key_asks_to_sign_in_again_without_a_request() {
        let auth = StoredAuth {
            dpop_key: "not a key".to_string(),
            ..expired_auth()
        };
        let (result, seen) = refresh_with(auth, vec![FakeReply::ok(tokens())]).await;
        assert!(matches!(
            result.unwrap_err(),
            GoogleAiPlanAuthError::SignInAgain
        ));
        assert!(seen.is_empty());
    }

    #[tokio::test]
    async fn fresh_token_is_returned_without_a_request() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let auth = StoredAuth {
            access: "still-fresh".to_string(),
            expires_at_ms: now_ms() + 10 * 60 * 1000,
            ..expired_auth()
        };
        let manager = manager(Some(auth), synthetic_client(), &endpoint.url);
        assert_eq!(manager.get_access_token().await.unwrap(), "still-fresh");
        assert!(endpoint.seen().is_empty());
    }

    #[tokio::test]
    async fn token_within_five_minutes_of_expiry_is_refreshed_and_saved() {
        let endpoint = fake_token_endpoint(vec![
            FakeReply::ok(tokens()).with_headers(&[("dpop-nonce", "n1")]),
        ]);
        let auth = StoredAuth {
            expires_at_ms: now_ms() + 4 * 60 * 1000,
            ..expired_auth()
        };
        let manager = manager(Some(auth.clone()), synthetic_client(), &endpoint.url);
        assert_eq!(manager.get_access_token().await.unwrap(), "new");
        let saved = manager.store().load_auth().unwrap().unwrap();
        assert_eq!(saved.access, "new");
        assert_eq!(saved.dpop_nonce.as_deref(), Some("n1"));
        assert_eq!(saved.dpop_key, auth.dpop_key);
        assert_eq!(saved.refresh, auth.refresh);
        assert_eq!(endpoint.seen().len(), 1);
    }

    #[tokio::test]
    async fn a_new_refresh_token_in_the_reply_replaces_the_stored_one_and_is_sent_next() {
        // The first reply rotates the refresh token and gives an access token
        // that is already due, so the second call refreshes again.
        let endpoint = fake_token_endpoint(vec![
            FakeReply::ok(serde_json::json!({
                "access_token": "new",
                "refresh_token": "synthetic-rotated-refresh",
                "expires_in": 0,
            })),
            FakeReply::ok(serde_json::json!({ "access_token": "newer", "expires_in": 3599 })),
        ]);
        let auth = expired_auth();
        let manager = manager(Some(auth.clone()), synthetic_client(), &endpoint.url);

        assert_eq!(manager.get_access_token().await.unwrap(), "new");
        let saved = manager.store().load_auth().unwrap().unwrap();
        assert_eq!(saved.refresh, "synthetic-rotated-refresh");
        assert_eq!(saved.dpop_key, auth.dpop_key);
        assert_eq!(saved.client_id, auth.client_id);

        assert_eq!(manager.get_access_token().await.unwrap(), "newer");
        let seen = endpoint.seen();
        let sent: Vec<&str> = seen
            .iter()
            .map(|request| request.form["refresh_token"].as_str())
            .collect();
        println!("refresh tokens sent: {sent:?}");
        assert_eq!(sent, ["refresh-token", "synthetic-rotated-refresh"]);
        let saved = manager.store().load_auth().unwrap().unwrap();
        assert_eq!(
            saved.refresh, "synthetic-rotated-refresh",
            "the second reply has no refresh token, so the rotated one stays"
        );
        assert_eq!(saved.access, "newer");
    }

    #[tokio::test]
    async fn a_reply_without_a_usable_refresh_token_leaves_the_stored_one_unchanged() {
        for (name, body) in [
            (
                "no refresh token",
                serde_json::json!({ "access_token": "new", "expires_in": 3599 }),
            ),
            (
                "an empty refresh token",
                serde_json::json!({
                    "access_token": "new",
                    "refresh_token": "",
                    "expires_in": 3599,
                }),
            ),
        ] {
            let endpoint = fake_token_endpoint(vec![FakeReply::ok(body)]);
            let manager = manager(Some(expired_auth()), synthetic_client(), &endpoint.url);
            assert_eq!(manager.get_access_token().await.unwrap(), "new", "{name}");
            let saved = manager.store().load_auth().unwrap().unwrap();
            println!("{name}: stored refresh token {:?}", saved.refresh);
            assert_eq!(saved.refresh, "refresh-token", "{name}");
            assert_eq!(endpoint.seen().len(), 1, "{name}");
        }
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_refresh() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let manager = Arc::new(manager(
            Some(expired_auth()),
            synthetic_client(),
            &endpoint.url,
        ));
        let calls = (0..8).map(|_| {
            let manager = manager.clone();
            tokio::spawn(async move { manager.get_access_token().await })
        });
        for call in futures_util::future::join_all(calls).await {
            assert_eq!(call.unwrap().unwrap(), "new");
        }
        assert_eq!(endpoint.seen().len(), 1, "one token request");
    }

    #[tokio::test]
    async fn manager_errors_distinguish_the_missing_pieces() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let other_client = GoogleAiPlanClient {
            client_id: "other-client".to_string(),
            client_secret: "other-secret".to_string(),
        };
        let fresh = StoredAuth {
            expires_at_ms: now_ms() + 60 * 60 * 1000,
            ..expired_auth()
        };
        let cases: [(&str, Option<StoredAuth>, GoogleAiPlanClient, &str); 3] = [
            ("not signed in", None, synthetic_client(), "Not signed in"),
            (
                "issued to another configured client",
                Some(fresh.clone()),
                other_client,
                "different OAuth client",
            ),
            (
                "issued to a configured client, the built-in one now in use",
                Some(fresh),
                GoogleAiPlanClient::built_in(),
                "different OAuth client",
            ),
        ];
        for (name, auth, client, expected) in cases {
            let error = manager(auth, client, &endpoint.url)
                .get_access_token()
                .await
                .unwrap_err();
            println!("{name}: {error}");
            assert!(error.to_string().contains(expected), "{name}: {error}");
            assert!(error.to_string().contains(LOGIN_COMMAND), "{name}: {error}");
        }
        assert!(endpoint.seen().is_empty());
    }

    #[tokio::test]
    async fn refresh_errors_never_carry_the_refresh_token_or_the_client_secret() {
        let refresh = expired_auth().refresh;
        let secret = synthetic_client().client_secret;
        let echo = format!("refresh_token={refresh} client_secret={secret}");
        // The refresh token starts 7 characters before the 512-character
        // cut of an unparsed body, so only redaction before the cut leaves
        // none of it.
        let long_html = format!("<html>{}{echo}</html>", "x".repeat(485));
        // (name, reply, text the error keeps)
        let tests = [
            (
                "JSON error quoting both",
                FakeReply {
                    status: 400,
                    headers: Vec::new(),
                    body: serde_json::json!({
                        "error": format!("invalid_request {refresh}"),
                        "error_description": format!("Bad request: {echo}"),
                    })
                    .to_string(),
                },
                "Bad request: refresh_token=[redacted] client_secret=[redacted]",
            ),
            (
                "page quoting both across the cut",
                FakeReply {
                    status: 502,
                    headers: Vec::new(),
                    body: long_html,
                },
                "refresh_token=[redact",
            ),
        ];
        for (name, reply, kept) in tests {
            let endpoint = fake_token_endpoint(vec![reply]);
            let manager = manager(Some(expired_auth()), synthetic_client(), &endpoint.url);
            let error = manager.get_access_token().await.unwrap_err();
            assert!(
                matches!(error, GoogleAiPlanAuthError::RefreshFailed { .. }),
                "{name}: {error:?}"
            );
            let printed = format!("{error} / {error:?}");
            println!("{name}: {printed}");
            assert!(printed.contains(kept), "{name}: {printed}");
            for leaked in [refresh.as_str(), secret.as_str()] {
                assert!(!printed.contains(leaked), "{name}: {leaked} in {printed}");
            }
        }
    }
}
