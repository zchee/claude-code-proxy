//! Access tokens for model calls, refreshed before they expire.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use super::token_store::{GoogleAiPlanTokenStore, StoredAuth};
use super::{LOGIN_COMMAND, TOKEN_URL, TokenReply, access_expiry_ms, now_ms, token_form};
use crate::auth::AuthStorage;
use crate::config::GoogleAiPlanClient;
use crate::providers::vertex::sanitize::redact_secrets;

/// Refresh this long before the access token expires, as Google's docs ask.
const REFRESH_SKEW: Duration = Duration::from_secs(5 * 60);

/// How many times one request refreshes before it gives up because another
/// writer keeps replacing the stored sign-in while each refresh is in flight.
const MAX_REFRESH_ROUNDS: usize = 3;

/// One consistent read of the stored sign-in: a fresh access token, the
/// persisted project ID, and a key that names the credential without exposing
/// the tokens, so a caller can discover and persist a project against the
/// exact sign-in it read.
#[derive(Clone, PartialEq, Eq)]
pub struct AuthSnapshot {
    /// A fresh access token for a model call.
    pub access_token: String,
    /// The persisted `cloudaicompanionProject` ID, if any.
    pub project_id: Option<String>,
    /// A SHA-256 digest of the OAuth client id and refresh token, in hex. It
    /// changes whenever either does, so it identifies the credential across a
    /// refresh or an account switch without ever carrying a secret.
    pub credential_key: String,
}

impl std::fmt::Debug for AuthSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthSnapshot")
            .field("access_token", &"<redacted>")
            .field("project_id", &self.project_id)
            .field("credential_key", &self.credential_key)
            .finish()
    }
}

/// A stable key for a credential: the SHA-256 digest of `client_id` and the
/// refresh token, which never leaves this process and holds no secret.
fn credential_key(client_id: &str, refresh: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(client_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(refresh.as_bytes());
    let digest = hasher.finalize();
    let mut key = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(key, "{byte:02x}");
    }
    key
}

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
    /// Google says the grant is gone (`invalid_grant`, `unauthorized_client`).
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
    /// Shared with the blocking pool, where every storage call runs.
    store: Arc<GoogleAiPlanTokenStore<S>>,
    client: GoogleAiPlanClient,
    token_url: String,
    http: reqwest::Client,
    /// Serializes the refresh and the project-id write of every caller of
    /// this manager. Another manager, in this process or another, has its
    /// own lock and is not serialized with this one.
    refresh_lock: Arc<Mutex<()>>,
}

impl<S: AuthStorage<StoredAuth> + 'static> GoogleAiPlanAuthManager<S> {
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
            store: Arc::new(store),
            client,
            token_url: token_url.into(),
            http: super::token_http_client().map_err(GoogleAiPlanAuthError::Transport)?,
            refresh_lock: Arc::new(Mutex::new(())),
        })
    }

    /// The credential store.
    pub fn store(&self) -> &GoogleAiPlanTokenStore<S> {
        &self.store
    }

    /// The stored `cloudaicompanionProject` ID, if one has been persisted.
    pub fn project_id(&self) -> Option<String> {
        self.store
            .load_auth()
            .ok()
            .flatten()
            .and_then(|auth| auth.project_id)
            .filter(|id| !id.is_empty())
    }

    /// Best-effort persistence of a discovered `cloudaicompanionProject` ID
    /// alongside the stored credential, but only while the stored credential
    /// is still the one `expected_key` named when the project was discovered.
    ///
    /// The key check drops the write when the credential rotated, the account
    /// was switched or the user logged out in between, so a project
    /// discovered for a superseded sign-in is never written back over the new
    /// tokens and never removes the logout marker. It is rediscovered on the
    /// next request instead. The refresh lock serializes this with a refresh
    /// through the same manager; for writers through another manager, in
    /// this process or another, the check and the write are not atomic, so a
    /// sign-in or logout completed in that window can still be overwritten
    /// (see [`GoogleAiPlanTokenStore::update_auth`]).
    pub async fn set_project_id(&self, expected_key: &str, project_id: String) {
        if project_id.is_empty() {
            return;
        }
        let guard = Arc::clone(&self.refresh_lock).lock_owned().await;
        let expected_key = expected_key.to_string();
        let result = self
            .with_store(guard, move |store| {
                store.update_auth(|mut auth| {
                    if credential_key(&auth.client_id, &auth.refresh) != expected_key
                        || auth.project_id.as_deref() == Some(project_id.as_str())
                    {
                        return None;
                    }
                    auth.project_id = Some(project_id);
                    Some(auth)
                })
            })
            .await;
        if let Err(error) = result {
            tracing::warn!("google-ai-plan: could not persist the project id: {error:#}");
        }
    }

    /// The access token for a model call: the stored one while it is more
    /// than five minutes from expiry, otherwise a refreshed one. Concurrent
    /// callers share one refresh.
    pub async fn get_access_token(&self) -> Result<String, GoogleAiPlanAuthError> {
        Ok(self.fresh_auth().await?.access)
    }

    /// A fresh access token together with the stored project ID and a key for
    /// the exact credential it read, in one consistent read. The refresh lock
    /// is released before this returns, so a later [`Self::set_project_id`]
    /// can take it without deadlocking.
    pub async fn get_auth_snapshot(&self) -> Result<AuthSnapshot, GoogleAiPlanAuthError> {
        let auth = self.fresh_auth().await?;
        Ok(AuthSnapshot {
            credential_key: credential_key(&auth.client_id, &auth.refresh),
            project_id: auth.project_id.clone().filter(|id| !id.is_empty()),
            access_token: auth.access,
        })
    }

    /// The usable stored sign-in, refreshed when it is within five minutes of
    /// expiry. Concurrent callers share one refresh.
    ///
    /// The refreshed credential is written only while the stored record is
    /// still, field for field, the one the refresh started from. Any
    /// difference means another writer replaced it while the refresh waited on
    /// the network (a logout, a sign-in, a project id saved through another
    /// manager), so nothing is written and the request decides again from
    /// what the store holds now: no sign-in after a logout, the stored record
    /// while it is fresh, otherwise another refresh of it. When the record
    /// changes during every one of [`MAX_REFRESH_ROUNDS`] refreshes and the
    /// last one left is not fresh either, the request fails with a store error
    /// and can be retried. Between managers
    /// the comparison and the write are not atomic (see
    /// [`GoogleAiPlanTokenStore::update_auth`]).
    async fn fresh_auth(&self) -> Result<StoredAuth, GoogleAiPlanAuthError> {
        let auth = self.usable_auth().await?;
        if is_fresh(&auth) {
            return Ok(auth);
        }
        for _ in 0..MAX_REFRESH_ROUNDS {
            let guard = Arc::clone(&self.refresh_lock).lock_owned().await;
            // Another caller may have refreshed while this one waited.
            let auth = self.usable_auth().await?;
            if is_fresh(&auth) {
                return Ok(auth);
            }
            let updated = refresh(&self.http, &self.token_url, &self.client, &auth).await?;
            let written = updated.clone();
            let saved = self
                .with_store(guard, move |store| {
                    store.update_auth(|current| (current == auth).then_some(written))
                })
                .await?;
            if saved {
                return Ok(updated);
            }
        }
        // The last declined save still gets the normal decision, short of
        // another refresh: a fresh record or a logout is answered as such.
        let auth = self.usable_auth().await?;
        if is_fresh(&auth) {
            return Ok(auth);
        }
        Err(GoogleAiPlanAuthError::Store(anyhow::anyhow!(
            "the stored sign-in was replaced during each of {MAX_REFRESH_ROUNDS} token refreshes; \
             try the request again"
        )))
    }

    /// The stored sign-in, checked against the client in use. The client is
    /// checked even while the token is fresh, so a configuration change shows
    /// at once rather than at the next refresh.
    async fn usable_auth(&self) -> Result<StoredAuth, GoogleAiPlanAuthError> {
        let auth = self
            .with_store((), |store| store.load_auth())
            .await?
            .ok_or(GoogleAiPlanAuthError::NotSignedIn)?;
        if auth.client_id != self.client.client_id {
            return Err(GoogleAiPlanAuthError::ClientChanged);
        }
        // A credential from an older DPoP-bound sign-in cannot be refreshed on
        // this DPoP-less path; ask the user to sign in again rather than let
        // every refresh fail. Cloud Code PA and native `agy` tokens have an
        // empty `dpop_key`, so their behavior is unchanged.
        if !auth.dpop_key.is_empty() {
            return Err(GoogleAiPlanAuthError::SignInAgain);
        }
        Ok(auth)
    }

    /// Runs `call` against the store on the blocking pool: a load can wait on
    /// the macOS Keychain and a write on the disk, and neither may stall the
    /// async runtime that serves other requests. `held` (a lock guard, or `()`)
    /// is dropped only after `call` returns, so a caller that is cancelled
    /// while the call runs does not release its lock early.
    async fn with_store<T, G>(
        &self,
        held: G,
        call: impl FnOnce(&GoogleAiPlanTokenStore<S>) -> anyhow::Result<T> + Send + 'static,
    ) -> Result<T, GoogleAiPlanAuthError>
    where
        T: Send + 'static,
        G: Send + 'static,
    {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || {
            let result = call(&store);
            drop(held);
            result
        })
        .await
        .map_err(|error| {
            GoogleAiPlanAuthError::Store(anyhow::anyhow!(
                "the credential store call failed: {error}"
            ))
        })?
        .map_err(GoogleAiPlanAuthError::Store)
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
async fn refresh(
    http: &reqwest::Client,
    token_url: &str,
    client: &GoogleAiPlanClient,
    auth: &StoredAuth,
) -> Result<StoredAuth, GoogleAiPlanAuthError> {
    let reply = post_refresh(http, token_url, client, &auth.refresh).await?;
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
        client_id: auth.client_id.clone(),
        // The DPoP-less path never carries these forward; a legacy DPoP
        // credential is caught before refresh and prompts a fresh sign-in.
        dpop_key: String::new(),
        dpop_nonce: None,
        project_id: auth.project_id.clone(),
    })
}

async fn post_refresh(
    http: &reqwest::Client,
    token_url: &str,
    client: &GoogleAiPlanClient,
    refresh_token: &str,
) -> Result<TokenReply, GoogleAiPlanAuthError> {
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
        .form(&form)
        .send()
        .await
        .map_err(GoogleAiPlanAuthError::Transport)?;
    let status = response.status().as_u16();
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
    Ok(TokenReply::from_parts(status, &text))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{FakeReply, fake_token_endpoint, synthetic_client};
    use super::*;
    use crate::auth::{FileAuthStore, InMemoryAuthStore};

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
            dpop_key: String::new(),
            dpop_nonce: None,
            project_id: None,
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
        (result, seen)
    }

    #[tokio::test]
    async fn refresh_sends_the_client_and_the_refresh_token_without_dpop() {
        let mut initial = expired_auth();
        initial.project_id = Some("cloudaicompanion-proj".to_string());
        let (result, seen) = refresh_with(initial, vec![FakeReply::ok(tokens())]).await;
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
        assert_eq!(request.dpop, None);
        assert_eq!(next.access, "new");
        assert_eq!(
            next.refresh, "refresh-token",
            "the reply has no new refresh token, so the sent one stays"
        );
        assert_eq!(next.project_id.as_deref(), Some("cloudaicompanion-proj"));
        assert!(next.expires_at_ms > now_ms() + 3_500_000);
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
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let auth = StoredAuth {
            expires_at_ms: now_ms() + 4 * 60 * 1000,
            ..expired_auth()
        };
        let manager = manager(Some(auth.clone()), synthetic_client(), &endpoint.url);
        assert_eq!(manager.get_access_token().await.unwrap(), "new");
        let saved = manager.store().load_auth().unwrap().unwrap();
        assert_eq!(saved.access, "new");
        assert_eq!(saved.refresh, auth.refresh);
        assert_eq!(endpoint.seen().len(), 1);
    }

    #[tokio::test]
    async fn set_project_id_persists_the_project_into_stored_auth() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let auth = expired_auth();
        let key = credential_key(&auth.client_id, &auth.refresh);
        let manager = manager(Some(auth), synthetic_client(), &endpoint.url);
        assert_eq!(manager.project_id(), None);
        manager
            .set_project_id(&key, "discovered-project-123".to_string())
            .await;
        assert_eq!(
            manager.project_id().as_deref(),
            Some("discovered-project-123")
        );
        let saved = manager.store().load_auth().unwrap().unwrap();
        assert_eq!(saved.project_id.as_deref(), Some("discovered-project-123"));
    }

    #[tokio::test]
    async fn an_auth_snapshot_reads_the_token_project_and_a_stable_key() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let auth = StoredAuth {
            expires_at_ms: now_ms() + 60 * 60 * 1000,
            project_id: Some("project-a".to_string()),
            ..expired_auth()
        };
        let manager = manager(Some(auth.clone()), synthetic_client(), &endpoint.url);
        let snapshot = manager.get_auth_snapshot().await.unwrap();
        assert_eq!(snapshot.access_token, auth.access);
        assert_eq!(snapshot.project_id.as_deref(), Some("project-a"));
        assert_eq!(
            snapshot.credential_key,
            credential_key(&auth.client_id, &auth.refresh)
        );
        // No refresh was needed, so the token endpoint stayed untouched.
        assert!(endpoint.seen().is_empty());
    }

    #[tokio::test]
    async fn a_legacy_dpop_credential_asks_for_a_fresh_sign_in_without_a_refresh() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let legacy = StoredAuth {
            expires_at_ms: 0,
            dpop_key: "legacy-dpop-key-material".to_string(),
            ..expired_auth()
        };
        let manager = manager(Some(legacy), synthetic_client(), &endpoint.url);
        assert!(matches!(
            manager.get_access_token().await,
            Err(GoogleAiPlanAuthError::SignInAgain)
        ));
        assert!(
            endpoint.seen().is_empty(),
            "a legacy credential must not reach the token endpoint"
        );
    }

    #[tokio::test]
    async fn a_rotated_credential_is_not_clobbered_by_a_stale_set_project_id() {
        // A refresh rotates the refresh token; a project discovered against the
        // old credential must not be written back over the rotated one.
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(serde_json::json!({
            "access_token": "new",
            "refresh_token": "synthetic-rotated-refresh",
            "expires_in": 3599,
        }))]);
        let auth = expired_auth();
        let stale_key = credential_key(&auth.client_id, &auth.refresh);
        let manager = manager(Some(auth), synthetic_client(), &endpoint.url);
        // Force the refresh, which rotates the refresh token in the store.
        assert_eq!(manager.get_access_token().await.unwrap(), "new");
        manager
            .set_project_id(&stale_key, "discovered-project-123".to_string())
            .await;
        let saved = manager.store().load_auth().unwrap().unwrap();
        assert_eq!(saved.refresh, "synthetic-rotated-refresh");
        assert_eq!(
            saved.project_id, None,
            "a project discovered for the old credential is not persisted"
        );
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

    /// How long a test waits for a step of an interleaving before it fails,
    /// so a regression that skips the step fails by name instead of hanging.
    /// Generous, because tests run in parallel on a loaded machine.
    const STEP_TIMEOUT: Duration = Duration::from_secs(20);

    /// Awaits `step`, failing the test with `what` when it takes longer than
    /// [`STEP_TIMEOUT`].
    async fn within<F: std::future::Future>(what: &str, step: F) -> F::Output {
        match tokio::time::timeout(STEP_TIMEOUT, step).await {
            Ok(output) => output,
            Err(_) => panic!("timed out after {STEP_TIMEOUT:?} waiting for {what}"),
        }
    }

    /// A token endpoint that announces each request, with its form body, on
    /// the returned receiver and answers it with `reply` only after the
    /// returned `Notify` is notified, so a test can act while a refresh waits
    /// on the network.
    async fn paused_token_endpoint(
        reply: serde_json::Value,
    ) -> (
        String,
        tokio::sync::mpsc::UnboundedReceiver<String>,
        Arc<tokio::sync::Notify>,
    ) {
        let (arrived_tx, arrived_rx) = tokio::sync::mpsc::unbounded_channel();
        let release = Arc::new(tokio::sync::Notify::new());
        let held = release.clone();
        let app = axum::Router::new().route(
            "/token",
            axum::routing::post(move |form: String| {
                let arrived_tx = arrived_tx.clone();
                let held = held.clone();
                let reply = reply.clone();
                async move {
                    let _ = arrived_tx.send(form);
                    held.notified().await;
                    axum::Json(reply)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/token", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, arrived_rx, release)
    }

    /// An expiry no test outlives (2100-01-01), fixed so that two calls of a
    /// fixture compare equal.
    const FAR_FUTURE_MS: u64 = 4_102_444_800_000;

    /// A native credential the fallback serves when nothing is stored.
    fn native_auth() -> StoredAuth {
        StoredAuth {
            access: "synthetic-native-access".to_string(),
            refresh: "synthetic-native-refresh".to_string(),
            expires_at_ms: FAR_FUTURE_MS,
            ..expired_auth()
        }
    }

    /// A sign-in completed after the refresh under test started.
    fn newer_auth() -> StoredAuth {
        StoredAuth {
            access: "synthetic-newer-access".to_string(),
            refresh: "synthetic-newer-refresh".to_string(),
            expires_at_ms: FAR_FUTURE_MS,
            ..expired_auth()
        }
    }

    /// A file-backed profile under a temporary directory. Every store opened
    /// on it stands for a separate process (the server, the CLI) sharing the
    /// same `auth.json`, and serves `native` as the native fallback.
    struct Profile {
        _dir: tempfile::TempDir,
        path: String,
        marker: std::path::PathBuf,
        native: Option<StoredAuth>,
    }

    impl Profile {
        fn new(native: Option<StoredAuth>) -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let provider_dir = dir.path().join("google-ai-plan");
            Self {
                path: provider_dir
                    .join("auth.json")
                    .to_string_lossy()
                    .into_owned(),
                marker: provider_dir.join("logged-out"),
                _dir: dir,
                native,
            }
        }

        fn store(&self) -> GoogleAiPlanTokenStore<FileAuthStore<StoredAuth>> {
            GoogleAiPlanTokenStore::with_native_override(
                FileAuthStore::new(self.path.clone(), self.path.clone()),
                self.native.clone(),
            )
        }

        /// What a load of `auth.json` alone returns, without the fallback.
        fn stored_file(&self) -> Option<StoredAuth> {
            FileAuthStore::new(self.path.clone(), self.path.clone())
                .load()
                .unwrap()
        }

        fn manager(
            &self,
            token_url: &str,
        ) -> Arc<GoogleAiPlanAuthManager<FileAuthStore<StoredAuth>>> {
            Arc::new(
                GoogleAiPlanAuthManager::with_token_url(
                    self.store(),
                    synthetic_client(),
                    token_url,
                )
                .unwrap(),
            )
        }
    }

    #[tokio::test]
    async fn a_logout_completed_during_a_refresh_is_not_undone_by_it() {
        let profile = Profile::new(Some(native_auth()));
        profile.store().save_auth(expired_auth()).unwrap();
        let (url, mut arrived, release) = paused_token_endpoint(tokens()).await;
        let manager = profile.manager(&url);
        let refreshing = tokio::spawn({
            let manager = manager.clone();
            async move { manager.get_access_token().await }
        });
        within("the refresh to reach the token endpoint", arrived.recv())
            .await
            .expect("the refresh reaches the token endpoint");

        // Another process logs out while the refresh waits on the network.
        profile.store().clear_auth().unwrap();
        release.notify_one();
        let result = within("the refresh to finish", refreshing).await.unwrap();
        println!("refresh after the logout: {result:?}");

        assert_eq!(
            profile.stored_file(),
            None,
            "the refresh wrote nothing back"
        );
        assert!(profile.marker.exists(), "the logout marker is still there");
        assert_eq!(
            profile.store().load_auth().unwrap(),
            None,
            "a later load returns no sign-in although a native one is available"
        );
        assert!(
            matches!(result, Err(GoogleAiPlanAuthError::NotSignedIn)),
            "the in-flight request sees the logout: {result:?}"
        );
    }

    #[tokio::test]
    async fn a_sign_in_completed_during_a_refresh_is_not_overwritten_by_it() {
        // (name, the record another process writes while the refresh waits)
        let tests = [
            ("a sign-in with another refresh token", newer_auth()),
            (
                // Same client and refresh token, so only the rest of the
                // record shows that someone wrote in between.
                "a newer record that kept the refresh token",
                StoredAuth {
                    access: "synthetic-newer-access".to_string(),
                    expires_at_ms: FAR_FUTURE_MS,
                    project_id: Some("synthetic-newer-project".to_string()),
                    ..expired_auth()
                },
            ),
        ];
        for (name, newer) in tests {
            let profile = Profile::new(Some(native_auth()));
            profile.store().save_auth(expired_auth()).unwrap();
            let (url, mut arrived, release) = paused_token_endpoint(tokens()).await;
            let manager = profile.manager(&url);
            let refreshing = tokio::spawn({
                let manager = manager.clone();
                async move { manager.get_access_token().await }
            });
            within("the refresh to reach the token endpoint", arrived.recv())
                .await
                .expect("the refresh reaches the token endpoint");

            // Another process writes a newer record while the refresh waits.
            profile.store().save_auth(newer.clone()).unwrap();
            release.notify_one();
            let result = within("the refresh to finish", refreshing).await.unwrap();
            let stored = profile.store().load_auth().unwrap();
            println!("{name}: request got {result:?}, store holds {stored:?}");

            let stored = stored.expect("a sign-in is stored");
            assert_eq!(stored.access, newer.access, "{name}: access token");
            assert_eq!(stored.expires_at_ms, newer.expires_at_ms, "{name}: expiry");
            assert_eq!(stored.project_id, newer.project_id, "{name}: project id");
            assert_eq!(stored, newer, "{name}: the newer record is still stored");
            assert_eq!(
                result.unwrap(),
                newer.access,
                "{name}: the in-flight request gets the newer record"
            );
            assert!(
                arrived.try_recv().is_err(),
                "{name}: the fresh newer record needs no second refresh"
            );
        }
    }

    #[tokio::test]
    async fn a_refresh_only_record_written_during_a_refresh_is_refreshed_not_returned() {
        let profile = Profile::new(Some(native_auth()));
        profile.store().save_auth(expired_auth()).unwrap();
        let (url, mut arrived, release) = paused_token_endpoint(tokens()).await;
        let manager = profile.manager(&url);
        let mut refreshing = tokio::spawn({
            let manager = manager.clone();
            async move { manager.get_access_token().await }
        });
        within("the refresh to reach the token endpoint", arrived.recv())
            .await
            .expect("the refresh reaches the token endpoint");

        // Another process stores a record that carries only a refresh token,
        // the shape a native credential without an access token has.
        let replacement = StoredAuth {
            access: String::new(),
            refresh: "synthetic-replacement-refresh".to_string(),
            expires_at_ms: 0,
            ..expired_auth()
        };
        profile.store().save_auth(replacement.clone()).unwrap();
        release.notify_one();

        // The request must refresh the replacement rather than hand out its
        // empty access token.
        let second = within("the replacement to be refreshed", async {
            tokio::select! {
                form = arrived.recv() => form.expect("the token endpoint is running"),
                result = &mut refreshing => panic!(
                    "the request returned without refreshing the replacement: {result:?}"
                ),
            }
        })
        .await;
        println!("second token request: {second}");
        assert!(
            second.contains("refresh_token=synthetic-replacement-refresh"),
            "the second refresh sends the replacement's refresh token: {second}"
        );
        release.notify_one();
        let result = within("the refresh to finish", refreshing).await.unwrap();
        println!("request after the replacement: {result:?}");

        assert_eq!(
            result.unwrap(),
            "new",
            "a refreshed access token is returned"
        );
        let stored = profile.stored_file().expect("the refresh is written");
        assert_eq!(stored.refresh, replacement.refresh);
        assert_eq!(stored.access, "new");
        assert!(is_fresh(&stored), "{stored:?}");
        assert!(arrived.try_recv().is_err(), "two token requests in all");
    }

    #[tokio::test]
    async fn a_record_that_changes_during_every_refresh_ends_in_a_retry_unless_the_last_is_fresh() {
        // (name, whether the record written during the last round is fresh)
        let tests = [
            ("an expired last record", false),
            ("a fresh last record", true),
        ];
        for (name, last_is_fresh) in tests {
            let profile = Profile::new(Some(native_auth()));
            profile.store().save_auth(expired_auth()).unwrap();
            let (url, mut arrived, release) = paused_token_endpoint(tokens()).await;
            let manager = profile.manager(&url);
            let refreshing = tokio::spawn({
                let manager = manager.clone();
                async move { manager.get_access_token().await }
            });

            // Each round, another process writes a different record while the
            // refresh waits on the network. The documented maximum is written
            // out here rather than read from the constant, so raising the
            // constant makes this test fail instead of following it.
            let rounds = 3;
            let mut last = expired_auth();
            for round in 0..rounds {
                let form = within("the next refresh round", arrived.recv())
                    .await
                    .expect("the token endpoint is running");
                println!("{name}: round {round}: {form}");
                assert!(
                    form.contains(&format!("refresh_token={}", last.refresh)),
                    "{name}: round {round} refreshes the record it read: {form}"
                );
                last = StoredAuth {
                    refresh: format!("synthetic-rewritten-refresh-{round}"),
                    ..expired_auth()
                };
                if last_is_fresh && round + 1 == rounds {
                    last.access = "synthetic-last-access".to_string();
                    last.expires_at_ms = FAR_FUTURE_MS;
                }
                profile.store().save_auth(last.clone()).unwrap();
                release.notify_one();
            }
            let result = within("the refresh rounds to end", refreshing)
                .await
                .unwrap();
            println!("{name}: request after {rounds} changed rounds: {result:?}");

            if last_is_fresh {
                assert_eq!(
                    result.unwrap(),
                    "synthetic-last-access",
                    "{name}: the fresh record is returned"
                );
            } else {
                let error = result.unwrap_err();
                assert!(
                    matches!(error, GoogleAiPlanAuthError::Store(_)),
                    "{name}: the request fails with the store error the server answers with a \
                     500: {error:?}"
                );
                assert!(
                    error.to_string().contains("try the request again"),
                    "{name}: {error}"
                );
            }
            assert!(
                arrived.try_recv().is_err(),
                "{name}: the rounds are bounded"
            );
            assert_eq!(
                profile.store().load_auth().unwrap(),
                Some(last),
                "{name}: the last record another process wrote is kept"
            );
        }
    }

    #[tokio::test]
    async fn a_project_id_save_after_a_logout_or_a_newer_sign_in_writes_nothing() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let signed_in = StoredAuth {
            expires_at_ms: now_ms() + 60 * 60 * 1000,
            ..expired_auth()
        };
        // (name, what another process does while discovery runs, the
        // sign-in a later load returns, whether the marker exists)
        type Interleave = fn(&GoogleAiPlanTokenStore<FileAuthStore<StoredAuth>>);
        let tests: [(&str, Interleave, Option<StoredAuth>, bool); 2] = [
            ("logout", |store| store.clear_auth().unwrap(), None, true),
            (
                "newer sign-in",
                |store| store.save_auth(newer_auth()).unwrap(),
                Some(newer_auth()),
                false,
            ),
        ];
        for (name, interleave, expected, marker) in tests {
            let profile = Profile::new(Some(native_auth()));
            profile.store().save_auth(signed_in.clone()).unwrap();
            let manager = profile.manager(&endpoint.url);
            let snapshot = manager.get_auth_snapshot().await.unwrap();

            interleave(&profile.store());
            manager
                .set_project_id(&snapshot.credential_key, "discovered-project".to_string())
                .await;

            let loaded = profile.store().load_auth().unwrap();
            println!("{name}: later load {loaded:?}");
            assert_eq!(loaded, expected, "{name}");
            assert_eq!(profile.marker.exists(), marker, "{name}");
        }
        assert!(endpoint.seen().is_empty());
    }

    #[tokio::test]
    async fn a_refreshed_native_credential_is_stored_for_the_next_request() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let profile = Profile::new(Some(StoredAuth {
            expires_at_ms: 0,
            ..native_auth()
        }));
        let manager = profile.manager(&endpoint.url);

        assert_eq!(manager.get_access_token().await.unwrap(), "new");
        let stored = profile.stored_file().expect("the refresh is written");
        assert_eq!(stored.access, "new");
        assert_eq!(stored.refresh, native_auth().refresh);
        assert_eq!(manager.get_access_token().await.unwrap(), "new");
        assert_eq!(endpoint.seen().len(), 1, "the stored refresh is reused");
    }

    /// A store that records the thread of every call.
    struct ThreadRecordingStore {
        inner: InMemoryAuthStore<StoredAuth>,
        threads: Arc<std::sync::Mutex<Vec<std::thread::ThreadId>>>,
    }

    impl ThreadRecordingStore {
        fn record(&self) {
            self.threads
                .lock()
                .unwrap()
                .push(std::thread::current().id());
        }
    }

    impl AuthStorage<StoredAuth> for ThreadRecordingStore {
        fn load(&self) -> anyhow::Result<Option<StoredAuth>> {
            self.record();
            self.inner.load()
        }

        fn save(&self, value: StoredAuth) -> anyhow::Result<()> {
            self.record();
            self.inner.save(value)
        }

        fn clear(&self) -> anyhow::Result<()> {
            self.record();
            self.inner.clear()
        }

        fn path(&self) -> String {
            self.inner.path()
        }
    }

    #[tokio::test]
    async fn storage_calls_run_off_the_async_runtime_thread() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let inner = InMemoryAuthStore::new();
        inner.save(expired_auth()).unwrap();
        let threads = Arc::default();
        let store = GoogleAiPlanTokenStore::new(ThreadRecordingStore {
            inner,
            threads: Arc::clone(&threads),
        });
        let manager =
            GoogleAiPlanAuthManager::with_token_url(store, synthetic_client(), &endpoint.url)
                .unwrap();

        let snapshot = manager.get_auth_snapshot().await.unwrap();
        manager
            .set_project_id(&snapshot.credential_key, "discovered-project".to_string())
            .await;

        let runtime_thread = std::thread::current().id();
        let threads = threads.lock().unwrap().clone();
        println!("runtime thread {runtime_thread:?}, storage threads {threads:?}");
        // Two loads before the refresh, the refresh's load and write, then
        // the project's load and write.
        assert_eq!(threads.len(), 6, "{threads:?}");
        assert!(
            threads.iter().all(|thread| *thread != runtime_thread),
            "a storage call ran on the async runtime thread: {threads:?}"
        );
        assert_eq!(manager.project_id().as_deref(), Some("discovered-project"));
    }

    /// A store whose next `save` waits, after announcing itself, until the
    /// test releases it.
    struct GatedStore {
        inner: InMemoryAuthStore<StoredAuth>,
        gate: std::sync::Mutex<Option<SaveGate>>,
    }

    struct SaveGate {
        arrived: tokio::sync::oneshot::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    }

    impl AuthStorage<StoredAuth> for GatedStore {
        fn load(&self) -> anyhow::Result<Option<StoredAuth>> {
            self.inner.load()
        }

        fn save(&self, value: StoredAuth) -> anyhow::Result<()> {
            let gate = self.gate.lock().unwrap().take();
            if let Some(gate) = gate {
                let _ = gate.arrived.send(());
                gate.release
                    .recv_timeout(STEP_TIMEOUT)
                    .expect("the test releases the gated save");
            }
            self.inner.save(value)
        }

        fn clear(&self) -> anyhow::Result<()> {
            self.inner.clear()
        }

        fn path(&self) -> String {
            self.inner.path()
        }
    }

    /// A manager over a [`GatedStore`] holding `auth`, with the receiver that
    /// announces the gated save and the sender that releases it.
    fn gated_manager(
        auth: StoredAuth,
        token_url: &str,
    ) -> (
        Arc<GoogleAiPlanAuthManager<GatedStore>>,
        tokio::sync::oneshot::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let inner = InMemoryAuthStore::new();
        inner.save(auth).unwrap();
        let (arrived_tx, arrived_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let store = GoogleAiPlanTokenStore::new(GatedStore {
            inner,
            gate: std::sync::Mutex::new(Some(SaveGate {
                arrived: arrived_tx,
                release: release_rx,
            })),
        });
        let manager = Arc::new(
            GoogleAiPlanAuthManager::with_token_url(store, synthetic_client(), token_url).unwrap(),
        );
        (manager, arrived_rx, release_tx)
    }

    #[tokio::test]
    async fn a_project_id_save_holds_the_refresh_lock_from_its_read_to_its_write() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(serde_json::json!({
            "access_token": "new",
            "refresh_token": "synthetic-rotated-refresh",
            "expires_in": 3599,
        }))]);
        let auth = expired_auth();
        let old_key = credential_key(&auth.client_id, &auth.refresh);
        let (manager, arrived, release) = gated_manager(auth, &endpoint.url);

        // The setter has read the credential and is paused in its write.
        let setting = tokio::spawn({
            let manager = manager.clone();
            async move {
                manager
                    .set_project_id(&old_key, "discovered-project-123".to_string())
                    .await;
            }
        });
        within("the setter to reach its write", arrived)
            .await
            .expect("the setter reaches its write");
        assert!(
            manager.refresh_lock.try_lock().is_err(),
            "the refresh lock is held between the setter's read and its write"
        );

        // A refresh that rotates the refresh token starts now and has to wait
        // for the setter to finish.
        let refreshing = tokio::spawn({
            let manager = manager.clone();
            async move { manager.get_access_token().await }
        });
        release.send(()).unwrap();
        within("the setter to finish", setting).await.unwrap();
        assert_eq!(
            within("the refresh to finish", refreshing)
                .await
                .unwrap()
                .unwrap(),
            "new"
        );

        let saved = manager.store().load_auth().unwrap().unwrap();
        println!(
            "after both: refresh rotated {}, project {:?}",
            saved.refresh == "synthetic-rotated-refresh",
            saved.project_id
        );
        assert_eq!(
            saved.refresh, "synthetic-rotated-refresh",
            "the rotated refresh token is not clobbered"
        );
        assert_eq!(
            saved.project_id.as_deref(),
            Some("discovered-project-123"),
            "the refresh read the credential after the setter wrote it"
        );
        assert_eq!(endpoint.seen().len(), 1);
    }

    #[tokio::test]
    async fn a_cancelled_project_id_save_keeps_the_refresh_lock_until_its_write_returns() {
        let endpoint = fake_token_endpoint(vec![FakeReply::ok(tokens())]);
        let auth = expired_auth();
        let key = credential_key(&auth.client_id, &auth.refresh);
        let (manager, arrived, release) = gated_manager(auth, &endpoint.url);

        let setting = tokio::spawn({
            let manager = manager.clone();
            async move {
                manager
                    .set_project_id(&key, "discovered-project-123".to_string())
                    .await;
            }
        });
        within("the setter to reach its write", arrived)
            .await
            .expect("the setter reaches its write");

        // The request that called the setter goes away while its write runs.
        setting.abort();
        let joined = within("the cancelled setter to finish", setting).await;
        println!("cancelled setter: {joined:?}");
        assert!(
            joined.is_err_and(|error| error.is_cancelled()),
            "the setter was cancelled during its write"
        );
        assert!(
            manager.refresh_lock.try_lock().is_err(),
            "the refresh lock stays held while the cancelled caller's write still runs"
        );

        release.send(()).unwrap();
        drop(
            within(
                "the write to release the refresh lock",
                manager.refresh_lock.lock(),
            )
            .await,
        );
        let saved = manager.store().load_auth().unwrap().unwrap();
        println!("after the write: project {:?}", saved.project_id);
        assert_eq!(
            saved.project_id.as_deref(),
            Some("discovered-project-123"),
            "the write of the cancelled caller completes"
        );
        assert!(endpoint.seen().is_empty());
    }

    #[test]
    fn debug_output_of_a_snapshot_hides_the_access_token() {
        let snapshot = AuthSnapshot {
            access_token: "ya29.synthetic-snapshot-access".to_string(),
            project_id: Some("project-a".to_string()),
            credential_key: credential_key("synthetic-client", "synthetic-refresh"),
        };
        let printed = format!("{snapshot:?} / {snapshot:#?}");
        assert!(
            !printed.contains("synthetic-snapshot-access"),
            "the access token is printed: {printed}"
        );
        assert!(printed.contains("project-a"), "{printed}");
    }
}
