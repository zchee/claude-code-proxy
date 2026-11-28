//! The stored Google AI Ultra plan credential.

use serde::{Deserialize, Serialize};

use crate::auth::{AuthStorage, FileAuthStore};
use crate::paths;

/// The provider directory the credential is stored under.
const PROVIDER_DIR: &str = "google-ai-plan";

/// A Google AI Ultra plan sign-in.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredAuth {
    /// The access token model calls send as a Bearer token.
    pub access: String,
    /// The refresh token.
    pub refresh: String,
    /// When the access token expires, in Unix milliseconds.
    pub expires_at_ms: u64,
    /// The OAuth client the tokens were issued to.
    pub client_id: String,
    /// Optional legacy DPoP private key, as written by
    /// [`super::dpop::DpopKey::to_stored`].
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub dpop_key: String,
    /// Google's latest `DPoP-Nonce` from a refresh; kept for compatibility
    /// with existing credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpop_nonce: Option<String>,
    /// The discovered Cloud Code PA `cloudaicompanionProject` ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

impl std::fmt::Debug for StoredAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredAuth")
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires_at_ms", &self.expires_at_ms)
            .field("client_id", &self.client_id)
            .field("dpop_key", &"<redacted>")
            .field("dpop_nonce", &self.dpop_nonce.as_ref().map(|_| "<set>"))
            .field("project_id", &self.project_id)
            .finish()
    }
}

/// Loads and saves the [`StoredAuth`].
pub struct GoogleAiPlanTokenStore<S: AuthStorage<StoredAuth>> {
    store: S,
    /// A synthetic native credential, so a test can exercise the native
    /// fallback without reading the developer's `$HOME` or Keychain.
    #[cfg(test)]
    native_override: Option<StoredAuth>,
}

impl<S: AuthStorage<StoredAuth>> GoogleAiPlanTokenStore<S> {
    /// Wraps a storage backend.
    pub fn new(store: S) -> Self {
        Self {
            store,
            #[cfg(test)]
            native_override: None,
        }
    }

    /// A store whose native fallback returns `native`, for tests only.
    #[cfg(test)]
    pub fn with_native_override(store: S, native: Option<StoredAuth>) -> Self {
        Self {
            store,
            native_override: native,
        }
    }

    /// The stored sign-in, or `None` when there is none.
    ///
    /// When nothing is stored, a native Antigravity (`agy`) credential is used
    /// as a fallback, unless an explicit logout left a marker next to the
    /// credential file: that marker suppresses the fallback until a sign-in
    /// removes it, so `auth logout` stops serving a native credential without
    /// touching Antigravity's own files.
    pub fn load_auth(&self) -> anyhow::Result<Option<StoredAuth>> {
        if let Some(auth) = self.store.load()? {
            return Ok(Some(auth));
        }
        if self.is_logged_out() {
            return Ok(None);
        }
        let path = self.store.path();
        if path.ends_with("google-ai-plan/auth.json") || path.ends_with("google-ai-plan\\auth.json")
        {
            return Ok(self.native_auth());
        }
        Ok(None)
    }

    /// Replaces the stored sign-in and tries to remove any logout marker, so
    /// a fresh sign-in restores normal fallback behavior; a marker that cannot
    /// be removed does not fail the sign-in. Only an explicit sign-in calls
    /// this; a background write goes through [`Self::update_auth`].
    pub fn save_auth(&self, auth: StoredAuth) -> anyhow::Result<()> {
        self.store.save(auth)?;
        self.clear_logout_marker();
        Ok(())
    }

    /// Rewrites the sign-in [`Self::load_auth`] returns now with `update`, and
    /// reports whether anything was written. Nothing is written when there is
    /// no sign-in, as after a logout, or when `update` returns `None`, which a
    /// caller uses to drop a write meant for a superseded credential. The
    /// logout marker is left as it is.
    ///
    /// The load and the write are atomic only for writers that share one auth
    /// manager, which serialize through its refresh lock. Each manager has its
    /// own lock and there is no lock between processes, so a logout or
    /// sign-in that another process, or another manager on the same store,
    /// completes after the load and before the write is overwritten.
    pub fn update_auth(
        &self,
        update: impl FnOnce(StoredAuth) -> Option<StoredAuth>,
    ) -> anyhow::Result<bool> {
        let Some(current) = self.load_auth()? else {
            return Ok(false);
        };
        let Some(updated) = update(current) else {
            return Ok(false);
        };
        self.store.save(updated)?;
        Ok(true)
    }

    /// Removes the stored sign-in and, for a file-backed store, writes a
    /// logout marker so the native fallback stops until a sign-in removes it. The
    /// provider directory is created when it is missing, and a marker that
    /// cannot be written fails the logout: without the marker a later load
    /// would still serve the native credential.
    pub fn clear_auth(&self) -> anyhow::Result<()> {
        self.store.clear()?;
        self.write_logout_marker()
    }

    /// Where the sign-in is stored.
    pub fn auth_path(&self) -> String {
        self.store.path()
    }

    /// The native fallback credential, or the test override when one is set.
    fn native_auth(&self) -> Option<StoredAuth> {
        #[cfg(test)]
        {
            self.native_override.clone()
        }
        #[cfg(not(test))]
        {
            load_agy_native_auth()
        }
    }

    /// The logout marker path next to a file-backed `auth.json`, or `None` for
    /// a store with no such file.
    fn logout_marker(&self) -> Option<std::path::PathBuf> {
        let path = self.store.path();
        let path = std::path::Path::new(&path);
        (path.file_name()? == "auth.json").then(|| path.with_file_name("logged-out"))
    }

    fn is_logged_out(&self) -> bool {
        self.logout_marker().is_some_and(|marker| marker.exists())
    }

    fn write_logout_marker(&self) -> anyhow::Result<()> {
        let Some(marker) = self.logout_marker() else {
            return Ok(());
        };
        write_private_marker(&marker).map_err(|error| {
            anyhow::anyhow!(
                "could not write the logout marker {}: {error}",
                marker.display()
            )
        })
    }

    fn clear_logout_marker(&self) {
        if let Some(marker) = self.logout_marker() {
            let _ = std::fs::remove_file(marker);
        }
    }
}

/// Writes an empty marker file readable and writable only by the owner,
/// creating its directory (owner-only too) when it is missing.
fn write_private_marker(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(dir)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map(drop)
}

/// Attempts to load a native Antigravity (`agy`) OAuth token from
/// `$HOME/.gemini/jetski-standalone-oauth-token` or, on macOS, from the
/// macOS Keychain.
///
/// Reads nothing when `CCP_CONFIG_DIR` is set: that variable selects an
/// isolated profile, and a sign-in from outside it must not leak in.
#[cfg(not(test))]
fn load_agy_native_auth() -> Option<StoredAuth> {
    if std::env::var_os("CCP_CONFIG_DIR").is_some() {
        return None;
    }
    if let Some(home) = std::env::var_os("HOME") {
        let token_path = std::path::Path::new(&home)
            .join(".gemini")
            .join("jetski-standalone-oauth-token");
        if let Ok(raw) = std::fs::read_to_string(token_path)
            && let Some(auth) = parse_agy_native_auth(&raw)
        {
            return Some(auth);
        }
    }
    // The Keychain lookup runs through the shared 10-second-bounded runner, so
    // a locked Keychain never blocks the caller indefinitely.
    #[cfg(target_os = "macos")]
    if let Ok(output) = crate::auth::run_security(&[
        "find-generic-password",
        "-s",
        "gemini",
        "-a",
        "antigravity",
        "-w",
    ]) && output.status.success()
        && let Ok(raw) = String::from_utf8(output.stdout)
        && let Some(auth) = parse_agy_native_auth(&raw)
    {
        return Some(auth);
    }
    None
}

/// Parses an Antigravity (`agy`) OAuth token JSON object into [`StoredAuth`].
pub(crate) fn parse_agy_native_auth(raw: &str) -> Option<StoredAuth> {
    let value: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    let obj = value.as_object()?;
    let access = obj
        .get("access_token")
        .or_else(|| obj.get("access"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_string();
    let refresh = obj
        .get("refresh_token")
        .or_else(|| obj.get("refresh"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())?
        .to_string();
    let expires_at_ms = if access.is_empty() {
        0
    } else {
        obj.get("expires_at_ms")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| {
                obj.get("expiry")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|s| s.trim().parse::<jiff::Timestamp>().ok())
                    .and_then(|ts| u64::try_from(ts.as_millisecond()).ok())
            })
            .unwrap_or(0)
    };
    Some(StoredAuth {
        access,
        refresh,
        expires_at_ms,
        client_id: super::client::built_in_client_id().to_string(),
        dpop_key: String::new(),
        dpop_nonce: None,
        project_id: None,
    })
}

/// The store at `<config dir>/google-ai-plan/auth.json`. The provider is new,
/// so there is no legacy location to read from.
pub fn file_store() -> GoogleAiPlanTokenStore<FileAuthStore<StoredAuth>> {
    let path = paths::provider_auth_file(PROVIDER_DIR)
        .to_string_lossy()
        .into_owned();
    GoogleAiPlanTokenStore::new(FileAuthStore::new(path.clone(), path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth() -> StoredAuth {
        StoredAuth {
            access: "synthetic-access".to_string(),
            refresh: "synthetic-refresh".to_string(),
            expires_at_ms: 1_795_608_000_000,
            client_id: "synthetic-client".to_string(),
            dpop_key: "synthetic-key".to_string(),
            dpop_nonce: Some("synthetic-nonce".to_string()),
            project_id: None,
        }
    }

    #[test]
    fn file_store_round_trips_the_credential() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir
            .path()
            .join(PROVIDER_DIR)
            .join("auth.json")
            .to_string_lossy()
            .into_owned();
        let store = GoogleAiPlanTokenStore::new(FileAuthStore::new(path.clone(), path.clone()));
        assert_eq!(store.load_auth().unwrap(), None);

        store.save_auth(auth()).unwrap();
        assert_eq!(store.load_auth().unwrap(), Some(auth()));

        let without_nonce = StoredAuth {
            dpop_key: String::new(),
            dpop_nonce: None,
            project_id: Some("cloudaicompanion-proj".to_string()),
            ..auth()
        };
        store.save_auth(without_nonce.clone()).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("dpop_nonce"), "{raw}");
        assert!(!raw.contains("dpop_key"), "{raw}");
        assert!(raw.contains("cloudaicompanion-proj"), "{raw}");
        assert_eq!(store.load_auth().unwrap(), Some(without_nonce));

        store.clear_auth().unwrap();
        assert_eq!(store.load_auth().unwrap(), None);
        assert_eq!(store.auth_path(), path);
    }

    #[test]
    fn file_store_lives_under_the_provider_directory() {
        let path = file_store().auth_path();
        assert!(
            path.ends_with(&format!("{PROVIDER_DIR}/auth.json"))
                || path.ends_with(&format!("{PROVIDER_DIR}\\auth.json")),
            "{path}"
        );
    }

    #[test]
    fn debug_output_hides_the_tokens_and_the_key() {
        let printed = format!("{:?}", auth());
        for secret in ["synthetic-access", "synthetic-refresh", "synthetic-key"] {
            assert!(!printed.contains(secret), "{secret} in {printed}");
        }
        assert!(printed.contains("synthetic-client"), "{printed}");
    }

    #[test]
    fn parses_agy_jetski_standalone_oauth_token_format() {
        let raw = r#"{
            "access_token": "ya29.synthetic-agy-access",
            "token_type": "Bearer",
            "refresh_token": "1//synthetic-agy-refresh",
            "expiry": "2026-11-25T12:00:00Z"
        }"#;
        let parsed = parse_agy_native_auth(raw).expect("valid agy token");
        assert_eq!(parsed.access, "ya29.synthetic-agy-access");
        assert_eq!(parsed.refresh, "1//synthetic-agy-refresh");
        assert_eq!(parsed.expires_at_ms, 1_795_608_000_000);
        // `assert!` rather than `assert_eq!`: a failure must not print the
        // built-in client id.
        assert!(
            parsed.client_id == super::super::client::built_in_client_id(),
            "the parsed sign-in does not carry the built-in client id"
        );
        assert!(parsed.dpop_key.is_empty());
        assert_eq!(parsed.dpop_nonce, None);
        assert_eq!(parsed.project_id, None);

        assert_eq!(parse_agy_native_auth("{}"), None);
        assert_eq!(
            parse_agy_native_auth(r#"{"access_token":"only-access"}"#),
            None
        );
    }

    #[test]
    fn logout_marker_suppresses_native_fallback_until_the_next_sign_in() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider_dir = dir.path().join(PROVIDER_DIR);
        std::fs::create_dir_all(&provider_dir).unwrap();
        let path = provider_dir
            .join("auth.json")
            .to_string_lossy()
            .into_owned();
        let native = StoredAuth {
            access: "ya29.synthetic-native-access".to_string(),
            refresh: "1//synthetic-native-refresh".to_string(),
            expires_at_ms: 1_795_608_000_000,
            client_id: "synthetic-client".to_string(),
            dpop_key: String::new(),
            dpop_nonce: None,
            project_id: None,
        };
        let store = GoogleAiPlanTokenStore::with_native_override(
            FileAuthStore::new(path.clone(), path.clone()),
            Some(native.clone()),
        );
        let marker = provider_dir.join("logged-out");

        // No stored sign-in and no marker: the native fallback is served.
        assert_eq!(store.load_auth().unwrap(), Some(native.clone()));

        // After logout the native fallback is suppressed and a marker exists.
        store.clear_auth().unwrap();
        assert!(marker.exists(), "logout writes the marker");
        assert_eq!(store.load_auth().unwrap(), None);

        // A fresh sign-in removes the marker and restores normal loading.
        store.save_auth(auth()).unwrap();
        assert!(!marker.exists(), "sign-in removes the marker");
        assert_eq!(store.load_auth().unwrap(), Some(auth()));

        // Logging out again removes the sign-in and suppresses the fallback.
        store.clear_auth().unwrap();
        assert!(marker.exists());
        assert_eq!(store.load_auth().unwrap(), None);
    }

    fn native() -> StoredAuth {
        StoredAuth {
            access: "ya29.synthetic-native-access".to_string(),
            refresh: "1//synthetic-native-refresh".to_string(),
            expires_at_ms: 1_795_608_000_000,
            client_id: "synthetic-client".to_string(),
            dpop_key: String::new(),
            dpop_nonce: None,
            project_id: None,
        }
    }

    #[test]
    fn logout_on_a_profile_without_a_provider_directory_suppresses_the_native_fallback() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider_dir = dir.path().join(PROVIDER_DIR);
        assert!(!provider_dir.exists(), "the profile starts empty");
        let path = provider_dir
            .join("auth.json")
            .to_string_lossy()
            .into_owned();
        let store = GoogleAiPlanTokenStore::with_native_override(
            FileAuthStore::new(path.clone(), path),
            Some(native()),
        );
        assert_eq!(store.load_auth().unwrap(), Some(native()));

        store.clear_auth().unwrap();

        let marker = provider_dir.join("logged-out");
        assert!(
            marker.is_file(),
            "logout creates the directory and the marker"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |path: &std::path::Path| {
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777
            };
            assert_eq!(mode(&provider_dir), 0o700, "the directory is private");
            assert_eq!(mode(&marker), 0o600, "the marker is private");
        }
        assert_eq!(store.load_auth().unwrap(), None);
    }

    #[test]
    fn logout_fails_when_the_marker_cannot_be_written() {
        let dir = tempfile::TempDir::new().unwrap();
        let provider_dir = dir.path().join(PROVIDER_DIR);
        // A directory where the marker file belongs: removing the (absent)
        // credential succeeds, writing the marker cannot.
        std::fs::create_dir_all(provider_dir.join("logged-out").join("blocker")).unwrap();
        let path = provider_dir
            .join("auth.json")
            .to_string_lossy()
            .into_owned();
        let store = GoogleAiPlanTokenStore::with_native_override(
            FileAuthStore::new(path.clone(), path),
            Some(native()),
        );

        let error = store
            .clear_auth()
            .expect_err("a logout without its marker must not report success");
        println!("logout error: {error:#}");
        assert!(
            format!("{error:#}").contains("logout marker"),
            "the error names the marker: {error:#}"
        );
    }

    #[test]
    fn update_auth_keeps_the_marker_and_writes_nothing_without_a_sign_in() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir
            .path()
            .join(PROVIDER_DIR)
            .join("auth.json")
            .to_string_lossy()
            .into_owned();
        let marker = dir.path().join(PROVIDER_DIR).join("logged-out");
        let store = GoogleAiPlanTokenStore::with_native_override(
            FileAuthStore::new(path.clone(), path.clone()),
            Some(native()),
        );
        let raw = FileAuthStore::<StoredAuth>::new(path.clone(), path);

        // The native fallback is a sign-in, so an update of it is written.
        let refreshed = StoredAuth {
            access: "synthetic-refreshed".to_string(),
            ..native()
        };
        assert!(store.update_auth(|_| Some(refreshed.clone())).unwrap());
        assert_eq!(raw.load().unwrap(), Some(refreshed.clone()));

        // A declined update writes nothing.
        assert!(!store.update_auth(|_| None).unwrap());
        assert_eq!(raw.load().unwrap(), Some(refreshed.clone()));

        // After a logout there is no sign-in to update, and the marker stays.
        store.clear_auth().unwrap();
        assert!(!store.update_auth(|_| Some(refreshed.clone())).unwrap());
        assert_eq!(raw.load().unwrap(), None);
        assert!(marker.exists());
        assert_eq!(store.load_auth().unwrap(), None);
    }
}
