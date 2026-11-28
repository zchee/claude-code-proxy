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
    /// The refresh token, bound to [`StoredAuth::dpop_key`].
    pub refresh: String,
    /// When the access token expires, in Unix milliseconds.
    pub expires_at_ms: u64,
    /// The OAuth client the tokens were issued to.
    pub client_id: String,
    /// The DPoP private key, as written by
    /// [`super::dpop::DpopKey::to_stored`].
    pub dpop_key: String,
    /// Google's latest `DPoP-Nonce` from a refresh; saves a
    /// `use_dpop_nonce` round trip on the next one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpop_nonce: Option<String>,
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
            .finish()
    }
}

/// Loads and saves the [`StoredAuth`].
pub struct GoogleAiPlanTokenStore<S: AuthStorage<StoredAuth>> {
    store: S,
}

impl<S: AuthStorage<StoredAuth>> GoogleAiPlanTokenStore<S> {
    /// Wraps a storage backend.
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// The stored sign-in, or `None` when there is none.
    pub fn load_auth(&self) -> anyhow::Result<Option<StoredAuth>> {
        self.store.load()
    }

    /// Replaces the stored sign-in.
    pub fn save_auth(&self, auth: StoredAuth) -> anyhow::Result<()> {
        self.store.save(auth)
    }

    /// Removes the stored sign-in.
    pub fn clear_auth(&self) -> anyhow::Result<()> {
        self.store.clear()
    }

    /// Where the sign-in is stored.
    pub fn auth_path(&self) -> String {
        self.store.path()
    }
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
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        println!(
            "stored credential members: {:?}",
            written.as_object().unwrap().keys()
        );
        assert_eq!(store.load_auth().unwrap(), Some(auth()));

        let without_nonce = StoredAuth {
            dpop_nonce: None,
            ..auth()
        };
        store.save_auth(without_nonce.clone()).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("dpop_nonce"), "{raw}");
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
}
