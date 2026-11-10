use serde::{Deserialize, Serialize};

use crate::auth::{AuthStorage, FileAuthStore};
use crate::paths;

/// The proxy's own Muse login. The minted API key, the account e-mail
/// address and the tier id are deliberately not stored: the key is minted
/// again in memory by every process, and the other two are not needed.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredAuth {
    pub access_token: String,
    pub obtained_at_ms: u64,
    pub client_id: String,
    pub oauth_host: String,
    #[serde(default)]
    pub user_full_name: Option<String>,
    pub is_subs_active: bool,
    #[serde(default)]
    pub subs_tier_name: Option<String>,
}

impl std::fmt::Debug for StoredAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredAuth")
            .field("access_token", &"[redacted]")
            .field("obtained_at_ms", &self.obtained_at_ms)
            .field("client_id", &self.client_id)
            .field("oauth_host", &self.oauth_host)
            .field("user_full_name", &self.user_full_name)
            .field("is_subs_active", &self.is_subs_active)
            .field("subs_tier_name", &self.subs_tier_name)
            .finish()
    }
}

pub struct MuseTokenStore<S: AuthStorage<StoredAuth>> {
    store: S,
}

impl<S: AuthStorage<StoredAuth>> MuseTokenStore<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }
    pub fn load_auth(&self) -> anyhow::Result<Option<StoredAuth>> {
        self.store.load()
    }
    pub fn save_auth(&self, auth: StoredAuth) -> anyhow::Result<()> {
        self.store.save(auth)
    }
    pub fn clear_auth(&self) -> anyhow::Result<()> {
        self.store.clear()
    }
    pub fn auth_path(&self) -> String {
        self.store.path()
    }
}

pub type FileTokenStore = MuseTokenStore<FileAuthStore<StoredAuth>>;

/// Store at `<configuration-root>/muse/auth.json`. Muse never had a legacy
/// location, so the primary path doubles as the legacy one and no other file
/// is consulted or removed.
pub fn file_store() -> FileTokenStore {
    let primary = paths::provider_auth_file("muse")
        .to_string_lossy()
        .into_owned();
    store_at(primary)
}

pub fn store_at(path: String) -> FileTokenStore {
    MuseTokenStore::new(FileAuthStore::new(path.clone(), path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StoredAuth {
        StoredAuth {
            access_token: "synthetic-access-token-value".into(),
            obtained_at_ms: 1_794_274_189_000,
            client_id: crate::providers::muse::auth::constants::CLIENT_ID.into(),
            oauth_host: "https://auth.meta.com".into(),
            user_full_name: Some("Synthetic Name".into()),
            is_subs_active: true,
            subs_tier_name: Some("Muse Code Max Usage".into()),
        }
    }

    #[test]
    fn stored_auth_round_trips_through_the_file_store() {
        let temp = tempfile::TempDir::new().unwrap();
        let store = store_at(
            temp.path()
                .join("muse")
                .join("auth.json")
                .to_string_lossy()
                .into_owned(),
        );
        assert!(store.load_auth().unwrap().is_none());
        store.save_auth(sample()).unwrap();
        assert_eq!(store.load_auth().unwrap(), Some(sample()));
        store.clear_auth().unwrap();
        assert!(store.load_auth().unwrap().is_none());
        store.clear_auth().unwrap();
    }

    #[test]
    fn stored_auth_rejects_unknown_fields() {
        let mut value = serde_json::to_value(sample()).unwrap();
        value["api_key"] = serde_json::json!("synthetic-api-key");
        let error = serde_json::from_value::<StoredAuth>(value).unwrap_err();
        assert!(error.to_string().contains("api_key"), "{error}");
    }

    #[test]
    fn stored_file_holds_no_key_email_or_tier_id() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("auth.json");
        let store = store_at(path.to_string_lossy().into_owned());
        store.save_auth(sample()).unwrap();
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut keys: Vec<_> = raw.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "access_token",
                "client_id",
                "is_subs_active",
                "oauth_host",
                "obtained_at_ms",
                "subs_tier_name",
                "user_full_name",
            ]
        );
    }

    #[test]
    fn debug_output_hides_the_access_token() {
        let rendered = format!("{:?}", sample());
        assert!(
            !rendered.contains("synthetic-access-token-value"),
            "{rendered}"
        );
        assert!(rendered.contains("[redacted]"));
    }

    #[cfg(unix)]
    #[test]
    fn saved_file_is_owner_read_write_only() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("muse").join("auth.json");
        let store = store_at(path.to_string_lossy().into_owned());
        store.save_auth(sample()).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
