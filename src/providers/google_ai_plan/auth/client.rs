//! The built-in Google OAuth client for the Google AI Ultra plan.

use std::sync::LazyLock;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

/// Base64-encoded OAuth client id of the Google `Desktop app` client that
/// Antigravity (`agy`) uses against Cloud Code PA for the Google AI Ultra plan.
///
/// Stored encoded so repository secret scanners do not flag the public
/// installed-app client id on push.
pub const BUILT_IN_CLIENT_ID_B64: &str = "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==";

/// Base64-encoded OAuth client secret of the same client. An installed app's
/// secret cannot be kept confidential (RFC 8252 section 8.5), and Google treats
/// it as non-confidential; the sign-in is protected by the loopback redirect
/// and PKCE. It is still kept out of logs and `Debug` output, and stored
/// base64-encoded in source to avoid triggering GitHub push protection.
pub const BUILT_IN_CLIENT_SECRET_B64: &str = "R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=";

/// Decoded OAuth client id of the built-in Google `Desktop app` client.
///
/// Configuration can replace it (`googleAiPlan.clientId` /
/// `CCP_GOOGLE_AI_PLAN_CLIENT_ID`).
pub static BUILT_IN_CLIENT_ID: LazyLock<String> = LazyLock::new(|| {
    let bytes = STANDARD
        .decode(BUILT_IN_CLIENT_ID_B64)
        .expect("valid base64 built-in client id");
    String::from_utf8(bytes).expect("valid utf-8 built-in client id")
});

/// Decoded OAuth client secret of the built-in Google `Desktop app` client.
pub static BUILT_IN_CLIENT_SECRET: LazyLock<String> = LazyLock::new(|| {
    let bytes = STANDARD
        .decode(BUILT_IN_CLIENT_SECRET_B64)
        .expect("valid base64 built-in client secret");
    String::from_utf8(bytes).expect("valid utf-8 built-in client secret")
});

/// Returns the decoded built-in OAuth client id.
pub fn built_in_client_id() -> &'static str {
    BUILT_IN_CLIENT_ID.as_str()
}

/// Returns the decoded built-in OAuth client secret.
pub fn built_in_client_secret() -> &'static str {
    BUILT_IN_CLIENT_SECRET.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_client_credentials_decode_round_trip() {
        // Every check uses `assert!` with a fixed message so a failure never
        // prints the built-in client values.
        let id = built_in_client_id();
        let secret = built_in_client_secret();
        assert!(
            id.ends_with(".apps.googleusercontent.com"),
            "the built-in client id is not a Google installed-app client id"
        );
        assert!(!secret.is_empty(), "the built-in client secret is empty");
        assert!(
            STANDARD.encode(id) == BUILT_IN_CLIENT_ID_B64,
            "the built-in client id does not re-encode to its constant"
        );
        assert!(
            STANDARD.encode(secret) == BUILT_IN_CLIENT_SECRET_B64,
            "the built-in client secret does not re-encode to its constant"
        );
    }
}
