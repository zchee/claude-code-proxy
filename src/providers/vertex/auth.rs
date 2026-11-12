//! Google credentials for Vertex AI, found with Application Default
//! Credentials (ADC). The proxy stores nothing: `gcloud`, a service account
//! key file or the metadata server owns the credential, and the
//! `google-cloud-auth` crate reads it, refreshes the access token and builds
//! the request headers.

use std::sync::Once;
use std::time::Duration;

use google_cloud_auth::credentials::{Builder, CacheableResource, Credentials};
use http::{HeaderMap, StatusCode};

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::providers::muse::sanitize::sanitize_upstream_text;

/// OAuth scope that covers every Vertex AI method.
pub const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// The command that creates user credentials in the ADC location.
pub const LOGIN_COMMAND: &str = "gcloud auth application-default login";

/// What a user does when no usable Google credential is found.
pub const LOGIN_GUIDANCE: &str = "Run `gcloud auth application-default login`, or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file";

/// Longest wait for an access token. Refreshing one is a single call to the
/// Google token endpoint or the metadata server; a host that never answers
/// must not hold the request or `auth status` forever.
pub const TOKEN_TIMEOUT: Duration = Duration::from_secs(30);

/// Why no Google request headers could be produced. Every variant reaches
/// clients as 401 `authentication_error` with [`LOGIN_GUIDANCE`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VertexAuthError {
    /// The credential could not be loaded: the file named by
    /// `GOOGLE_APPLICATION_CREDENTIALS` is missing or malformed, or its
    /// `type` is not supported.
    Unavailable(String),
    /// The credential loaded but no access token could be obtained.
    Token(String),
}

impl VertexAuthError {
    /// The reason without the guidance.
    pub fn reason(&self) -> &str {
        match self {
            VertexAuthError::Unavailable(reason) | VertexAuthError::Token(reason) => reason,
        }
    }

    /// 401 `authentication_error` naming the reason and what to run.
    pub fn into_provider_error(self) -> ProviderError {
        ProviderError::new(
            StatusCode::UNAUTHORIZED,
            ProviderErrorKind::Authentication,
            self.to_string(),
        )
    }
}

impl std::fmt::Display for VertexAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VertexAuthError::Unavailable(reason) => {
                write!(
                    f,
                    "Google credentials are not available: {reason}. {LOGIN_GUIDANCE}"
                )
            }
            VertexAuthError::Token(reason) => {
                write!(
                    f,
                    "Google access token could not be obtained: {reason}. {LOGIN_GUIDANCE}"
                )
            }
        }
    }
}

impl std::error::Error for VertexAuthError {}

impl From<VertexAuthError> for ProviderError {
    fn from(error: VertexAuthError) -> Self {
        error.into_provider_error()
    }
}

/// Source of the headers that authenticate a Vertex AI request.
#[derive(Clone)]
pub enum VertexCredentials {
    /// Application Default Credentials, as resolved by `google-cloud-auth`.
    Adc(Credentials),
    /// The same headers every time, for tests against a local upstream.
    Fixed(HeaderMap),
}

// Hand-written so that neither a header value nor the crate's internal state
// (which holds the token) can reach a log line through `{:?}`.
impl std::fmt::Debug for VertexCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VertexCredentials::Adc(_) => f.write_str("VertexCredentials::Adc"),
            VertexCredentials::Fixed(_) => f.write_str("VertexCredentials::Fixed"),
        }
    }
}

impl VertexCredentials {
    /// Loads Application Default Credentials with the cloud-platform scope.
    ///
    /// The crate's lookup order applies: the file named by
    /// `GOOGLE_APPLICATION_CREDENTIALS`, then
    /// `~/.config/gcloud/application_default_credentials.json`
    /// (`%APPDATA%\gcloud\...` on Windows), then the metadata server. A
    /// successful return does not mean a credential exists: without a file
    /// the crate picks the metadata server, and only [`Self::headers`] finds
    /// out whether one answers.
    ///
    /// # Errors
    ///
    /// Fails when called outside a Tokio runtime (the crate starts its token
    /// refresh task while building), or when the file is missing, malformed
    /// or of an unsupported type.
    pub fn adc() -> Result<Self, VertexAuthError> {
        // Checked before anything is read, so a misuse never touches the
        // credential file.
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(VertexAuthError::Unavailable(
                "credentials must be loaded inside the async runtime".to_string(),
            ));
        }
        install_default_crypto_provider();
        Builder::default()
            .with_scopes([CLOUD_PLATFORM_SCOPE])
            .build()
            .map(VertexCredentials::Adc)
            .map_err(|error| VertexAuthError::Unavailable(build_error_reason(&error)))
    }

    /// Credentials that always return `headers`.
    pub fn fixed(headers: HeaderMap) -> Self {
        VertexCredentials::Fixed(headers)
    }

    /// The headers for one request: `authorization` and, when a quota
    /// project is configured, `x-goog-user-project`. The crate caches the
    /// token and refreshes it before it expires.
    ///
    /// # Errors
    ///
    /// Fails when no token can be obtained within [`TOKEN_TIMEOUT`].
    pub async fn headers(&self) -> Result<HeaderMap, VertexAuthError> {
        let credentials = match self {
            VertexCredentials::Fixed(headers) => return Ok(headers.clone()),
            VertexCredentials::Adc(credentials) => credentials,
        };
        let fetched = tokio::time::timeout(TOKEN_TIMEOUT, credentials.headers(Default::default()))
            .await
            .map_err(|_| {
                VertexAuthError::Token(format!(
                    "no answer within {} seconds",
                    TOKEN_TIMEOUT.as_secs()
                ))
            })?;
        match fetched {
            Ok(CacheableResource::New { data, .. }) => Ok(data),
            // Only returned when the caller passes an entity tag, which this
            // call never does.
            Ok(CacheableResource::NotModified) => Err(VertexAuthError::Token(
                "the credential library returned no headers".to_string(),
            )),
            Err(error) => Err(VertexAuthError::Token(sanitize_upstream_text(
                &error.to_string(),
                &[],
            ))),
        }
    }
}

/// The bearer token in `headers`, for scrubbing upstream text that might echo
/// it. `None` when there is no `authorization: Bearer ...` header.
pub fn access_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(http::header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    let token = token.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
}

/// Makes ring the process-wide rustls `CryptoProvider`, once.
///
/// `google-cloud-auth` is built without its TLS provider feature (that
/// feature pulls in aws-lc), and in that form it signs service account
/// assertions and opens its own HTTPS connections with whatever provider the
/// application installed, panicking when there is none. Every other
/// connection of the proxy already uses ring, so installing ring as the
/// default changes nothing for them. Another component installing a provider
/// first is fine: the crate then uses that one.
pub fn install_default_crypto_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // An error means a provider is already installed, which is all the
        // crate needs.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Describes a load failure without repeating file contents: a parse error
/// from the crate can quote a value from the file, so it gets a fixed text.
fn build_error_reason(error: &google_cloud_auth::build_errors::Error) -> String {
    if error.is_parsing() {
        return "the credentials file is not valid credentials JSON".to_string();
    }
    sanitize_upstream_text(&error.to_string(), &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_TOKEN: &str =
        "ya29.synthetic-access-token-0123456789abcdefghijklmnopqrstuvwxyz";

    fn bearer_headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
        headers.insert("x-goog-user-project", "synthetic-quota".parse().unwrap());
        headers
    }

    #[tokio::test]
    async fn fixed_credentials_synthetic_return_the_same_headers_every_time() {
        let credentials = VertexCredentials::fixed(bearer_headers(SYNTHETIC_TOKEN));
        for _ in 0..2 {
            let headers = credentials.headers().await.unwrap();
            assert_eq!(
                headers.get(http::header::AUTHORIZATION).unwrap(),
                &format!("Bearer {SYNTHETIC_TOKEN}")
            );
            assert_eq!(
                headers.get("x-goog-user-project").unwrap(),
                "synthetic-quota"
            );
        }
    }

    #[test]
    fn debug_output_synthetic_never_shows_header_values() {
        let credentials = VertexCredentials::fixed(bearer_headers(SYNTHETIC_TOKEN));
        let shown = format!("{credentials:?}");
        assert_eq!(shown, "VertexCredentials::Fixed");
        assert!(!shown.contains(SYNTHETIC_TOKEN));
    }

    #[test]
    fn access_token_synthetic_reads_only_a_bearer_authorization() {
        assert_eq!(
            access_token(&bearer_headers(SYNTHETIC_TOKEN)),
            Some(SYNTHETIC_TOKEN)
        );
        let mut lower = HeaderMap::new();
        lower.insert(
            http::header::AUTHORIZATION,
            "bearer synthetic-lowercase-scheme".parse().unwrap(),
        );
        assert_eq!(access_token(&lower), Some("synthetic-lowercase-scheme"));
        let mut basic = HeaderMap::new();
        basic.insert(
            http::header::AUTHORIZATION,
            "Basic c3ludGhldGljOnN5bnRoZXRpYw==".parse().unwrap(),
        );
        assert_eq!(access_token(&basic), None);
        let mut empty = HeaderMap::new();
        empty.insert(http::header::AUTHORIZATION, "Bearer ".parse().unwrap());
        assert_eq!(access_token(&empty), None);
        assert_eq!(access_token(&HeaderMap::new()), None);
    }

    #[test]
    fn auth_errors_synthetic_become_401_authentication_errors_with_the_gcloud_command() {
        for error in [
            VertexAuthError::Unavailable("synthetic load failure".to_string()),
            VertexAuthError::Token("synthetic token failure".to_string()),
        ] {
            let reason = error.reason().to_string();
            let mapped = error.into_provider_error();
            assert_eq!(mapped.status, StatusCode::UNAUTHORIZED);
            assert_eq!(mapped.error_type(), "authentication_error");
            assert!(mapped.message.contains(&reason), "{}", mapped.message);
            assert!(
                mapped
                    .message
                    .contains("gcloud auth application-default login"),
                "{}",
                mapped.message
            );
            assert!(
                mapped.message.contains("GOOGLE_APPLICATION_CREDENTIALS"),
                "{}",
                mapped.message
            );
        }
    }

    #[test]
    fn adc_outside_a_runtime_is_an_error_and_reads_nothing() {
        // No runtime here, so the function must return before the crate looks
        // for a credential file.
        let error = VertexCredentials::adc().unwrap_err();
        assert!(
            matches!(&error, VertexAuthError::Unavailable(reason) if reason.contains("runtime")),
            "{error:?}"
        );
    }

    #[test]
    fn installing_the_crypto_provider_twice_leaves_one_installed() {
        install_default_crypto_provider();
        install_default_crypto_provider();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }
}
