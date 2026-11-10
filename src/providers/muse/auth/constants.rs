//! Fixed protocol values for the Meta login and key mint. The OAuth host does
//! not list the device endpoints in its discovery document, so the paths are
//! fixed here rather than discovered.

use std::time::Duration;

/// Public client id of the Muse Code CLI's device-code login.
pub const CLIENT_ID: &str = "1031625952748946";

pub const DEVICE_AUTHORIZATION_PATH: &str = "/oidc/device/authorization/";
pub const DEVICE_TOKEN_PATH: &str = "/oidc/device/token/";
pub const GRANT_DEVICE_CODE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// The mint is served at the API host root, not under the Model API's `/v1`.
pub const MINT_PATH: &str = "/muse-code/key";
pub const MINT_API_VERSION: &str = "1.0.0";
pub const MINT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const MINT_TIMEOUT: Duration = Duration::from_secs(20);

/// Upper bound on any login or mint response body the proxy reads.
pub const MAX_AUTH_RESPONSE_BYTES: usize = 64 * 1024;

/// Where an account without an active subscription can subscribe.
pub const ACCOUNTS_CENTER_URL: &str = "https://accountscenter.meta.com/muse_code/";

/// Command that creates a new proxy login, quoted in every error that asks for one.
pub const LOGIN_COMMAND: &str = "claude-code-proxy muse auth login";
