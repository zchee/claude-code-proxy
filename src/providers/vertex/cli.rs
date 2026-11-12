//! `claude-code-proxy vertex auth ...`. The proxy keeps no Google login of its
//! own, so `login`, `device` and `logout` point at `gcloud`, and `status`
//! reports what Application Default Credentials and the Vertex settings
//! resolve to in this process.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::auth::{
    LOGIN_COMMAND, LOGIN_GUIDANCE, VertexAuthError, VertexCredentials, access_token,
};
use crate::config::{
    ResolvedVertexConfig, VERTEX_ENDPOINTS_KEY, VertexEndpoints, VertexLocationOverridesSetting,
    VertexSetting, vertex_host,
};
use crate::provider::CliHandlers;

/// The command runner exits 1 only for this exact message; every other
/// failure exits 2.
const NOT_AUTHENTICATED: &str = "Not authenticated";

/// File name gcloud writes Application Default Credentials to, inside its
/// configuration directory.
const ADC_FILE_NAME: &str = "application_default_credentials.json";

/// Variables that decide where Application Default Credentials come from,
/// read the way `google-cloud-auth` reads them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdcEnvironment {
    /// `GOOGLE_APPLICATION_CREDENTIALS`. Set, even to an empty value, it is
    /// the only file the library tries.
    pub credentials_file: Option<PathBuf>,
    /// The well-known file: `$HOME/.config/gcloud/application_default_credentials.json`,
    /// or `%APPDATA%\gcloud\application_default_credentials.json` on Windows.
    pub well_known_file: Option<PathBuf>,
    /// `GOOGLE_CLOUD_QUOTA_PROJECT`, which overrides every other quota
    /// project source.
    pub quota_project: Option<String>,
    /// `CLOUDSDK_CONFIG`, gcloud's configuration directory. The library does
    /// not read it.
    pub cloudsdk_config: Option<PathBuf>,
}

impl AdcEnvironment {
    /// Reads the variables of this process.
    pub fn from_process() -> Self {
        Self {
            credentials_file: std::env::var_os("GOOGLE_APPLICATION_CREDENTIALS").map(PathBuf::from),
            well_known_file: well_known_file(),
            quota_project: std::env::var("GOOGLE_CLOUD_QUOTA_PROJECT").ok(),
            cloudsdk_config: std::env::var_os("CLOUDSDK_CONFIG")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
        }
    }
}

#[cfg(target_os = "windows")]
fn well_known_file() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|root| PathBuf::from(root).join("gcloud").join(ADC_FILE_NAME))
}

#[cfg(not(target_os = "windows"))]
fn well_known_file() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|root| {
        PathBuf::from(root)
            .join(".config")
            .join("gcloud")
            .join(ADC_FILE_NAME)
    })
}

/// Why `status` failed after printing its report.
#[derive(Debug)]
pub enum StatusFailure {
    /// A Vertex setting is missing or invalid. The error names the setting.
    Configuration(anyhow::Error),
    /// No credential was found or no access token could be obtained.
    NotAuthenticated,
}

impl StatusFailure {
    /// Guidance for stderr, and the error the command runner maps to an exit
    /// status: exactly "Not authenticated" (exit 1) when a Google login is
    /// the fix, the configuration error (exit 2) otherwise.
    pub fn into_guidance_and_error(self) -> (Option<String>, anyhow::Error) {
        match self {
            StatusFailure::Configuration(error) => (None, error),
            StatusFailure::NotAuthenticated => (
                Some(format!("{LOGIN_GUIDANCE}.")),
                anyhow::anyhow!(NOT_AUTHENTICATED),
            ),
        }
    }
}

/// What `status` prints, and how it ends.
#[derive(Debug)]
pub struct StatusReport {
    /// One fact per line. Never contains a token, key or secret.
    pub text: String,
    /// `None` when everything needed for a request resolved.
    pub failure: Option<StatusFailure>,
}

/// The two fields `status` reads from a credentials file. Deserializing into
/// this struct keeps every other field (tokens, keys, secrets) out of memory
/// that the report is built from.
#[derive(Debug, Deserialize)]
struct AdcFileFields {
    #[serde(rename = "type")]
    kind: Option<String>,
    quota_project_id: Option<String>,
}

enum AdcFileRead {
    Fields(AdcFileFields),
    Missing,
    Unreadable(String),
}

fn read_adc_file(path: &Path) -> AdcFileRead {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return AdcFileRead::Missing,
        Err(error) => return AdcFileRead::Unreadable(format!("cannot be read: {error}")),
    };
    match serde_json::from_slice::<AdcFileFields>(&bytes) {
        Ok(fields) => AdcFileRead::Fields(fields),
        // The position only: serde's message can quote a value from the file.
        Err(error) => AdcFileRead::Unreadable(format!(
            "is not valid credentials JSON (line {}, column {})",
            error.line(),
            error.column()
        )),
    }
}

/// Credential types whose file `quota_project_id` the library uses. For
/// `service_account` and `external_account` it reads the quota project only
/// from `GOOGLE_CLOUD_QUOTA_PROJECT`.
const FILE_QUOTA_PROJECT_TYPES: [&str; 3] = [
    "authorized_user",
    "external_account_authorized_user",
    "impersonated_service_account",
];

/// Builds the `status` report from the environment, the resolved settings
/// and the outcome of one attempt to obtain request headers.
///
/// Only the `type` and `quota_project_id` fields of the credentials file are
/// read. A configuration problem decides the failure before a credential
/// problem does, because no request can be sent without a project.
pub fn status_report(
    env: &AdcEnvironment,
    config: &ResolvedVertexConfig,
    token: &Result<(), VertexAuthError>,
) -> StatusReport {
    let mut lines = Vec::new();
    let mut file_fields = None;
    // The library uses the metadata server only when no file name is set
    // and the default file does not exist; any other read failure is an
    // error, not a fallback.
    let mut metadata_server = false;

    match (&env.credentials_file, &env.well_known_file) {
        (Some(path), _) => {
            lines.push(format!(
                "Credentials: {} (GOOGLE_APPLICATION_CREDENTIALS)",
                path.display()
            ));
            match read_adc_file(path) {
                AdcFileRead::Fields(fields) => file_fields = Some(fields),
                AdcFileRead::Missing => lines.push("Credentials file: does not exist".to_string()),
                AdcFileRead::Unreadable(reason) => {
                    lines.push(format!("Credentials file: {reason}"));
                }
            }
        }
        (None, Some(path)) => match read_adc_file(path) {
            AdcFileRead::Fields(fields) => {
                lines.push(format!(
                    "Credentials: {} (default location)",
                    path.display()
                ));
                file_fields = Some(fields);
            }
            AdcFileRead::Missing => {
                lines.push(metadata_server_line(path));
                metadata_server = true;
            }
            AdcFileRead::Unreadable(reason) => {
                lines.push(format!(
                    "Credentials: {} (default location)",
                    path.display()
                ));
                lines.push(format!("Credentials file: {reason}"));
            }
        },
        (None, None) => {
            lines.push(
                "Credentials: no file (the home directory is not set); the library falls back to the metadata server, which exists only on Google Cloud"
                    .to_string(),
            );
            metadata_server = true;
        }
    }
    if let Some(fields) = &file_fields {
        lines.push(format!(
            "Credential type: {}",
            fields
                .kind
                .as_deref()
                .unwrap_or("unknown (the file has no type field)")
        ));
    } else if metadata_server {
        lines.push("Credential type: metadata server".to_string());
    }
    if let Some(note) = cloudsdk_config_note(env) {
        lines.push(note);
    }

    let project = config.project();
    lines.push(match &project {
        Ok(value) => setting_line("Project", value, config.project_setting.as_ref()),
        Err(error) => format!("Project: {error}"),
    });
    let location = config.location();
    lines.push(match &location {
        Ok(value) => setting_line("Location", value, Some(&config.location_setting)),
        Err(error) => format!("Location: {error}"),
    });
    let base_url = config.base_url();
    lines.push(match (&base_url, &location) {
        (Ok(Some(url)), _) => setting_line(
            "Endpoint",
            &without_query(url),
            config.base_url_setting.as_ref(),
        ),
        (Ok(None), Ok(location)) => format!(
            "Endpoint: https://{} (derived from the location)",
            vertex_host(location)
        ),
        (Ok(None), Err(_)) => "Endpoint: unknown (the location is invalid)".to_string(),
        (Err(error), _) => format!("Endpoint: {error}"),
    });
    let location_overrides = config.location_overrides().map(drop);
    lines.push(location_overrides_line(
        config.location_overrides_setting.as_ref(),
        &location_overrides,
    ));
    lines.push(quota_project_line(env, file_fields.as_ref()));
    lines.push(match token {
        Ok(()) => "Access token obtained: true".to_string(),
        Err(error) => format!("Access token obtained: false ({})", error.reason()),
    });

    let failure = match (project, location, base_url, location_overrides) {
        (Err(error), _, _, _)
        | (_, Err(error), _, _)
        | (_, _, Err(error), _)
        | (_, _, _, Err(error)) => Some(StatusFailure::Configuration(error)),
        _ if token.is_err() => Some(StatusFailure::NotAuthenticated),
        _ => None,
    };
    StatusReport {
        text: lines.join("\n"),
        failure,
    }
}

/// One line per deployed endpoint of `vertex.endpoints`: its model name,
/// endpoint ID, project and where that came from, location, `model`, and
/// where its host comes from. Nothing is sent to the endpoint: a host not
/// configured is shown as not read yet. An invalid entry shows the error a
/// request to it would get.
pub fn endpoint_lines(endpoints: &VertexEndpoints, config: &ResolvedVertexConfig) -> Vec<String> {
    if endpoints.unreadable {
        let error = endpoints
            .first_error()
            .map_or_else(String::new, |error| error.to_string());
        return vec![format!("Deployed endpoints: {error}")];
    }
    if endpoints.entries.is_empty() {
        return vec![format!(
            "Deployed endpoints: none ({VERTEX_ENDPOINTS_KEY} in config.json)"
        )];
    }
    endpoints
        .entries
        .iter()
        .map(|entry| {
            let endpoint = match &entry.config {
                Ok(endpoint) => endpoint,
                Err(error) => return format!("Deployed endpoint: {error}"),
            };
            let name = &entry.name;
            let project = match (&endpoint.project, config.project()) {
                (Some(project), _) => format!("{project} ({VERTEX_ENDPOINTS_KEY}.{name})"),
                (None, Ok(project)) => match &config.project_setting {
                    Some(setting) => format!("{project} ({})", setting.source),
                    None => project.to_string(),
                },
                (None, Err(error)) => format!("unknown ({error})"),
            };
            let host = match &endpoint.host {
                Some(host) => format!("{host} ({VERTEX_ENDPOINTS_KEY}.{name})"),
                None => "not read yet (endpoints.get reads dedicatedEndpointDns on the first request)"
                    .to_string(),
            };
            let tools = if endpoint.tools {
                ""
            } else {
                ", tools off (a request with tools is refused)"
            };
            format!(
                "Deployed endpoint vertex/endpoint/{name}: endpoint {}, project {project}, location {}, model {}, host {host}{tools}",
                endpoint.endpoint, endpoint.location, endpoint.model
            )
        })
        .collect()
}

fn metadata_server_line(well_known: &Path) -> String {
    format!(
        "Credentials: no file at {}; the library falls back to the metadata server, which exists only on Google Cloud",
        well_known.display()
    )
}

/// gcloud honours `CLOUDSDK_CONFIG` for its own files, so `gcloud auth
/// application-default login` run with it writes the credentials there, but
/// the library looks only in the default directory.
fn cloudsdk_config_note(env: &AdcEnvironment) -> Option<String> {
    if env.credentials_file.is_some() {
        return None;
    }
    let config_dir = env.cloudsdk_config.as_ref()?;
    let default_dir = env.well_known_file.as_ref().and_then(|file| file.parent());
    if default_dir == Some(config_dir.as_path()) {
        return None;
    }
    Some(format!(
        "Note: CLOUDSDK_CONFIG is {}, but the ADC lookup does not follow CLOUDSDK_CONFIG. To use the credentials of that gcloud configuration, set GOOGLE_APPLICATION_CREDENTIALS={}",
        config_dir.display(),
        config_dir.join(ADC_FILE_NAME).display()
    ))
}

fn quota_project_line(env: &AdcEnvironment, file: Option<&AdcFileFields>) -> String {
    if let Some(project) = &env.quota_project {
        return format!("Quota project: {project} (GOOGLE_CLOUD_QUOTA_PROJECT)");
    }
    let Some(file) = file else {
        return "Quota project: none".to_string();
    };
    match (file.quota_project_id.as_deref(), file.kind.as_deref()) {
        (Some(project), Some(kind)) if FILE_QUOTA_PROJECT_TYPES.contains(&kind) => {
            format!("Quota project: {project} (quota_project_id in the credentials file)")
        }
        (Some(_), Some(kind)) => format!(
            "Quota project: none (the library ignores the file's quota_project_id for {kind} credentials; set GOOGLE_CLOUD_QUOTA_PROJECT)"
        ),
        _ => "Quota project: none".to_string(),
    }
}

/// `Location overrides: none`, the entries as `<key>=<location>` with their
/// source, or the error that names the setting. A request reads the same
/// setting and fails with that error, so it is shown before any request is
/// sent.
fn location_overrides_line(
    setting: Option<&VertexLocationOverridesSetting>,
    validated: &anyhow::Result<()>,
) -> String {
    if let Err(error) = validated {
        return format!("Location overrides: {error}");
    }
    let Some(setting) = setting else {
        return "Location overrides: none".to_string();
    };
    if setting.entries.is_empty() {
        return format!("Location overrides: none ({})", setting.source);
    }
    let entries: Vec<String> = setting
        .entries
        .iter()
        .map(|entry| format!("{}={}", entry.key, entry.location.as_deref().unwrap_or("")))
        .collect();
    format!(
        "Location overrides: {} ({})",
        entries.join(", "),
        setting.source
    )
}

fn setting_line(label: &str, value: &str, setting: Option<&VertexSetting>) -> String {
    match setting {
        Some(setting) => format!("{label}: {value} ({})", setting.source),
        None => format!("{label}: {value}"),
    }
}

/// The URL as shown to the user: a query string could carry a credential.
fn without_query(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) => {
            parsed.set_query(None);
            parsed.set_fragment(None);
            parsed.to_string()
        }
        Err(_) => "[invalid URL]".to_string(),
    }
}

/// Loads the credentials and asks for request headers once. Runs in its own
/// single-threaded runtime because the command runner is synchronous and the
/// library starts a background task while loading.
fn check_token() -> Result<(), VertexAuthError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            VertexAuthError::Unavailable(format!("the async runtime did not start: {error}"))
        })?;
    runtime.block_on(async {
        let headers = VertexCredentials::adc()?.headers().await?;
        if access_token(&headers).is_none() {
            return Err(VertexAuthError::Token(
                "the credential library returned no bearer token".to_string(),
            ));
        }
        Ok(())
    })
}

/// Message for `login` and `device`.
pub fn login_message() -> String {
    format!(
        "Vertex AI uses Google Application Default Credentials, which the proxy does not store. Run `{LOGIN_COMMAND}`, or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file"
    )
}

/// Message for `logout`.
pub fn logout_message() -> &'static str {
    "Vertex AI credentials are managed by gcloud, not by the proxy. Run `gcloud auth application-default revoke` to remove them"
}

/// The `vertex auth` command handlers.
pub struct VertexCli;

/// The handlers the provider hands to the command runner.
pub static VERTEX_CLI: VertexCli = VertexCli;

impl CliHandlers for VertexCli {
    fn login(&self) -> anyhow::Result<()> {
        anyhow::bail!(login_message())
    }

    fn device(&self) -> anyhow::Result<()> {
        self.login()
    }

    fn status(&self) -> anyhow::Result<()> {
        let config = crate::config::vertex_config();
        let endpoints = crate::config::vertex_endpoints();
        let report = status_report(&AdcEnvironment::from_process(), &config, &check_token());
        println!("{}", report.text);
        for line in endpoint_lines(&endpoints, &config) {
            println!("{line}");
        }
        // A broken endpoint entry fails the command like any other setting,
        // after a problem the report found itself.
        let failure = report
            .failure
            .or_else(|| endpoints.first_error().map(StatusFailure::Configuration));
        let Some(failure) = failure else {
            return Ok(());
        };
        let (guidance, error) = failure.into_guidance_and_error();
        if let Some(guidance) = guidance {
            eprintln!("{guidance}");
        }
        Err(error)
    }

    fn logout(&self) -> anyhow::Result<()> {
        anyhow::bail!(logout_message())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VertexSettingSource;

    const SYNTHETIC_CLIENT_SECRET: &str = "synthetic-client-secret-d41d8cd98f00b204";
    const SYNTHETIC_REFRESH_TOKEN: &str = "1//synthetic-refresh-token-e9800998ecf8427e";
    const SYNTHETIC_PRIVATE_KEY: &str =
        "-----BEGIN PRIVATE KEY-----\nsynthetic0private0key0body\n-----END PRIVATE KEY-----\n";

    /// Field names as gcloud writes a user credential file.
    fn authorized_user_json(quota_project: Option<&str>) -> String {
        let mut value = serde_json::json!({
            "account": "",
            "client_id": "synthetic-client-id.apps.googleusercontent.com",
            "client_secret": SYNTHETIC_CLIENT_SECRET,
            "refresh_token": SYNTHETIC_REFRESH_TOKEN,
            "type": "authorized_user",
            "universe_domain": "googleapis.com"
        });
        if let Some(project) = quota_project {
            value["quota_project_id"] = project.into();
        }
        value.to_string()
    }

    /// Field names of a downloaded service account key.
    fn service_account_json() -> String {
        serde_json::json!({
            "type": "service_account",
            "project_id": "synthetic-project",
            "private_key_id": "synthetic-private-key-id",
            "private_key": SYNTHETIC_PRIVATE_KEY,
            "client_email": "synthetic@synthetic-project.iam.gserviceaccount.com",
            "client_id": "000000000000000000000",
            "token_uri": "https://oauth2.googleapis.com/token",
            "quota_project_id": "synthetic-sa-quota"
        })
        .to_string()
    }

    fn write(dir: &tempfile::TempDir, name: &str, contents: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn setting(value: &str, source: VertexSettingSource) -> VertexSetting {
        VertexSetting {
            value: value.to_string(),
            source,
        }
    }

    fn config(project: Option<&str>) -> ResolvedVertexConfig {
        ResolvedVertexConfig {
            project_setting: project
                .map(|value| setting(value, VertexSettingSource::Env("CCP_VERTEX_PROJECT"))),
            location_setting: setting("global", VertexSettingSource::Default),
            base_url_setting: None,
            location_overrides_setting: None,
        }
    }

    fn assert_no_secrets(text: &str) {
        for secret in [
            SYNTHETIC_CLIENT_SECRET,
            SYNTHETIC_REFRESH_TOKEN,
            "synthetic0private0key0body",
            "BEGIN PRIVATE KEY",
        ] {
            assert!(!text.contains(secret), "report leaked {secret:?}:\n{text}");
        }
    }

    #[test]
    fn status_synthetic_user_file_from_env_reports_type_and_file_quota_project() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(
            &dir,
            "adc.json",
            &authorized_user_json(Some("synthetic-quota")),
        );
        let env = AdcEnvironment {
            credentials_file: Some(path.clone()),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert_eq!(
            report.text,
            format!(
                "Credentials: {} (GOOGLE_APPLICATION_CREDENTIALS)\nCredential type: authorized_user\nProject: synthetic-project (CCP_VERTEX_PROJECT)\nLocation: global (default)\nEndpoint: https://aiplatform.googleapis.com (derived from the location)\nLocation overrides: none\nQuota project: synthetic-quota (quota_project_id in the credentials file)\nAccess token obtained: true",
                path.display()
            )
        );
        assert!(report.failure.is_none());
        assert_no_secrets(&report.text);
    }

    #[test]
    fn status_synthetic_quota_project_env_wins_over_the_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(
            &dir,
            "adc.json",
            &authorized_user_json(Some("synthetic-quota")),
        );
        let env = AdcEnvironment {
            credentials_file: Some(path),
            quota_project: Some("synthetic-env-quota".to_string()),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert!(
            report
                .text
                .contains("Quota project: synthetic-env-quota (GOOGLE_CLOUD_QUOTA_PROJECT)"),
            "{}",
            report.text
        );
    }

    #[test]
    fn status_synthetic_service_account_file_quota_project_is_reported_as_ignored() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(&dir, "key.json", &service_account_json());
        let env = AdcEnvironment {
            credentials_file: Some(path),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert!(
            report.text.contains("Credential type: service_account"),
            "{}",
            report.text
        );
        assert!(
            report.text.contains(
                "Quota project: none (the library ignores the file's quota_project_id for service_account credentials; set GOOGLE_CLOUD_QUOTA_PROJECT)"
            ),
            "{}",
            report.text
        );
        assert!(
            !report.text.contains("synthetic-sa-quota"),
            "{}",
            report.text
        );
        assert_no_secrets(&report.text);
    }

    #[test]
    fn status_synthetic_user_file_without_quota_project_reports_none() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(&dir, ADC_FILE_NAME, &authorized_user_json(None));
        let env = AdcEnvironment {
            well_known_file: Some(path.clone()),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert!(
            report.text.starts_with(&format!(
                "Credentials: {} (default location)\n",
                path.display()
            )),
            "{}",
            report.text
        );
        assert!(
            report.text.contains("\nQuota project: none\n"),
            "{}",
            report.text
        );
        assert_no_secrets(&report.text);
    }

    #[test]
    fn status_synthetic_without_any_file_names_the_metadata_server() {
        let dir = tempfile::TempDir::new().unwrap();
        let absent = dir.path().join("gcloud").join(ADC_FILE_NAME);
        let env = AdcEnvironment {
            well_known_file: Some(absent.clone()),
            ..AdcEnvironment::default()
        };
        let token = Err(VertexAuthError::Token(
            "synthetic metadata server failure".to_string(),
        ));
        let report = status_report(&env, &config(Some("synthetic-project")), &token);
        assert!(
            report.text.starts_with(&format!(
                "Credentials: no file at {}; the library falls back to the metadata server, which exists only on Google Cloud\nCredential type: metadata server\n",
                absent.display()
            )),
            "{}",
            report.text
        );
        assert!(
            report
                .text
                .ends_with("Access token obtained: false (synthetic metadata server failure)"),
            "{}",
            report.text
        );
        let (guidance, error) = report.failure.unwrap().into_guidance_and_error();
        assert_eq!(error.to_string(), "Not authenticated");
        let guidance = guidance.unwrap();
        assert!(
            guidance.contains("gcloud auth application-default login"),
            "{guidance}"
        );
    }

    #[test]
    fn status_synthetic_missing_env_file_is_reported_and_not_replaced_by_the_default() {
        let dir = tempfile::TempDir::new().unwrap();
        let default_file = write(
            &dir,
            ADC_FILE_NAME,
            &authorized_user_json(Some("synthetic-quota")),
        );
        let env = AdcEnvironment {
            credentials_file: Some(dir.path().join("missing.json")),
            well_known_file: Some(default_file),
            ..AdcEnvironment::default()
        };
        let token = Err(VertexAuthError::Unavailable(
            "synthetic missing file".to_string(),
        ));
        let report = status_report(&env, &config(Some("synthetic-project")), &token);
        assert!(
            report.text.contains("Credentials file: does not exist"),
            "{}",
            report.text
        );
        assert!(!report.text.contains("Credential type"), "{}", report.text);
        assert!(!report.text.contains("synthetic-quota"), "{}", report.text);
        assert!(matches!(
            report.failure,
            Some(StatusFailure::NotAuthenticated)
        ));
    }

    #[test]
    fn status_synthetic_malformed_file_reports_position_without_contents() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(
            &dir,
            "adc.json",
            &format!(
                "{{\"type\": \"authorized_user\", \"refresh_token\": \"{SYNTHETIC_REFRESH_TOKEN}\", "
            ),
        );
        let env = AdcEnvironment {
            credentials_file: Some(path),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert!(
            report
                .text
                .contains("Credentials file: is not valid credentials JSON (line 1, column"),
            "{}",
            report.text
        );
        assert_no_secrets(&report.text);
    }

    #[test]
    fn status_synthetic_malformed_default_file_is_not_the_metadata_server() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(&dir, ADC_FILE_NAME, "{\"type\": ");
        let env = AdcEnvironment {
            well_known_file: Some(path.clone()),
            ..AdcEnvironment::default()
        };
        let token = Err(VertexAuthError::Unavailable(
            "synthetic parse failure".to_string(),
        ));
        let report = status_report(&env, &config(Some("synthetic-project")), &token);
        assert!(
            report.text.starts_with(&format!(
                "Credentials: {} (default location)\nCredentials file: is not valid credentials JSON (line 1, column",
                path.display()
            )),
            "{}",
            report.text
        );
        assert!(!report.text.contains("metadata server"), "{}", report.text);
        assert!(!report.text.contains("Credential type"), "{}", report.text);
        assert!(matches!(
            report.failure,
            Some(StatusFailure::NotAuthenticated)
        ));
    }

    #[test]
    fn status_synthetic_type_of_wrong_shape_does_not_echo_the_value() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = write(
            &dir,
            "adc.json",
            &format!("{{\"type\": [\"{SYNTHETIC_CLIENT_SECRET}\"]}}"),
        );
        let env = AdcEnvironment {
            credentials_file: Some(path),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert!(
            report.text.contains("is not valid credentials JSON"),
            "{}",
            report.text
        );
        assert_no_secrets(&report.text);
    }

    #[test]
    fn status_synthetic_cloudsdk_config_elsewhere_gets_a_note() {
        let dir = tempfile::TempDir::new().unwrap();
        let default_file = dir
            .path()
            .join(".config")
            .join("gcloud")
            .join(ADC_FILE_NAME);
        let other = dir.path().join("other-gcloud");
        let env = AdcEnvironment {
            well_known_file: Some(default_file),
            cloudsdk_config: Some(other.clone()),
            ..AdcEnvironment::default()
        };
        let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
        assert!(
            report.text.contains(&format!(
                "Note: CLOUDSDK_CONFIG is {}, but the ADC lookup does not follow CLOUDSDK_CONFIG. To use the credentials of that gcloud configuration, set GOOGLE_APPLICATION_CREDENTIALS={}",
                other.display(),
                other.join(ADC_FILE_NAME).display()
            )),
            "{}",
            report.text
        );
    }

    #[test]
    fn status_synthetic_cloudsdk_config_note_is_absent_when_it_cannot_matter() {
        let dir = tempfile::TempDir::new().unwrap();
        let default_dir = dir.path().join(".config").join("gcloud");
        let default_file = default_dir.join(ADC_FILE_NAME);
        // The default directory itself, written with a trailing separator.
        let same = AdcEnvironment {
            well_known_file: Some(default_file.clone()),
            cloudsdk_config: Some(PathBuf::from(format!("{}/", default_dir.display()))),
            ..AdcEnvironment::default()
        };
        // GOOGLE_APPLICATION_CREDENTIALS decides the file on its own.
        let explicit = AdcEnvironment {
            credentials_file: Some(dir.path().join("key.json")),
            well_known_file: Some(default_file),
            cloudsdk_config: Some(dir.path().join("other-gcloud")),
            ..AdcEnvironment::default()
        };
        for env in [same, explicit] {
            let report = status_report(&env, &config(Some("synthetic-project")), &Ok(()));
            assert!(!report.text.contains("CLOUDSDK_CONFIG"), "{}", report.text);
        }
    }

    #[test]
    fn status_synthetic_unset_project_fails_with_the_setting_names() {
        let token = Err(VertexAuthError::Token("synthetic".to_string()));
        let report = status_report(&AdcEnvironment::default(), &config(None), &token);
        let expected = "Vertex project is not set: set CCP_VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT or vertex.project in config.json";
        assert!(
            report.text.contains(&format!("Project: {expected}")),
            "{}",
            report.text
        );
        let failure = report.failure.unwrap();
        assert!(
            matches!(failure, StatusFailure::Configuration(_)),
            "{failure:?}"
        );
        let (guidance, error) = failure.into_guidance_and_error();
        assert_eq!(guidance, None);
        // Any message other than "Not authenticated" exits 2.
        assert_eq!(error.to_string(), expected);
    }

    #[test]
    fn status_synthetic_invalid_location_and_base_url_name_their_sources() {
        let resolved = ResolvedVertexConfig {
            project_setting: Some(setting(
                "synthetic-project",
                VertexSettingSource::Env("GOOGLE_CLOUD_PROJECT"),
            )),
            location_setting: setting(
                "US-Central1",
                VertexSettingSource::Env("CCP_VERTEX_LOCATION"),
            ),
            base_url_setting: Some(setting(
                "http://relay.example",
                VertexSettingSource::ConfigFile("vertex.baseUrl"),
            )),
            location_overrides_setting: None,
        };
        let report = status_report(&AdcEnvironment::default(), &resolved, &Ok(()));
        assert!(
            report
                .text
                .contains("Location: Invalid Vertex setting CCP_VERTEX_LOCATION:"),
            "{}",
            report.text
        );
        assert!(
            report
                .text
                .contains("Endpoint: Invalid Vertex setting vertex.baseUrl in config.json:"),
            "{}",
            report.text
        );
        let (_, error) = report.failure.unwrap().into_guidance_and_error();
        assert!(error.to_string().contains("CCP_VERTEX_LOCATION"), "{error}");
    }

    #[test]
    fn status_synthetic_regional_location_and_base_url_are_shown_with_their_sources() {
        let regional = ResolvedVertexConfig {
            location_setting: setting(
                "asia-northeast1",
                VertexSettingSource::ConfigFile("vertex.location"),
            ),
            ..config(Some("synthetic-project"))
        };
        let report = status_report(&AdcEnvironment::default(), &regional, &Ok(()));
        assert!(
            report.text.contains(
                "Location: asia-northeast1 (vertex.location in config.json)\nEndpoint: https://asia-northeast1-aiplatform.googleapis.com (derived from the location)"
            ),
            "{}",
            report.text
        );

        let relayed = ResolvedVertexConfig {
            base_url_setting: Some(setting(
                "http://127.0.0.1:9999/relay?synthetic-key=value",
                VertexSettingSource::Env("CCP_VERTEX_BASE_URL"),
            )),
            ..config(Some("synthetic-project"))
        };
        let report = status_report(&AdcEnvironment::default(), &relayed, &Ok(()));
        assert!(
            report
                .text
                .contains("Endpoint: http://127.0.0.1:9999/relay (CCP_VERTEX_BASE_URL)"),
            "{}",
            report.text
        );
        assert!(!report.text.contains("synthetic-key"), "{}", report.text);
    }

    #[test]
    fn login_device_and_logout_point_at_gcloud() {
        let login = VERTEX_CLI.login().unwrap_err().to_string();
        assert!(
            login.contains("`gcloud auth application-default login`"),
            "{login}"
        );
        assert!(login.contains("GOOGLE_APPLICATION_CREDENTIALS"), "{login}");
        assert_eq!(VERTEX_CLI.device().unwrap_err().to_string(), login);
        let logout = VERTEX_CLI.logout().unwrap_err().to_string();
        assert!(
            logout.contains("`gcloud auth application-default revoke`"),
            "{logout}"
        );
    }

    fn overrides(
        entries: &[(&str, Option<&str>)],
        source: VertexSettingSource,
    ) -> Option<VertexLocationOverridesSetting> {
        Some(VertexLocationOverridesSetting {
            entries: entries
                .iter()
                .map(
                    |(key, location)| crate::config::VertexLocationOverrideEntry {
                        key: (*key).to_string(),
                        location: location.map(str::to_string),
                    },
                )
                .collect(),
            readable: true,
            source,
        })
    }

    #[test]
    fn status_synthetic_location_overrides_are_shown_with_their_source() {
        let report = status_report(
            &AdcEnvironment::default(),
            &config(Some("synthetic-project")),
            &Ok(()),
        );
        assert!(
            report.text.contains("\nLocation overrides: none\n"),
            "{}",
            report.text
        );

        let set = ResolvedVertexConfig {
            location_overrides_setting: overrides(
                &[
                    ("mistralai", Some("europe-west4")),
                    ("anthropic/claude-opus-4-6", Some("us-east5")),
                ],
                VertexSettingSource::Env("CCP_VERTEX_LOCATION_OVERRIDES"),
            ),
            ..config(Some("synthetic-project"))
        };
        let report = status_report(&AdcEnvironment::default(), &set, &Ok(()));
        assert!(
            report.text.contains(
                "Endpoint: https://aiplatform.googleapis.com (derived from the location)\nLocation overrides: mistralai=europe-west4, anthropic/claude-opus-4-6=us-east5 (CCP_VERTEX_LOCATION_OVERRIDES)\nQuota project: none"
            ),
            "{}",
            report.text
        );
        assert!(report.failure.is_none(), "{:?}", report.failure);
    }

    #[test]
    fn status_synthetic_malformed_location_overrides_fail_as_configuration_naming_the_setting() {
        let tests = [
            (
                "environment entry without a location",
                overrides(
                    &[("mistralai", None)],
                    VertexSettingSource::Env("CCP_VERTEX_LOCATION_OVERRIDES"),
                ),
                "Invalid Vertex setting CCP_VERTEX_LOCATION_OVERRIDES: entry \"mistralai\" has no location; an entry is <key>=<location> in the variable and \"<key>\": \"<location>\" in config.json",
            ),
            (
                "config file value that is not an object",
                Some(VertexLocationOverridesSetting {
                    entries: Vec::new(),
                    readable: false,
                    source: VertexSettingSource::ConfigFile("vertex.locationOverrides"),
                }),
                "Invalid Vertex setting vertex.locationOverrides in config.json: must be an object of \"<key>\": \"<location>\" members, for example {\"mistralai\": \"europe-west4\"}",
            ),
        ];
        for (name, setting, expected) in tests {
            let resolved = ResolvedVertexConfig {
                location_overrides_setting: setting,
                ..config(Some("synthetic-project"))
            };
            // A token was obtained: the setting alone decides the failure.
            let report = status_report(&AdcEnvironment::default(), &resolved, &Ok(()));
            assert!(
                report
                    .text
                    .contains(&format!("\nLocation overrides: {expected}\n")),
                "{name}: {}",
                report.text
            );
            let failure = report
                .failure
                .unwrap_or_else(|| panic!("{name}: no failure"));
            assert!(
                matches!(failure, StatusFailure::Configuration(_)),
                "{name}: {failure:?}"
            );
            let (guidance, error) = failure.into_guidance_and_error();
            assert_eq!(guidance, None, "{name}");
            // Any message other than "Not authenticated" exits 2.
            assert_eq!(error.to_string(), expected, "{name}");
        }
    }

    fn synthetic_endpoints(file: &str) -> VertexEndpoints {
        let dir = tempfile::TempDir::new().unwrap();
        write(&dir, "config.json", file);
        crate::config::resolve_vertex_endpoints(dir.path())
    }

    #[test]
    fn endpoint_lines_synthetic_show_each_endpoint_and_where_its_host_comes_from() {
        let endpoints = synthetic_endpoints(
            r#"{"vertex":{"endpoints":{
                "my-llm":{"endpoint":"1234567890123456789","location":"us-central1","model":"gemma-2-9b-it"},
                "hf":{"endpoint":"9876543210987654321","location":"europe-west4","model":"tgi",
                      "project":"other-project","tools":false,
                      "host":"https://9876543210987654321.europe-west4-123456789.prediction.vertexai.goog"}
            }}}"#,
        );
        let mut lines = endpoint_lines(&endpoints, &config(Some("synthetic-project")));
        lines.sort_unstable();
        assert_eq!(
            lines,
            [
                "Deployed endpoint vertex/endpoint/hf: endpoint 9876543210987654321, project other-project (vertex.endpoints.hf), location europe-west4, model tgi, host https://9876543210987654321.europe-west4-123456789.prediction.vertexai.goog (vertex.endpoints.hf), tools off (a request with tools is refused)",
                "Deployed endpoint vertex/endpoint/my-llm: endpoint 1234567890123456789, project synthetic-project (CCP_VERTEX_PROJECT), location us-central1, model gemma-2-9b-it, host not read yet (endpoints.get reads dedicatedEndpointDns on the first request)",
            ]
        );
        // Without a Vertex project, an endpoint without its own says so.
        let lines = endpoint_lines(
            &synthetic_endpoints(
                r#"{"vertex":{"endpoints":{"my-llm":{"endpoint":"1","location":"us-central1","model":"m"}}}}"#,
            ),
            &config(None),
        );
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].contains("project unknown (Vertex project is not set"),
            "{lines:?}"
        );
    }

    #[test]
    fn endpoint_lines_synthetic_report_none_invalid_and_unreadable_settings() {
        assert_eq!(
            endpoint_lines(&synthetic_endpoints("{}"), &config(Some("p"))),
            ["Deployed endpoints: none (vertex.endpoints in config.json)"]
        );
        assert_eq!(
            endpoint_lines(
                &synthetic_endpoints(r#"{"vertex":{"endpoints":[]}}"#),
                &config(Some("p"))
            ),
            [
                "Deployed endpoints: Invalid Vertex setting vertex.endpoints in config.json: must be an object of \"<name>\": {\"endpoint\", \"location\", \"model\"} members"
            ]
        );
        let endpoints = synthetic_endpoints(
            r#"{"vertex":{"endpoints":{"bad":{"endpoint":"1","location":"global","model":"m",
                "host":"https://user:synthetic-pass@1.us-central1-9.prediction.vertexai.goog"}}}}"#,
        );
        let lines = endpoint_lines(&endpoints, &config(Some("p")));
        assert_eq!(
            lines,
            [
                "Deployed endpoint: Invalid Vertex setting vertex.endpoints in config.json: endpoint \"bad\": location must be the endpoint's region, such as us-central1; an endpoint is never in global"
            ]
        );
        assert!(!lines.join("\n").contains("synthetic-pass"));
        // The same error fails the command.
        assert_eq!(
            endpoints.first_error().map(|error| error.to_string()),
            Some(
                lines[0]
                    .trim_start_matches("Deployed endpoint: ")
                    .to_string()
            )
        );
    }
}
