use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::paths;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasProvider {
    Codex,
    Kimi,
}

impl AliasProvider {
    pub fn as_str(&self) -> &str {
        match self {
            AliasProvider::Codex => "codex",
            AliasProvider::Kimi => "kimi",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub bind_address: String,
    pub port: u16,
    pub alias_provider: AliasProvider,
    pub log_verbose: bool,
    pub log_stderr: bool,
    pub config_dir: PathBuf,
}

#[derive(Deserialize)]
struct FileConfig {
    #[serde(rename = "bindAddress")]
    pub bind_address: Option<String>,
    pub port: Option<u16>,
    #[serde(rename = "aliasProvider")]
    pub alias_provider: Option<String>,
    #[serde(rename = "autoReviewModel")]
    pub auto_review_model: Option<String>,
    pub log: Option<FileLog>,
    pub kimi: Option<KimiConfig>,
    pub codex: Option<CodexConfig>,
    pub cursor: Option<CursorConfig>,
    pub grok: Option<GrokConfig>,
    pub muse: Option<MuseConfig>,
    pub opencode: Option<OpenCodeConfig>,
    pub vertex: Option<VertexConfig>,
}

#[derive(Deserialize, Clone)]
struct CodexConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "originator")]
    pub originator: Option<String>,
    #[serde(rename = "userAgent")]
    pub user_agent: Option<String>,
    #[serde(rename = "previousResponseId")]
    pub previous_response_id: Option<bool>,
    #[serde(rename = "requestCompression")]
    pub request_compression: Option<bool>,
    #[serde(rename = "serverCompaction")]
    pub server_compaction: Option<bool>,
    #[serde(rename = "responsesApi")]
    pub responses_api: Option<bool>,
    #[serde(rename = "fullLane")]
    pub full_lane: Option<bool>,
    #[serde(rename = "imagesApi")]
    pub images_api: Option<bool>,
    #[serde(rename = "imagesBaseUrl")]
    pub images_base_url: Option<String>,
    #[serde(rename = "transcriptionsApi")]
    pub transcriptions_api: Option<bool>,
    #[serde(rename = "serviceTier")]
    pub service_tier: Option<String>,
    #[serde(rename = "reasoningSummary")]
    pub reasoning_summary: Option<String>,
    #[serde(rename = "effort")]
    pub effort: Option<String>,
    #[serde(rename = "model")]
    pub model: Option<String>,
    pub transport: Option<String>,
    #[serde(rename = "headerTimeoutMs")]
    pub header_timeout_ms: Option<u64>,
}

#[derive(Deserialize, Clone)]
struct CursorConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "clientVersion")]
    pub client_version: Option<String>,
    #[serde(rename = "agentBundle")]
    pub agent_bundle: Option<String>,
}

#[derive(Deserialize, Clone)]
struct KimiConfig {
    #[serde(rename = "userAgent")]
    pub user_agent: Option<String>,
    #[serde(rename = "oauthHost")]
    pub oauth_host: Option<String>,
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
}

#[derive(Deserialize, Clone)]
struct GrokConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "clientVersion")]
    pub client_version: Option<String>,
}

#[derive(Deserialize, Clone)]
struct MuseConfig {
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    #[serde(rename = "oauthHost")]
    pub oauth_host: Option<String>,
    #[serde(rename = "mintBaseUrl")]
    pub mint_base_url: Option<String>,
    #[serde(rename = "userAgent")]
    pub user_agent: Option<String>,
}

#[derive(Deserialize, Clone)]
struct OpenCodeConfig {
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
}

#[derive(Deserialize, Clone)]
struct VertexConfig {
    pub project: Option<String>,
    pub location: Option<String>,
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    // Read as any JSON value: a typed map would make one wrong value drop
    // the whole config file, because a file that fails to parse is ignored.
    #[serde(rename = "locationOverrides")]
    pub location_overrides: Option<serde_json::Value>,
    // Read as any JSON value for the same reason: one bad endpoint entry
    // must not drop the rest of the file.
    pub endpoints: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct FileLog {
    pub verbose: Option<bool>,
    pub stderr: Option<bool>,
}

fn parse_alias(raw: &str) -> Option<AliasProvider> {
    match raw {
        "codex" => Some(AliasProvider::Codex),
        "kimi" => Some(AliasProvider::Kimi),
        _ => None,
    }
}

fn read_file_config(config_dir: &Path) -> Option<FileConfig> {
    let path = config_dir.join("config.json");
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn load_config() -> LoadedConfig {
    let env = paths::DirResolverEnv::default();
    let config_dir = paths::resolve_config_dir(&env);
    load_config_from_env(&env.env, config_dir)
}

pub fn load_config_for_env(env: &HashMap<String, String>) -> LoadedConfig {
    let home = env
        .get("HOME")
        .or_else(|| env.get("USERPROFILE"))
        .cloned()
        .unwrap_or_else(|| "/".to_string());
    let resolver_env = paths::DirResolverEnv {
        platform: std::env::consts::OS.to_string(),
        env: env.clone(),
        home,
    };
    let config_dir = paths::resolve_config_dir(&resolver_env);
    load_config_from_env(env, config_dir)
}

fn load_config_from_env(env: &HashMap<String, String>, config_dir: PathBuf) -> LoadedConfig {
    let file = read_file_config(&config_dir);

    let mut out = LoadedConfig {
        bind_address: "127.0.0.1".to_string(),
        port: 18765,
        alias_provider: AliasProvider::Codex,
        log_verbose: false,
        log_stderr: false,
        config_dir: config_dir.clone(),
    };

    if let Some(raw) = env.get("CCP_BIND_ADDRESS") {
        out.bind_address = raw.clone();
    } else if let Some(bind_address) = file.as_ref().and_then(|f| f.bind_address.clone()) {
        out.bind_address = bind_address;
    }

    if let Some(raw) = env.get("CCP_ALIAS_PROVIDER") {
        if let Some(alias) = parse_alias(raw) {
            out.alias_provider = alias;
        }
    } else if let Some(alias_provider) = file
        .as_ref()
        .and_then(|f| f.alias_provider.as_deref())
        .and_then(parse_alias)
    {
        out.alias_provider = alias_provider;
    }

    if let Some(raw) = env.get("PORT") {
        if let Ok(port) = raw.parse::<u16>() {
            out.port = port;
        }
    } else if let Some(port) = file.as_ref().and_then(|f| f.port) {
        out.port = port;
    }

    if env.contains_key("CCP_LOG_VERBOSE") {
        out.log_verbose = true;
    } else if let Some(value) = file
        .as_ref()
        .and_then(|f| f.log.as_ref().and_then(|v| v.verbose))
    {
        out.log_verbose = value;
    }

    if env.contains_key("CCP_LOG_STDERR") {
        out.log_stderr = true;
    } else if let Some(value) = file
        .as_ref()
        .and_then(|f| f.log.as_ref().and_then(|v| v.stderr))
    {
        out.log_stderr = value;
    }

    out
}

pub fn config_path() -> PathBuf {
    paths::config_dir().join("config.json")
}

pub fn port() -> u16 {
    load_config().port
}

pub fn bind_address() -> String {
    load_config().bind_address
}

pub fn alias_provider() -> AliasProvider {
    load_config().alias_provider
}

pub fn log_verbose() -> bool {
    load_config().log_verbose
}

pub fn log_stderr() -> bool {
    load_config().log_stderr
}

pub fn config_override_summary_lines(cfg: &LoadedConfig) -> Vec<String> {
    let file = read_file_config(&cfg.config_dir);
    let env: HashMap<_, _> = std::env::vars().collect();
    // Listed from the resolved sources so the summary cannot disagree with
    // what the provider uses, including blank values that fall through.
    let vertex = resolve_vertex_config(&env, &cfg.config_dir);
    let vertex_endpoints = resolve_vertex_endpoints(&cfg.config_dir);
    let vertex_settings = [
        ("vertex.project", vertex.project_setting.as_ref()),
        ("vertex.location", Some(&vertex.location_setting)),
        ("vertex.baseUrl", vertex.base_url_setting.as_ref()),
    ];
    let mut out = Vec::new();
    if env.contains_key("CCP_BIND_ADDRESS") {
        out.push("bindAddress (env)".to_string());
    }
    if env.contains_key("PORT") {
        out.push("port (env)".to_string());
    }
    if env.contains_key("CCP_ALIAS_PROVIDER") {
        out.push("aliasProvider (env)".to_string());
    }
    if env.contains_key("CCP_LOG_VERBOSE") {
        out.push("log.verbose (env)".to_string());
    }
    if env.contains_key("CCP_LOG_STDERR") {
        out.push("log.stderr (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_RESPONSES_API") {
        out.push("codex.responsesApi (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_FULL_LANE") {
        out.push("codex.fullLane (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_IMAGES_API") {
        out.push("codex.imagesApi (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_IMAGES_BASE_URL") {
        out.push("codex.imagesBaseUrl (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_TRANSCRIPTIONS_API") {
        out.push("codex.transcriptionsApi (env)".to_string());
    }
    if env.contains_key("CCP_KIMI_OAUTH_HOST") {
        out.push("kimi.oauthHost (env)".to_string());
    }
    if env.contains_key("CCP_KIMI_BASE_URL") {
        out.push("kimi.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_CURSOR_BASE_URL") {
        out.push("cursor.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_CURSOR_CLIENT_VERSION") {
        out.push("cursor.clientVersion (env)".to_string());
    }
    if env.contains_key("CCP_KIMI_USER_AGENT") {
        out.push("kimi.userAgent (env)".to_string());
    }
    if env.contains_key("CCP_GROK_BASE_URL") {
        out.push("grok.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_GROK_CLIENT_VERSION") {
        out.push("grok.clientVersion (env)".to_string());
    }
    if env.contains_key("CCP_MUSE_BASE_URL") {
        out.push("muse.baseUrl (env)".to_string());
    }
    if env.contains_key("CCP_MUSE_OAUTH_HOST") {
        out.push("muse.oauthHost (env)".to_string());
    }
    if env.contains_key("CCP_MUSE_MINT_BASE_URL") {
        out.push("muse.mintBaseUrl (env)".to_string());
    }
    if env.contains_key("CCP_MUSE_USER_AGENT") {
        out.push("muse.userAgent (env)".to_string());
    }
    if env.contains_key("CCP_OPENCODE_API_KEY") {
        out.push("opencode.apiKey (env)".to_string());
    } else if env.contains_key("OPENCODE_API_KEY") {
        out.push("opencode.apiKey (OpenCode env)".to_string());
    }
    if env.contains_key("CCP_OPENCODE_BASE_URL") {
        out.push("opencode.baseUrl (env)".to_string());
    }
    for (key, setting) in vertex_settings {
        match setting.map(|setting| setting.source) {
            Some(VertexSettingSource::Env(name)) if name.starts_with("CCP_") => {
                out.push(format!("{key} (env)"));
            }
            Some(VertexSettingSource::Env(_)) => out.push(format!("{key} (Google env)")),
            _ => {}
        }
    }
    let location_overrides = vertex.location_overrides_setting.as_ref();
    if location_overrides
        .is_some_and(|setting| matches!(setting.source, VertexSettingSource::Env(_)))
    {
        out.push(format!("{VERTEX_LOCATION_OVERRIDES_KEY} (env)"));
    }
    if env
        .get("CCP_CODEX_REASONING_SUMMARY")
        .is_some_and(|raw| !raw.is_empty())
    {
        out.push("CCP_CODEX_REASONING_SUMMARY (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_SERVER_COMPACTION") {
        out.push("CCP_CODEX_SERVER_COMPACTION (env)".to_string());
    }
    if env.contains_key("CCP_CODEX_REQUEST_COMPRESSION") {
        out.push("CCP_CODEX_REQUEST_COMPRESSION (env)".to_string());
    }
    if env
        .get("CCP_AUTO_REVIEW_MODEL")
        .is_some_and(|raw| !raw.is_empty())
    {
        out.push("CCP_AUTO_REVIEW_MODEL (env)".to_string());
    }
    if let Some(file_cfg) = file {
        if let Some(bind_address) = file_cfg.bind_address {
            out.push(format!("bindAddress: {bind_address}"));
        }
        if let Some(p) = file_cfg.port {
            out.push(format!("port: {p}"));
        }
        if let Some(alias) = file_cfg.alias_provider {
            out.push(format!("aliasProvider: {alias}"));
        }
        if file_cfg
            .auto_review_model
            .is_some_and(|model| !model.is_empty())
        {
            out.push("autoReviewModel (config)".to_string());
        }
        if let Some(log) = file_cfg.log {
            if let Some(v) = log.verbose {
                out.push(format!("log.verbose: {v}"));
            }
            if let Some(v) = log.stderr {
                out.push(format!("log.stderr: {v}"));
            }
        }
        if let Some(opencode) = file_cfg.opencode {
            if opencode.api_key.is_some_and(|raw| !raw.is_empty()) {
                out.push("opencode.apiKey (config)".to_string());
            }
            if let Some(url) = opencode.base_url.filter(|raw| !raw.is_empty()) {
                out.push(format!("opencode.baseUrl: {url}"));
            }
        }
        for (key, setting) in vertex_settings {
            let Some(setting) = setting
                .filter(|setting| matches!(setting.source, VertexSettingSource::ConfigFile(_)))
            else {
                continue;
            };
            // The base URL receives the access token, and a rejected value may
            // carry user info, so only its presence is listed.
            if key == "vertex.baseUrl" {
                out.push(format!("{key} (config)"));
            } else {
                out.push(format!("{key}: {}", setting.value));
            }
        }
        if let Some(setting) = location_overrides
            .filter(|setting| matches!(setting.source, VertexSettingSource::ConfigFile(_)))
        {
            if setting.readable {
                let entries: Vec<String> = setting
                    .entries
                    .iter()
                    .map(|entry| {
                        format!("{}={}", entry.key, entry.location.as_deref().unwrap_or("?"))
                    })
                    .collect();
                out.push(format!(
                    "{VERTEX_LOCATION_OVERRIDES_KEY}: {}",
                    entries.join(", ")
                ));
            } else {
                out.push(format!("{VERTEX_LOCATION_OVERRIDES_KEY} (config)"));
            }
        }
        if let Some(line) = vertex_endpoints.summary_line() {
            out.push(line);
        }
        if let Some(codex) = file_cfg.codex {
            if codex
                .reasoning_summary
                .is_some_and(|value| !value.is_empty())
            {
                out.push("codex.reasoningSummary (config)".to_string());
            }
            if let Some(enabled) = codex.server_compaction {
                out.push(format!("codex.serverCompaction: {enabled}"));
            }
            if let Some(enabled) = codex.request_compression {
                out.push(format!("codex.requestCompression: {enabled}"));
            }
            if codex.responses_api == Some(true) {
                out.push("codex.responsesApi: true".to_string());
            }
            if codex.full_lane == Some(true) {
                out.push("codex.fullLane: true".to_string());
            }
            if codex.images_api == Some(true) {
                out.push("codex.imagesApi: true".to_string());
            }
            if codex.images_base_url.is_some() {
                out.push("codex.imagesBaseUrl (config)".to_string());
            }
            if codex.transcriptions_api == Some(true) {
                out.push("codex.transcriptionsApi: true".to_string());
            }
            if let Some(ms) = codex.header_timeout_ms {
                out.push(format!("codex.headerTimeoutMs: {ms}"));
            }
        }
    }
    out
}

pub fn grok_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_GROK_BASE_URL") {
        return raw.clone();
    }
    if let Some(grok) = read_file_config(&paths::config_dir()).and_then(|f| f.grok)
        && let Some(url) = grok.base_url
    {
        return url;
    }
    "https://cli-chat-proxy.grok.com/v1".to_string()
}

pub fn grok_client_version() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_GROK_CLIENT_VERSION") {
        return raw.clone();
    }
    if let Some(grok) = read_file_config(&paths::config_dir()).and_then(|f| f.grok)
        && let Some(version) = grok.client_version
    {
        return version;
    }
    "1.0.13".to_string()
}

// ---------------------------------------------------------------------------
// Muse config
// ---------------------------------------------------------------------------

fn muse_file_config() -> Option<MuseConfig> {
    read_file_config(&paths::config_dir()).and_then(|f| f.muse)
}

/// Model API base URL. The key mint also returns one, which the provider
/// accepts only when it points at this URL's host over https.
pub fn muse_base_url() -> anyhow::Result<String> {
    muse_endpoint(
        "CCP_MUSE_BASE_URL",
        "muse.baseUrl",
        |muse| muse.base_url,
        "https://api.meta.ai/v1",
    )
}

/// Origin of the device-code OAuth endpoints.
pub fn muse_oauth_host() -> anyhow::Result<String> {
    muse_endpoint(
        "CCP_MUSE_OAUTH_HOST",
        "muse.oauthHost",
        |muse| muse.oauth_host,
        "https://auth.meta.com",
    )
}

/// Origin of the API key mint. The mint lives at the API host root, not under
/// the `/v1` prefix of the Model API.
pub fn muse_mint_base_url() -> anyhow::Result<String> {
    muse_endpoint(
        "CCP_MUSE_MINT_BASE_URL",
        "muse.mintBaseUrl",
        |muse| muse.mint_base_url,
        "https://api.meta.ai",
    )
}

/// Reads one Muse endpoint. These hosts receive the Meta access token and the
/// API key, so a value that is not https (or plain http to a loopback host)
/// or that carries user info is an error naming the setting, never a silent
/// fallback to the default.
fn muse_endpoint(
    env_name: &str,
    file_key: &str,
    from_file: impl FnOnce(MuseConfig) -> Option<String>,
    default: &str,
) -> anyhow::Result<String> {
    muse_setting(
        env_name,
        file_key,
        from_file,
        default,
        validate_token_endpoint,
    )
}

/// Reads one Muse setting with env over config.json over the default and
/// returns it trimmed, as validated. An invalid value is an error naming the
/// setting; the value is not repeated in it, since it may hold credentials.
fn muse_setting(
    env_name: &str,
    file_key: &str,
    from_file: impl FnOnce(MuseConfig) -> Option<String>,
    default: &str,
    validate: fn(&str) -> Result<(), &'static str>,
) -> anyhow::Result<String> {
    // An empty or blank value means "not set", as for the other providers'
    // base URLs, and falls through to the next source.
    let set = |raw: String| {
        let trimmed = raw.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    };
    let (value, setting) = match std::env::var(env_name).ok().and_then(set) {
        Some(value) => (value, env_name.to_string()),
        None => match muse_file_config().and_then(from_file).and_then(set) {
            Some(value) => (value, format!("{file_key} in config.json")),
            None => return Ok(default.to_string()),
        },
    };
    validate(&value)
        .map_err(|reason| anyhow::anyhow!("Invalid Muse setting {setting}: {reason}"))?;
    Ok(value)
}

/// Checks a URL that receives a credential (a bearer token, an access token
/// or an API key): https, or plain http only to a loopback host, and no user
/// info. The reasons are fixed strings so callers can name the setting without
/// repeating a value that may itself carry a credential.
fn validate_token_endpoint(value: &str) -> Result<(), &'static str> {
    let url = url::Url::parse(value).map_err(|_| "not a valid URL")?;
    let loopback = match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        // `localhost.` is the same name written fully qualified.
        Some(url::Host::Domain(domain)) => domain
            .strip_suffix('.')
            .unwrap_or(domain)
            .eq_ignore_ascii_case("localhost"),
        None => return Err("the URL has no host"),
    };
    match url.scheme() {
        "https" => {}
        "http" if loopback => {}
        _ => return Err("must use https (plain http only for a loopback host)"),
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("must not contain a user name or password");
    }
    Ok(())
}

/// User-Agent for every Muse request, including those that carry the access
/// token or the API key. The OAuth host redirects requests that carry none,
/// so the default is never empty.
pub fn muse_user_agent() -> anyhow::Result<String> {
    muse_setting(
        "CCP_MUSE_USER_AGENT",
        "muse.userAgent",
        |muse| muse.user_agent,
        &format!("claude-code-proxy/{}", env!("CARGO_PKG_VERSION")),
        validate_muse_user_agent,
    )
}

const MAX_MUSE_USER_AGENT_LEN: usize = 256;

fn validate_muse_user_agent(value: &str) -> Result<(), &'static str> {
    if value.len() > MAX_MUSE_USER_AGENT_LEN {
        return Err("must be at most 256 characters");
    }
    if !value.bytes().all(|byte| (0x20..=0x7e).contains(&byte)) {
        return Err("must contain only printable ASCII characters");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Grok tool-image policy (CCP_GROK_TOOL_IMAGE)
// ---------------------------------------------------------------------------

/// How the Grok translator treats Anthropic `image` blocks (tool results and
/// top-level user messages). `omit` is the safe default: degrade to the L1
/// placeholder string. `reattach` keeps the placeholder in the tool output and
/// additionally appends a user message carrying the images as `input_image`
/// data URLs. `inline` sends the tool output itself as an array of
/// `input_text` + `input_image` parts (string-only outputs still serialize as
/// plain strings). `reject` restores the pre-L1 hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrokToolImageMode {
    Omit,
    Reattach,
    Inline,
    Reject,
}

pub fn parse_grok_tool_image_mode(raw: Option<&str>) -> GrokToolImageMode {
    match raw.map(str::trim) {
        Some("reattach") => GrokToolImageMode::Reattach,
        Some("inline") => GrokToolImageMode::Inline,
        Some("reject") => GrokToolImageMode::Reject,
        // Any unknown/empty value degrades to the safe default.
        _ => GrokToolImageMode::Omit,
    }
}

pub fn grok_tool_image_mode() -> GrokToolImageMode {
    parse_grok_tool_image_mode(std::env::var("CCP_GROK_TOOL_IMAGE").ok().as_deref())
}

/// Warn once at startup when an unknown mode was requested. Called from the
/// Grok provider constructor rather than per request.
pub fn warn_grok_tool_image_mode_once(log: &crate::logging::Logger) {
    match std::env::var("CCP_GROK_TOOL_IMAGE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some(other) if !matches!(other, "" | "omit" | "reattach" | "inline" | "reject") => {
            let mut fields = serde_json::Map::new();
            fields.insert(
                "value".to_string(),
                serde_json::Value::String(other.to_string()),
            );
            log.warn(
                "unrecognized CCP_GROK_TOOL_IMAGE value; falling back to omit",
                Some(fields),
            );
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Grok hosted-search policy (CCP_GROK_HOSTED_SEARCH)
// ---------------------------------------------------------------------------

/// Whether the Grok translator replaces caller search tools with xAI-hosted
/// search and requires hosted tool use on explicit search turns.
///
/// The disabled policy preserves caller tools, instructions, and tool choice.
/// It adds `x_search` only to X-specific turns because the caller has no
/// equivalent access to xAI's X index.
///
/// The enabled policy favors xAI-hosted search and citations. Hosted tools
/// replace caller search implementations, matching turns receive search
/// guidance, and explicit search turns use `tool_choice: required`.
///
/// Set `CCP_GROK_HOSTED_SEARCH` to `1`, `on`, or `true` to enable this policy.
pub fn parse_grok_hosted_search(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), Some("1" | "on" | "true"))
}

pub fn grok_hosted_search() -> bool {
    parse_grok_hosted_search(std::env::var("CCP_GROK_HOSTED_SEARCH").ok().as_deref())
}

// ---------------------------------------------------------------------------
// Grok hosted-search block shape (CCP_GROK_SEARCH_BLOCKS)
// ---------------------------------------------------------------------------

/// How a hosted search that xAI ran is reported to the client.
///
/// `Text` projects the search query into a standard `text` block.
///
/// `Native` preserves the Anthropic server-tool shape: `server_tool_use`
/// followed by `web_search_tool_result` or `x_search_tool_result`. Select this
/// shape for clients that consume hosted-tool blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrokSearchBlocks {
    Text,
    Native,
}

pub fn parse_grok_search_blocks(raw: Option<&str>) -> GrokSearchBlocks {
    match raw.map(str::trim) {
        Some("native") => GrokSearchBlocks::Native,
        // Text is the compatibility-safe fallback for empty or unknown values.
        _ => GrokSearchBlocks::Text,
    }
}

pub fn grok_search_blocks() -> GrokSearchBlocks {
    parse_grok_search_blocks(std::env::var("CCP_GROK_SEARCH_BLOCKS").ok().as_deref())
}

struct ResolvedOpenCodeConfig {
    api_key: Option<String>,
    api_key_source: Option<&'static str>,
    base_url: String,
}

fn resolve_opencode_config(
    env: &HashMap<String, String>,
    config_dir: &Path,
) -> ResolvedOpenCodeConfig {
    let file = read_file_config(config_dir).and_then(|file| file.opencode);
    let file_key = file
        .as_ref()
        .and_then(|config| config.api_key.as_ref())
        .filter(|value| !value.is_empty());
    let (api_key, api_key_source) = if let Some(value) = env
        .get("CCP_OPENCODE_API_KEY")
        .filter(|value| !value.is_empty())
    {
        (Some(value.clone()), Some("CCP_OPENCODE_API_KEY"))
    } else if let Some(value) = env
        .get("OPENCODE_API_KEY")
        .filter(|value| !value.is_empty())
    {
        (Some(value.clone()), Some("OPENCODE_API_KEY"))
    } else if let Some(value) = file_key {
        (Some(value.clone()), Some("config.json"))
    } else {
        (None, None)
    };
    let base_url = env
        .get("CCP_OPENCODE_BASE_URL")
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| {
            file.as_ref()
                .and_then(|config| config.base_url.as_ref())
                .filter(|value| !value.is_empty())
                .cloned()
        })
        .unwrap_or_else(|| "https://opencode.ai/zen/go/v1".to_string());

    ResolvedOpenCodeConfig {
        api_key,
        api_key_source,
        base_url,
    }
}

pub fn opencode_api_key() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    resolve_opencode_config(&env, &paths::config_dir()).api_key
}

pub fn opencode_api_key_source() -> Option<&'static str> {
    let env: HashMap<_, _> = std::env::vars().collect();
    resolve_opencode_config(&env, &paths::config_dir()).api_key_source
}

pub fn opencode_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    resolve_opencode_config(&env, &paths::config_dir()).base_url
}

// ---------------------------------------------------------------------------
// Vertex config
// ---------------------------------------------------------------------------

/// Location used when no setting names one. `global` serves every Gemini
/// model and needs no regional quota.
pub const VERTEX_DEFAULT_LOCATION: &str = "global";

/// Where a resolved Vertex setting came from. `Display` gives the name a user
/// edits to change it, which is also how errors refer to the setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexSettingSource {
    /// The named environment variable.
    Env(&'static str),
    /// The named key of the `vertex` object in `config.json`.
    ConfigFile(&'static str),
    /// The built-in default.
    Default,
}

impl std::fmt::Display for VertexSettingSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VertexSettingSource::Env(name) => f.write_str(name),
            VertexSettingSource::ConfigFile(key) => write!(f, "{key} in config.json"),
            VertexSettingSource::Default => f.write_str("default"),
        }
    }
}

/// One resolved Vertex setting: the trimmed value and where it came from.
/// The value is not validated yet; the accessors on [`ResolvedVertexConfig`]
/// do that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexSetting {
    /// The value, with surrounding whitespace removed.
    pub value: String,
    /// The environment variable, config key or default it came from.
    pub source: VertexSettingSource,
}

/// Environment variable holding the location overrides as
/// `<key>=<location>` entries separated by commas.
pub const VERTEX_LOCATION_OVERRIDES_ENV: &str = "CCP_VERTEX_LOCATION_OVERRIDES";

/// `config.json` key holding the location overrides as an object of
/// `"<key>": "<location>"` members.
pub const VERTEX_LOCATION_OVERRIDES_KEY: &str = "vertex.locationOverrides";

/// One location override as read: the trimmed key and location, not
/// validated yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexLocationOverrideEntry {
    /// `<publisher>` or `<publisher>/<model-id>`.
    pub key: String,
    /// The location, or `None` when the entry has none: an environment entry
    /// without `=`, or a `config.json` value that is not a string.
    pub location: Option<String>,
}

/// The location overrides as read from the one source that sets them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexLocationOverridesSetting {
    /// The entries in the order they were read.
    pub entries: Vec<VertexLocationOverrideEntry>,
    /// False when `config.json` gives a value that is not an object; there
    /// are no entries then.
    pub readable: bool,
    /// The environment variable or config key the entries came from.
    pub source: VertexSettingSource,
}

/// Validated location overrides, each key at most once. A key is a
/// publisher (`mistralai`) or a publisher and a model ID
/// (`anthropic/claude-opus-4-6`); which one applies to a model is decided by
/// the model's location rule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VertexLocationOverrides {
    entries: Vec<(String, String)>,
}

impl VertexLocationOverrides {
    /// The location set for exactly `key`, if any.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(entry, _)| entry == key)
            .map(|(_, location)| location.as_str())
    }

    /// Whether no override is set.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Vertex settings after applying the precedence `CCP_VERTEX_*` over the
/// standard Google Cloud variables over `config.json`. Values are kept even
/// when they are invalid so that a status command can show what was read and
/// from where; the accessors return the validated value or an error that names
/// the setting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedVertexConfig {
    /// Google Cloud project ID. `None` when no source sets it: there is no
    /// default, because guessing a project would bill the wrong one.
    pub project_setting: Option<VertexSetting>,
    /// Vertex AI location, `global` unless a source sets another.
    pub location_setting: VertexSetting,
    /// Replacement for the host derived from the location, for a relay or a
    /// local test server.
    pub base_url_setting: Option<VertexSetting>,
    /// Locations for single publishers or models, in place of the location
    /// above. `None` when no source sets any.
    pub location_overrides_setting: Option<VertexLocationOverridesSetting>,
}

impl ResolvedVertexConfig {
    /// The validated project ID.
    ///
    /// # Errors
    ///
    /// Fails when no source sets a project, naming every setting that can, or
    /// when the value cannot be a project ID, naming the setting it came from.
    /// The project goes into the request path, so `/`, `?` and `#` never pass.
    pub fn project(&self) -> anyhow::Result<&str> {
        let Some(setting) = &self.project_setting else {
            anyhow::bail!(
                "Vertex project is not set: set CCP_VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT or vertex.project in config.json"
            );
        };
        validated_vertex_setting(setting, validate_vertex_project)
    }

    /// The validated location.
    ///
    /// # Errors
    ///
    /// Fails when the value is not lowercase letters, digits and `-`, naming
    /// the setting it came from. The location goes into both the host name
    /// and the request path.
    pub fn location(&self) -> anyhow::Result<&str> {
        validated_vertex_setting(&self.location_setting, validate_vertex_location)
    }

    /// The validated base URL override, or `None` to use the host derived
    /// from the location (see [`vertex_host`]).
    ///
    /// # Errors
    ///
    /// The URL receives the Google access token, so it must use https (plain
    /// http only for a loopback host) and carry no user info. A value that
    /// does not is an error naming the setting; the value itself is left out
    /// because it may contain credentials.
    pub fn base_url(&self) -> anyhow::Result<Option<&str>> {
        self.base_url_setting
            .as_ref()
            .map(|setting| validated_vertex_setting(setting, validate_token_endpoint))
            .transpose()
    }

    /// The validated location overrides; empty when no source sets any.
    ///
    /// # Errors
    ///
    /// Fails, naming the setting the overrides came from, when an entry's
    /// key is not `<publisher>` or `<publisher>/<model-id>`, when its
    /// location is missing or fails the same check as the location, or when
    /// a key is listed twice. A key that fails the check is referred to by
    /// its position rather than quoted. An override never names a deployed
    /// endpoint (`endpoint/<name>`): an endpoint has its own location.
    pub fn location_overrides(&self) -> anyhow::Result<VertexLocationOverrides> {
        let Some(setting) = &self.location_overrides_setting else {
            return Ok(VertexLocationOverrides::default());
        };
        let invalid =
            |reason: String| anyhow::anyhow!("Invalid Vertex setting {}: {reason}", setting.source);
        if !setting.readable {
            return Err(invalid(
                "must be an object of \"<key>\": \"<location>\" members, for example {\"mistralai\": \"europe-west4\"}".to_string(),
            ));
        }
        let mut entries: Vec<(String, String)> = Vec::with_capacity(setting.entries.len());
        for (index, entry) in setting.entries.iter().enumerate() {
            let key = &entry.key;
            validate_location_override_key(key)
                .map_err(|reason| invalid(format!("entry {}: {reason}", index + 1)))?;
            let Some(location) = &entry.location else {
                return Err(invalid(format!(
                    "entry {key:?} has no location; an entry is <key>=<location> in the variable and \"<key>\": \"<location>\" in config.json"
                )));
            };
            validate_vertex_location(location)
                .map_err(|reason| invalid(format!("the location of entry {key:?} {reason}")))?;
            if entries.iter().any(|(seen, _)| seen == key) {
                return Err(invalid(format!("entry {key:?} is listed more than once")));
            }
            entries.push((key.clone(), location.clone()));
        }
        Ok(VertexLocationOverrides { entries })
    }
}

fn validated_vertex_setting(
    setting: &VertexSetting,
    validate: fn(&str) -> Result<(), &'static str>,
) -> anyhow::Result<&str> {
    validate(&setting.value)
        .map_err(|reason| anyhow::anyhow!("Invalid Vertex setting {}: {reason}", setting.source))?;
    Ok(&setting.value)
}

fn validate_vertex_project(value: &str) -> Result<(), &'static str> {
    let mut bytes = value.bytes();
    let starts_alphanumeric = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphanumeric());
    if !starts_alphanumeric
        || !bytes
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
    {
        return Err(
            "must be a Google Cloud project ID: letters, digits, '.', '_', ':' and '-', starting with a letter or digit",
        );
    }
    Ok(())
}

fn validate_vertex_location(value: &str) -> Result<(), &'static str> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(
            "must be a Vertex AI location such as global, us-central1 or europe-west4: lowercase letters, digits and '-'",
        );
    }
    Ok(())
}

/// Longest publisher accepted in a location override key.
const MAX_OVERRIDE_PUBLISHER_BYTES: usize = 64;

/// Longest model ID accepted in a location override key; the same bound as
/// a requested model ID.
const MAX_OVERRIDE_MODEL_ID_BYTES: usize = 256;

fn validate_location_override_key(key: &str) -> Result<(), &'static str> {
    let (publisher, model) = match key.split_once('/') {
        Some((publisher, model)) => (publisher, Some(model)),
        None => (key, None),
    };
    let starts_alphanumeric = |value: &str| {
        value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
    };
    let publisher_valid = starts_alphanumeric(publisher)
        && publisher.len() <= MAX_OVERRIDE_PUBLISHER_BYTES
        && publisher
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
    let model_valid = model.is_none_or(|model| {
        starts_alphanumeric(model)
            && model.len() <= MAX_OVERRIDE_MODEL_ID_BYTES
            && model
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-@".contains(&byte))
    });
    if !publisher_valid || !model_valid {
        return Err(
            "the key must be <publisher> or <publisher>/<model-id>: a publisher is lowercase letters, digits and '-', a model ID is letters, digits, '.', '_', '-' and '@', and each starts with a letter or digit",
        );
    }
    if publisher == "endpoint" {
        return Err("a deployed endpoint (endpoint/<name>) takes no location override");
    }
    Ok(())
}

/// Reads `CCP_VERTEX_LOCATION_OVERRIDES`: entries separated by `,`, each
/// `<key>=<location>`. Empty entries, such as one after a trailing comma,
/// are skipped.
fn location_override_env_entries(raw: &str) -> Vec<VertexLocationOverrideEntry> {
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| match entry.split_once('=') {
            Some((key, location)) => VertexLocationOverrideEntry {
                key: key.trim().to_string(),
                location: Some(location.trim().to_string()),
            },
            None => VertexLocationOverrideEntry {
                key: entry.to_string(),
                location: None,
            },
        })
        .collect()
}

/// Reads `vertex.locationOverrides`. `null` and an empty object set
/// nothing; any other value that is not an object is kept as unreadable so
/// the accessor can report it.
fn location_override_file_setting(
    value: &serde_json::Value,
) -> Option<VertexLocationOverridesSetting> {
    let source = VertexSettingSource::ConfigFile(VERTEX_LOCATION_OVERRIDES_KEY);
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Object(members) if members.is_empty() => None,
        serde_json::Value::Object(members) => Some(VertexLocationOverridesSetting {
            entries: members
                .iter()
                .map(|(key, location)| VertexLocationOverrideEntry {
                    key: key.trim().to_string(),
                    location: location
                        .as_str()
                        .map(|location| location.trim().to_string()),
                })
                .collect(),
            readable: true,
            source,
        }),
        _ => Some(VertexLocationOverridesSetting {
            entries: Vec::new(),
            readable: false,
            source,
        }),
    }
}

/// Resolves the Vertex settings from `env` and `config_dir/config.json`.
///
/// Each value is taken from the first source that sets it to something other
/// than blank: `CCP_VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `vertex.project`
/// for the project; `CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION`,
/// `vertex.location`, then `global` for the location; `CCP_VERTEX_BASE_URL`,
/// `vertex.baseUrl` for the base URL; `CCP_VERTEX_LOCATION_OVERRIDES`,
/// `vertex.locationOverrides` for the location overrides, where the variable
/// replaces the file's overrides as a whole rather than being merged with
/// them. Reading the Google Cloud variables lets
/// a shell already set up for `gcloud` or the Gemini CLI work unchanged, while
/// the `CCP_` names can point the proxy elsewhere without touching them.
pub fn resolve_vertex_config(
    env: &HashMap<String, String>,
    config_dir: &Path,
) -> ResolvedVertexConfig {
    let file = read_file_config(config_dir).and_then(|file| file.vertex);
    let pick = |env_names: &[&'static str],
                file_key: &'static str,
                file_value: Option<&String>|
     -> Option<VertexSetting> {
        let set = |raw: &String| {
            let trimmed = raw.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        };
        env_names
            .iter()
            .find_map(|name| {
                env.get(*name).and_then(set).map(|value| VertexSetting {
                    value,
                    source: VertexSettingSource::Env(name),
                })
            })
            .or_else(|| {
                file_value.and_then(set).map(|value| VertexSetting {
                    value,
                    source: VertexSettingSource::ConfigFile(file_key),
                })
            })
    };

    let project_setting = pick(
        &["CCP_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT"],
        "vertex.project",
        file.as_ref().and_then(|config| config.project.as_ref()),
    );
    let location_setting = pick(
        &["CCP_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"],
        "vertex.location",
        file.as_ref().and_then(|config| config.location.as_ref()),
    )
    .unwrap_or_else(|| VertexSetting {
        value: VERTEX_DEFAULT_LOCATION.to_string(),
        source: VertexSettingSource::Default,
    });
    let base_url_setting = pick(
        &["CCP_VERTEX_BASE_URL"],
        "vertex.baseUrl",
        file.as_ref().and_then(|config| config.base_url.as_ref()),
    );

    let location_overrides_setting = env
        .get(VERTEX_LOCATION_OVERRIDES_ENV)
        .map(|raw| location_override_env_entries(raw))
        .filter(|entries| !entries.is_empty())
        .map(|entries| VertexLocationOverridesSetting {
            entries,
            readable: true,
            source: VertexSettingSource::Env(VERTEX_LOCATION_OVERRIDES_ENV),
        })
        .or_else(|| {
            file.as_ref()
                .and_then(|config| config.location_overrides.as_ref())
                .and_then(location_override_file_setting)
        });

    ResolvedVertexConfig {
        project_setting,
        location_setting,
        base_url_setting,
        location_overrides_setting,
    }
}

/// Vertex settings for this process: its environment and the config file in
/// the resolved config directory.
pub fn vertex_config() -> ResolvedVertexConfig {
    let env: HashMap<_, _> = std::env::vars().collect();
    resolve_vertex_config(&env, &paths::config_dir())
}

/// Host name that serves Vertex AI in `location`: the global endpoint for
/// `global`, the multi-region endpoints for `us` and `eu`, and the regional
/// endpoint for everything else. `location` must already be validated.
pub fn vertex_host(location: &str) -> String {
    match location {
        "global" => "aiplatform.googleapis.com".to_string(),
        "us" | "eu" => format!("aiplatform.{location}.rep.googleapis.com"),
        region => format!("{region}-aiplatform.googleapis.com"),
    }
}

/// `config.json` key holding the deployed endpoints: an object of
/// `"<name>": {...}` members, where `<name>` is what a request names as
/// `vertex/endpoint/<name>`.
pub const VERTEX_ENDPOINTS_KEY: &str = "vertex.endpoints";

/// The keys an endpoint entry may have.
const VERTEX_ENDPOINT_KEYS: [&str; 6] =
    ["endpoint", "location", "model", "project", "host", "tools"];

/// Longest endpoint name. The name is part of a model name and of messages.
const MAX_VERTEX_ENDPOINT_NAME_BYTES: usize = 64;

/// Longest endpoint ID. The ID becomes one segment of the request path.
const MAX_VERTEX_ENDPOINT_ID_BYTES: usize = 128;

/// Longest `model` value. It is sent in the request body only.
const MAX_VERTEX_ENDPOINT_MODEL_BYTES: usize = 512;

/// Host suffix of a Vertex AI dedicated endpoint: Google documents its DNS
/// name as `{endpoint_id}.{region}-{uid}.prediction.vertexai.goog`.
pub const VERTEX_DEDICATED_HOST_SUFFIX: &str = ".prediction.vertexai.goog";

/// One deployed endpoint, validated. The proxy only calls it; the user
/// deploys the model and creates the endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexEndpointConfig {
    /// The endpoint ID, the last segment of its resource name.
    pub endpoint: String,
    /// The endpoint's region. Never `global`.
    pub location: String,
    /// The `model` value the serving container accepts, such as vLLM's
    /// `--served-model-name`.
    pub model: String,
    /// The project of the endpoint, when it is not the Vertex project.
    pub project: Option<String>,
    /// The dedicated endpoint URL as `scheme://host[:port]`, when the user
    /// set one. Without it the proxy reads the endpoint once to learn it.
    pub host: Option<String>,
    /// False when requests that carry tools are refused before they are
    /// sent.
    pub tools: bool,
}

/// One entry of `vertex.endpoints` as read: its name and either the
/// validated settings or an error that names the setting and the entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexEndpointEntry {
    /// The member name as written in `config.json`.
    pub name: String,
    /// The settings, or the message saying what is wrong with them.
    pub config: Result<VertexEndpointConfig, String>,
}

/// The deployed endpoints of `config.json`. Each entry is validated on its
/// own, so a broken entry makes only requests to that endpoint fail.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VertexEndpoints {
    /// The entries in the order they were read.
    pub entries: Vec<VertexEndpointEntry>,
    /// Set when `vertex.endpoints` is present but is not an object; there
    /// are no entries then.
    pub unreadable: bool,
}

impl VertexEndpoints {
    /// The entry named `name`, if any.
    pub fn get(&self, name: &str) -> Option<&VertexEndpointEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// The names of the entries whose settings are valid, in the order
    /// they were read.
    pub fn usable_names(&self) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .filter(|entry| entry.config.is_ok())
            .map(|entry| entry.name.as_str())
    }

    /// The first problem: the setting is not an object, or an entry is
    /// invalid.
    pub fn first_error(&self) -> Option<anyhow::Error> {
        if self.unreadable {
            return Some(anyhow::anyhow!(unreadable_endpoints_message()));
        }
        self.entries
            .iter()
            .find_map(|entry| entry.config.as_ref().err())
            .map(|message| anyhow::anyhow!("{message}"))
    }

    /// The startup summary line: the names of the entries, or `None` when
    /// the setting is absent. Only names are listed: a host receives the
    /// access token.
    fn summary_line(&self) -> Option<String> {
        if self.unreadable {
            return Some(format!("{VERTEX_ENDPOINTS_KEY} (config, not an object)"));
        }
        if self.entries.is_empty() {
            return None;
        }
        let names: Vec<&str> = self
            .entries
            .iter()
            .map(|entry| {
                if is_vertex_endpoint_name(&entry.name) {
                    entry.name.as_str()
                } else {
                    "[invalid name]"
                }
            })
            .collect();
        Some(format!("{VERTEX_ENDPOINTS_KEY}: {}", names.join(", ")))
    }
}

fn unreadable_endpoints_message() -> String {
    format!(
        "Invalid Vertex setting {VERTEX_ENDPOINTS_KEY} in config.json: must be an object of \"<name>\": {{\"endpoint\", \"location\", \"model\"}} members"
    )
}

/// Whether `name` can be used as `vertex/endpoint/<name>`: letters, digits,
/// `_` and `-`, at most 64 bytes.
pub fn is_vertex_endpoint_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_VERTEX_ENDPOINT_NAME_BYTES
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

/// Reads `vertex.endpoints` from `config_dir/config.json`. There is no
/// environment variable: an endpoint takes several values at once.
pub fn resolve_vertex_endpoints(config_dir: &Path) -> VertexEndpoints {
    let value = read_file_config(config_dir)
        .and_then(|file| file.vertex)
        .and_then(|vertex| vertex.endpoints);
    vertex_endpoints_from_value(value.as_ref())
}

/// The deployed endpoints of this process's config file.
pub fn vertex_endpoints() -> VertexEndpoints {
    resolve_vertex_endpoints(&paths::config_dir())
}

/// Validates every member of `vertex.endpoints`. `null` and absence set
/// nothing.
fn vertex_endpoints_from_value(value: Option<&serde_json::Value>) -> VertexEndpoints {
    match value {
        None | Some(serde_json::Value::Null) => VertexEndpoints::default(),
        Some(serde_json::Value::Object(members)) => VertexEndpoints {
            entries: members
                .iter()
                .map(|(name, value)| VertexEndpointEntry {
                    name: name.clone(),
                    config: validate_vertex_endpoint(name, value),
                })
                .collect(),
            unreadable: false,
        },
        Some(_) => VertexEndpoints {
            entries: Vec::new(),
            unreadable: true,
        },
    }
}

/// One entry's settings, or a message naming the setting and the entry.
/// The value of `host` is never repeated, because a rejected value may
/// carry credentials.
fn validate_vertex_endpoint(
    name: &str,
    value: &serde_json::Value,
) -> Result<VertexEndpointConfig, String> {
    if !is_vertex_endpoint_name(name) {
        return Err(format!(
            "Invalid Vertex setting {VERTEX_ENDPOINTS_KEY} in config.json: endpoint name {name:?} must be letters, digits, '_' and '-', at most {MAX_VERTEX_ENDPOINT_NAME_BYTES} of them"
        ));
    }
    let invalid = |reason: String| {
        format!(
            "Invalid Vertex setting {VERTEX_ENDPOINTS_KEY} in config.json: endpoint {name:?}: {reason}"
        )
    };
    let serde_json::Value::Object(fields) = value else {
        return Err(invalid(
            "must be an object with the keys endpoint, location and model, and optionally project, host and tools".to_string(),
        ));
    };
    if let Some(key) = fields
        .keys()
        .find(|key| !VERTEX_ENDPOINT_KEYS.contains(&key.as_str()))
    {
        return Err(invalid(format!(
            "unknown key {key:?}; the keys are endpoint, location, model, project, host and tools"
        )));
    }
    let text = |key: &str| -> Result<Option<String>, String> {
        match fields.get(key) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(raw)) if !raw.trim().is_empty() => {
                Ok(Some(raw.trim().to_string()))
            }
            Some(_) => Err(invalid(format!("{key} must be a non-empty string"))),
        }
    };
    let required = |key: &str| -> Result<String, String> {
        text(key)?.ok_or_else(|| invalid(format!("{key} is required")))
    };

    let endpoint = required("endpoint")?;
    validate_vertex_endpoint_id(&endpoint).map_err(|reason| invalid(reason.to_string()))?;
    let location = required("location")?;
    validate_vertex_location(&location).map_err(|reason| invalid(format!("location {reason}")))?;
    if location == VERTEX_DEFAULT_LOCATION {
        return Err(invalid(
            "location must be the endpoint's region, such as us-central1; an endpoint is never in global".to_string(),
        ));
    }
    let model = required("model")?;
    if model.len() > MAX_VERTEX_ENDPOINT_MODEL_BYTES || model.chars().any(char::is_control) {
        return Err(invalid(format!(
            "model must be at most {MAX_VERTEX_ENDPOINT_MODEL_BYTES} bytes without control characters"
        )));
    }
    let project = text("project")?;
    if let Some(project) = &project {
        validate_vertex_project(project).map_err(|reason| invalid(format!("project {reason}")))?;
    }
    let host = text("host")?
        .map(|host| validate_vertex_endpoint_host(&host))
        .transpose()
        .map_err(|reason| invalid(format!("host {reason}")))?;
    let tools = match fields.get("tools") {
        None | Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Bool(tools)) => *tools,
        Some(_) => return Err(invalid("tools must be true or false".to_string())),
    };
    Ok(VertexEndpointConfig {
        endpoint,
        location,
        model,
        project,
        host,
        tools,
    })
}

/// An endpoint ID: lowercase letters, digits and `-`, starting with a letter
/// or digit. Vertex AI generates numeric IDs; a user-chosen ID uses the same
/// characters.
fn validate_vertex_endpoint_id(value: &str) -> Result<(), &'static str> {
    let starts_alphanumeric = value
        .bytes()
        .next()
        .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit());
    if !starts_alphanumeric
        || value.len() > MAX_VERTEX_ENDPOINT_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(
            "endpoint must be the endpoint ID: lowercase letters, digits and '-', starting with a letter or digit",
        );
    }
    Ok(())
}

/// A configured `host`: the dedicated endpoint URL, or its DNS name alone,
/// which gets `https://` in front. Checked as
/// [`validate_vertex_dedicated_url`] checks the URL the endpoint reports.
fn validate_vertex_endpoint_host(value: &str) -> Result<String, &'static str> {
    if value.contains("://") {
        validate_vertex_dedicated_url(value)
    } else {
        validate_vertex_dedicated_url(&format!("https://{value}"))
    }
}

/// A URL the access token may be sent to for a deployed endpoint, returned
/// as `scheme://host[:port]`: https to a host under
/// `.prediction.vertexai.goog`, or a loopback host over https or plain
/// http. No user info, path, query or fragment: the request path is
/// appended once, by the proxy.
///
/// # Errors
///
/// A fixed reason, so callers can report it without repeating a value that
/// may carry a credential.
pub fn validate_vertex_dedicated_url(value: &str) -> Result<String, &'static str> {
    let url = url::Url::parse(value).map_err(|_| "is not a valid URL")?;
    let (loopback, dedicated) = match url.host() {
        Some(url::Host::Ipv4(ip)) => (ip.is_loopback(), false),
        Some(url::Host::Ipv6(ip)) => (ip.is_loopback(), false),
        Some(url::Host::Domain(domain)) => {
            let domain = domain.strip_suffix('.').unwrap_or(domain);
            let dedicated = domain.len() > VERTEX_DEDICATED_HOST_SUFFIX.len()
                && domain
                    .to_ascii_lowercase()
                    .ends_with(VERTEX_DEDICATED_HOST_SUFFIX);
            (domain.eq_ignore_ascii_case("localhost"), dedicated)
        }
        None => return Err("has no host"),
    };
    match url.scheme() {
        "https" => {}
        "http" if loopback => {}
        _ => return Err("must use https (plain http only for a loopback host)"),
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("must not contain a user name or password");
    }
    if !loopback && !dedicated {
        return Err(
            "must be a dedicated endpoint host, https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a loopback host",
        );
    }
    if !matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some() {
        return Err("must be a scheme and a host only, without a path, query or fragment");
    }
    Ok(url.origin().ascii_serialization())
}

pub fn is_verbose() -> bool {
    log_verbose()
}

pub fn kimi_oauth_host() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_KIMI_OAUTH_HOST") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(kimi) = file.kimi
        && let Some(host) = kimi.oauth_host
    {
        return host;
    }
    "https://auth.kimi.com".to_string()
}

pub fn kimi_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_KIMI_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(kimi) = file.kimi
        && let Some(url) = kimi.base_url
    {
        return url;
    }
    "https://api.kimi.com/coding/v1".to_string()
}

pub fn kimi_user_agent(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_KIMI_USER_AGENT") {
        return raw.clone();
    }
    if let Some(raw) = env.get("CCP_USER_AGENT") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(kimi) = file.kimi
        && let Some(ua) = kimi.user_agent
    {
        return ua;
    }
    default.to_string()
}

// ---------------------------------------------------------------------------
// Codex config
// ---------------------------------------------------------------------------

pub fn codex_base_url(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_BASE_URL") {
        return raw.clone();
    }
    if let Some(raw) = env.get("CLAUDE_CODE_PROXY_CODEX_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(url) = codex.base_url
    {
        return url;
    }
    default.to_string()
}

pub fn codex_originator(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_ORIGINATOR") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(val) = codex.originator
    {
        return val;
    }
    default.to_string()
}

pub fn codex_user_agent(default: &str) -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_USER_AGENT") {
        return raw.clone();
    }
    if let Some(raw) = env.get("CCP_USER_AGENT") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(ua) = codex.user_agent
    {
        return ua;
    }
    default.to_string()
}

pub fn codex_previous_response_id() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_PREVIOUS_RESPONSE_ID") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(val) = codex.previous_response_id
    {
        return val;
    }
    false
}

pub fn codex_request_compression() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_REQUEST_COMPRESSION") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => return true,
            "0" | "false" | "no" | "off" => return false,
            _ => {}
        }
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.request_compression
    {
        return enabled;
    }
    false
}

pub fn codex_server_compaction() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_SERVER_COMPACTION") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => return true,
            "0" | "false" | "no" | "off" => return false,
            _ => {}
        }
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.server_compaction
    {
        return enabled;
    }
    false
}

pub fn codex_responses_api() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_RESPONSES_API") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.responses_api
    {
        return enabled;
    }
    false
}

pub fn codex_full_lane() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_FULL_LANE") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.full_lane
    {
        return enabled;
    }
    false
}

pub fn codex_images_api() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_IMAGES_API") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.images_api
    {
        return enabled;
    }
    false
}

pub fn codex_transcriptions_api() -> bool {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_TRANSCRIPTIONS_API") {
        return matches!(raw.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(enabled) = codex.transcriptions_api
    {
        return enabled;
    }
    false
}

pub fn codex_images_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_IMAGES_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(url) = codex.images_base_url
    {
        return url;
    }
    "https://chatgpt.com/backend-api/codex".to_string()
}

pub fn codex_service_tier() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_SERVICE_TIER") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
    {
        return codex.service_tier;
    }
    None
}

pub fn codex_effort() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_EFFORT") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
    {
        return codex.effort;
    }
    None
}

pub fn codex_reasoning_summary() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env
        .get("CCP_CODEX_REASONING_SUMMARY")
        .filter(|raw| !raw.is_empty())
    {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(summary) = codex.reasoning_summary.filter(|raw| !raw.is_empty())
    {
        return Some(summary);
    }
    None
}

pub fn codex_model() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_MODEL") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
    {
        return codex.model;
    }
    None
}

pub fn auto_review_model() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env
        .get("CCP_AUTO_REVIEW_MODEL")
        .filter(|raw| !raw.is_empty())
    {
        return Some(raw.clone());
    }
    read_file_config(&paths::config_dir())
        .and_then(|file| file.auto_review_model)
        .filter(|model| !model.is_empty())
}

// ---------------------------------------------------------------------------
// Codex transport config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexTransport {
    Http,
    WebSocket,
    Auto,
}

impl CodexTransport {
    pub fn as_str(self) -> &'static str {
        match self {
            CodexTransport::Http => "http",
            CodexTransport::WebSocket => "websocket",
            CodexTransport::Auto => "auto",
        }
    }
}

fn parse_codex_transport(raw: &str) -> Option<CodexTransport> {
    match raw {
        "http" => Some(CodexTransport::Http),
        "websocket" => Some(CodexTransport::WebSocket),
        "auto" => Some(CodexTransport::Auto),
        _ => None,
    }
}

pub fn codex_transport() -> CodexTransport {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CODEX_TRANSPORT")
        && let Some(transport) = parse_codex_transport(raw)
    {
        return transport;
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(transport) = codex.transport.as_deref().and_then(parse_codex_transport)
    {
        return transport;
    }
    if codex_request_compression() {
        return CodexTransport::Http;
    }
    CodexTransport::Auto
}

/// How long an HTTP-transport request waits for the Codex response headers.
///
/// Codex withholds the response head until the model produces its first
/// output, so the wait tracks how much reasoning the request asks for rather
/// than the health of the connection. A large request to a high-effort model
/// can hold the head for minutes, and the timeout firing fails the request
/// outright, so the default is generous. Lower it only where a request that
/// slow is better failed than waited out. Values below
/// `CODEX_MIN_HEADER_TIMEOUT_MS` are ignored rather than clamped, so a typo
/// does not hide behind a working default.
pub const CODEX_DEFAULT_HEADER_TIMEOUT_MS: u64 = 300_000;
pub const CODEX_MIN_HEADER_TIMEOUT_MS: u64 = 1_000;

pub fn codex_header_timeout_ms() -> u64 {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(ms) = env
        .get("CCP_CODEX_HEADER_TIMEOUT_MS")
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|ms| *ms >= CODEX_MIN_HEADER_TIMEOUT_MS)
    {
        return ms;
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(codex) = file.codex
        && let Some(ms) = codex
            .header_timeout_ms
            .filter(|ms| *ms >= CODEX_MIN_HEADER_TIMEOUT_MS)
    {
        return ms;
    }
    CODEX_DEFAULT_HEADER_TIMEOUT_MS
}

// ---------------------------------------------------------------------------
// Cursor config
// ---------------------------------------------------------------------------

pub fn cursor_base_url() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CURSOR_BASE_URL") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(cursor) = file.cursor
        && let Some(url) = cursor.base_url
    {
        return url;
    }
    "https://api2.cursor.sh".to_string()
}

pub fn cursor_client_version() -> String {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CURSOR_CLIENT_VERSION") {
        return raw.clone();
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(cursor) = file.cursor
        && let Some(version) = cursor.client_version
    {
        return version;
    }
    detect_cursor_agent_version().unwrap_or_else(|| "cli-2026.07.23-e383d2b".to_string())
}

fn detect_cursor_agent_version() -> Option<String> {
    let output = std::process::Command::new("cursor-agent")
        .arg("--version")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8(output.stdout).ok()?;
    let version = version.lines().next()?.trim();
    if version.is_empty() {
        return None;
    }
    Some(if version.starts_with("cli-") {
        version.to_string()
    } else {
        format!("cli-{version}")
    })
}

pub fn cursor_agent_bundle() -> Option<String> {
    let env: HashMap<_, _> = std::env::vars().collect();
    if let Some(raw) = env.get("CCP_CURSOR_AGENT_BUNDLE") {
        return Some(raw.clone());
    }
    let config_dir = paths::config_dir();
    if let Some(file) = read_file_config(&config_dir)
        && let Some(cursor) = file.cursor
        && let Some(bundle) = cursor.agent_bundle
    {
        return Some(bundle);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    /// Clears the environment knobs the config accessors read and points the
    /// config dir at `config`, so neither the developer's shell nor their real
    /// config.json can leak into the test. Every change is restored when the
    /// guards drop, including `CCP_CONFIG_DIR`: a plain `remove_var` here would
    /// strip a `CCP_CONFIG_DIR` the developer exported to isolate the whole
    /// run, and every later test in the process that reads config through
    /// `paths::config_dir()` would fall back to the real config file.
    fn isolated_env(config: &tempfile::TempDir) -> Vec<EnvGuard> {
        let mut guards = vec![
            EnvGuard::unset("CCP_BIND_ADDRESS"),
            EnvGuard::unset("CCP_CODEX_TRANSPORT"),
            EnvGuard::unset("CCP_LOG_VERBOSE"),
            EnvGuard::unset("CCP_LOG_STDERR"),
            EnvGuard::unset("CCP_CODEX_REASONING_SUMMARY"),
            EnvGuard::unset("CCP_CODEX_REQUEST_COMPRESSION"),
            EnvGuard::unset("CCP_CODEX_SERVER_COMPACTION"),
            EnvGuard::unset("CCP_CODEX_RESPONSES_API"),
            EnvGuard::unset("CCP_CODEX_FULL_LANE"),
            EnvGuard::unset("CCP_CODEX_IMAGES_API"),
            EnvGuard::unset("CCP_CODEX_IMAGES_BASE_URL"),
            EnvGuard::unset("CCP_CODEX_TRANSCRIPTIONS_API"),
            EnvGuard::unset("CCP_CODEX_HEADER_TIMEOUT_MS"),
            EnvGuard::unset("CCP_AUTO_REVIEW_MODEL"),
            EnvGuard::unset("CCP_GROK_CLIENT_VERSION"),
            EnvGuard::unset("CCP_MUSE_BASE_URL"),
            EnvGuard::unset("CCP_MUSE_OAUTH_HOST"),
            EnvGuard::unset("CCP_MUSE_MINT_BASE_URL"),
            EnvGuard::unset("CCP_MUSE_USER_AGENT"),
        ];
        guards.push(EnvGuard::set("CCP_CONFIG_DIR", config.path()));
        guards
    }

    fn config_env(config: &tempfile::TempDir) -> HashMap<String, String> {
        HashMap::from([(
            "CCP_CONFIG_DIR".to_string(),
            config.path().to_string_lossy().into_owned(),
        )])
    }

    #[test]
    fn grok_client_version_defaults_and_overrides() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        assert_eq!(grok_client_version(), "1.0.13");

        std::fs::write(
            config.path().join("config.json"),
            r#"{"grok":{"clientVersion":"1.0.14"}}"#,
        )
        .unwrap();
        assert_eq!(grok_client_version(), "1.0.14");

        let _version = EnvGuard::set("CCP_GROK_CLIENT_VERSION", "1.0.15");
        assert_eq!(grok_client_version(), "1.0.15");
    }

    #[test]
    fn muse_settings_default_then_config_then_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        assert_eq!(muse_base_url().unwrap(), "https://api.meta.ai/v1");
        assert_eq!(muse_oauth_host().unwrap(), "https://auth.meta.com");
        assert_eq!(muse_mint_base_url().unwrap(), "https://api.meta.ai");
        assert_eq!(
            muse_user_agent().unwrap(),
            format!("claude-code-proxy/{}", env!("CARGO_PKG_VERSION"))
        );

        std::fs::write(
            config.path().join("config.json"),
            r#"{"muse":{"baseUrl":"https://file.example/v1","oauthHost":"https://auth.file.example","mintBaseUrl":"https://mint.file.example","userAgent":"file-agent/1"}}"#,
        )
        .unwrap();
        assert_eq!(muse_base_url().unwrap(), "https://file.example/v1");
        assert_eq!(muse_oauth_host().unwrap(), "https://auth.file.example");
        assert_eq!(muse_mint_base_url().unwrap(), "https://mint.file.example");
        assert_eq!(muse_user_agent().unwrap(), "file-agent/1");

        let _base = EnvGuard::set("CCP_MUSE_BASE_URL", "http://127.0.0.1:9/v1");
        let _oauth = EnvGuard::set("CCP_MUSE_OAUTH_HOST", "http://127.0.0.1:9");
        let _mint = EnvGuard::set("CCP_MUSE_MINT_BASE_URL", "http://127.0.0.1:10");
        let _ua = EnvGuard::set("CCP_MUSE_USER_AGENT", "env-agent/2");
        assert_eq!(muse_base_url().unwrap(), "http://127.0.0.1:9/v1");
        assert_eq!(muse_oauth_host().unwrap(), "http://127.0.0.1:9");
        assert_eq!(muse_mint_base_url().unwrap(), "http://127.0.0.1:10");
        assert_eq!(muse_user_agent().unwrap(), "env-agent/2");
    }

    #[test]
    fn muse_settings_partial_config_keeps_other_defaults() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        std::fs::write(
            config.path().join("config.json"),
            r#"{"muse":{"oauthHost":"https://auth.file.example"}}"#,
        )
        .unwrap();
        assert_eq!(muse_oauth_host().unwrap(), "https://auth.file.example");
        assert_eq!(muse_base_url().unwrap(), "https://api.meta.ai/v1");
        assert_eq!(muse_mint_base_url().unwrap(), "https://api.meta.ai");
    }

    #[test]
    fn muse_endpoint_synthetic_unsafe_values_are_rejected_with_the_setting_name() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        type Getter = fn() -> anyhow::Result<String>;
        let getters: [(&str, &str, Getter); 3] = [
            ("CCP_MUSE_BASE_URL", "baseUrl", muse_base_url),
            ("CCP_MUSE_OAUTH_HOST", "oauthHost", muse_oauth_host),
            ("CCP_MUSE_MINT_BASE_URL", "mintBaseUrl", muse_mint_base_url),
        ];
        const NOT_HTTPS: &str = "must use https (plain http only for a loopback host)";
        const USER_INFO: &str = "must not contain a user name or password";
        let rejected = [
            ("http://relay.example/v1", NOT_HTTPS),
            ("http://10.0.0.1/v1", NOT_HTTPS),
            ("ftp://127.0.0.1/v1", NOT_HTTPS),
            ("https://user:synthetic-pass@api.meta.ai/v1", USER_INFO),
            ("https://user@api.meta.ai/v1", USER_INFO),
            ("http://user@127.0.0.1:9/v1", USER_INFO),
            ("api.meta.ai/v1", "not a valid URL"),
        ];
        for (env_name, file_key, getter) in getters {
            for (value, reason) in rejected {
                let guard = EnvGuard::set(env_name, value);
                let error = getter().unwrap_err().to_string();
                assert_eq!(
                    error,
                    format!("Invalid Muse setting {env_name}: {reason}"),
                    "{value}"
                );
                assert!(!error.contains("synthetic-pass"), "{error}");
                drop(guard);

                std::fs::write(
                    config.path().join("config.json"),
                    serde_json::json!({"muse": {file_key: value}}).to_string(),
                )
                .unwrap();
                assert_eq!(
                    getter().unwrap_err().to_string(),
                    format!("Invalid Muse setting muse.{file_key} in config.json: {reason}"),
                    "{value}"
                );
                std::fs::remove_file(config.path().join("config.json")).unwrap();
            }
            for accepted in [
                "http://127.0.0.1:9/v1",
                "http://127.1.2.3:9",
                "http://[::1]:9/v1",
                "http://localhost:9",
                "http://localhost.:9",
                "http://LocalHost.:9/v1",
                "https://relay.example:8443/v1",
            ] {
                let _guard = EnvGuard::set(env_name, accepted);
                assert_eq!(getter().unwrap(), accepted);
            }
        }
    }

    #[test]
    fn muse_endpoint_synthetic_empty_values_fall_through_to_the_next_source() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        type Getter = fn() -> anyhow::Result<String>;
        let getters: [(&str, &str, Getter, &str); 3] = [
            (
                "CCP_MUSE_BASE_URL",
                "baseUrl",
                muse_base_url,
                "https://api.meta.ai/v1",
            ),
            (
                "CCP_MUSE_OAUTH_HOST",
                "oauthHost",
                muse_oauth_host,
                "https://auth.meta.com",
            ),
            (
                "CCP_MUSE_MINT_BASE_URL",
                "mintBaseUrl",
                muse_mint_base_url,
                "https://api.meta.ai",
            ),
        ];
        let path = config.path().join("config.json");
        for (env_name, file_key, getter, default) in getters {
            for blank in ["", "   ", "\t\n"] {
                // Blank environment, nothing in the file: the default.
                let env = EnvGuard::set(env_name, blank);
                assert_eq!(getter().unwrap(), default, "{env_name}={blank:?}");

                // Blank environment, a file value: the file value.
                std::fs::write(
                    &path,
                    serde_json::json!({"muse": {file_key: "https://file.example/v1"}}).to_string(),
                )
                .unwrap();
                assert_eq!(getter().unwrap(), "https://file.example/v1", "{env_name}");

                // Blank in both: the default.
                std::fs::write(
                    &path,
                    serde_json::json!({"muse": {file_key: blank}}).to_string(),
                )
                .unwrap();
                assert_eq!(getter().unwrap(), default, "{file_key}={blank:?}");
                drop(env);
                assert_eq!(getter().unwrap(), default, "{file_key}={blank:?}");
                std::fs::remove_file(&path).unwrap();
            }
        }
    }

    #[test]
    fn muse_endpoint_synthetic_padded_values_are_returned_trimmed() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        type Getter = fn() -> anyhow::Result<String>;
        let getters: [(&str, &str, Getter); 3] = [
            ("CCP_MUSE_BASE_URL", "baseUrl", muse_base_url),
            ("CCP_MUSE_OAUTH_HOST", "oauthHost", muse_oauth_host),
            ("CCP_MUSE_MINT_BASE_URL", "mintBaseUrl", muse_mint_base_url),
        ];
        let path = config.path().join("config.json");
        for (env_name, file_key, getter) in getters {
            let env = EnvGuard::set(env_name, " https://relay.example/v1 \t\n");
            assert_eq!(getter().unwrap(), "https://relay.example/v1", "{env_name}");
            drop(env);
            std::fs::write(
                &path,
                serde_json::json!({"muse": {file_key: "\thttp://localhost.:9 "}}).to_string(),
            )
            .unwrap();
            assert_eq!(getter().unwrap(), "http://localhost.:9", "{file_key}");
            std::fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn muse_user_agent_synthetic_printable_ascii_up_to_256_characters_is_accepted() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        let longest = "a".repeat(256);
        for (value, expected) in [
            // The Muse CLI's own User-Agents on the Model API and the mint.
            (
                "muse-build/1.4.14 (non-interactive; macos-aarch64; build dec7d7497fecdad135ec5835a8161dbc92e55b58)",
                "muse-build/1.4.14 (non-interactive; macos-aarch64; build dec7d7497fecdad135ec5835a8161dbc92e55b58)",
            ),
            ("muse-build/0.1.0", "muse-build/0.1.0"),
            ("  curl/8.7.1 ", "curl/8.7.1"),
            (
                "~!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./",
                "~!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./",
            ),
            (longest.as_str(), longest.as_str()),
        ] {
            let _ua = EnvGuard::set("CCP_MUSE_USER_AGENT", value);
            assert_eq!(muse_user_agent().unwrap(), expected, "{value:?}");
        }
    }

    #[test]
    fn muse_user_agent_synthetic_invalid_values_are_rejected_with_the_setting_name() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        const TOO_LONG: &str = "must be at most 256 characters";
        const NOT_PRINTABLE: &str = "must contain only printable ASCII characters";
        let path = config.path().join("config.json");
        let too_long = format!("synthetic-marker-{}", "a".repeat(240));
        for (value, reason) in [
            (too_long.as_str(), TOO_LONG),
            ("synthetic-marker\r\nX-Injected: 1", NOT_PRINTABLE),
            ("synthetic-marker\tagent", NOT_PRINTABLE),
            ("synthetic-marker\u{7f}", NOT_PRINTABLE),
            ("synthetic-marker\u{1b}[2J", NOT_PRINTABLE),
            ("synthetic-marker/\u{e9}", NOT_PRINTABLE),
        ] {
            let guard = EnvGuard::set("CCP_MUSE_USER_AGENT", value);
            let error = muse_user_agent().unwrap_err().to_string();
            assert_eq!(
                error,
                format!("Invalid Muse setting CCP_MUSE_USER_AGENT: {reason}"),
                "{value:?}"
            );
            assert!(!error.contains("synthetic-marker"), "{error}");
            drop(guard);

            std::fs::write(
                &path,
                serde_json::json!({"muse": {"userAgent": value}}).to_string(),
            )
            .unwrap();
            assert_eq!(
                muse_user_agent().unwrap_err().to_string(),
                format!("Invalid Muse setting muse.userAgent in config.json: {reason}"),
                "{value:?}"
            );
            std::fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn muse_user_agent_synthetic_blank_values_fall_through_to_the_next_source() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        let default = format!("claude-code-proxy/{}", env!("CARGO_PKG_VERSION"));
        let path = config.path().join("config.json");
        for blank in ["", "   ", "\t\n"] {
            let env = EnvGuard::set("CCP_MUSE_USER_AGENT", blank);
            assert_eq!(muse_user_agent().unwrap(), default, "{blank:?}");
            std::fs::write(
                &path,
                serde_json::json!({"muse": {"userAgent": "file-agent/1"}}).to_string(),
            )
            .unwrap();
            assert_eq!(muse_user_agent().unwrap(), "file-agent/1", "{blank:?}");
            std::fs::write(
                &path,
                serde_json::json!({"muse": {"userAgent": blank}}).to_string(),
            )
            .unwrap();
            assert_eq!(muse_user_agent().unwrap(), default, "{blank:?}");
            drop(env);
            assert_eq!(muse_user_agent().unwrap(), default, "{blank:?}");
            std::fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn muse_env_overrides_are_listed() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        let loaded = load_config_for_env(&config_env(&config));
        let before = config_override_summary_lines(&loaded);
        assert!(!before.iter().any(|line| line.starts_with("muse.")));

        let _base = EnvGuard::set("CCP_MUSE_BASE_URL", "http://127.0.0.1:9/v1");
        let _oauth = EnvGuard::set("CCP_MUSE_OAUTH_HOST", "http://127.0.0.1:9");
        let _mint = EnvGuard::set("CCP_MUSE_MINT_BASE_URL", "http://127.0.0.1:10");
        let _ua = EnvGuard::set("CCP_MUSE_USER_AGENT", "env-agent/2");
        let lines = config_override_summary_lines(&loaded);
        for expected in [
            "muse.baseUrl (env)",
            "muse.oauthHost (env)",
            "muse.mintBaseUrl (env)",
            "muse.userAgent (env)",
        ] {
            assert!(lines.iter().any(|line| line == expected), "{lines:?}");
        }
    }

    #[test]
    fn opencode_config_reads_file_and_env_precedence() {
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"opencode":{"apiKey":"file-key","baseUrl":"https://file.example/v1"}}"#,
        )
        .unwrap();
        let mut env = HashMap::new();
        let resolved = resolve_opencode_config(&env, config.path());
        assert_eq!(resolved.api_key.as_deref(), Some("file-key"));
        assert_eq!(resolved.api_key_source, Some("config.json"));
        assert_eq!(resolved.base_url, "https://file.example/v1");

        env.insert("OPENCODE_API_KEY".into(), "standard-key".into());
        let resolved = resolve_opencode_config(&env, config.path());
        assert_eq!(resolved.api_key.as_deref(), Some("standard-key"));
        assert_eq!(resolved.api_key_source, Some("OPENCODE_API_KEY"));

        env.insert("CCP_OPENCODE_API_KEY".into(), "ccp-key".into());
        env.insert(
            "CCP_OPENCODE_BASE_URL".into(),
            "https://env.example/v1".into(),
        );
        let resolved = resolve_opencode_config(&env, config.path());
        assert_eq!(resolved.api_key.as_deref(), Some("ccp-key"));
        assert_eq!(resolved.api_key_source, Some("CCP_OPENCODE_API_KEY"));
        assert_eq!(resolved.base_url, "https://env.example/v1");
    }

    fn vertex_env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    fn vertex_setting(value: &str, source: VertexSettingSource) -> Option<VertexSetting> {
        Some(VertexSetting {
            value: value.to_string(),
            source,
        })
    }

    const VERTEX_FILE: &str = r#"{"vertex":{"project":"file-project","location":"europe-west4","baseUrl":"https://file.example"}}"#;

    #[test]
    fn vertex_config_takes_ccp_env_over_google_env_over_config_file() {
        use VertexSettingSource::{ConfigFile, Default, Env};
        let with_file = tempfile::TempDir::new().unwrap();
        std::fs::write(with_file.path().join("config.json"), VERTEX_FILE).unwrap();
        let without_file = tempfile::TempDir::new().unwrap();
        let all_env = [
            ("CCP_VERTEX_PROJECT", "ccp-project"),
            ("GOOGLE_CLOUD_PROJECT", "google-project"),
            ("CCP_VERTEX_LOCATION", "us-central1"),
            ("GOOGLE_CLOUD_LOCATION", "asia-northeast1"),
            ("CCP_VERTEX_BASE_URL", "https://env.example"),
        ];
        let google_env = [
            ("GOOGLE_CLOUD_PROJECT", "google-project"),
            ("GOOGLE_CLOUD_LOCATION", "asia-northeast1"),
        ];
        // Name, environment, config directory, expected resolution.
        type Case<'a> = (
            &'a str,
            &'a [(&'a str, &'a str)],
            &'a tempfile::TempDir,
            ResolvedVertexConfig,
        );
        let cases: [Case; 5] = [
            (
                "nothing set: no project, the global location, no base URL",
                &[],
                &without_file,
                ResolvedVertexConfig {
                    project_setting: None,
                    location_setting: vertex_setting("global", Default).unwrap(),
                    base_url_setting: None,
                    location_overrides_setting: None,
                },
            ),
            (
                "config.json only",
                &[],
                &with_file,
                ResolvedVertexConfig {
                    project_setting: vertex_setting("file-project", ConfigFile("vertex.project")),
                    location_setting: vertex_setting("europe-west4", ConfigFile("vertex.location"))
                        .unwrap(),
                    base_url_setting: vertex_setting(
                        "https://file.example",
                        ConfigFile("vertex.baseUrl"),
                    ),
                    location_overrides_setting: None,
                },
            ),
            (
                "Google env over config.json; the base URL has no Google variable",
                &google_env,
                &with_file,
                ResolvedVertexConfig {
                    project_setting: vertex_setting("google-project", Env("GOOGLE_CLOUD_PROJECT")),
                    location_setting: vertex_setting(
                        "asia-northeast1",
                        Env("GOOGLE_CLOUD_LOCATION"),
                    )
                    .unwrap(),
                    base_url_setting: vertex_setting(
                        "https://file.example",
                        ConfigFile("vertex.baseUrl"),
                    ),
                    location_overrides_setting: None,
                },
            ),
            (
                "CCP env over Google env over config.json",
                &all_env,
                &with_file,
                ResolvedVertexConfig {
                    project_setting: vertex_setting("ccp-project", Env("CCP_VERTEX_PROJECT")),
                    location_setting: vertex_setting("us-central1", Env("CCP_VERTEX_LOCATION"))
                        .unwrap(),
                    base_url_setting: vertex_setting(
                        "https://env.example",
                        Env("CCP_VERTEX_BASE_URL"),
                    ),
                    location_overrides_setting: None,
                },
            ),
            (
                "Google env without config.json",
                &google_env,
                &without_file,
                ResolvedVertexConfig {
                    project_setting: vertex_setting("google-project", Env("GOOGLE_CLOUD_PROJECT")),
                    location_setting: vertex_setting(
                        "asia-northeast1",
                        Env("GOOGLE_CLOUD_LOCATION"),
                    )
                    .unwrap(),
                    base_url_setting: None,
                    location_overrides_setting: None,
                },
            ),
        ];
        for (name, env, config, expected) in cases {
            let resolved = resolve_vertex_config(&vertex_env(env), config.path());
            assert_eq!(resolved, expected, "{name}");
        }
    }

    #[test]
    fn vertex_config_blank_values_fall_through_and_padded_values_are_trimmed() {
        use VertexSettingSource::{ConfigFile, Default, Env};
        let config = tempfile::TempDir::new().unwrap();
        let path = config.path().join("config.json");
        for blank in ["", "   ", "\t\n"] {
            // Blank CCP values fall through to the Google variables.
            let env = vertex_env(&[
                ("CCP_VERTEX_PROJECT", blank),
                ("GOOGLE_CLOUD_PROJECT", " google-project "),
                ("CCP_VERTEX_LOCATION", blank),
                ("GOOGLE_CLOUD_LOCATION", "\tus-east5\n"),
                ("CCP_VERTEX_BASE_URL", blank),
            ]);
            let resolved = resolve_vertex_config(&env, config.path());
            assert_eq!(
                resolved.project_setting,
                vertex_setting("google-project", Env("GOOGLE_CLOUD_PROJECT")),
                "{blank:?}"
            );
            assert_eq!(
                Some(resolved.location_setting),
                vertex_setting("us-east5", Env("GOOGLE_CLOUD_LOCATION")),
                "{blank:?}"
            );
            assert_eq!(resolved.base_url_setting, None, "{blank:?}");

            // Blank environment values fall through to config.json, and its
            // padded values come back trimmed.
            std::fs::write(
                &path,
                serde_json::json!({"vertex": {
                    "project": " file-project\t",
                    "location": " eu ",
                    "baseUrl": "\thttp://localhost.:9 ",
                }})
                .to_string(),
            )
            .unwrap();
            let env = vertex_env(&[
                ("CCP_VERTEX_PROJECT", blank),
                ("GOOGLE_CLOUD_PROJECT", blank),
                ("CCP_VERTEX_LOCATION", blank),
                ("GOOGLE_CLOUD_LOCATION", blank),
                ("CCP_VERTEX_BASE_URL", blank),
            ]);
            let resolved = resolve_vertex_config(&env, config.path());
            assert_eq!(resolved.project().unwrap(), "file-project", "{blank:?}");
            assert_eq!(
                resolved
                    .project_setting
                    .as_ref()
                    .map(|setting| setting.source),
                Some(ConfigFile("vertex.project")),
                "{blank:?}"
            );
            assert_eq!(resolved.location().unwrap(), "eu", "{blank:?}");
            assert_eq!(
                resolved.base_url().unwrap(),
                Some("http://localhost.:9"),
                "{blank:?}"
            );

            // Blank everywhere: no project, the default location, no base URL.
            std::fs::write(
                &path,
                serde_json::json!({"vertex": {
                    "project": blank,
                    "location": blank,
                    "baseUrl": blank,
                }})
                .to_string(),
            )
            .unwrap();
            let resolved = resolve_vertex_config(&env, config.path());
            assert_eq!(resolved.project_setting, None, "{blank:?}");
            assert_eq!(
                Some(resolved.location_setting.clone()),
                vertex_setting("global", Default),
                "{blank:?}"
            );
            assert_eq!(resolved.base_url().unwrap(), None, "{blank:?}");
            std::fs::remove_file(&path).unwrap();
        }
    }

    fn override_entries(pairs: &[(&str, Option<&str>)]) -> Vec<VertexLocationOverrideEntry> {
        pairs
            .iter()
            .map(|(key, location)| VertexLocationOverrideEntry {
                key: key.to_string(),
                location: location.map(str::to_string),
            })
            .collect()
    }

    #[test]
    fn vertex_location_overrides_synthetic_come_from_env_over_config_file_with_sources() {
        use VertexSettingSource::{ConfigFile, Env};
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"vertex":{"project":"file-project","locationOverrides":{"anthropic/claude-opus-4-6":" us-east5 ","mistralai":"europe-west4"}}}"#,
        )
        .unwrap();
        let file_setting = VertexLocationOverridesSetting {
            entries: override_entries(&[
                ("anthropic/claude-opus-4-6", Some("us-east5")),
                ("mistralai", Some("europe-west4")),
            ]),
            readable: true,
            source: ConfigFile("vertex.locationOverrides"),
        };
        // Blank or entry-less variables fall through to the file.
        for raw in [None, Some(""), Some("  "), Some(" , ,")] {
            let env = match raw {
                Some(raw) => vertex_env(&[("CCP_VERTEX_LOCATION_OVERRIDES", raw)]),
                None => HashMap::new(),
            };
            let resolved = resolve_vertex_config(&env, config.path());
            assert_eq!(
                resolved.location_overrides_setting.as_ref(),
                Some(&file_setting),
                "{raw:?}"
            );
            assert_eq!(resolved.project().unwrap(), "file-project");
            let overrides = resolved.location_overrides().unwrap();
            assert_eq!(overrides.get("anthropic/claude-opus-4-6"), Some("us-east5"));
            assert_eq!(overrides.get("mistralai"), Some("europe-west4"));
            assert_eq!(overrides.get("anthropic"), None);
        }

        // The variable replaces the file's overrides as a whole.
        let env = vertex_env(&[(
            "CCP_VERTEX_LOCATION_OVERRIDES",
            " xai = us ,anthropic/claude-opus-4-5@20251101=us-east5,, zai-org=global,",
        )]);
        let resolved = resolve_vertex_config(&env, config.path());
        assert_eq!(
            resolved.location_overrides_setting,
            Some(VertexLocationOverridesSetting {
                entries: override_entries(&[
                    ("xai", Some("us")),
                    ("anthropic/claude-opus-4-5@20251101", Some("us-east5")),
                    ("zai-org", Some("global")),
                ]),
                readable: true,
                source: Env("CCP_VERTEX_LOCATION_OVERRIDES"),
            })
        );
        let overrides = resolved.location_overrides().unwrap();
        assert_eq!(overrides.get("xai"), Some("us"));
        assert_eq!(
            overrides.get("anthropic/claude-opus-4-5@20251101"),
            Some("us-east5")
        );
        assert_eq!(overrides.get("zai-org"), Some("global"));
        assert_eq!(overrides.get("mistralai"), None);

        // Nothing set anywhere, and an empty or null object in the file.
        let empty = tempfile::TempDir::new().unwrap();
        for file in [
            None,
            Some(r#"{"vertex":{"locationOverrides":{}}}"#),
            Some(r#"{"vertex":{"locationOverrides":null}}"#),
        ] {
            if let Some(file) = file {
                std::fs::write(empty.path().join("config.json"), file).unwrap();
            }
            let resolved = resolve_vertex_config(&HashMap::new(), empty.path());
            assert_eq!(resolved.location_overrides_setting, None, "{file:?}");
            assert!(
                resolved.location_overrides().unwrap().is_empty(),
                "{file:?}"
            );
        }
    }

    #[test]
    fn vertex_location_overrides_synthetic_invalid_entries_name_the_setting() {
        const KEY_REASON: &str = "the key must be <publisher> or <publisher>/<model-id>: a publisher is lowercase letters, digits and '-', a model ID is letters, digits, '.', '_', '-' and '@', and each starts with a letter or digit";
        const LOCATION_REASON: &str = "must be a Vertex AI location such as global, us-central1 or europe-west4: lowercase letters, digits and '-'";
        const NO_LOCATION: &str = "has no location; an entry is <key>=<location> in the variable and \"<key>\": \"<location>\" in config.json";
        let config = tempfile::TempDir::new().unwrap();
        let env_prefix = "Invalid Vertex setting CCP_VERTEX_LOCATION_OVERRIDES: ";
        let env_cases = [
            ("anthropic", format!("entry \"anthropic\" {NO_LOCATION}")),
            (
                "anthropic=",
                format!("the location of entry \"anthropic\" {LOCATION_REASON}"),
            ),
            (
                "anthropic=US",
                format!("the location of entry \"anthropic\" {LOCATION_REASON}"),
            ),
            (
                "anthropic=us/../x",
                format!("the location of entry \"anthropic\" {LOCATION_REASON}"),
            ),
            ("Anthropic=us", format!("entry 1: {KEY_REASON}")),
            ("anthropic=us,=eu", format!("entry 2: {KEY_REASON}")),
            ("anthropic/claude:x=us", format!("entry 1: {KEY_REASON}")),
            ("a/b/c=us", format!("entry 1: {KEY_REASON}")),
            ("anthropic/=us", format!("entry 1: {KEY_REASON}")),
            ("-meta=us", format!("entry 1: {KEY_REASON}")),
            (
                "endpoint=us-central1",
                "entry 1: a deployed endpoint (endpoint/<name>) takes no location override"
                    .to_string(),
            ),
            (
                "xai=us,endpoint/synthetic=us-central1",
                "entry 2: a deployed endpoint (endpoint/<name>) takes no location override"
                    .to_string(),
            ),
            (
                "anthropic=us,anthropic=eu",
                "entry \"anthropic\" is listed more than once".to_string(),
            ),
        ];
        for (raw, reason) in env_cases {
            let env = vertex_env(&[("CCP_VERTEX_LOCATION_OVERRIDES", raw)]);
            let error = resolve_vertex_config(&env, config.path())
                .location_overrides()
                .unwrap_err()
                .to_string();
            assert_eq!(error, format!("{env_prefix}{reason}"), "{raw:?}");
        }
        // A key that fails the check is not quoted back.
        let env = vertex_env(&[("CCP_VERTEX_LOCATION_OVERRIDES", "Synthetic-Secret-Key=us")]);
        let error = resolve_vertex_config(&env, config.path())
            .location_overrides()
            .unwrap_err()
            .to_string();
        assert!(!error.contains("Synthetic-Secret-Key"), "{error}");

        let file_prefix = "Invalid Vertex setting vertex.locationOverrides in config.json: ";
        let path = config.path().join("config.json");
        let file_cases = [
            (
                r#"{"vertex":{"project":"file-project","locationOverrides":{"mistralai":5}}}"#,
                format!("entry \"mistralai\" {NO_LOCATION}"),
            ),
            (
                r#"{"vertex":{"project":"file-project","locationOverrides":{"mistralai":"EUROPE"}}}"#,
                format!("the location of entry \"mistralai\" {LOCATION_REASON}"),
            ),
            (
                r#"{"vertex":{"project":"file-project","locationOverrides":"mistralai=europe-west4"}}"#,
                "must be an object of \"<key>\": \"<location>\" members, for example {\"mistralai\": \"europe-west4\"}".to_string(),
            ),
            (
                r#"{"vertex":{"project":"file-project","locationOverrides":["mistralai"]}}"#,
                "must be an object of \"<key>\": \"<location>\" members, for example {\"mistralai\": \"europe-west4\"}".to_string(),
            ),
        ];
        for (file, reason) in file_cases {
            std::fs::write(&path, file).unwrap();
            let resolved = resolve_vertex_config(&HashMap::new(), config.path());
            // A wrong override does not cost the rest of the file.
            assert_eq!(resolved.project().unwrap(), "file-project", "{file}");
            let error = resolved.location_overrides().unwrap_err().to_string();
            assert_eq!(error, format!("{file_prefix}{reason}"), "{file}");
        }
    }

    #[test]
    fn vertex_project_missing_is_an_error_naming_every_setting() {
        let config = tempfile::TempDir::new().unwrap();
        let resolved = resolve_vertex_config(&HashMap::new(), config.path());
        let error = resolved.project().unwrap_err().to_string();
        assert_eq!(
            error,
            "Vertex project is not set: set CCP_VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT or vertex.project in config.json"
        );
    }

    #[test]
    fn vertex_project_synthetic_values_are_checked_before_they_reach_the_url() {
        const REASON: &str = "must be a Google Cloud project ID: letters, digits, '.', '_', ':' and '-', starting with a letter or digit";
        let config = tempfile::TempDir::new().unwrap();
        let path = config.path().join("config.json");
        let rejected = [
            "my-project/locations/us",
            "my-project?alt=json",
            "my-project#fragment",
            "my project",
            "-leading-dash",
            ".leading-dot",
            "my-project%2F",
            "プロジェクト",
        ];
        for value in rejected {
            for env_name in ["CCP_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT"] {
                let resolved =
                    resolve_vertex_config(&vertex_env(&[(env_name, value)]), config.path());
                assert_eq!(
                    resolved.project().unwrap_err().to_string(),
                    format!("Invalid Vertex setting {env_name}: {REASON}"),
                    "{env_name}={value:?}"
                );
            }
            std::fs::write(
                &path,
                serde_json::json!({"vertex": {"project": value}}).to_string(),
            )
            .unwrap();
            let resolved = resolve_vertex_config(&HashMap::new(), config.path());
            assert_eq!(
                resolved.project().unwrap_err().to_string(),
                format!("Invalid Vertex setting vertex.project in config.json: {REASON}"),
                "{value:?}"
            );
            std::fs::remove_file(&path).unwrap();
        }
        for accepted in [
            "my-project",
            "my-project-123456",
            "0starts-with-a-digit",
            "example.com:my-project",
            "Mixed_Case.project",
        ] {
            let resolved = resolve_vertex_config(
                &vertex_env(&[("CCP_VERTEX_PROJECT", accepted)]),
                config.path(),
            );
            assert_eq!(resolved.project().unwrap(), accepted);
        }
    }

    #[test]
    fn vertex_location_synthetic_values_must_be_lowercase_letters_digits_and_dashes() {
        const REASON: &str = "must be a Vertex AI location such as global, us-central1 or europe-west4: lowercase letters, digits and '-'";
        let config = tempfile::TempDir::new().unwrap();
        let path = config.path().join("config.json");
        let rejected = [
            "US-CENTRAL1",
            "Global",
            "us-central1/publishers",
            "us_central1",
            "us-central1?x=1",
            "us central1",
            "evil.example.com#",
        ];
        for value in rejected {
            for env_name in ["CCP_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION"] {
                let resolved =
                    resolve_vertex_config(&vertex_env(&[(env_name, value)]), config.path());
                assert_eq!(
                    resolved.location().unwrap_err().to_string(),
                    format!("Invalid Vertex setting {env_name}: {REASON}"),
                    "{env_name}={value:?}"
                );
            }
            std::fs::write(
                &path,
                serde_json::json!({"vertex": {"location": value}}).to_string(),
            )
            .unwrap();
            let resolved = resolve_vertex_config(&HashMap::new(), config.path());
            assert_eq!(
                resolved.location().unwrap_err().to_string(),
                format!("Invalid Vertex setting vertex.location in config.json: {REASON}"),
                "{value:?}"
            );
            std::fs::remove_file(&path).unwrap();
        }
        for accepted in ["global", "us", "eu", "us-central1", "asia-northeast1"] {
            let resolved = resolve_vertex_config(
                &vertex_env(&[("CCP_VERTEX_LOCATION", accepted)]),
                config.path(),
            );
            assert_eq!(resolved.location().unwrap(), accepted);
        }
    }

    #[test]
    fn vertex_base_url_synthetic_unsafe_values_are_rejected_without_repeating_them() {
        const NOT_HTTPS: &str = "must use https (plain http only for a loopback host)";
        const USER_INFO: &str = "must not contain a user name or password";
        let config = tempfile::TempDir::new().unwrap();
        let path = config.path().join("config.json");
        let rejected = [
            ("http://relay.example/v1", NOT_HTTPS),
            ("http://10.0.0.1", NOT_HTTPS),
            ("ftp://127.0.0.1", NOT_HTTPS),
            (
                "https://user:synthetic-pass@aiplatform.googleapis.com",
                USER_INFO,
            ),
            ("https://user@aiplatform.googleapis.com", USER_INFO),
            ("http://user@127.0.0.1:9", USER_INFO),
            ("aiplatform.googleapis.com", "not a valid URL"),
        ];
        for (value, reason) in rejected {
            let resolved = resolve_vertex_config(
                &vertex_env(&[("CCP_VERTEX_BASE_URL", value)]),
                config.path(),
            );
            let error = resolved.base_url().unwrap_err().to_string();
            assert_eq!(
                error,
                format!("Invalid Vertex setting CCP_VERTEX_BASE_URL: {reason}"),
                "{value}"
            );
            assert!(!error.contains("synthetic-pass"), "{error}");

            std::fs::write(
                &path,
                serde_json::json!({"vertex": {"baseUrl": value}}).to_string(),
            )
            .unwrap();
            let resolved = resolve_vertex_config(&HashMap::new(), config.path());
            assert_eq!(
                resolved.base_url().unwrap_err().to_string(),
                format!("Invalid Vertex setting vertex.baseUrl in config.json: {reason}"),
                "{value}"
            );
            std::fs::remove_file(&path).unwrap();
        }
        for accepted in [
            "http://127.0.0.1:9",
            "http://[::1]:9/v1",
            "http://localhost:9",
            "https://relay.example:8443",
        ] {
            let resolved = resolve_vertex_config(
                &vertex_env(&[("CCP_VERTEX_BASE_URL", accepted)]),
                config.path(),
            );
            assert_eq!(resolved.base_url().unwrap(), Some(accepted));
        }
    }

    #[test]
    fn vertex_host_follows_the_location_kind() {
        let cases = [
            ("global", "aiplatform.googleapis.com"),
            ("us", "aiplatform.us.rep.googleapis.com"),
            ("eu", "aiplatform.eu.rep.googleapis.com"),
            (
                "asia-northeast1",
                "asia-northeast1-aiplatform.googleapis.com",
            ),
            ("us-central1", "us-central1-aiplatform.googleapis.com"),
            ("us-east5", "us-east5-aiplatform.googleapis.com"),
        ];
        for (location, host) in cases {
            assert_eq!(vertex_host(location), host, "{location}");
        }
    }

    #[test]
    fn vertex_setting_source_names_the_setting_to_edit() {
        let cases = [
            (
                VertexSettingSource::Env("CCP_VERTEX_PROJECT"),
                "CCP_VERTEX_PROJECT",
            ),
            (
                VertexSettingSource::Env("GOOGLE_CLOUD_LOCATION"),
                "GOOGLE_CLOUD_LOCATION",
            ),
            (
                VertexSettingSource::ConfigFile("vertex.baseUrl"),
                "vertex.baseUrl in config.json",
            ),
            (VertexSettingSource::Default, "default"),
        ];
        for (source, expected) in cases {
            assert_eq!(source.to_string(), expected);
        }
    }

    #[test]
    fn vertex_overrides_are_listed_with_their_source() {
        let _lock = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _guards = isolated_env(&config);
        let _vertex_guards = [
            EnvGuard::unset("CCP_VERTEX_PROJECT"),
            EnvGuard::unset("CCP_VERTEX_LOCATION"),
            EnvGuard::unset("CCP_VERTEX_BASE_URL"),
            EnvGuard::unset("GOOGLE_CLOUD_PROJECT"),
            EnvGuard::unset("GOOGLE_CLOUD_LOCATION"),
        ];
        let loaded = load_config_for_env(&config_env(&config));
        let vertex_lines = || -> Vec<String> {
            config_override_summary_lines(&loaded)
                .into_iter()
                .filter(|line| line.starts_with("vertex."))
                .collect()
        };
        assert_eq!(vertex_lines(), Vec::<String>::new());

        std::fs::write(
            config.path().join("config.json"),
            r#"{"vertex":{"project":"file-project","location":"europe-west4","baseUrl":"https://user:synthetic-pass@file.example"}}"#,
        )
        .unwrap();
        assert_eq!(
            vertex_lines(),
            [
                "vertex.project: file-project",
                "vertex.location: europe-west4",
                "vertex.baseUrl (config)",
            ]
        );

        let _project = EnvGuard::set("GOOGLE_CLOUD_PROJECT", "google-project");
        let _location = EnvGuard::set("GOOGLE_CLOUD_LOCATION", "us-central1");
        assert_eq!(
            vertex_lines(),
            [
                "vertex.project (Google env)",
                "vertex.location (Google env)",
                "vertex.baseUrl (config)",
            ]
        );

        // A blank CCP value is unset and leaves the Google variable in effect.
        let _blank_project = EnvGuard::set("CCP_VERTEX_PROJECT", " ");
        let _ccp_location = EnvGuard::set("CCP_VERTEX_LOCATION", "us-east5");
        let _ccp_base_url = EnvGuard::set("CCP_VERTEX_BASE_URL", "http://127.0.0.1:9");
        let lines = vertex_lines();
        assert_eq!(
            lines,
            [
                "vertex.project (Google env)",
                "vertex.location (env)",
                "vertex.baseUrl (env)",
            ]
        );
        assert!(
            !config_override_summary_lines(&loaded)
                .iter()
                .any(|line| line.contains("synthetic-pass")),
            "{lines:?}"
        );
    }

    #[test]
    fn bind_address_defaults_to_loopback() {
        let config = tempfile::TempDir::new().unwrap();
        let env = config_env(&config);

        assert_eq!(load_config_for_env(&env).bind_address, "127.0.0.1");
    }

    #[test]
    fn bind_address_reads_config_and_env_takes_precedence() {
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"bindAddress":"192.0.2.10"}"#,
        )
        .unwrap();
        let mut env = config_env(&config);

        assert_eq!(load_config_for_env(&env).bind_address, "192.0.2.10");
        env.insert("CCP_BIND_ADDRESS".to_string(), "0.0.0.0".to_string());
        assert_eq!(load_config_for_env(&env).bind_address, "0.0.0.0");
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let previous = std::env::var_os(key);
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.previous.take() {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn isolated_env_restores_an_exported_config_dir() {
        let _guard = ENV_LOCK.lock().unwrap();
        let exported = tempfile::TempDir::new().unwrap();
        let _exported_env = EnvGuard::set("CCP_CONFIG_DIR", exported.path());
        {
            let config = tempfile::TempDir::new().unwrap();
            let _env = isolated_env(&config);
            assert_eq!(paths::config_dir(), config.path());
        }
        assert_eq!(paths::config_dir(), exported.path());
    }

    #[test]
    fn codex_transport_defaults_to_auto() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        assert_eq!(codex_transport(), CodexTransport::Auto);
    }

    #[test]
    fn codex_transport_reads_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        let _transport = EnvGuard::set("CCP_CODEX_TRANSPORT", "auto");
        assert_eq!(codex_transport(), CodexTransport::Auto);
    }

    #[test]
    fn codex_transport_env_websocket() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        let _transport = EnvGuard::set("CCP_CODEX_TRANSPORT", "websocket");
        assert_eq!(codex_transport(), CodexTransport::WebSocket);
    }

    #[test]
    fn codex_transport_invalid_env_falls_back_to_auto() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        let _transport = EnvGuard::set("CCP_CODEX_TRANSPORT", "invalid");
        assert_eq!(codex_transport(), CodexTransport::Auto);
    }

    #[test]
    fn codex_transport_empty_env_falls_back_to_auto() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        let _transport = EnvGuard::set("CCP_CODEX_TRANSPORT", "");
        assert_eq!(codex_transport(), CodexTransport::Auto);
    }

    #[test]
    fn codex_transport_file_and_env_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        for (file_value, expected) in [
            ("http", CodexTransport::Http),
            ("websocket", CodexTransport::WebSocket),
            ("auto", CodexTransport::Auto),
        ] {
            std::fs::write(
                config.path().join("config.json"),
                serde_json::json!({"codex": {"transport": file_value}}).to_string(),
            )
            .unwrap();
            assert_eq!(codex_transport(), expected);
            for env_value in ["", "invalid"] {
                let _transport = EnvGuard::set("CCP_CODEX_TRANSPORT", env_value);
                assert_eq!(codex_transport(), expected);
            }
            for (env_value, expected) in [
                ("http", CodexTransport::Http),
                ("websocket", CodexTransport::WebSocket),
                ("auto", CodexTransport::Auto),
            ] {
                let _transport = EnvGuard::set("CCP_CODEX_TRANSPORT", env_value);
                assert_eq!(codex_transport(), expected);
            }
        }
    }

    #[test]
    fn codex_transport_invalid_or_empty_file_falls_back_to_auto() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        for value in ["", "invalid"] {
            std::fs::write(
                config.path().join("config.json"),
                serde_json::json!({"codex": {"transport": value}}).to_string(),
            )
            .unwrap();
            assert_eq!(codex_transport(), CodexTransport::Auto);
        }
    }

    #[test]
    fn codex_header_timeout_defaults_overrides_and_floors() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);

        assert_eq!(codex_header_timeout_ms(), CODEX_DEFAULT_HEADER_TIMEOUT_MS);
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"headerTimeoutMs":600000}}"#,
        )
        .unwrap();
        assert_eq!(codex_header_timeout_ms(), 600_000);
        {
            let _timeout_env = EnvGuard::set("CCP_CODEX_HEADER_TIMEOUT_MS", "120000");
            assert_eq!(codex_header_timeout_ms(), 120_000);
        }
        // Below the floor is ignored, not clamped: a value that small is a typo,
        // and clamping it would hide the typo behind a working default.
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"headerTimeoutMs":5}}"#,
        )
        .unwrap();
        assert_eq!(codex_header_timeout_ms(), CODEX_DEFAULT_HEADER_TIMEOUT_MS);
        let _timeout_env = EnvGuard::set("CCP_CODEX_HEADER_TIMEOUT_MS", "0");
        assert_eq!(codex_header_timeout_ms(), CODEX_DEFAULT_HEADER_TIMEOUT_MS);
    }

    #[test]
    fn parse_codex_transport_variants() {
        assert_eq!(parse_codex_transport("http"), Some(CodexTransport::Http));
        assert_eq!(
            parse_codex_transport("websocket"),
            Some(CodexTransport::WebSocket)
        );
        assert_eq!(parse_codex_transport("auto"), Some(CodexTransport::Auto));
        assert_eq!(parse_codex_transport(""), None);
        assert_eq!(parse_codex_transport("HTTP"), None);
        assert_eq!(parse_codex_transport("ws"), None);
    }

    #[test]
    fn codex_transport_as_str() {
        assert_eq!(CodexTransport::Http.as_str(), "http");
        assert_eq!(CodexTransport::WebSocket.as_str(), "websocket");
        assert_eq!(CodexTransport::Auto.as_str(), "auto");
    }

    #[test]
    fn log_env_presence_enables_legacy_verbose_and_stderr() {
        let config = tempfile::TempDir::new().unwrap();
        let mut env = config_env(&config);
        env.insert("CCP_LOG_VERBOSE".to_string(), "0".to_string());
        env.insert("CCP_LOG_STDERR".to_string(), String::new());

        let loaded = load_config_for_env(&env);
        assert!(loaded.log_verbose);
        assert!(loaded.log_stderr);
    }

    #[test]
    fn log_config_values_apply_without_env() {
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"log":{"verbose":true,"stderr":true}}"#,
        )
        .unwrap();
        let env = config_env(&config);

        let loaded = load_config_for_env(&env);
        assert!(loaded.log_verbose);
        assert!(loaded.log_stderr);
    }

    #[test]
    fn codex_responses_api_defaults_to_disabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);

        assert!(!codex_responses_api());
    }

    #[test]
    fn codex_responses_api_reads_config_and_env_takes_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"responsesApi":true}}"#,
        )
        .unwrap();
        let _env = isolated_env(&config);

        assert!(codex_responses_api());
        let _responses_env = EnvGuard::set("CCP_CODEX_RESPONSES_API", "false");
        assert!(!codex_responses_api());
    }

    #[test]
    fn codex_full_lane_defaults_off_reads_config_and_env_takes_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        assert!(!codex_full_lane());

        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"fullLane":true}}"#,
        )
        .unwrap();
        assert!(codex_full_lane());
        let _full_lane_env = EnvGuard::set("CCP_CODEX_FULL_LANE", "false");
        assert!(!codex_full_lane());
    }

    #[test]
    fn codex_full_lane_accepts_boolean_env_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);
        for value in ["1", "true", "TRUE", "yes"] {
            let _flag = EnvGuard::set("CCP_CODEX_FULL_LANE", value);
            assert!(codex_full_lane(), "{value}");
        }
        for value in ["0", "false", "no", "invalid", ""] {
            let _flag = EnvGuard::set("CCP_CODEX_FULL_LANE", value);
            assert!(!codex_full_lane(), "{value}");
        }
    }

    #[test]
    fn codex_responses_api_accepts_enabled_env_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);

        for value in ["1", "true", "TRUE", "yes"] {
            let _responses_env = EnvGuard::set("CCP_CODEX_RESPONSES_API", value);
            assert!(codex_responses_api(), "{value}");
        }
    }

    #[test]
    fn codex_images_api_defaults_to_disabled_and_env_overrides_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"imagesApi":true,"imagesBaseUrl":"https://chatgpt.com/backend-api/codex-custom"}}"#,
        )
        .unwrap();
        let _env = isolated_env(&config);

        assert!(codex_images_api());
        assert_eq!(
            codex_images_base_url(),
            "https://chatgpt.com/backend-api/codex-custom"
        );
        let _enabled_env = EnvGuard::set("CCP_CODEX_IMAGES_API", "false");
        let _base_env = EnvGuard::set(
            "CCP_CODEX_IMAGES_BASE_URL",
            "https://chatgpt.com/backend-api/codex",
        );
        assert!(!codex_images_api());
        assert_eq!(
            codex_images_base_url(),
            "https://chatgpt.com/backend-api/codex"
        );
    }

    #[test]
    fn codex_transcriptions_api_defaults_to_disabled_and_env_overrides_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"transcriptionsApi":true}}"#,
        )
        .unwrap();
        let _env = isolated_env(&config);

        assert!(codex_transcriptions_api());
        let _enabled_env = EnvGuard::set("CCP_CODEX_TRANSCRIPTIONS_API", "false");
        assert!(!codex_transcriptions_api());
    }

    #[test]
    fn codex_reasoning_summary_reads_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"reasoningSummary":"off"}}"#,
        )
        .unwrap();
        let _env = isolated_env(&config);

        assert_eq!(codex_reasoning_summary().as_deref(), Some("off"));
    }

    #[test]
    fn codex_reasoning_summary_env_overrides_config_and_empty_falls_through() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"reasoningSummary":"off"}}"#,
        )
        .unwrap();
        let _env = isolated_env(&config);
        {
            let _summary_env = EnvGuard::set("CCP_CODEX_REASONING_SUMMARY", "auto");
            assert_eq!(codex_reasoning_summary().as_deref(), Some("auto"));
        }
        {
            let _summary_env = EnvGuard::set("CCP_CODEX_REASONING_SUMMARY", "");
            assert_eq!(codex_reasoning_summary().as_deref(), Some("off"));
        }
    }

    #[test]
    fn auto_review_model_reads_top_level_config_and_env_takes_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"autoReviewModel":"grok-4.5"}"#,
        )
        .unwrap();
        let _env = isolated_env(&config);

        assert_eq!(auto_review_model().as_deref(), Some("grok-4.5"));
        {
            let _model_env = EnvGuard::set("CCP_AUTO_REVIEW_MODEL", "gpt-5.6-terra");
            assert_eq!(auto_review_model().as_deref(), Some("gpt-5.6-terra"));
        }
        {
            let _model_env = EnvGuard::set("CCP_AUTO_REVIEW_MODEL", "");
            assert_eq!(auto_review_model().as_deref(), Some("grok-4.5"));
        }
    }

    #[test]
    fn codex_server_compaction_defaults_and_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);

        assert!(!codex_server_compaction());
        {
            let _enabled_env = EnvGuard::set("CCP_CODEX_SERVER_COMPACTION", "on");
            assert!(codex_server_compaction());
        }
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"serverCompaction":true}}"#,
        )
        .unwrap();
        assert!(codex_server_compaction());
        let _disabled_env = EnvGuard::set("CCP_CODEX_SERVER_COMPACTION", "false");
        assert!(!codex_server_compaction());
    }

    #[test]
    fn codex_request_compression_defaults_and_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let config = tempfile::TempDir::new().unwrap();
        let _env = isolated_env(&config);

        assert!(!codex_request_compression());
        {
            let _enabled_env = EnvGuard::set("CCP_CODEX_REQUEST_COMPRESSION", "on");
            assert!(codex_request_compression());
        }
        std::fs::write(
            config.path().join("config.json"),
            r#"{"codex":{"requestCompression":true}}"#,
        )
        .unwrap();
        assert!(codex_request_compression());
        assert_eq!(codex_transport(), CodexTransport::Http);
        {
            let _transport_env = EnvGuard::set("CCP_CODEX_TRANSPORT", "auto");
            assert_eq!(codex_transport(), CodexTransport::Auto);
        }
        let _disabled_env = EnvGuard::set("CCP_CODEX_REQUEST_COMPRESSION", "false");
        assert!(!codex_request_compression());
        assert_eq!(codex_transport(), CodexTransport::Auto);
    }

    fn synthetic_endpoints(file: &str) -> VertexEndpoints {
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(config.path().join("config.json"), file).unwrap();
        resolve_vertex_endpoints(config.path())
    }

    #[test]
    fn vertex_endpoints_synthetic_valid_entries_are_read_with_their_defaults() {
        let endpoints = synthetic_endpoints(
            r#"{"vertex":{"endpoints":{
                "my-llm":{"endpoint":"1234567890123456789","location":"us-central1","model":"gemma-2-9b-it"},
                "hf_tgi":{"endpoint":"9876543210987654321","location":"europe-west4","model":"tgi",
                          "project":"other-project","host":"9876543210987654321.europe-west4-123456789.prediction.vertexai.goog",
                          "tools":false},
                "local":{"endpoint":"synthetic-endpoint","location":"us-east5","model":"synthetic/model",
                         "host":"http://127.0.0.1:8080/","tools":null,"project":null}
            }}}"#,
        );
        assert!(!endpoints.unreadable);
        assert!(endpoints.first_error().is_none(), "{endpoints:?}");
        let tests = [
            (
                "my-llm",
                VertexEndpointConfig {
                    endpoint: "1234567890123456789".to_string(),
                    location: "us-central1".to_string(),
                    model: "gemma-2-9b-it".to_string(),
                    project: None,
                    host: None,
                    tools: true,
                },
            ),
            (
                "hf_tgi",
                VertexEndpointConfig {
                    endpoint: "9876543210987654321".to_string(),
                    location: "europe-west4".to_string(),
                    model: "tgi".to_string(),
                    project: Some("other-project".to_string()),
                    // A DNS name alone gets https in front.
                    host: Some(
                        "https://9876543210987654321.europe-west4-123456789.prediction.vertexai.goog"
                            .to_string(),
                    ),
                    tools: false,
                },
            ),
            (
                "local",
                VertexEndpointConfig {
                    endpoint: "synthetic-endpoint".to_string(),
                    location: "us-east5".to_string(),
                    model: "synthetic/model".to_string(),
                    project: None,
                    host: Some("http://127.0.0.1:8080".to_string()),
                    tools: true,
                },
            ),
        ];
        for (name, expected) in tests {
            let entry = endpoints
                .get(name)
                .unwrap_or_else(|| panic!("{name} is missing"));
            assert_eq!(entry.config.as_ref(), Ok(&expected), "{name}");
        }
        let mut names: Vec<&str> = endpoints.usable_names().collect();
        names.sort_unstable();
        assert_eq!(names, ["hf_tgi", "local", "my-llm"]);
    }

    #[test]
    fn vertex_endpoints_synthetic_invalid_entries_name_the_setting_and_the_endpoint() {
        let prefix = "Invalid Vertex setting vertex.endpoints in config.json: ";
        let tests = [
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"http://synthetic.example"}"#,
                "endpoint \"e\": host must use https (plain http only for a loopback host)",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"http://123.us-central1-9.prediction.vertexai.goog"}"#,
                "endpoint \"e\": host must use https (plain http only for a loopback host)",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"https://synthetic.example"}"#,
                "endpoint \"e\": host must be a dedicated endpoint host, https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a loopback host",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"https://prediction.vertexai.goog"}"#,
                "endpoint \"e\": host must be a dedicated endpoint host, https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a loopback host",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"https://123.us-central1-9.prediction.vertexai.goog.synthetic.example"}"#,
                "endpoint \"e\": host must be a dedicated endpoint host, https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a loopback host",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"https://123.us-central1-9.prediction.vertexai.goog/v1/projects/p"}"#,
                "endpoint \"e\": host must be a scheme and a host only, without a path, query or fragment",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","host":"https://user:synthetic-pass@123.us-central1-9.prediction.vertexai.goog"}"#,
                "endpoint \"e\": host must not contain a user name or password",
            ),
            (
                r#"{"endpoint":"123","location":"global","model":"m"}"#,
                "endpoint \"e\": location must be the endpoint's region, such as us-central1; an endpoint is never in global",
            ),
            (
                r#"{"endpoint":"123","location":"US-CENTRAL1","model":"m"}"#,
                "endpoint \"e\": location must be a Vertex AI location such as global, us-central1 or europe-west4: lowercase letters, digits and '-'",
            ),
            (
                r#"{"location":"us-central1","model":"m"}"#,
                "endpoint \"e\": endpoint is required",
            ),
            (
                r#"{"endpoint":"123","model":"m"}"#,
                "endpoint \"e\": location is required",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"  "}"#,
                "endpoint \"e\": model must be a non-empty string",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1"}"#,
                "endpoint \"e\": model is required",
            ),
            (
                r#"{"endpoint":"projects/p/locations/l/endpoints/123","location":"us-central1","model":"m"}"#,
                "endpoint \"e\": endpoint must be the endpoint ID: lowercase letters, digits and '-', starting with a letter or digit",
            ),
            (
                r#"{"endpoint":123,"location":"us-central1","model":"m"}"#,
                "endpoint \"e\": endpoint must be a non-empty string",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","project":"bad/project"}"#,
                "endpoint \"e\": project must be a Google Cloud project ID: letters, digits, '.', '_', ':' and '-', starting with a letter or digit",
            ),
            (
                r#"{"endpoint":"123","location":"us-central1","model":"m","tools":"no"}"#,
                "endpoint \"e\": tools must be true or false",
            ),
            (
                r#"{"endpointId":"123","location":"us-central1","model":"m"}"#,
                "endpoint \"e\": unknown key \"endpointId\"; the keys are endpoint, location, model, project, host and tools",
            ),
            (
                r#""123""#,
                "endpoint \"e\": must be an object with the keys endpoint, location and model, and optionally project, host and tools",
            ),
        ];
        for (entry, reason) in tests {
            let endpoints = synthetic_endpoints(&format!(
                r#"{{"vertex":{{"endpoints":{{"e":{entry},"ok":{{"endpoint":"1","location":"us-central1","model":"m"}}}}}}}}"#
            ));
            let expected = format!("{prefix}{reason}");
            assert_eq!(
                endpoints.get("e").map(|entry| entry.config.clone()),
                Some(Err(expected.clone())),
                "{entry}"
            );
            assert!(!expected.contains("synthetic-pass"), "{entry}");
            // The broken entry leaves the other one usable.
            assert_eq!(
                endpoints.usable_names().collect::<Vec<_>>(),
                ["ok"],
                "{entry}"
            );
            assert_eq!(
                endpoints.first_error().map(|error| error.to_string()),
                Some(expected),
                "{entry}"
            );
        }

        let endpoints = synthetic_endpoints(
            r#"{"vertex":{"endpoints":{"my llm":{"endpoint":"1","location":"us-central1","model":"m"}}}}"#,
        );
        assert_eq!(
            endpoints.get("my llm").map(|entry| entry.config.clone()),
            Some(Err(format!(
                "{prefix}endpoint name \"my llm\" must be letters, digits, '_' and '-', at most 64 of them"
            )))
        );
        assert_eq!(endpoints.usable_names().count(), 0);
    }

    #[test]
    fn vertex_endpoints_synthetic_absent_null_and_non_objects() {
        for file in [
            r#"{}"#,
            r#"{"vertex":{"project":"p"}}"#,
            r#"{"vertex":{"endpoints":null}}"#,
            r#"{"vertex":{"endpoints":{}}}"#,
        ] {
            let endpoints = synthetic_endpoints(file);
            assert_eq!(endpoints, VertexEndpoints::default(), "{file}");
            assert!(endpoints.first_error().is_none(), "{file}");
            assert_eq!(endpoints.summary_line(), None, "{file}");
        }
        for file in [
            r#"{"vertex":{"endpoints":["my-llm"]}}"#,
            r#"{"vertex":{"endpoints":"my-llm"}}"#,
        ] {
            let endpoints = synthetic_endpoints(file);
            assert!(endpoints.unreadable, "{file}");
            assert!(endpoints.entries.is_empty(), "{file}");
            assert_eq!(
                endpoints.first_error().map(|error| error.to_string()),
                Some(
                    "Invalid Vertex setting vertex.endpoints in config.json: must be an object of \"<name>\": {\"endpoint\", \"location\", \"model\"} members"
                        .to_string()
                ),
                "{file}"
            );
            assert_eq!(
                endpoints.summary_line().as_deref(),
                Some("vertex.endpoints (config, not an object)"),
                "{file}"
            );
        }
        // The rest of the file is still read when an endpoint is broken.
        let config = tempfile::TempDir::new().unwrap();
        std::fs::write(
            config.path().join("config.json"),
            r#"{"vertex":{"project":"file-project","endpoints":{"e":42}}}"#,
        )
        .unwrap();
        let resolved = resolve_vertex_config(&HashMap::new(), config.path());
        assert_eq!(resolved.project().unwrap(), "file-project");
    }

    #[test]
    fn vertex_endpoints_synthetic_summary_lists_names_and_never_a_host() {
        let endpoints = synthetic_endpoints(
            r#"{"vertex":{"endpoints":{
                "my-llm":{"endpoint":"1","location":"us-central1","model":"m",
                          "host":"https://user:synthetic-pass@1.us-central1-9.prediction.vertexai.goog"},
                "bad name":{"endpoint":"2","location":"us-central1","model":"m"}
            }}}"#,
        );
        let line = endpoints.summary_line().expect("a summary line");
        assert!(line.starts_with("vertex.endpoints: "), "{line}");
        assert!(line.contains("my-llm"), "{line}");
        assert!(line.contains("[invalid name]"), "{line}");
        assert!(!line.contains("bad name"), "{line}");
        assert!(!line.contains("synthetic-pass"), "{line}");
        assert!(!line.contains("prediction.vertexai.goog"), "{line}");
    }

    #[test]
    fn vertex_dedicated_url_synthetic_takes_the_documented_form_and_refuses_others() {
        let accepted = [
            (
                "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog",
                "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog",
            ),
            (
                "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog/",
                "https://1234567890123456789.us-central1-987654321.prediction.vertexai.goog",
            ),
            (
                "https://1234567890123456789.us-central1-abc.PREDICTION.vertexai.goog",
                "https://1234567890123456789.us-central1-abc.prediction.vertexai.goog",
            ),
            ("http://127.0.0.1:18080", "http://127.0.0.1:18080"),
            ("http://[::1]:18080/", "http://[::1]:18080"),
            ("https://localhost:18443", "https://localhost:18443"),
        ];
        for (value, expected) in accepted {
            assert_eq!(
                validate_vertex_dedicated_url(value).as_deref(),
                Ok(expected),
                "{value}"
            );
        }
        let refused = [
            // A DNS name without a scheme is not a URL: nothing is put in
            // front of what the endpoint reports.
            (
                "1234567890123456789.us-central1-987654321.prediction.vertexai.goog",
                "is not a valid URL",
            ),
            (
                "http://1234567890123456789.us-central1-987654321.prediction.vertexai.goog",
                "must use https (plain http only for a loopback host)",
            ),
            (
                "https://us-central1-aiplatform.googleapis.com",
                "must be a dedicated endpoint host, https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a loopback host",
            ),
            (
                "https://1.us-central1-9.prediction.vertexai.goog?x=1",
                "must be a scheme and a host only, without a path, query or fragment",
            ),
            (
                "ftp://1.us-central1-9.prediction.vertexai.goog",
                "must use https (plain http only for a loopback host)",
            ),
        ];
        for (value, reason) in refused {
            assert_eq!(validate_vertex_dedicated_url(value), Err(reason), "{value}");
        }
    }
}
