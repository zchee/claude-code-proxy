//! LM Studio's local server. It speaks the Anthropic Messages API itself
//! (LM Studio 0.4.1 and later), so requests are forwarded untranslated; the
//! model list is read whenever a listing is asked for, because models are
//! downloaded and removed while the proxy runs. The list comes from the
//! server's REST API, or, while a server on this machine is stopped, from the
//! models on this machine's disk as `lms ls` reports them.

use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::Duration;

use async_trait::async_trait;
use axum::{Json, http::StatusCode, response::IntoResponse, response::Response};
use reqwest::Client;
use serde::Deserialize;

use crate::{
    anthropic::{
        json_error,
        schema::{CountTokensResponse, MessagesRequest},
    },
    config::LmStudioSettings,
    provider::{CliHandlers, Provider, RequestContext},
    providers::{anthropic_passthrough, kimi::count_tokens},
};

/// Every LM Studio model is addressed as `lmstudio/<model key>`, so its
/// names never collide with another provider's.
pub const MODEL_PREFIX: &str = "lmstudio/";

/// A listing must not hang on a server that accepts the connection but does
/// not answer; a refused connection fails at once.
const CATALOG_TIMEOUT: Duration = Duration::from_secs(3);

/// `lms ls` asks the LM Studio daemon, which may be starting up.
const LMS_TIMEOUT: Duration = Duration::from_secs(10);

pub struct LmStudioProvider {
    upstream: Result<Upstream, String>,
    /// The `lms` CLI that lists this machine's models while its server is
    /// stopped. Only set when the server is on this machine.
    lms: Option<PathBuf>,
    catalog: RwLock<Vec<String>>,
}

/// Why the server's catalog could not be read.
enum CatalogError {
    /// Nothing answered: the server is not running.
    Unreachable(String),
    /// The server answered, but not with a usable list.
    Failed(String),
}

struct Upstream {
    client: Client,
    settings: LmStudioSettings,
}

/// The part of `GET /api/v1/models` the proxy reads.
#[derive(Deserialize)]
struct Catalog {
    models: Vec<CatalogEntry>,
}

#[derive(Deserialize)]
struct CatalogEntry {
    #[serde(rename = "type")]
    kind: String,
    key: String,
}

/// The part of one `lms ls --json` entry the proxy reads.
#[derive(Deserialize)]
struct DiskEntry {
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "modelKey")]
    model_key: String,
    /// Set for a model on another device reached through LM Link; `null`
    /// for a model on this machine.
    #[serde(rename = "deviceIdentifier", default)]
    device_identifier: Option<String>,
}

impl LmStudioProvider {
    /// The provider for this process's settings, or `None` when LM Studio
    /// is turned off. Invalid settings still give a provider, so listings
    /// and requests report the setting instead of the provider vanishing.
    pub fn configured() -> Option<Self> {
        let settings = crate::config::lmstudio_settings()?;
        let lms = settings
            .as_ref()
            .is_ok_and(|settings| is_loopback_origin(&settings.base_url))
            .then(find_lms)
            .flatten();
        Some(Self::new(settings.map_err(|error| error.to_string()), lms))
    }

    pub(crate) fn new(settings: Result<LmStudioSettings, String>, lms: Option<PathBuf>) -> Self {
        let upstream = settings.and_then(|settings| {
            Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(5))
                .build()
                .map(|client| Upstream { client, settings })
                .map_err(|error| format!("Could not build the LM Studio HTTP client: {error}"))
        });
        Self {
            upstream,
            lms,
            catalog: RwLock::new(Vec::new()),
        }
    }

    fn set_catalog(&self, models: Vec<String>) {
        // A poisoned lock only means another thread panicked mid-write; the
        // list it guards is replaced whole, so it is safe to keep using.
        *self
            .catalog
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = models;
    }

    async fn fetch_catalog(upstream: &Upstream) -> Result<Vec<String>, CatalogError> {
        let base = &upstream.settings.base_url;
        let response = upstream
            .authorized(upstream.client.get(format!("{base}/api/v1/models")))
            .timeout(CATALOG_TIMEOUT)
            .send()
            .await
            .map_err(|error| {
                if error.is_timeout() {
                    CatalogError::Failed(format!(
                        "LM Studio at {base} did not answer within {}s",
                        CATALOG_TIMEOUT.as_secs()
                    ))
                } else {
                    CatalogError::Unreachable(format!(
                        "LM Studio server is not reachable at {base}; start it with `lms server start`"
                    ))
                }
            })?;
        Self::read_catalog(response, upstream)
            .await
            .map_err(CatalogError::Failed)
    }

    async fn read_catalog(
        response: reqwest::Response,
        upstream: &Upstream,
    ) -> Result<Vec<String>, String> {
        let base = &upstream.settings.base_url;
        let status = response.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            let hint = if upstream.settings.api_key.is_some() {
                "check the API token in CCP_LMSTUDIO_API_KEY"
            } else {
                "set CCP_LMSTUDIO_API_KEY to an API token from LM Studio's server settings"
            };
            return Err(format!(
                "LM Studio at {base} requires authentication (HTTP {}): {hint}",
                status.as_u16()
            ));
        }
        if status == StatusCode::NOT_FOUND {
            return Err(format!(
                "LM Studio at {base} has no /api/v1/models; LM Studio 0.4.1 or later is required"
            ));
        }
        if !status.is_success() {
            return Err(format!(
                "LM Studio at {base} answered HTTP {}",
                status.as_u16()
            ));
        }
        let catalog: Catalog = response.json().await.map_err(|error| {
            format!("LM Studio at {base} returned a model list the proxy cannot read: {error}")
        })?;
        // Embedding models cannot answer a Messages request.
        let mut models: Vec<String> = catalog
            .models
            .into_iter()
            .filter(|entry| entry.kind == "llm" && !entry.key.is_empty())
            .map(|entry| format!("{MODEL_PREFIX}{}", entry.key))
            .collect();
        models.sort_unstable();
        models.dedup();
        Ok(models)
    }

    /// The LLMs on this machine's disk, as `lms ls` reports them. Models on
    /// other devices reached through LM Link are left out: a stopped server
    /// on this machine cannot reach them either.
    async fn read_disk_catalog(lms: &Path) -> Result<Vec<String>, String> {
        let run = tokio::process::Command::new(lms)
            .args(["ls", "--json", "--llm"])
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output();
        let output = tokio::time::timeout(LMS_TIMEOUT, run)
            .await
            .map_err(|_| format!("`lms ls` did not finish within {}s", LMS_TIMEOUT.as_secs()))?
            .map_err(|error| format!("could not run {}: {error}", lms.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let reason = stderr
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("");
            return Err(format!(
                "`lms ls` failed ({}): {}",
                output.status,
                reason.trim()
            ));
        }
        let entries: Vec<DiskEntry> = serde_json::from_slice(&output.stdout).map_err(|error| {
            format!("`lms ls --json` printed a list the proxy cannot read: {error}")
        })?;
        let mut models: Vec<String> = entries
            .into_iter()
            .filter(|entry| {
                entry.kind == "llm"
                    && entry.device_identifier.is_none()
                    && !entry.model_key.is_empty()
            })
            .map(|entry| format!("{MODEL_PREFIX}{}", entry.model_key))
            .collect();
        models.sort_unstable();
        models.dedup();
        Ok(models)
    }

    /// The upstream, and the LM Studio model key the request names, or the
    /// error response to send instead.
    fn resolve<'a>(
        &'a self,
        body: &MessagesRequest,
    ) -> Result<(&'a Upstream, String), Box<Response>> {
        let upstream = self.upstream.as_ref().map_err(|error| {
            Box::new(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                error.clone(),
            ))
        })?;
        let requested = body.model.as_deref().unwrap_or_default();
        let normalized = crate::registry::normalize_incoming_model(requested);
        match normalized.strip_prefix(MODEL_PREFIX) {
            Some(key) if !key.is_empty() => Ok((upstream, key.to_string())),
            _ => Err(Box::new(json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("LM Studio models are named {MODEL_PREFIX}<model key>, not '{requested}'"),
            ))),
        }
    }
}

impl Upstream {
    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        // LM Studio accepts the bearer form on every endpoint.
        match &self.settings.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }
}

#[async_trait]
impl Provider for LmStudioProvider {
    fn name(&self) -> &'static str {
        "lmstudio"
    }

    fn supported_models(&self) -> Vec<String> {
        self.catalog
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn has_live_catalog(&self) -> bool {
        true
    }

    async fn refresh_models(&self) -> Result<Option<String>, String> {
        let upstream = self.upstream.as_ref().map_err(Clone::clone)?;
        let listed = match Self::fetch_catalog(upstream).await {
            Ok(models) => Ok((models, None)),
            Err(CatalogError::Failed(error)) => Err(error),
            Err(CatalogError::Unreachable(error)) => match &self.lms {
                None => Err(error),
                Some(lms) => match Self::read_disk_catalog(lms).await {
                    Ok(models) => Ok((
                        models,
                        Some(format!(
                            "listed from this machine's disk: {error} before using them"
                        )),
                    )),
                    Err(disk_error) => Err(format!("{error} ({disk_error})")),
                },
            },
        };
        match listed {
            Ok((models, caveat)) => {
                self.set_catalog(models);
                Ok(caveat)
            }
            Err(error) => {
                // Nothing could be read; do not keep listing an old answer.
                self.set_catalog(Vec::new());
                Err(error)
            }
        }
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &LmStudioAuth
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let (upstream, key) = match self.resolve(&body) {
            Ok(resolved) => resolved,
            Err(response) => return *response,
        };
        let base = &upstream.settings.base_url;
        let request = upstream.authorized(upstream.client.post(format!("{base}/v1/messages")));
        anthropic_passthrough::forward(request, body, &key, &ctx, &format!("LM Studio at {base}"))
            .await
    }

    /// LM Studio has no count_tokens endpoint, so the count is estimated here
    /// as for the other providers without one.
    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let key = match self.resolve(&body) {
            Ok((_, key)) => key,
            Err(response) => return *response,
        };
        let input_tokens = count_tokens::count_tokens(&body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &key);
            monitor.usage_updated(&ctx.req_id, Some(input_tokens), None);
        }
        (StatusCode::OK, Json(CountTokensResponse { input_tokens })).into_response()
    }
}

/// Whether `origin` names this machine, whose disk `lms ls` describes.
fn is_loopback_origin(origin: &str) -> bool {
    match url::Url::parse(origin)
        .ok()
        .as_ref()
        .and_then(url::Url::host)
    {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(domain)) => domain
            .strip_suffix('.')
            .unwrap_or(domain)
            .eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

/// The `lms` CLI on `PATH`, else where LM Studio installs it.
fn find_lms() -> Option<PathBuf> {
    let on_path = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join("lms"))
            .find(|candidate| candidate.is_file())
    });
    on_path.or_else(|| {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        let bin = PathBuf::from(home).join(".lmstudio").join("bin");
        ["lms", "lms.exe"]
            .into_iter()
            .map(|name| bin.join(name))
            .find(|candidate| candidate.is_file())
    })
}

struct LmStudioAuth;

impl CliHandlers for LmStudioAuth {
    fn login(&self) -> anyhow::Result<()> {
        anyhow::bail!(
            "LM Studio does not need a login; set CCP_LMSTUDIO_API_KEY when its server requires authentication"
        )
    }
    fn device(&self) -> anyhow::Result<()> {
        self.login()
    }
    fn status(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn logout(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::{
        Router,
        body::Body,
        extract::Request,
        http::{HeaderMap, header::CONTENT_TYPE},
        routing::{get, post},
    };
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    const CATALOG: &str = r#"{"models":[
        {"type":"llm","publisher":"OBLITERATUS","key":"gemma-4-12b-obliterated","display_name":"Gemma 4 12B"},
        {"type":"embedding","publisher":"nomic-ai","key":"text-embedding-nomic-embed-text-v1.5"},
        {"type":"llm","publisher":"ibm","key":"ibm/granite-4-micro","loaded_instances":[]}
    ]}"#;

    /// An LM Studio stand-in on a free loopback port. Every request it sees
    /// is counted, and `routes` decides what it answers.
    async fn fake_lmstudio(
        routes: Router,
    ) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = routes.layer(axum::middleware::from_fn(
            move |request: Request, next: axum::middleware::Next| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    next.run(request).await
                }
            },
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (base, hits, task)
    }

    fn provider(base_url: &str, api_key: Option<&str>) -> LmStudioProvider {
        provider_with_lms(base_url, api_key, None)
    }

    fn provider_with_lms(
        base_url: &str,
        api_key: Option<&str>,
        lms: Option<PathBuf>,
    ) -> LmStudioProvider {
        LmStudioProvider::new(
            Ok(LmStudioSettings {
                base_url: base_url.to_string(),
                api_key: api_key.map(str::to_string),
            }),
            lms,
        )
    }

    /// A stand-in `lms` CLI: a shell script that prints `stdout` and exits
    /// with `code`, and records each run in `runs` next to it.
    fn fake_lms(dir: &tempfile::TempDir, stdout: &str, code: i32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let json = dir.path().join("ls.json");
        std::fs::write(&json, stdout).unwrap();
        let runs = dir.path().join("runs");
        let script = dir.path().join("lms");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\necho \"$*\" >> '{}'\ncat '{}'\nexit {code}\n",
                runs.display(),
                json.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    fn fake_lms_runs(dir: &tempfile::TempDir) -> Vec<String> {
        std::fs::read_to_string(dir.path().join("runs"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// A loopback base URL where nothing listens.
    async fn stopped_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        base
    }

    const LMS_LS: &str = r#"[
        {"type":"llm","modelKey":"qwen3.8-27b-obliterated","deviceIdentifier":null,"path":"OBLITERATUS/Qwen3.8-27B"},
        {"type":"llm","modelKey":"remote-only-70b","deviceIdentifier":"9f1c0de","path":"x/remote"},
        {"type":"embedding","modelKey":"text-embedding-nomic-embed-text-v1.5","deviceIdentifier":null},
        {"type":"llm","modelKey":"clef-omni","path":"mlx-community/clef-omni-4bit"}
    ]"#;

    fn proxy(provider: LmStudioProvider) -> (Arc<crate::Registry>, Router) {
        let providers: Vec<Arc<dyn Provider>> = vec![Arc::new(provider)];
        let registry = Arc::new(crate::Registry::from_providers(
            crate::config::AliasProvider::Codex,
            providers,
        ));
        (registry.clone(), crate::server::app(registry))
    }

    async fn post_json(app: &Router, path: &str, body: Value) -> (StatusCode, HeaderMap, Vec<u8>) {
        let request = axum::http::Request::post(path)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, headers, bytes.to_vec())
    }

    #[tokio::test]
    async fn listing_shows_the_downloaded_llms_and_not_the_embedding_models() {
        let routes = Router::new().route(
            "/api/v1/models",
            get(|headers: HeaderMap| async move {
                assert_eq!(
                    headers.get("authorization").and_then(|v| v.to_str().ok()),
                    Some("Bearer lm-secret"),
                    "the catalog request must carry the configured token"
                );
                ([(CONTENT_TYPE, "application/json")], CATALOG)
            }),
        );
        let (base, _hits, task) = fake_lmstudio(routes).await;
        let (registry, _) = proxy(provider(&base, Some("lm-secret")));

        assert_eq!(
            registry.grouped_models().get("lmstudio"),
            Some(&Vec::new()),
            "nothing is listed before the catalog is read"
        );
        let notes = registry.refresh_live_catalogs().await;
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(
            registry.grouped_models().get("lmstudio").cloned(),
            Some(vec![
                "lmstudio/gemma-4-12b-obliterated".to_string(),
                "lmstudio/ibm/granite-4-micro".to_string(),
            ])
        );
        task.abort();
    }

    #[tokio::test]
    async fn the_proxy_models_endpoint_reads_the_catalog_on_each_call() {
        let routes = Router::new().route(
            "/api/v1/models",
            get(|| async { ([(CONTENT_TYPE, "application/json")], CATALOG) }),
        );
        let (base, hits, task) = fake_lmstudio(routes).await;
        let (_, app) = proxy(provider(&base, None));
        for expected_hits in 1..=2 {
            let request = axum::http::Request::get("/v1/models")
                .body(Body::empty())
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let listing: Value = serde_json::from_slice(&bytes).unwrap();
            let ids: Vec<&str> = listing["data"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|entry| entry["id"].as_str())
                .collect();
            assert_eq!(
                ids,
                [
                    "lmstudio/gemma-4-12b-obliterated",
                    "lmstudio/ibm/granite-4-micro"
                ]
            );
            assert_eq!(hits.load(Ordering::SeqCst), expected_hits);
        }
        task.abort();
    }

    #[tokio::test]
    async fn messages_reach_lmstudio_under_the_bare_model_key_and_stream_back_unchanged() {
        const SSE: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\n";
        let routes = Router::new().route(
            "/v1/messages",
            post(
                |headers: HeaderMap, Json(payload): Json<Value>| async move {
                    assert_eq!(payload["model"], "ibm/granite-4-micro");
                    assert_eq!(payload["stream"], true);
                    assert_eq!(payload["max_tokens"], 64);
                    assert_eq!(payload["tools"][0]["name"], "get_weather");
                    assert_eq!(
                        headers.get("authorization").and_then(|v| v.to_str().ok()),
                        Some("Bearer lm-secret")
                    );
                    assert_eq!(
                        headers
                            .get("anthropic-version")
                            .and_then(|v| v.to_str().ok()),
                        Some("2023-06-01")
                    );
                    ([(CONTENT_TYPE, "text/event-stream")], SSE)
                },
            ),
        );
        let (base, hits, task) = fake_lmstudio(routes).await;
        let (_, app) = proxy(provider(&base, Some("lm-secret")));
        let (status, headers, body) = post_json(
            &app,
            "/v1/messages",
            json!({
                "model": "lmstudio/ibm/granite-4-micro[1m]",
                "max_tokens": 64,
                "stream": true,
                "tools": [{"name":"get_weather","input_schema":{"type":"object"}}],
                "messages": [{"role":"user","content":"Weather?"}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        assert_eq!(
            headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(String::from_utf8(body).unwrap(), SSE);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        task.abort();
    }

    #[tokio::test]
    async fn an_lmstudio_error_keeps_its_status_and_body() {
        const NOT_LOADED: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"No models loaded."}}"#;
        let routes = Router::new().route(
            "/v1/messages",
            post(|| async {
                (
                    StatusCode::BAD_REQUEST,
                    [(CONTENT_TYPE, "application/json")],
                    NOT_LOADED,
                )
            }),
        );
        let (base, _hits, task) = fake_lmstudio(routes).await;
        let (_, app) = proxy(provider(&base, None));
        let (status, _, body) = post_json(
            &app,
            "/v1/messages",
            json!({"model":"lmstudio/gemma-4-12b-obliterated","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(String::from_utf8(body).unwrap(), NOT_LOADED);
        task.abort();
    }

    #[tokio::test]
    async fn token_counts_are_estimated_without_calling_lmstudio() {
        let (base, hits, task) = fake_lmstudio(Router::new()).await;
        let (_, app) = proxy(provider(&base, None));
        let (status, _, body) = post_json(
            &app,
            "/v1/messages/count_tokens",
            json!({
                "model": "lmstudio/gemma-4-12b-obliterated",
                "system": "You are terse.",
                "messages": [{"role":"user","content":"Count the tokens in this sentence, please."}]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let counted: Value = serde_json::from_slice(&body).unwrap();
        assert!(counted["input_tokens"].as_u64().unwrap() > 0, "{counted}");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        task.abort();
    }

    #[tokio::test]
    async fn a_stopped_server_is_reported_and_lists_nothing() {
        let routes = Router::new().route(
            "/api/v1/models",
            get(|| async { ([(CONTENT_TYPE, "application/json")], CATALOG) }),
        );
        let (base, _hits, task) = fake_lmstudio(routes).await;
        let (registry, app) = proxy(provider(&base, None));
        assert!(registry.refresh_live_catalogs().await.is_empty());
        assert_eq!(registry.supported_models_for("lmstudio").len(), 2);

        task.abort();
        let _ = task.await;
        let notes = registry.refresh_live_catalogs().await;
        let Some(crate::provider::CatalogNote::Unavailable(message)) = notes.get("lmstudio") else {
            panic!("expected lmstudio to be unavailable: {notes:?}");
        };
        assert!(message.contains(&base), "{message}");
        assert!(message.contains("lms server start"), "{message}");
        assert!(registry.supported_models_for("lmstudio").is_empty());

        let (status, _, body) = post_json(
            &app,
            "/v1/messages",
            json!({"model":"lmstudio/gemma-4-12b-obliterated","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let error: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with(&format!("Could not reach LM Studio at {base}")),
            "{error}"
        );
    }

    #[tokio::test]
    async fn catalog_failures_name_the_cause() {
        let tests: [(&str, StatusCode, Option<&str>, &str); 4] = [
            (
                "no token",
                StatusCode::UNAUTHORIZED,
                None,
                "set CCP_LMSTUDIO_API_KEY",
            ),
            (
                "wrong token",
                StatusCode::FORBIDDEN,
                Some("bad"),
                "check the API token",
            ),
            (
                "old LM Studio",
                StatusCode::NOT_FOUND,
                None,
                "0.4.1 or later",
            ),
            (
                "server error",
                StatusCode::INTERNAL_SERVER_ERROR,
                None,
                "answered HTTP 500",
            ),
        ];
        for (name, status, api_key, expected) in tests {
            let routes = Router::new().route("/api/v1/models", get(move || async move { status }));
            let (base, _hits, task) = fake_lmstudio(routes).await;
            let provider = provider(&base, api_key);
            let error = provider.refresh_models().await.expect_err(name);
            assert!(error.contains(expected), "{name}: {error}");
            task.abort();
        }
    }

    #[tokio::test]
    async fn invalid_settings_are_reported_instead_of_hiding_the_provider() {
        let provider = LmStudioProvider::new(
            Err("Invalid LM Studio setting CCP_LMSTUDIO_BASE_URL".into()),
            None,
        );
        let error = provider.refresh_models().await.unwrap_err();
        assert!(error.contains("CCP_LMSTUDIO_BASE_URL"), "{error}");
        let (_, app) = proxy(provider);
        let (status, _, body) = post_json(
            &app,
            "/v1/messages",
            json!({"model":"lmstudio/x","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(String::from_utf8_lossy(&body).contains("CCP_LMSTUDIO_BASE_URL"));
    }

    #[tokio::test]
    async fn a_stopped_local_server_still_lists_the_llms_on_this_machines_disk() {
        let dir = tempfile::TempDir::new().unwrap();
        let lms = fake_lms(&dir, LMS_LS, 0);
        let base = stopped_server().await;
        let (registry, _) = proxy(provider_with_lms(&base, None, Some(lms)));

        let notes = registry.refresh_live_catalogs().await;
        assert_eq!(
            registry.supported_models_for("lmstudio"),
            ["lmstudio/clef-omni", "lmstudio/qwen3.8-27b-obliterated"],
            "only LLMs on this machine, not embeddings or LM Link devices"
        );
        let Some(crate::provider::CatalogNote::Caveat(caveat)) = notes.get("lmstudio") else {
            panic!("expected a caveat: {notes:?}");
        };
        assert!(
            caveat.contains("listed from this machine's disk"),
            "{caveat}"
        );
        assert!(caveat.contains(&base), "{caveat}");
        assert!(caveat.contains("lms server start"), "{caveat}");
        assert_eq!(fake_lms_runs(&dir), ["ls --json --llm"]);
    }

    #[tokio::test]
    async fn a_running_server_is_listed_without_running_lms() {
        let dir = tempfile::TempDir::new().unwrap();
        let lms = fake_lms(&dir, LMS_LS, 0);
        let routes = Router::new().route(
            "/api/v1/models",
            get(|| async { ([(CONTENT_TYPE, "application/json")], CATALOG) }),
        );
        let (base, _hits, task) = fake_lmstudio(routes).await;
        let provider = provider_with_lms(&base, None, Some(lms));
        assert_eq!(provider.refresh_models().await, Ok(None));
        assert_eq!(provider.supported_models().len(), 2);
        assert!(fake_lms_runs(&dir).is_empty());
        task.abort();
    }

    #[tokio::test]
    async fn a_server_that_answers_with_an_error_is_not_hidden_behind_the_disk_list() {
        let dir = tempfile::TempDir::new().unwrap();
        let lms = fake_lms(&dir, LMS_LS, 0);
        let routes =
            Router::new().route("/api/v1/models", get(|| async { StatusCode::UNAUTHORIZED }));
        let (base, _hits, task) = fake_lmstudio(routes).await;
        let provider = provider_with_lms(&base, None, Some(lms));
        let error = provider.refresh_models().await.unwrap_err();
        assert!(error.contains("requires authentication"), "{error}");
        assert!(provider.supported_models().is_empty());
        assert!(fake_lms_runs(&dir).is_empty());
        task.abort();
    }

    #[tokio::test]
    async fn a_failing_lms_is_reported_with_the_stopped_server() {
        let dir = tempfile::TempDir::new().unwrap();
        let lms = fake_lms(&dir, "", 1);
        let base = stopped_server().await;
        let provider = provider_with_lms(&base, None, Some(lms));
        let error = provider.refresh_models().await.unwrap_err();
        assert!(error.contains("not reachable"), "{error}");
        assert!(error.contains("`lms ls` failed"), "{error}");
        assert!(provider.supported_models().is_empty());
    }

    #[test]
    fn only_a_server_on_this_machine_uses_the_disk_list() {
        for (origin, loopback) in [
            ("http://127.0.0.1:1234", true),
            ("http://[::1]:1234", true),
            ("http://localhost:1234", true),
            ("https://lmstudio.example:8443", false),
            ("http://192.168.1.20:1234", false),
        ] {
            assert_eq!(is_loopback_origin(origin), loopback, "{origin}");
        }
    }

    #[test]
    fn only_the_prefix_routes_to_lmstudio() {
        let (registry, _) = proxy(provider("http://127.0.0.1:9", None));
        for model in [
            "lmstudio/gemma-4-12b-obliterated",
            "lmstudio/ibm/granite-4-micro",
            "lmstudio/not-downloaded-yet",
            "lmstudio/gemma-4-12b-obliterated[1m]",
        ] {
            assert_eq!(
                registry.provider_for_model(model, None).map(|p| p.name()),
                Some("lmstudio"),
                "{model}"
            );
        }
        for model in ["gemma-4-12b-obliterated", "lmstudio", "local/gguf"] {
            assert!(
                registry.provider_for_model(model, None).is_none(),
                "{model}"
            );
        }
    }
}
