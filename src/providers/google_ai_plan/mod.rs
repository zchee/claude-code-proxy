//! Google AI Ultra plan: Gemini models on the quota of the user's own
//! Google AI Ultra subscription.
//!
//! The user signs in with Google through a `Desktop app` OAuth client that
//! Google approved for the Ultra plan ([`auth`]). The proxy ships one, the
//! client Antigravity (`agy`) uses against Cloud Code PA ([`auth::client`]);
//! configuration can replace it with the user's own. Model calls go to Cloud Code PA's
//! `v1internal:streamGenerateContent` with the access token as a Bearer
//! token, for the Cloud AI Companion project that `v1internal:loadCodeAssist`
//! (and `v1internal:onboardUser` when needed) finds for the sign-in
//! ([`client`], [`translate`]).
//!
//! The upstream is always asked for a stream. A non-streaming client gets a
//! message built by the same translator, so both modes decide content, stop
//! reason and errors the same way and share its frame and content limits.
//! The buffered reply also fails when the whole upstream body exceeds 16 MiB,
//! which a live stream of many small frames does not. A stream that closes
//! before a `finishReason` is an error and is not sent again; the client
//! retries by the error it gets.

pub mod auth;
pub mod client;
pub mod errors;
pub mod models;
pub mod translate;

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use axum::{
    Json,
    body::Body,
    http::{HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value, json};

use crate::anthropic::{
    error::json_error,
    schema::{CountTokensResponse, MessagesRequest},
    sse::parse_sse_events,
};
use crate::auth::{AuthStorage, FileAuthStore};
use crate::config::{GoogleAiPlanClient, GoogleAiPlanSettings};
use crate::monitor::usage_from_anthropic_sse;
use crate::provider::{
    CliHandlers, Generation, GenerationBody, Provider, ProviderError, ProviderErrorKind,
    RequestContext,
};
use crate::providers::codex::native::NativeResponseOutcome;
use crate::providers::grok::translate::stream::SseDecoder;
use crate::providers::kimi::count_tokens;
use crate::providers::opencode::chat::redact_every_string;
use crate::providers::vertex::sanitize::{redact_secrets, sanitize_vertex_text};
use crate::providers::vertex::translate::stream::{
    POST_TERMINAL_READ_LIMIT, Sanitizer, StreamSinks,
};
use crate::providers::vertex::{MAX_BUFFERED_RESPONSE_BYTES, status_for_error_type};
use crate::registry::normalize_incoming_model;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use self::auth::manager::{GoogleAiPlanAuthError, GoogleAiPlanAuthManager};
use self::auth::token_store::{GoogleAiPlanTokenStore, StoredAuth, file_store};
use self::client::{InteractionsClient, SendError, byte_stream};
use self::errors::{classify_discovery_error, classify_error};
use self::translate::request::prepare_request;
use self::translate::stream::{
    LiveStreamTranslator, MAX_SSE_FRAME_BYTES, StreamError, TranslatorConfig,
};

/// Traffic-capture summary of a streaming response.
const STREAM_SUMMARY_NAME: &str = "061-google-ai-plan-stream-summary";

/// Traffic-capture summary of a non-streaming response.
const BUFFERED_SUMMARY_NAME: &str = "061-google-ai-plan-buffered-summary";

/// What a model call needs besides the settings.
struct Ready<S: AuthStorage<StoredAuth>> {
    client: InteractionsClient,
    auth: GoogleAiPlanAuthManager<S>,
}

/// Builds the [`Ready`] parts from the settings, on the first request.
type BuildReady<S> = fn(&GoogleAiPlanSettings) -> anyhow::Result<Ready<S>>;

/// The `google-ai-plan` provider.
pub struct GoogleAiPlanProvider<S: AuthStorage<StoredAuth> + 'static = FileAuthStore<StoredAuth>> {
    /// The settings read when the provider was built, or why they are
    /// invalid. An invalid setting is reported by name in every request.
    settings: Result<GoogleAiPlanSettings, String>,
    ready: OnceLock<Arc<Ready<S>>>,
    build: BuildReady<S>,
}

impl GoogleAiPlanProvider {
    /// The server's provider. The settings are read now; the HTTP clients
    /// are built on the first request, because building them loads the
    /// system root certificates and every CLI command builds every
    /// provider.
    pub fn new() -> Self {
        Self::with_settings(
            crate::config::google_ai_plan_settings().map_err(|error| format!("{error:#}")),
        )
    }

    /// A provider for `settings`, signed in through the credential file. An
    /// `Err` is the message every request answers with.
    pub fn with_settings(settings: Result<GoogleAiPlanSettings, String>) -> Self {
        Self {
            settings,
            ready: OnceLock::new(),
            build: |settings| {
                Ok(Ready {
                    client: InteractionsClient::new(settings)?,
                    auth: GoogleAiPlanAuthManager::new(file_store(), settings.client.clone())?,
                })
            },
        }
    }
}

impl Default for GoogleAiPlanProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: AuthStorage<StoredAuth> + 'static> GoogleAiPlanProvider<S> {
    /// A provider from its parts, for tests against local servers: point the
    /// settings' base URL at a fake Cloud Code PA endpoint and build `auth`
    /// with an in-memory store and the URL of a fake token endpoint.
    pub fn with_parts(
        settings: GoogleAiPlanSettings,
        client: InteractionsClient,
        auth: GoogleAiPlanAuthManager<S>,
    ) -> Self {
        Self {
            settings: Ok(settings),
            ready: OnceLock::from(Arc::new(Ready { client, auth })),
            build: |_| anyhow::bail!("the parts were given at construction"),
        }
    }

    fn settings(&self) -> Result<&GoogleAiPlanSettings, ProviderError> {
        self.settings.as_ref().map_err(|error| {
            ProviderError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ProviderErrorKind::Api,
                error.clone(),
            )
        })
    }

    /// The HTTP clients, built on first use. A failure is returned and not
    /// kept, so the next request tries again.
    fn ready(&self) -> Result<&Arc<Ready<S>>, ProviderError> {
        let settings = self.settings()?;
        if let Some(ready) = self.ready.get() {
            return Ok(ready);
        }
        let built = (self.build)(settings).map_err(|error| {
            ProviderError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ProviderErrorKind::Api,
                format!("Google AI Ultra plan transport is unavailable: {error:#}"),
            )
        })?;
        // Concurrent first requests may each build one; all use the first.
        Ok(self.ready.get_or_init(|| Arc::new(built)))
    }

    /// Resolves the model, translates the request, gets the access token and
    /// sends the request. The streaming, non-streaming and OpenAI-compatible
    /// paths share this, so they get the same validation and error mapping.
    async fn start(
        &self,
        body: &MessagesRequest,
        ctx: &RequestContext,
    ) -> Result<Started, Failure> {
        let requested = normalize_incoming_model(body.model.as_deref().unwrap_or_default());
        let model = models::resolve(&requested).map_err(invalid_request)?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model.upstream_model);
        }
        let mut translated = prepare_request(body, &model).map_err(invalid_request)?;
        let ready = self.ready()?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let snapshot = ready.auth.get_auth_snapshot().await.map_err(auth_error)?;
        let token = snapshot.access_token;
        let project_id = ready
            .client
            .ensure_project(
                &token,
                &snapshot.credential_key,
                snapshot.project_id.as_deref(),
            )
            .await
            .map_err(|error| send_failure(error, &model.upstream_model, &token, true))?;
        if snapshot.project_id.as_deref() != Some(project_id.as_str()) {
            ready
                .auth
                .set_project_id(&snapshot.credential_key, project_id.clone())
                .await;
        }
        translated.project = project_id;
        let response = ready
            .client
            .send(&translated, &token, ctx.traffic.as_deref())
            .await
            .map_err(|error| send_failure(error, &model.upstream_model, &token, false))?;
        let config = TranslatorConfig {
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            model: requested,
            signature_model: model.upstream_model.clone(),
            estimated_input_tokens: count_tokens::count_tokens(body),
            emit_thinking: translated.emit_thinking,
        };
        Ok(Started {
            response,
            config,
            sanitize: sanitizer(&token),
            redact: redactor(&token),
            model_id: model.upstream_model,
        })
    }
}

/// One request the upstream accepted: the response body is still unread.
struct Started {
    response: reqwest::Response,
    config: TranslatorConfig,
    /// Applied to upstream prose before it reaches the client, the log or
    /// the outcome.
    sanitize: Sanitizer,
    /// Takes the access token out of every string a capture records.
    redact: Sanitizer,
    model_id: String,
}

/// A request that failed before any response body: the error and whether
/// the client may retry it.
struct Failure {
    error: ProviderError,
    should_retry: bool,
}

impl From<ProviderError> for Failure {
    fn from(error: ProviderError) -> Self {
        Self {
            error,
            should_retry: true,
        }
    }
}

/// The sanitiser for upstream text a client or the log reads: the access
/// token removed, verbatim or percent-encoded once, plus the general
/// clean-up and length bound the Vertex AI path applies to Google prose.
fn sanitizer(token: &str) -> Sanitizer {
    let token = token.to_string();
    Arc::new(move |text: &str| sanitize_vertex_text(text, &[token.as_str()]))
}

/// The capture redactor: the access token becomes `[redacted]` in any
/// string, and nothing else changes.
fn redactor(token: &str) -> Sanitizer {
    let token = token.to_string();
    Arc::new(move |text: &str| redact_secrets(text, &[token.as_str()]))
}

/// The answer for a failed sign-in lookup or refresh.
fn auth_error(error: GoogleAiPlanAuthError) -> Failure {
    let (status, kind) = match &error {
        GoogleAiPlanAuthError::NotSignedIn
        | GoogleAiPlanAuthError::ClientChanged
        | GoogleAiPlanAuthError::SignInAgain => {
            (StatusCode::UNAUTHORIZED, ProviderErrorKind::Authentication)
        }
        GoogleAiPlanAuthError::RefreshFailed { .. } | GoogleAiPlanAuthError::Transport(_) => {
            (StatusCode::BAD_GATEWAY, ProviderErrorKind::Api)
        }
        GoogleAiPlanAuthError::Store(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, ProviderErrorKind::Api)
        }
    };
    ProviderError::new(status, kind, sanitize_vertex_text(&error.to_string(), &[])).into()
}

/// The answer for a model call that produced no stream. A non-2xx status is
/// decided by [`classify_error`], with the access token removed from the
/// message again on top of the client's redaction of the body.
fn send_failure(error: SendError, model_id: &str, token: &str, discovery: bool) -> Failure {
    match error {
        SendError::Status {
            status,
            headers,
            body,
        } => {
            let classified = if discovery {
                classify_discovery_error(status, &headers, &body, model_id)
            } else {
                classify_error(status, &headers, &body, model_id)
            };
            let should_retry = classified.should_retry;
            let mut error = classified.into_provider_error();
            error.message = redact_secrets(&error.message, &[token]);
            Failure {
                error,
                should_retry,
            }
        }
        SendError::Discovery(_) => ProviderError::new(
            StatusCode::BAD_GATEWAY,
            ProviderErrorKind::Api,
            error.to_string(),
        )
        .into(),
        SendError::HeaderTimeout(_) => ProviderError::new(
            StatusCode::GATEWAY_TIMEOUT,
            ProviderErrorKind::Api,
            error.to_string(),
        )
        .into(),
        SendError::Unreachable => ProviderError::new(
            StatusCode::BAD_GATEWAY,
            ProviderErrorKind::Api,
            error.to_string(),
        )
        .into(),
        SendError::InvalidUrl | SendError::Encode => ProviderError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ProviderErrorKind::Api,
            error.to_string(),
        )
        .into(),
    }
}

fn invalid_request(error: impl std::fmt::Display) -> Failure {
    ProviderError::new(
        StatusCode::BAD_REQUEST,
        ProviderErrorKind::InvalidRequest,
        error.to_string(),
    )
    .into()
}

#[async_trait]
impl<S: AuthStorage<StoredAuth> + 'static> Provider for GoogleAiPlanProvider<S> {
    fn name(&self) -> &'static str {
        "google-ai-plan"
    }

    fn supported_models(&self) -> Vec<String> {
        models::advertised_models()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &GOOGLE_AI_PLAN_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let started = match self.start(&body, &ctx).await {
            Ok(started) => started,
            Err(failure) => return failure_response(failure),
        };
        if !body.stream {
            return buffered_response(started, &ctx).await;
        }
        // A stream that ends in an error event is reported through the
        // outcome, which the server reads when the body ends, so the request
        // is recorded as failed with the sanitised message the client saw.
        let outcome = NativeResponseOutcome::default();
        let body = stream_response_body(started, &ctx, Some(outcome.clone()));
        let mut response = sse_response(body);
        response.extensions_mut().insert(outcome);
        response
    }

    /// Answered locally with the estimator the Vertex AI Gemini path uses;
    /// nothing is sent upstream.
    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let requested = normalize_incoming_model(body.model.as_deref().unwrap_or_default());
        let model = match models::resolve(&requested) {
            Ok(model) => model,
            Err(error) => return failure_response(invalid_request(error)),
        };
        if let Err(error) = self.settings() {
            return failure_response(error.into());
        }
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model.upstream_model);
        }
        let tokens = count_tokens::count_tokens(&body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(&ctx.req_id, Some(tokens), None);
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }

    /// The `x-should-retry: false` of a final 429 cannot be carried by a
    /// [`ProviderError`]; the caller gets the status and message only. The
    /// resolved model is the prefixed name: the OpenAI-style routes echo it,
    /// and the bare upstream ID would route a client's next request to
    /// Vertex AI.
    async fn generate_anthropic_stream(
        &self,
        mut body: MessagesRequest,
        ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        body.stream = true;
        let started = self
            .start(&body, &ctx)
            .await
            .map_err(|failure| failure.error)?;
        let resolved_model = format!("{}{}", models::MODEL_PREFIX, started.model_id);
        Ok(Generation {
            body: GenerationBody::LiveSse(stream_response_body(started, &ctx, None)),
            resolved_model,
        })
    }
}

fn failure_response(failure: Failure) -> Response {
    let Failure {
        error,
        should_retry,
    } = failure;
    let mut response = json_error(error.status, error.error_type(), error.message);
    let headers = response.headers_mut();
    if let Some(value) = error
        .retry_after
        .as_deref()
        .and_then(|value| HeaderValue::from_str(value).ok())
    {
        headers.insert(http::header::RETRY_AFTER, value);
    }
    if !should_retry {
        headers.insert(
            HeaderName::from_static("x-should-retry"),
            HeaderValue::from_static("false"),
        );
    }
    response
}

fn sse_response(body: Body) -> Response {
    (
        [
            (http::header::CONTENT_TYPE, "text/event-stream"),
            (http::header::CACHE_CONTROL, "no-cache"),
            (http::header::CONNECTION, "keep-alive"),
        ],
        body,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

type UpstreamBody = Pin<Box<dyn Stream<Item = Result<Bytes, ()>> + Send>>;

/// The Anthropic SSE body of a started request. The upstream is read until
/// the `finishReason` (and the short window after it for a usage-only chunk)
/// or until it closes; a failure ends the body with
/// one `error` event, recorded in `outcome`.
fn stream_response_body(
    started: Started,
    ctx: &RequestContext,
    outcome: Option<NativeResponseOutcome>,
) -> Body {
    let Started {
        response,
        config,
        sanitize,
        redact,
        ..
    } = started;
    let state = PlanStream {
        upstream: Box::pin(byte_stream(response)),
        translator: LiveStreamTranslator::streaming(config),
        capture: ctx.traffic.as_ref().map(|_| UpstreamCapture::new()),
        sinks: StreamSinks {
            req_id: ctx.req_id.clone(),
            monitor: ctx.monitor.clone(),
            traffic: ctx.traffic.clone(),
            outcome,
            sanitize,
        },
        redact,
        received: 0,
        read_deadline: None,
        done: false,
    };
    let stream = futures_util::stream::unfold(state, |mut state| async move {
        state
            .next_output()
            .await
            .map(|bytes| (Ok::<Bytes, Infallible>(Bytes::from(bytes)), state))
    });
    Body::from_stream(stream)
}

struct PlanStream {
    upstream: UpstreamBody,
    translator: LiveStreamTranslator,
    capture: Option<UpstreamCapture>,
    sinks: StreamSinks,
    redact: Sanitizer,
    /// Upstream bytes read so far.
    received: u64,
    /// When the post-terminal read window ends, set once the finish reason
    /// arrives so a held-open upstream still completes promptly.
    read_deadline: Option<tokio::time::Instant>,
    done: bool,
}

impl PlanStream {
    async fn next_output(&mut self) -> Option<Vec<u8>> {
        if self.done {
            return None;
        }
        loop {
            let next = if self.translator.has_finish() {
                let deadline = *self
                    .read_deadline
                    .get_or_insert_with(|| tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT);
                match tokio::time::timeout_at(deadline, self.upstream.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        return match self.translator.finish_after_read_limit() {
                            Ok(output) => self.complete(output),
                            Err(error) => Some(self.fail(&error)),
                        };
                    }
                }
            } else {
                self.upstream.next().await
            };
            let chunk = match next {
                Some(Ok(chunk)) => chunk,
                Some(Err(())) => return Some(self.fail(&StreamError::Transport)),
                None => {
                    return match self.translator.finish() {
                        Ok(output) => self.complete(output),
                        Err(error) => Some(self.fail(&error)),
                    };
                }
            };
            if self.received == 0
                && let Some(monitor) = self.sinks.monitor.as_ref()
            {
                monitor.generation_started(&self.sinks.req_id);
            }
            self.received = self.received.saturating_add(chunk.len() as u64);
            if let Some(capture) = self.capture.as_mut() {
                capture.upstream(&chunk, &self.sinks.sanitize, &self.redact);
            }
            match self.translator.push(&chunk) {
                // The response is complete: the upstream is not read again.
                Ok(output) if self.translator.is_finished() => return self.complete(output),
                Ok(output) if output.is_empty() => {}
                Ok(output) => {
                    self.forward(&output);
                    return Some(output);
                }
                Err(error) => return Some(self.fail(&error)),
            }
        }
    }

    fn complete(&mut self, output: Vec<u8>) -> Option<Vec<u8>> {
        self.done = true;
        if !output.is_empty() {
            self.forward(&output);
        }
        self.finish_capture(None);
        (!output.is_empty()).then_some(output)
    }

    fn fail(&mut self, error: &StreamError) -> Vec<u8> {
        self.done = true;
        let sanitize = self.sinks.sanitize.clone();
        let message = error.message_with(&*sanitize);
        self.log_failure(error);
        let output = error.sse_event(&*sanitize);
        if let Some(outcome) = self.sinks.outcome.as_ref() {
            outcome.fail(message);
        }
        self.capture_downstream(&output);
        self.finish_capture(Some(error.kind()));
        output
    }

    /// Records a failed stream in the proxy log. The server logs a
    /// streaming request as completed once the response headers are sent,
    /// so without this line a failure inside the stream leaves no trace.
    /// The error's `Display` holds no upstream prose.
    fn log_failure(&self, error: &StreamError) {
        let mut fields = Map::new();
        fields.insert("reqId".into(), json!(self.sinks.req_id));
        fields.insert("model".into(), json!(self.translator.model()));
        fields.insert("kind".into(), json!(error.kind()));
        fields.insert("error".into(), json!(error.to_string()));
        fields.insert("bytes".into(), json!(self.received));
        crate::logging::create_logger("google-ai-plan")
            .warn("google_ai_plan_stream_failed", Some(fields));
    }

    fn forward(&mut self, output: &[u8]) {
        if let Some(monitor) = self.sinks.monitor.as_ref() {
            let (input_tokens, output_tokens) = usage_from_anthropic_sse(output);
            monitor.stream_progress(
                &self.sinks.req_id,
                output.len() as u64,
                parse_sse_events(output).len() as u64,
                input_tokens,
                output_tokens,
            );
        }
        self.capture_downstream(output);
    }

    fn capture_downstream(&mut self, bytes: &[u8]) {
        if let Some(capture) = self.capture.as_mut() {
            capture.downstream(bytes, &self.redact);
        }
    }

    fn finish_capture(&mut self, error: Option<&'static str>) {
        if let (Some(capture), Some(traffic)) = (self.capture.take(), self.sinks.traffic.as_ref()) {
            let kind = if error.is_none() {
                "stream_completion"
            } else {
                "stream_error"
            };
            capture.finish(traffic, kind, error, STREAM_SUMMARY_NAME);
        }
    }
}

impl Drop for PlanStream {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        if let (Some(capture), Some(traffic)) = (self.capture.take(), self.sinks.traffic.as_ref()) {
            capture.finish(
                traffic,
                "stream_abandoned",
                Some("downstream_body_dropped"),
                STREAM_SUMMARY_NAME,
            );
        }
    }
}

/// Keys whose string values carry free-form upstream prose: `error.message`,
/// `candidates[].finishMessage` and `promptFeedback.blockReasonMessage`.
const CAPTURE_PROSE_KEYS: [&str; 3] = ["message", "finishMessage", "blockReasonMessage"];

/// A copy of `value` with every string under a [`CAPTURE_PROSE_KEYS`] key, at
/// any depth, passed through `sanitize`. Any depth, because the upstream wraps
/// `candidates` and `promptFeedback` in `response`. A non-string under such a
/// key is searched like any other value; every other string is kept as is.
fn sanitize_capture_prose(value: &Value, sanitize: &dyn Fn(&str) -> String) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, item)| {
                    let item = match item {
                        Value::String(text) if CAPTURE_PROSE_KEYS.contains(&key.as_str()) => {
                            Value::String(sanitize(text))
                        }
                        _ => sanitize_capture_prose(item, sanitize),
                    };
                    (key.clone(), item)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| sanitize_capture_prose(item, sanitize))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Traffic capture of one response. Upstream events are recorded decoded,
/// with every `message`, `finishMessage` and `blockReasonMessage` (where
/// Google puts prose) through the sanitiser and the access token redacted
/// from every string, key and event name; never as raw bytes.
struct UpstreamCapture {
    decoder: SseDecoder,
    events: StreamTrafficCapture,
    bytes: u64,
    chunks: u64,
}

impl UpstreamCapture {
    fn new() -> Self {
        Self {
            decoder: SseDecoder::with_max_frame_bytes(MAX_SSE_FRAME_BYTES),
            events: StreamTrafficCapture::default(),
            bytes: 0,
            chunks: 0,
        }
    }

    fn upstream(&mut self, chunk: &[u8], sanitize: &Sanitizer, redact: &Sanitizer) {
        self.bytes = self.bytes.saturating_add(chunk.len() as u64);
        self.chunks = self.chunks.saturating_add(1);
        let Ok(events) = self.decoder.push(chunk) else {
            self.events.malformed("decoder", "malformed_sse");
            return;
        };
        for event in events {
            let data = event.data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<Value>(data) {
                Ok(value) => {
                    let cleaned = sanitize_capture_prose(&value, &**sanitize);
                    let name = event.event.as_deref().map(|name| redact(name));
                    self.events
                        .upstream_event(name.as_deref(), &redact_every_string(&cleaned, &**redact));
                }
                Err(_) => self.events.malformed("json", "malformed_event"),
            }
        }
    }

    fn downstream(&mut self, bytes: &[u8], redact: &Sanitizer) {
        for event in parse_sse_events(bytes) {
            if let Ok(value) = serde_json::from_str::<Value>(&event.data) {
                self.events.downstream_event(
                    event.event.as_deref().unwrap_or("message"),
                    redact_every_string(&value, &**redact),
                );
            }
        }
    }

    /// Writes the summary. `error` is the kind of the failure, if any.
    fn finish(mut self, traffic: &TrafficCapture, kind: &str, error: Option<&str>, name: &str) {
        if let Some(error) = error {
            self.events.malformed("translation", error);
        }
        self.events.finish_named(
            traffic,
            json!({"kind": kind, "bytes": self.bytes, "chunks": self.chunks}),
            name,
        );
    }
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

/// Reads the upstream response until the `finishReason` and answers
/// with one Anthropic message, or with the error the stream would have
/// ended in.
async fn buffered_response(started: Started, ctx: &RequestContext) -> Response {
    let Started {
        response,
        config,
        sanitize,
        redact,
        ..
    } = started;
    let mut capture = ctx.traffic.as_ref().map(|_| UpstreamCapture::new());
    let result = read_message(
        byte_stream(response),
        config,
        &sanitize,
        &redact,
        capture.as_mut(),
    )
    .await;
    match result {
        Ok(message) => {
            if let (Some(traffic), Some(capture)) = (ctx.traffic.as_ref(), capture) {
                traffic.write_json(
                    "051-downstream-response",
                    &redact_every_string(&message, &*redact),
                );
                capture.finish(traffic, "buffered_completion", None, BUFFERED_SUMMARY_NAME);
            }
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.usage_updated(
                    &ctx.req_id,
                    message
                        .pointer("/usage/input_tokens")
                        .and_then(Value::as_u64),
                    message
                        .pointer("/usage/output_tokens")
                        .and_then(Value::as_u64),
                );
            }
            (StatusCode::OK, Json(message)).into_response()
        }
        Err(error) => {
            let (status, kind) = status_for_error_type(error.error_type());
            let mapped = ProviderError::new(status, kind, error.message_with(&*sanitize));
            if let (Some(traffic), Some(capture)) = (ctx.traffic.as_ref(), capture) {
                traffic.write_json(
                    "051-downstream-response",
                    &redact_every_string(
                        &json!({
                            "type": "error",
                            "error": {"type": mapped.error_type(), "message": &mapped.message},
                        }),
                        &*redact,
                    ),
                );
                capture.finish(
                    traffic,
                    "buffered_error",
                    Some(error.kind()),
                    BUFFERED_SUMMARY_NAME,
                );
            }
            failure_response(mapped.into())
        }
    }
}

/// Feeds the upstream body to one translator chunk by chunk until the
/// response is complete. More than [`MAX_BUFFERED_RESPONSE_BYTES`] is an
/// error: an upstream that never stops sending must not grow the message
/// without bound.
async fn read_message<U>(
    upstream: U,
    config: TranslatorConfig,
    sanitize: &Sanitizer,
    redact: &Sanitizer,
    mut capture: Option<&mut UpstreamCapture>,
) -> Result<Value, StreamError>
where
    U: Stream<Item = Result<Bytes, ()>>,
{
    let mut upstream = std::pin::pin!(upstream);
    let mut translator = LiveStreamTranslator::new(config);
    let mut read = 0usize;
    let mut read_deadline: Option<tokio::time::Instant> = None;
    loop {
        let next = if translator.has_finish() {
            let deadline = *read_deadline
                .get_or_insert_with(|| tokio::time::Instant::now() + POST_TERMINAL_READ_LIMIT);
            match tokio::time::timeout_at(deadline, upstream.next()).await {
                Ok(next) => next,
                Err(_) => {
                    translator.finish_after_read_limit()?;
                    return translator.response();
                }
            }
        } else {
            upstream.next().await
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|()| StreamError::Transport)?;
        read = read.saturating_add(chunk.len());
        if read > MAX_BUFFERED_RESPONSE_BYTES {
            return Err(StreamError::Malformed("the response exceeds 16 MiB"));
        }
        if let Some(capture) = capture.as_deref_mut() {
            capture.upstream(&chunk, sanitize, redact);
        }
        translator.push(&chunk)?;
        if translator.is_finished() {
            return translator.response();
        }
    }
    translator.finish()?;
    translator.response()
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// `claude-code-proxy google-ai-plan auth ...`.
pub struct GoogleAiPlanCli;

/// The provider's CLI handlers.
pub static GOOGLE_AI_PLAN_CLI: GoogleAiPlanCli = GoogleAiPlanCli;

impl GoogleAiPlanCli {
    /// The OAuth client in use, from the settings of this process.
    fn client() -> anyhow::Result<GoogleAiPlanClient> {
        Ok(crate::config::google_ai_plan_settings()?.client)
    }
}

impl CliHandlers for GoogleAiPlanCli {
    fn login(&self) -> anyhow::Result<()> {
        let client = Self::client()?;
        let store = file_store();
        auth::login::login(&client, &store)?;
        println!(
            "Google AI Ultra plan sign-in saved in {}",
            store.auth_path()
        );
        Ok(())
    }

    /// There is no device-code sign-in: the browser sign-in accepts the
    /// final URL pasted from a browser on another machine.
    fn device(&self) -> anyhow::Result<()> {
        self.login()
    }

    fn status(&self) -> anyhow::Result<()> {
        let report = status_report(&Self::client()?, &file_store(), auth::now_ms());
        for line in &report.lines {
            println!("{line}");
        }
        report.outcome
    }

    fn logout(&self) -> anyhow::Result<()> {
        let store = file_store();
        store.clear_auth()?;
        println!("Google AI Ultra plan credentials removed");
        Ok(())
    }
}

/// What `auth status` prints, and how it ends.
struct StatusReport {
    /// The lines to print, in order.
    lines: Vec<String>,
    /// The command's result: `Not authenticated` without a sign-in, the
    /// [`GoogleAiPlanAuthError::ClientChanged`] error for a sign-in issued to
    /// another client, or the error that kept the credential from being read.
    outcome: anyhow::Result<()>,
}

/// The status of the sign-in in `store` for `client` at `now_ms`. Only the
/// auth path is reported before an error, so a sign-in that cannot be used is
/// never shown as authenticated.
fn status_report<S: AuthStorage<StoredAuth>>(
    client: &GoogleAiPlanClient,
    store: &GoogleAiPlanTokenStore<S>,
    now_ms: u64,
) -> StatusReport {
    let mut lines = vec![format!("Auth path: {}", store.auth_path())];
    let stored = match store.load_auth() {
        Ok(Some(stored)) => stored,
        Ok(None) => {
            return StatusReport {
                lines,
                outcome: Err(anyhow::anyhow!("Not authenticated")),
            };
        }
        Err(error) => {
            return StatusReport {
                lines,
                outcome: Err(error),
            };
        }
    };
    if stored.client_id != client.client_id {
        return StatusReport {
            lines,
            outcome: Err(GoogleAiPlanAuthError::ClientChanged.into()),
        };
    }
    // A credential from an older DPoP-bound sign-in cannot be refreshed on the
    // DPoP-less path, so it is reported as needing a fresh sign-in rather than
    // shown as authenticated.
    if !stored.dpop_key.is_empty() {
        return StatusReport {
            lines,
            outcome: Err(GoogleAiPlanAuthError::SignInAgain.into()),
        };
    }
    let kind = if *client == GoogleAiPlanClient::built_in() {
        "built-in"
    } else {
        "configured"
    };
    lines.push("Authenticated: true".to_string());
    lines.push(format!("OAuth client: {kind}"));
    lines.push(match stored.expires_at_ms.checked_sub(now_ms) {
        Some(left) if left > 0 => format!("Access token expires in {}s", left / 1000),
        _ => "Access token expired; it is refreshed on the next request".to_string(),
    });
    StatusReport {
        lines,
        outcome: Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use axum::Router;
    use axum::http::{HeaderMap, Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::auth::dpop::DpopKey;
    use super::auth::test_support::{FakeReply, fake_token_endpoint, synthetic_client};
    use super::*;
    use crate::auth::InMemoryAuthStore;
    use crate::monitor::{EndpointKind, MonitorHandle};
    use crate::providers::vertex::client::client_builder;
    use crate::traffic::test_capture;

    /// A synthetic access token, shaped like Google's so the general
    /// sanitiser would also catch it.
    const TOKEN: &str = "ya29.synthetic-access-token-a1b2c3d4e5f6";
    const MODEL: &str = "google-ai-plan/gemini-4-hydrogen";
    const USER_AGENT: &str = "synthetic-agent/1.0";

    /// What the fake Cloud Code PA endpoint answers.
    #[derive(Clone)]
    struct Reply {
        status: u16,
        headers: Vec<(&'static str, String)>,
        chunks: Vec<String>,
        /// Keep the body open after the last chunk.
        hold_open: bool,
        /// Never send the response headers.
        never_answer: bool,
    }

    impl Reply {
        fn sse(body: String) -> Self {
            Self {
                status: 200,
                headers: vec![("content-type", "text/event-stream".to_string())],
                chunks: vec![body],
                hold_open: false,
                never_answer: false,
            }
        }

        fn error(status: u16, body: String) -> Self {
            Self {
                status,
                headers: vec![("content-type", "application/json".to_string())],
                chunks: vec![body],
                hold_open: false,
                never_answer: false,
            }
        }

        fn with_header(mut self, name: &'static str, value: &str) -> Self {
            self.headers.push((name, value.to_string()));
            self
        }
    }

    /// One request the fake endpoint received.
    #[derive(Debug, Clone)]
    struct Seen {
        path: String,
        headers: HeaderMap,
        body: Value,
    }

    struct FakeUpstream {
        base_url: String,
        seen: Arc<Mutex<Vec<Seen>>>,
    }

    impl FakeUpstream {
        fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }
    }

    async fn fake_upstream(reply: Reply) -> FakeUpstream {
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let recorded = seen.clone();
        let app = Router::new().route(
            "/{*path}",
            axum::routing::post(move |uri: http::Uri, headers: HeaderMap, body: Bytes| {
                let recorded = recorded.clone();
                let reply = reply.clone();
                async move {
                    if uri.path().ends_with(":loadCodeAssist") {
                        let mut response = Response::new(Body::from(
                            json!({"cloudaicompanionProject": "synthetic-project"}).to_string(),
                        ));
                        *response.status_mut() = StatusCode::OK;
                        response.headers_mut().insert(
                            http::header::CONTENT_TYPE,
                            HeaderValue::from_static("application/json"),
                        );
                        return response;
                    }
                    if uri.path().ends_with(":onboardUser") {
                        let mut response = Response::new(Body::from(
                            json!({"response": {"cloudaicompanionProject": {"id": "synthetic-project"}}})
                                .to_string(),
                        ));
                        *response.status_mut() = StatusCode::OK;
                        response.headers_mut().insert(
                            http::header::CONTENT_TYPE,
                            HeaderValue::from_static("application/json"),
                        );
                        return response;
                    }
                    recorded.lock().unwrap().push(Seen {
                        path: uri.path().to_string(),
                        headers,
                        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
                    });
                    if reply.never_answer {
                        std::future::pending::<()>().await;
                    }
                    let chunks = reply
                        .chunks
                        .into_iter()
                        .map(|chunk| Ok::<Bytes, Infallible>(Bytes::from(chunk)));
                    let body = if reply.hold_open {
                        Body::from_stream(
                            futures_util::stream::iter(chunks)
                                .chain(futures_util::stream::pending()),
                        )
                    } else {
                        Body::from_stream(futures_util::stream::iter(chunks))
                    };
                    let mut response = Response::new(body);
                    *response.status_mut() = StatusCode::from_u16(reply.status).unwrap();
                    for (name, value) in reply.headers {
                        response
                            .headers_mut()
                            .insert(name, HeaderValue::from_str(&value).unwrap());
                    }
                    response
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        FakeUpstream {
            base_url: format!("http://{addr}"),
            seen,
        }
    }

    fn signed_in(access: &str, expires_at_ms: u64) -> StoredAuth {
        StoredAuth {
            access: access.to_string(),
            refresh: "synthetic-refresh-token".to_string(),
            expires_at_ms,
            client_id: synthetic_client().client_id,
            // A Cloud Code PA sign-in carries no DPoP key.
            dpop_key: String::new(),
            dpop_nonce: None,
            project_id: None,
        }
    }

    fn fresh() -> StoredAuth {
        signed_in(TOKEN, auth::now_ms() + 3_600_000)
    }

    type TestProvider = GoogleAiPlanProvider<InMemoryAuthStore<StoredAuth>>;

    fn provider_with(
        base_url: &str,
        auth: Option<StoredAuth>,
        token_url: &str,
        header_timeout: Duration,
    ) -> TestProvider {
        let settings = GoogleAiPlanSettings {
            base_url: base_url.to_string(),
            client: synthetic_client(),
            user_agent: USER_AGENT.to_string(),
        };
        let store = GoogleAiPlanTokenStore::new(InMemoryAuthStore::new());
        if let Some(auth) = auth {
            store.save_auth(auth).unwrap();
        }
        let manager =
            GoogleAiPlanAuthManager::with_token_url(store, synthetic_client(), token_url).unwrap();
        let client = InteractionsClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            &settings,
        )
        .with_response_header_timeout(header_timeout);
        GoogleAiPlanProvider::with_parts(settings, client, manager)
    }

    fn provider(base_url: &str, auth: Option<StoredAuth>) -> TestProvider {
        // No test here refreshes through this URL: nothing listens on it.
        provider_with(
            base_url,
            auth,
            "http://127.0.0.1:9/token",
            Duration::from_secs(30),
        )
    }

    fn app(provider: impl Provider + 'static) -> Router {
        let providers: Vec<Arc<dyn Provider>> = vec![Arc::new(provider)];
        crate::server::app(Arc::new(crate::Registry::from_providers(
            crate::config::AliasProvider::Codex,
            providers,
        )))
    }

    /// `step`, failing the test after 20 seconds instead of hanging.
    async fn bounded<T>(step: impl std::future::Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(20), step)
            .await
            .expect("finished within 20 seconds")
    }

    async fn post(app: &Router, path: &str, body: Value) -> (StatusCode, HeaderMap, String) {
        let request = Request::post(path)
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = tokio::time::timeout(Duration::from_secs(20), app.clone().oneshot(request))
            .await
            .expect("the proxy answers")
            .unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = tokio::time::timeout(Duration::from_secs(20), response.into_body().collect())
            .await
            .expect("the body ends")
            .unwrap()
            .to_bytes();
        (
            status,
            headers,
            String::from_utf8_lossy(&bytes).into_owned(),
        )
    }

    fn request(stream: bool) -> Value {
        json!({
            "model": MODEL,
            "max_tokens": 1024,
            "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "tools": [{"name": "read_file", "description": "Read a file", "input_schema": {
                "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]
            }}],
            "messages": [{"role": "user", "content": "Read a.py"}]
        })
    }

    fn frame(data: Value) -> String {
        format!("data: {data}\n\n")
    }

    fn frames(events: &[Value]) -> String {
        events.iter().cloned().map(frame).collect()
    }

    /// Thought, text and a function call, completed with `finishReason: "STOP"`.
    fn tool_turn_events() -> Vec<Value> {
        vec![
            json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"thought": true, "text": "Need the file."}]
                        }
                    }]
                },
                "traceId": "trace-1"
            }),
            json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [
                                {"thought": true, "thoughtSignature": "c3ludGhldGljLXNpZw"},
                                {"text": "Reading it."}
                            ]
                        }
                    }]
                },
                "traceId": "trace-1"
            }),
            json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "args": {"path": "a.py"}
                                }
                            }]
                        },
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 114,
                        "cachedContentTokenCount": 10,
                        "candidatesTokenCount": 26,
                        "thoughtsTokenCount": 44,
                        "totalTokenCount": 184
                    }
                },
                "traceId": "trace-1"
            }),
        ]
    }

    fn tool_turn() -> String {
        frames(&tool_turn_events()) + "data: [DONE]\n\n"
    }

    fn events(body: &str) -> Vec<(String, Value)> {
        parse_sse_events(body.as_bytes())
            .into_iter()
            .map(|event| {
                let data = serde_json::from_str(&event.data)
                    .unwrap_or_else(|error| panic!("{error}: {}", event.data));
                (event.event.unwrap_or_default(), data)
            })
            .collect()
    }

    fn names(events: &[(String, Value)]) -> Vec<&str> {
        events.iter().map(|(name, _)| name.as_str()).collect()
    }

    fn header<'a>(seen: &'a Seen, name: &str) -> &'a str {
        seen.headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_else(|| panic!("no {name} header: {:?}", seen.headers))
    }

    fn assert_upstream_request(seen: &[Seen], token: &str) {
        assert_eq!(seen.len(), 1, "one upstream call: {seen:?}");
        let seen = &seen[0];
        assert_eq!(seen.path, "/v1internal:streamGenerateContent");
        assert_eq!(header(seen, "authorization"), format!("Bearer {token}"));
        assert_eq!(header(seen, "accept"), "text/event-stream");
        assert_eq!(header(seen, "content-type"), "application/json");
        assert_eq!(header(seen, "user-agent"), USER_AGENT);
        assert_eq!(seen.body["project"], "synthetic-project", "{}", seen.body);
        assert_eq!(seen.body["model"], "gemini-4-hydrogen", "{}", seen.body);
        assert_eq!(seen.body["userAgent"], "antigravity", "{}", seen.body);
        assert_eq!(seen.body["requestType"], "agent", "{}", seen.body);
    }

    #[tokio::test]
    async fn streaming_turn_with_thinking_text_and_a_tool_call_reaches_the_client() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let (status, headers, body) = post(
            &app(provider(&upstream.base_url, Some(fresh()))),
            "/v1/messages",
            request(true),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(headers["content-type"], "text/event-stream", "{body}");
        assert_upstream_request(&upstream.seen(), TOKEN);

        let events = events(&body);
        assert_eq!(
            names(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
            "{body}"
        );
        assert_eq!(events[0].1["message"]["model"], MODEL, "{body}");
        assert_eq!(events[1].1["content_block"]["type"], "thinking", "{body}");
        assert_eq!(events[2].1["delta"]["thinking"], "Need the file.", "{body}");
        let signature = events[3].1["delta"]["signature"]
            .as_str()
            .unwrap_or_default();
        assert!(
            signature.starts_with("ccp:google-ai-plan:v1:gemini-4-hydrogen:"),
            "{body}"
        );
        assert_eq!(events[6].1["delta"]["text"], "Reading it.", "{body}");
        assert_eq!(events[8].1["content_block"]["type"], "tool_use", "{body}");
        assert_eq!(events[8].1["content_block"]["name"], "read_file", "{body}");
        assert_eq!(
            events[9].1["delta"]["partial_json"], "{\"path\":\"a.py\"}",
            "{body}"
        );
        assert_eq!(events[11].1["delta"]["stop_reason"], "tool_use", "{body}");
        assert_eq!(
            events[11].1["usage"],
            json!({"input_tokens":104,"output_tokens":70,"cache_creation_input_tokens":0,"cache_read_input_tokens":10}),
            "{body}"
        );
    }

    #[tokio::test]
    async fn non_streaming_turn_is_one_message_with_the_same_content() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let (status, _, body) = post(
            &app(provider(&upstream.base_url, Some(fresh()))),
            "/v1/messages",
            request(false),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_upstream_request(&upstream.seen(), TOKEN);
        let message: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(message["model"], MODEL, "{body}");
        assert_eq!(message["stop_reason"], "tool_use", "{body}");
        let kinds: Vec<&str> = message["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["thinking", "text", "tool_use"], "{body}");
        assert_eq!(message["content"][1]["text"], "Reading it.", "{body}");
        assert_eq!(
            message["content"][2]["input"],
            json!({"path":"a.py"}),
            "{body}"
        );
        assert_eq!(
            message["usage"],
            json!({"input_tokens":104,"output_tokens":70,"cache_creation_input_tokens":0,"cache_read_input_tokens":10}),
            "{body}"
        );
    }

    #[tokio::test]
    async fn streaming_reports_progress_and_usage_to_the_monitor() {
        // The upstream stays open after the `finishReason`: the
        // response must end on its own.
        let mut reply = Reply::sse(frames(&tool_turn_events()));
        reply.hold_open = true;
        let upstream = fake_upstream(reply).await;
        let provider = provider(&upstream.base_url, Some(fresh()));
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "req_1",
            Some("session_1".into()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.provider_selected("req_1", "google-ai-plan", MODEL, None);
        let ctx = RequestContext {
            req_id: "req_1".to_string(),
            session_id: Some("session_1".to_string()),
            session_seq: Some(1),
            provider: "google-ai-plan".to_string(),
            traffic: None,
            monitor: Some(monitor.clone()),
        };
        let body: MessagesRequest = serde_json::from_value(request(true)).unwrap();
        let response = bounded(provider.handle_messages(body, ctx)).await;
        monitor.request_completed("req_1", response.status().as_u16(), None, None);
        let bytes = bounded(response.into_body().collect())
            .await
            .unwrap()
            .to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        assert_eq!(events(&text).last().unwrap().0, "message_stop", "{text}");
        let snapshot = monitor.snapshot();
        let recent = snapshot
            .recent
            .iter()
            .find(|request| request.request_id == "req_1")
            .expect("the request is listed");
        assert_eq!(
            recent.model.as_deref(),
            Some("google-ai-plan/gemini-4-hydrogen → gemini-4-hydrogen")
        );
        assert_eq!(recent.input_tokens, Some(104));
        assert_eq!(recent.output_tokens, Some(70));
        assert!(recent.streamed_bytes > 0);
    }

    #[tokio::test]
    async fn a_request_without_a_sign_in_is_a_401_naming_the_login_command() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let app = app(provider(&upstream.base_url, None));
        for stream in [false, true] {
            let (status, _, body) = post(&app, "/v1/messages", request(stream)).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
            let error: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(error["error"]["type"], "authentication_error", "{body}");
            assert!(body.contains(auth::LOGIN_COMMAND), "{body}");
        }
        assert!(
            upstream.seen().is_empty(),
            "no upstream call without a sign-in"
        );
    }

    #[tokio::test]
    async fn a_legacy_dpop_sign_in_is_a_401_asking_to_sign_in_again() {
        // A credential from an older DPoP-bound sign-in cannot be refreshed on
        // the DPoP-less path, so the proxy asks for a fresh sign-in without
        // calling the token endpoint or the upstream.
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let legacy = StoredAuth {
            dpop_key: DpopKey::generate().unwrap().to_stored(),
            ..fresh()
        };
        let app = app(provider(&upstream.base_url, Some(legacy)));
        let (status, _, body) = post(&app, "/v1/messages", request(false)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
        let error: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error["error"]["type"], "authentication_error", "{body}");
        assert!(body.contains(auth::LOGIN_COMMAND), "{body}");
        assert!(
            upstream.seen().is_empty(),
            "a legacy credential never reaches the upstream"
        );
    }

    #[tokio::test]
    async fn an_expired_access_token_is_refreshed_before_the_model_call() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let token_endpoint = fake_token_endpoint(vec![FakeReply::ok(
            json!({"access_token": "ya29.synthetic-refreshed-token-0f9e8d", "expires_in": 3599}),
        )]);
        let provider = provider_with(
            &upstream.base_url,
            Some(signed_in("ya29.synthetic-expired-token", 0)),
            &token_endpoint.url,
            Duration::from_secs(30),
        );
        let (status, _, body) = post(&app(provider), "/v1/messages", request(false)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let refreshes = token_endpoint.seen();
        assert_eq!(refreshes.len(), 1, "one refresh");
        assert_eq!(refreshes[0].form["grant_type"], "refresh_token");
        assert!(
            refreshes[0].dpop.is_none(),
            "the refresh sends no DPoP header"
        );
        assert_upstream_request(&upstream.seen(), "ya29.synthetic-refreshed-token-0f9e8d");
    }

    #[tokio::test]
    async fn a_refused_refresh_is_a_502_without_the_refresh_token_or_the_client_secret() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let refresh = "synthetic-refresh-token";
        let secret = synthetic_client().client_secret;
        let token_endpoint = fake_token_endpoint(vec![FakeReply {
            status: 400,
            headers: Vec::new(),
            body: json!({
                "error": "invalid_request",
                "error_description": format!("cannot use {refresh} with {secret}"),
            })
            .to_string(),
        }]);
        let provider = provider_with(
            &upstream.base_url,
            Some(signed_in("ya29.synthetic-expired-token", 0)),
            &token_endpoint.url,
            Duration::from_secs(30),
        );
        let (status, _, body) = post(&app(provider), "/v1/messages", request(false)).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
        assert!(body.contains("api_error"), "{body}");
        assert!(
            body.contains("Google token refresh failed: 400 invalid_request cannot use [redacted] with [redacted]"),
            "{body}"
        );
        for leaked in [refresh, secret.as_str()] {
            assert!(!body.contains(leaked), "{leaked} in {body}");
        }
        assert!(upstream.seen().is_empty(), "no model call without a token");
    }

    /// The body of a 429 or 403 the way Google sends one, echoing the access
    /// token.
    fn google_error(code: u16, status: &str, message: &str) -> String {
        json!({"error":{"code":code,"status":status,"message":format!("{message} (token {TOKEN})")}})
            .to_string()
    }

    #[tokio::test]
    async fn upstream_errors_are_passed_on_with_their_retry_decision() {
        let daily = google_error(
            429,
            "RESOURCE_EXHAUSTED",
            "You've reached today's AI Ultra limit for gemini-4-hydrogen in third-party apps.",
        );
        let not_on_plan = google_error(
            429,
            "RESOURCE_EXHAUSTED",
            "Rate limit exceeded for model gemini-4-argon (limit: 0 tokens per day)",
        );
        // (name, reply, model, status, error type, retry-after, x-should-retry, text)
        type Case = (
            &'static str,
            Reply,
            &'static str,
            StatusCode,
            &'static str,
            Option<&'static str>,
            Option<&'static str>,
            &'static str,
        );
        let tests: Vec<Case> = vec![
            (
                "per-minute 429: retried after Google's delay",
                Reply::error(
                    429,
                    google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded."),
                )
                .with_header("retry-after", "17"),
                MODEL,
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                Some("17"),
                None,
                "Quota exceeded.",
            ),
            (
                "daily limit: final",
                Reply::error(429, daily).with_header("retry-after", "19104"),
                MODEL,
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                None,
                Some("false"),
                "Daily limit reached for gemini-4-hydrogen",
            ),
            (
                "model outside the plan: final",
                Reply::error(429, not_on_plan),
                "google-ai-plan/gemini-4-argon",
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                None,
                Some("false"),
                "gemini-4-argon is not available on the Google AI Ultra plan",
            ),
            (
                "401: sign in again",
                Reply::error(
                    401,
                    google_error(401, "UNAUTHENTICATED", "Invalid credentials."),
                ),
                MODEL,
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                None,
                None,
                auth::LOGIN_COMMAND,
            ),
            (
                "403: no eligible plan",
                Reply::error(403, google_error(403, "PERMISSION_DENIED", "Forbidden.")),
                MODEL,
                StatusCode::FORBIDDEN,
                "permission_error",
                None,
                None,
                "does not have an eligible Google AI Ultra plan",
            ),
        ];
        for (name, reply, model, status, error_type, retry_after, should_retry, text) in tests {
            let upstream = fake_upstream(reply).await;
            let app = app(provider(&upstream.base_url, Some(fresh())));
            for stream in [false, true] {
                let mut body = request(stream);
                body["model"] = json!(model);
                let (got, headers, body) = post(&app, "/v1/messages", body).await;
                assert_eq!(got, status, "{name} (stream {stream}): {body}");
                let error: Value = serde_json::from_str(&body)
                    .unwrap_or_else(|error| panic!("{name}: {error}: {body}"));
                assert_eq!(error["error"]["type"], error_type, "{name}: {body}");
                assert!(body.contains(text), "{name}: {body}");
                assert!(!body.contains(TOKEN), "{name}: the token leaked: {body}");
                let header = |key: &str| headers.get(key).map(|value| value.to_str().unwrap());
                assert_eq!(header("retry-after"), retry_after, "{name}: {headers:?}");
                assert_eq!(
                    header("x-should-retry"),
                    should_retry,
                    "{name}: {headers:?}"
                );
            }
            assert_eq!(upstream.seen().len(), 2, "{name}: not retried by the proxy");
        }
    }

    #[tokio::test]
    async fn a_stream_that_closes_before_completion_ends_in_one_error_event() {
        let mut cut = tool_turn_events();
        cut.pop();
        let upstream = fake_upstream(Reply::sse(frames(&cut))).await;
        let app = app(provider(&upstream.base_url, Some(fresh())));
        let (status, _, body) = post(&app, "/v1/messages", request(true)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let events = events(&body);
        let errors: Vec<_> = events.iter().filter(|(name, _)| name == "error").collect();
        assert_eq!(errors.len(), 1, "{body}");
        assert_eq!(events.last().unwrap().0, "error", "{body}");
        assert!(
            !names(&events).contains(&"message_stop"),
            "no message_stop: {body}"
        );
        assert_eq!(errors[0].1["error"]["type"], "api_error", "{body}");
        assert!(
            errors[0].1["error"]["message"]
                .as_str()
                .unwrap()
                .contains("ended without a finishReason"),
            "{body}"
        );

        let (status, _, body) = post(&app, "/v1/messages", request(false)).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
        assert!(body.contains("ended without a finishReason"), "{body}");
        assert_eq!(upstream.seen().len(), 2, "not sent again");
    }

    #[tokio::test]
    async fn the_upstream_is_not_read_after_interaction_completed() {
        let mut reply = Reply::sse(frames(&tool_turn_events()));
        reply.hold_open = true;
        let upstream = fake_upstream(reply).await;
        let app = app(provider(&upstream.base_url, Some(fresh())));
        for stream in [true, false] {
            // `post` fails the test when the body does not end.
            let (status, _, body) = post(&app, "/v1/messages", request(stream)).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            if stream {
                assert_eq!(events(&body).last().unwrap().0, "message_stop", "{body}");
            } else {
                assert!(body.contains("\"stop_reason\":\"tool_use\""), "{body}");
            }
        }
    }

    #[tokio::test]
    async fn an_upstream_that_sends_no_headers_is_a_504() {
        let mut reply = Reply::sse(String::new());
        reply.never_answer = true;
        let upstream = fake_upstream(reply).await;
        let provider = provider_with(
            &upstream.base_url,
            Some(fresh()),
            "http://127.0.0.1:9/token",
            Duration::from_millis(300),
        );
        let (status, _, body) = post(&app(provider), "/v1/messages", request(true)).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert!(body.contains("no response headers within"), "{body}");
        assert!(body.contains("api_error"), "{body}");
    }

    fn files_under(root: &std::path::Path) -> Vec<(std::path::PathBuf, String)> {
        let mut out = Vec::new();
        let mut dirs = vec![root.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    dirs.push(path);
                } else {
                    let text = String::from_utf8_lossy(&std::fs::read(&path).unwrap()).into_owned();
                    out.push((path, text));
                }
            }
        }
        out
    }

    #[tokio::test]
    async fn the_access_token_stays_out_of_every_capture_file_and_error() {
        let echo = format!("echo {TOKEN} and {}", TOKEN.replace('.', "%2E"));
        let in_stream_error = frames(&[
            json!({
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": "partial"}]
                        }
                    }]
                }
            }),
            json!({"error":{"code":"invalid_request","message":echo}}),
        ]);
        let completed_with_echo = frames(&[json!({
            "response": {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": echo}]
                    },
                    "finishReason": "STOP"
                }]
            }
        })]);
        let replies = [
            (
                "error body",
                Reply::error(400, google_error(400, "INVALID_ARGUMENT", &echo)),
            ),
            ("in-stream error", Reply::sse(in_stream_error)),
            ("echoed text", Reply::sse(completed_with_echo)),
        ];
        for (name, reply) in replies {
            let upstream = fake_upstream(reply).await;
            let provider = provider(&upstream.base_url, Some(fresh()));
            for stream in [true, false] {
                let dir = tempfile::TempDir::new().unwrap();
                let ctx = RequestContext {
                    req_id: "req_capture".to_string(),
                    session_id: None,
                    session_seq: None,
                    provider: "google-ai-plan".to_string(),
                    traffic: Some(Arc::new(test_capture(dir.path().join("traffic")))),
                    monitor: None,
                };
                let body: MessagesRequest = serde_json::from_value(request(stream)).unwrap();
                let response = bounded(provider.handle_messages(body, ctx)).await;
                let outcome = response
                    .extensions()
                    .get::<NativeResponseOutcome>()
                    .cloned();
                let bytes = bounded(response.into_body().collect())
                    .await
                    .unwrap()
                    .to_bytes();
                let text = String::from_utf8_lossy(&bytes);
                // Only a streaming response that was started carries an
                // outcome, and it holds a failure only when the stream failed.
                match (name, stream) {
                    ("in-stream error", true) => {
                        let failure = outcome
                            .as_ref()
                            .and_then(NativeResponseOutcome::failure)
                            .unwrap_or_else(|| panic!("{name}: no failure recorded: {text}"));
                        assert!(
                            failure.starts_with(
                                "Google AI Ultra plan stream reported an error (invalid_request): echo "
                            ),
                            "{name}: {failure}"
                        );
                        assert!(failure.contains("[redacted]"), "{name}: {failure}");
                    }
                    ("echoed text", true) => {
                        let outcome = outcome.as_ref().expect("a stream carries an outcome");
                        assert_eq!(outcome.failure(), None, "{name}: {text}");
                    }
                    _ => assert!(outcome.is_none(), "{name}/{stream}: unexpected outcome"),
                }
                let percent = TOKEN.replace('.', "%2E");
                for secret in [TOKEN, percent.as_str(), "synthetic-access-token"] {
                    // Model output reaches the client as it is; error prose
                    // does not.
                    if name != "echoed text" {
                        assert!(!text.contains(secret), "{name}/{stream}: response: {text}");
                    }
                    if let Some(failure) = outcome.as_ref().and_then(NativeResponseOutcome::failure)
                    {
                        assert!(
                            !failure.contains(secret),
                            "{name}/{stream}: outcome: {failure}"
                        );
                    }
                }
                let files = files_under(dir.path());
                assert!(
                    files.iter().any(|(path, _)| path
                        .to_string_lossy()
                        .contains("021-upstream-request-metadata")),
                    "{name}/{stream}: no request metadata: {files:?}"
                );
                // An HTTP error ends the request before any body is read.
                let read_a_body = name != "error body";
                assert_eq!(
                    files
                        .iter()
                        .any(|(path, _)| path.to_string_lossy().contains("061-google-ai-plan")),
                    read_a_body,
                    "{name}/{stream}: summary: {files:?}"
                );
                for (path, content) in &files {
                    for secret in [TOKEN, percent.as_str(), "synthetic-access-token"] {
                        assert!(
                            !content.contains(secret),
                            "{name}/{stream}: {} holds the token:\n{content}",
                            path.display()
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn count_tokens_is_answered_locally() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let app = app(provider(&upstream.base_url, None));
        let (status, _, body) = post(&app, "/v1/messages/count_tokens", request(false)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let expected = count_tokens::count_tokens(&serde_json::from_value(request(false)).unwrap());
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap(),
            json!({"input_tokens": expected})
        );
        let mut bad = request(false);
        bad["model"] = json!("google-ai-plan/gemini:4");
        let (status, _, body) = post(&app, "/v1/messages/count_tokens", bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(upstream.seen().is_empty(), "nothing sent upstream");
    }

    #[tokio::test]
    async fn an_invalid_setting_is_reported_by_name_in_every_request() {
        let message =
            "Invalid Google AI Ultra plan setting CCP_GOOGLE_AI_PLAN_BASE_URL: must use https";
        let app = app(GoogleAiPlanProvider::with_settings(
            Err(message.to_string()),
        ));
        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            for stream in [false, true] {
                let (status, _, body) = post(&app, path, request(stream)).await;
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{path}: {body}");
                assert!(
                    body.contains("CCP_GOOGLE_AI_PLAN_BASE_URL"),
                    "{path}: {body}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_malformed_request_is_a_400_before_any_upstream_call() {
        let upstream = fake_upstream(Reply::sse(tool_turn())).await;
        let app = app(provider(&upstream.base_url, Some(fresh())));
        let mut stop = request(false);
        stop["stop_sequences"] = json!(["END"]);
        let mut bad_model = request(false);
        bad_model["model"] = json!("google-ai-plan/");
        for body in [stop, bad_model] {
            let (status, _, text) = post(&app, "/v1/messages", body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
            assert!(text.contains("invalid_request_error"), "{text}");
        }
        assert!(upstream.seen().is_empty());
    }

    #[test]
    fn status_reports_the_client_kind_and_the_expiry_without_secrets() {
        let store = GoogleAiPlanTokenStore::new(InMemoryAuthStore::new());
        let path = format!("Auth path: {}", store.auth_path());

        let report = status_report(&synthetic_client(), &store, 0);
        assert_eq!(report.lines, std::slice::from_ref(&path));
        assert_eq!(report.outcome.unwrap_err().to_string(), "Not authenticated");

        store.save_auth(signed_in(TOKEN, 90_000)).unwrap();
        // (name, now, expiry line)
        let tests = [
            ("live token", 30_000, "Access token expires in 60s"),
            (
                "expired token",
                120_000,
                "Access token expired; it is refreshed on the next request",
            ),
            (
                "expiring this millisecond",
                90_000,
                "Access token expired; it is refreshed on the next request",
            ),
        ];
        for (name, now, expiry) in tests {
            let report = status_report(&synthetic_client(), &store, now);
            assert!(report.outcome.is_ok(), "{name}: {:?}", report.outcome);
            assert_eq!(
                report.lines,
                [
                    path.clone(),
                    "Authenticated: true".to_string(),
                    "OAuth client: configured".to_string(),
                    expiry.to_string(),
                ],
                "{name}"
            );
        }

        let mut built_in = signed_in(TOKEN, 90_000);
        built_in.client_id = GoogleAiPlanClient::built_in().client_id;
        store.save_auth(built_in).unwrap();
        let report = status_report(&GoogleAiPlanClient::built_in(), &store, 0);
        assert_eq!(report.lines[2], "OAuth client: built-in");

        store.save_auth(signed_in(TOKEN, 90_000)).unwrap();
        let other = GoogleAiPlanClient {
            client_id: "synthetic-other-client".to_string(),
            client_secret: "synthetic-other-secret".to_string(),
        };
        let report = status_report(&other, &store, 0);
        assert_eq!(
            report.lines,
            [path],
            "a sign-in for another client is not shown as authenticated"
        );
        let error = report.outcome.unwrap_err().to_string();
        assert!(error.contains("different OAuth client"), "{error}");
        for secret in ["synthetic-other-secret", TOKEN, "synthetic-refresh-token"] {
            assert!(!error.contains(secret), "{error}");
        }
    }

    #[test]
    fn status_reports_a_legacy_dpop_sign_in_as_needing_a_fresh_login() {
        let store = GoogleAiPlanTokenStore::new(InMemoryAuthStore::new());
        let legacy = StoredAuth {
            dpop_key: DpopKey::generate().unwrap().to_stored(),
            ..signed_in(TOKEN, 90_000)
        };
        store.save_auth(legacy).unwrap();
        let report = status_report(&synthetic_client(), &store, 0);
        let error = report
            .outcome
            .expect_err("a legacy credential is not authenticated")
            .to_string();
        assert!(error.contains(auth::LOGIN_COMMAND), "{error}");
        assert_eq!(
            report.lines,
            [format!("Auth path: {}", store.auth_path())],
            "a legacy credential is not shown as authenticated"
        );
    }

    /// Marks a string as passed through the sanitiser, so the expected
    /// value shows exactly which strings were cleaned.
    fn marked(text: &str) -> String {
        format!("<clean:{text}>")
    }

    #[test]
    fn capture_sanitises_every_prose_field_and_keeps_the_rest() {
        let cases = [
            (
                "finishMessage in each wrapped candidate",
                json!({"response": {"candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "kept text"}]},
                        "finishReason": "SAFETY",
                        "finishMessage": "first prose"
                    },
                    {"finishReason": "RECITATION", "finishMessage": "second prose"}
                ]}}),
                json!({"response": {"candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "kept text"}]},
                        "finishReason": "SAFETY",
                        "finishMessage": "<clean:first prose>"
                    },
                    {"finishReason": "RECITATION", "finishMessage": "<clean:second prose>"}
                ]}}),
            ),
            (
                "blockReasonMessage in wrapped promptFeedback",
                json!({"response": {"promptFeedback": {
                    "blockReason": "SAFETY",
                    "blockReasonMessage": "block prose"
                }}}),
                json!({"response": {"promptFeedback": {
                    "blockReason": "SAFETY",
                    "blockReasonMessage": "<clean:block prose>"
                }}}),
            ),
            (
                "unwrapped top-level fields",
                json!({
                    "candidates": [{"finishReason": "OTHER", "finishMessage": "top prose"}],
                    "promptFeedback": {"blockReason": "OTHER", "blockReasonMessage": "top block"}
                }),
                json!({
                    "candidates": [{"finishReason": "OTHER", "finishMessage": "<clean:top prose>"}],
                    "promptFeedback": {
                        "blockReason": "OTHER",
                        "blockReasonMessage": "<clean:top block>"
                    }
                }),
            ),
            (
                "error message still sanitised",
                json!({"error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "err prose"
                }}),
                json!({"error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "<clean:err prose>"
                }}),
            ),
            (
                "non-string under a prose key is searched, not dropped",
                json!({
                    "finishMessage": {"detail": "inner", "message": "nested prose"},
                    "message": [1, null]
                }),
                json!({
                    "finishMessage": {"detail": "inner", "message": "<clean:nested prose>"},
                    "message": [1, null]
                }),
            ),
            (
                "non-object values pass through",
                json!(["plain", 3, true, null]),
                json!(["plain", 3, true, null]),
            ),
        ];
        for (name, input, expected) in cases {
            let cleaned = sanitize_capture_prose(&input, &|text: &str| marked(text));
            assert_eq!(cleaned, expected, "{name}");
        }
    }

    #[test]
    fn capture_prose_fields_lose_the_access_token() {
        let echo = format!("echo {TOKEN}");
        let event = json!({"response": {
            "candidates": [{"finishReason": "SAFETY", "finishMessage": echo}],
            "promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": echo}
        }});
        let sanitize = sanitizer(TOKEN);
        let cleaned = sanitize_capture_prose(&event, &*sanitize);
        let response = &cleaned["response"];
        for (field, text) in [
            ("finishMessage", &response["candidates"][0]["finishMessage"]),
            (
                "blockReasonMessage",
                &response["promptFeedback"]["blockReasonMessage"],
            ),
        ] {
            let text = text
                .as_str()
                .unwrap_or_else(|| panic!("{field}: not a string: {cleaned}"));
            assert!(!text.contains(TOKEN), "{field} kept the token: {text}");
        }
        assert_eq!(
            response["candidates"][0]["finishReason"], "SAFETY",
            "{cleaned}"
        );
        assert_eq!(
            response["promptFeedback"]["blockReason"], "SAFETY",
            "{cleaned}"
        );
    }
}
