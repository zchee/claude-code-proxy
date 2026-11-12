//! Google Cloud Vertex AI provider: Anthropic Messages requests with
//! Application Default Credentials, for Gemini models (sent as
//! `streamGenerateContent`), for the xAI Grok, OpenAI and Meta Muse partner
//! models (sent as OpenAI Responses requests) and for Mistral (sent in
//! Mistral's chat format on `streamRawPredict`).
//!
//! The upstream is always asked for a stream. A non-streaming client gets a
//! message built from the same translation, so both modes decide the stop
//! reason and content the same way.
//!
//! Anthropic Claude models (`vertex/anthropic/<model-id>`) are passed
//! through as Anthropic Messages instead; see [`claude`].

pub mod auth;
mod buffered;
pub mod claude;
pub mod cli;
pub mod client;
mod mistral;
pub mod models;
mod responses;
pub mod sanitize;
pub mod translate;

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use axum::{
    Json,
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};

use crate::anthropic::{
    error::json_error,
    schema::{CountTokensResponse, MessagesRequest},
};
use crate::config::ResolvedVertexConfig;
use crate::provider::{
    CliHandlers, Generation, GenerationBody, Provider, ProviderError, ProviderErrorKind,
    RequestContext,
};
use crate::providers::codex::native::NativeResponseOutcome;
use crate::providers::grok::translate::stream::SseDecoder;
use crate::providers::kimi::count_tokens;
use crate::providers::opencode::chat::sanitize_upstream_messages;
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use self::auth::{VertexAuthError, VertexCredentials};
use self::buffered::{BufferedFailure, BufferedSink, ReadEnd, read_upstream};
use self::cli::VERTEX_CLI;
use self::client::{VertexClient, VertexError, VertexResponse, VertexTarget};
use self::models::{ModelError, ResolvedModel, WireFormat};
use self::sanitize::sanitize_vertex_text;
use self::translate::request::prepare_request;
use self::translate::stream::{
    LiveStreamTranslator, MAX_SSE_FRAME_BYTES, Sanitizer, StreamError, StreamSinks,
    TranslatorConfig, sanitize_capture, stream_body,
};

/// Largest upstream body read for a non-streaming response. The body is
/// translated as it arrives rather than kept, but an upstream that never
/// stops sending must not grow the message without bound.
pub const MAX_BUFFERED_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Traffic-capture summary of a non-streaming response. The streaming one is
/// `061-vertex-stream-summary`; the name says which path produced the files.
const BUFFERED_SUMMARY_NAME: &str = "061-vertex-buffered-summary";

/// What every request needs once the settings and HTTP client exist.
struct Ready {
    config: ResolvedVertexConfig,
    client: VertexClient,
    /// Set by the first request that obtains an access token with them, and
    /// kept from then on. Until then each request looks for Application
    /// Default Credentials again, so a login or key file supplied after a
    /// failed request is found without a restart.
    credentials: OnceLock<VertexCredentials>,
}

/// One request that Vertex AI accepted: the response body is still unread.
struct Started {
    upstream: VertexResponse,
    reception: Reception,
    sanitize: Sanitizer,
    model_id: String,
}

/// How the upstream body of a started request is translated.
enum Reception {
    /// Gemini `streamGenerateContent`.
    Gemini(TranslatorConfig),
    /// An OpenAI Responses stream (xAI Grok, OpenAI, Meta Muse).
    /// `model` is the name the client asked for.
    Responses { message_id: String, model: String },
    /// A Mistral Chat-Completions-shaped stream. `model` is the name the
    /// client asked for.
    Mistral { message_id: String, model: String },
}

/// The `vertex` provider.
pub struct VertexProvider {
    ready: OnceLock<Arc<Ready>>,
}

impl VertexProvider {
    /// The server's provider. Settings, credentials and the HTTP client are
    /// created on first use rather than here: building the client loads the
    /// system root certificates, loading credentials reads files and needs
    /// the async runtime, and every CLI command and server start constructs
    /// every provider.
    pub fn new() -> Self {
        Self {
            ready: OnceLock::new(),
        }
    }

    /// A provider with fixed settings, credentials and client, for tests
    /// against a local upstream (point the settings' base URL at it).
    pub fn with_client(
        client: VertexClient,
        credentials: VertexCredentials,
        config: ResolvedVertexConfig,
    ) -> Self {
        Self {
            ready: OnceLock::from(Arc::new(Ready {
                config,
                client,
                credentials: OnceLock::from(credentials),
            })),
        }
    }

    /// The settings and HTTP client, built on first use. A failure is
    /// returned and not kept, so the next request tries again.
    fn ready(&self) -> Result<&Arc<Ready>, ProviderError> {
        if let Some(ready) = self.ready.get() {
            return Ok(ready);
        }
        let config = crate::config::vertex_config();
        check_settings(&config)?;
        let client = VertexClient::new().map_err(|error| {
            ProviderError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ProviderErrorKind::Api,
                format!("Vertex AI transport is unavailable: {error}"),
            )
        })?;
        let built = Arc::new(Ready {
            config,
            client,
            credentials: OnceLock::new(),
        });
        // Concurrent first requests may each build one; all use the first.
        Ok(self.ready.get_or_init(|| built))
    }

    /// The credentials for a request. Without credentials that have already
    /// produced a token, Application Default Credentials are resolved and a
    /// token is fetched; only when that succeeds are they kept. The crate
    /// resolves ADC once, when the credentials are built (no file means the
    /// metadata server, for good), so keeping a selection that never
    /// produced a token would keep failing until the proxy restarts. The
    /// crate caches the token, so the request's own header fetch that
    /// follows does not ask again.
    async fn credentials(ready: &Ready) -> Result<&VertexCredentials, ProviderError> {
        if let Some(credentials) = ready.credentials.get() {
            return Ok(credentials);
        }
        let loaded = VertexCredentials::adc().map_err(VertexAuthError::into_provider_error)?;
        loaded
            .headers()
            .await
            .map_err(VertexAuthError::into_provider_error)?;
        // Concurrent first requests may each load one; all use the first.
        Ok(ready.credentials.get_or_init(|| loaded))
    }

    /// Resolves the model, translates the request and sends it. The
    /// streaming, non-streaming and OpenAI-compatible paths share this, so
    /// they get the same validation and error mapping.
    async fn start(
        &self,
        body: &MessagesRequest,
        ctx: &RequestContext,
    ) -> Result<Started, ProviderError> {
        let requested = body.model.clone().unwrap_or_default();
        let model = models::resolve(&requested).map_err(invalid_request)?;
        require_served(&requested, &model)?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model.upstream_model);
        }
        match model.wire {
            WireFormat::GenerateContent => self.start_gemini(body, ctx, requested, model).await,
            WireFormat::Responses { route } => {
                let translated =
                    responses::prepare_request(body, &model).map_err(invalid_request)?;
                let upstream = self
                    .send_partner(
                        &model,
                        |target| responses::url(target, &model, route),
                        &translated,
                        ctx,
                    )
                    .await?;
                Ok(Started {
                    sanitize: sanitizer(&upstream),
                    upstream,
                    reception: Reception::Responses {
                        message_id: new_message_id(),
                        model: requested,
                    },
                    model_id: model.upstream_model,
                })
            }
            WireFormat::MistralChat => {
                let translated = mistral::prepare_request(body, &model).map_err(invalid_request)?;
                let upstream = self
                    .send_partner(
                        &model,
                        |target| mistral::url(target, &model),
                        &translated,
                        ctx,
                    )
                    .await?;
                Ok(Started {
                    sanitize: sanitizer(&upstream),
                    upstream,
                    reception: Reception::Mistral {
                        message_id: new_message_id(),
                        model: requested,
                    },
                    model_id: model.upstream_model,
                })
            }
            // Claude requests are passed through by `claude` before this
            // point and never translated here.
            WireFormat::AnthropicMessages => Err(family_not_supported(&requested, &model)),
        }
    }

    /// Translates a Gemini request and sends it as `streamGenerateContent`.
    async fn start_gemini(
        &self,
        body: &MessagesRequest,
        ctx: &RequestContext,
        requested: String,
        model: ResolvedModel,
    ) -> Result<Started, ProviderError> {
        let translated = prepare_request(body, &model).map_err(invalid_request)?;
        let ready = self.ready()?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let credentials = Self::credentials(ready).await?;
        let upstream = ready
            .client
            .stream_generate_content(
                &ready.config,
                &model,
                &translated,
                credentials,
                ctx.traffic.as_deref(),
            )
            .await
            .map_err(VertexError::into_provider_error)?;
        let config = TranslatorConfig {
            message_id: new_message_id(),
            model: requested,
            signature_model: model.upstream_model.clone(),
            estimated_input_tokens: count_tokens::count_tokens(body),
            emit_thinking: translated.includes_thoughts(),
        };
        Ok(Started {
            sanitize: sanitizer(&upstream),
            upstream,
            reception: Reception::Gemini(config),
            model_id: model.upstream_model,
        })
    }

    /// Sends a partner model's request to the URL `url` builds for the
    /// model's target. The target is chosen before credentials are looked
    /// up, so a model that is not served in the configured location gets
    /// its 400, naming the settings, without a credential lookup or an
    /// upstream call. A 403 or 404 carries
    /// [`VertexError::with_partner_guidance`].
    async fn send_partner(
        &self,
        model: &ResolvedModel,
        url: impl FnOnce(&VertexTarget) -> Result<reqwest::Url, VertexError>,
        body: &Value,
        ctx: &RequestContext,
    ) -> Result<VertexResponse, ProviderError> {
        let ready = self.ready()?;
        let target = VertexTarget::for_model(&ready.config, model)
            .map_err(VertexError::into_provider_error)?;
        let url = url(&target).map_err(VertexError::into_provider_error)?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let credentials = Self::credentials(ready).await?;
        ready
            .client
            .post_sse(
                &target,
                &model.upstream_model,
                url,
                body,
                credentials,
                ctx.traffic.as_deref(),
            )
            .await
            .map_err(|error| {
                error
                    .with_partner_guidance(&target, &model.upstream_model)
                    .into_provider_error()
            })
    }
}

fn new_message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

/// The sanitiser for text from `upstream`. The token is copied before the
/// body is consumed: upstream prose inside the stream may echo it, and the
/// sanitiser must know it.
fn sanitizer(upstream: &VertexResponse) -> Sanitizer {
    let secrets: Vec<String> = upstream.access_token.iter().cloned().collect();
    Arc::new(move |text: &str| {
        let secrets: Vec<&str> = secrets.iter().map(String::as_str).collect();
        sanitize_vertex_text(text, &secrets)
    })
}

impl Default for VertexProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for VertexProvider {
    fn name(&self) -> &'static str {
        "vertex"
    }

    fn supported_models(&self) -> Vec<String> {
        models::advertised_models()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &VERTEX_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        if let Ok(model) = models::resolve(body.model.as_deref().unwrap_or_default())
            && model.wire == WireFormat::AnthropicMessages
        {
            return claude::handle_messages(self, body, model, ctx).await;
        }
        let started = match self.start(&body, &ctx).await {
            Ok(started) => started,
            Err(error) => return error_response(error),
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

    /// Answered locally with the shared estimator; nothing is sent upstream.
    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let requested = body.model.as_deref().unwrap_or_default();
        let model = match models::resolve(requested) {
            Ok(model) => model,
            Err(error) => return error_response(invalid_request(error)),
        };
        if let Err(error) = require_served(requested, &model) {
            return error_response(error);
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

    async fn generate_anthropic_stream(
        &self,
        mut body: MessagesRequest,
        ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        if let Ok(model) = models::resolve(body.model.as_deref().unwrap_or_default())
            && model.wire == WireFormat::AnthropicMessages
        {
            return claude::generate_anthropic_stream(self, body, model, ctx).await;
        }
        body.stream = true;
        let started = self.start(&body, &ctx).await?;
        let resolved_model = started.model_id.clone();
        Ok(Generation {
            body: GenerationBody::LiveSse(stream_response_body(started, &ctx, None)),
            resolved_model,
        })
    }
}

/// The Anthropic SSE body of a started request, translated by its wire's
/// translator. A failure the client sees as an `error` event is recorded in
/// `outcome`.
fn stream_response_body(
    started: Started,
    ctx: &RequestContext,
    outcome: Option<NativeResponseOutcome>,
) -> Body {
    let Started {
        upstream,
        reception,
        sanitize,
        ..
    } = started;
    let upstream = upstream.into_byte_stream();
    match reception {
        Reception::Gemini(config) => stream_body(
            upstream,
            config,
            StreamSinks {
                req_id: ctx.req_id.clone(),
                monitor: ctx.monitor.clone(),
                traffic: ctx.traffic.clone(),
                outcome,
                sanitize,
            },
        ),
        Reception::Responses { message_id, model } => report_error_events(
            responses::stream_body(upstream, message_id, model, ctx, sanitize),
            outcome,
        ),
        Reception::Mistral { message_id, model } => report_error_events(
            mistral::stream_body(upstream, message_id, model, ctx, sanitize),
            outcome,
        ),
    }
}

/// Records the message of the first `error` event of `body` in `outcome`,
/// for the shared wirings, which do not report to it themselves. Each chunk
/// is decoded on its own: every error event those wirings write is a whole
/// chunk, and a forwarded upstream chunk never carries a complete one (the
/// Messages pass-through replaces an upstream error event with its own).
fn report_error_events(body: Body, outcome: Option<NativeResponseOutcome>) -> Body {
    let Some(outcome) = outcome else {
        return body;
    };
    Body::from_stream(body.into_data_stream().map(move |chunk| {
        if let Ok(bytes) = chunk.as_ref()
            && let Ok(events) = SseDecoder::default().push(bytes)
        {
            for event in events
                .iter()
                .filter(|event| event.event.as_deref() == Some("error"))
            {
                let message = serde_json::from_str::<Value>(&event.data)
                    .ok()
                    .and_then(|value| {
                        value
                            .pointer("/error/message")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "Vertex AI stream failed".to_string());
                outcome.fail(message);
            }
        }
        chunk
    }))
}

/// Reads the whole upstream response and answers with one Anthropic message,
/// or with the error the stream would have ended in.
async fn buffered_response(started: Started, ctx: &RequestContext) -> Response {
    let Started {
        upstream,
        reception,
        sanitize,
        ..
    } = started;
    let mut capture = ctx
        .traffic
        .as_ref()
        .map(|_| BufferedCapture::for_reception(&reception));
    let upstream = upstream.into_byte_stream();
    let result = match reception {
        Reception::Gemini(config) => read_message(upstream, config, &sanitize, capture.as_mut())
            .await
            .map_err(BufferedFailure::Read),
        Reception::Responses { message_id, model } => {
            responses::read_response(upstream, &message_id, &model, &sanitize, capture.as_mut())
                .await
        }
        Reception::Mistral { message_id, model } => {
            mistral::read_response(upstream, &message_id, &model, &sanitize, capture.as_mut()).await
        }
    };
    match result {
        Ok(message) => {
            if let (Some(traffic), Some(capture)) = (ctx.traffic.as_ref(), capture) {
                traffic.write_json("051-downstream-response", &message);
                capture.finish(traffic, None);
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
        Err(failure) => {
            let mapped = buffered_error(&failure, &sanitize);
            if let (Some(traffic), Some(capture)) = (ctx.traffic.as_ref(), capture) {
                traffic.write_json(
                    "051-downstream-response",
                    &json!({
                        "type": "error",
                        "error": {"type": mapped.error_type(), "message": &mapped.message},
                    }),
                );
                capture.finish_with_kind(traffic, Some(failure.kind()));
            }
            error_response(mapped)
        }
    }
}

/// The non-streaming answer for a failure: [`stream_error`] for a failed
/// read, a 502 with the translator's sanitised message otherwise.
fn buffered_error(failure: &BufferedFailure, sanitize: &Sanitizer) -> ProviderError {
    match failure {
        BufferedFailure::Read(error) => stream_error(error, sanitize),
        BufferedFailure::Translate(error) => ProviderError::new(
            StatusCode::BAD_GATEWAY,
            ProviderErrorKind::Api,
            error.to_string(),
        ),
    }
}

/// Feeds the upstream body to one Gemini translator chunk by chunk.
///
/// The response is complete when the upstream closes, or
/// [`translate::stream::POST_TERMINAL_READ_LIMIT`] after the `finishReason`
/// when it stays open: the same rule the streaming path applies. More than
/// [`MAX_BUFFERED_RESPONSE_BYTES`] is an error. The reading loop is the one
/// every wire's non-streaming path shares.
async fn read_message<S>(
    upstream: S,
    config: TranslatorConfig,
    sanitize: &Sanitizer,
    capture: Option<&mut BufferedCapture>,
) -> Result<Value, StreamError>
where
    S: Stream<Item = Result<Bytes, VertexError>>,
{
    let mut translator = LiveStreamTranslator::new(config);
    match read_upstream(upstream, &mut translator, sanitize, capture).await? {
        // The Gemini translator never settles a body itself.
        ReadEnd::Closed | ReadEnd::Settled => translator.finish()?,
        ReadEnd::ReadLimit => translator.finish_after_read_limit()?,
    };
    translator.response()
}

impl BufferedSink for LiveStreamTranslator {
    fn push(&mut self, chunk: &[u8]) -> Result<(), StreamError> {
        LiveStreamTranslator::push(self, chunk).map(drop)
    }

    fn terminal_events(&self) -> usize {
        usize::from(self.is_terminal())
    }
}

/// Traffic capture of a non-streaming response. The upstream body is
/// recorded as decoded events whose upstream prose went through the
/// sanitiser, never as the raw bytes: Google error and finish messages can
/// quote the request, and key-based redaction cannot see inside prose.
struct BufferedCapture {
    decoder: SseDecoder,
    events: StreamTrafficCapture,
    bytes: u64,
    chunks: u64,
    /// Cleans one decoded upstream event with the sanitiser.
    clean: CleanEvent,
}

/// Cleans one decoded upstream event: the value and the sanitiser.
type CleanEvent = fn(&Value, &dyn Fn(&str) -> String) -> Value;

impl BufferedCapture {
    /// The capture of a Gemini response: [`sanitize_capture`] cleans the
    /// places Gemini puts upstream prose.
    fn new() -> Self {
        Self {
            decoder: SseDecoder::with_max_frame_bytes(MAX_SSE_FRAME_BYTES),
            events: StreamTrafficCapture::default(),
            bytes: 0,
            chunks: 0,
            clean: sanitize_capture,
        }
    }

    /// The capture for `reception`. A partner model's events put upstream
    /// prose under a `message` key at any depth (`response.error.message`
    /// of a failed Responses event, a Mistral error's top-level `message`),
    /// so every such string is cleaned, as the shared translators clean the
    /// events they capture.
    fn for_reception(reception: &Reception) -> Self {
        match reception {
            Reception::Gemini(_) => Self::new(),
            Reception::Responses { .. } | Reception::Mistral { .. } => Self {
                clean: sanitize_upstream_messages,
                ..Self::new()
            },
        }
    }

    fn upstream(&mut self, chunk: &[u8], sanitize: &Sanitizer) {
        self.bytes = self.bytes.saturating_add(chunk.len() as u64);
        self.chunks = self.chunks.saturating_add(1);
        let Ok(events) = self.decoder.push(chunk) else {
            self.events.malformed("decoder", "malformed_sse");
            return;
        };
        for event in events {
            match serde_json::from_str::<Value>(&event.data) {
                Ok(value) => self
                    .events
                    .upstream_event(event.event.as_deref(), &(self.clean)(&value, &**sanitize)),
                Err(_) => self.events.malformed("json", "malformed_event"),
            }
        }
    }

    fn finish(self, traffic: &TrafficCapture, error: Option<&StreamError>) {
        self.finish_with_kind(traffic, error.map(StreamError::kind));
    }

    /// Writes the summary; `error` is the kind of the failure, if any.
    fn finish_with_kind(mut self, traffic: &TrafficCapture, error: Option<&str>) {
        if let Some(kind) = error {
            self.events.malformed("translation", kind);
        }
        self.events.finish_named(
            traffic,
            json!({
                "kind": if error.is_none() { "buffered_completion" } else { "buffered_error" },
                "bytes": self.bytes,
                "chunks": self.chunks,
            }),
            BUFFERED_SUMMARY_NAME,
        );
    }
}

/// The non-streaming answer for a stream that failed: the error type the
/// streaming error event carries, with a matching status. An in-stream
/// upstream error with a 4xx code keeps that code.
fn stream_error(error: &StreamError, sanitize: &Sanitizer) -> ProviderError {
    let (status, kind) = match error.error_type() {
        "rate_limit_error" => (StatusCode::TOO_MANY_REQUESTS, ProviderErrorKind::RateLimit),
        "authentication_error" => (StatusCode::UNAUTHORIZED, ProviderErrorKind::Authentication),
        "permission_error" => (StatusCode::FORBIDDEN, ProviderErrorKind::Permission),
        "invalid_request_error" => {
            let status = match error {
                StreamError::Upstream {
                    code: Some(code @ 400..=499),
                    ..
                } => StatusCode::from_u16(*code).unwrap_or(StatusCode::BAD_REQUEST),
                _ => StatusCode::BAD_REQUEST,
            };
            (status, ProviderErrorKind::InvalidRequest)
        }
        _ => (StatusCode::BAD_GATEWAY, ProviderErrorKind::Api),
    };
    ProviderError::new(status, kind, error.message_with(&**sanitize))
}

/// Validates every setting a request reads: project, location, base URL
/// and location overrides. It runs before any credential lookup, so a
/// missing or malformed setting is reported by its name rather than as a
/// credential failure.
fn check_settings(config: &ResolvedVertexConfig) -> Result<(), ProviderError> {
    VertexTarget::from_config(config).map_err(VertexError::into_provider_error)?;
    config
        .location_overrides()
        .map_err(|error| VertexError::configuration(error).into_provider_error())?;
    Ok(())
}

/// Refuses a model whose family this version does not send
/// ([`models::Family::is_served`]). The 400 comes before any setting or
/// credential is read.
fn require_served(requested: &str, model: &ResolvedModel) -> Result<(), ProviderError> {
    if model.family.is_served() {
        Ok(())
    } else {
        Err(family_not_supported(requested, model))
    }
}

fn family_not_supported(requested: &str, model: &ResolvedModel) -> ProviderError {
    invalid_request(ModelError::FamilyNotSupported {
        requested: requested.to_string(),
        family: model.family.label(),
    })
}

fn invalid_request(error: impl std::fmt::Display) -> ProviderError {
    ProviderError::new(
        StatusCode::BAD_REQUEST,
        ProviderErrorKind::InvalidRequest,
        error.to_string(),
    )
}

fn error_response(error: ProviderError) -> Response {
    let response = json_error(error.status, error.error_type(), error.message);
    match error.retry_after {
        Some(retry_after) => ([(http::header::RETRY_AFTER, retry_after)], response).into_response(),
        None => response,
    }
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::stream;

    use super::translate::stream::POST_TERMINAL_READ_LIMIT;
    use super::*;

    /// Shaped like a Google OAuth access token.
    const SYNTHETIC_TOKEN: &str =
        "ya29.synthetic-access-token-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ";

    fn translator_config() -> TranslatorConfig {
        TranslatorConfig {
            message_id: "msg_synthetic".to_string(),
            model: "vertex/gemini-3.5-flash".to_string(),
            signature_model: "gemini-3.5-flash".to_string(),
            estimated_input_tokens: 7,
            emit_thinking: false,
        }
    }

    fn token_sanitizer() -> Sanitizer {
        Arc::new(|text: &str| sanitize_vertex_text(text, &[SYNTHETIC_TOKEN]))
    }

    fn frame(value: Value) -> Result<Bytes, VertexError> {
        Ok(Bytes::from(format!("data: {value}\n\n")))
    }

    fn text_chunk(text: &str) -> Result<Bytes, VertexError> {
        frame(json!({"candidates":[{"content":{"role":"model","parts":[{"text":text}]}}]}))
    }

    fn stop_chunk() -> Result<Bytes, VertexError> {
        frame(json!({
            "candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":3,"thoughtsTokenCount":2}
        }))
    }

    #[test]
    fn construction_resolves_nothing() {
        let provider = VertexProvider::new();
        assert!(provider.ready.get().is_none());
        assert_eq!(provider.name(), "vertex");
        let models = provider.supported_models();
        assert_eq!(models.len(), 47, "{models:?}");
        assert!(
            models
                .iter()
                .any(|model| model == "vertex/gemini-3.5-flash")
        );
    }

    #[tokio::test]
    async fn read_message_synthetic_completes_when_the_upstream_closes() {
        let upstream = stream::iter([text_chunk("Hello"), stop_chunk()]);
        let message = read_message(upstream, translator_config(), &token_sanitizer(), None)
            .await
            .expect("a closed stream after STOP is a complete message");
        assert_eq!(message["model"], "vertex/gemini-3.5-flash");
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(message["content"][0]["type"], "text");
        assert_eq!(message["content"][0]["text"], "Hello!");
        assert_eq!(message["usage"]["input_tokens"], 11);
        assert_eq!(message["usage"]["output_tokens"], 5);
    }

    #[tokio::test(start_paused = true)]
    async fn read_message_synthetic_upstream_left_open_after_stop_completes_after_the_read_limit() {
        let upstream = stream::iter([text_chunk("Hello"), stop_chunk()]).chain(stream::pending());
        let started = tokio::time::Instant::now();
        let message = tokio::time::timeout(
            Duration::from_secs(60),
            read_message(upstream, translator_config(), &token_sanitizer(), None),
        )
        .await
        .expect("an upstream that stays open must not hold the response")
        .expect("STOP followed by silence is a complete message");
        assert_eq!(started.elapsed(), POST_TERMINAL_READ_LIMIT);
        assert_eq!(message["stop_reason"], "end_turn");
        assert_eq!(message["content"][0]["text"], "Hello!");
    }

    #[tokio::test(start_paused = true)]
    async fn read_message_synthetic_upstream_silent_before_any_finish_reason_is_not_cut() {
        // Without a finishReason there is no deadline: a model may think for
        // a long time before it sends anything.
        let upstream = stream::iter([text_chunk("Hello")]).chain(stream::pending());
        let pending = tokio::time::timeout(
            Duration::from_secs(600),
            read_message(upstream, translator_config(), &token_sanitizer(), None),
        )
        .await;
        assert!(pending.is_err(), "the read ended without a finishReason");
    }

    #[tokio::test]
    async fn read_message_synthetic_close_without_finish_reason_is_incomplete() {
        let upstream = stream::iter([text_chunk("Hello")]);
        let error = read_message(upstream, translator_config(), &token_sanitizer(), None)
            .await
            .expect_err("a stream without finishReason is not a message");
        assert_eq!(error, StreamError::Incomplete);
    }

    #[tokio::test]
    async fn read_message_synthetic_broken_connection_is_a_transport_error() {
        let upstream = stream::iter([
            text_chunk("Hello"),
            Err(VertexError::stream_failed("Vertex AI stream broke off")),
        ]);
        let error = read_message(upstream, translator_config(), &token_sanitizer(), None)
            .await
            .expect_err("a broken connection is not a message");
        assert_eq!(error, StreamError::Transport);
    }

    #[tokio::test]
    async fn read_message_synthetic_body_over_the_cap_is_rejected() {
        // Each chunk is one complete SSE comment frame: the decoder discards
        // it and no frame comes near its own limit, so only the byte cap can
        // end the read.
        let mut filler = vec![b' '; 1024 * 1024];
        filler[0] = b':';
        filler[1024 * 1024 - 2] = b'\n';
        filler[1024 * 1024 - 1] = b'\n';
        let filler = Bytes::from(filler);
        let chunks = (0..=MAX_BUFFERED_RESPONSE_BYTES / filler.len())
            .map(move |_| Ok::<Bytes, VertexError>(filler.clone()));
        let error = read_message(
            stream::iter(chunks),
            translator_config(),
            &token_sanitizer(),
            None,
        )
        .await
        .expect_err("more than the cap is not a message");
        assert_eq!(error, StreamError::Malformed("the response exceeds 16 MiB"));
    }

    #[tokio::test]
    async fn buffered_capture_synthetic_keeps_echoed_secrets_out_of_the_files() {
        let echoed = format!("blocked because the request carried {SYNTHETIC_TOKEN}");
        let upstream = stream::iter([
            text_chunk("Hello"),
            frame(json!({
                "candidates":[{"content":{"role":"model","parts":[]},"finishReason":"SAFETY","finishMessage":echoed}]
            })),
        ]);
        let root = tempfile::tempdir().expect("temporary capture directory");
        let traffic = crate::traffic::test_capture(root.path().to_path_buf());
        let sanitize = token_sanitizer();
        let mut capture = BufferedCapture::new();
        let error = read_message(upstream, translator_config(), &sanitize, Some(&mut capture))
            .await
            .expect_err("SAFETY is not a successful finish");
        capture.finish(&traffic, Some(&error));

        let mapped = stream_error(&error, &sanitize);
        assert_eq!(mapped.status, StatusCode::BAD_GATEWAY);
        assert_eq!(mapped.error_type(), "api_error");
        assert!(
            mapped.message.contains("finishReason SAFETY"),
            "{}",
            mapped.message
        );
        assert!(mapped.message.contains("[redacted]"), "{}", mapped.message);
        assert!(
            !mapped.message.contains(SYNTHETIC_TOKEN),
            "{}",
            mapped.message
        );

        let mut files = Vec::new();
        let mut pending = vec![root.path().to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).expect("capture directory is readable") {
                let path = entry.expect("capture entry").path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    files.push(path);
                }
            }
        }
        let names: Vec<String> = files
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_string))
            .collect();
        assert!(
            names
                .iter()
                .any(|name| name.ends_with(&format!("{BUFFERED_SUMMARY_NAME}.json"))),
            "{names:?}"
        );
        assert!(
            names
                .iter()
                .any(|name| name.ends_with("032-upstream-response-body.sse")),
            "{names:?}"
        );
        for path in files {
            let text = std::fs::read_to_string(&path).expect("capture file is text");
            assert!(
                !text.contains(SYNTHETIC_TOKEN),
                "{} leaked the token: {text}",
                path.display()
            );
        }
    }

    #[test]
    fn stream_errors_synthetic_keep_the_streaming_error_type_and_get_a_matching_status() {
        let sanitize = token_sanitizer();
        let cases = [
            (
                StreamError::Upstream {
                    code: Some(429),
                    status: "RESOURCE_EXHAUSTED".to_string(),
                    upstream_message: Some("quota".to_string()),
                },
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
            (
                StreamError::Upstream {
                    code: None,
                    status: "UNAUTHENTICATED".to_string(),
                    upstream_message: None,
                },
                StatusCode::UNAUTHORIZED,
                "authentication_error",
            ),
            (
                StreamError::Upstream {
                    code: Some(403),
                    status: "PERMISSION_DENIED".to_string(),
                    upstream_message: None,
                },
                StatusCode::FORBIDDEN,
                "permission_error",
            ),
            (
                StreamError::Upstream {
                    code: Some(404),
                    status: "NOT_FOUND".to_string(),
                    upstream_message: None,
                },
                StatusCode::NOT_FOUND,
                "invalid_request_error",
            ),
            (
                StreamError::Upstream {
                    code: None,
                    status: "INVALID_ARGUMENT".to_string(),
                    upstream_message: None,
                },
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                StreamError::PromptBlocked {
                    block_reason: "PROHIBITED_CONTENT".to_string(),
                    upstream_message: None,
                },
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                StreamError::Upstream {
                    code: Some(503),
                    status: "UNAVAILABLE".to_string(),
                    upstream_message: None,
                },
                StatusCode::BAD_GATEWAY,
                "api_error",
            ),
            (
                StreamError::Incomplete,
                StatusCode::BAD_GATEWAY,
                "api_error",
            ),
            (StreamError::Transport, StatusCode::BAD_GATEWAY, "api_error"),
        ];
        for (error, status, error_type) in cases {
            let mapped = stream_error(&error, &sanitize);
            assert_eq!(mapped.status, status, "{error:?}");
            assert_eq!(mapped.error_type(), error_type, "{error:?}");
            assert_eq!(mapped.message, error.message_with(&*sanitize), "{error:?}");
        }
    }

    #[test]
    fn stream_errors_synthetic_run_upstream_prose_through_the_sanitiser() {
        let error = StreamError::Upstream {
            code: Some(400),
            status: "INVALID_ARGUMENT".to_string(),
            upstream_message: Some(format!("token {SYNTHETIC_TOKEN} is not allowed here")),
        };
        let mapped = stream_error(&error, &token_sanitizer());
        assert!(
            !mapped.message.contains(SYNTHETIC_TOKEN),
            "{}",
            mapped.message
        );
        assert!(
            mapped
                .message
                .contains("token [redacted] is not allowed here"),
            "{}",
            mapped.message
        );
    }

    #[tokio::test]
    async fn count_tokens_synthetic_rejects_other_publishers_and_needs_no_settings() {
        let provider = VertexProvider::new();
        let context = RequestContext {
            req_id: "req_synthetic".to_string(),
            session_id: None,
            session_seq: None,
            provider: "vertex".to_string(),
            traffic: None,
            monitor: None,
        };
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "vertex/openai/gpt-oss-120b-maas",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .expect("valid request");
        let response = provider.handle_count_tokens(body, context.clone()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .expect("valid request");
        let response = provider.handle_count_tokens(body, context).await;
        assert_eq!(response.status(), StatusCode::OK);
        // Counting never builds the request state.
        assert!(provider.ready.get().is_none());
    }
    #[tokio::test]
    async fn partner_families_synthetic_get_a_400_saying_this_version_does_not_support_them() {
        let provider = VertexProvider::new();
        let context = RequestContext {
            req_id: "req_synthetic".to_string(),
            session_id: None,
            session_seq: None,
            provider: "vertex".to_string(),
            traffic: None,
            monitor: None,
        };
        for (model, family) in [
            (
                "vertex/openai/gpt-oss-120b-maas",
                "Model Garden managed open model (<publisher>/<model-id>-maas)",
            ),
            (
                "vertex/endpoint/synthetic",
                "deployed endpoint (endpoint/<name>)",
            ),
        ] {
            for stream in [false, true] {
                let body: MessagesRequest = serde_json::from_value(json!({
                    "model": model,
                    "stream": stream,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hello"}]
                }))
                .expect("valid request");
                let response = provider.handle_messages(body, context.clone()).await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{model}");
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["type"], "invalid_request_error", "{value}");
                let message = value["error"]["message"].as_str().unwrap();
                assert_eq!(
                    message,
                    format!(
                        "this version of claude-code-proxy does not support the {family} model family on Vertex AI (requested {model:?}); supported families: Gemini, Anthropic Claude, xAI Grok, OpenAI, Meta Muse, Mistral"
                    ),
                    "{model}"
                );
            }
        }
        // Refused before any setting, credential or client is built.
        assert!(provider.ready.get().is_none());
    }
    #[test]
    fn check_settings_synthetic_rejects_a_malformed_override_by_its_setting_name() {
        let config_dir = tempfile::TempDir::new().unwrap();
        let config = |overrides: &str| {
            let env = std::collections::HashMap::from([
                (
                    "CCP_VERTEX_PROJECT".to_string(),
                    "synthetic-project".to_string(),
                ),
                (
                    "CCP_VERTEX_LOCATION_OVERRIDES".to_string(),
                    overrides.to_string(),
                ),
            ]);
            crate::config::resolve_vertex_config(&env, config_dir.path())
        };
        assert!(check_settings(&config("mistralai=europe-west4")).is_ok());
        for (overrides, reason) in [
            (
                "mistralai=EUROPE",
                "the location of entry \"mistralai\" must be a Vertex AI location",
            ),
            ("mistralai", "entry \"mistralai\" has no location"),
            ("endpoint=us-central1", "entry 1: a deployed endpoint"),
        ] {
            let error = check_settings(&config(overrides)).unwrap_err();
            assert_eq!(
                error.status,
                StatusCode::INTERNAL_SERVER_ERROR,
                "{overrides}"
            );
            assert_eq!(error.error_type(), "api_error", "{overrides}");
            let prefix = format!(
                "Invalid Vertex configuration: Invalid Vertex setting CCP_VERTEX_LOCATION_OVERRIDES: {reason}"
            );
            assert!(error.message.starts_with(&prefix), "{}", error.message);
        }
        // The project is still checked first.
        let env = std::collections::HashMap::from([(
            "CCP_VERTEX_LOCATION_OVERRIDES".to_string(),
            "mistralai=EUROPE".to_string(),
        )]);
        let error = check_settings(&crate::config::resolve_vertex_config(
            &env,
            config_dir.path(),
        ))
        .unwrap_err();
        assert!(
            error.message.contains("Vertex project is not set"),
            "{}",
            error.message
        );
    }

    fn synthetic_context() -> RequestContext {
        RequestContext {
            req_id: "req_synthetic".to_string(),
            session_id: None,
            session_seq: None,
            provider: "vertex".to_string(),
            traffic: None,
            monitor: None,
        }
    }

    #[tokio::test]
    async fn responses_families_synthetic_count_tokens_locally_without_settings() {
        let provider = VertexProvider::new();
        for model in [
            "vertex/xai/grok-4.7",
            "vertex/openai/gpt-6.1-sol",
            "vertex/meta/muse-spark-1.3",
        ] {
            let body: MessagesRequest = serde_json::from_value(json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello world"}]
            }))
            .expect("valid request");
            let response = provider
                .handle_count_tokens(body, synthetic_context())
                .await;
            assert_eq!(response.status(), StatusCode::OK, "{model}");
        }
        assert!(provider.ready.get().is_none());
    }

    #[tokio::test]
    async fn responses_families_synthetic_hosted_web_search_is_a_400_before_any_setting() {
        let provider = VertexProvider::new();
        for model in [
            "vertex/xai/grok-4.7",
            "vertex/openai/gpt-6.1-sol",
            "vertex/meta/muse-spark-1.3",
        ] {
            for stream in [false, true] {
                let body: MessagesRequest = serde_json::from_value(json!({
                    "model": model,
                    "stream": stream,
                    "max_tokens": 16,
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                    "messages": [{"role": "user", "content": "news?"}]
                }))
                .expect("valid request");
                let response = provider.handle_messages(body, synthetic_context()).await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{model}");
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["type"], "invalid_request_error", "{value}");
                let message = value["error"]["message"].as_str().unwrap();
                assert!(
                    message.starts_with("Vertex AI does not serve the hosted web_search tool"),
                    "{message}"
                );
            }
        }
        assert!(provider.ready.get().is_none());
    }

    #[tokio::test]
    async fn report_error_events_synthetic_records_the_first_error_message() {
        let outcome = NativeResponseOutcome::default();
        let chunks = [
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Vertex AI Responses stream is invalid\"}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"second\"}}\n\n",
        ]
        .map(|chunk| Ok::<Bytes, std::io::Error>(Bytes::from_static(chunk.as_bytes())));
        let body = report_error_events(
            Body::from_stream(stream::iter(chunks)),
            Some(outcome.clone()),
        );
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("message_start"));
        assert_eq!(
            outcome.failure().as_deref(),
            Some("Vertex AI Responses stream is invalid")
        );

        let outcome = NativeResponseOutcome::default();
        let body = report_error_events(
            Body::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            Some(outcome.clone()),
        );
        axum::body::to_bytes(body, usize::MAX).await.unwrap();
        assert_eq!(outcome.failure(), None);
    }

    #[tokio::test]
    async fn mistral_synthetic_outside_its_locations_is_a_400_naming_the_settings() {
        // An unroutable base URL: if the location were not refused first,
        // the request would fail to connect instead of reaching anyone.
        let config_dir = tempfile::TempDir::new().unwrap();
        let env = std::collections::HashMap::from([
            (
                "CCP_VERTEX_PROJECT".to_string(),
                "synthetic-project".to_string(),
            ),
            (
                "CCP_VERTEX_BASE_URL".to_string(),
                "http://127.0.0.1:9".to_string(),
            ),
        ]);
        let config = crate::config::resolve_vertex_config(&env, config_dir.path());
        let provider = VertexProvider::with_client(
            VertexClient::with_client(client::client_builder().no_proxy().build().unwrap()),
            VertexCredentials::fixed(http::HeaderMap::new()),
            config,
        );
        for stream in [false, true] {
            let body: MessagesRequest = serde_json::from_value(json!({
                "model": "vertex/mistralai/mistral-medium-3",
                "stream": stream,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .expect("valid request");
            let response = provider.handle_messages(body, synthetic_context()).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "stream {stream}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error", "{value}");
            assert_eq!(
                value["error"]["message"],
                "Vertex AI models of publisher mistralai are not served in location global: set \
                 CCP_VERTEX_LOCATION_OVERRIDES=mistralai=<location>, \"mistralai\": \"<location>\" \
                 in vertex.locationOverrides in config.json, or the location itself \
                 (CCP_VERTEX_LOCATION, GOOGLE_CLOUD_LOCATION or vertex.location), with \
                 us-central1 or europe-west4 as <location>",
                "{value}"
            );
        }
    }

    #[tokio::test]
    async fn mistral_synthetic_counts_tokens_locally_without_settings() {
        let provider = VertexProvider::new();
        let body: MessagesRequest = serde_json::from_value(json!({
            "model": "vertex/mistralai/codestral-2",
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .expect("valid request");
        let response = provider
            .handle_count_tokens(body, synthetic_context())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(provider.ready.get().is_none());
    }

    /// Every file under `root`, with its text.
    fn synthetic_files_under(root: &std::path::Path) -> Vec<(std::path::PathBuf, String)> {
        let mut files = Vec::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).expect("capture directory is readable") {
                let path = entry.expect("capture entry").path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    let text = std::fs::read_to_string(&path).expect("capture file is text");
                    files.push((path, text));
                }
            }
        }
        files
    }

    /// A provider whose requests go to `base_url` with `SYNTHETIC_TOKEN`
    /// as the bearer token.
    fn synthetic_local_provider(base_url: &str, location: &str) -> VertexProvider {
        use crate::config::{VertexSetting, VertexSettingSource};
        let setting = |value: &str, source| VertexSetting {
            value: value.to_string(),
            source,
        };
        let config = ResolvedVertexConfig {
            project_setting: Some(setting(
                "synthetic-project",
                VertexSettingSource::Env("CCP_VERTEX_PROJECT"),
            )),
            location_setting: setting(location, VertexSettingSource::Env("CCP_VERTEX_LOCATION")),
            base_url_setting: Some(setting(
                base_url,
                VertexSettingSource::Env("CCP_VERTEX_BASE_URL"),
            )),
            location_overrides_setting: None,
        };
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            format!("Bearer {SYNTHETIC_TOKEN}").parse().unwrap(),
        );
        VertexProvider::with_client(
            VertexClient::with_client(client::client_builder().no_proxy().build().unwrap()),
            VertexCredentials::fixed(headers),
            config,
        )
    }

    /// A local server that answers every request with `sse`.
    async fn synthetic_sse_server(sse: String) -> String {
        let app = axum::Router::new().route(
            "/{*path}",
            axum::routing::post(move || {
                let sse = sse.clone();
                async move { ([(http::header::CONTENT_TYPE, "text/event-stream")], sse) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn partner_buffered_capture_synthetic_keeps_nested_echoed_secrets_out_of_every_file() {
        let echoed = format!("the request carried {SYNTHETIC_TOKEN}");
        let cases = [
            (
                "Responses failure with the error under response",
                "vertex/openai/gpt-6.1-sol",
                "global",
                format!(
                    "data: {}\n\n",
                    json!({
                        "type":"response.failed",
                        "response":{"status":"failed","error":{"code":"server_error","message":echoed}}
                    })
                ),
            ),
            (
                "Mistral error with a top-level message",
                "vertex/mistralai/mistral-medium-3",
                "us-central1",
                format!(
                    "data: {}\n\n",
                    json!({"object":"error","type":"invalid_request_error","message":echoed})
                ),
            ),
        ];
        for (case, model, location, sse) in cases {
            let base = synthetic_sse_server(sse).await;
            let provider = synthetic_local_provider(&base, location);
            let root = tempfile::tempdir().expect("temporary capture directory");
            let mut ctx = synthetic_context();
            ctx.traffic = Some(Arc::new(crate::traffic::test_capture(
                root.path().to_path_buf(),
            )));
            let body: MessagesRequest = serde_json::from_value(json!({
                "model": model,
                "max_tokens": 64,
                "stream": false,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap();
            let response = provider.handle_messages(body, ctx).await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{case}");
            let answer = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let answer = String::from_utf8_lossy(&answer);
            assert!(!answer.contains(SYNTHETIC_TOKEN), "{case}: {answer}");

            let files = synthetic_files_under(root.path());
            let upstream_events: Vec<_> = files
                .iter()
                .filter(|(path, _)| path.to_string_lossy().contains("-040-upstream-event"))
                .collect();
            assert!(!upstream_events.is_empty(), "{case}: {files:?}");
            assert!(
                upstream_events
                    .iter()
                    .any(|(_, text)| text.contains("the request carried [redacted]")),
                "{case}: {upstream_events:?}"
            );
            assert!(
                files.iter().any(|(path, _)| path
                    .to_string_lossy()
                    .ends_with(&format!("{BUFFERED_SUMMARY_NAME}.json"))),
                "{case}: {files:?}"
            );
            for (path, text) in &files {
                assert!(
                    !text.contains(SYNTHETIC_TOKEN),
                    "{case}: {} leaked the token: {text}",
                    path.display()
                );
            }
        }
    }
}
