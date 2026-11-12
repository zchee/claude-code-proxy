//! Google Cloud Vertex AI provider: Anthropic Messages requests for Gemini
//! models, sent as `streamGenerateContent` with Application Default
//! Credentials.
//!
//! The upstream is always asked for a stream. A non-streaming client gets a
//! message built from the same translation state, so both modes decide the
//! stop reason and content the same way.

pub mod auth;
pub mod cli;
pub mod client;
pub mod models;
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
use crate::traffic::{StreamTrafficCapture, TrafficCapture};

use self::auth::{VertexAuthError, VertexCredentials};
use self::cli::VERTEX_CLI;
use self::client::{VertexClient, VertexError, VertexResponse, VertexTarget};
use self::sanitize::sanitize_vertex_text;
use self::translate::request::prepare_request;
use self::translate::stream::{
    LiveStreamTranslator, MAX_SSE_FRAME_BYTES, POST_TERMINAL_READ_LIMIT, Sanitizer, StreamError,
    StreamSinks, TranslatorConfig, sanitize_capture, stream_body,
};

/// Largest upstream body read for a non-streaming response. The body is
/// translated as it arrives rather than kept, but an upstream that never
/// stops sending must not grow the message without bound.
pub const MAX_BUFFERED_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Traffic-capture summary of a non-streaming response. The streaming one is
/// `061-vertex-stream-summary`; the name says which path produced the files.
const BUFFERED_SUMMARY_NAME: &str = "061-vertex-buffered-summary";

/// What every request needs once the settings, credentials and HTTP client
/// exist.
struct Ready {
    config: ResolvedVertexConfig,
    credentials: VertexCredentials,
    client: VertexClient,
}

/// One request that Vertex AI accepted: the response body is still unread.
struct Started {
    upstream: VertexResponse,
    config: TranslatorConfig,
    sanitize: Sanitizer,
    model_id: String,
}

/// The `vertex` provider.
pub struct VertexProvider {
    ready: OnceLock<Arc<Ready>>,
}

impl VertexProvider {
    /// The server's provider. Settings, credentials and the HTTP client are
    /// created on the first request rather than here: building the client
    /// loads the system root certificates, loading credentials reads files
    /// and needs the async runtime, and every CLI command and server start
    /// constructs every provider.
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
                credentials,
                client,
            })),
        }
    }

    /// The shared request state, built on first use. A failure is returned
    /// and not kept, so the next request tries again.
    fn ready(&self) -> Result<&Arc<Ready>, ProviderError> {
        if let Some(ready) = self.ready.get() {
            return Ok(ready);
        }
        let config = crate::config::vertex_config();
        // Checked first so a missing project is reported by its setting
        // names before any credential file is read.
        VertexTarget::from_config(&config).map_err(VertexError::into_provider_error)?;
        let credentials = VertexCredentials::adc().map_err(VertexAuthError::into_provider_error)?;
        let client = VertexClient::new().map_err(|error| {
            ProviderError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ProviderErrorKind::Api,
                format!("Vertex AI transport is unavailable: {error}"),
            )
        })?;
        let built = Arc::new(Ready {
            config,
            credentials,
            client,
        });
        // Concurrent first requests may each build one; all use the first.
        Ok(self.ready.get_or_init(|| built))
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
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model.id);
        }
        let translated = prepare_request(body, &model).map_err(invalid_request)?;
        let ready = self.ready()?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let upstream = ready
            .client
            .stream_generate_content(
                &ready.config,
                &model.id,
                &translated,
                &ready.credentials,
                ctx.traffic.as_deref(),
            )
            .await
            .map_err(VertexError::into_provider_error)?;
        // Copied before the body is consumed: upstream prose inside the
        // stream may echo the token, and the sanitiser must know it.
        let secrets: Vec<String> = upstream.access_token.iter().cloned().collect();
        let sanitize: Sanitizer = Arc::new(move |text: &str| {
            let secrets: Vec<&str> = secrets.iter().map(String::as_str).collect();
            sanitize_vertex_text(text, &secrets)
        });
        let config = TranslatorConfig {
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            model: requested,
            signature_model: model.id.clone(),
            estimated_input_tokens: count_tokens::count_tokens(body),
            emit_thinking: translated.includes_thoughts(),
        };
        Ok(Started {
            upstream,
            config,
            sanitize,
            model_id: model.id,
        })
    }
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
        let body = stream_body(
            started.upstream.into_byte_stream(),
            started.config,
            StreamSinks {
                req_id: ctx.req_id.clone(),
                monitor: ctx.monitor.clone(),
                traffic: ctx.traffic.clone(),
                outcome: Some(outcome.clone()),
                sanitize: started.sanitize,
            },
        );
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
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &model.id);
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
        body.stream = true;
        let started = self.start(&body, &ctx).await?;
        let body = stream_body(
            started.upstream.into_byte_stream(),
            started.config,
            StreamSinks {
                req_id: ctx.req_id.clone(),
                monitor: ctx.monitor.clone(),
                traffic: ctx.traffic.clone(),
                outcome: None,
                sanitize: started.sanitize,
            },
        );
        Ok(Generation {
            body: GenerationBody::LiveSse(body),
            resolved_model: started.model_id,
        })
    }
}

/// Reads the whole upstream response and answers with one Anthropic message,
/// or with the error the stream would have ended in.
async fn buffered_response(started: Started, ctx: &RequestContext) -> Response {
    let Started {
        upstream,
        config,
        sanitize,
        ..
    } = started;
    let mut capture = ctx.traffic.as_ref().map(|_| BufferedCapture::new());
    let result = read_message(
        upstream.into_byte_stream(),
        config,
        &sanitize,
        capture.as_mut(),
    )
    .await;
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
        Err(error) => {
            let mapped = stream_error(&error, &sanitize);
            if let (Some(traffic), Some(capture)) = (ctx.traffic.as_ref(), capture) {
                traffic.write_json(
                    "051-downstream-response",
                    &json!({
                        "type": "error",
                        "error": {"type": mapped.error_type(), "message": &mapped.message},
                    }),
                );
                capture.finish(traffic, Some(&error));
            }
            error_response(mapped)
        }
    }
}

/// Feeds the upstream body to one translator chunk by chunk.
///
/// The response is complete when the upstream closes, or
/// [`POST_TERMINAL_READ_LIMIT`] after the `finishReason` when it stays open:
/// the same rule the streaming path applies. More than
/// [`MAX_BUFFERED_RESPONSE_BYTES`] is an error.
async fn read_message<S>(
    upstream: S,
    config: TranslatorConfig,
    sanitize: &Sanitizer,
    mut capture: Option<&mut BufferedCapture>,
) -> Result<Value, StreamError>
where
    S: Stream<Item = Result<Bytes, VertexError>>,
{
    let mut upstream = std::pin::pin!(upstream);
    let mut translator = LiveStreamTranslator::new(config);
    let mut deadline = None;
    let mut read = 0usize;
    loop {
        let next = if translator.is_terminal() {
            let deadline = *deadline
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
        let chunk = match next {
            Some(Ok(chunk)) => chunk,
            Some(Err(_)) => return Err(StreamError::Transport),
            None => {
                translator.finish()?;
                return translator.response();
            }
        };
        read = read.saturating_add(chunk.len());
        if read > MAX_BUFFERED_RESPONSE_BYTES {
            return Err(StreamError::Malformed("the response exceeds 16 MiB"));
        }
        if let Some(capture) = capture.as_deref_mut() {
            capture.upstream(&chunk, sanitize);
        }
        translator.push(&chunk)?;
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
}

impl BufferedCapture {
    fn new() -> Self {
        Self {
            decoder: SseDecoder::with_max_frame_bytes(MAX_SSE_FRAME_BYTES),
            events: StreamTrafficCapture::default(),
            bytes: 0,
            chunks: 0,
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
                Ok(value) => self.events.upstream_event(
                    event.event.as_deref(),
                    &sanitize_capture(&value, &**sanitize),
                ),
                Err(_) => self.events.malformed("json", "malformed_event"),
            }
        }
    }

    fn finish(mut self, traffic: &TrafficCapture, error: Option<&StreamError>) {
        if let Some(error) = error {
            self.events.malformed("translation", error.kind());
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
        assert_eq!(models.len(), 18, "{models:?}");
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
            "model": "vertex/anthropic/claude-sonnet-5",
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
}
