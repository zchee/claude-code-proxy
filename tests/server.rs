#[path = "support/isolated_codex.rs"]
mod isolated_codex;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::response::IntoResponse;
use claude_code_proxy::{
    MessagesRequest,
    anthropic::MAX_ANTHROPIC_REQUEST_BYTES,
    config::AliasProvider,
    monitor::{MonitorHandle, RequestStatus},
    provider::{CliHandlers, Generation, GenerationBody, Provider, ProviderError, RequestContext},
    registry::Registry,
    request_identity::ConversationIdentity,
    server::{
        AppFeatures, app, app_with_features, app_with_monitor, app_with_options,
        bind_proxy_listener,
    },
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tower::util::ServiceExt;

fn body_string(json: &str) -> Body {
    Body::from(json.to_string())
}

struct FakeCli;

impl CliHandlers for FakeCli {
    fn login(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn device(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn status(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn logout(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

static FAKE_CLI: FakeCli = FakeCli;

struct FakeProvider {
    name: &'static str,
    models: Vec<String>,
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supported_models(&self) -> Vec<String> {
        self.models.clone()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &FAKE_CLI
    }

    async fn handle_messages(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        (StatusCode::NOT_IMPLEMENTED, "unused").into_response()
    }

    async fn handle_count_tokens(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        (StatusCode::NOT_IMPLEMENTED, "unused").into_response()
    }

    async fn generate_anthropic_stream(
        &self,
        body: MessagesRequest,
        _ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        let model = body.model.unwrap_or_default();
        let sse = format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_fake\",\"model\":{model:?},\"usage\":{{\"input_tokens\":2}}}}}}\n\nevent: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\nevent: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{name}\"}}}}\n\nevent: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\nevent: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":1}}}}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
            name = self.name,
        );
        Ok(Generation {
            body: GenerationBody::BufferedSse(sse.into()),
            resolved_model: model,
        })
    }
}

struct TranslatingProvider {
    name: &'static str,
    model: &'static str,
    captured: Arc<Mutex<Option<Value>>>,
}

#[async_trait]
impl Provider for TranslatingProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.model.to_string()]
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &FAKE_CLI
    }

    async fn handle_messages(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        (StatusCode::NOT_IMPLEMENTED, "unused").into_response()
    }

    async fn handle_count_tokens(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        (StatusCode::NOT_IMPLEMENTED, "unused").into_response()
    }

    async fn generate_anthropic_stream(
        &self,
        body: MessagesRequest,
        _ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        let translated = match self.name {
            "kimi" => serde_json::to_value(
                claude_code_proxy::providers::kimi::translate::request::translate_request(
                    &body,
                    claude_code_proxy::providers::kimi::translate::request::TranslateOptions {
                        session_id: None,
                    },
                )
                .unwrap(),
            )
            .unwrap(),
            "grok" => serde_json::to_value(
                claude_code_proxy::providers::grok::translate::request::translate_request(
                    &body,
                    self.model.to_string(),
                )
                .unwrap(),
            )
            .unwrap(),
            _ => unreachable!(),
        };
        *self.captured.lock().unwrap() = Some(translated);
        let sse = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fake\",\"model\":\"test\",\"usage\":{\"input_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        Ok(Generation {
            body: GenerationBody::BufferedSse(sse.into()),
            resolved_model: self.model.to_string(),
        })
    }
}

fn translating_registry(
    name: &'static str,
    model: &'static str,
    captured: Arc<Mutex<Option<Value>>>,
) -> Arc<Registry> {
    Arc::new(Registry::from_providers(
        AliasProvider::Kimi,
        vec![Arc::new(TranslatingProvider {
            name,
            model,
            captured,
        }) as Arc<dyn Provider>],
    ))
}

type CapturedIdentity = (Option<ConversationIdentity>, Option<String>);

struct IdentityCaptureProvider {
    captured: Arc<Mutex<Vec<CapturedIdentity>>>,
}

struct RequestCaptureProvider {
    captured: Arc<Mutex<Vec<MessagesRequest>>>,
}

#[async_trait]
impl Provider for RequestCaptureProvider {
    fn name(&self) -> &'static str {
        "codex"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["gpt-5.5".to_string()]
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &FAKE_CLI
    }

    async fn handle_messages(
        &self,
        body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        self.captured.lock().unwrap().push(body);
        (StatusCode::OK, "captured").into_response()
    }

    async fn handle_count_tokens(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
    ) -> axum::response::Response {
        self.handle_messages(body, ctx).await
    }
}

#[async_trait]
impl Provider for IdentityCaptureProvider {
    fn name(&self) -> &'static str {
        "codex"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["gpt-5.5".to_string(), "gpt-6-luna".to_string()]
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &FAKE_CLI
    }

    async fn handle_messages(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, "legacy path").into_response()
    }

    async fn handle_messages_with_conversation_identity(
        &self,
        _body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> axum::response::Response {
        self.captured
            .lock()
            .unwrap()
            .push((conversation_identity, ctx.session_id));
        (StatusCode::OK, "captured").into_response()
    }

    async fn handle_count_tokens(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> axum::response::Response {
        (StatusCode::OK, "counted").into_response()
    }
}

fn routed_registry() -> Arc<Registry> {
    Arc::new(Registry::from_providers(
        AliasProvider::Kimi,
        vec![
            Arc::new(FakeProvider {
                name: "kimi",
                models: vec!["kimi-k2.6".to_string()],
            }) as Arc<dyn Provider>,
            Arc::new(FakeProvider {
                name: "grok",
                models: vec!["grok-4.5".to_string()],
            }),
            Arc::new(FakeProvider {
                name: "cursor",
                models: vec!["cursor".to_string()],
            }),
        ],
    ))
}

async fn call_identity_ingress(
    app: &axum::Router,
    path: &str,
    headers: &[(&str, &str)],
    body: Value,
) -> StatusCode {
    let mut request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    app.clone()
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap()
        .status()
}

#[tokio::test]
async fn messages_ingress_forwards_only_strict_conversation_identity() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(IdentityCaptureProvider {
        captured: captured.clone(),
    }) as Arc<dyn Provider>;
    let app = app(Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [provider],
    )));
    let normal_body = || {
        json!({
            "model": "gpt-5.5",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}]
        })
    };

    let cases = [
        (
            vec![("x-claude-code-session-id", "session-main")],
            normal_body(),
        ),
        (
            vec![
                ("x-claude-code-session-id", "session-agent"),
                ("x-claude-code-agent-id", "agent-direct"),
            ],
            normal_body(),
        ),
        (
            vec![
                ("x-claude-code-session-id", "session-nested"),
                ("x-claude-code-agent-id", "agent-child"),
                ("x-claude-code-parent-agent-id", "agent-parent"),
            ],
            normal_body(),
        ),
        (
            vec![
                ("x-claude-code-session-id", "session-malformed-agent"),
                ("x-claude-code-agent-id", "malformed agent"),
            ],
            normal_body(),
        ),
        (vec![], normal_body()),
        (
            vec![("x-claude-code-session-id", " \tsession-trimmed\t ")],
            normal_body(),
        ),
        (
            vec![
                ("x-claude-code-session-id", "session-auto-review"),
                ("x-claude-code-agent-id", "agent-auto-review"),
            ],
            json!({
                "model": "gpt-5.5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "review"}],
                "system": [{
                    "type": "text",
                    "text": "You are a security monitor for autonomous AI coding agents. Review this turn."
                }]
            }),
        ),
    ];

    for (headers, body) in cases {
        assert_eq!(
            call_identity_ingress(&app, "/v1/messages", &headers, body).await,
            StatusCode::OK
        );
    }
    assert_eq!(
        call_identity_ingress(
            &app,
            "/v1/messages/count_tokens",
            &[("x-claude-code-session-id", "session-count")],
            normal_body(),
        )
        .await,
        StatusCode::OK
    );

    assert_eq!(
        *captured.lock().unwrap(),
        vec![
            (
                Some(ConversationIdentity::Main("session-main".to_string())),
                Some("session-main".to_string()),
            ),
            (
                Some(ConversationIdentity::Agent(
                    "session-agent".to_string(),
                    "agent-direct".to_string(),
                )),
                Some("session-agent".to_string()),
            ),
            (
                Some(ConversationIdentity::Agent(
                    "session-nested".to_string(),
                    "agent-child".to_string(),
                )),
                Some("session-nested".to_string()),
            ),
            (None, Some("session-malformed-agent".to_string())),
            (None, None),
            (
                Some(ConversationIdentity::Main("session-trimmed".to_string())),
                Some(" \tsession-trimmed\t ".to_string()),
            ),
            (None, Some("session-auto-review".to_string())),
        ]
    );
}

#[tokio::test]
async fn bind_error_names_address_and_port() {
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = occupied.local_addr().unwrap().port();

    let err = bind_proxy_listener("127.0.0.1", port)
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains(&format!("127.0.0.1:{port}")));
    assert!(err.contains("failed to bind proxy listener"));
}

#[tokio::test]
async fn configurable_bind_address_accepts_all_interfaces() {
    let listener = bind_proxy_listener("0.0.0.0", 0).await.unwrap();
    assert_eq!(listener.local_addr().unwrap().ip().to_string(), "0.0.0.0");
}

#[tokio::test]
async fn invalid_bind_address_is_actionable() {
    let err = bind_proxy_listener("not-an-ip", 18765)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid proxy bind address"));
    assert!(err.contains("not-an-ip"));
}

#[tokio::test]
async fn healthz_returns_ok() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    assert_eq!(body, json!({"ok": true}));
}

#[tokio::test]
async fn invalid_json_request_is_json_error() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    let error_type = value["error"]["type"].as_str().unwrap_or("");
    assert_eq!(error_type, "invalid_request_error");
}

async fn message_thread_response(
    body: Value,
) -> (
    axum::response::Response,
    Vec<MessagesRequest>,
    MonitorHandle,
) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let registry = Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [Arc::new(RequestCaptureProvider {
            captured: captured.clone(),
        }) as Arc<dyn Provider>],
    ));
    let monitor = MonitorHandle::new(10);
    let response = app_with_monitor(registry, Some(monitor.clone()))
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-beta", "message-threads-2026-08-12")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let requests = captured.lock().unwrap().clone();
    (response, requests, monitor)
}

async fn assert_message_thread_rejected(body: Value) {
    let (response, captured, monitor) = message_thread_response(body).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(!request_id(&response).is_empty());
    assert_eq!(captured.len(), 0);
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent.len(), 1);
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert_eq!(state.recent[0].http_status, Some(400));
    assert!(
        state.recent[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("capability_rejected: beta_header:message-threads-2026-08-12")
    );
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("capability_rejected: beta_header:message-threads-2026-08-12")
    );
}

async fn assert_message_thread_dispatched(mut body: Value) {
    let (response, captured, _) = message_thread_response(body.clone()).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(captured.len(), 1);
    assert!(!captured[0].extra.contains_key("thread"));
    body.as_object_mut().unwrap().remove("thread");
    let expected: MessagesRequest = serde_json::from_value(body).unwrap();
    assert_eq!(
        serde_json::to_value(&captured[0]).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
}

#[tokio::test]
async fn message_thread_continue_tool_result_is_rejected_before_dispatch() {
    assert_message_thread_rejected(json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_test", "content": "done"
        }]}],
        "thread": {"type": "continue", "previous_message_id": "msg_test"}
    }))
    .await;
}

#[tokio::test]
async fn message_thread_continue_text_is_rejected_before_dispatch() {
    assert_message_thread_rejected(json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "next question"}],
        "thread": {"type": "continue", "previous_message_id": "msg_test"}
    }))
    .await;
}

#[tokio::test]
async fn message_thread_unknown_type_is_rejected_before_dispatch() {
    assert_message_thread_rejected(json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hello"}],
        "thread": {"type": "fork"}
    }))
    .await;
}

#[tokio::test]
async fn message_thread_missing_type_is_rejected_before_dispatch() {
    assert_message_thread_rejected(json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hello"}],
        "thread": {}
    }))
    .await;
}

#[tokio::test]
async fn message_thread_create_tool_result_dispatches_without_thread() {
    assert_message_thread_dispatched(json!({
        "model": "gpt-5.5",
        "max_tokens": 32,
        "system": "Keep the full history.",
        "messages": [
            {"role": "user", "content": "run the tool"},
            {"role": "assistant", "content": [{
                "type": "tool_use", "id": "call_test", "name": "test", "input": {}
            }]},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "call_test", "content": "done"
            }]}
        ],
        "thread": {"type": "create"}
    }))
    .await;
}

#[tokio::test]
async fn message_thread_create_text_dispatches_without_thread() {
    assert_message_thread_dispatched(json!({
        "model": "gpt-5.5",
        "max_tokens": 32,
        "system": "Keep the full history.",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "next question"}
        ],
        "thread": {"type": "create"}
    }))
    .await;
}

#[tokio::test]
async fn message_thread_absent_text_dispatches_unchanged() {
    assert_message_thread_dispatched(json!({
        "model": "gpt-5.5",
        "max_tokens": 32,
        "system": "Keep the full history.",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .await;
}

#[tokio::test]
async fn message_thread_absent_tool_result_dispatches_unchanged() {
    assert_message_thread_dispatched(json!({
        "model": "gpt-5.5",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_test", "content": "done"
        }]}]
    }))
    .await;
}

#[tokio::test]
async fn empty_body_is_invalid_json() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_model_returns_400_with_summary() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"messages":[{"role":"user","content":"hello"}],"model":"not-a-model"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    let message = body["error"]["message"].as_str().unwrap_or("");
    assert!(message.contains("Unknown model \"not-a-model\""));
    assert!(message.contains("Supported:"));
}

fn request_id(response: &axum::response::Response) -> &str {
    response
        .headers()
        .get("request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
}

async fn messages_response(app: axum::Router, model: &str) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(body_string(
                &json!({"model": model, "messages": [{"role": "user", "content": "hello"}]})
                    .to_string(),
            ))
            .unwrap(),
    )
    .await
    .unwrap()
}

// Claude Code populates the `requestId` field of every transcript record from
// the `request-id` response header. Consumers that de-duplicate those records
// by request id count each request twice without it, because a transcript
// legitimately repeats a record and the id is what resolves the repeat.
#[tokio::test]
async fn successful_response_carries_a_request_id() {
    let registry = || {
        Arc::new(Registry::from_providers(
            AliasProvider::Codex,
            [Arc::new(IdentityCaptureProvider {
                captured: Arc::new(Mutex::new(Vec::new())),
            }) as Arc<dyn Provider>],
        ))
    };
    let first = messages_response(app(registry()), "gpt-5.5").await;
    let second = messages_response(app(registry()), "gpt-5.5").await;

    assert_eq!(first.status(), StatusCode::OK);
    assert!(!request_id(&first).is_empty());
    assert_ne!(request_id(&first), request_id(&second));
}

// Errors are de-duplicated by the same key as successes, and a failed turn is
// the record most worth correlating with a proxy log.
#[tokio::test]
async fn rejected_request_carries_a_request_id() {
    let response =
        messages_response(app(Arc::new(Registry::with_default_alias())), "not-a-model").await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(!request_id(&response).is_empty());
}

// A provider that answers with a failure status leaves through a different
// return path than a success.
#[tokio::test]
async fn provider_failure_response_carries_a_request_id() {
    let response = messages_response(app(routed_registry()), "kimi-k2.6").await;

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    assert!(!request_id(&response).is_empty());
}

#[tokio::test]
async fn missing_model_returns_400() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    let error_type = body["error"]["type"].as_str().unwrap_or("");
    assert_eq!(error_type, "invalid_request_error");
}

async fn error_body(response: axum::response::Response) -> Value {
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap()
}

// Builds a valid JSON /v1/messages body of exactly `total_len` bytes that has
// no "model", so the handler parses it and then fails on the missing model.
// That failure proves the body cleared the size gate without touching a
// provider.
fn padded_messages_body_without_model(total_len: usize) -> String {
    let prefix = r#"{"messages":[{"role":"user","content":"hello"}],"padding":""#;
    let suffix = r#""}"#;
    let padding = total_len - prefix.len() - suffix.len();
    let mut body = String::with_capacity(total_len);
    body.push_str(prefix);
    body.extend(std::iter::repeat_n('a', padding));
    body.push_str(suffix);
    assert_eq!(body.len(), total_len);
    body
}

#[tokio::test]
async fn messages_body_over_16mib_clears_size_gate() {
    const OLD_LIMIT: usize = 16 * 1024 * 1024;
    const { assert!(MAX_ANTHROPIC_REQUEST_BYTES > OLD_LIMIT) };
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(padded_messages_body_without_model(
                    OLD_LIMIT + 1,
                )))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = error_body(response).await;
    let message = body["error"]["message"].as_str().unwrap_or("");
    assert!(
        message.starts_with("Missing \"model\""),
        "body over 16 MiB should reach model validation, got: {message}"
    );
}

#[tokio::test]
async fn messages_body_at_limit_clears_size_gate() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(padded_messages_body_without_model(
                    MAX_ANTHROPIC_REQUEST_BYTES,
                )))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = error_body(response).await;
    let message = body["error"]["message"].as_str().unwrap_or("");
    assert!(
        message.starts_with("Missing \"model\""),
        "body at the limit should reach model validation, got: {message}"
    );
}

#[tokio::test]
async fn anthropic_bodies_over_limit_return_request_too_large() {
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let response = app(Arc::new(Registry::with_default_alias()))
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(padded_messages_body_without_model(
                        MAX_ANTHROPIC_REQUEST_BYTES + 1,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "request_too_large");
    }
}

#[tokio::test]
async fn known_model_reaches_codex_provider() {
    if isolated_codex::rerun_without_credentials("known_model_reaches_codex_provider") {
        return;
    }
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // A known model reaches the concrete provider's authentication boundary.
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = error_body(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Not authenticated")
    );
}

#[tokio::test]
async fn count_tokens_routes_to_provider() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Codex provider is now concrete, so count_tokens should succeed
    let status = response.status();
    assert!(
        status != StatusCode::NOT_IMPLEMENTED,
        "count_tokens should no longer return 501 for codex models"
    );
}

#[tokio::test]
async fn context_window_hint_is_removed_before_provider_dispatch() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.6-luna[1m]","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn opus_5_alias_routes_to_provider() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"claude-opus-5","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn image_routes_reject_variations_wrong_media_and_oversized_generation() {
    let features = AppFeatures {
        responses_api: false,
        images_api: true,
        transcriptions_api: false,
    };
    let variation = app_with_features(Arc::new(Registry::with_default_alias()), None, features)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/variations")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(variation.status(), StatusCode::NOT_FOUND);

    let wrong_media = app_with_features(Arc::new(Registry::with_default_alias()), None, features)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/generations")
                .header("content-type", "multipart/form-data; boundary=x")
                .body(Body::from("--x--\r\n"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wrong_media.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let oversized = app_with_features(Arc::new(Registry::with_default_alias()), None, features)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(vec![b'x'; 256 * 1024 + 1]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn monitor_tracks_image_endpoint_without_session_affinity() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_features(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
        AppFeatures {
            responses_api: false,
            images_api: true,
            transcriptions_api: false,
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();

    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), 1);
    assert_eq!(state.recent[0].endpoint.label(), "images");
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert!(state.recent[0].session_seq.is_none());
    assert!(state.recent[0].traffic_capture_path.is_none());
}

#[tokio::test]
async fn image_edit_accepts_multipart_and_validates_fields() {
    let app = app_with_features(
        Arc::new(Registry::with_default_alias()),
        None,
        AppFeatures {
            responses_api: false,
            images_api: true,
            transcriptions_api: false,
        },
    );
    let boundary = "ccp-image-test";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input.png\"\r\nContent-Type: image/png\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/edits")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["param"], "prompt");
}

#[tokio::test]
async fn image_routes_are_independently_opt_in() {
    let disabled = app_with_features(
        Arc::new(Registry::with_default_alias()),
        None,
        AppFeatures {
            responses_api: false,
            images_api: false,
            transcriptions_api: false,
        },
    );
    let response = disabled
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let enabled = app_with_features(
        Arc::new(Registry::with_default_alias()),
        None,
        AppFeatures {
            responses_api: false,
            images_api: true,
            transcriptions_api: false,
        },
    );
    let response = enabled
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn transcription_route_is_independently_opt_in_and_validates_multipart() {
    let disabled = app_with_features(
        Arc::new(Registry::with_default_alias()),
        None,
        AppFeatures {
            responses_api: false,
            images_api: false,
            transcriptions_api: false,
        },
    );
    let response = disabled
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/audio/transcriptions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let enabled = app_with_features(
        Arc::new(Registry::with_default_alias()),
        None,
        AppFeatures {
            responses_api: false,
            images_api: false,
            transcriptions_api: true,
        },
    );
    let boundary = "ccp-transcription-test";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ngpt-4o-mini-transcribe\r\n--{boundary}--\r\n"
    );
    let response = enabled
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["param"], "file");
}

#[tokio::test]
async fn transcription_route_rejects_non_audio_uploads() {
    let app = app_with_features(
        Arc::new(Registry::with_default_alias()),
        None,
        AppFeatures {
            responses_api: false,
            images_api: false,
            transcriptions_api: true,
        },
    );
    let boundary = "ccp-transcription-type-test";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"notes.txt\"\r\nContent-Type: text/plain\r\n\r\nnot audio\r\n--{boundary}--\r\n"
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn native_responses_route_is_disabled_by_default_option() {
    let app = app_with_options(Arc::new(Registry::with_default_alias()), None, false);
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(body_string(r#"{"model":"gpt-5.4","input":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn enabled_native_responses_route_uses_openai_errors() {
    let app = app_with_options(Arc::new(Registry::with_default_alias()), None, true);
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    assert!(body.get("type").is_none());
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "invalid_json");
}

#[tokio::test]
async fn chat_completions_route_uses_responses_api_gate() {
    let disabled = app_with_options(Arc::new(Registry::with_default_alias()), None, false);
    let response = disabled
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let enabled = app_with_options(Arc::new(Registry::with_default_alias()), None, true);
    let response = enabled
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "invalid_json");
}

#[tokio::test]
async fn openai_routes_select_non_codex_providers_and_aliases() {
    for (uri, request, expected) in [
        (
            "/v1/chat/completions",
            json!({"model":"kimi-k2.6","messages":[{"role":"user","content":"hello"}]}),
            "kimi",
        ),
        (
            "/v1/responses",
            json!({"model":"grok-4.5","input":"hello"}),
            "grok",
        ),
        (
            "/v1/chat/completions",
            json!({"model":"cursor:gpt-5.5","messages":[{"role":"user","content":"hello"}]}),
            "cursor",
        ),
        (
            "/v1/responses",
            json!({"model":"sonnet","input":"hello"}),
            "kimi",
        ),
    ] {
        let response = app_with_options(routed_registry(), None, true)
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(body_string(&request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri} {request}");
        let value: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let text = if uri.ends_with("responses") {
            value["output"][0]["content"][0]["text"].as_str()
        } else {
            value["choices"][0]["message"]["content"].as_str()
        };
        assert_eq!(text, Some(expected));
    }
}

#[tokio::test]
async fn openai_routes_preserve_serial_tool_calls_upstream() {
    for (provider, model, uri, body, expected_choice) in [
        (
            "kimi",
            "kimi-k2.6",
            "/v1/chat/completions",
            json!({
                "model":"kimi-k2.6",
                "messages":[{"role":"user","content":"look up x"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "tool_choice":{"type":"function","function":{"name":"lookup"}},
                "parallel_tool_calls":false
            }),
            json!({"type":"function","function":{"name":"lookup"}}),
        ),
        (
            "grok",
            "grok-4.5",
            "/v1/responses",
            json!({
                "model":"grok-4.5",
                "input":"look up x",
                "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
                "tool_choice":"none",
                "parallel_tool_calls":false
            }),
            json!("none"),
        ),
    ] {
        let captured = Arc::new(Mutex::new(None));
        let response = app_with_options(
            translating_registry(provider, model, captured.clone()),
            None,
            true,
        )
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header("content-type", "application/json")
                .body(body_string(&body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let translated = captured.lock().unwrap().clone().unwrap();
        assert_eq!(translated["parallel_tool_calls"], false);
        assert_eq!(translated["tool_choice"], expected_choice);
    }
}

#[tokio::test]
async fn routed_openai_streams_use_surface_specific_events() {
    let chat = app_with_options(routed_registry(), None, true)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"kimi-k2.6","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hello"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let chat = String::from_utf8(
        axum::body::to_bytes(chat.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(chat.contains("chat.completion.chunk"));
    assert!(chat.contains("\"total_tokens\":3"));
    assert!(chat.ends_with("data: [DONE]\n\n"));

    let responses = app_with_options(routed_registry(), None, true)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"grok-4.5","stream":true,"input":"hello"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let responses = String::from_utf8(
        axum::body::to_bytes(responses.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(responses.contains("event: response.created"));
    assert!(responses.contains("event: response.completed"));
    assert!(responses.contains("\"sequence_number\":0"));
}

#[tokio::test]
async fn non_codex_validation_uses_openai_errors_before_generation() {
    let monitor = MonitorHandle::new(10);
    let response = app_with_options(routed_registry(), Some(monitor.clone()), true)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("x-client-request-id", "invalid-routed-request")
                .body(body_string(
                    r#"{"model":"kimi-k2.6","messages":[{"role":"user","content":"hello"}],"temperature":0.5}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let value: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(value["error"]["param"], "temperature");
    assert_eq!(value["error"]["code"], "unsupported_parameter");
    assert!(
        claude_code_proxy::session::existing_session_now(Some("invalid-routed-request")).is_none()
    );
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.recent[0].session_seq, None);
    assert_eq!(snapshot.recent[0].provider, None);
}

#[tokio::test]
async fn chat_completions_validation_returns_openai_parameter_errors() {
    let app = app_with_options(Arc::new(Registry::with_default_alias()), None, true);
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"model":"gpt-5.6-sol","messages":[{"role":"user","content":"hello"}],"max_tokens":100}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["error"]["param"], "max_tokens");
    assert_eq!(body["error"]["code"], "unsupported_parameter");
}

#[tokio::test]
async fn unknown_routes_use_anthropic_not_found_error() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/nope")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap();
    assert_eq!(body["type"].as_str().unwrap_or(""), "error");
}

#[tokio::test]
async fn monitor_records_successful_request_events() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_monitor(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages/count_tokens")
                .header("content-type", "application/json")
                .header("x-claude-code-session-id", "project-session")
                .body(body_string(
                    r##"{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.177.45c"},{"type":"text","text":"You are a Claude agent, built on Anthropic's Claude Agent SDK.","cache_control":{"type":"ephemeral"}},{"type":"text","text":"\nYou are an interactive agent.\n\n# Environment\nYou have been invoked in the following environment: \n - Primary working directory: /projects/example\n - Is a git repository: true","cache_control":{"type":"ephemeral"}}],"output_config":{"effort":"high"}}"##,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let state = monitor.snapshot();
    assert_eq!(state.active.len(), 1);
    assert!(state.recent.is_empty());

    let _body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent.len(), 1);
    assert_eq!(state.recent[0].status, RequestStatus::Completed);
    assert_eq!(state.recent[0].http_status, Some(200));
    assert_eq!(
        state.recent[0].session_id.as_deref(),
        Some("project-session")
    );
    assert!(state.recent[0].session_seq.is_some());
    assert_eq!(state.recent[0].project.as_deref(), Some("example"));
    assert_eq!(state.sessions[0].project.as_deref(), Some("example"));
    assert_eq!(state.recent[0].provider.as_deref(), Some("codex"));
    assert_eq!(state.recent[0].model.as_deref(), Some("gpt-5.4"));
    assert_eq!(state.recent[0].effort.as_deref(), Some("high"));
    assert!(state.recent[0].input_tokens.is_some());
}

#[tokio::test]
async fn monitor_records_invalid_json_failure() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_monitor(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .body(body_string("{"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert_eq!(state.recent[0].http_status, Some(400));
    let error = state.recent[0].error.as_deref().unwrap_or("");
    assert!(error.starts_with("Invalid JSON:"));
}

#[tokio::test]
async fn monitor_records_unknown_model_failure() {
    let monitor = MonitorHandle::new(10);
    let app = app_with_monitor(
        Arc::new(Registry::with_default_alias()),
        Some(monitor.clone()),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(body_string(
                    r#"{"messages":[{"role":"user","content":"hello"}],"model":"not-a-model"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let state = monitor.snapshot();
    assert!(state.active.is_empty());
    assert_eq!(state.recent[0].status, RequestStatus::Failed);
    assert_eq!(state.recent[0].http_status, Some(400));
    let error = state.recent[0].error.as_deref().unwrap_or("");
    assert!(error.starts_with("Unknown model \"not-a-model\""));
    assert!(error.contains("Supported:"));
}

async fn get_models(app: axum::Router, uri: &str) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    (status, value)
}

#[tokio::test]
async fn models_endpoint_lists_supported_models() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let (status, value) = get_models(app, "/v1/models").await;

    assert_eq!(status, StatusCode::OK);
    let data = value["data"].as_array().unwrap();
    assert!(!data.is_empty());
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"gpt-5.6-sol"));
    assert!(ids.contains(&"gpt-6.1-sol"));
    assert!(ids.contains(&"gpt-6.1-sol-fast"));
    assert!(ids.contains(&"gpt-6.1-terra"));
    assert!(ids.contains(&"gpt-6.1-terra-fast"));
    assert!(ids.contains(&"gpt-6-astra-ultrafast"));
    assert!(ids.contains(&"gpt-6.1-sol-ultrafast"));
    for id in [
        "gpt-6.2-astra",
        "gpt-6.2-astra-fast",
        "gpt-6.2-astra-ultrafast",
        "gpt-6.2-sol",
        "gpt-6.2-sol-fast",
        "gpt-6.2-sol-ultrafast",
    ] {
        assert!(ids.contains(&id), "{id}");
    }
    assert!(!ids.contains(&"gpt-6-sol-ultrafast"));
    assert!(ids.contains(&"grok-4.6"));
    assert!(ids.contains(&"grok-4.7-build-fast"));
    assert!(ids.contains(&"grok-4.7-fast"));
    for entry in data {
        assert_eq!(entry["type"], "model");
        assert!(entry["display_name"].as_str().is_some());
    }
    assert_eq!(value["has_more"], json!(false));
    assert_eq!(value["first_id"], data[0]["id"]);
    assert_eq!(value["last_id"], data[data.len() - 1]["id"]);
}

#[tokio::test]
async fn models_endpoint_includes_claude_prefixed_aliases_for_discovery() {
    // Claude Code's gateway model discovery ignores ids that don't start with
    // "claude" or "anthropic", so the alias entries are what make
    // CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 useful at all.
    let app = app(Arc::new(Registry::with_default_alias()));
    let (status, value) = get_models(app, "/v1/models?limit=1000").await;

    assert_eq!(status, StatusCode::OK);
    let ids: Vec<&str> = value["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.iter().any(|id| id.starts_with("claude-")));
    assert!(ids.contains(&"claude-opus-5"));
}

#[tokio::test]
async fn models_endpoint_respects_limit() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let (status, value) = get_models(app, "/v1/models?limit=2").await;

    assert_eq!(status, StatusCode::OK);
    let data = value["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(value["has_more"], json!(true));
    assert_eq!(value["last_id"], data[1]["id"]);
}

#[tokio::test]
async fn models_endpoint_tolerates_unknown_query_params() {
    let app = app(Arc::new(Registry::with_default_alias()));
    let (status, _) = get_models(app, "/v1/models?limit=1000&after_id=x").await;
    assert_eq!(status, StatusCode::OK);
}

mod muse {
    use super::*;
    use axum::routing::post;
    use claude_code_proxy::providers::muse::MuseProvider;
    use claude_code_proxy::providers::muse::auth::constants::MINT_TIMEOUT;
    use claude_code_proxy::providers::muse::auth::manager::{
        CALLER_DEADLINE, HttpMinter, MuseAuthManager,
    };
    use claude_code_proxy::providers::muse::auth::mint::{self, MintTarget};
    use claude_code_proxy::providers::muse::auth::token_store::{StoredAuth, store_at};
    use claude_code_proxy::providers::muse::client::{MuseResponsesClient, client_builder};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const MODELS: [&str; 5] = [
        "muse-spark-1.3",
        "muse-spark-1.2",
        "muse-wonder-1.0",
        "muse-wonder-full-1.0",
        "muse-ember-1.0",
    ];

    /// The text-only stream recorded on the wire for "Reply with the single
    /// word ok", event for event.
    const OBSERVED_TEXT_STREAM: &[(&str, &str)] = &[
        (
            "response.created",
            r#"{"response": {"background": false, "created_at": 1794274334.0, "error": null, "id": "resp_6af2741e267d9e80b98c465e", "incomplete_details": null, "instructions": "<instructions omitted: NoneType>", "model": "muse-spark-1.3", "object": "response", "output": [], "parallel_tool_calls": true, "reasoning": {"effort": "high"}, "service_tier": "default", "status": "in_progress", "store": true, "temperature": 1.0, "tool_choice": "auto", "tools": "<tools omitted: list>", "top_logprobs": 0, "top_p": 1.0, "truncation": "disabled"}, "sequence_number": 0, "type": "response.created"}"#,
        ),
        (
            "response.in_progress",
            r#"{"response": {"background": false, "created_at": 1794274334.0, "error": null, "id": "resp_6af2741e267d9e80b98c465e", "incomplete_details": null, "instructions": "<instructions omitted: NoneType>", "model": "muse-spark-1.3", "object": "response", "output": [], "parallel_tool_calls": true, "reasoning": {"effort": "high"}, "service_tier": "default", "status": "in_progress", "store": true, "temperature": 1.0, "tool_choice": "auto", "tools": "<tools omitted: list>", "top_logprobs": 0, "top_p": 1.0, "truncation": "disabled"}, "sequence_number": 1, "type": "response.in_progress"}"#,
        ),
        (
            "response.output_item.added",
            r#"{"item": {"id": "rs_6af2741e267d9e80b98c465e:rs_01a1c3159692719f9ff568e42c3d6492", "status": "in_progress", "summary": [], "type": "reasoning"}, "output_index": 0, "sequence_number": 2, "type": "response.output_item.added"}"#,
        ),
        (
            "response.output_item.done",
            r#"{"item": {"id": "rs_6af2741e267d9e80b98c465e:rs_01a1c3159692719f9ff568e42c3d6492", "status": "completed", "summary": [], "type": "reasoning"}, "output_index": 0, "sequence_number": 3, "type": "response.output_item.done"}"#,
        ),
        (
            "response.output_item.added",
            r#"{"item": {"content": [], "id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "role": "assistant", "status": "in_progress", "type": "message"}, "output_index": 1, "sequence_number": 4, "type": "response.output_item.added"}"#,
        ),
        (
            "response.content_part.added",
            r#"{"content_index": 0, "item_id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "output_index": 1, "part": {"annotations": [], "logprobs": [], "text": "", "type": "output_text"}, "sequence_number": 5, "type": "response.content_part.added"}"#,
        ),
        (
            "response.output_text.delta",
            r#"{"content_index": 0, "delta": "ok", "item_id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "logprobs": [], "output_index": 1, "sequence_number": 6, "type": "response.output_text.delta"}"#,
        ),
        (
            "response.output_text.done",
            r#"{"content_index": 0, "item_id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "logprobs": [], "output_index": 1, "sequence_number": 7, "text": "ok", "type": "response.output_text.done"}"#,
        ),
        (
            "response.content_part.done",
            r#"{"content_index": 0, "item_id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "output_index": 1, "part": {"annotations": [], "logprobs": [], "text": "ok", "type": "output_text"}, "sequence_number": 8, "type": "response.content_part.done"}"#,
        ),
        (
            "response.output_item.done",
            r#"{"item": {"content": [{"annotations": [], "logprobs": [], "text": "ok", "type": "output_text"}], "id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "role": "assistant", "status": "completed", "type": "message"}, "output_index": 1, "sequence_number": 9, "type": "response.output_item.done"}"#,
        ),
        (
            "response.completed",
            r#"{"response": {"background": false, "completed_at": 1794274335.0, "created_at": 1794274334.0, "error": null, "id": "resp_6af2741e267d9e80b98c465e", "incomplete_details": null, "instructions": "<instructions omitted: NoneType>", "model": "muse-spark-1.3", "object": "response", "output": [{"id": "rs_6af2741e267d9e80b98c465e:rs_01a1c3159692719f9ff568e42c3d6492", "status": "completed", "summary": [], "type": "reasoning"}, {"content": [{"annotations": [], "logprobs": [], "text": "ok", "type": "output_text"}], "id": "msg_01a1c3159c0872d48e1f1d4210d984ae", "role": "assistant", "status": "completed", "type": "message"}], "parallel_tool_calls": true, "reasoning": {"effort": "high"}, "service_tier": "default", "status": "completed", "store": true, "temperature": 1.0, "tool_choice": "auto", "tools": "<tools omitted: list>", "top_logprobs": 0, "top_p": 1.0, "truncation": "disabled", "usage": {"input_tokens": 14, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 307, "output_tokens_details": {"reasoning_tokens": 297}, "total_tokens": 321}}, "sequence_number": 10, "type": "response.completed"}"#,
        ),
        (
            "response.subscription_usage",
            r#"{"subscription":{"tier":"12345678901234567","weekly":{"resets_at":1794787200,"used_percent":0},"window":{"resets_at":1794291326,"used_percent":1,"window_duration_mins":300}},"type":"response.subscription_usage"}"#,
        ),
    ];

    const OBSERVED_404: &str = r#"{"error":{"code":"model_not_found","message":"The requested model was not found. It may not exist, or this API key may not have access to it. Check the model name; GET /v1/models lists the models this API key can use.","param":null,"type":"invalid_request_error"}}"#;

    fn observed_sse() -> String {
        let mut sse = String::new();
        for (event, data) in OBSERVED_TEXT_STREAM {
            sse.push_str(&format!("event: {event}\ndata: {data}\n\n"));
        }
        sse.push_str("data: [DONE]\n\n");
        sse
    }

    fn observed_mint_body() -> String {
        json!({
            "api_key": "synthetic-minted-api-key-48-chars-0000000000000",
            "base_url": "https://api.meta.ai/v1",
            "has_payment_method": false,
            "require_payment": false,
            "is_subs_active": true,
            "can_subscribe": false,
            "show_subs_upsell": true,
            "user_full_name": "Synthetic Person",
            "user_email": "synthetic.person@example.com",
            "user_avatar_url": null,
            "payment_method": null,
            "action_url": null,
            "subs_tier_id": "12345678901234567",
            "subs_tier_name": "Muse Code Max Usage",
            "is_subs_upgrade_available": false,
            "subs_usage": {
                "window": {"used_percent": 1, "window_duration_mins": 300, "resets_at": 1794291326},
                "weekly": {"used_percent": 0, "resets_at": 1794787200},
                "tier": "12345678901234567"
            }
        })
        .to_string()
    }

    /// Model API and key mint on one local server. `error` replaces every
    /// Model API answer when set.
    struct Upstream {
        _dir: tempfile::TempDir,
        registry: Arc<Registry>,
        responses: Arc<AtomicUsize>,
        mints: Arc<AtomicUsize>,
        bodies: Arc<Mutex<Vec<Value>>>,
    }

    async fn spawn_upstream(error: Option<(StatusCode, &'static str)>) -> Upstream {
        let responses = Arc::new(AtomicUsize::new(0));
        let mints = Arc::new(AtomicUsize::new(0));
        let bodies: Arc<Mutex<Vec<Value>>> = Arc::default();
        let (seen, recorded, minted) = (responses.clone(), bodies.clone(), mints.clone());
        let app = axum::Router::new()
            .route(
                "/v1/responses",
                post(move |body: String| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    recorded
                        .lock()
                        .unwrap()
                        .push(serde_json::from_str(&body).unwrap_or(Value::Null));
                    async move {
                        match error {
                            // A 200 answer is a stream the test composed.
                            Some((status, body)) => (
                                status,
                                [(
                                    "content-type",
                                    if status == StatusCode::OK {
                                        "text/event-stream"
                                    } else {
                                        "application/json"
                                    },
                                )],
                                body.to_string(),
                            )
                                .into_response(),
                            None => (
                                StatusCode::OK,
                                [("content-type", "text/event-stream")],
                                observed_sse(),
                            )
                                .into_response(),
                        }
                    }
                }),
            )
            .route(
                "/muse-code/key",
                post(move || {
                    minted.fetch_add(1, Ordering::SeqCst);
                    async move {
                        (
                            StatusCode::OK,
                            [("content-type", "application/json")],
                            observed_mint_body(),
                        )
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("auth.json").to_string_lossy().into_owned();
        store_at(path.clone())
            .save_auth(StoredAuth {
                access_token: "synthetic-oauth-access-token".into(),
                obtained_at_ms: 1_794_274_189_000,
                client_id: "1031625952748946".into(),
                oauth_host: "https://auth.meta.com".into(),
                user_full_name: None,
                is_subs_active: true,
                subs_tier_name: None,
            })
            .unwrap();
        let model_base = format!("{base}/v1");
        let minter = Arc::new(HttpMinter::with_client(
            mint::async_client_builder(MINT_TIMEOUT)
                .no_proxy()
                .build()
                .unwrap(),
            MintTarget {
                mint_base_url: base,
                model_base_url: model_base.clone(),
                user_agent: "claude-code-proxy/test".into(),
            },
        ));
        let manager = Arc::new(MuseAuthManager::with_minter(
            store_at(path),
            minter,
            MINT_TIMEOUT,
            CALLER_DEADLINE,
        ));
        let client = MuseResponsesClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            model_base,
            "claude-code-proxy/test".into(),
            manager,
        )
        .unwrap();
        let registry = Arc::new(Registry::from_providers(
            AliasProvider::Codex,
            vec![Arc::new(MuseProvider::with_client(client)) as Arc<dyn Provider>],
        ));
        Upstream {
            _dir: dir,
            registry,
            responses,
            mints,
            bodies,
        }
    }

    async fn post_json(app: axum::Router, uri: &str, body: Value) -> (StatusCode, String) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(body_string(&body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    fn messages_body(model: &str, stream: bool) -> Value {
        json!({
            "model": model,
            "max_tokens": 256,
            "stream": stream,
            "messages": [{"role": "user", "content": "Reply with the single word ok"}]
        })
    }

    fn event_names(sse: &str) -> Vec<&str> {
        sse.lines()
            .filter_map(|line| line.strip_prefix("event: "))
            .collect()
    }

    #[tokio::test]
    async fn each_model_streams_messages_in_anthropic_order() {
        for model in MODELS {
            let upstream = spawn_upstream(None).await;
            let (status, sse) = post_json(
                app(upstream.registry.clone()),
                "/v1/messages",
                messages_body(model, true),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{model}: {sse}");
            let events = event_names(&sse);
            assert_eq!(events.first(), Some(&"message_start"), "{model}: {sse}");
            assert_eq!(events.last(), Some(&"message_stop"), "{model}: {sse}");
            let delta = events.iter().position(|e| *e == "message_delta").unwrap();
            let first_block = events
                .iter()
                .position(|e| *e == "content_block_start")
                .unwrap();
            assert!(first_block < delta, "{model}: {events:?}");
            assert!(sse.contains(r#""text":"ok""#), "{model}: {sse}");
            assert!(!sse.contains("subscription_usage"), "{model}: {sse}");
            assert!(!sse.contains("event: error"), "{model}: {sse}");
            let sent = upstream.bodies.lock().unwrap()[0].clone();
            assert_eq!(sent["model"], model);
            assert_eq!(sent["store"], false);
            assert_eq!(sent["include"], json!(["reasoning.encrypted_content"]));
        }
    }

    #[tokio::test]
    async fn each_model_answers_non_streaming_messages_with_a_message_object() {
        for model in MODELS {
            let upstream = spawn_upstream(None).await;
            let (status, body) = post_json(
                app(upstream.registry.clone()),
                "/v1/messages",
                messages_body(model, false),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            let message: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(message["type"], "message", "{model}");
            assert_eq!(message["stop_reason"], "end_turn", "{model}");
            assert!(
                message["usage"]["output_tokens"].as_u64().is_some(),
                "{model}"
            );
            let text: Vec<_> = message["content"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|block| block["type"] == "text")
                .map(|block| block["text"].as_str().unwrap())
                .collect();
            assert_eq!(text, ["ok"], "{model}");
        }
    }

    #[tokio::test]
    async fn count_tokens_is_answered_locally_for_each_model() {
        let upstream = spawn_upstream(None).await;
        for model in MODELS {
            let (status, body) = post_json(
                app(upstream.registry.clone()),
                "/v1/messages/count_tokens",
                json!({"model": model, "messages": [{"role": "user", "content": "hello"}]}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert!(value["input_tokens"].as_u64().unwrap() > 0, "{model}");
        }
        assert_eq!(upstream.responses.load(Ordering::SeqCst), 0);
        assert_eq!(upstream.mints.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn openai_routes_return_success_shapes_streaming_and_not() {
        let upstream = spawn_upstream(None).await;
        let (status, body) = post_json(
            app_with_options(upstream.registry.clone(), None, true),
            "/v1/chat/completions",
            json!({"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "Reply with the single word ok"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "ok", "{body}");

        let (status, body) = post_json(
            app_with_options(upstream.registry.clone(), None, true),
            "/v1/chat/completions",
            json!({"model": "muse-spark-1.3", "stream": true, "messages": [{"role": "user", "content": "Reply with the single word ok"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains(r#""content":"ok""#), "{body}");
        assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");

        let (status, body) = post_json(
            app_with_options(upstream.registry.clone(), None, true),
            "/v1/responses",
            json!({"model": "muse-spark-1.3", "input": "Reply with the single word ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        let text: Vec<_> = value["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "message")
            .flat_map(|item| item["content"].as_array().unwrap().clone())
            .filter_map(|part| part["text"].as_str().map(str::to_string))
            .collect();
        assert_eq!(text, ["ok"], "{body}");

        let (status, body) = post_json(
            app_with_options(upstream.registry.clone(), None, true),
            "/v1/responses",
            json!({"model": "muse-spark-1.3", "stream": true, "input": "Reply with the single word ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("event: response.completed"), "{body}");
        assert!(body.contains(r#""delta":"ok""#), "{body}");
    }

    #[tokio::test]
    async fn openai_routes_carry_the_upstream_status_and_code_on_error() {
        let upstream = spawn_upstream(Some((StatusCode::NOT_FOUND, OBSERVED_404))).await;
        for (uri, body) in [
            (
                "/v1/chat/completions",
                json!({"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "hello"}]}),
            ),
            (
                "/v1/responses",
                json!({"model": "muse-spark-1.3", "input": "hello"}),
            ),
        ] {
            let (status, text) = post_json(
                app_with_options(upstream.registry.clone(), None, true),
                uri,
                body,
            )
            .await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}: {text}");
            let value: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(value["error"]["code"], "model_not_found", "{uri}");
            assert_eq!(value["error"]["type"], "invalid_request_error", "{uri}");
            assert!(
                value["error"]["message"]
                    .as_str()
                    .unwrap()
                    .ends_with("(upstream HTTP 404)"),
                "{uri}: {text}"
            );
        }
    }

    #[tokio::test]
    async fn monitor_shows_muse_requests_models_and_upstream_errors() {
        let upstream = spawn_upstream(None).await;
        let monitor = MonitorHandle::new(10);
        let (status, _) = post_json(
            app_with_monitor(upstream.registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            messages_body("muse-wonder-full-1.0", false),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let snapshot = monitor.snapshot();
        let recent = &snapshot.recent[0];
        assert_eq!(recent.status, RequestStatus::Completed);
        assert_eq!(recent.provider.as_deref(), Some("muse"));
        assert_eq!(recent.model.as_deref(), Some("muse-wonder-full-1.0"));
        assert_eq!(recent.output_tokens, Some(307));

        let failing = spawn_upstream(Some((StatusCode::NOT_FOUND, OBSERVED_404))).await;
        let monitor = MonitorHandle::new(10);
        let (status, _) = post_json(
            app_with_monitor(failing.registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            messages_body("muse-spark-1.3", false),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let snapshot = monitor.snapshot();
        let recent = &snapshot.recent[0];
        assert_eq!(recent.status, RequestStatus::Failed);
        assert_eq!(recent.provider.as_deref(), Some("muse"));
        assert_eq!(recent.http_status, Some(404));
        let error = recent.error.as_deref().unwrap();
        assert!(
            error.contains("The requested model was not found"),
            "{error}"
        );
        assert!(error.ends_with("(upstream HTTP 404)"), "{error}");
    }

    const ACCESS_TOKEN: &str = "synthetic-oauth-access-token";
    const API_KEY: &str = "synthetic-minted-api-key-48-chars-0000000000000";
    const EMAIL: &str = "synthetic.person@example.com";

    /// Upstream failures that carry both credentials and an e-mail address
    /// in every text field: an HTTP error body, a `response.failed` frame
    /// and an `error` frame.
    fn hostile_answers() -> Vec<(StatusCode, &'static str)> {
        let text = format!("key {API_KEY} token {ACCESS_TOKEN} owner {EMAIL}");
        let http = json!({
            "error": {"code": API_KEY, "message": text, "param": EMAIL, "type": "invalid_request_error"},
            "detail": format!("detail {ACCESS_TOKEN} {EMAIL}"),
            "title": format!("title {API_KEY}")
        });
        let failed = json!({
            "type": "response.failed",
            "response": {"status": "failed", "error": {"code": API_KEY, "message": text, "param": EMAIL}}
        });
        let error = json!({"type": "error", "code": API_KEY, "message": text, "param": EMAIL});
        let leak = |body: String| -> &'static str { Box::leak(body.into_boxed_str()) };
        vec![
            (StatusCode::BAD_REQUEST, leak(http.to_string())),
            (
                StatusCode::OK,
                leak(format!(
                    "event: response.failed\ndata: {failed}\n\ndata: [DONE]\n\n"
                )),
            ),
            (
                StatusCode::OK,
                leak(format!("event: error\ndata: {error}\n\ndata: [DONE]\n\n")),
            ),
        ]
    }

    #[tokio::test]
    async fn monitor_error_text_of_a_synthetic_hostile_upstream_failure_leaks_nothing() {
        for (status, body) in hostile_answers() {
            for stream in [false, true] {
                let upstream = spawn_upstream(Some((status, body))).await;
                let monitor = MonitorHandle::new(10);
                let (answered, downstream) = post_json(
                    app_with_monitor(upstream.registry.clone(), Some(monitor.clone())),
                    "/v1/messages",
                    messages_body("muse-spark-1.3", stream),
                )
                .await;
                let case = format!("upstream {status}, stream {stream}");
                // A failure inside a stream arrives with the stream's 200.
                if stream && status == StatusCode::OK {
                    assert_eq!(answered, StatusCode::OK, "{case}");
                    assert!(event_names(&downstream).contains(&"error"), "{case}");
                } else {
                    assert!(
                        answered.is_client_error() || answered.is_server_error(),
                        "{case}"
                    );
                }
                let snapshot = monitor.snapshot();
                let recent = &snapshot.recent[0];
                assert_eq!(recent.status, RequestStatus::Failed, "{case}");
                let error = recent.error.as_deref().unwrap();
                // The scrubbed upstream text is what the monitor shows.
                assert!(
                    error.contains("key [redacted] token [redacted] owner [redacted]"),
                    "{case}: {error}"
                );
                assert!(
                    error.ends_with(&format!("(upstream HTTP {})", status.as_u16())),
                    "{case}: {error}"
                );
                for (place, text) in [("monitor", error), ("response", downstream.as_str())] {
                    for secret in [API_KEY, ACCESS_TOKEN, EMAIL, "@example.com"] {
                        assert!(
                            !text.contains(secret),
                            "{case}: {place} leaked {secret}: {text}"
                        );
                    }
                }
            }
        }
    }

    fn expected_cache_key(identity: &str) -> String {
        use sha2::{Digest, Sha256};
        let digest = hex::encode(Sha256::digest(identity.as_bytes()));
        format!("ccp:{}", &digest[..32])
    }

    #[tokio::test]
    async fn conversation_identity_becomes_the_prompt_cache_key() {
        let upstream = spawn_upstream(None).await;
        let app = app(upstream.registry.clone());
        for (headers, expected) in [
            (
                vec![("x-claude-code-session-id", "session-main")],
                Some(expected_cache_key("main\0session-main")),
            ),
            (
                vec![
                    ("x-claude-code-session-id", "session-agent"),
                    ("x-claude-code-agent-id", "agent-direct"),
                ],
                Some(expected_cache_key("agent\0session-agent\0agent-direct")),
            ),
            (vec![], None),
        ] {
            let status = call_identity_ingress(
                &app,
                "/v1/messages",
                &headers,
                messages_body("muse-spark-1.3", false),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{headers:?}");
            let sent = upstream.bodies.lock().unwrap().pop().unwrap();
            assert_eq!(
                sent.get("prompt_cache_key").and_then(Value::as_str),
                expected.as_deref(),
                "{headers:?}"
            );
            // The raw session id never leaves the proxy.
            assert!(!sent.to_string().contains("session-"), "{sent}");
        }
    }

    fn model_ids(value: &Value) -> Vec<String> {
        value["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn models_endpoint_lists_exactly_the_five_muse_models() {
        let upstream = spawn_upstream(None).await;
        let (status, value) =
            get_models(app(upstream.registry.clone()), "/v1/models?limit=1000").await;
        assert_eq!(status, StatusCode::OK);
        let mut ids = model_ids(&value);
        ids.sort_unstable();
        let mut expected: Vec<String> = MODELS.iter().map(|model| model.to_string()).collect();
        expected.sort_unstable();
        assert_eq!(ids, expected);

        // In the full catalog the five are listed, the hidden image model is
        // not, and the only contributor ids are the two OpenCode Go serves.
        let registry = Arc::new(Registry::with_default_alias());
        let (status, value) = get_models(app(registry.clone()), "/v1/models?limit=1000").await;
        assert_eq!(status, StatusCode::OK);
        let ids = model_ids(&value);
        for model in MODELS {
            assert!(ids.iter().any(|id| id == model), "{model}");
        }
        assert!(!ids.iter().any(|id| id == "muse-image-1.0"));
        for id in ids.iter().filter(|id| id.ends_with("-contributor")) {
            assert_eq!(
                registry.provider_for_model(id, None).unwrap().name(),
                "opencode",
                "{id}"
            );
        }
    }
}

mod google_ai_plan {
    use super::*;
    use claude_code_proxy::auth::InMemoryAuthStore;
    use claude_code_proxy::config::{GoogleAiPlanClient, GoogleAiPlanSettings};
    use claude_code_proxy::providers::google_ai_plan::GoogleAiPlanProvider;
    use claude_code_proxy::providers::google_ai_plan::auth::manager::GoogleAiPlanAuthManager;
    use claude_code_proxy::providers::google_ai_plan::auth::token_store::{
        GoogleAiPlanTokenStore, StoredAuth,
    };
    use claude_code_proxy::providers::google_ai_plan::client::InteractionsClient;
    use claude_code_proxy::providers::vertex::client::client_builder;

    const TOKEN: &str = "ya29.synthetic-plan-server-token-5e6f7a";

    /// One text turn the way Cloud Code PA streams it.
    fn text_turn() -> String {
        [json!({
            "response": {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "ok"}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 1
                }
            }
        })]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
    }

    /// A fake Cloud Code PA endpoint that answers `loadCodeAssist` with a
    /// project ID and `streamGenerateContent` with `text_turn`, counting the
    /// generation calls.
    async fn registry() -> (Arc<Registry>, Arc<Mutex<Vec<Value>>>) {
        let bodies: Arc<Mutex<Vec<Value>>> = Arc::default();
        let seen = bodies.clone();
        let upstream = axum::Router::new().route(
            "/{*path}",
            axum::routing::post(move |uri: http::Uri, body: axum::body::Bytes| {
                let seen = seen.clone();
                async move {
                    if uri.path().ends_with(":loadCodeAssist") {
                        return (
                            [(http::header::CONTENT_TYPE, "application/json")],
                            json!({"cloudaicompanionProject": "synthetic-project"}).to_string(),
                        );
                    }
                    seen.lock()
                        .unwrap()
                        .push(serde_json::from_slice(&body).unwrap_or(Value::Null));
                    (
                        [(http::header::CONTENT_TYPE, "text/event-stream")],
                        text_turn(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let client = GoogleAiPlanClient {
            client_id: "synthetic-server-client".to_string(),
            client_secret: "synthetic-server-secret".to_string(),
        };
        let settings = GoogleAiPlanSettings {
            base_url: format!("http://{addr}"),
            client: client.clone(),
            user_agent: "synthetic-agent/1".to_string(),
        };
        let store = GoogleAiPlanTokenStore::new(InMemoryAuthStore::new());
        store
            .save_auth(StoredAuth {
                access: TOKEN.to_string(),
                refresh: "synthetic-refresh".to_string(),
                expires_at_ms: u64::MAX / 2,
                client_id: client.client_id.clone(),
                dpop_key: String::new(),
                dpop_nonce: None,
                project_id: None,
            })
            .unwrap();
        let auth =
            GoogleAiPlanAuthManager::with_token_url(store, client, "http://127.0.0.1:9/token")
                .unwrap();
        let interactions = InteractionsClient::with_client(
            client_builder().no_proxy().build().unwrap(),
            &settings,
        );
        let provider = GoogleAiPlanProvider::with_parts(settings, interactions, auth);
        let providers: Vec<Arc<dyn Provider>> = vec![Arc::new(provider)];
        (
            Arc::new(Registry::from_providers(AliasProvider::Codex, providers)),
            bodies,
        )
    }

    async fn post_json(app: axum::Router, uri: &str, body: Value) -> (StatusCode, String) {
        let request = Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let bounded = std::time::Duration::from_secs(20);
        let response = tokio::time::timeout(bounded, app.oneshot(request))
            .await
            .expect("the proxy answers within 20 seconds")
            .unwrap();
        let status = response.status();
        let bytes = tokio::time::timeout(
            bounded,
            http_body_util::BodyExt::collect(response.into_body()),
        )
        .await
        .expect("the body ends within 20 seconds")
        .unwrap()
        .to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn openai_routes_answer_from_the_plan_streaming_and_not() {
        let (registry, bodies) = registry().await;
        let model = "google-ai-plan/gemini-4-neon";
        let (status, body) = post_json(
            app_with_options(registry.clone(), None, true),
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": "Say ok"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["content"], "ok", "{body}");
        // The echoed name routes back here; the bare upstream ID would
        // route to Vertex AI.
        assert_eq!(value["model"], model, "{body}");

        let (status, body) = post_json(
            app_with_options(registry.clone(), None, true),
            "/v1/chat/completions",
            json!({"model": model, "stream": true, "messages": [{"role": "user", "content": "Say ok"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains(r#""content":"ok""#), "{body}");
        assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
        assert!(
            body.contains(&format!(r#""model":"{model}""#)),
            "every chunk names the routable model: {body}"
        );
        assert!(!body.contains(r#""model":"gemini-4-neon""#), "{body}");

        let (status, body) = post_json(
            app_with_options(registry.clone(), None, true),
            "/v1/responses",
            json!({"model": model, "input": "Say ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["model"], model, "{body}");
        assert!(body.contains(r#""text":"ok""#), "{body}");
        assert!(!body.contains(TOKEN), "{body}");

        let (status, body) = post_json(
            app_with_options(registry.clone(), None, true),
            "/v1/responses",
            json!({"model": model, "stream": true, "input": "Say ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("event: response.completed"), "{body}");
        assert!(body.contains(r#""delta":"ok""#), "{body}");
        assert!(
            body.contains(&format!(r#""model":"{model}""#)),
            "the response names the routable model: {body}"
        );
        assert!(!body.contains(r#""model":"gemini-4-neon""#), "{body}");

        let sent = bodies.lock().unwrap().clone();
        assert_eq!(sent.len(), 4, "{sent:?}");
        for body in sent {
            assert_eq!(body["project"], "synthetic-project", "{body}");
            assert_eq!(body["model"], "gemini-4-neon", "{body}");
            assert_eq!(body["userAgent"], "antigravity", "{body}");
            assert_eq!(body["requestType"], "agent", "{body}");
        }
    }
}
