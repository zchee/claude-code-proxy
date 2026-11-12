//! The Vertex AI provider driven through the proxy's HTTP routes against a
//! local stand-in for Vertex AI. Vertex traffic was not observed when these
//! were written, so every upstream frame is synthetic and built from the
//! documented `streamGenerateContent` response shape.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode, Uri, header};
use axum::response::Response;
use bytes::Bytes;
use claude_code_proxy::{
    config::{AliasProvider, resolve_vertex_config},
    monitor::{MonitorHandle, RequestStatus, snapshot::MonitorSnapshot},
    provider::Provider,
    providers::vertex::{
        VertexProvider,
        auth::VertexCredentials,
        client::{VertexClient, client_builder},
        translate::thought_signature::ENVELOPE_PREFIX,
    },
    registry::Registry,
    server::{app_with_monitor, app_with_options},
};
use futures_util::{Stream, StreamExt, stream};
use serde_json::{Value, json};
use tower::util::ServiceExt;

/// Shaped like a Google OAuth access token.
const SYNTHETIC_TOKEN: &str =
    "ya29.synthetic-access-token-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ";

/// A thought signature as Gemini returns it: opaque base64.
const SYNTHETIC_SIGNATURE: &str =
    "CiQBVKhc7synthetic0signature0AAAAbbbbCCCCddddEEEEffff0123456789GGGGhhhhIIIIjjjjKKKK==";

/// Inline image data of the length a screenshot would have, as base64.
const SYNTHETIC_IMAGE_DATA: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg";

const MODEL: &str = "gemini-3.5-flash";

const EXPECTED_PATH: &str = "/v1/projects/ccp-project/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent?alt=sse";

/// One scripted answer of the stand-in.
struct Answer {
    status: StatusCode,
    headers: Vec<(&'static str, String)>,
    body: Vec<String>,
    hold_open: bool,
    /// Sends this many body chunks, then waits until the flag is set
    /// before sending the rest.
    gate: Option<(usize, Arc<AtomicBool>)>,
}

impl Answer {
    /// A 200 event stream with one `data:` frame per value, closed after
    /// the last one.
    fn sse(frames: &[Value]) -> Self {
        Self {
            status: StatusCode::OK,
            headers: vec![("content-type", "text/event-stream".to_string())],
            body: frames
                .iter()
                .map(|frame| format!("data: {frame}\r\n\r\n"))
                .collect(),
            hold_open: false,
            gate: None,
        }
    }

    /// [`Self::sse`], but after the first `sent` frames the stand-in stays
    /// silent, with the connection open, until `release` is set.
    fn sse_gated(frames: &[Value], sent: usize, release: Arc<AtomicBool>) -> Self {
        Self {
            gate: Some((sent, release)),
            ..Self::sse(frames)
        }
    }

    /// [`Self::sse`], but the connection stays open after the last frame.
    fn sse_held_open(frames: &[Value]) -> Self {
        Self {
            hold_open: true,
            ..Self::sse(frames)
        }
    }

    /// A non-success answer with a JSON body.
    fn error(status: u16, body: Value) -> Self {
        Self {
            status: StatusCode::from_u16(status).expect("valid status"),
            headers: vec![("content-type", "application/json".to_string())],
            body: vec![body.to_string()],
            hold_open: false,
            gate: None,
        }
    }

    fn into_response(self) -> Response {
        let mut chunks: Vec<Result<Bytes, Infallible>> = self
            .body
            .into_iter()
            .map(|chunk| Ok(Bytes::from(chunk)))
            .collect();
        let chunks: Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>> = match self.gate
        {
            Some((sent, release)) => {
                let rest = chunks.split_off(sent.min(chunks.len()));
                let wait = stream::once(async move {
                    while !release.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                })
                .filter_map(|()| async { None::<Result<Bytes, Infallible>> });
                Box::pin(stream::iter(chunks).chain(wait).chain(stream::iter(rest)))
            }
            None => Box::pin(stream::iter(chunks)),
        };
        let body: Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>> = if self.hold_open
        {
            Box::pin(chunks.chain(stream::pending()))
        } else {
            chunks
        };
        let mut response = Response::new(Body::from_stream(body));
        *response.status_mut() = self.status;
        for (name, value) in self.headers {
            response.headers_mut().insert(
                name,
                HeaderValue::from_str(&value).expect("valid header value"),
            );
        }
        response
    }
}

/// What the stand-in received.
#[derive(Debug, Clone)]
struct Recorded {
    path_and_query: String,
    authorization: Option<String>,
    accept: Option<String>,
    body: Value,
}

#[derive(Clone)]
struct StandIn {
    answers: Arc<Mutex<VecDeque<Answer>>>,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

async fn answer_next(
    State(stand_in): State<StandIn>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let header_text = |name: header::HeaderName| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    stand_in.requests.lock().unwrap().push(Recorded {
        path_and_query: uri
            .path_and_query()
            .map(|value| value.as_str().to_string())
            .unwrap_or_default(),
        authorization: header_text(header::AUTHORIZATION),
        accept: header_text(header::ACCEPT),
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
    });
    let next = stand_in.answers.lock().unwrap().pop_front();
    match next {
        Some(answer) => answer.into_response(),
        None => Answer::error(
            500,
            json!({"error":{"code":500,"message":"no scripted answer left","status":"INTERNAL"}}),
        )
        .into_response(),
    }
}

/// A loopback stand-in for Vertex AI that answers each request with the next
/// scripted answer and records what it received.
struct FakeVertex {
    base: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    server: tokio::task::JoinHandle<()>,
}

impl FakeVertex {
    async fn start(answers: Vec<Answer>) -> Self {
        let stand_in = StandIn {
            answers: Arc::new(Mutex::new(VecDeque::from(answers))),
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let requests = stand_in.requests.clone();
        let app = Router::new().fallback(answer_next).with_state(stand_in);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            base: format!("http://{address}"),
            requests,
            server,
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for FakeVertex {
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// A provider aimed at `upstream` with fixed credentials. `CCP_VERTEX_PROJECT`
/// is `ccp-project`; `extra_env` adds more variables.
fn provider(upstream: &FakeVertex, extra_env: &[(&str, &str)]) -> VertexProvider {
    let mut env: HashMap<String, String> = HashMap::from([
        ("CCP_VERTEX_PROJECT".to_string(), "ccp-project".to_string()),
        ("CCP_VERTEX_BASE_URL".to_string(), upstream.base.clone()),
    ]);
    for (name, value) in extra_env {
        env.insert((*name).to_string(), (*value).to_string());
    }
    let config_dir = tempfile::tempdir().expect("temporary config directory");
    let config = resolve_vertex_config(&env, config_dir.path());
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {SYNTHETIC_TOKEN}")).unwrap(),
    );
    let client = VertexClient::with_client(client_builder().no_proxy().build().unwrap());
    VertexProvider::with_client(client, VertexCredentials::fixed(headers), config)
}

fn registry(provider: VertexProvider) -> Arc<Registry> {
    Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [Arc::new(provider) as Arc<dyn Provider>],
    ))
}

async fn post(app: Router, uri: &str, body: &Value) -> (StatusCode, HeaderMap, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, headers, String::from_utf8(bytes.to_vec()).unwrap())
}

/// The `data:` payloads of an Anthropic SSE body, with their event names.
fn sse_events(body: &str) -> Vec<(String, Value)> {
    let mut events = Vec::new();
    for frame in body.split("\n\n") {
        let mut name = String::new();
        let mut data = String::new();
        for line in frame.lines() {
            if let Some(value) = line.strip_prefix("event: ") {
                name = value.to_string();
            } else if let Some(value) = line.strip_prefix("data: ") {
                data.push_str(value);
            }
        }
        if !data.is_empty() {
            events.push((name, serde_json::from_str(&data).unwrap()));
        }
    }
    events
}

/// Rebuilds the content blocks a streaming client would assemble.
fn streamed_content(events: &[(String, Value)]) -> Vec<Value> {
    let mut blocks: Vec<Value> = Vec::new();
    let mut partial_json: Vec<String> = Vec::new();
    for (name, data) in events {
        match name.as_str() {
            "content_block_start" => {
                blocks.push(data["content_block"].clone());
                partial_json.push(String::new());
            }
            "content_block_delta" => {
                let index = data["index"].as_u64().unwrap() as usize;
                let delta = &data["delta"];
                match delta["type"].as_str().unwrap() {
                    "text_delta" => {
                        let text = blocks[index]["text"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string()
                            + delta["text"].as_str().unwrap();
                        blocks[index]["text"] = json!(text);
                    }
                    "thinking_delta" => {
                        let text = blocks[index]["thinking"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string()
                            + delta["thinking"].as_str().unwrap();
                        blocks[index]["thinking"] = json!(text);
                    }
                    "signature_delta" => {
                        blocks[index]["signature"] = delta["signature"].clone();
                    }
                    "input_json_delta" => {
                        partial_json[index].push_str(delta["partial_json"].as_str().unwrap());
                    }
                    other => panic!("unexpected delta {other}"),
                }
            }
            _ => {}
        }
    }
    for (block, json_text) in blocks.iter_mut().zip(partial_json) {
        if block["type"] == "tool_use" && !json_text.is_empty() {
            block["input"] = serde_json::from_str(&json_text).unwrap();
        }
    }
    blocks
}

fn stop_reason(events: &[(String, Value)]) -> Option<String> {
    events
        .iter()
        .find(|(name, _)| name == "message_delta")
        .and_then(|(_, data)| data["delta"]["stop_reason"].as_str())
        .map(str::to_string)
}

fn text_frame(text: &str) -> Value {
    json!({"candidates":[{"content":{"role":"model","parts":[{"text":text}]}}]})
}

fn finish_frame(reason: &str) -> Value {
    json!({
        "candidates":[{"content":{"role":"model","parts":[]},"finishReason":reason}],
        "usageMetadata":{
            "promptTokenCount":40,
            "cachedContentTokenCount":10,
            "candidatesTokenCount":6,
            "thoughtsTokenCount":4
        }
    })
}

/// Two parallel `read_file` calls; Gemini signs only the first.
fn parallel_calls_frame() -> Value {
    json!({"candidates":[{"content":{"role":"model","parts":[
        {"functionCall":{"name":"read_file","args":{"path":"src/a.rs"}},"thoughtSignature":SYNTHETIC_SIGNATURE},
        {"functionCall":{"name":"read_file","args":{"path":"src/b.rs"}}}
    ]}}]})
}

fn read_file_tool() -> Value {
    json!({
        "name":"read_file",
        "description":"Read a file",
        "input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    })
}

fn message_request(model: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "max_tokens": 256,
        "stream": stream,
        "messages": [{"role":"user","content":"Say hello"}]
    })
}

fn error_message(body: &str) -> String {
    let value: Value = serde_json::from_str(body).unwrap_or_else(|_| panic!("not JSON: {body}"));
    value["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn vertex_synthetic_non_streaming_text_reaches_the_documented_url_with_the_ccp_project() {
    let upstream = FakeVertex::start(vec![Answer::sse(&[
        text_frame("Hello"),
        text_frame(" there"),
        finish_frame("STOP"),
    ])])
    .await;
    // CCP_VERTEX_PROJECT wins over GOOGLE_CLOUD_PROJECT.
    let provider = provider(&upstream, &[("GOOGLE_CLOUD_PROJECT", "google-project")]);
    let monitor = MonitorHandle::new(10);
    let (status, _, body) = post(
        app_with_monitor(registry(provider), Some(monitor.clone())),
        "/v1/messages",
        &message_request("vertex/gemini-3.5-flash", false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let message: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(message["type"], "message");
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["model"], "vertex/gemini-3.5-flash");
    assert_eq!(
        message["content"],
        json!([{"type":"text","text":"Hello there"}])
    );
    assert_eq!(message["stop_reason"], "end_turn");
    // input = prompt - cached; output = candidates + thoughts.
    assert_eq!(message["usage"]["input_tokens"], 30);
    assert_eq!(message["usage"]["cache_read_input_tokens"], 10);
    assert_eq!(message["usage"]["output_tokens"], 10);

    let requests = upstream.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.path_and_query, EXPECTED_PATH);
    assert_eq!(
        request.authorization.as_deref(),
        Some(format!("Bearer {SYNTHETIC_TOKEN}").as_str())
    );
    assert_eq!(request.accept.as_deref(), Some("text/event-stream"));
    assert_eq!(
        request.body["contents"],
        json!([{"role":"user","parts":[{"text":"Say hello"}]}])
    );
    assert_eq!(request.body["generationConfig"]["maxOutputTokens"], 256);
    assert!(request.body.get("model").is_none(), "{}", request.body);

    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), 1);
    assert_eq!(state.recent[0].status, RequestStatus::Completed);
    assert_eq!(state.recent[0].provider.as_deref(), Some("vertex"));
    assert!(
        state.recent[0]
            .model
            .as_deref()
            .is_some_and(|model| model.contains(MODEL)),
        "{:?}",
        state.recent[0].model
    );
    assert_eq!(state.recent[0].output_tokens, Some(10));
}

#[tokio::test]
async fn vertex_synthetic_streaming_text_is_sent_as_anthropic_events() {
    let upstream = FakeVertex::start(vec![Answer::sse(&[
        text_frame("Hello"),
        text_frame(" there"),
        finish_frame("STOP"),
    ])])
    .await;
    let (status, headers, body) = post(
        app_with_monitor(registry(provider(&upstream, &[])), None),
        "/v1/messages",
        &message_request(MODEL, true),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let events = sse_events(&body);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names.first(), Some(&"message_start"), "{body}");
    assert_eq!(names.last(), Some(&"message_stop"), "{body}");
    assert!(!names.contains(&"error"), "{body}");
    assert_eq!(events[0].1["message"]["model"], MODEL);
    assert_eq!(
        streamed_content(&events),
        vec![json!({"type":"text","text":"Hello there"})]
    );
    assert_eq!(stop_reason(&events).as_deref(), Some("end_turn"));
    assert_eq!(upstream.requests()[0].path_and_query, EXPECTED_PATH);
}

#[tokio::test]
async fn vertex_synthetic_streaming_and_non_streaming_agree_on_tool_use_and_max_tokens() {
    for (frames, expected_stop) in [
        (
            vec![
                text_frame("Reading."),
                parallel_calls_frame(),
                finish_frame("STOP"),
            ],
            "tool_use",
        ),
        (
            vec![text_frame("Cut"), finish_frame("MAX_TOKENS")],
            "max_tokens",
        ),
    ] {
        let upstream = FakeVertex::start(vec![Answer::sse(&frames), Answer::sse(&frames)]).await;
        let registry = registry(provider(&upstream, &[]));
        let request = |stream: bool| {
            json!({
                "model": MODEL,
                "max_tokens": 256,
                "stream": stream,
                "tools": [read_file_tool()],
                "messages": [{"role":"user","content":"Read both files"}]
            })
        };
        let (status, _, buffered) = post(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &request(false),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{buffered}");
        let (status, _, streamed) = post(
            app_with_monitor(registry, None),
            "/v1/messages",
            &request(true),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{streamed}");

        let buffered: Value = serde_json::from_str(&buffered).unwrap();
        let events = sse_events(&streamed);
        assert_eq!(buffered["stop_reason"], expected_stop);
        assert_eq!(stop_reason(&events).as_deref(), Some(expected_stop));
        // The message IDs differ per request, and tool-use IDs are derived
        // from them; compare everything else.
        let without_ids = |blocks: Vec<Value>| -> Vec<Value> {
            blocks
                .into_iter()
                .map(|mut block| {
                    if let Some(object) = block.as_object_mut() {
                        object.remove("id");
                    }
                    block
                })
                .collect()
        };
        let buffered_content = without_ids(buffered["content"].as_array().unwrap().clone());
        assert_eq!(
            buffered_content,
            without_ids(streamed_content(&events)),
            "{buffered}"
        );
        if expected_stop == "tool_use" {
            let content = buffered["content"].as_array().unwrap();
            let tool_uses: Vec<&Value> = content
                .iter()
                .filter(|block| block["type"] == "tool_use")
                .collect();
            assert_eq!(tool_uses.len(), 2, "{buffered}");
            assert_eq!(tool_uses[0]["input"], json!({"path":"src/a.rs"}));
            assert_eq!(tool_uses[1]["input"], json!({"path":"src/b.rs"}));
            assert_ne!(tool_uses[0]["id"], tool_uses[1]["id"]);
        }
    }
}

#[tokio::test]
async fn vertex_synthetic_signature_from_the_first_turn_goes_back_on_the_first_call_only() {
    let upstream = FakeVertex::start(vec![
        Answer::sse(&[parallel_calls_frame(), finish_frame("STOP")]),
        Answer::sse(&[text_frame("Both read."), finish_frame("STOP")]),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    let first_turn = json!({
        "model": MODEL,
        "max_tokens": 256,
        "tools": [read_file_tool()],
        "messages": [{"role":"user","content":"Read both files"}]
    });
    let (status, _, body) = post(
        app_with_monitor(registry.clone(), None),
        "/v1/messages",
        &first_turn,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: Value = serde_json::from_str(&body).unwrap();
    let content = first["content"].as_array().unwrap().clone();
    let thinking = content
        .iter()
        .find(|block| block["type"] == "thinking")
        .unwrap_or_else(|| panic!("the signature is carried in a thinking block: {body}"));
    let signature = thinking["signature"].as_str().unwrap();
    assert!(signature.starts_with(ENVELOPE_PREFIX), "{signature}");
    assert!(signature.ends_with(SYNTHETIC_SIGNATURE), "{signature}");
    let tool_ids: Vec<String> = content
        .iter()
        .filter(|block| block["type"] == "tool_use")
        .map(|block| block["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(tool_ids.len(), 2, "{body}");

    // Claude Code sends the assistant turn back unchanged with the results.
    let second_turn = json!({
        "model": MODEL,
        "max_tokens": 256,
        "tools": [read_file_tool()],
        "messages": [
            {"role":"user","content":"Read both files"},
            {"role":"assistant","content":content},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":tool_ids[0],"content":"fn a() {}"},
                {"type":"tool_result","tool_use_id":tool_ids[1],"content":"fn b() {}"}
            ]}
        ]
    });
    let (status, _, body) = post(
        app_with_monitor(registry, None),
        "/v1/messages",
        &second_turn,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let requests = upstream.requests();
    assert_eq!(requests.len(), 2);
    let contents = requests[1].body["contents"].as_array().unwrap().clone();
    assert_eq!(contents.len(), 3, "{}", requests[1].body);
    assert_eq!(contents[1]["role"], "model");
    let calls = contents[1]["parts"].as_array().unwrap();
    assert_eq!(calls.len(), 2, "{}", contents[1]);
    assert_eq!(calls[0]["functionCall"]["name"], "read_file");
    assert_eq!(calls[0]["functionCall"]["args"], json!({"path":"src/a.rs"}));
    assert_eq!(calls[0]["thoughtSignature"], SYNTHETIC_SIGNATURE);
    assert_eq!(calls[1]["functionCall"]["args"], json!({"path":"src/b.rs"}));
    assert!(calls[1].get("thoughtSignature").is_none(), "{}", calls[1]);
    assert!(calls[0]["functionCall"].get("id").is_none(), "{}", calls[0]);
    let results = contents[2]["parts"].as_array().unwrap();
    assert_eq!(contents[2]["role"], "user");
    assert_eq!(
        results[0]["functionResponse"],
        json!({"name":"read_file","response":{"output":"fn a() {}"}})
    );
    assert_eq!(
        results[1]["functionResponse"],
        json!({"name":"read_file","response":{"output":"fn b() {}"}})
    );
    // Thinking text never goes upstream.
    assert!(
        !requests[1].body.to_string().contains(ENVELOPE_PREFIX),
        "{}",
        requests[1].body
    );
}

#[tokio::test]
async fn vertex_synthetic_history_without_a_usable_signature_gets_the_skip_value() {
    let tool_use =
        json!({"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"src/a.rs"}});
    let other_model_signature =
        format!("{ENVELOPE_PREFIX}gemini-3.1-pro-preview:{SYNTHETIC_SIGNATURE}");
    let histories = [
        // The thinking block was removed from the history.
        json!([tool_use.clone()]),
        // The signature came from another model.
        json!([
            {"type":"thinking","thinking":"","signature":other_model_signature},
            tool_use.clone()
        ]),
    ];
    let upstream = FakeVertex::start(
        histories
            .iter()
            .map(|_| Answer::sse(&[text_frame("ok"), finish_frame("STOP")]))
            .collect(),
    )
    .await;
    let registry = registry(provider(&upstream, &[]));
    for history in &histories {
        let request = json!({
            "model": MODEL,
            "max_tokens": 256,
            "tools": [read_file_tool()],
            "messages": [
                {"role":"user","content":"Read the file"},
                {"role":"assistant","content":history},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"fn a() {}"}]}
            ]
        });
        let (status, _, body) = post(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &request,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{history}: {body}");
    }
    let requests = upstream.requests();
    assert_eq!(requests.len(), histories.len());
    for (request, history) in requests.iter().zip(&histories) {
        let call = &request.body["contents"][1]["parts"][0];
        assert_eq!(call["functionCall"]["name"], "read_file", "{history}");
        // The value Google documents for a call it did not sign, written
        // out rather than taken from the code.
        assert_eq!(
            call["thoughtSignature"], "skip_thought_signature_validator",
            "{history}: {}",
            request.body
        );
        assert!(
            !request.body.to_string().contains(SYNTHETIC_SIGNATURE),
            "{history}: {}",
            request.body
        );
    }
}

#[tokio::test]
async fn vertex_synthetic_upstream_errors_map_to_anthropic_error_types() {
    let google_error = |code: u16, status: &str, message: &str| json!({"error":{"code":code,"message":message,"status":status}});
    let rate_limited = json!({"error":{
        "code":429,
        "message":"Resource exhausted.",
        "status":"RESOURCE_EXHAUSTED",
        "details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"2.540s"}]
    }});
    let cases = [
        (
            Answer::error(
                401,
                google_error(
                    401,
                    "UNAUTHENTICATED",
                    "Request had invalid authentication credentials.",
                ),
            ),
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            vec![
                "gcloud auth application-default login",
                "(upstream HTTP 401)",
            ],
        ),
        (
            Answer::error(
                403,
                google_error(
                    403,
                    "PERMISSION_DENIED",
                    "Permission denied on resource project ccp-project.",
                ),
            ),
            StatusCode::FORBIDDEN,
            "permission_error",
            vec![
                "roles/aiplatform.user",
                "aiplatform.googleapis.com",
                "set-quota-project ccp-project",
                "(upstream HTTP 403)",
            ],
        ),
        (
            Answer::error(
                404,
                google_error(404, "NOT_FOUND", "Publisher Model was not found."),
            ),
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            vec![
                "Model gemini-3.5-flash is not available in project ccp-project / location global",
                "(upstream HTTP 404)",
            ],
        ),
        (
            Answer::error(429, rate_limited),
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            vec!["gemini-3.5-flash", "global", "(upstream HTTP 429)"],
        ),
        (
            Answer::error(
                400,
                google_error(
                    400,
                    "INVALID_ARGUMENT",
                    "Request contains an invalid argument.",
                ),
            ),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            vec![
                "Request contains an invalid argument.",
                "(upstream HTTP 400)",
            ],
        ),
        (
            Answer::error(
                503,
                google_error(503, "UNAVAILABLE", "The service is currently unavailable."),
            ),
            StatusCode::BAD_GATEWAY,
            "api_error",
            vec!["(upstream HTTP 503)"],
        ),
    ];
    let expected: Vec<_> = cases
        .iter()
        .map(|(_, status, kind, needles)| (*status, *kind, needles.clone()))
        .collect();
    let upstream = FakeVertex::start(cases.into_iter().map(|(answer, ..)| answer).collect()).await;
    let registry = registry(provider(&upstream, &[]));
    for (status, kind, needles) in expected {
        let (actual, headers, body) = post(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(MODEL, false),
        )
        .await;
        assert_eq!(actual, status, "{body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["type"], "error", "{body}");
        assert_eq!(value["error"]["type"], kind, "{body}");
        let message = error_message(&body);
        for needle in needles {
            assert!(message.contains(needle), "missing {needle:?}: {message}");
        }
        let retry_after = headers
            .get(header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok());
        if status == StatusCode::TOO_MANY_REQUESTS {
            // 2.540 s is rounded up.
            assert_eq!(retry_after, Some("3"), "{body}");
        } else {
            assert_eq!(retry_after, None, "{body}");
        }
    }
}

#[tokio::test]
async fn vertex_synthetic_echoed_secrets_are_redacted_in_responses_and_the_monitor() {
    let envelope = format!("{ENVELOPE_PREFIX}{MODEL}:{SYNTHETIC_SIGNATURE}");
    let echo = format!(
        "token {SYNTHETIC_TOKEN} signature {envelope} image {SYNTHETIC_IMAGE_DATA} owner synthetic.person@example.com"
    );
    let http_error = json!({"error":{"code":400,"message":echo,"status":"INVALID_ARGUMENT"}});
    let stream_error = json!({"error":{"code":500,"message":echo,"status":"INTERNAL"}});
    let blocked = json!({
        "candidates":[{"content":{"role":"model","parts":[]},"finishReason":"SAFETY","finishMessage":echo}]
    });
    let upstream = FakeVertex::start(vec![
        Answer::error(400, http_error),
        Answer::sse(&[text_frame("partial"), stream_error.clone()]),
        Answer::sse(&[text_frame("partial"), stream_error]),
        Answer::sse(&[text_frame("partial"), blocked.clone()]),
        Answer::sse(&[text_frame("partial"), blocked]),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    let monitor = MonitorHandle::new(10);
    let mut downstream = String::new();
    for stream in [false, false, true, false, true] {
        let (status, _, body) = post(
            app_with_monitor(registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            &message_request(MODEL, stream),
        )
        .await;
        if stream {
            // The status is already 200 when the stream fails; the failure
            // is the final error event.
            assert_eq!(status, StatusCode::OK, "{body}");
            let events = sse_events(&body);
            let (name, data) = events.last().unwrap();
            assert_eq!(name, "error", "{body}");
            assert!(
                data["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("[redacted]"),
                "{body}"
            );
        } else {
            assert_ne!(status, StatusCode::OK, "{body}");
            assert!(error_message(&body).contains("[redacted]"), "{body}");
        }
        downstream.push_str(&body);
    }

    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), 5);
    let mut monitored = String::new();
    for request in &state.recent {
        assert_eq!(request.status, RequestStatus::Failed, "{request:?}");
        let error = request.error.as_deref().unwrap_or_default();
        assert!(!error.is_empty(), "{request:?}");
        monitored.push_str(error);
        monitored.push('\n');
    }
    assert!(monitored.contains("finishReason SAFETY"), "{monitored}");
    // Every field the monitor can show or send to an attached dashboard, not
    // only the error strings.
    let snapshot = serde_json::to_string(&MonitorSnapshot::from(state.clone()))
        .expect("the monitor snapshot serialises");
    assert!(snapshot.contains("[redacted]"), "{snapshot}");
    for (place, text) in [
        ("responses", &downstream),
        ("monitor errors", &monitored),
        ("monitor snapshot", &snapshot),
    ] {
        for secret in [
            SYNTHETIC_TOKEN,
            SYNTHETIC_SIGNATURE,
            SYNTHETIC_IMAGE_DATA,
            "synthetic.person@example.com",
        ] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
}

#[tokio::test]
async fn vertex_synthetic_upstream_left_open_after_the_finish_reason_completes_both_modes() {
    let frames = [text_frame("Hello"), finish_frame("STOP")];
    let upstream = FakeVertex::start(vec![
        Answer::sse_held_open(&frames),
        Answer::sse_held_open(&frames),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    for stream in [false, true] {
        let started = Instant::now();
        let (status, _, body) = tokio::time::timeout(
            Duration::from_secs(20),
            post(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(MODEL, stream),
            ),
        )
        .await
        .unwrap_or_else(|_| panic!("stream={stream}: an open upstream held the response"));
        let elapsed = started.elapsed();
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        // The response waits for a late usage-only chunk, then completes.
        assert!(
            elapsed >= Duration::from_millis(1900),
            "stream={stream}: completed after {elapsed:?}, before the read limit"
        );
        if stream {
            let events = sse_events(&body);
            assert_eq!(
                events.last().map(|(name, _)| name.as_str()),
                Some("message_stop")
            );
            assert_eq!(stop_reason(&events).as_deref(), Some("end_turn"));
        } else {
            let message: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(message["stop_reason"], "end_turn");
            assert_eq!(message["content"][0]["text"], "Hello");
        }
    }
}

#[tokio::test]
async fn vertex_synthetic_upstream_silent_before_the_finish_reason_is_not_cut_off() {
    // A model can think for a long time before its next chunk. Without a
    // finishReason the proxy has no deadline of its own: only the upstream
    // ends the response. The silence below is longer than the 2 s the proxy
    // waits after a finishReason.
    for stream in [false, true] {
        let release = Arc::new(AtomicBool::new(false));
        let upstream = FakeVertex::start(vec![Answer::sse_gated(
            &[text_frame("Thinking it over."), finish_frame("STOP")],
            1,
            release.clone(),
        )])
        .await;
        let app = app_with_monitor(registry(provider(&upstream, &[])), None);
        let body = message_request(MODEL, stream);
        let request = tokio::spawn(async move { post(app, "/v1/messages", &body).await });
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(
            !request.is_finished(),
            "stream={stream}: the response ended while the upstream was silent"
        );
        assert_eq!(upstream.requests().len(), 1, "stream={stream}");
        release.store(true, Ordering::SeqCst);
        let (status, _, body) = tokio::time::timeout(Duration::from_secs(20), request)
            .await
            .unwrap_or_else(|_| panic!("stream={stream}: no answer after the upstream finished"))
            .expect("the request task completes");
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        if stream {
            let events = sse_events(&body);
            assert_eq!(
                streamed_content(&events),
                vec![json!({"type":"text","text":"Thinking it over."})],
                "{body}"
            );
            assert_eq!(stop_reason(&events).as_deref(), Some("end_turn"));
        } else {
            let message: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(message["content"][0]["text"], "Thinking it over.");
            assert_eq!(message["stop_reason"], "end_turn");
        }
    }
}

#[tokio::test]
async fn vertex_synthetic_openai_chat_completions_route_returns_gemini_text() {
    let upstream = FakeVertex::start(vec![Answer::sse(&[
        text_frame("Hello from Gemini"),
        finish_frame("STOP"),
    ])])
    .await;
    let (status, _, body) = post(
        app_with_options(registry(provider(&upstream, &[])), None, true),
        "/v1/chat/completions",
        &json!({"model":"vertex/gemini-3.5-flash","messages":[{"role":"user","content":"hello"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        value["choices"][0]["message"]["content"], "Hello from Gemini",
        "{body}"
    );
    assert_eq!(upstream.requests()[0].path_and_query, EXPECTED_PATH);
}

#[tokio::test]
async fn vertex_count_tokens_is_answered_locally() {
    let upstream = FakeVertex::start(Vec::new()).await;
    let (status, _, body) = post(
        app_with_monitor(registry(provider(&upstream, &[])), None),
        "/v1/messages/count_tokens",
        &json!({"model":"vertex/gemini-3.5-flash","messages":[{"role":"user","content":"hello world"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert!(value["input_tokens"].as_u64().unwrap() > 0, "{body}");
    assert!(upstream.requests().is_empty());
}

#[tokio::test]
async fn vertex_synthetic_unregistered_ids_are_sent_and_other_publishers_are_refused() {
    let upstream =
        FakeVertex::start(vec![Answer::sse(&[text_frame("ok"), finish_frame("STOP")])]).await;
    let registry = registry(provider(&upstream, &[]));
    let (status, _, body) = post(
        app_with_monitor(registry.clone(), None),
        "/v1/messages",
        &message_request("vertex/gemini-9-unreleased", false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        upstream.requests()[0].path_and_query,
        "/v1/projects/ccp-project/locations/global/publishers/google/models/gemini-9-unreleased:streamGenerateContent?alt=sse"
    );
    // An unregistered ID gets the smallest level every Gemini model accepts.
    assert_eq!(
        upstream.requests()[0].body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "LOW"
    );

    for model in [
        "vertex/anthropic/claude-sonnet-5",
        "vertex/gemini-3.5-flash:predict",
    ] {
        let (status, _, body) = post(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(model, false),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{model}: {body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error", "{body}");
    }
    assert_eq!(upstream.requests().len(), 1);
}

#[tokio::test]
async fn vertex_models_are_listed_by_the_models_route() {
    let upstream = FakeVertex::start(Vec::new()).await;
    let response = app_with_monitor(registry(provider(&upstream, &[])), None)
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    for model in [
        "gemini-3.5-flash",
        "vertex/gemini-3.5-flash",
        "vertex/gemini-4-argon",
    ] {
        assert!(
            body.contains(&format!("\"{model}\"")),
            "missing {model}: {body}"
        );
    }
}
