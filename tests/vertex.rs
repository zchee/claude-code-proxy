//! The Vertex AI provider driven through the proxy's HTTP routes against a
//! local stand-in for Vertex AI. Vertex traffic was not observed when these
//! were written, so every upstream frame is synthetic and built from the
//! documented response shapes: `streamGenerateContent` for Gemini, the
//! Anthropic Messages events for Claude, OpenAI Responses events for xAI
//! Grok, OpenAI and Meta Muse, and `chat.completion.chunk` objects for
//! Mistral, Model Garden's managed open models and deployed endpoints.

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
    config::{AliasProvider, resolve_vertex_config, resolve_vertex_endpoints},
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

    /// An answer whose body is exactly `chunks`, one body chunk each,
    /// closed after the last one.
    fn raw(status: u16, content_type: &str, chunks: Vec<String>) -> Self {
        Self {
            status: StatusCode::from_u16(status).expect("valid status"),
            headers: vec![("content-type", content_type.to_string())],
            body: chunks,
            hold_open: false,
            gate: None,
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
    method: Method,
    path_and_query: String,
    authorization: Option<String>,
    accept: Option<String>,
    /// Every request header as received, repeated names included.
    headers: HeaderMap,
    body: Value,
}

#[derive(Clone)]
struct StandIn {
    answers: Arc<Mutex<VecDeque<Answer>>>,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

async fn answer_next(
    State(stand_in): State<StandIn>,
    method: Method,
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
        method,
        path_and_query: uri
            .path_and_query()
            .map(|value| value.as_str().to_string())
            .unwrap_or_default(),
        authorization: header_text(header::AUTHORIZATION),
        accept: header_text(header::ACCEPT),
        headers: headers.clone(),
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
    post_with_headers(app, uri, body, &[]).await
}

/// [`post`] with extra request headers. A name given twice is sent as two
/// header lines.
async fn post_with_headers(
    app: Router,
    uri: &str,
    body: &Value,
    headers: &[(&str, &str)],
) -> (StatusCode, HeaderMap, String) {
    let mut request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = app
        .oneshot(request.body(Body::from(body.to_string())).unwrap())
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
        "vertex/ai21/jamba-1.5-large",
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

// Partner model families. Every expected URL, model value and body field
// below is written out from Google's partner-model documentation, not taken
// from the provider's constants.

/// The longest any request of the partner tests may take. Only the
/// read-limit tests wait on purpose, and they wait about two seconds.
const REQUEST_DEADLINE: Duration = Duration::from_secs(30);

/// [`post_with_headers`] that fails the test instead of hanging when the
/// proxy does not answer within [`REQUEST_DEADLINE`].
async fn send(
    app: Router,
    uri: &str,
    body: &Value,
    headers: &[(&str, &str)],
) -> (StatusCode, HeaderMap, String) {
    tokio::time::timeout(REQUEST_DEADLINE, post_with_headers(app, uri, body, headers))
        .await
        .unwrap_or_else(|_| panic!("no answer from {uri} within {REQUEST_DEADLINE:?}: {body}"))
}

fn content_type(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
}

/// Asserts that a streaming answer ended in the proxy's `error` event with
/// `message`, and never told the client the message was complete.
fn assert_stream_failed_with(status: StatusCode, body: &str, message: &str) {
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = sse_events(body);
    let (name, data) = events.last().unwrap_or_else(|| panic!("no event: {body}"));
    assert_eq!(name, "error", "{body}");
    assert_eq!(data["type"], "error", "{body}");
    assert_eq!(data["error"]["type"], "api_error", "{body}");
    assert_eq!(data["error"]["message"], message, "{body}");
    assert!(
        !events.iter().any(|(name, _)| name == "message_stop"),
        "{body}"
    );
}

// Anthropic Claude.

const CLAUDE_MODEL: &str = "vertex/anthropic/claude-opus-5-5";

const CLAUDE_STREAM_PATH: &str = "/v1/projects/ccp-project/locations/global/publishers/anthropic/models/claude-opus-5-5:streamRawPredict";

const CLAUDE_RAW_PATH: &str = "/v1/projects/ccp-project/locations/global/publishers/anthropic/models/claude-opus-5-5:rawPredict";

/// The fixed message a client sees when a Claude stream fails.
const CLAUDE_STREAM_INVALID: &str = "Vertex AI Messages stream is invalid";

/// Shaped like an Anthropic thinking signature: standard base64 whose bytes
/// do not start with `ccp:`.
const ANTHROPIC_SIGNATURE: &str = "EqQBCkgIBRABGAIiQLsyntheticAnthropicSignature0123456789";

/// A thinking signature as this proxy's Codex translator makes it.
const PROXY_SIGNATURE: &str = "ccp:codex:v1:cnNfc3ludGhldGlj:gAAAAABsyntheticEncryptedReasoning";

/// base64url of `ccp:kimi:v1:synthetic-reasoning-state`: the form this
/// proxy's Kimi and OpenCode translators make.
const PROXY_SIGNATURE_BASE64URL: &str = "Y2NwOmtpbWk6djE6c3ludGhldGljLXJlYXNvbmluZy1zdGF0ZQ";

/// One Anthropic SSE event, written as Claude sends it.
fn claude_event(data: &Value) -> String {
    let name = data["type"].as_str().expect("an event has a type");
    format!("event: {name}\ndata: {data}\n\n")
}

/// A complete Claude stream answering `text`, one event per body chunk.
fn claude_stream(text: &str) -> Vec<String> {
    [
        json!({"type":"message_start","message":{
            "id":"msg_vrtx_01Synthetic","type":"message","role":"assistant","model":"claude-opus-5-5",
            "content":[],"stop_reason":null,"stop_sequence":null,
            "usage":{"input_tokens":12,"output_tokens":1}
        }}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"ping"}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}),
        json!({"type":"message_stop"}),
    ]
    .iter()
    .map(claude_event)
    .collect()
}

/// Claude's answer to a non-streaming request, in the shape of Google's
/// example response.
fn claude_message(text: &str) -> String {
    json!({
        "id":"msg_vrtx_01Synthetic","type":"message","role":"assistant","model":"claude-opus-5-5",
        "content":[{"type":"text","text":text}],
        "stop_reason":"end_turn","stop_sequence":null,
        "usage":{"input_tokens":727,"output_tokens":308}
    })
    .to_string()
}

fn claude_json(body: String) -> Answer {
    Answer::raw(200, "application/json", vec![body])
}

fn claude_sse(chunks: Vec<String>) -> Answer {
    Answer::raw(200, "text/event-stream", chunks)
}

#[tokio::test]
async fn vertex_claude_synthetic_stream_reaches_stream_raw_predict_and_passes_events_through() {
    let events = claude_stream("Hello from Claude");
    let upstream = FakeVertex::start(vec![claude_sse(events.clone())]).await;
    let monitor = MonitorHandle::new(10);
    let request = json!({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "stream": true,
        "system": "Be brief.",
        "messages": [{"role":"user","content":"Say hello"}]
    });
    let (status, headers, body) = send(
        app_with_monitor(registry(provider(&upstream, &[])), Some(monitor.clone())),
        "/v1/messages",
        &request,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(content_type(&headers), Some("text/event-stream"));
    // The events are forwarded, not rebuilt.
    assert_eq!(body, events.concat());

    let requests = upstream.requests();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert_eq!(sent.path_and_query, CLAUDE_STREAM_PATH);
    assert_eq!(sent.accept.as_deref(), Some("text/event-stream"));
    assert_eq!(
        sent.authorization.as_deref(),
        Some(format!("Bearer {SYNTHETIC_TOKEN}").as_str())
    );
    // The URL names the model; the body must not.
    assert!(sent.body.get("model").is_none(), "{}", sent.body);
    assert_eq!(sent.body["anthropic_version"], "vertex-2023-10-16");
    // The documented streaming recipe sets both the method and the flag.
    assert_eq!(sent.body["stream"], true, "{}", sent.body);
    assert_eq!(sent.body["max_tokens"], 256, "{}", sent.body);
    assert_eq!(sent.body["system"], "Be brief.", "{}", sent.body);
    assert_eq!(sent.body["messages"][0]["role"], "user", "{}", sent.body);
    assert_eq!(
        sent.body["messages"][0]["content"], "Say hello",
        "{}",
        sent.body
    );
    assert!(
        sent.headers.get("anthropic-beta").is_none(),
        "{:?}",
        sent.headers
    );

    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), 1);
    assert_eq!(
        state.recent[0].status,
        RequestStatus::Completed,
        "{:?}",
        state.recent[0]
    );
    assert_eq!(state.recent[0].provider.as_deref(), Some("vertex"));
}

#[tokio::test]
async fn vertex_claude_synthetic_non_streaming_reaches_raw_predict_and_returns_the_json_as_is() {
    let answer = claude_message("Hello from Claude");
    let dated_answer = claude_message("Hello from Opus 4.5");
    let upstream = FakeVertex::start(vec![
        claude_json(answer.clone()),
        claude_json(dated_answer.clone()),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    let (status, headers, body) = send(
        app_with_monitor(registry.clone(), None),
        "/v1/messages",
        &message_request(CLAUDE_MODEL, false),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(content_type(&headers), Some("application/json"));
    assert_eq!(body, answer);

    // A dated ID keeps its `@` in the URL, and a version the client chose
    // is kept.
    let mut dated = message_request("vertex/anthropic/claude-opus-4-5@20251101", false);
    dated["anthropic_version"] = json!("2026-10-21");
    let (status, _, body) = send(
        app_with_monitor(registry, None),
        "/v1/messages",
        &dated,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, dated_answer);

    let requests = upstream.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].path_and_query, CLAUDE_RAW_PATH);
    assert_eq!(requests[0].accept.as_deref(), Some("application/json"));
    assert!(
        requests[0].body.get("model").is_none(),
        "{}",
        requests[0].body
    );
    assert_eq!(requests[0].body["anthropic_version"], "vertex-2023-10-16");
    assert_eq!(requests[0].body["stream"], false, "{}", requests[0].body);
    assert_eq!(
        requests[1].path_and_query,
        "/v1/projects/ccp-project/locations/global/publishers/anthropic/models/claude-opus-4-5@20251101:rawPredict"
    );
    assert!(
        requests[1].body.get("model").is_none(),
        "{}",
        requests[1].body
    );
    assert_eq!(requests[1].body["anthropic_version"], "2026-10-21");
}

#[tokio::test]
async fn vertex_claude_synthetic_beta_headers_reach_vertex_as_one_header_of_listed_values() {
    let upstream = FakeVertex::start(vec![
        claude_json(claude_message("one")),
        claude_json(claude_message("two")),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    // Two header lines, each mixing values on the Vertex list with values
    // that are not on it, and one listed value sent twice.
    let (status, _, body) = send(
        app_with_monitor(registry.clone(), None),
        "/v1/messages",
        &message_request(CLAUDE_MODEL, false),
        &[
            (
                "anthropic-beta",
                "claude-code-20250219, oauth-2025-04-20, interleaved-thinking-2025-05-14",
            ),
            (
                "anthropic-beta",
                "context-1m-2025-08-07,synthetic-unlisted-2099-01-01,claude-code-20250219",
            ),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Nothing on the list.
    let (status, _, body) = send(
        app_with_monitor(registry, None),
        "/v1/messages",
        &message_request(CLAUDE_MODEL, false),
        &[
            ("anthropic-beta", "oauth-2025-04-20"),
            ("anthropic-beta", "synthetic-unlisted-2099-01-01"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let requests = upstream.requests();
    assert_eq!(requests.len(), 2);
    // Vertex AI reads only one anthropic-beta header.
    let sent: Vec<&HeaderValue> = requests[0]
        .headers
        .get_all("anthropic-beta")
        .iter()
        .collect();
    assert_eq!(sent.len(), 1, "{:?}", requests[0].headers);
    let mut values: Vec<&str> = sent[0]
        .to_str()
        .expect("an ASCII header")
        .split(',')
        .map(str::trim)
        .collect();
    values.sort_unstable();
    assert_eq!(
        values,
        [
            "claude-code-20250219",
            "context-1m-2025-08-07",
            "interleaved-thinking-2025-05-14"
        ],
        "{:?}",
        sent[0]
    );
    assert_eq!(
        requests[1].headers.get_all("anthropic-beta").iter().count(),
        0,
        "{:?}",
        requests[1].headers
    );
    // A beta never travels in the body on Vertex AI.
    for request in &requests {
        assert!(
            request.body.get("anthropic_beta").is_none(),
            "{}",
            request.body
        );
    }
}

#[tokio::test]
async fn vertex_claude_synthetic_thinking_claude_cannot_verify_is_dropped_and_its_own_is_kept() {
    let upstream = FakeVertex::start(vec![claude_json(claude_message("ok"))]).await;
    let request = json!({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "messages": [
            {"role":"user","content":"First question."},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"Codex reasoning","signature":PROXY_SIGNATURE},
                {"type":"text","text":"First answer."}
            ]},
            {"role":"user","content":"Second question."},
            // Only a thinking block another provider made: the message is
            // left out rather than sent empty.
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"Kimi reasoning","signature":PROXY_SIGNATURE_BASE64URL}
            ]},
            {"role":"user","content":"Third question."},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"Unsigned reasoning","signature":""},
                {"type":"thinking","thinking":"Claude reasoning","signature":ANTHROPIC_SIGNATURE},
                {"type":"redacted_thinking","data":"synthetic-redacted-thinking"},
                {"type":"text","text":"Third answer."}
            ]},
            {"role":"user","content":"Last question."}
        ]
    });
    let (status, _, body) = send(
        app_with_monitor(registry(provider(&upstream, &[])), None),
        "/v1/messages",
        &request,
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let requests = upstream.requests();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0].body;
    let text = sent.to_string();
    for dropped in [
        PROXY_SIGNATURE,
        PROXY_SIGNATURE_BASE64URL,
        "Codex reasoning",
        "Kimi reasoning",
        "Unsigned reasoning",
    ] {
        assert!(!text.contains(dropped), "{dropped} reached Claude: {sent}");
    }
    let messages = sent["messages"].as_array().expect("messages");
    for message in messages {
        assert_ne!(message["content"], json!([]), "{sent}");
    }
    let assistants: Vec<&Value> = messages
        .iter()
        .filter(|message| message["role"] == "assistant")
        .collect();
    assert_eq!(assistants.len(), 2, "{sent}");
    assert_eq!(
        assistants[0]["content"],
        json!([{"type":"text","text":"First answer."}]),
        "{sent}"
    );
    assert_eq!(
        assistants[1]["content"],
        json!([
            {"type":"thinking","thinking":"Claude reasoning","signature":ANTHROPIC_SIGNATURE},
            {"type":"redacted_thinking","data":"synthetic-redacted-thinking"},
            {"type":"text","text":"Third answer."}
        ]),
        "{sent}"
    );
    for question in [
        "First question.",
        "Second question.",
        "Third question.",
        "Last question.",
    ] {
        assert!(text.contains(question), "{question} is missing: {sent}");
    }
}

#[tokio::test]
async fn vertex_claude_synthetic_stream_error_or_early_close_ends_in_the_fixed_error() {
    let mut cut_short = claude_stream("Hello");
    // The connection closes after message_delta, before message_stop.
    cut_short.truncate(6);
    let upstream = FakeVertex::start(vec![
        claude_sse(vec![
            claude_stream("unused").remove(0),
            claude_event(
                &json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
            ),
        ]),
        claude_sse(cut_short),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    let monitor = MonitorHandle::new(10);
    for case in ["error event", "closed before message_stop"] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            &message_request(CLAUDE_MODEL, true),
            &[],
        )
        .await;
        assert_stream_failed_with(status, &body, CLAUDE_STREAM_INVALID);
        // The upstream's own error text is not forwarded.
        assert!(!body.contains("Overloaded"), "{case}: {body}");
    }
    assert_eq!(upstream.requests().len(), 2);

    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), 2);
    for request in &state.recent {
        assert_eq!(request.status, RequestStatus::Failed, "{request:?}");
        assert!(
            request
                .error
                .as_deref()
                .is_some_and(|error| error.contains(CLAUDE_STREAM_INVALID)),
            "{request:?}"
        );
    }
}

#[tokio::test]
async fn vertex_claude_synthetic_upstream_errors_with_anthropic_bodies_keep_their_message() {
    let anthropic_error =
        |kind: &str, message: &str| json!({"type":"error","error":{"type":kind,"message":message}});
    let cases = [
        (
            400,
            anthropic_error(
                "invalid_request_error",
                "messages.0.content: text content blocks must be non-empty",
            ),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            vec![
                "messages.0.content: text content blocks must be non-empty",
                "(upstream HTTP 400)",
            ],
        ),
        (
            429,
            anthropic_error(
                "rate_limit_error",
                "Number of request tokens has exceeded your rate limit",
            ),
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            vec![
                "Number of request tokens has exceeded your rate limit",
                "claude-opus-5-5",
                "(upstream HTTP 429)",
            ],
        ),
        (
            529,
            anthropic_error("overloaded_error", "Overloaded"),
            StatusCode::BAD_GATEWAY,
            "api_error",
            vec!["Overloaded", "(upstream HTTP 529)"],
        ),
    ];
    let mut answers = Vec::new();
    for (status, body, ..) in &cases {
        for _ in 0..2 {
            answers.push(Answer::error(*status, body.clone()));
        }
    }
    let upstream = FakeVertex::start(answers).await;
    let registry = registry(provider(&upstream, &[]));
    for (_, _, status, kind, needles) in &cases {
        for stream in [false, true] {
            let (actual, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(CLAUDE_MODEL, stream),
                &[],
            )
            .await;
            assert_eq!(actual, *status, "stream={stream}: {body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["type"], "error", "{body}");
            assert_eq!(value["error"]["type"], *kind, "{body}");
            let message = error_message(&body);
            for needle in needles {
                assert!(message.contains(needle), "missing {needle:?}: {message}");
            }
            // The message is taken out of the body, not the body itself.
            assert!(!message.contains("\"type\""), "{message}");
        }
    }
    assert_eq!(upstream.requests().len(), 6);
}

// xAI Grok, OpenAI and Meta Muse: OpenAI Responses.

/// A Responses family as Google documents it.
struct ResponsesFamily {
    /// The name the client asks for.
    model: &'static str,
    /// Settings for the provider; Muse is asked for in a location it is not
    /// served in, so the test shows the location it is sent to anyway.
    env: &'static [(&'static str, &'static str)],
    /// Where Vertex AI takes the request for project `ccp-project`.
    path: &'static str,
    /// The `model` value of the request body.
    upstream_model: &'static str,
}

const RESPONSES_FAMILIES: [ResponsesFamily; 3] = [
    ResponsesFamily {
        model: "vertex/xai/grok-4.7",
        env: &[],
        path: "/v1/projects/ccp-project/locations/global/endpoints/openapi/responses",
        upstream_model: "xai/grok-4.7",
    },
    ResponsesFamily {
        model: "vertex/openai/gpt-6.1-sol",
        env: &[],
        path: "/v1/projects/ccp-project/locations/global/publishers/openai/v1/responses",
        upstream_model: "gpt-6.1-sol",
    },
    ResponsesFamily {
        model: "vertex/meta/muse-spark-1.3",
        env: &[("CCP_VERTEX_LOCATION", "us-central1")],
        path: "/v1/projects/ccp-project/locations/global/publishers/meta/v1/responses",
        upstream_model: "muse-spark-1.3",
    },
];

/// The fixed message a client sees when a Responses stream fails.
const RESPONSES_STREAM_INVALID: &str = "Vertex AI Responses stream is invalid";

fn responses_completed() -> Value {
    json!({"type":"response.completed","response":{
        "id":"resp_synthetic","status":"completed","incomplete_details":null,
        "usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}
    }})
}

/// A complete Responses stream answering `text` through one delta.
fn responses_text(text: &str) -> Vec<Value> {
    vec![
        json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_up","role":"assistant","content":[]}}),
        json!({"type":"response.output_text.delta","output_index":0,"item_id":"msg_up","delta":text}),
        json!({"type":"response.output_text.done","output_index":0,"item_id":"msg_up","text":text}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_up","role":"assistant","status":"completed"}}),
        responses_completed(),
    ]
}

/// A complete Responses stream calling `read_file` on `src/a.rs`.
fn responses_tool_call() -> Vec<Value> {
    let arguments = r#"{"path":"src/a.rs"}"#;
    vec![
        json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{
            "type":"function_call","id":"fc_up","call_id":"call_synthetic_1","name":"read_file","arguments":""
        }}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_up","delta":r#"{"path":"#}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_up","delta":r#""src/a.rs"}"#}),
        json!({"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_up","arguments":arguments}),
        json!({"type":"response.output_item.done","output_index":0,"item":{
            "type":"function_call","id":"fc_up","call_id":"call_synthetic_1","name":"read_file","arguments":arguments
        }}),
        responses_completed(),
    ]
}

fn responses_request(model: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "max_tokens": 200_000,
        "stream": stream,
        "system": "Be brief.",
        "output_config": {"effort": "high"},
        "tools": [read_file_tool()],
        "messages": [{"role":"user","content":"Say hello"}]
    })
}

/// Asserts the client got the text `expected` as a complete message.
fn assert_text_answer(stream: bool, body: &str, model: &str, expected: &str) {
    if stream {
        let events = sse_events(body);
        assert_eq!(
            events.last().map(|(name, _)| name.as_str()),
            Some("message_stop"),
            "{body}"
        );
        assert!(!events.iter().any(|(name, _)| name == "error"), "{body}");
        assert_eq!(events[0].1["message"]["model"], model, "{body}");
        assert_eq!(
            streamed_content(&events),
            vec![json!({"type":"text","text":expected})],
            "{body}"
        );
        assert_eq!(stop_reason(&events).as_deref(), Some("end_turn"), "{body}");
    } else {
        let message: Value = serde_json::from_str(body).unwrap();
        assert_eq!(message["model"], model, "{body}");
        assert_eq!(
            message["content"],
            json!([{"type":"text","text":expected}]),
            "{body}"
        );
        assert_eq!(message["stop_reason"], "end_turn", "{body}");
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_requests_follow_each_publishers_documented_shape() {
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(vec![
            Answer::sse(&responses_text("Hello")),
            Answer::sse(&responses_text("Hello")),
        ])
        .await;
        let registry = registry(provider(&upstream, family.env));
        for stream in [false, true] {
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &responses_request(family.model, stream),
                &[],
            )
            .await;
            assert_eq!(
                status,
                StatusCode::OK,
                "{} stream={stream}: {body}",
                family.model
            );
            assert_text_answer(stream, &body, family.model, "Hello");
        }

        let requests = upstream.requests();
        assert_eq!(requests.len(), 2, "{}", family.model);
        for sent in &requests {
            let model = family.model;
            let body = &sent.body;
            assert_eq!(sent.path_and_query, family.path, "{model}");
            // The upstream is asked for a stream in both modes.
            assert_eq!(sent.accept.as_deref(), Some("text/event-stream"), "{model}");
            assert_eq!(body["stream"], true, "{model}: {body}");
            assert_eq!(body["model"], family.upstream_model, "{model}: {body}");
            assert_eq!(body["store"], false, "{model}: {body}");
            assert_eq!(body["instructions"], "Be brief.", "{model}: {body}");
            for absent in ["include", "prompt_cache_key"] {
                assert!(body.get(absent).is_none(), "{model}: {absent} in {body}");
            }
            for absent in ["/text/verbosity", "/reasoning/summary"] {
                assert!(
                    body.pointer(absent).is_none(),
                    "{model}: {absent} in {body}"
                );
            }
            assert_eq!(body["tools"][0]["type"], "function", "{model}: {body}");
            assert_eq!(body["tools"][0]["name"], "read_file", "{model}: {body}");
            if family.upstream_model == "gpt-6.1-sol" {
                // OpenAI documents an adjustable effort and an output limit
                // of 128,000 tokens.
                assert_eq!(body["reasoning"]["effort"], "high", "{model}: {body}");
                assert_eq!(body["max_output_tokens"], 128_000, "{model}: {body}");
            } else {
                assert!(body.get("reasoning").is_none(), "{model}: {body}");
                assert_eq!(body["max_output_tokens"], 200_000, "{model}: {body}");
            }
        }
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_thinking_from_the_history_is_not_replayed() {
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(vec![Answer::sse(&responses_text("Shorter."))]).await;
        let request = json!({
            "model": family.model,
            "max_tokens": 256,
            "messages": [
                {"role":"user","content":"Define proxy."},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"Codex reasoning","signature":PROXY_SIGNATURE},
                    {"type":"thinking","thinking":"Claude reasoning","signature":ANTHROPIC_SIGNATURE},
                    {"type":"redacted_thinking","data":"synthetic-redacted-thinking"},
                    {"type":"text","text":"A go-between."}
                ]},
                {"role":"user","content":"Shorter."}
            ]
        });
        let (status, _, body) = send(
            app_with_monitor(registry(provider(&upstream, family.env)), None),
            "/v1/messages",
            &request,
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}: {body}", family.model);
        let requests = upstream.requests();
        assert_eq!(requests.len(), 1);
        let sent = &requests[0].body;
        let text = sent.to_string();
        let input = sent["input"].as_array().expect("input items");
        assert!(
            input.iter().all(|item| item["type"] != "reasoning"),
            "{}: {sent}",
            family.model
        );
        for leaked in [
            "encrypted_content",
            "gAAAAABsyntheticEncryptedReasoning",
            ANTHROPIC_SIGNATURE,
            "Codex reasoning",
            "Claude reasoning",
            "synthetic-redacted-thinking",
        ] {
            assert!(
                !text.contains(leaked),
                "{} sent {leaked}: {sent}",
                family.model
            );
        }
        // The visible text of that turn still goes.
        assert!(text.contains("A go-between."), "{}: {sent}", family.model);
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_hosted_web_search_is_refused_without_an_upstream_request() {
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(Vec::new()).await;
        let registry = registry(provider(&upstream, family.env));
        for stream in [false, true] {
            let request = json!({
                "model": family.model,
                "max_tokens": 256,
                "stream": stream,
                "tools": [{"type":"web_search_20250305","name":"web_search","max_uses":5}],
                "messages": [{"role":"user","content":"Any news?"}]
            });
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &request,
                &[],
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{} stream={stream}: {body}",
                family.model
            );
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error", "{body}");
            let message = error_message(&body);
            assert!(message.contains("web_search"), "{message}");
            assert!(message.contains("remove the web search tool"), "{message}");
        }
        assert!(upstream.requests().is_empty(), "{}", family.model);
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_tool_call_goes_out_and_its_result_comes_back() {
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(vec![
            Answer::sse(&responses_tool_call()),
            Answer::sse(&responses_text("It defines a.")),
        ])
        .await;
        let registry = registry(provider(&upstream, family.env));
        let first_turn = json!({
            "model": family.model,
            "max_tokens": 256,
            "tools": [read_file_tool()],
            "messages": [{"role":"user","content":"Read src/a.rs"}]
        });
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &first_turn,
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}: {body}", family.model);
        let first: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(first["stop_reason"], "tool_use", "{body}");
        let content = first["content"].as_array().unwrap().clone();
        let calls: Vec<&Value> = content
            .iter()
            .filter(|block| block["type"] == "tool_use")
            .collect();
        assert_eq!(calls.len(), 1, "{body}");
        assert_eq!(calls[0]["id"], "call_synthetic_1", "{body}");
        assert_eq!(calls[0]["name"], "read_file", "{body}");
        assert_eq!(calls[0]["input"], json!({"path":"src/a.rs"}), "{body}");

        // Claude Code sends the call back with its result, and streams the
        // answer this time.
        let second_turn = json!({
            "model": family.model,
            "max_tokens": 256,
            "stream": true,
            "tools": [read_file_tool()],
            "messages": [
                {"role":"user","content":"Read src/a.rs"},
                {"role":"assistant","content":content},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"call_synthetic_1","content":"fn a() {}"}
                ]}
            ]
        });
        let (status, _, body) = send(
            app_with_monitor(registry, None),
            "/v1/messages",
            &second_turn,
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{}: {body}", family.model);
        assert_text_answer(true, &body, family.model, "It defines a.");

        let requests = upstream.requests();
        assert_eq!(requests.len(), 2);
        let input = requests[1].body["input"].as_array().expect("input items");
        let call = input
            .iter()
            .find(|item| item["type"] == "function_call")
            .unwrap_or_else(|| panic!("no function_call: {}", requests[1].body));
        assert_eq!(call["call_id"], "call_synthetic_1", "{call}");
        assert_eq!(call["name"], "read_file", "{call}");
        let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"path":"src/a.rs"}), "{call}");
        let output = input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .unwrap_or_else(|| panic!("no function_call_output: {}", requests[1].body));
        assert_eq!(output["call_id"], "call_synthetic_1", "{output}");
        assert!(
            output["output"].to_string().contains("fn a() {}"),
            "{output}"
        );
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_unlisted_or_late_events_end_in_the_fixed_error() {
    let mut unlisted = responses_text("Hello");
    unlisted.insert(
        2,
        json!({"type":"response.refusal.delta","output_index":0,"item_id":"msg_up","delta":"No."}),
    );
    let mut late = responses_text("Hello");
    late.push(json!({"type":"response.output_text.delta","output_index":0,"item_id":"msg_up","delta":"late"}));
    let cases = [
        ("unlisted_event_type", unlisted),
        ("event_after_completion", late),
    ];
    for family in &RESPONSES_FAMILIES {
        let answers = cases
            .iter()
            .flat_map(|(_, frames)| [Answer::sse(frames), Answer::sse(frames)])
            .collect();
        let upstream = FakeVertex::start(answers).await;
        let registry = registry(provider(&upstream, family.env));
        for (kind, _) in &cases {
            for stream in [false, true] {
                let (status, _, body) = send(
                    app_with_monitor(registry.clone(), None),
                    "/v1/messages",
                    &message_request(family.model, stream),
                    &[],
                )
                .await;
                if stream {
                    assert_stream_failed_with(status, &body, RESPONSES_STREAM_INVALID);
                } else {
                    assert_eq!(
                        status,
                        StatusCode::BAD_GATEWAY,
                        "{} {kind}: {body}",
                        family.model
                    );
                    let value: Value = serde_json::from_str(&body).unwrap();
                    assert_eq!(value["error"]["type"], "api_error", "{body}");
                    let message = error_message(&body);
                    assert!(
                        message.contains(&format!("{RESPONSES_STREAM_INVALID}: {kind}")),
                        "{} {kind}: {message}",
                        family.model
                    );
                }
            }
        }
        assert_eq!(upstream.requests().len(), 4, "{}", family.model);
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_upstream_left_open_after_the_completion_completes_both_modes() {
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(vec![
            Answer::sse_held_open(&responses_text("Hello")),
            Answer::sse_held_open(&responses_text("Hello")),
        ])
        .await;
        let registry = registry(provider(&upstream, family.env));
        for stream in [false, true] {
            let started = Instant::now();
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(family.model, stream),
                &[],
            )
            .await;
            let elapsed = started.elapsed();
            assert_eq!(
                status,
                StatusCode::OK,
                "{} stream={stream}: {body}",
                family.model
            );
            // Nothing may follow the completion, so the proxy waits out its
            // two-second limit before it settles the message.
            assert!(
                elapsed >= Duration::from_millis(1900),
                "{} stream={stream}: completed after {elapsed:?}, before the read limit",
                family.model
            );
            assert_text_answer(stream, &body, family.model, "Hello");
        }
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_snapshot_text_that_no_delta_delivered_is_an_error() {
    let frames = vec![
        json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_up","role":"assistant","content":[]}}),
        json!({"type":"response.output_text.done","output_index":0,"item_id":"msg_up","text":"Text only in a snapshot"}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_up","role":"assistant","status":"completed"}}),
        responses_completed(),
    ];
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(vec![Answer::sse(&frames), Answer::sse(&frames)]).await;
        let registry = registry(provider(&upstream, family.env));
        for stream in [false, true] {
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(family.model, stream),
                &[],
            )
            .await;
            if stream {
                assert_stream_failed_with(status, &body, RESPONSES_STREAM_INVALID);
            } else {
                assert_eq!(status, StatusCode::BAD_GATEWAY, "{}: {body}", family.model);
                assert!(
                    error_message(&body).contains("snapshot_without_delta"),
                    "{}: {body}",
                    family.model
                );
            }
            // The text was never given to the client as if it were complete.
            assert!(!body.contains("message_stop"), "{body}");
        }
    }
}

// Mistral: Chat-Completions-shaped `streamRawPredict`.

const MISTRAL_MODEL: &str = "vertex/mistralai/mistral-medium-3";

/// Mistral is served in `us-central1` and `europe-west4` only.
const MISTRAL_LOCATION: (&str, &str) = ("CCP_VERTEX_LOCATION", "us-central1");

const MISTRAL_PATH: &str = "/v1/projects/ccp-project/locations/us-central1/publishers/mistralai/models/mistral-medium-3:streamRawPredict";

/// The fixed message a client sees when a Mistral stream fails.
const MISTRAL_STREAM_INVALID: &str = "Vertex AI Mistral stream is invalid";

/// One `chat.completion.chunk` as Google's Mistral example shows it.
fn mistral_chunk(delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id":"synthetic-mistral","object":"chat.completion.chunk","model":"mistral-medium-3",
        "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]
    })
}

/// A Mistral stream answering `text`. Google does not show an end marker
/// for Mistral, so there is no `[DONE]`.
fn mistral_text(text: &str) -> Vec<Value> {
    let mut finish = mistral_chunk(json!({}), Some("stop"));
    finish["usage"] = json!({"prompt_tokens":7,"total_tokens":9,"completion_tokens":2});
    vec![
        mistral_chunk(json!({"role":"assistant","content":text}), None),
        finish,
    ]
}

#[tokio::test]
async fn vertex_mistral_synthetic_requests_reach_stream_raw_predict_and_complete_without_done() {
    let upstream = FakeVertex::start(vec![
        Answer::sse(&mistral_text("Hello")),
        Answer::sse(&mistral_text("Hello")),
    ])
    .await;
    let registry = registry(provider(&upstream, &[MISTRAL_LOCATION]));
    for stream in [false, true] {
        let request = json!({
            "model": MISTRAL_MODEL,
            "max_tokens": 256,
            "stream": stream,
            "output_config": {"effort": "high"},
            "tools": [read_file_tool()],
            "messages": [{"role":"user","content":"Say hello"}]
        });
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &request,
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        assert_text_answer(stream, &body, MISTRAL_MODEL, "Hello");
        if !stream {
            let message: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(message["usage"]["input_tokens"], 7, "{body}");
            assert_eq!(message["usage"]["output_tokens"], 2, "{body}");
        }
    }
    let requests = upstream.requests();
    assert_eq!(requests.len(), 2);
    for sent in &requests {
        let body = &sent.body;
        // The streaming method and flag for both client modes.
        assert_eq!(sent.path_and_query, MISTRAL_PATH);
        assert_eq!(sent.accept.as_deref(), Some("text/event-stream"));
        assert_eq!(body["stream"], true, "{body}");
        assert_eq!(body["model"], "mistral-medium-3", "{body}");
        assert_eq!(body["max_tokens"], 256, "{body}");
        for absent in ["stream_options", "reasoning_effort"] {
            assert!(body.get(absent).is_none(), "{absent} in {body}");
        }
        assert_eq!(body["messages"].as_array().map(Vec::len), Some(1), "{body}");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file", "{body}");
    }
}

#[tokio::test]
async fn vertex_mistral_synthetic_global_location_is_refused_naming_the_settings() {
    let upstream = FakeVertex::start(Vec::new()).await;
    // No location set: the default is global, where Mistral is not served.
    let registry = registry(provider(&upstream, &[]));
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(MISTRAL_MODEL, stream),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "stream={stream}: {body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error", "{body}");
        let message = error_message(&body);
        for needle in [
            "mistralai",
            "global",
            "CCP_VERTEX_LOCATION_OVERRIDES",
            "vertex.locationOverrides",
            "CCP_VERTEX_LOCATION",
            "us-central1",
            "europe-west4",
        ] {
            assert!(message.contains(needle), "missing {needle:?}: {message}");
        }
    }
    assert!(upstream.requests().is_empty());
}

#[tokio::test]
async fn vertex_mistral_synthetic_content_after_the_finish_reason_is_an_error() {
    let mut frames = mistral_text("Hello");
    frames.push(mistral_chunk(json!({"content":"late"}), None));
    let upstream = FakeVertex::start(vec![Answer::sse(&frames), Answer::sse(&frames)]).await;
    let registry = registry(provider(&upstream, &[MISTRAL_LOCATION]));
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(MISTRAL_MODEL, stream),
            &[],
        )
        .await;
        if stream {
            assert_stream_failed_with(status, &body, MISTRAL_STREAM_INVALID);
        } else {
            assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], "api_error", "{body}");
            assert!(
                error_message(&body).contains("content after finish_reason"),
                "{body}"
            );
        }
    }
    assert_eq!(upstream.requests().len(), 2);
}

// Model Garden's managed open models.

/// The six managed open models of Google's open-model catalog: the name the
/// client asks for, where Vertex AI takes the request for project
/// `ccp-project`, and the `model` value of the request body. The general
/// open-model guide calls `v1` in `global`; the Llama 4 page calls
/// `v1beta1` in `us-east5`.
const OPEN_MODELS: [(&str, &str, &str); 6] = [
    (
        "vertex/openai/gpt-oss-120b-maas",
        "/v1/projects/ccp-project/locations/global/endpoints/openapi/chat/completions",
        "openai/gpt-oss-120b-maas",
    ),
    (
        "vertex/zai-org/glm-5.2-maas",
        "/v1/projects/ccp-project/locations/global/endpoints/openapi/chat/completions",
        "zai-org/glm-5.2-maas",
    ),
    (
        "vertex/google/gemma-4-26b-a4b-it-maas",
        "/v1/projects/ccp-project/locations/global/endpoints/openapi/chat/completions",
        "google/gemma-4-26b-a4b-it-maas",
    ),
    (
        "vertex/meta/llama-4-maverick-17b-128e-instruct-maas",
        "/v1beta1/projects/ccp-project/locations/us-east5/endpoints/openapi/chat/completions",
        "meta/llama-4-maverick-17b-128e-instruct-maas",
    ),
    (
        "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
        "/v1beta1/projects/ccp-project/locations/us-east5/endpoints/openapi/chat/completions",
        "meta/llama-4-scout-17b-16e-instruct-maas",
    ),
    (
        "vertex/deepseek-ai/deepseek-v4.1-flash-maas",
        "/v1/projects/ccp-project/locations/global/endpoints/openapi/chat/completions",
        "deepseek-ai/deepseek-v4.1-flash-maas",
    ),
];

/// The fixed message a client sees when an open-model stream fails.
const OPEN_MODEL_STREAM_INVALID: &str = "Vertex AI open model stream is invalid";

/// One `chat.completion.chunk` as Google's open-model examples show it.
fn open_model_chunk(delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id":"synthetic-open-model","object":"chat.completion.chunk","model":"zai-org/glm-5.2-maas",
        "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]
    })
}

/// The last chunk of Google's open-model streaming example: `usage` and no
/// choices.
fn open_model_usage_chunk() -> Value {
    json!({
        "id":"synthetic-open-model","object":"chat.completion.chunk","model":"zai-org/glm-5.2-maas",
        "choices":[],
        "usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}
    })
}

/// An open-model stream answering `text`: the text, the finish reason, the
/// usage chunk and `data: [DONE]`.
fn open_model_text(text: &str) -> Vec<Value> {
    vec![
        open_model_chunk(json!({"role":"assistant","content":text}), None),
        open_model_chunk(json!({}), Some("stop")),
        open_model_usage_chunk(),
    ]
}

/// A 200 event stream with one `data:` frame per value, then
/// `data: [DONE]`, closed after it.
fn open_model_sse(frames: &[Value]) -> Answer {
    let mut chunks: Vec<String> = frames
        .iter()
        .map(|frame| format!("data: {frame}\r\n\r\n"))
        .collect();
    chunks.push("data: [DONE]\r\n\r\n".to_string());
    Answer::raw(200, "text/event-stream", chunks)
}

/// The request every open-model test sends: a system prompt, effort `max`,
/// adaptive thinking and one tool.
fn open_model_request(model: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "max_tokens": 256,
        "stream": stream,
        "system": "Be brief.",
        "output_config": {"effort": "max"},
        "thinking": {"type": "adaptive"},
        "tools": [read_file_tool()],
        "messages": [{"role":"user","content":"Say hello"}]
    })
}

/// The body every open model gets for [`open_model_request`], before the
/// model's own thinking field.
fn open_model_expected_body(upstream_model: &str) -> Value {
    json!({
        "model": upstream_model,
        "messages": [
            {"role":"system","content":"Be brief."},
            {"role":"user","content":"Say hello"}
        ],
        "tools": [{"type":"function","function":{
            "name":"read_file",
            "description":"Read a file",
            "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
        }}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 256
    })
}

#[tokio::test]
async fn vertex_open_models_synthetic_each_row_reaches_its_documented_url_with_its_body() {
    let upstream = FakeVertex::start(
        OPEN_MODELS
            .iter()
            .flat_map(|_| {
                [
                    open_model_sse(&open_model_text("Hello")),
                    open_model_sse(&open_model_text("Hello")),
                ]
            })
            .collect(),
    )
    .await;
    let registry = registry(provider(&upstream, &[]));
    for (model, ..) in OPEN_MODELS {
        for stream in [false, true] {
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &open_model_request(model, stream),
                &[],
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{model} stream={stream}: {body}");
            assert_text_answer(stream, &body, model, "Hello");
        }
    }
    let requests = upstream.requests();
    assert_eq!(requests.len(), OPEN_MODELS.len() * 2);
    for (sent, (model, path, upstream_model)) in requests
        .iter()
        .zip(OPEN_MODELS.iter().flat_map(|row| [row, row]))
    {
        assert_eq!(sent.path_and_query, *path, "{model}");
        assert_eq!(sent.accept.as_deref(), Some("text/event-stream"), "{model}");
        assert_eq!(
            sent.authorization.as_deref(),
            Some(format!("Bearer {SYNTHETIC_TOKEN}").as_str()),
            "{model}"
        );
        // The thinking field each model's documentation gives: effort for
        // gpt-oss (max becomes high), the chat-template switch for GLM and
        // Gemma, nothing for Llama and DeepSeek.
        let mut expected = open_model_expected_body(upstream_model);
        match *upstream_model {
            "openai/gpt-oss-120b-maas" => expected["reasoning_effort"] = json!("high"),
            "zai-org/glm-5.2-maas" | "google/gemma-4-26b-a4b-it-maas" => {
                expected["chat_template_kwargs"] = json!({"enable_thinking": true});
            }
            _ => {}
        }
        assert_eq!(sent.body, expected, "{model}");
    }
}

#[tokio::test]
async fn vertex_open_models_synthetic_glm_thinking_off_unless_claude_code_turns_it_on() {
    let upstream = FakeVertex::start(vec![
        open_model_sse(&open_model_text("Hello")),
        open_model_sse(&open_model_text("Hello")),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    for thinking in [json!({"type": "disabled"}), Value::Null] {
        let mut request = message_request("vertex/zai-org/glm-5.2-maas", false);
        if !thinking.is_null() {
            request["thinking"] = thinking;
        }
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &request,
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    for sent in upstream.requests() {
        assert_eq!(
            sent.body["chat_template_kwargs"],
            json!({"enable_thinking": false}),
            "{}",
            sent.body
        );
        assert!(sent.body.get("reasoning_effort").is_none(), "{}", sent.body);
    }
}

#[tokio::test]
async fn vertex_open_models_synthetic_reasoning_effort_on_the_chat_completions_route_turns_glm_thinking_on()
 {
    // The OpenAI-compatible route sets an effort and never a thinking
    // setting, so the effort alone decides enable_thinking for GLM.
    let upstream = FakeVertex::start(vec![
        open_model_sse(&open_model_text("Hello")),
        open_model_sse(&open_model_text("Hello")),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    for (effort, enabled) in [(Some("high"), true), (None, false)] {
        let mut request = json!({
            "model": "vertex/zai-org/glm-5.2-maas",
            "messages": [{"role": "user", "content": "hello"}]
        });
        if let Some(effort) = effort {
            request["reasoning_effort"] = json!(effort);
        }
        let (status, _, body) = tokio::time::timeout(
            REQUEST_DEADLINE,
            post(
                app_with_options(registry.clone(), None, true),
                "/v1/chat/completions",
                &request,
            ),
        )
        .await
        .unwrap_or_else(|_| panic!("effort {effort:?}: no answer from /v1/chat/completions"));
        assert_eq!(status, StatusCode::OK, "effort {effort:?}: {body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            value["choices"][0]["message"]["content"], "Hello",
            "effort {effort:?}: {body}"
        );
        let sent = upstream.requests().last().cloned().unwrap();
        assert_eq!(
            sent.body["chat_template_kwargs"],
            json!({"enable_thinking": enabled}),
            "effort {effort:?}: {}",
            sent.body
        );
        assert!(sent.body.get("reasoning_effort").is_none(), "{}", sent.body);
    }
    assert_eq!(upstream.requests().len(), 2);
}

#[tokio::test]
async fn vertex_open_models_synthetic_reasoning_becomes_thinking_in_both_modes() {
    let frames = vec![
        open_model_chunk(
            json!({"role":"assistant","reasoning_content":"Thinking it over"}),
            None,
        ),
        open_model_chunk(json!({"content":"Hello"}), None),
        open_model_chunk(json!({}), Some("stop")),
        open_model_usage_chunk(),
    ];
    let upstream = FakeVertex::start(vec![open_model_sse(&frames), open_model_sse(&frames)]).await;
    let registry = registry(provider(&upstream, &[]));
    let model = "vertex/zai-org/glm-5.2-maas";
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &open_model_request(model, stream),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        let content = if stream {
            let events = sse_events(&body);
            assert_eq!(
                events.last().map(|(name, _)| name.as_str()),
                Some("message_stop"),
                "{body}"
            );
            assert_eq!(stop_reason(&events).as_deref(), Some("end_turn"), "{body}");
            let usage = events
                .iter()
                .find(|(name, _)| name == "message_delta")
                .map(|(_, data)| data["usage"].clone())
                .unwrap_or_default();
            assert_eq!(usage["output_tokens"], 3, "{body}");
            streamed_content(&events)
        } else {
            let message: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(message["stop_reason"], "end_turn", "{body}");
            assert_eq!(message["usage"]["input_tokens"], 11, "{body}");
            assert_eq!(message["usage"]["output_tokens"], 3, "{body}");
            message["content"].as_array().cloned().unwrap_or_default()
        };
        assert_eq!(content.len(), 2, "{body}");
        assert_eq!(content[0]["type"], "thinking", "{body}");
        assert_eq!(content[0]["thinking"], "Thinking it over", "{body}");
        assert_eq!(content[1], json!({"type":"text","text":"Hello"}), "{body}");
    }
}

#[tokio::test]
async fn vertex_open_models_synthetic_forced_tool_choice_on_gpt_oss_is_refused_without_a_request() {
    let upstream = FakeVertex::start(Vec::new()).await;
    let registry = registry(provider(&upstream, &[]));
    for tool_choice in [
        json!({"type":"any"}),
        json!({"type":"tool","name":"read_file"}),
    ] {
        for stream in [false, true] {
            let mut request = open_model_request("vertex/openai/gpt-oss-120b-maas", stream);
            request["tool_choice"] = tool_choice.clone();
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &request,
                &[],
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{tool_choice}: {body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error", "{body}");
            assert_eq!(
                error_message(&body),
                "Vertex AI model openai/gpt-oss-120b-maas does not support a forced tool choice (tool_choice any or tool); use tool_choice auto or none"
            );
        }
    }
    assert!(upstream.requests().is_empty());
}

#[tokio::test]
async fn vertex_open_models_synthetic_glm_takes_every_tool_choice() {
    let upstream = FakeVertex::start(vec![
        open_model_sse(&open_model_text("Hello")),
        open_model_sse(&open_model_text("Hello")),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    for tool_choice in [
        json!({"type":"any"}),
        json!({"type":"tool","name":"read_file"}),
    ] {
        let mut request = open_model_request("vertex/zai-org/glm-5.2-maas", false);
        request["tool_choice"] = tool_choice;
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &request,
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let requests = upstream.requests();
    assert_eq!(requests[0].body["tool_choice"], "required");
    assert_eq!(
        requests[1].body["tool_choice"],
        json!({"type":"function","function":{"name":"read_file"}})
    );
}

#[tokio::test]
async fn vertex_open_models_synthetic_content_after_the_finish_reason_is_an_error() {
    let mut frames = open_model_text("Hello");
    frames.insert(2, open_model_chunk(json!({"content":"late"}), None));
    let upstream = FakeVertex::start(vec![open_model_sse(&frames), open_model_sse(&frames)]).await;
    let registry = registry(provider(&upstream, &[]));
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request("vertex/deepseek-ai/deepseek-v4.1-flash-maas", stream),
            &[],
        )
        .await;
        if stream {
            assert_stream_failed_with(status, &body, OPEN_MODEL_STREAM_INVALID);
        } else {
            assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], "api_error", "{body}");
            assert_eq!(
                error_message(&body),
                "Vertex AI open model stream sent content after finish_reason"
            );
        }
    }
    assert_eq!(upstream.requests().len(), 2);
}

#[tokio::test]
async fn vertex_chat_completions_synthetic_upstream_error_with_the_upstream_left_open_answers_at_once()
 {
    // A non-streaming request whose upstream sends an error event and then
    // keeps the connection open. The streaming path fails at that event; the
    // buffered path must too, instead of waiting for a terminal event that
    // never comes. The stand-in holds back a finishing chunk behind a gate
    // that opens only after the answer has arrived, and never closes the
    // connection, so an answer at all proves the proxy did not wait for the
    // upstream to finish or close. The watchdog only catches a hang; it is
    // not a bound on how fast a slow host must answer.
    let cases = [
        (
            "vertex/zai-org/glm-5.2-maas",
            &[][..],
            "Vertex AI open model upstream error: overloaded",
        ),
        (
            MISTRAL_MODEL,
            &[MISTRAL_LOCATION][..],
            "Vertex AI Mistral upstream error: overloaded",
        ),
    ];
    for (model, extra_env, expected) in cases {
        let release = Arc::new(AtomicBool::new(false));
        let upstream = FakeVertex::start(vec![Answer {
            hold_open: true,
            ..Answer::sse_gated(
                &[
                    json!({"error":{"message":"overloaded"}}),
                    json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
                ],
                1,
                release.clone(),
            )
        }])
        .await;
        let app = app_with_monitor(registry(provider(&upstream, extra_env)), None);
        let (status, _, body) = tokio::time::timeout(
            Duration::from_secs(20),
            post(app, "/v1/messages", &message_request(model, false)),
        )
        .await
        .unwrap_or_else(|_| panic!("{model}: the answer hung behind an upstream left open"));
        // The gate was still shut when the answer came back.
        assert!(!release.swap(true, Ordering::SeqCst), "{model}");
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{model}: {body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["type"], "api_error", "{model}: {body}");
        assert_eq!(error_message(&body), expected, "{model}");
        assert_eq!(upstream.requests().len(), 1, "{model}");
    }
}

#[tokio::test]
async fn vertex_open_models_synthetic_404_names_model_garden_with_the_model_project_and_location() {
    let not_found = google_error(404, "NOT_FOUND", "Publisher Model was not found.");
    let upstream = FakeVertex::start(
        (0..4)
            .map(|_| Answer::error(404, not_found.clone()))
            .collect(),
    )
    .await;
    let registry = registry(provider(&upstream, &[]));
    for (model, upstream_model, location) in [
        (
            "vertex/openai/gpt-oss-120b-maas",
            "openai/gpt-oss-120b-maas",
            "global",
        ),
        (
            "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
            "meta/llama-4-scout-17b-16e-instruct-maas",
            "us-east5",
        ),
    ] {
        for stream in [false, true] {
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(model, stream),
                &[],
            )
            .await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{model}: {body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error", "{body}");
            let message = error_message(&body);
            let expected = format!(
                ". Also check that model {upstream_model} is enabled in Model Garden for project ccp-project in location {location} and, for a preview model, that the project has been granted access; the status alone does not say which of these is the cause (upstream HTTP 404)"
            );
            assert!(message.ends_with(&expected), "{model}: {message}");
            assert!(message.contains("Check the model ID"), "{model}: {message}");
        }
    }
    assert_eq!(upstream.requests().len(), 4);
}

#[tokio::test]
async fn vertex_open_models_synthetic_llama_location_moves_only_with_a_model_key() {
    let upstream = FakeVertex::start(vec![
        open_model_sse(&open_model_text("maverick")),
        open_model_sse(&open_model_text("scout")),
        open_model_sse(&open_model_text("glm")),
        open_model_sse(&open_model_text("unlisted")),
    ])
    .await;
    let registry = registry(provider(
        &upstream,
        &[
            ("CCP_VERTEX_LOCATION", "us-central1"),
            (
                "CCP_VERTEX_LOCATION_OVERRIDES",
                "meta=us, meta/llama-4-maverick-17b-128e-instruct-maas=europe-west4",
            ),
        ],
    ));
    let cases = [
        (
            "vertex/meta/llama-4-maverick-17b-128e-instruct-maas",
            "/v1beta1/projects/ccp-project/locations/europe-west4/endpoints/openapi/chat/completions",
        ),
        (
            "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
            "/v1beta1/projects/ccp-project/locations/us-east5/endpoints/openapi/chat/completions",
        ),
        (
            "vertex/zai-org/glm-5.2-maas",
            "/v1/projects/ccp-project/locations/us-central1/endpoints/openapi/chat/completions",
        ),
        (
            "vertex/meta/llama-synthetic-5-maas",
            "/v1/projects/ccp-project/locations/us/endpoints/openapi/chat/completions",
        ),
    ];
    for (model, _) in &cases {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(model, false),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
    }
    let requests = upstream.requests();
    assert_eq!(requests.len(), cases.len());
    for (request, (model, path)) in requests.iter().zip(&cases) {
        assert_eq!(request.path_and_query, *path, "{model}");
    }
    assert_eq!(
        requests[3].body["model"], "meta/llama-synthetic-5-maas",
        "{}",
        requests[3].body
    );
}

#[tokio::test]
async fn vertex_open_models_synthetic_echoed_secrets_are_redacted_in_errors() {
    let echo = format!("saw token {SYNTHETIC_TOKEN} for synthetic.person@example.com");
    let upstream = FakeVertex::start(vec![
        Answer::error(
            400,
            json!({"object":"error","message":echo,"type":"invalid_request_error","code":null}),
        ),
        open_model_sse(&[json!({"error":{"message":echo}})]),
        open_model_sse(&[
            open_model_chunk(json!({"role":"assistant","content":"partial"}), None),
            json!({"error":{"message":echo}}),
        ]),
    ])
    .await;
    let registry = registry(provider(&upstream, &[]));
    let monitor = MonitorHandle::new(20);
    let mut downstream = String::new();
    for stream in [false, false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            &message_request("vertex/zai-org/glm-5.2-maas", stream),
            &[],
        )
        .await;
        if stream {
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(
                sse_events(&body).last().map(|(name, _)| name.as_str()),
                Some("error"),
                "{body}"
            );
        } else {
            assert_ne!(status, StatusCode::OK, "{body}");
            assert!(error_message(&body).contains("[redacted]"), "{body}");
        }
        downstream.push_str(&body);
    }
    assert_eq!(upstream.requests().len(), 3);
    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), 3);
    let mut monitored = String::new();
    for request in &state.recent {
        assert_eq!(request.status, RequestStatus::Failed, "{request:?}");
        monitored.push_str(request.error.as_deref().unwrap_or_default());
        monitored.push('\n');
    }
    for (place, text) in [("responses", &downstream), ("monitor errors", &monitored)] {
        for secret in [SYNTHETIC_TOKEN, "synthetic.person@example.com"] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
}

#[tokio::test]
async fn vertex_open_models_synthetic_count_tokens_is_answered_locally() {
    let upstream = FakeVertex::start(Vec::new()).await;
    let registry = registry(provider(&upstream, &[]));
    for (model, ..) in OPEN_MODELS {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages/count_tokens",
            &message_request(model, false),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert!(value["input_tokens"].as_u64().is_some(), "{body}");
    }
    assert!(upstream.requests().is_empty());
}

// Every family together.

/// Every name the provider advertises, as Google's model pages spell them.
const VERTEX_NAMES: [&str; 53] = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-4-argon",
    "gemini-4-carbon",
    "vertex/gemini-3-flash-preview",
    "vertex/gemini-3.1-flash-lite",
    "vertex/gemini-3.1-pro-preview",
    "vertex/gemini-3.5-flash",
    "vertex/gemini-3.5-flash-lite",
    "vertex/gemini-3.7-flash",
    "vertex/gemini-3.8-flash",
    "vertex/gemini-4-argon",
    "vertex/gemini-4-carbon",
    "vertex/anthropic/claude-fable-5-2",
    "vertex/anthropic/claude-fable-5-1",
    "vertex/anthropic/claude-fable-5",
    "vertex/anthropic/claude-opus-5-5",
    "vertex/anthropic/claude-opus-5",
    "vertex/anthropic/claude-opus-4-8",
    "vertex/anthropic/claude-opus-4-7",
    "vertex/anthropic/claude-sonnet-5-5",
    "vertex/anthropic/claude-sonnet-5",
    "vertex/anthropic/claude-haiku-5-5",
    "vertex/anthropic/claude-opus-4-6",
    "vertex/anthropic/claude-sonnet-4-6",
    "vertex/anthropic/claude-opus-4-5@20251101",
    "vertex/anthropic/claude-haiku-4-5@20251001",
    "vertex/xai/grok-4.7",
    "vertex/xai/grok-4.6",
    "vertex/xai/grok-4.20-reasoning",
    "vertex/xai/grok-4.20-non-reasoning",
    "vertex/xai/grok-4.3",
    "vertex/meta/muse-spark-1.3",
    "vertex/meta/muse-wonder-1.0",
    "vertex/meta/muse-ember-1.0",
    "vertex/openai/gpt-6-astra",
    "vertex/openai/gpt-6.1-sol",
    "vertex/openai/gpt-6.1-terra",
    "vertex/openai/gpt-6-luna",
    "vertex/mistralai/mistral-medium-3",
    "vertex/mistralai/mistral-small-2503",
    "vertex/mistralai/codestral-2",
    "vertex/openai/gpt-oss-120b-maas",
    "vertex/zai-org/glm-5.2-maas",
    "vertex/google/gemma-4-26b-a4b-it-maas",
    "vertex/meta/llama-4-maverick-17b-128e-instruct-maas",
    "vertex/meta/llama-4-scout-17b-16e-instruct-maas",
    "vertex/deepseek-ai/deepseek-v4.1-flash-maas",
];

#[tokio::test]
async fn vertex_models_route_lists_every_name_of_every_served_family() {
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
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let mut listed: Vec<&str> = value["data"]
        .as_array()
        .expect("a model list")
        .iter()
        .map(|model| model["id"].as_str().expect("a model ID"))
        .collect();
    listed.sort_unstable();
    let mut expected = VERTEX_NAMES.to_vec();
    expected.sort_unstable();
    assert_eq!(listed, expected);
    assert!(upstream.requests().is_empty());
}

#[test]
fn vertex_partner_names_route_to_vertex_and_bare_ids_stay_with_their_providers() {
    let registry = Registry::new(AliasProvider::Codex);
    let owner = |model: &str| {
        registry
            .provider_for_model(model, None)
            .map(|provider| provider.name())
    };
    for model in VERTEX_NAMES {
        assert_eq!(owner(model), Some("vertex"), "{model}");
    }
    // The partner IDs without the prefix belong to the providers that
    // already serve them, or to nobody; never to Vertex.
    for model in VERTEX_NAMES
        .iter()
        .filter_map(|name| name.strip_prefix("vertex/"))
        .filter(|name| name.contains('/'))
    {
        let (publisher, id) = model.split_once('/').unwrap();
        for bare in [model, id] {
            assert_ne!(owner(bare), Some("vertex"), "{publisher}: {bare}");
        }
    }
    for (model, provider) in [
        ("claude-opus-5-5", "codex"),
        ("claude-sonnet-5", "codex"),
        ("grok-4.7", "grok"),
        ("muse-spark-1.3", "muse"),
    ] {
        assert_eq!(owner(model), Some(provider), "{model}");
    }
}

#[tokio::test]
async fn vertex_location_overrides_synthetic_model_key_beats_publisher_key_and_muse_stays_global() {
    let upstream = FakeVertex::start(vec![
        claude_json(claude_message("opus")),
        claude_json(claude_message("sonnet")),
        Answer::sse(&responses_text("muse")),
        Answer::sse(&mistral_text("codestral")),
        Answer::sse(&mistral_text("medium")),
    ])
    .await;
    let registry = registry(provider(
        &upstream,
        &[(
            "CCP_VERTEX_LOCATION_OVERRIDES",
            "anthropic=us, anthropic/claude-opus-5-5=eu, meta=us, mistralai=us-central1, mistralai/codestral-2=europe-west4",
        )],
    ));
    let cases = [
        (
            "vertex/anthropic/claude-opus-5-5",
            "/v1/projects/ccp-project/locations/eu/publishers/anthropic/models/claude-opus-5-5:rawPredict",
        ),
        (
            "vertex/anthropic/claude-sonnet-5",
            "/v1/projects/ccp-project/locations/us/publishers/anthropic/models/claude-sonnet-5:rawPredict",
        ),
        (
            "vertex/meta/muse-spark-1.3",
            "/v1/projects/ccp-project/locations/global/publishers/meta/v1/responses",
        ),
        (
            "vertex/mistralai/codestral-2",
            "/v1/projects/ccp-project/locations/europe-west4/publishers/mistralai/models/codestral-2:streamRawPredict",
        ),
        (
            "vertex/mistralai/mistral-medium-3",
            "/v1/projects/ccp-project/locations/us-central1/publishers/mistralai/models/mistral-medium-3:streamRawPredict",
        ),
    ];
    for (model, _) in &cases {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(model, false),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
    }
    let requests = upstream.requests();
    assert_eq!(requests.len(), cases.len());
    for (request, (model, path)) in requests.iter().zip(&cases) {
        assert_eq!(request.path_and_query, *path, "{model}");
    }
}

#[tokio::test]
async fn vertex_partner_synthetic_echoed_secrets_are_redacted_in_responses_and_the_monitor() {
    let echo = format!("saw token {SYNTHETIC_TOKEN} for synthetic.person@example.com");
    let echo = echo.as_str();
    let upstream = FakeVertex::start(vec![
        // Claude: an Anthropic error body, and an error event in a stream.
        Answer::error(
            400,
            json!({"type":"error","error":{"type":"invalid_request_error","message":echo}}),
        ),
        claude_sse(vec![
            claude_stream("unused").remove(0),
            claude_event(&json!({"type":"error","error":{"type":"api_error","message":echo}})),
        ]),
        // Grok: an OpenAI error body, and a failed response in both modes.
        Answer::error(
            400,
            json!({"error":{"message":echo,"type":"invalid_request_error","code":null}}),
        ),
        Answer::sse(&[json!({"type":"response.failed","response":{
            "status":"failed","error":{"code":"server_error","message":echo}
        }})]),
        Answer::sse(&[
            json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
            json!({"type":"response.failed","response":{
                "status":"failed","error":{"code":"server_error","message":echo}
            }}),
        ]),
        // Mistral: its own error body, which has no `error` object, and an
        // error chunk in both modes.
        Answer::error(
            400,
            json!({"object":"error","message":echo,"type":"invalid_request_error","code":null}),
        ),
        Answer::sse(&[json!({"error":{"message":echo}})]),
        Answer::sse(&[
            mistral_chunk(json!({"role":"assistant","content":"partial"}), None),
            json!({"error":{"message":echo}}),
        ]),
    ])
    .await;
    let registry = registry(provider(
        &upstream,
        &[("CCP_VERTEX_LOCATION_OVERRIDES", "mistralai=us-central1")],
    ));
    let monitor = MonitorHandle::new(20);
    let requests = [
        (CLAUDE_MODEL, false),
        (CLAUDE_MODEL, true),
        ("vertex/xai/grok-4.7", false),
        ("vertex/xai/grok-4.7", false),
        ("vertex/xai/grok-4.7", true),
        (MISTRAL_MODEL, false),
        (MISTRAL_MODEL, false),
        (MISTRAL_MODEL, true),
    ];
    let mut downstream = String::new();
    for (model, stream) in requests {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            &message_request(model, stream),
            &[],
        )
        .await;
        if stream {
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            let events = sse_events(&body);
            assert_eq!(
                events.last().map(|(name, _)| name.as_str()),
                Some("error"),
                "{model}: {body}"
            );
        } else {
            assert_ne!(status, StatusCode::OK, "{model}: {body}");
            // The upstream message is kept, with the secrets taken out.
            assert!(
                error_message(&body).contains("[redacted]"),
                "{model}: {body}"
            );
        }
        downstream.push_str(&body);
    }
    assert_eq!(upstream.requests().len(), requests.len());

    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), requests.len());
    let mut monitored = String::new();
    for request in &state.recent {
        assert_eq!(request.status, RequestStatus::Failed, "{request:?}");
        monitored.push_str(request.error.as_deref().unwrap_or_default());
        monitored.push('\n');
    }
    let snapshot = serde_json::to_string(&MonitorSnapshot::from(state.clone()))
        .expect("the monitor snapshot serialises");
    for (place, text) in [
        ("responses", &downstream),
        ("monitor errors", &monitored),
        ("monitor snapshot", &snapshot),
    ] {
        for secret in [SYNTHETIC_TOKEN, "synthetic.person@example.com"] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
}

// Partner errors, buffered settling, refusals and framing.

/// Google's error body for `status`.
fn google_error(code: u16, status: &str, message: &str) -> Value {
    json!({"error":{"code":code,"message":message,"status":status}})
}

#[tokio::test]
async fn vertex_partner_synthetic_403_and_404_name_model_garden_with_the_model_project_and_location()
 {
    // One family per wire, with the upstream model value and the location
    // each request goes to.
    let families = [
        (CLAUDE_MODEL, "claude-opus-5-5", "global", vec![]),
        ("vertex/xai/grok-4.7", "xai/grok-4.7", "global", vec![]),
        (
            MISTRAL_MODEL,
            "mistral-medium-3",
            "us-central1",
            vec![MISTRAL_LOCATION],
        ),
    ];
    let errors = [
        (
            403,
            google_error(
                403,
                "PERMISSION_DENIED",
                "Permission denied on resource project ccp-project.",
            ),
            StatusCode::FORBIDDEN,
            "permission_error",
        ),
        (
            404,
            google_error(404, "NOT_FOUND", "Publisher Model was not found."),
            StatusCode::NOT_FOUND,
            "invalid_request_error",
        ),
    ];
    for (model, upstream_model, location, env) in &families {
        let answers = errors
            .iter()
            .flat_map(|(code, body, ..)| {
                [
                    Answer::error(*code, body.clone()),
                    Answer::error(*code, body.clone()),
                ]
            })
            .collect();
        let upstream = FakeVertex::start(answers).await;
        let registry = registry(provider(&upstream, env));
        for (code, _, status, kind) in &errors {
            for stream in [false, true] {
                let (actual, _, body) = send(
                    app_with_monitor(registry.clone(), None),
                    "/v1/messages",
                    &message_request(model, stream),
                    &[],
                )
                .await;
                assert_eq!(actual, *status, "{model} stream={stream}: {body}");
                let value: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(value["error"]["type"], *kind, "{model}: {body}");
                let message = error_message(&body);
                let expected = format!(
                    ". Also check that model {upstream_model} is enabled in Model Garden for project ccp-project in location {location} and, for a preview model, that the project has been granted access; the status alone does not say which of these is the cause (upstream HTTP {code})"
                );
                assert!(message.ends_with(&expected), "{model} {code}: {message}");
                // The guidance every Vertex 403 and 404 already gave stays.
                let general = if *code == 403 {
                    "roles/aiplatform.user"
                } else {
                    "Check the model ID"
                };
                assert!(message.contains(general), "{model} {code}: {message}");
            }
        }
        assert_eq!(upstream.requests().len(), 4, "{model}");
    }
}

#[tokio::test]
async fn vertex_partner_synthetic_400_keeps_its_message_without_model_garden_guidance() {
    // Mistral's own error shape carries the message at the top level.
    let upstream = FakeVertex::start(vec![Answer::error(
        400,
        json!({
            "object":"error",
            "message":"Invalid model: mistral-medium-3",
            "type":"invalid_request_error",
            "param":null,
            "code":"1500"
        }),
    )])
    .await;
    let (status, _, body) = send(
        app_with_monitor(registry(provider(&upstream, &[MISTRAL_LOCATION])), None),
        "/v1/messages",
        &message_request(MISTRAL_MODEL, false),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(
        error_message(&body),
        "Invalid model: mistral-medium-3 (upstream HTTP 400)"
    );
}

#[tokio::test]
async fn vertex_claude_synthetic_non_streaming_error_object_in_a_200_is_an_error_response() {
    let cases = [
        (
            "overloaded_error",
            "Overloaded",
            StatusCode::BAD_GATEWAY,
            "api_error",
        ),
        (
            "rate_limit_error",
            "Number of request tokens has exceeded your rate limit",
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
        ),
        (
            "invalid_request_error",
            "max_tokens: field required",
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
    ];
    let upstream = FakeVertex::start(
        cases
            .iter()
            .map(|(kind, message, ..)| {
                claude_json(
                    json!({"type":"error","error":{"type":kind,"message":message}}).to_string(),
                )
            })
            .collect(),
    )
    .await;
    let registry = registry(provider(&upstream, &[]));
    let monitor = MonitorHandle::new(10);
    for (upstream_kind, upstream_message, status, kind) in &cases {
        let (actual, _, body) = send(
            app_with_monitor(registry.clone(), Some(monitor.clone())),
            "/v1/messages",
            &message_request(CLAUDE_MODEL, false),
            &[],
        )
        .await;
        assert_eq!(actual, *status, "{upstream_kind}: {body}");
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["type"], "error", "{body}");
        assert_eq!(value["error"]["type"], *kind, "{body}");
        assert_eq!(
            error_message(&body),
            format!("Vertex AI returned a Claude {upstream_kind}: {upstream_message}")
        );
    }
    assert_eq!(upstream.requests().len(), cases.len());
    let state = monitor.snapshot();
    assert_eq!(state.recent.len(), cases.len());
    for request in &state.recent {
        assert_eq!(request.status, RequestStatus::Failed, "{request:?}");
    }
}

#[tokio::test]
async fn vertex_claude_synthetic_crlf_framed_stream_passes_through_byte_for_byte() {
    let events: Vec<String> = claude_stream("Hello over CRLF")
        .iter()
        .map(|event| event.replace('\n', "\r\n"))
        .collect();
    let whole = events.concat();
    // The same bytes again, cut between the CR and the LF of every line end.
    let mut split = Vec::new();
    let mut rest = whole.as_str();
    while let Some(position) = rest.find("\r\n") {
        split.push(rest[..=position].to_string());
        rest = &rest[position + 1..];
    }
    split.push(rest.to_string());
    split.retain(|chunk| !chunk.is_empty());
    assert_eq!(split.concat(), whole);
    let upstream = FakeVertex::start(vec![claude_sse(events), claude_sse(split)]).await;
    let registry = registry(provider(&upstream, &[]));
    for case in ["one event per chunk", "cut between CR and LF"] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(CLAUDE_MODEL, true),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{case}: {body}");
        assert_eq!(body, whole, "{case}");
    }
}

/// Claude Code's file-reading tool. Its calls are the ones the Responses
/// translator holds and repairs.
fn read_tool() -> Value {
    json!({
        "name":"Read",
        "description":"Reads a file from the local filesystem.",
        "input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}
    })
}

/// A `Read` call whose arguments are complete JSON followed by 1,024
/// spaces, the stall the translator repairs into a finished call. No
/// completion follows.
fn responses_stalled_read_call() -> Vec<Value> {
    vec![
        json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{
            "type":"function_call","id":"fc_up","call_id":"call_read_1","name":"Read","arguments":""
        }}),
        json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_up",
            "delta":format!("{{\"file_path\":\"/tmp/a\"}}{}", " ".repeat(1024))}),
    ]
}

#[tokio::test]
async fn vertex_responses_synthetic_repaired_read_call_returns_while_the_upstream_stays_open() {
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(vec![
            Answer::sse_held_open(&responses_stalled_read_call()),
            Answer::sse_held_open(&responses_stalled_read_call()),
        ])
        .await;
        let registry = registry(provider(&upstream, family.env));
        for stream in [false, true] {
            let request = json!({
                "model": family.model,
                "max_tokens": 256,
                "stream": stream,
                "tools": [read_tool()],
                "messages": [{"role":"user","content":"Read /tmp/a"}]
            });
            let started = Instant::now();
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &request,
                &[],
            )
            .await;
            let elapsed = started.elapsed();
            assert_eq!(
                status,
                StatusCode::OK,
                "{} stream={stream}: {body}",
                family.model
            );
            // The repaired call is the end of the message: no completion is
            // awaited and no read window runs.
            assert!(
                elapsed < Duration::from_millis(1500),
                "{} stream={stream}: answered after {elapsed:?}",
                family.model
            );
            let (blocks, stop) = if stream {
                let events = sse_events(&body);
                assert_eq!(
                    events.last().map(|(name, _)| name.as_str()),
                    Some("message_stop"),
                    "{body}"
                );
                (streamed_content(&events), stop_reason(&events))
            } else {
                let message: Value = serde_json::from_str(&body).unwrap();
                (
                    message["content"].as_array().unwrap().clone(),
                    message["stop_reason"].as_str().map(str::to_string),
                )
            };
            assert_eq!(stop.as_deref(), Some("tool_use"), "{body}");
            let call = blocks
                .iter()
                .find(|block| block["type"] == "tool_use")
                .unwrap_or_else(|| panic!("no tool_use: {body}"));
            assert_eq!(call["name"], "Read", "{body}");
            assert_eq!(call["input"], json!({"file_path":"/tmp/a"}), "{body}");
        }
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_buffered_failure_or_unlisted_event_returns_while_the_upstream_stays_open()
 {
    let failed = vec![
        json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
        json!({"type":"response.failed","response":{
            "id":"resp_synthetic","status":"failed",
            "error":{"code":"server_error","message":"The server had an error while processing the request"}
        }}),
    ];
    let unlisted = vec![
        json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_up","role":"assistant","content":[]}}),
        json!({"type":"response.refusal.delta","output_index":0,"item_id":"msg_up","delta":"No."}),
    ];
    let cases = [
        (
            "upstream failure",
            failed,
            "The server had an error while processing the request",
        ),
        (
            "unlisted event",
            unlisted,
            "Vertex AI Responses stream is invalid: unlisted_event_type",
        ),
    ];
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(
            cases
                .iter()
                .map(|(_, frames, _)| Answer::sse_held_open(frames))
                .collect(),
        )
        .await;
        let registry = registry(provider(&upstream, family.env));
        for (case, _, needle) in &cases {
            let started = Instant::now();
            let (status, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(family.model, false),
                &[],
            )
            .await;
            let elapsed = started.elapsed();
            assert!(
                elapsed < Duration::from_millis(1500),
                "{} {case}: answered after {elapsed:?}",
                family.model
            );
            assert_eq!(
                status,
                StatusCode::BAD_GATEWAY,
                "{} {case}: {body}",
                family.model
            );
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], "api_error", "{body}");
            assert!(
                error_message(&body).contains(needle),
                "{} {case}: {body}",
                family.model
            );
        }
    }
}

#[tokio::test]
async fn vertex_responses_synthetic_snapshot_carrying_a_refusal_is_an_error() {
    let message_added = json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_up","role":"assistant","content":[]}});
    let refusal = json!({"type":"refusal","refusal":"I can't help with that."});
    let snapshots = [
        json!({"type":"response.content_part.added","output_index":0,"item_id":"msg_up","content_index":0,"part":refusal}),
        json!({"type":"response.content_part.done","output_index":0,"item_id":"msg_up","content_index":0,"part":refusal}),
        json!({"type":"response.output_item.done","output_index":0,"item":{
            "type":"message","id":"msg_up","role":"assistant","status":"completed","content":[refusal]
        }}),
    ];
    let wires: Vec<Vec<Value>> = snapshots
        .iter()
        .map(|snapshot| {
            vec![
                json!({"type":"response.created","response":{"id":"resp_synthetic","status":"in_progress"}}),
                message_added.clone(),
                snapshot.clone(),
                responses_completed(),
            ]
        })
        .collect();
    for family in &RESPONSES_FAMILIES {
        let upstream = FakeVertex::start(
            wires
                .iter()
                .flat_map(|wire| [Answer::sse(wire), Answer::sse(wire)])
                .collect(),
        )
        .await;
        let registry = registry(provider(&upstream, family.env));
        for wire in &wires {
            let shape = wire[2]["type"].as_str().unwrap();
            for stream in [false, true] {
                let (status, _, body) = send(
                    app_with_monitor(registry.clone(), None),
                    "/v1/messages",
                    &message_request(family.model, stream),
                    &[],
                )
                .await;
                if stream {
                    assert_stream_failed_with(status, &body, RESPONSES_STREAM_INVALID);
                } else {
                    assert_eq!(
                        status,
                        StatusCode::BAD_GATEWAY,
                        "{} {shape}: {body}",
                        family.model
                    );
                    assert!(
                        error_message(&body)
                            .contains("Vertex AI Responses stream is invalid: refusal_in_snapshot"),
                        "{} {shape}: {body}",
                        family.model
                    );
                }
                // The refusal is not passed off as an answer.
                assert!(!body.contains("I can't help with that."), "{shape}: {body}");
            }
        }
        assert_eq!(upstream.requests().len(), 6, "{}", family.model);
    }
}

// Deployed endpoints. The paths, the dedicated DNS form and the cold-start
// wording below are written out from Google's endpoint documentation, not
// taken from the provider's constants. The proxy never deploys anything:
// every endpoint here is one the test says the user already has.

const ENDPOINT_MODEL: &str = "vertex/endpoint/my-llm";

const ENDPOINT_CHAT_PATH: &str =
    "/v1/projects/ccp-project/locations/us-central1/endpoints/1234567890123456789/chat/completions";

const ENDPOINT_GET_PATH: &str =
    "/v1/projects/ccp-project/locations/us-central1/endpoints/1234567890123456789";

/// The `my-llm` entry of the documented example configuration, with
/// `extra` merged in.
fn endpoint_entry(extra: Value) -> Value {
    let mut entry = json!({
        "endpoint": "1234567890123456789",
        "location": "us-central1",
        "model": "gemma-2-9b-it"
    });
    if let (Some(entry), Some(extra)) = (entry.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            entry.insert(key.clone(), value.clone());
        }
    }
    entry
}

/// A provider whose Vertex base URL is `upstream` and whose
/// `vertex.endpoints` is `endpoints`.
fn endpoint_provider(upstream: &FakeVertex, endpoints: Value) -> VertexProvider {
    let env: HashMap<String, String> = HashMap::from([
        ("CCP_VERTEX_PROJECT".to_string(), "ccp-project".to_string()),
        ("CCP_VERTEX_BASE_URL".to_string(), upstream.base.clone()),
    ]);
    let config_dir = tempfile::tempdir().expect("temporary config directory");
    std::fs::write(
        config_dir.path().join("config.json"),
        json!({"vertex": {"endpoints": endpoints}}).to_string(),
    )
    .unwrap();
    let config = resolve_vertex_config(&env, config_dir.path());
    let endpoints = resolve_vertex_endpoints(config_dir.path());
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {SYNTHETIC_TOKEN}")).unwrap(),
    );
    let client = VertexClient::with_client(client_builder().no_proxy().build().unwrap());
    VertexProvider::with_client_and_endpoints(
        client,
        VertexCredentials::fixed(headers),
        config,
        endpoints,
    )
}

/// What `endpoints.get` answers for a dedicated endpoint whose
/// `dedicatedEndpointDns` is `dns`.
fn endpoint_metadata(fields: Value) -> Answer {
    let mut metadata = json!({
        "name": "projects/ccp-project/locations/us-central1/endpoints/1234567890123456789",
        "displayName": "synthetic-endpoint",
        "deployedModels": [{"id": "1", "displayName": "gemma-2-9b-it"}]
    });
    if let (Some(metadata), Some(fields)) = (metadata.as_object_mut(), fields.as_object()) {
        for (key, value) in fields {
            metadata.insert(key.clone(), value.clone());
        }
    }
    Answer::raw(200, "application/json", vec![metadata.to_string()])
}

fn endpoint_answer_text(stream: bool, body: &str) -> String {
    if stream {
        let events = sse_events(body);
        assert_eq!(
            events.last().map(|(name, _)| name.as_str()),
            Some("message_stop"),
            "{body}"
        );
        let content = streamed_content(&events);
        content[0]["text"].as_str().unwrap_or_default().to_string()
    } else {
        let message: Value = serde_json::from_str(body).unwrap();
        message["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }
}

#[tokio::test]
async fn vertex_endpoint_synthetic_configured_host_gets_the_request_and_endpoints_get_is_not_called()
 {
    let upstream = FakeVertex::start(vec![
        open_model_sse(&open_model_text("Hello")),
        open_model_sse(&open_model_text("Hello")),
    ])
    .await;
    let provider = endpoint_provider(
        &upstream,
        json!({"my-llm": endpoint_entry(json!({"host": upstream.base}))}),
    );
    let registry = registry(provider);
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &open_model_request(ENDPOINT_MODEL, stream),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        assert_eq!(endpoint_answer_text(stream, &body), "Hello", "{body}");
    }
    let requests = upstream.requests();
    assert_eq!(requests.len(), 2, "{requests:?}");
    for sent in requests {
        assert_eq!(sent.method, Method::POST);
        assert_eq!(sent.path_and_query, ENDPOINT_CHAT_PATH);
        assert_eq!(
            sent.authorization.as_deref(),
            Some(format!("Bearer {SYNTHETIC_TOKEN}").as_str())
        );
        assert_eq!(sent.accept.as_deref(), Some("text/event-stream"));
        // The configured model, the tools as translated, a stream with
        // usage, and neither an effort nor a thinking switch.
        assert_eq!(
            sent.body,
            json!({
                "model": "gemma-2-9b-it",
                "messages": [
                    {"role":"system","content":"Be brief."},
                    {"role":"user","content":"Say hello"}
                ],
                "tools": [{"type":"function","function":{
                    "name":"read_file",
                    "description":"Read a file",
                    "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
                }}],
                "stream": true,
                "stream_options": {"include_usage": true},
                "max_tokens": 256
            })
        );
    }
}

#[tokio::test]
async fn vertex_endpoint_synthetic_endpoints_get_runs_once_and_its_dedicated_url_is_used_as_written()
 {
    // The dedicated host is a second stand-in, so a request that went to
    // the base URL instead would be seen there.
    let dedicated = FakeVertex::start(vec![
        open_model_sse(&open_model_text("Hello")),
        open_model_sse(&open_model_text("Hello")),
    ])
    .await;
    let lookup = FakeVertex::start(vec![endpoint_metadata(json!({
        "dedicatedEndpointEnabled": true,
        "dedicatedEndpointDns": dedicated.base,
    }))])
    .await;
    let registry = registry(endpoint_provider(
        &lookup,
        json!({"my-llm": endpoint_entry(json!({}))}),
    ));
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(ENDPOINT_MODEL, stream),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        assert_eq!(endpoint_answer_text(stream, &body), "Hello", "{body}");
    }
    let lookups = lookup.requests();
    assert_eq!(lookups.len(), 1, "endpoints.get runs once: {lookups:?}");
    assert_eq!(lookups[0].method, Method::GET);
    assert_eq!(lookups[0].path_and_query, ENDPOINT_GET_PATH);
    assert_eq!(
        lookups[0].authorization.as_deref(),
        Some(format!("Bearer {SYNTHETIC_TOKEN}").as_str())
    );
    let requests = dedicated.requests();
    assert_eq!(requests.len(), 2, "{requests:?}");
    for sent in requests {
        // The path is appended once, to the reported URL as written.
        assert_eq!(sent.method, Method::POST);
        assert_eq!(sent.path_and_query, ENDPOINT_CHAT_PATH);
        assert_eq!(sent.body["model"], "gemma-2-9b-it");
    }
}

#[tokio::test]
async fn vertex_endpoint_synthetic_shared_and_dns_less_endpoints_fail_before_any_inference_call() {
    let dedicated = FakeVertex::start(Vec::new()).await;
    let shared = "Vertex AI endpoint my-llm is not a dedicated endpoint (endpoint \
                  1234567890123456789 in project ccp-project, location us-central1 \
                  (vertex.endpoints.my-llm in config.json)): endpoints.get did not report \
                  dedicatedEndpointEnabled: true. A shared endpoint does not stream (no SSE) and \
                  limits a request to 60 seconds and 1.5 MB, so claude-code-proxy needs a \
                  dedicated endpoint: deploy the model to a dedicated endpoint, or, if this \
                  endpoint is one, set \"host\" in vertex.endpoints.my-llm to its dedicated URL";
    let unusable = |reason: &str| {
        format!(
            "Vertex AI endpoint my-llm is a dedicated endpoint, but endpoints.get returned {reason} \
             (endpoint 1234567890123456789 in project ccp-project, location us-central1 \
             (vertex.endpoints.my-llm in config.json)). The proxy sends the access token only to \
             https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog and never to the \
             shared host, which a dedicated endpoint does not answer on; set \"host\" in \
             vertex.endpoints.my-llm to the endpoint's dedicated URL (upstream HTTP 200)"
        )
    };
    let cases = [
        (
            "shared endpoint",
            json!({"dedicatedEndpointEnabled": false, "dedicatedEndpointDns": dedicated.base}),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            shared.to_string(),
        ),
        (
            "no dedicated flag",
            json!({"dedicatedEndpointDns": dedicated.base}),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            shared.to_string(),
        ),
        (
            "dedicated without a DNS name",
            json!({"dedicatedEndpointEnabled": true}),
            StatusCode::BAD_GATEWAY,
            "api_error",
            unusable("no dedicatedEndpointDns"),
        ),
        (
            "a DNS name with a path",
            json!({
                "dedicatedEndpointEnabled": true,
                "dedicatedEndpointDns": "1234567890123456789.us-central1-987654321.prediction.vertexai.goog/v1"
            }),
            StatusCode::BAD_GATEWAY,
            "api_error",
            unusable(
                "a dedicatedEndpointDns that must be a scheme and a host only, without a path, \
                 query or fragment",
            ),
        ),
        (
            "a host outside prediction.vertexai.goog",
            json!({
                "dedicatedEndpointEnabled": true,
                "dedicatedEndpointDns": "https://synthetic.example"
            }),
            StatusCode::BAD_GATEWAY,
            "api_error",
            unusable(
                "a dedicatedEndpointDns that must be a dedicated endpoint host, \
                 https://<endpoint-id>.<region>-<uid>.prediction.vertexai.goog, or a loopback host",
            ),
        ),
    ];
    for (case, fields, status, error_type, expected) in cases {
        // A failed lookup is not kept, so each request reads the endpoint.
        let lookup = FakeVertex::start(vec![
            endpoint_metadata(fields.clone()),
            endpoint_metadata(fields),
        ])
        .await;
        let registry = registry(endpoint_provider(
            &lookup,
            json!({"my-llm": endpoint_entry(json!({}))}),
        ));
        for stream in [false, true] {
            let (answered, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(ENDPOINT_MODEL, stream),
                &[],
            )
            .await;
            assert_eq!(answered, status, "{case}, stream={stream}: {body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], error_type, "{case}: {body}");
            assert_eq!(error_message(&body), expected, "{case}, stream={stream}");
        }
        let lookups = lookup.requests();
        assert_eq!(lookups.len(), 2, "{case}: {lookups:?}");
        assert!(
            lookups
                .iter()
                .all(|sent| sent.method == Method::GET && sent.path_and_query == ENDPOINT_GET_PATH),
            "{case}: {lookups:?}"
        );
    }
    assert!(
        dedicated.requests().is_empty(),
        "{:?}",
        dedicated.requests()
    );
}

#[tokio::test]
async fn vertex_endpoint_synthetic_cold_start_429_is_a_503_and_other_429s_are_rate_limits() {
    // The autoscaling guide's wording for a request to a scale-to-zero
    // endpoint without a ready replica.
    let not_ready = google_error(
        429,
        "RESOURCE_EXHAUSTED",
        "Model is not yet ready for inference. Please wait and then try your request again",
    );
    let quota = google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded.");
    let upstream = FakeVertex::start(vec![
        Answer::error(429, not_ready.clone()),
        Answer::error(429, not_ready),
        Answer::error(429, quota.clone()),
        Answer::error(429, quota),
    ])
    .await;
    let registry = registry(endpoint_provider(
        &upstream,
        json!({"my-llm": endpoint_entry(json!({"host": upstream.base}))}),
    ));
    let cases = [
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            "Vertex AI endpoint my-llm is starting: RESOURCE_EXHAUSTED: Model is not yet ready \
             for inference. Please wait and then try your request again. An endpoint that scales \
             to zero answers this until a replica is ready; it is not a rate limit. Retry in a \
             few minutes (upstream HTTP 429)",
        ),
        (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "RESOURCE_EXHAUSTED: Quota exceeded.. Vertex AI endpoint 1234567890123456789 in \
             project ccp-project, location us-central1 (vertex.endpoints.my-llm in config.json) \
             is rate limited or its serving container is overloaded; retry later (upstream HTTP \
             429)",
        ),
    ];
    for (status, error_type, expected) in cases {
        for stream in [false, true] {
            let (answered, _, body) = send(
                app_with_monitor(registry.clone(), None),
                "/v1/messages",
                &message_request(ENDPOINT_MODEL, stream),
                &[],
            )
            .await;
            assert_eq!(answered, status, "stream={stream}: {body}");
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["error"]["type"], error_type, "{body}");
            assert_eq!(error_message(&body), expected, "stream={stream}");
        }
    }
    assert_eq!(upstream.requests().len(), 4);
}

#[tokio::test]
async fn vertex_endpoint_synthetic_an_ordinary_429_is_a_rate_limit_with_its_retry_after() {
    let quota = google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded.");
    let answer = || {
        let mut answer = Answer::error(429, quota.clone());
        answer.headers.push(("retry-after", "17".to_string()));
        answer
    };
    let upstream = FakeVertex::start(vec![answer(), answer()]).await;
    let registry = registry(endpoint_provider(
        &upstream,
        json!({"my-llm": endpoint_entry(json!({"host": upstream.base}))}),
    ));
    for stream in [false, true] {
        let (status, headers, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(ENDPOINT_MODEL, stream),
            &[],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::TOO_MANY_REQUESTS,
            "stream={stream}: {body}"
        );
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["error"]["type"], "rate_limit_error", "{body}");
        assert_eq!(
            error_message(&body),
            "RESOURCE_EXHAUSTED: Quota exceeded.. Vertex AI endpoint 1234567890123456789 in \
             project ccp-project, location us-central1 (vertex.endpoints.my-llm in config.json) \
             is rate limited or its serving container is overloaded; retry later (upstream HTTP \
             429)",
            "stream={stream}"
        );
        assert_eq!(
            headers
                .get(header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("17"),
            "stream={stream}: {body}"
        );
    }
    assert_eq!(upstream.requests().len(), 2);
}

#[tokio::test]
async fn vertex_endpoint_synthetic_tools_off_refuses_tools_and_tools_on_sends_them() {
    let upstream = FakeVertex::start(vec![open_model_sse(&open_model_text("Hello"))]).await;
    let registry = registry(endpoint_provider(
        &upstream,
        json!({
            "no-tools": endpoint_entry(json!({"host": upstream.base, "tools": false})),
            "my-llm": endpoint_entry(json!({"host": upstream.base}))
        }),
    ));
    for stream in [false, true] {
        let (status, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &open_model_request("vertex/endpoint/no-tools", stream),
            &[],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "stream={stream}: {body}");
        assert_eq!(
            error_message(&body),
            "Vertex AI endpoint no-tools is configured with \"tools\": false in vertex.endpoints, \
             so a request that carries tools is not sent to it; send the request without tools, \
             or set \"tools\": true once the endpoint's serving container supports tool calling"
        );
    }
    assert!(upstream.requests().is_empty(), "{:?}", upstream.requests());

    // Whether a container can call tools cannot be read from the
    // endpoint, so an endpoint left at the default gets them.
    let (status, _, body) = send(
        app_with_monitor(registry.clone(), None),
        "/v1/messages",
        &open_model_request(ENDPOINT_MODEL, false),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let sent = upstream.requests();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].body["tools"][0]["function"]["name"], "read_file");
}

#[tokio::test]
async fn vertex_endpoint_synthetic_bad_settings_name_the_endpoint_and_nothing_is_sent() {
    let upstream = FakeVertex::start(Vec::new()).await;
    let registry = registry(endpoint_provider(
        &upstream,
        json!({
            "plain-http": endpoint_entry(json!({"host": "http://synthetic.example"})),
            "in-global": endpoint_entry(json!({"location": "global"})),
            "bad name": endpoint_entry(json!({}))
        }),
    ));
    let prefix =
        "Invalid Vertex configuration: Invalid Vertex setting vertex.endpoints in config.json: ";
    let cases = [
        (
            "vertex/endpoint/plain-http",
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "{prefix}endpoint \"plain-http\": host must use https (plain http only for a loopback host)"
            ),
        ),
        (
            "vertex/endpoint/in-global",
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "{prefix}endpoint \"in-global\": location must be the endpoint's region, such as us-central1; an endpoint is never in global"
            ),
        ),
        (
            "vertex/endpoint/bad name",
            StatusCode::BAD_REQUEST,
            "invalid Vertex AI endpoint name in \"vertex/endpoint/bad name\"; the name after \
             vertex/endpoint/ is a key of vertex.endpoints in config.json: letters, digits, '_' \
             and '-'"
                .to_string(),
        ),
        (
            "vertex/endpoint/missing",
            StatusCode::BAD_REQUEST,
            "Vertex AI endpoint \"missing\" is not configured: add \"missing\": {\"endpoint\": \
             \"<endpoint-id>\", \"location\": \"<region>\", \"model\": \"<model the container \
             serves>\"} to vertex.endpoints in config.json and restart the proxy; configured \
             endpoints: none"
                .to_string(),
        ),
    ];
    for (model, status, expected) in cases {
        let (answered, _, body) = send(
            app_with_monitor(registry.clone(), None),
            "/v1/messages",
            &message_request(model, false),
            &[],
        )
        .await;
        assert_eq!(answered, status, "{model}: {body}");
        assert_eq!(error_message(&body), expected, "{model}");
    }
    assert!(upstream.requests().is_empty(), "{:?}", upstream.requests());
}

#[tokio::test]
async fn vertex_endpoint_synthetic_lookup_403_offers_the_host_setting() {
    let upstream = FakeVertex::start(vec![Answer::error(
        403,
        google_error(
            403,
            "PERMISSION_DENIED",
            "Permission 'aiplatform.endpoints.get' denied on resource.",
        ),
    )])
    .await;
    let registry = registry(endpoint_provider(
        &upstream,
        json!({"my-llm": endpoint_entry(json!({}))}),
    ));
    let (status, _, body) = send(
        app_with_monitor(registry, None),
        "/v1/messages",
        &message_request(ENDPOINT_MODEL, false),
        &[],
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(value["error"]["type"], "permission_error", "{body}");
    assert_eq!(
        error_message(&body),
        "PERMISSION_DENIED: Permission 'aiplatform.endpoints.get' denied on resource.. Reading \
         Vertex AI endpoint 1234567890123456789 in project ccp-project, location us-central1 \
         (vertex.endpoints.my-llm in config.json) with endpoints.get needs the permission \
         aiplatform.endpoints.get (included in roles/aiplatform.user): grant it, or set \"host\" \
         in vertex.endpoints.my-llm to the endpoint's dedicated URL so the proxy does not read \
         the endpoint (upstream HTTP 403)"
    );
    let requests = upstream.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, Method::GET);
}

#[tokio::test]
async fn vertex_endpoint_synthetic_models_route_lists_usable_endpoints_and_counts_tokens_locally() {
    let upstream = FakeVertex::start(Vec::new()).await;
    let registry = registry(endpoint_provider(
        &upstream,
        json!({
            "my-llm": endpoint_entry(json!({})),
            "in-global": endpoint_entry(json!({"location": "global"}))
        }),
    ));
    let response = app_with_monitor(registry.clone(), None)
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
    assert!(body.contains("\"vertex/endpoint/my-llm\""), "{body}");
    assert!(!body.contains("in-global"), "{body}");

    let (status, _, body) = post(
        app_with_monitor(registry, None),
        "/v1/messages/count_tokens",
        &json!({"model": ENDPOINT_MODEL, "messages": [{"role":"user","content":"hello world"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let value: Value = serde_json::from_str(&body).unwrap();
    assert!(value["input_tokens"].as_u64().unwrap() > 0, "{body}");
    assert!(upstream.requests().is_empty());
}
