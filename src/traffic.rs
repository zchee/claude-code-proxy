use serde_json::{Map, Value};
use std::fs::{self, File, create_dir_all};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use crate::config::AliasProvider;
use crate::logging::REDACT_KEYS;
use crate::paths;
use crate::providers::codex::translate::reasoning_signature::is_proxy_reasoning_signature;
use crate::providers::google_ai_plan::translate::thought_signature::is_google_ai_plan_thought_signature;
use crate::providers::muse::sanitize::{redact_secrets, sanitize_error_value};
use crate::providers::muse::translate::reasoning_signature::is_muse_reasoning_signature;
use crate::providers::vertex::translate::thought_signature::is_vertex_thought_signature;

#[derive(Debug)]
pub struct TrafficCapture {
    root: PathBuf,
    artifact_counter: Mutex<usize>,
    event_counter: Mutex<usize>,
}

pub const MAX_SSE_CAPTURE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_STREAM_CAPTURE_EVENT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_STREAM_CAPTURE_EVENTS: usize = 1_024;
pub const MAX_STREAM_CAPTURE_FRAME_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub struct TrafficCaptureOptions {
    pub req_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub provider: Option<String>,
    pub state_dir_override: Option<PathBuf>,
}

pub fn traffic_capture_enabled() -> bool {
    traffic_capture_enabled_for_env(&std::env::vars().collect())
}

pub fn traffic_capture_enabled_for_env(env: &std::collections::HashMap<String, String>) -> bool {
    match env.get("CCP_TRAFFIC_LOG").map(String::as_str) {
        Some(v) => matches!(v, "1" | "true" | "yes"),
        None => false,
    }
}

pub fn create_traffic_capture(opts: TrafficCaptureOptions) -> Option<TrafficCapture> {
    if !traffic_capture_enabled() {
        return None;
    }
    let state_root = opts
        .state_dir_override
        .unwrap_or_else(paths::state_dir)
        .join("traffic")
        .join(sanitize_path_part(
            opts.session_id.as_deref().unwrap_or("no-session"),
        ))
        .join(format!(
            "{:06}-{}-{}",
            opts.session_seq.unwrap_or(0),
            sanitize_path_part(opts.provider.as_deref().unwrap_or("unknown-provider")),
            sanitize_path_part(&opts.req_id),
        ));

    Some(TrafficCapture {
        root: state_root,
        artifact_counter: Mutex::new(0),
        event_counter: Mutex::new(0),
    })
}

impl TrafficCapture {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn write_json(&self, name: &str, value: &Value) {
        let value = redact_traffic(value);
        let payload = serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| "{}".to_string())
            .into_bytes();
        let path = self.next_artifact_path(name, true);
        let _ = write_bytes(path, &payload);
    }

    pub fn write_text(&self, name: &str, text: &str) {
        let file = if name.ends_with(".txt") {
            name.to_string()
        } else {
            format!("{name}.txt")
        };
        let path = self.next_artifact_path(&file, false);
        let _ = write_bytes(path, text.as_bytes());
    }

    pub fn write_bytes(&self, name: &str, value: &[u8]) {
        let path = self.next_artifact_path(name, false);
        let _ = write_bytes(path, value);
    }

    pub fn write_json_event(&self, name: &str, value: &Value) {
        let value = redact_traffic(value);
        let payload = serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| "{}".to_string())
            .into_bytes();
        let path = self.next_event_path(name, true);
        let _ = write_bytes(path, &payload);
    }

    pub fn stream_capture(&self) -> StreamTrafficCapture {
        StreamTrafficCapture::default()
    }

    fn next_artifact_path(&self, name: &str, ensure_ext_json: bool) -> PathBuf {
        let mut counter: MutexGuard<'_, usize> = self
            .artifact_counter
            .lock()
            .unwrap_or_else(|_| self.artifact_counter.lock().unwrap());
        *counter += 1;
        let file = if ensure_ext_json && !name.ends_with(".json") {
            format!("{name}.json")
        } else {
            name.to_string()
        };
        self.root
            .join(format!("{:03}-{}", *counter, sanitize_path_part(&file)))
    }

    fn next_event_path(&self, name: &str, ensure_ext_json: bool) -> PathBuf {
        let mut counter: MutexGuard<'_, usize> = self
            .event_counter
            .lock()
            .unwrap_or_else(|_| self.event_counter.lock().unwrap());
        *counter += 1;
        let file = if ensure_ext_json && !name.ends_with(".json") {
            format!("{name}.json")
        } else {
            name.to_string()
        };
        self.root
            .join("events")
            .join(format!("{:06}-{}", *counter, sanitize_path_part(&file)))
    }
}

#[cfg(test)]
pub(crate) fn test_capture(root: PathBuf) -> TrafficCapture {
    TrafficCapture {
        root,
        artifact_counter: Mutex::new(0),
        event_counter: Mutex::new(0),
    }
}

fn write_bytes(path: PathBuf, value: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
        if let Ok(meta) = fs::metadata(parent) {
            set_mode(parent, 0o700);
            if meta.is_dir() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perm = meta.permissions();
                    perm.set_mode(0o700);
                    let _ = fs::set_permissions(parent, perm);
                }
            }
        }
    }
    let mut out = File::create(&path)?;
    out.write_all(value)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = out.metadata()?.permissions();
        perm.set_mode(0o600);
        let _ = fs::set_permissions(&path, perm);
    }
    Ok(())
}

pub struct StreamTrafficCapture {
    upstream_sse: Vec<u8>,
    upstream_events: Vec<Value>,
    downstream_events: Vec<Value>,
    malformed: Vec<Value>,
    upstream_event_bytes: usize,
    downstream_event_bytes: usize,
    upstream_sse_truncated: u64,
    upstream_events_truncated: u64,
    downstream_events_truncated: u64,
    malformed_truncated: u64,
    upstream_frames_truncated: u64,
}

impl Default for StreamTrafficCapture {
    fn default() -> Self {
        Self {
            upstream_sse: Vec::with_capacity(MAX_SSE_CAPTURE_BYTES.min(64 * 1024)),
            upstream_events: Vec::new(),
            downstream_events: Vec::new(),
            malformed: Vec::new(),
            upstream_event_bytes: 0,
            downstream_event_bytes: 0,
            upstream_sse_truncated: 0,
            upstream_events_truncated: 0,
            downstream_events_truncated: 0,
            malformed_truncated: 0,
            upstream_frames_truncated: 0,
        }
    }
}

impl StreamTrafficCapture {
    pub fn upstream_event(&mut self, event: Option<&str>, value: &Value) {
        let value = redact_traffic(value);
        let frame = serde_json::to_vec(&value).unwrap_or_default();
        let event = event.unwrap_or("message");
        let frame_len = event.len().saturating_add(frame.len()).saturating_add(16);
        if frame_len > MAX_STREAM_CAPTURE_FRAME_BYTES {
            self.upstream_frames_truncated = self.upstream_frames_truncated.saturating_add(1);
        } else if self.upstream_sse.len().saturating_add(frame_len) <= MAX_SSE_CAPTURE_BYTES {
            self.upstream_sse.extend_from_slice(b"event: ");
            self.upstream_sse.extend_from_slice(event.as_bytes());
            self.upstream_sse.extend_from_slice(b"\ndata: ");
            self.upstream_sse.extend_from_slice(&frame);
            self.upstream_sse.extend_from_slice(b"\n\n");
        } else {
            self.upstream_sse_truncated = self.upstream_sse_truncated.saturating_add(1);
        }
        self.push_event(true, serde_json::json!({"event":event,"data":value}));
    }

    pub fn malformed(&mut self, stage: &str, kind: &str) {
        if self.malformed.len() < MAX_STREAM_CAPTURE_EVENTS {
            self.malformed
                .push(serde_json::json!({"stage":stage,"kind":kind}));
        } else {
            self.malformed_truncated = self.malformed_truncated.saturating_add(1);
        }
    }

    pub fn downstream_event(&mut self, event: &str, data: Value) {
        self.push_event(
            false,
            serde_json::json!({"event":event,"data":redact_traffic(&data)}),
        );
    }

    fn push_event(&mut self, upstream: bool, value: Value) {
        let bytes = serde_json::to_vec(&value).map_or(0, |value| value.len());
        let (events, total, truncated) = if upstream {
            (
                &mut self.upstream_events,
                &mut self.upstream_event_bytes,
                &mut self.upstream_events_truncated,
            )
        } else {
            (
                &mut self.downstream_events,
                &mut self.downstream_event_bytes,
                &mut self.downstream_events_truncated,
            )
        };
        if events.len() < MAX_STREAM_CAPTURE_EVENTS
            && total.saturating_add(bytes) <= MAX_STREAM_CAPTURE_EVENT_BYTES
        {
            *total += bytes;
            events.push(value);
        } else {
            *truncated = truncated.saturating_add(1);
        }
    }

    pub fn finish(self, traffic: &TrafficCapture, completion: Value) {
        self.finish_named(traffic, completion, "061-grok-stream-summary");
    }

    pub fn finish_named(self, traffic: &TrafficCapture, completion: Value, summary_name: &str) {
        let upstream_event_count = self.upstream_events.len();
        let downstream_event_count = self.downstream_events.len();
        if !self.upstream_sse.is_empty() {
            traffic.write_bytes("032-upstream-response-body.sse", &self.upstream_sse);
        }
        traffic.write_json(
            "033-upstream-response-capture",
            &serde_json::json!({
                "truncated": self.upstream_sse_truncated > 0 || self.upstream_frames_truncated > 0 || self.upstream_events_truncated > 0,
                "captured_bytes": self.upstream_sse.len(),
                "truncated_frames": self.upstream_sse_truncated,
                "oversized_frames": self.upstream_frames_truncated,
                "captured_events": self.upstream_events.len(),
                "captured_event_bytes": self.upstream_event_bytes,
                "truncated_events": self.upstream_events_truncated,
                "malformed": self.malformed,
                "truncated_malformed": self.malformed_truncated,
            }),
        );
        for value in self.upstream_events {
            traffic.write_json_event("040-upstream-event", &value);
        }
        for value in self.downstream_events {
            traffic.write_json_event("050-downstream-event", &value);
        }
        traffic.write_json(
            summary_name,
            &serde_json::json!({
                "completion": completion,
                "upstream_sse": {
                    "captured_bytes": self.upstream_sse.len(),
                    "truncated_frames": self.upstream_sse_truncated,
                    "oversized_frames": self.upstream_frames_truncated,
                },
                "upstream_events": {
                    "captured": upstream_event_count,
                    "captured_bytes": self.upstream_event_bytes,
                    "truncated": self.upstream_events_truncated,
                },
                "downstream_events": {
                    "captured": downstream_event_count,
                    "captured_bytes": self.downstream_event_bytes,
                    "truncated": self.downstream_events_truncated,
                },
            }),
        );
    }
}

pub fn sanitize_path_part(input: &str) -> String {
    let cleaned: String = input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    let truncated = if cleaned.len() > 160 {
        &cleaned[..160]
    } else {
        &cleaned
    };
    if truncated.is_empty() {
        "unknown".to_string()
    } else {
        truncated.to_string()
    }
}

pub fn redact_traffic(value: &Value) -> Value {
    redact_traffic_with_depth(value, 0)
}

fn redact_traffic_with_depth(value: &Value, depth: u16) -> Value {
    if depth > 100 {
        return Value::String("[depth-limit]".to_string());
    }

    match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                let normalized = key.to_lowercase();
                if normalized == "image_url" {
                    // Grok/OpenAI `input_image` parts carry data URLs; never
                    // persist the payload in traffic captures.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "data" && looks_like_image_source(map) {
                    // Anthropic image blocks carry raw base64 under
                    // `source.data`; redact it for the same reason.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "data" && looks_like_inline_data(map) {
                    // Gemini `inlineData` parts carry raw base64 media beside
                    // `mimeType`, and Interactions image content beside
                    // `mime_type`.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "thoughtsignature" || normalized == "thought_signature" {
                    // Gemini thought signatures are opaque model state replayed
                    // with function calls; only their size is kept.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "encrypted_content" {
                    // Codex `compaction` and `reasoning` items carry an opaque
                    // blob that replays a whole conversation's context; it is
                    // bulk payload, not debug signal, so only its size stays.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "signature"
                    && value.as_str().is_some_and(|signature| {
                        is_proxy_reasoning_signature(signature)
                            || is_muse_reasoning_signature(signature)
                            || is_vertex_thought_signature(signature)
                            || is_google_ai_plan_thought_signature(signature)
                    })
                {
                    // The Codex and Muse translators replay reasoning as an
                    // Anthropic thinking signature of the form
                    // `ccp:<provider>:v1:<encoded-id>:<encrypted_content>`,
                    // which carries the same opaque conversation handle as the
                    // `encrypted_content` key; the Vertex translator carries a
                    // Gemini thought signature as `ccp:vertex:v1:<model>:<sig>`,
                    // and the Google AI Ultra plan translator as
                    // `ccp:google-ai-plan:v1:<model>:<sig>`.
                    // Only proxy-owned signatures are touched; foreign ones
                    // (for example, real Anthropic signatures) are preserved.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if normalized == "signature" && looks_like_interactions_thought(map) {
                    // Interactions `thought` steps and `thought_signature`
                    // deltas carry the raw Gemini thought signature under a
                    // plain `signature` key.
                    out.insert(key.clone(), redact_traffic_value(value));
                } else if REDACT_KEYS.contains(&normalized.as_str())
                    || matches!(
                        normalized.as_str(),
                        "token"
                            | "bearer_token"
                            | "oauth_token"
                            | "oauth_access_token"
                            | "oauth_refresh_token"
                            | "client_secret"
                            | "secret"
                            | "password"
                            | "email"
                            | "user_id"
                            | "account_id"
                            | "identity"
                            | "identity_id"
                            | "subject"
                            | "sub"
                    )
                {
                    out.insert(key.clone(), redact_traffic_value(value));
                } else {
                    out.insert(key.clone(), redact_traffic_with_depth(value, depth + 1));
                }
            }
            Value::Object(out)
        }
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| redact_traffic_with_depth(value, depth + 1))
                .collect(),
        ),
        Value::String(text) if is_data_url(text) => {
            Value::String(format!("[redacted data-url len={}]", text.len()))
        }
        _ => value.clone(),
    }
}

/// Prepares a Muse stream event, or an event rendered from one, for a
/// traffic capture. Key-based redaction cannot see a credential embedded in a
/// string, so every string has the request's `secrets` replaced. The parts
/// that carry upstream error text (a whole `error` event, a top-level `error`
/// object, `response.error`) go through [`sanitize_error_value`], the same
/// rules the client applies to HTTP error bodies. Conversation content keeps
/// everything except the secrets.
pub fn sanitize_muse_capture(value: &Value, secrets: &[&str]) -> Value {
    let Some(object) = value.as_object() else {
        return replace_secrets(value, secrets, 0);
    };
    if object.get("type").and_then(Value::as_str) == Some("error") {
        return sanitize_error_value(value, secrets);
    }
    let mut object = object.clone();
    if let Some(error) = object.get_mut("error") {
        *error = sanitize_error_value(error, secrets);
    }
    if let Some(error) = object
        .get_mut("response")
        .and_then(|response| response.get_mut("error"))
    {
        *error = sanitize_error_value(error, secrets);
    }
    replace_secrets(&Value::Object(object), secrets, 0)
}

fn replace_secrets(value: &Value, secrets: &[&str], depth: u16) -> Value {
    if depth > 100 {
        return Value::String("[depth-limit]".to_string());
    }
    match value {
        Value::String(text) => Value::String(redact_secrets(text, secrets)),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| replace_secrets(value, secrets, depth + 1))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), replace_secrets(value, secrets, depth + 1)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn is_data_url(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.starts_with("data:image/") && lower.contains(";base64,")
}

/// An object shaped like an Anthropic image source: `type: "base64"` and
/// either an image `media_type` or none at all. Used to redact only image
/// payloads, not arbitrary `data` keys. Missing `media_type` is treated as
/// an image: the capture is written before translation, so a malformed block
/// that the translator would reject must still not persist its payload.
fn looks_like_image_source(map: &Map<String, Value>) -> bool {
    if !map
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|source_type| source_type.eq_ignore_ascii_case("base64"))
    {
        return false;
    }
    match map.get("media_type").and_then(Value::as_str) {
        Some(media_type) => media_type.to_ascii_lowercase().starts_with("image/"),
        None => true,
    }
}

/// An object shaped like a Gemini `Blob` (`inlineData`): a `mimeType` key
/// beside `data`. Any media type is redacted, not only images.
fn looks_like_inline_data(map: &Map<String, Value>) -> bool {
    map.keys()
        .any(|key| key.eq_ignore_ascii_case("mimetype") || key.eq_ignore_ascii_case("mime_type"))
}

fn looks_like_interactions_thought(map: &Map<String, Value>) -> bool {
    map.get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "thought" || kind == "thought_signature")
}

fn redact_traffic_value(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(format!("[redacted len={}]", s.len())),
        // Structured values (e.g. Kimi's `{url: ...}` image_url) keep their
        // shape so captures stay parseable; only the payload leaves.
        Value::Object(_) | Value::Array(_) => redact_traffic(value),
        _ => Value::String("[redacted]".to_string()),
    }
}

fn set_mode(path: &Path, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perm = meta.permissions();
            perm.set_mode(mode);
            let _ = fs::set_permissions(path, perm);
        }
    }
}

#[allow(dead_code)]
fn _provider_alias(_provider: &str) -> Option<AliasProvider> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_traffic_strips_input_image_data_urls() {
        let value = serde_json::json!({
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "what color?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}
                ]}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "base64 payload leaked: {rendered}"
        );
        assert!(
            !rendered.contains("data:image/png;base64,iVBOR"),
            "data URL payload leaked: {rendered}"
        );
        // The image part is still structurally visible (redacted marker).
        assert!(rendered.contains("input_image"));
    }

    #[test]
    fn redact_traffic_strips_inline_base64_image_values() {
        let value = serde_json::json!({
            "output": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(!rendered.contains("iVBORw0KGgo"));
    }

    #[test]
    fn redact_traffic_strips_anthropic_image_source_data() {
        // The incoming Anthropic request body carries raw base64 under
        // `source.data` (no data: URL wrapper) — it must not persist either.
        let value = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
                    }}
                ]
            }]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "source.data base64 leaked: {rendered}"
        );
        assert!(rendered.contains("redacted"));
        // Structure is preserved.
        assert!(rendered.contains("image/png"));
    }

    #[test]
    fn redact_traffic_strips_compaction_encrypted_content() {
        // Server-side compaction replays the transcript as an opaque blob in
        // the upstream request; it must not persist, but its size should.
        let value = serde_json::json!({
            "model": "gpt-5",
            "input": [
                {"type": "compaction", "encrypted_content": "gAAAAABopaque-compacted-transcript"},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hello"}
                ]}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("gAAAAA"),
            "encrypted_content leaked: {rendered}"
        );
        assert_eq!(
            redacted["input"][0]["encrypted_content"],
            "[redacted len=34]"
        );
        assert_eq!(redacted["input"][0]["type"], "compaction");
        assert_eq!(redacted["input"][1]["content"][0]["text"], "hello");
        assert_eq!(redacted["model"], "gpt-5");
    }

    #[test]
    fn redact_traffic_strips_reasoning_encrypted_content_in_events() {
        // Upstream response items carry the same blob under `reasoning`.
        let value = serde_json::json!({
            "type": "response.output_item.done",
            "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAAAopaque"}
        });
        let redacted = redact_traffic(&value);
        assert_eq!(redacted["item"]["encrypted_content"], "[redacted len=12]");
        assert_eq!(redacted["item"]["id"], "rs_1");
    }

    #[test]
    fn redact_traffic_strips_proxy_signature_delta_in_downstream_event() {
        // The Codex translator emits a 050 downstream `signature_delta` whose
        // signature is `ccp:codex:v1:<encoded-id>:<encrypted_content>`.
        let signature = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let value = serde_json::json!({
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature}
            }
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("ccp:codex:v1:"),
            "proxy signature leaked: {rendered}"
        );
        assert!(
            !rendered.contains("gAAAAABopaque"),
            "encrypted payload leaked: {rendered}"
        );
        assert_eq!(
            redacted["data"]["delta"]["signature"],
            format!("[redacted len={}]", signature.len())
        );
        // Neighboring structure survives.
        assert_eq!(redacted["data"]["delta"]["type"], "signature_delta");
        assert_eq!(redacted["data"]["index"], 0);
    }

    #[test]
    fn redact_traffic_strips_proxy_signature_in_thinking_blocks() {
        // Buffered downstream content and the replayed incoming Anthropic
        // request both carry the signature on a `thinking` block.
        let signature = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let value = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "reasoned", "signature": signature},
                {"type": "text", "text": "answer"}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("ccp:codex:v1:"),
            "proxy signature leaked: {rendered}"
        );
        assert_eq!(
            redacted["content"][0]["signature"],
            format!("[redacted len={}]", signature.len())
        );
        assert_eq!(redacted["content"][0]["thinking"], "reasoned");
        assert_eq!(redacted["content"][1]["text"], "answer");
    }

    #[test]
    fn redact_traffic_keeps_foreign_signatures() {
        // Real Anthropic signatures and other opaque values are not
        // proxy-owned and must survive capture redaction unchanged.
        let value = serde_json::json!({
            "delta": {"type": "signature_delta", "signature": "ErUBCkYIBRgCIkA-real"},
            "content": [{"type": "thinking", "signature": "another-opaque-signature"}]
        });
        let redacted = redact_traffic(&value);
        assert_eq!(redacted["delta"]["signature"], "ErUBCkYIBRgCIkA-real");
        assert_eq!(
            redacted["content"][0]["signature"],
            "another-opaque-signature"
        );
    }

    #[test]
    fn traffic_capture_redacts_proxy_signatures_at_json_boundary() {
        // Exercise write_json_event (050 downstream) and write_json (010
        // incoming request) end to end, where redaction is actually applied.
        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        let signature = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let foreign = "ErUBCkYIBRgCIkA-real-anthropic-signature";

        capture.write_json_event(
            "050-downstream-event",
            &serde_json::json!({
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": signature}
                }
            }),
        );
        capture.write_json(
            "010-anthropic-request",
            &serde_json::json!({
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "kept", "signature": signature},
                        {"type": "thinking", "thinking": "kept", "signature": foreign}
                    ]
                }]
            }),
        );

        let mut captured = String::new();
        for dir in [capture.root().to_path_buf(), capture.root().join("events")] {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() {
                    captured.push_str(&std::fs::read_to_string(path).unwrap());
                }
            }
        }

        assert!(
            !captured.contains("ccp:codex:v1:"),
            "proxy signature leaked to capture: {captured}"
        );
        assert!(
            !captured.contains("gAAAAABopaque"),
            "encrypted payload leaked to capture: {captured}"
        );
        assert!(
            captured.contains(foreign),
            "foreign signature lost: {captured}"
        );
        assert!(captured.contains("kept"));
    }

    #[test]
    fn redact_traffic_keeps_unrelated_data_keys() {
        let value = serde_json::json!({"data": "some-non-image-payload", "count": 3});
        let redacted = redact_traffic(&value);
        assert_eq!(redacted["data"], "some-non-image-payload");
    }

    #[test]
    fn redact_traffic_strips_data_urls_in_array_shaped_tool_output() {
        // L2b inline mode: function_call_output.output is an array of content
        // parts — the image_url inside must be redacted just like message parts.
        let value = serde_json::json!({
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": [
                    {"type": "input_text", "text": "screenshot"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}
                ]}
            ]
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "base64 payload leaked: {rendered}"
        );
        assert!(
            !rendered.contains("data:image/png;base64,iVBOR"),
            "data URL payload leaked: {rendered}"
        );
        // Structure and text survive.
        assert!(rendered.contains("input_image"));
        assert!(rendered.contains("screenshot"));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn redact_traffic_strips_uppercase_media_type_image_source() {
        // Captures are written before translation validates anything, so a
        // non-standard casing must still not persist its payload.
        let value = serde_json::json!({
            "source": {"type": "base64", "media_type": "IMAGE/PNG", "data": "QUJDREVGRw=="}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("QUJDREVGRw"),
            "payload leaked: {rendered}"
        );
        assert!(
            redacted["source"]["data"]
                .as_str()
                .unwrap()
                .contains("redacted")
        );
    }

    #[test]
    fn redact_traffic_strips_uppercase_image_source_type() {
        let value = serde_json::json!({
            "source": {"type": "BASE64", "media_type": "image/png", "data": "QUJDREVGRw=="}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("QUJDREVGRw"),
            "payload leaked: {rendered}"
        );
    }

    #[test]
    fn redact_traffic_strips_image_source_without_media_type() {
        // A malformed block the translator would reject must still be redacted
        // in the pre-translation capture.
        let value = serde_json::json!({
            "source": {"type": "base64", "data": "QUJDREVGRw=="}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("QUJDREVGRw"),
            "payload leaked: {rendered}"
        );
    }

    #[test]
    fn redact_traffic_strips_uppercase_data_url_scheme() {
        let value = serde_json::json!({
            "image_url": "DATA:IMAGE/PNG;base64,iVBORw0KGgoAAAANSUhEUg"
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "data URL payload leaked: {rendered}"
        );
    }

    #[test]
    fn redact_traffic_keeps_structured_image_url_shape() {
        // Kimi-style object image_url: redact the payload but keep the object
        // structure so captures stay parseable.
        let value = serde_json::json!({
            "image_url": {"url": "DATA:IMAGE/PNG;BASE64,iVBORw0KGgoAAAANSUhEUg"}
        });
        let redacted = redact_traffic(&value);
        assert!(redacted["image_url"].is_object(), "shape lost: {redacted}");
        let rendered = redacted.to_string();
        assert!(
            !rendered.contains("iVBORw0KGgo"),
            "nested payload leaked: {rendered}"
        );
    }

    #[test]
    fn traffic_redacts_proxy_authorization() {
        let redacted = redact_traffic(&serde_json::json!({
            "headers": {
                "proxy-authorization": "Basic dXNlcjpwYXNz",
                "x-safe": "kept"
            }
        }));

        assert_eq!(redacted["headers"]["x-safe"], "kept");
        assert_eq!(
            redacted["headers"]["proxy-authorization"],
            "[redacted len=18]"
        );
    }

    const MUSE_SIGNATURE: &str =
        "ccp:muse:v1:cnNfNmFmMjc0NGQ4YWI1YjhkNTM1ODQzMWU:gAAAAABopaque-muse-replay";
    const MUSE_SECRET: &str = "synthetic-muse-api-key-48-chars-0123456789abcdef";

    fn captured_files(capture: &TrafficCapture) -> String {
        let mut captured = String::new();
        for dir in [capture.root().to_path_buf(), capture.root().join("events")] {
            let Ok(entries) = std::fs::read_dir(dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_file() {
                    captured.push_str(&std::fs::read_to_string(path).unwrap());
                }
            }
        }
        captured
    }

    #[test]
    fn traffic_capture_redacts_muse_signatures_in_requests_responses_and_deltas() {
        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        let codex = "ccp:codex:v1:cnNfMQ:gAAAAABopaque-reasoning-replay";
        let foreign = "ErUBCkYIBRgCIkA-real-anthropic-signature";

        capture.write_json(
            "010-anthropic-request",
            &serde_json::json!({
                "messages": [{"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "kept", "signature": MUSE_SIGNATURE},
                    {"type": "thinking", "thinking": "kept", "signature": codex},
                    {"type": "thinking", "thinking": "kept", "signature": foreign}
                ]}]
            }),
        );
        capture.write_json(
            "051-downstream-response",
            &serde_json::json!({
                "type": "message",
                "content": [{"type": "thinking", "thinking": "kept", "signature": MUSE_SIGNATURE}]
            }),
        );
        let mut stream = capture.stream_capture();
        stream.downstream_event(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": MUSE_SIGNATURE}
            }),
        );
        stream.finish_named(&capture, serde_json::json!({}), "061-muse-stream-summary");

        let captured = captured_files(&capture);
        assert!(
            !captured.contains("ccp:muse:v1:"),
            "Muse signature leaked to capture: {captured}"
        );
        assert!(
            !captured.contains("ccp:codex:v1:"),
            "Codex signature leaked to capture: {captured}"
        );
        assert!(!captured.contains("gAAAAABopaque"), "{captured}");
        assert!(captured.contains(foreign), "foreign signature lost");
        assert_eq!(
            captured
                .matches(&format!("[redacted len={}]", MUSE_SIGNATURE.len()))
                .count(),
            3,
            "{captured}"
        );
    }

    fn assert_error_fields_removed(captured: &str) {
        for leaked in [MUSE_SECRET, "owner@example.com", "billing@meta.example"] {
            assert!(
                !captured.contains(leaked),
                "{leaked} leaked to capture: {captured}"
            );
        }
    }

    #[test]
    fn synthetic_muse_error_bodies_and_events_are_sanitised_before_capture() {
        let secrets = [MUSE_SECRET];
        let message = format!("key {MUSE_SECRET} of owner@example.com was refused");
        let http_body = serde_json::json!({"error": {
            "code": MUSE_SECRET,
            "message": message,
            "param": "billing@meta.example",
            "type": "authentication_error"
        }});
        let error_event = serde_json::json!({
            "type": "error",
            "code": "owner@example.com",
            "param": MUSE_SECRET,
            "message": message,
            "error": {"code": "has spaces", "param": "billing@meta.example", "message": message}
        });
        let failed = serde_json::json!({
            "type": "response.failed",
            "response": {"status": "failed", "error": {
                "code": "server_error",
                "param": "owner@example.com",
                "message": message
            }}
        });
        let problem = serde_json::json!({
            "title": "Authentication Error owner@example.com",
            "detail": format!("token {MUSE_SECRET}"),
            "status": 401
        });

        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        // HTTP error bodies take the client's path; stream events take the
        // capture path. Both apply the same field rules.
        capture.write_json(
            "030-upstream-error",
            &sanitize_error_value(&http_body, &secrets),
        );
        capture.write_json(
            "031-upstream-error",
            &sanitize_error_value(&problem, &secrets),
        );
        let mut stream = capture.stream_capture();
        for value in [&error_event, &failed] {
            let event = value["type"].as_str().unwrap();
            stream.upstream_event(Some(event), &sanitize_muse_capture(value, &secrets));
        }
        stream.downstream_event(
            "error",
            sanitize_muse_capture(
                &serde_json::json!({"type": "error", "error": {"type": "api_error", "message": message}}),
                &secrets,
            ),
        );
        stream.finish_named(&capture, serde_json::json!({}), "061-muse-stream-summary");

        let captured = captured_files(&capture);
        assert_error_fields_removed(&captured);
        assert!(captured.contains("key [redacted] of [redacted] was refused"));
        assert!(captured.contains("authentication_error"));

        // A valid code survives the sanitiser; the capture's key-based rules
        // then redact `code` as they do for every provider.
        let sanitised = sanitize_muse_capture(&failed, &secrets);
        assert_eq!(sanitised["response"]["error"]["code"], "server_error");
        assert_eq!(sanitised["response"]["error"]["param"], Value::Null);

        let sanitised = sanitize_muse_capture(&error_event, &secrets);
        assert_eq!(sanitised["code"], Value::Null);
        assert_eq!(sanitised["param"], Value::Null);
        assert_eq!(sanitised["error"]["code"], Value::Null);
        assert_eq!(sanitised["error"]["param"], Value::Null);
        let sanitised = sanitize_muse_capture(&http_body, &secrets);
        assert_eq!(sanitised, sanitize_error_value(&http_body, &secrets));
        assert_eq!(sanitised["error"]["code"], Value::Null);
        assert_eq!(sanitised["error"]["type"], "authentication_error");
    }

    #[test]
    fn synthetic_long_muse_error_text_is_redacted_before_it_is_cut() {
        let message = format!("{}{MUSE_SECRET}", "x".repeat(490));
        let sanitised = sanitize_muse_capture(
            &serde_json::json!({"error": {"message": message}}),
            &[MUSE_SECRET],
        );
        let text = sanitised["error"]["message"].as_str().unwrap();
        assert!(!text.contains("synthetic-muse"), "{text}");
        assert_eq!(text.chars().count(), 500);
    }

    #[test]
    fn muse_conversation_content_keeps_email_addresses_but_not_secrets() {
        let output = serde_json::json!({
            "type": "response.output_text.delta",
            "delta": format!("Write to owner@example.com, not {MUSE_SECRET}"),
            "item_id": "msg_01a1c3159c0872d48e1f1d4210d984ae"
        });
        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        let mut stream = capture.stream_capture();
        stream.upstream_event(
            Some("response.output_text.delta"),
            &sanitize_muse_capture(&output, &[MUSE_SECRET]),
        );
        stream.finish_named(&capture, serde_json::json!({}), "061-muse-stream-summary");
        let captured = captured_files(&capture);
        assert!(captured.contains("owner@example.com"), "{captured}");
        assert!(!captured.contains(MUSE_SECRET), "{captured}");
        assert!(captured.contains("not [redacted]"), "{captured}");
    }

    const VERTEX_ENVELOPE: &str = "ccp:vertex:v1:gemini-3.5-flash:c3ludGhldGljLWdlbWluaS10aG91Z2h0";
    const VERTEX_SIGNATURE: &str = "c3ludGhldGljLWdlbWluaS10aG91Z2h0";

    #[test]
    fn redact_traffic_strips_vertex_envelopes_thought_signatures_and_inline_data() {
        let value = serde_json::json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "kept", "signature": VERTEX_ENVELOPE}
            ]}],
            "contents": [
                {"role": "model", "parts": [
                    {"functionCall": {"name": "Read", "args": {"path": "a"}}, "thoughtSignature": VERTEX_SIGNATURE}
                ]},
                {"role": "user", "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}},
                    {"functionResponse": {"name": "Read", "response": {"output": "kept"}, "parts": [
                        {"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0xLjQKJcfsj6IKNSAwIG9iago"}}
                    ]}}
                ]}
            ],
            "delta": {"type": "signature_delta", "signature": VERTEX_ENVELOPE},
            "thought_signature": VERTEX_SIGNATURE
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        for leaked in [
            VERTEX_SIGNATURE,
            "ccp:vertex:v1:",
            "iVBORw0KGgo",
            "JVBERi0x",
        ] {
            assert!(!rendered.contains(leaked), "{leaked} leaked: {rendered}");
        }
        assert_eq!(
            redacted["contents"][0]["parts"][0]["thoughtSignature"],
            format!("[redacted len={}]", VERTEX_SIGNATURE.len())
        );
        assert_eq!(
            redacted["delta"]["signature"],
            format!("[redacted len={}]", VERTEX_ENVELOPE.len())
        );
        assert_eq!(
            redacted["contents"][1]["parts"][0]["inlineData"]["mimeType"],
            "image/png"
        );
        assert_eq!(
            redacted["contents"][0]["parts"][0]["functionCall"]["name"],
            "Read"
        );
        assert_eq!(redacted["messages"][0]["content"][0]["thinking"], "kept");
        assert!(rendered.contains("\"output\":\"kept\""), "{rendered}");
    }

    #[test]
    fn traffic_capture_redacts_vertex_signatures_in_stream_events() {
        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        let mut stream = capture.stream_capture();
        stream.upstream_event(
            None,
            &serde_json::json!({"candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "Read", "args": {}}, "thoughtSignature": VERTEX_SIGNATURE}
            ]}}]}),
        );
        stream.downstream_event(
            "content_block_delta",
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": VERTEX_ENVELOPE}}),
        );
        stream.finish_named(&capture, serde_json::json!({}), "061-vertex-stream-summary");
        let captured = captured_files(&capture);
        assert!(!captured.contains(VERTEX_SIGNATURE), "{captured}");
        assert!(!captured.contains("ccp:vertex:v1:"), "{captured}");
        assert!(captured.contains("signature_delta"), "{captured}");
    }

    const PLAN_ENVELOPE: &str =
        "ccp:google-ai-plan:v1:gemini-4-hydrogen:c3ludGhldGljLWludGVyYWN0aW9ucw";
    const PLAN_SIGNATURE: &str = "c3ludGhldGljLWludGVyYWN0aW9ucw";

    #[test]
    fn redact_traffic_strips_google_ai_plan_envelopes_raw_signatures_and_images() {
        let value = serde_json::json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "kept", "signature": PLAN_ENVELOPE}
            ]}],
            "input": [
                {"type": "thought", "signature": PLAN_SIGNATURE, "summary": [{"type": "text", "text": "kept"}]},
                {"type": "function_call", "id": "call_1", "name": "Read", "arguments": {}},
                {"type": "user_input", "content": [
                    {"type": "image", "mime_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}
                ]}
            ],
            "step": {"type": "thought", "signature": PLAN_SIGNATURE},
            "delta": {"type": "thought_signature", "signature": PLAN_SIGNATURE}
        });
        let redacted = redact_traffic(&value);
        let rendered = redacted.to_string();
        for leaked in [PLAN_SIGNATURE, "ccp:google-ai-plan:v1:", "iVBORw0KGgo"] {
            assert!(!rendered.contains(leaked), "{leaked} leaked: {rendered}");
        }
        assert_eq!(
            redacted["input"][0]["signature"],
            format!("[redacted len={}]", PLAN_SIGNATURE.len())
        );
        assert_eq!(redacted["input"][0]["summary"][0]["text"], "kept");
        assert_eq!(redacted["input"][1]["name"], "Read");
        assert_eq!(redacted["input"][2]["content"][0]["mime_type"], "image/png");
        assert_eq!(redacted["messages"][0]["content"][0]["thinking"], "kept");
    }

    #[test]
    fn traffic_capture_redacts_google_ai_plan_signatures_in_stream_events() {
        let temp = tempfile::TempDir::new().unwrap();
        let capture = test_capture(temp.path().join("traffic"));
        let mut stream = capture.stream_capture();
        stream.upstream_event(
            Some("step.delta"),
            &serde_json::json!({"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": PLAN_SIGNATURE}}),
        );
        stream.downstream_event(
            "content_block_delta",
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": PLAN_ENVELOPE}}),
        );
        stream.finish_named(
            &capture,
            serde_json::json!({}),
            "061-google-ai-plan-stream-summary",
        );
        let captured = captured_files(&capture);
        assert!(!captured.contains(PLAN_SIGNATURE), "{captured}");
        assert!(!captured.contains("ccp:google-ai-plan:v1:"), "{captured}");
        assert!(captured.contains("signature_delta"), "{captured}");
    }

    #[test]
    fn redact_traffic_keeps_signatures_that_are_not_interactions_thoughts() {
        let value = serde_json::json!({
            "signature": "ccp:google-ai-planish:v1:not-ours",
            "other": {"type": "text", "signature": "kept"}
        });
        assert_eq!(redact_traffic(&value), value);
    }

    #[test]
    fn redact_traffic_keeps_data_without_a_mime_type_and_vertex_like_foreign_values() {
        let value = serde_json::json!({
            "data": "some-non-image-payload",
            "signature": "ccp:vertexish:v1:not-ours",
            "nested": {"data": "kept", "media": "png"}
        });
        let redacted = redact_traffic(&value);
        assert_eq!(redacted, value);
    }
}
