use serde_json::{Value, json};

use super::reducer::ReducerEvent;
use crate::anthropic::sse::{SseEvent, encode_sse_event};
use crate::providers::muse::client::MuseError;

/// Largest SSE frame kept for parsing. The terminal event repeats the whole
/// output, including reasoning blobs of up to 8 MiB, so the bound is wide.
pub const MAX_SSE_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Message of the error event sent when a known event breaks the stream's
/// structure.
pub const INVALID_STREAM_MESSAGE: &str = "Muse stream is invalid";

/// One decoded SSE frame, or the reason a frame could not be read. Unreadable
/// frames are reported rather than failing the stream so the caller can
/// ignore them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedFrame {
    Event(SseEvent),
    Malformed(&'static str),
}

/// Incremental SSE decoder that accepts any chunk boundary and the three
/// line endings.
#[derive(Default)]
pub struct SseDecoder {
    frame: Vec<u8>,
    line_has_bytes: bool,
    skip_lf: bool,
    oversized: bool,
}

impl SseDecoder {
    pub fn push(&mut self, input: &[u8]) -> Vec<DecodedFrame> {
        let mut frames = Vec::new();
        for &byte in input {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\n' => self.end_line(&mut frames),
                b'\r' => {
                    self.end_line(&mut frames);
                    self.skip_lf = true;
                }
                _ => {
                    self.line_has_bytes = true;
                    self.push_byte(byte);
                }
            }
        }
        frames
    }

    fn push_byte(&mut self, byte: u8) {
        if self.frame.len() >= MAX_SSE_FRAME_BYTES {
            self.oversized = true;
        } else {
            self.frame.push(byte);
        }
    }

    fn end_line(&mut self, frames: &mut Vec<DecodedFrame>) {
        if self.line_has_bytes {
            self.line_has_bytes = false;
            self.push_byte(b'\n');
            return;
        }
        if self.oversized {
            frames.push(DecodedFrame::Malformed("oversized_frame"));
        } else if let Some(frame) = parse_frame(&self.frame) {
            frames.push(frame);
        }
        self.frame.clear();
        self.oversized = false;
    }
}

fn parse_frame(frame: &[u8]) -> Option<DecodedFrame> {
    let Ok(frame) = std::str::from_utf8(frame) else {
        return Some(DecodedFrame::Malformed("invalid_utf8"));
    };
    let mut event = None;
    let mut data = Vec::new();
    for line in frame.lines() {
        if line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = Some(value.to_string()),
            "data" => data.push(value),
            _ => {}
        }
    }
    if data.is_empty() {
        return None;
    }
    Some(DecodedFrame::Event(SseEvent {
        event,
        data: data.join("\n"),
    }))
}

/// Renders [`ReducerEvent`]s as Anthropic Messages SSE.
pub struct StreamRenderer {
    message_id: String,
    model: String,
    estimated_input_tokens: u64,
    started: bool,
}

impl StreamRenderer {
    /// `estimated_input_tokens` seeds the client's usage counter in
    /// `message_start` until the terminal `message_delta` carries the real
    /// count.
    pub fn new(message_id: String, model: String, estimated_input_tokens: u64) -> Self {
        Self {
            message_id,
            model,
            estimated_input_tokens,
            started: false,
        }
    }

    pub fn render(&mut self, events: &[ReducerEvent]) -> Vec<u8> {
        let mut out = Vec::new();
        for event in events {
            // The reader turns a failure into the client's error, and the
            // allowance frame is never shown to the client.
            if matches!(
                event,
                ReducerEvent::SubscriptionUsage(_) | ReducerEvent::Failed(_)
            ) {
                continue;
            }
            if !self.started {
                self.started = true;
                emit(
                    &mut out,
                    "message_start",
                    json!({"type":"message_start","message":{"id":self.message_id,"type":"message","role":"assistant","model":self.model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":self.estimated_input_tokens,"output_tokens":0}}}),
                );
            }
            render_event(&mut out, event);
        }
        out
    }
}

/// The Anthropic `error` SSE event for a failed stream, with the same type and
/// message (including the upstream status) as the non-streaming error.
pub fn error_event(error: &MuseError) -> Vec<u8> {
    let error_type = error.clone().into_provider_error().error_type();
    let data =
        json!({"type":"error","error":{"type":error_type,"message":error.message_with_status()}});
    encode_sse_event(Some("error"), &data.to_string())
}

fn emit(out: &mut Vec<u8>, event: &str, data: Value) {
    out.extend(encode_sse_event(Some(event), &data.to_string()));
}

fn render_event(out: &mut Vec<u8>, event: &ReducerEvent) {
    match event {
        ReducerEvent::ThinkingStart(index) => emit(
            out,
            "content_block_start",
            json!({"type":"content_block_start","index":index,"content_block":{"type":"thinking","thinking":"","signature":""}}),
        ),
        ReducerEvent::ThinkingDelta(index, text) => emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":text}}),
        ),
        ReducerEvent::ThinkingSignature(index, signature) => emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"signature_delta","signature":signature}}),
        ),
        ReducerEvent::TextStart(index) => emit(
            out,
            "content_block_start",
            json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}),
        ),
        ReducerEvent::TextDelta(index, text) => emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
        ),
        ReducerEvent::ToolStart(index, id, name) => emit(
            out,
            "content_block_start",
            json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}),
        ),
        ReducerEvent::ToolDelta(index, json) => emit(
            out,
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":json}}),
        ),
        ReducerEvent::ThinkingStop(index)
        | ReducerEvent::TextStop(index)
        | ReducerEvent::ToolStop(index) => emit(
            out,
            "content_block_stop",
            json!({"type":"content_block_stop","index":index}),
        ),
        ReducerEvent::Finish {
            stop_reason,
            input_tokens,
            output_tokens,
            ..
        } => {
            // Input usage is published only when the upstream reported it,
            // so the client keeps the seeded estimate otherwise.
            let mut usage = json!({"output_tokens":output_tokens});
            if let Some(input_tokens) = input_tokens {
                usage["input_tokens"] = json!(input_tokens);
            }
            emit(
                out,
                "message_delta",
                json!({"type":"message_delta","delta":{"stop_reason":stop_reason,"stop_sequence":null},"usage":usage}),
            );
            emit(out, "message_stop", json!({"type":"message_stop"}));
        }
        ReducerEvent::Failed(_) | ReducerEvent::SubscriptionUsage(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::*;
    use super::super::reducer::Reducer;
    use super::*;
    use crate::anthropic::sse::parse_sse_events;

    fn decode_all(input: &[u8]) -> Vec<DecodedFrame> {
        SseDecoder::default().push(input)
    }

    fn render(stream: &str) -> Vec<Value> {
        let mut reducer = Reducer::new(vec![]);
        let mut renderer = StreamRenderer::new("msg_1".into(), "muse-spark-1.3".into(), 99);
        let mut out = Vec::new();
        for frame in SseDecoder::default().push(stream.as_bytes()) {
            let DecodedFrame::Event(event) = frame else {
                continue;
            };
            let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            out.extend(renderer.render(&reducer.push(&value).expect("valid")));
        }
        parse_sse_events(&out)
            .into_iter()
            .map(|event| serde_json::from_str(&event.data).expect("rendered JSON"))
            .collect()
    }

    #[test]
    fn decoder_accepts_every_chunk_boundary_and_line_ending() {
        let input = b": note\r\nevent: response.output_text.delta\r\ndata: first\r\ndata: second\r\n\r\ndata: [DONE]\n\n";
        let expected = vec![
            DecodedFrame::Event(SseEvent {
                event: Some("response.output_text.delta".into()),
                data: "first\nsecond".into(),
            }),
            DecodedFrame::Event(SseEvent {
                event: None,
                data: "[DONE]".into(),
            }),
        ];
        for split in 0..=input.len() {
            let mut decoder = SseDecoder::default();
            let mut frames = decoder.push(&input[..split]);
            frames.extend(decoder.push(&input[split..]));
            assert_eq!(frames, expected, "split at {split}");
        }
    }

    #[test]
    fn decoder_reports_unreadable_frames_and_keeps_going() {
        let mut input = b"data: \xff\n\n".to_vec();
        input.extend_from_slice(b"data: ");
        input.extend(std::iter::repeat_n(b'x', MAX_SSE_FRAME_BYTES + 10));
        input.extend_from_slice(b"\n\nid: 7\n\ndata: after\n\n");
        let frames = decode_all(&input);
        assert_eq!(
            frames,
            vec![
                DecodedFrame::Malformed("invalid_utf8"),
                DecodedFrame::Malformed("oversized_frame"),
                DecodedFrame::Event(SseEvent {
                    event: None,
                    data: "after".into(),
                }),
            ]
        );
    }

    #[test]
    fn observed_text_stream_renders_the_anthropic_sequence_without_the_allowance() {
        let events = render(&(join(TEXT_ONLY) + &usage_frame() + DONE));
        let types: Vec<&str> = events
            .iter()
            .filter_map(|event| event["type"].as_str())
            .collect();
        assert_eq!(
            types,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[0]["message"]["usage"]["input_tokens"], 99);
        assert_eq!(events[0]["message"]["model"], "muse-spark-1.3");
        assert_eq!(events[2]["delta"]["text"], "ok");
        assert_eq!(events[4]["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[4]["usage"]["input_tokens"], 14);
        assert_eq!(events[4]["usage"]["output_tokens"], 307);
        let rendered = serde_json::to_string(&events).expect("json");
        assert!(!rendered.contains("subscription"), "{rendered}");
        assert!(!rendered.contains("used_percent"), "{rendered}");
    }

    #[test]
    fn observed_function_call_renders_tool_use_with_the_call_id() {
        let events = render(&(join(FUNCTION_CALL) + &synthetic_completed(3, 4)));
        assert_eq!(
            events[1]["content_block"],
            json!({"type":"tool_use","id":"call_01a1c315c9c87310a50529eaae77c51e","name":"get_weather","input":{}})
        );
        assert_eq!(events[2]["delta"]["type"], "input_json_delta");
        assert_eq!(events[2]["delta"]["partial_json"], "{\"city\":\"Paris\"}");
        assert_eq!(events[4]["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn synthetic_signed_reasoning_renders_signature_before_block_stop() {
        let stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0)
            + &join(SUMMARY_PART)
            + &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, "blob")
            + &synthetic_completed(1, 1);
        let events = render(&stream);
        assert_eq!(events[1]["content_block"]["type"], "thinking");
        assert_eq!(events[2]["delta"]["type"], "thinking_delta");
        assert_eq!(events[3]["delta"]["type"], "signature_delta");
        assert!(
            events[3]["delta"]["signature"]
                .as_str()
                .is_some_and(|signature| signature.starts_with("ccp:muse:v1:"))
        );
        assert_eq!(events[4]["type"], "content_block_stop");
    }

    #[test]
    fn missing_input_usage_is_not_published_as_zero() {
        let data = r#"{"type": "response.completed", "response": {"usage": {"output_tokens": 2}}}"#;
        let events = render(&frame("response.completed", data));
        let delta = events
            .iter()
            .find(|event| event["type"] == "message_delta")
            .expect("message_delta");
        assert!(delta["usage"].get("input_tokens").is_none(), "{delta}");
        assert_eq!(delta["usage"]["output_tokens"], 2);
    }

    #[test]
    fn a_failure_is_left_to_the_reader() {
        let events = render(&frame(
            "response.failed",
            r#"{"type": "response.failed", "response": {"error": {"message": "overloaded"}}}"#,
        ));
        assert_eq!(events, Vec::<Value>::new());
    }

    #[test]
    fn error_events_carry_the_upstream_status() {
        let events: Vec<Value> = [
            error_event(&MuseError::stream_ended()),
            error_event(&MuseError::stream_failed("Muse stream failed: overloaded")),
        ]
        .iter()
        .flat_map(|bytes| parse_sse_events(bytes))
        .map(|event| serde_json::from_str(&event.data).expect("JSON"))
        .collect();
        assert_eq!(
            events,
            vec![
                json!({"type":"error","error":{"type":"api_error","message":"Muse stream ended before completion (upstream HTTP 200)"}}),
                json!({"type":"error","error":{"type":"api_error","message":"Muse stream failed: overloaded (upstream HTTP 200)"}}),
            ]
        );
    }
}
