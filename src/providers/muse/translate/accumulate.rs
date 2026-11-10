use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::anthropic::sse::parse_sse_events;
use crate::providers::muse::client::MuseError;

const INVALID_RESPONSE: &str = "Muse response is invalid";

struct Block {
    value: Value,
    tool_arguments: Option<String>,
}

/// Assembles a non-streaming Anthropic Messages response from the SSE the
/// reader task rendered, so both response modes share one translation. The
/// reader reports its own failures; this only fails on SSE it cannot assemble,
/// including an `error` event.
pub fn accumulate_anthropic_sse(sse: &[u8]) -> Result<Value, MuseError> {
    let invalid = || MuseError::stream_failed(INVALID_RESPONSE);
    let mut message: Option<Map<String, Value>> = None;
    let mut blocks: BTreeMap<u64, Block> = BTreeMap::new();
    let mut stop_reason = None;
    let mut input_tokens = None;
    let mut output_tokens = 0;
    let mut stopped = false;
    for event in parse_sse_events(sse) {
        let value: Value = serde_json::from_str(&event.data).map_err(|_| invalid())?;
        match value.get("type").and_then(Value::as_str) {
            Some("error") => return Err(invalid()),
            Some("message_start") => {
                message = value
                    .get("message")
                    .and_then(Value::as_object)
                    .cloned()
                    .map(Some)
                    .ok_or_else(invalid)?;
            }
            Some("content_block_start") => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or_else(invalid)?;
                let block = value.get("content_block").cloned().ok_or_else(invalid)?;
                let tool_arguments = (block.get("type").and_then(Value::as_str)
                    == Some("tool_use"))
                .then(String::new);
                blocks.insert(
                    index,
                    Block {
                        value: block,
                        tool_arguments,
                    },
                );
            }
            Some("content_block_delta") => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or_else(invalid)?;
                let block = blocks.get_mut(&index).ok_or_else(invalid)?;
                apply_delta(block, value.get("delta").ok_or_else(invalid)?).ok_or_else(invalid)?;
            }
            Some("message_delta") => {
                stop_reason = value
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                input_tokens = value.pointer("/usage/input_tokens").and_then(Value::as_u64);
                output_tokens = value
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
            }
            Some("message_stop") => stopped = true,
            _ => {}
        }
    }
    let (Some(mut message), true, Some(stop_reason)) = (message, stopped, stop_reason) else {
        return Err(invalid());
    };
    let truncated = stop_reason == "max_tokens";
    let mut content = Vec::with_capacity(blocks.len());
    for block in blocks.into_values() {
        let mut value = block.value;
        if let Some(arguments) = block.tool_arguments {
            let input = if arguments.is_empty() {
                Ok(json!({}))
            } else {
                serde_json::from_str::<Value>(&arguments)
            };
            // A call cut off by the output limit has no complete input; the
            // streaming client sees the same partial call.
            value["input"] = match input {
                Ok(input) if input.is_object() => input,
                _ if truncated => json!({}),
                _ => return Err(invalid()),
            };
        }
        content.push(value);
    }
    message.insert("content".into(), Value::Array(content));
    message.insert("stop_reason".into(), Value::String(stop_reason));
    message.insert("stop_sequence".into(), Value::Null);
    // The non-streaming shape always carries an input count; an absent
    // upstream value is zero here, as in the other Responses providers.
    message.insert(
        "usage".into(),
        json!({"input_tokens": input_tokens.unwrap_or(0), "output_tokens": output_tokens}),
    );
    Ok(Value::Object(message))
}

fn apply_delta(block: &mut Block, delta: &Value) -> Option<()> {
    // Appends in place: copying the text so far for every delta would make a
    // long reply quadratic.
    let append = |value: &mut Value, key: &str, text: &str| -> Option<()> {
        let Value::String(field) = value.get_mut(key)? else {
            return None;
        };
        field.push_str(text);
        Some(())
    };
    match delta.get("type").and_then(Value::as_str)? {
        "text_delta" => append(&mut block.value, "text", delta.get("text")?.as_str()?),
        "thinking_delta" => append(
            &mut block.value,
            "thinking",
            delta.get("thinking")?.as_str()?,
        ),
        "signature_delta" => {
            block.value["signature"] = Value::String(delta.get("signature")?.as_str()?.to_string());
            Some(())
        }
        "input_json_delta" => {
            block
                .tool_arguments
                .as_mut()?
                .push_str(delta.get("partial_json")?.as_str()?);
            Some(())
        }
        _ => Some(()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::*;
    use super::super::reducer::Reducer;
    use super::super::stream::{DecodedFrame, SseDecoder, StreamRenderer, error_event};
    use super::*;

    fn rendered(stream: &str) -> Vec<u8> {
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
        out
    }

    #[test]
    fn observed_text_stream_assembles_a_message() {
        let message =
            accumulate_anthropic_sse(&rendered(&(join(TEXT_ONLY) + &usage_frame() + DONE)))
                .expect("assembles");
        assert_eq!(
            message,
            json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "muse-spark-1.3",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 14, "output_tokens": 307}
            })
        );
    }

    #[test]
    fn observed_function_call_assembles_tool_use_input() {
        let message = accumulate_anthropic_sse(&rendered(
            &(join(FUNCTION_CALL) + &synthetic_completed(3, 4)),
        ))
        .expect("assembles");
        assert_eq!(
            message["content"],
            json!([{
                "type": "tool_use",
                "id": "call_01a1c315c9c87310a50529eaae77c51e",
                "name": "get_weather",
                "input": {"city": "Paris"}
            }])
        );
        assert_eq!(message["stop_reason"], "tool_use");
    }

    #[test]
    fn synthetic_signed_reasoning_keeps_text_and_signature() {
        let stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0)
            + &join(SUMMARY_PART)
            + &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, "blob")
            + &join(&TEXT_ONLY[4..10])
            + &synthetic_completed(1, 1);
        let message = accumulate_anthropic_sse(&rendered(&stream)).expect("assembles");
        let thinking = &message["content"][0];
        assert_eq!(thinking["type"], "thinking");
        assert_eq!(thinking["thinking"], SUMMARY_TEXT);
        assert!(
            thinking["signature"]
                .as_str()
                .is_some_and(|signature| signature.starts_with("ccp:muse:v1:"))
        );
        assert_eq!(message["content"][1], json!({"type": "text", "text": "ok"}));
    }

    #[test]
    fn observed_incomplete_assembles_max_tokens() {
        let message = accumulate_anthropic_sse(&rendered(
            &(frame("response.incomplete", INCOMPLETE) + &usage_frame()),
        ))
        .expect("assembles");
        assert_eq!(message["stop_reason"], "max_tokens");
        assert_eq!(message["content"], json!([]));
        assert_eq!(
            message["usage"],
            json!({"input_tokens": 20, "output_tokens": 16})
        );
    }

    #[test]
    fn synthetic_many_small_deltas_assemble_the_whole_text_and_thinking() {
        let text_delta = TEXT_ONLY[6].1;
        let summary_delta = SUMMARY_PART[1].1;
        let mut stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0);
        for _ in 0..2_000 {
            stream += &frame("response.reasoning_summary_text.delta", summary_delta);
        }
        stream += &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, "blob");
        stream += &join(&TEXT_ONLY[4..6]);
        for _ in 0..2_000 {
            stream += &frame("response.output_text.delta", text_delta);
        }
        stream += &join(&TEXT_ONLY[7..10]);
        stream += &synthetic_completed(1, 1);
        let message = accumulate_anthropic_sse(&rendered(&stream)).expect("assembles");
        assert_eq!(
            message["content"][0]["thinking"],
            SUMMARY_TEXT.repeat(2_000)
        );
        assert_eq!(message["content"][1]["text"], "ok".repeat(2_000));
    }

    #[test]
    fn an_error_event_or_a_missing_message_stop_is_not_assembled() {
        let mut failed = rendered(&join(&TEXT_ONLY[4..7]));
        failed.extend(error_event(&MuseError::stream_ended()));
        let truncated = rendered(&join(&TEXT_ONLY[4..7]));
        for sse in [failed, truncated] {
            assert_eq!(
                accumulate_anthropic_sse(&sse),
                Err(MuseError::stream_failed(INVALID_RESPONSE))
            );
        }
    }
}
