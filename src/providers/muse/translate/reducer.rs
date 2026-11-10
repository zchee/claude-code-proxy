use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use super::reasoning_signature::{ReasoningReplay, encode_reasoning_signature};
use crate::providers::muse::sanitize::{sanitize_upstream_text, sanitize_upstream_token};
use crate::providers::muse::usage::SubscriptionUsage;

const MAX_TOOL_ARGUMENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_OPEN_ITEMS: usize = 256;
/// Separates two reasoning summary parts inside one thinking block.
const SUMMARY_PART_SEPARATOR: &str = "\n\n";

/// Intermediate events between the Responses stream and the Anthropic
/// renderers. Indices are Anthropic content-block indices.
#[derive(Debug, Clone, PartialEq)]
pub enum ReducerEvent {
    ThinkingStart(usize),
    ThinkingDelta(usize, String),
    ThinkingSignature(usize, String),
    ThinkingStop(usize),
    TextStart(usize),
    TextDelta(usize, String),
    TextStop(usize),
    ToolStart(usize, String, String),
    ToolDelta(usize, String),
    ToolStop(usize),
    Finish {
        stop_reason: &'static str,
        /// `None` when the upstream omitted or malformed the field; a present
        /// zero is a real value.
        input_tokens: Option<u64>,
        output_tokens: u64,
        /// The `incomplete_details.reason` of a `response.incomplete` that is
        /// not `max_output_tokens`, already reduced to a safe token.
        unexpected_incomplete_reason: Option<String>,
    },
    /// A `response.failed` or `error` event. The message is sanitised.
    Failed(String),
    /// The allowance frame. Renderers never forward it to the client.
    SubscriptionUsage(SubscriptionUsage),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Open,
    Finished,
    Failed,
}

enum Item {
    Reasoning {
        block: Option<usize>,
        last_summary_index: Option<u64>,
        has_text: bool,
    },
    Message {
        /// Content index to Anthropic block index, for text blocks still open.
        open: BTreeMap<u64, usize>,
        /// Content indices that already produced text.
        streamed: Vec<u64>,
    },
    Call {
        index: usize,
        arguments: String,
    },
    Ignored,
}

/// Turns decoded Responses events into [`ReducerEvent`]s. Items are told
/// apart by `item_id`, falling back to `output_index`, so interleaved items
/// stay separate.
pub struct Reducer {
    secrets: Vec<String>,
    next_index: usize,
    items: HashMap<String, Item>,
    item_by_output_index: HashMap<u64, String>,
    saw_tool: bool,
    phase: Phase,
}

impl Reducer {
    /// `secrets` are the credential values this request used; they are
    /// removed from any upstream error text.
    pub fn new(secrets: Vec<String>) -> Self {
        Self {
            secrets,
            next_index: 0,
            items: HashMap::new(),
            item_by_output_index: HashMap::new(),
            saw_tool: false,
            phase: Phase::Open,
        }
    }

    /// True once the content-terminal event (`response.completed` or
    /// `response.incomplete`) has been processed.
    pub fn finished(&self) -> bool {
        self.phase == Phase::Finished
    }

    /// True once a `response.failed` or `error` event has been processed.
    pub fn failed(&self) -> bool {
        self.phase == Phase::Failed
    }

    /// Processes one decoded event. Unknown event types are ignored. After
    /// the stream has finished or failed only the allowance frame is still
    /// reported. An error means a known event broke the stream's structure.
    pub fn push(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let Some(kind) = value.get("type").and_then(Value::as_str) else {
            return Ok(vec![]);
        };
        if kind == "response.subscription_usage" {
            return Ok(value
                .get("subscription")
                .and_then(|usage| serde_json::from_value(usage.clone()).ok())
                .map(ReducerEvent::SubscriptionUsage)
                .into_iter()
                .collect());
        }
        if self.phase != Phase::Open {
            return Ok(vec![]);
        }
        match kind {
            "response.output_item.added" => self.item_added(value),
            "response.output_item.done" => self.item_done(value),
            "response.reasoning_summary_text.delta" => self.summary_delta(value),
            "response.output_text.delta" => self.text_delta(value),
            "response.output_text.done" => self.text_done(value),
            "response.function_call_arguments.delta" => self.arguments_delta(value),
            "response.function_call_arguments.done" => self.arguments_done(value),
            "response.completed" | "response.incomplete" => self.terminal(kind, value),
            "response.failed" => {
                let error = value.pointer("/response/error");
                Ok(vec![
                    self.fail(error.and_then(|error| error.get("message"))),
                ])
            }
            "error" => {
                let message = value
                    .get("message")
                    .or_else(|| value.pointer("/error/message"));
                Ok(vec![self.fail(message)])
            }
            _ => Ok(vec![]),
        }
    }

    fn fail(&mut self, message: Option<&Value>) -> ReducerEvent {
        self.phase = Phase::Failed;
        let secrets: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
        let detail = message
            .and_then(Value::as_str)
            .map(|message| sanitize_upstream_text(message, &secrets))
            .filter(|message| !message.is_empty());
        ReducerEvent::Failed(match detail {
            Some(detail) => format!("Muse stream failed: {detail}"),
            None => "Muse stream failed".to_string(),
        })
    }

    fn allocate(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn register(
        &mut self,
        id: String,
        output_index: Option<u64>,
        item: Item,
    ) -> anyhow::Result<()> {
        if self.items.contains_key(&id) {
            anyhow::bail!("duplicate output item id");
        }
        if self.items.len() >= MAX_OPEN_ITEMS {
            anyhow::bail!("too many open output items");
        }
        if let Some(output_index) = output_index {
            self.item_by_output_index.insert(output_index, id.clone());
        }
        self.items.insert(id, item);
        Ok(())
    }

    /// Finds the item an event refers to. Text and reasoning deltas for an
    /// item that was never announced create it, so a stream that skips the
    /// `output_item.added` frame still renders.
    fn item_key(&mut self, value: &Value, create: impl FnOnce() -> Item) -> anyhow::Result<String> {
        let output_index = value.get("output_index").and_then(Value::as_u64);
        let key = match value.get("item_id").and_then(Value::as_str) {
            Some(id) => Some(id.to_string()),
            None => output_index.and_then(|index| self.item_by_output_index.get(&index).cloned()),
        };
        match key {
            Some(key) if self.items.contains_key(&key) => Ok(key),
            Some(key) => {
                self.register(key.clone(), output_index, create())?;
                Ok(key)
            }
            None => {
                let key = format!("output_index:{}", output_index.unwrap_or(u64::MAX));
                if !self.items.contains_key(&key) {
                    self.register(key.clone(), output_index, create())?;
                }
                Ok(key)
            }
        }
    }

    fn item_added(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let item = value
            .get("item")
            .filter(|item| item.is_object())
            .ok_or_else(|| anyhow::anyhow!("output item is invalid"))?;
        let output_index = value.get("output_index").and_then(Value::as_u64);
        let id = item_id(item, output_index);
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => {
                self.register(id, output_index, new_reasoning())?;
                Ok(vec![])
            }
            Some("message") => {
                self.register(id, output_index, new_message())?;
                Ok(vec![])
            }
            Some("function_call") => {
                let (call_id, name) = call_identity(item)?;
                let index = self.allocate();
                self.register(
                    id,
                    output_index,
                    Item::Call {
                        index,
                        arguments: String::new(),
                    },
                )?;
                self.saw_tool = true;
                Ok(vec![ReducerEvent::ToolStart(index, call_id, name)])
            }
            _ => {
                self.register(id, output_index, Item::Ignored)?;
                Ok(vec![])
            }
        }
    }

    fn summary_delta(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let delta = value
            .get("delta")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("reasoning summary delta is invalid"))?;
        let summary_index = value.get("summary_index").and_then(Value::as_u64);
        let key = self.item_key(value, new_reasoning)?;
        if delta.is_empty() {
            return Ok(vec![]);
        }
        let next_index = self.next_index;
        let Some(Item::Reasoning {
            block,
            last_summary_index,
            has_text,
        }) = self.items.get_mut(&key)
        else {
            anyhow::bail!("reasoning summary delta targets another item type");
        };
        let mut out = Vec::new();
        let index = match *block {
            Some(index) => index,
            None => {
                *block = Some(next_index);
                out.push(ReducerEvent::ThinkingStart(next_index));
                next_index
            }
        };
        if *has_text && summary_index != *last_summary_index {
            out.push(ReducerEvent::ThinkingDelta(
                index,
                SUMMARY_PART_SEPARATOR.to_string(),
            ));
        }
        *last_summary_index = summary_index;
        *has_text = true;
        out.push(ReducerEvent::ThinkingDelta(index, delta.to_string()));
        if index == next_index {
            self.next_index += 1;
        }
        Ok(out)
    }

    fn text_delta(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let delta = value
            .get("delta")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("text delta is invalid"))?;
        if delta.is_empty() {
            return Ok(vec![]);
        }
        let content_index = value
            .get("content_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let key = self.item_key(value, new_message)?;
        let next_index = self.next_index;
        let Some(Item::Message { open, streamed }) = self.items.get_mut(&key) else {
            anyhow::bail!("text delta targets another item type");
        };
        let mut out = Vec::new();
        let index = match open.get(&content_index) {
            Some(index) => *index,
            None => {
                open.insert(content_index, next_index);
                if !streamed.contains(&content_index) {
                    streamed.push(content_index);
                }
                out.push(ReducerEvent::TextStart(next_index));
                next_index
            }
        };
        if index == next_index {
            self.next_index += 1;
        }
        out.push(ReducerEvent::TextDelta(index, delta.to_string()));
        Ok(out)
    }

    /// Closes the text block of one content part. A part that arrives only
    /// as a completed text, with no deltas, is emitted whole.
    fn text_done(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let content_index = value
            .get("content_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let text = value.get("text").and_then(Value::as_str).unwrap_or("");
        let key = self.item_key(value, new_message)?;
        let next_index = self.next_index;
        let Some(Item::Message { open, streamed }) = self.items.get_mut(&key) else {
            anyhow::bail!("text completion targets another item type");
        };
        if let Some(index) = open.remove(&content_index) {
            return Ok(vec![ReducerEvent::TextStop(index)]);
        }
        if streamed.contains(&content_index) || text.is_empty() {
            return Ok(vec![]);
        }
        streamed.push(content_index);
        self.next_index += 1;
        Ok(vec![
            ReducerEvent::TextStart(next_index),
            ReducerEvent::TextDelta(next_index, text.to_string()),
            ReducerEvent::TextStop(next_index),
        ])
    }

    fn call_key(&self, value: &Value) -> anyhow::Result<String> {
        let key = match value.get("item_id").and_then(Value::as_str) {
            Some(id) => Some(id.to_string()),
            None => value
                .get("output_index")
                .and_then(Value::as_u64)
                .and_then(|index| self.item_by_output_index.get(&index).cloned()),
        };
        key.filter(|key| matches!(self.items.get(key), Some(Item::Call { .. })))
            .ok_or_else(|| anyhow::anyhow!("function call event is out of order"))
    }

    fn arguments_delta(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let delta = value
            .get("delta")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("function arguments delta is invalid"))?;
        let key = self.call_key(value)?;
        let Some(Item::Call { index, arguments }) = self.items.get_mut(&key) else {
            anyhow::bail!("function call event is out of order");
        };
        if arguments.len().saturating_add(delta.len()) > MAX_TOOL_ARGUMENT_BYTES {
            anyhow::bail!("function arguments exceed the size limit");
        }
        if delta.is_empty() {
            return Ok(vec![]);
        }
        arguments.push_str(delta);
        Ok(vec![ReducerEvent::ToolDelta(*index, delta.to_string())])
    }

    fn arguments_done(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let key = self.call_key(value)?;
        let Some(Item::Call { index, arguments }) = self.items.get_mut(&key) else {
            anyhow::bail!("function call event is out of order");
        };
        fill_arguments(
            *index,
            arguments,
            value.get("arguments").and_then(Value::as_str),
        )
    }

    fn item_done(&mut self, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let item = value
            .get("item")
            .filter(|item| item.is_object())
            .ok_or_else(|| anyhow::anyhow!("completed output item is invalid"))?;
        let output_index = value.get("output_index").and_then(Value::as_u64);
        let id = item_id(item, output_index);
        let known = self.items.remove(&id);
        if let Some(output_index) = output_index {
            self.item_by_output_index.remove(&output_index);
        }
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => Ok(self.reasoning_done(item, known)),
            Some("message") => Ok(self.message_done(item, known)),
            Some("function_call") => self.call_done(item, known),
            _ => Ok(vec![]),
        }
    }

    /// Closes a reasoning item's thinking block with its replay signature.
    /// An item with no summary text still becomes an empty thinking block
    /// when it carries a replayable blob; one with neither produces nothing.
    fn reasoning_done(&mut self, item: &Value, known: Option<Item>) -> Vec<ReducerEvent> {
        let block = match known {
            Some(Item::Reasoning { block, .. }) => block,
            _ => None,
        };
        let signature = replay_from_item(item)
            .as_ref()
            .and_then(encode_reasoning_signature);
        let mut out = Vec::new();
        let index = match (block, &signature) {
            (Some(index), _) => index,
            (None, Some(_)) => {
                let index = self.allocate();
                out.push(ReducerEvent::ThinkingStart(index));
                index
            }
            (None, None) => return out,
        };
        if let Some(signature) = signature {
            out.push(ReducerEvent::ThinkingSignature(index, signature));
        }
        out.push(ReducerEvent::ThinkingStop(index));
        out
    }

    fn message_done(&mut self, item: &Value, known: Option<Item>) -> Vec<ReducerEvent> {
        let (open, streamed) = match known {
            Some(Item::Message { open, streamed }) => (open, streamed),
            _ => (BTreeMap::new(), Vec::new()),
        };
        let mut out: Vec<ReducerEvent> = open.into_values().map(ReducerEvent::TextStop).collect();
        let parts = item
            .get("content")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for (content_index, part) in (0u64..).zip(parts) {
            let text = part.get("text").and_then(Value::as_str).unwrap_or("");
            if part.get("type").and_then(Value::as_str) != Some("output_text")
                || text.is_empty()
                || streamed.contains(&content_index)
            {
                continue;
            }
            let index = self.allocate();
            out.extend([
                ReducerEvent::TextStart(index),
                ReducerEvent::TextDelta(index, text.to_string()),
                ReducerEvent::TextStop(index),
            ]);
        }
        out
    }

    fn call_done(
        &mut self,
        item: &Value,
        known: Option<Item>,
    ) -> anyhow::Result<Vec<ReducerEvent>> {
        let mut out = Vec::new();
        let (index, mut arguments) = match known {
            Some(Item::Call { index, arguments }) => (index, arguments),
            _ => {
                let (call_id, name) = call_identity(item)?;
                let index = self.allocate();
                self.saw_tool = true;
                out.push(ReducerEvent::ToolStart(index, call_id, name));
                (index, String::new())
            }
        };
        out.extend(fill_arguments(
            index,
            &mut arguments,
            item.get("arguments").and_then(Value::as_str),
        )?);
        validate_arguments(&arguments)?;
        out.push(ReducerEvent::ToolStop(index));
        Ok(out)
    }

    fn terminal(&mut self, kind: &str, value: &Value) -> anyhow::Result<Vec<ReducerEvent>> {
        let incomplete = kind == "response.incomplete";
        let response = value.get("response").unwrap_or(value);
        let reason = response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str);
        let output_limit = incomplete && reason == Some("max_output_tokens");
        let mut closing = BTreeMap::new();
        for item in self.items.drain().map(|(_, item)| item) {
            match item {
                Item::Reasoning {
                    block: Some(index), ..
                } => {
                    closing.insert(index, ReducerEvent::ThinkingStop(index));
                }
                Item::Message { open, .. } => {
                    for index in open.into_values() {
                        closing.insert(index, ReducerEvent::TextStop(index));
                    }
                }
                Item::Call { index, arguments } => {
                    // Only a call cut off by the output limit may stay
                    // half-written (the client sees `max_tokens`); any other
                    // ending must not hand out invalid tool input.
                    if !output_limit {
                        validate_arguments(&arguments)?;
                    }
                    closing.insert(index, ReducerEvent::ToolStop(index));
                }
                Item::Reasoning { block: None, .. } | Item::Ignored => {}
            }
        }
        self.item_by_output_index.clear();
        let usage = response.get("usage");
        let input_tokens = usage
            .and_then(|usage| usage.get("input_tokens"))
            .and_then(Value::as_u64);
        let output_tokens = usage
            .and_then(|usage| usage.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let (stop_reason, unexpected_incomplete_reason) = match (incomplete, reason) {
            (true, Some("max_output_tokens")) => ("max_tokens", None),
            (true, reason) => {
                let secrets: Vec<&str> = self.secrets.iter().map(String::as_str).collect();
                (
                    "end_turn",
                    Some(
                        reason
                            .and_then(|reason| sanitize_upstream_token(reason, &secrets))
                            .unwrap_or_else(|| "unspecified".to_string()),
                    ),
                )
            }
            (false, _) if self.saw_tool => ("tool_use", None),
            (false, _) => ("end_turn", None),
        };
        self.phase = Phase::Finished;
        let mut out: Vec<ReducerEvent> = closing.into_values().collect();
        out.push(ReducerEvent::Finish {
            stop_reason,
            input_tokens,
            output_tokens,
            unexpected_incomplete_reason,
        });
        Ok(out)
    }
}

fn new_reasoning() -> Item {
    Item::Reasoning {
        block: None,
        last_summary_index: None,
        has_text: false,
    }
}

fn new_message() -> Item {
    Item::Message {
        open: BTreeMap::new(),
        streamed: Vec::new(),
    }
}

fn item_id(item: &Value, output_index: Option<u64>) -> String {
    match item.get("id").and_then(Value::as_str) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => format!("output_index:{}", output_index.unwrap_or(u64::MAX)),
    }
}

/// The Anthropic `tool_use` id is the upstream `call_id`, which is what a
/// `function_call_output` must name on the next turn.
fn call_identity(item: &Value) -> anyhow::Result<(String, String)> {
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| anyhow::anyhow!("function call id is invalid"))?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| anyhow::anyhow!("function call name is invalid"))?;
    Ok((call_id.to_string(), name.to_string()))
}

/// Uses the complete arguments of a `.done` frame when no delta arrived, and
/// rejects a complete value that contradicts the deltas.
fn fill_arguments(
    index: usize,
    arguments: &mut String,
    complete: Option<&str>,
) -> anyhow::Result<Vec<ReducerEvent>> {
    match complete {
        Some(complete) if arguments.is_empty() && !complete.is_empty() => {
            if complete.len() > MAX_TOOL_ARGUMENT_BYTES {
                anyhow::bail!("function arguments exceed the size limit");
            }
            arguments.push_str(complete);
            Ok(vec![ReducerEvent::ToolDelta(index, complete.to_string())])
        }
        Some(complete) if !complete.is_empty() && complete != arguments => {
            anyhow::bail!("function arguments disagree with their deltas")
        }
        _ => Ok(vec![]),
    }
}

/// Empty arguments stand for a call without input.
fn validate_arguments(arguments: &str) -> anyhow::Result<()> {
    if arguments.is_empty() || serde_json::from_str::<Value>(arguments).is_ok_and(|v| v.is_object())
    {
        Ok(())
    } else {
        anyhow::bail!("function arguments are not a JSON object")
    }
}

fn replay_from_item(item: &Value) -> Option<ReasoningReplay> {
    let id = item.get("id").and_then(Value::as_str)?;
    let encrypted_content = item.get("encrypted_content").and_then(Value::as_str)?;
    Some(ReasoningReplay {
        id: id.to_string(),
        encrypted_content: encrypted_content.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::*;
    use super::super::reasoning_signature::decode_reasoning_signature;
    use super::*;
    use crate::providers::muse::usage::UsageWindow;

    fn reduce(frames: &str) -> Vec<ReducerEvent> {
        reduce_with(&mut Reducer::new(vec![]), frames)
    }

    fn reduce_with(reducer: &mut Reducer, frames: &str) -> Vec<ReducerEvent> {
        let mut out = Vec::new();
        for event in crate::anthropic::sse::parse_sse_events(frames.as_bytes()) {
            let Ok(value) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            out.extend(reducer.push(&value).expect("valid event"));
        }
        out
    }

    fn observed_usage() -> SubscriptionUsage {
        SubscriptionUsage {
            window: Some(UsageWindow {
                used_percent: 1,
                resets_at: 1_794_291_326,
                window_duration_mins: Some(300),
            }),
            weekly: Some(UsageWindow {
                used_percent: 0,
                resets_at: 1_794_787_200,
                window_duration_mins: None,
            }),
        }
    }

    #[test]
    fn observed_text_only_stream_becomes_one_text_block() {
        let mut reducer = Reducer::new(vec![]);
        let events = reduce_with(&mut reducer, &(join(TEXT_ONLY) + &usage_frame() + DONE));
        assert_eq!(
            events,
            vec![
                ReducerEvent::TextStart(0),
                ReducerEvent::TextDelta(0, "ok".into()),
                ReducerEvent::TextStop(0),
                ReducerEvent::Finish {
                    stop_reason: "end_turn",
                    input_tokens: Some(14),
                    output_tokens: 307,
                    unexpected_incomplete_reason: None,
                },
                ReducerEvent::SubscriptionUsage(observed_usage()),
            ]
        );
        assert!(reducer.finished());
    }

    #[test]
    fn observed_function_call_frames_with_a_synthetic_terminal_become_tool_use() {
        let events = reduce(&(join(FUNCTION_CALL) + &synthetic_completed(40, 12)));
        assert_eq!(
            events,
            vec![
                ReducerEvent::ToolStart(
                    0,
                    "call_01a1c315c9c87310a50529eaae77c51e".into(),
                    "get_weather".into()
                ),
                ReducerEvent::ToolDelta(0, "{\"city\":\"Paris\"}".into()),
                ReducerEvent::ToolStop(0),
                ReducerEvent::Finish {
                    stop_reason: "tool_use",
                    input_tokens: Some(40),
                    output_tokens: 12,
                    unexpected_incomplete_reason: None,
                },
            ]
        );
    }

    #[test]
    fn observed_summary_part_with_a_synthetic_blob_becomes_a_signed_thinking_block() {
        let stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0)
            + &join(SUMMARY_PART)
            + &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, "gAAAA-blob")
            + &synthetic_completed(30, 50);
        let events = reduce(&stream);
        assert_eq!(events[0], ReducerEvent::ThinkingStart(0));
        assert_eq!(
            events[1],
            ReducerEvent::ThinkingDelta(0, SUMMARY_TEXT.into())
        );
        let ReducerEvent::ThinkingSignature(0, signature) = &events[2] else {
            panic!("expected a signature, got {events:?}");
        };
        let replay = decode_reasoning_signature(signature).expect("decodes");
        assert_eq!(replay.id, SUMMARY_ITEM_ID);
        assert_eq!(replay.encrypted_content, "gAAAA-blob");
        assert_eq!(events[3], ReducerEvent::ThinkingStop(0));
        assert!(matches!(
            events[4],
            ReducerEvent::Finish {
                stop_reason: "end_turn",
                ..
            }
        ));
    }

    #[test]
    fn synthetic_two_summary_parts_are_joined_by_a_blank_line() {
        let second = SUMMARY_PART
            .iter()
            .map(|(event, data)| {
                (
                    *event,
                    data.replace("\"summary_index\": 0", "\"summary_index\": 1")
                        .replace(SUMMARY_TEXT, "Second part."),
                )
            })
            .map(|(event, data)| frame(event, &data))
            .collect::<String>();
        let stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0)
            + &join(SUMMARY_PART)
            + &second
            + &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, "blob")
            + &synthetic_completed(1, 1);
        let text: String = reduce(&stream)
            .into_iter()
            .filter_map(|event| match event {
                ReducerEvent::ThinkingDelta(0, text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(text, format!("{SUMMARY_TEXT}\n\nSecond part."));
    }

    #[test]
    fn synthetic_reasoning_with_a_blob_and_no_summary_is_an_empty_signed_block() {
        let stream = synthetic_reasoning_added(SUMMARY_ITEM_ID, 0)
            + &synthetic_reasoning_done(SUMMARY_ITEM_ID, 0, "blob")
            + &synthetic_completed(1, 1);
        let events = reduce(&stream);
        assert_eq!(events[0], ReducerEvent::ThinkingStart(0));
        assert!(matches!(&events[1], ReducerEvent::ThinkingSignature(0, _)));
        assert_eq!(events[2], ReducerEvent::ThinkingStop(0));
    }

    #[test]
    fn observed_incomplete_terminal_maps_to_max_tokens() {
        let events = reduce(&(frame("response.incomplete", INCOMPLETE) + &usage_frame()));
        assert_eq!(
            events,
            vec![
                ReducerEvent::Finish {
                    stop_reason: "max_tokens",
                    input_tokens: Some(20),
                    output_tokens: 16,
                    unexpected_incomplete_reason: None,
                },
                ReducerEvent::SubscriptionUsage(observed_usage()),
            ]
        );
    }

    #[test]
    fn synthetic_incomplete_with_another_reason_ends_the_turn_and_reports_it() {
        let data = INCOMPLETE.replace("max_output_tokens\"}", "content_filter\"}");
        let events = reduce(&frame("response.incomplete", &data));
        assert_eq!(
            events,
            vec![ReducerEvent::Finish {
                stop_reason: "end_turn",
                input_tokens: Some(20),
                output_tokens: 16,
                unexpected_incomplete_reason: Some("content_filter".into()),
            }]
        );
    }

    #[test]
    fn synthetic_two_parallel_function_calls_stay_separate() {
        let call = |n: u64, delta: &str| {
            let added = frame(
                "response.output_item.added",
                &format!(
                    r#"{{"item": {{"arguments": "", "call_id": "call_{n}", "id": "fc_{n}", "name": "get_weather", "status": "in_progress", "type": "function_call"}}, "output_index": {n}, "type": "response.output_item.added"}}"#
                ),
            );
            let delta = frame(
                "response.function_call_arguments.delta",
                &format!(
                    r#"{{"delta": {}, "item_id": "fc_{n}", "output_index": {n}, "type": "response.function_call_arguments.delta"}}"#,
                    serde_json::to_string(delta).expect("string")
                ),
            );
            let done = frame(
                "response.output_item.done",
                &format!(
                    r#"{{"item": {{"call_id": "call_{n}", "id": "fc_{n}", "name": "get_weather", "type": "function_call"}}, "output_index": {n}, "type": "response.output_item.done"}}"#
                ),
            );
            (added, delta, done)
        };
        let (added_a, delta_a, done_a) = call(1, "{\"city\":\"Paris\"}");
        let (added_b, delta_b, done_b) = call(2, "{\"city\":\"Rome\"}");
        let stream = added_a
            + &added_b
            + &delta_b
            + &delta_a
            + &done_b
            + &done_a
            + &synthetic_completed(1, 1);
        let events = reduce(&stream);
        assert_eq!(
            events,
            vec![
                ReducerEvent::ToolStart(0, "call_1".into(), "get_weather".into()),
                ReducerEvent::ToolStart(1, "call_2".into(), "get_weather".into()),
                ReducerEvent::ToolDelta(1, "{\"city\":\"Rome\"}".into()),
                ReducerEvent::ToolDelta(0, "{\"city\":\"Paris\"}".into()),
                ReducerEvent::ToolStop(1),
                ReducerEvent::ToolStop(0),
                ReducerEvent::Finish {
                    stop_reason: "tool_use",
                    input_tokens: Some(1),
                    output_tokens: 1,
                    unexpected_incomplete_reason: None,
                },
            ]
        );
    }

    #[test]
    fn terminal_without_item_events_is_an_empty_successful_response() {
        // The hidden image model streamed only `response.created` and
        // `response.completed`.
        let events = reduce(&(join(&TEXT_ONLY[..1]) + &synthetic_completed(5, 0)));
        assert_eq!(
            events,
            vec![ReducerEvent::Finish {
                stop_reason: "end_turn",
                input_tokens: Some(5),
                output_tokens: 0,
                unexpected_incomplete_reason: None,
            }]
        );
    }

    #[test]
    fn synthetic_unknown_events_are_ignored() {
        let stream = frame(
            "response.audit",
            r#"{"type": "response.audit", "anything": [1, 2]}"#,
        ) + &frame("response.mystery", r#"{"no_type_field": true}"#)
            + &join(&TEXT_ONLY[4..])
            + &usage_frame();
        let events = reduce(&stream);
        assert_eq!(events.len(), 5, "{events:?}");
        assert!(matches!(events[3], ReducerEvent::Finish { .. }));
    }

    #[test]
    fn synthetic_failed_and_error_events_are_sanitised() {
        let token = "tok_abcdefghijklmnopqrstuvwxyz0123456789";
        let failed = frame(
            "response.failed",
            &format!(
                r#"{{"type": "response.failed", "response": {{"status": "failed", "error": {{"code": "server_error", "message": "key {token} owned by someone@example.com failed"}}}}}}"#
            ),
        );
        let mut reducer = Reducer::new(vec![token.to_string()]);
        let events = reduce_with(&mut reducer, &failed);
        assert_eq!(
            events,
            vec![ReducerEvent::Failed(
                "Muse stream failed: key [redacted] owned by [redacted] failed".into()
            )]
        );
        assert!(reducer.failed());
        assert!(!reducer.finished());

        for data in [
            format!(r#"{{"type": "error", "message": "bad {token}"}}"#),
            format!(r#"{{"type": "error", "error": {{"message": "bad {token}"}}}}"#),
        ] {
            let mut reducer = Reducer::new(vec![token.to_string()]);
            let events = reduce_with(&mut reducer, &frame("error", &data));
            assert_eq!(
                events,
                vec![ReducerEvent::Failed(
                    "Muse stream failed: bad [redacted]".into()
                )]
            );
        }

        let events = reduce(&frame("error", r#"{"type": "error"}"#));
        assert_eq!(
            events,
            vec![ReducerEvent::Failed("Muse stream failed".into())]
        );
    }

    #[test]
    fn synthetic_long_failure_text_is_cut_after_redaction() {
        let token = "secret-token-value-that-spans-the-cut";
        let message = format!("{}{token}", "x".repeat(490));
        let failed = frame(
            "response.failed",
            &serde_json::json!({
                "type": "response.failed",
                "response": {"error": {"message": message}}
            })
            .to_string(),
        );
        let events = reduce_with(&mut Reducer::new(vec![token.into()]), &failed);
        let [ReducerEvent::Failed(text)] = events.as_slice() else {
            panic!("expected a failure, got {events:?}");
        };
        assert!(!text.contains("secret-token"), "{text}");
        assert!(text.ends_with("[redacted]"), "{text}");
    }

    #[test]
    fn only_the_allowance_frame_is_reported_after_the_terminal() {
        let mut reducer = Reducer::new(vec![]);
        reduce_with(&mut reducer, &join(TEXT_ONLY));
        assert!(reducer.finished());
        let late = join(&TEXT_ONLY[4..7]) + &frame("error", r#"{"type": "error"}"#);
        assert_eq!(reduce_with(&mut reducer, &late), vec![]);
        assert_eq!(
            reduce_with(&mut reducer, &usage_frame()),
            vec![ReducerEvent::SubscriptionUsage(observed_usage())]
        );
    }

    #[test]
    fn synthetic_completed_response_with_a_half_written_call_is_an_error() {
        let stream = join(&FUNCTION_CALL[..1])
            + &frame(
                "response.function_call_arguments.delta",
                r#"{"delta": "{\"city\":", "item_id": "fc_01a1c315c9c87310a50529eaae77c51e", "output_index": 2, "type": "response.function_call_arguments.delta"}"#,
            );
        let mut reducer = Reducer::new(vec![]);
        reduce_with(&mut reducer, &stream);
        let completed: Value = serde_json::from_str(
            &synthetic_completed(1, 1)["event: response.completed\ndata: ".len()..],
        )
        .expect("json");
        assert!(reducer.push(&completed).is_err());

        // Another incomplete reason does not excuse the half-written call.
        let mut reducer = Reducer::new(vec![]);
        reduce_with(&mut reducer, &stream);
        let filtered: Value =
            serde_json::from_str(&INCOMPLETE.replace("max_output_tokens\"}", "content_filter\"}"))
                .expect("json");
        assert!(reducer.push(&filtered).is_err());

        // The same call cut off by the output limit closes without failing.
        let mut reducer = Reducer::new(vec![]);
        reduce_with(&mut reducer, &stream);
        let incomplete: Value = serde_json::from_str(INCOMPLETE).expect("json");
        let events = reducer.push(&incomplete).expect("incomplete closes");
        assert_eq!(events[0], ReducerEvent::ToolStop(0));
        assert!(matches!(
            events[1],
            ReducerEvent::Finish {
                stop_reason: "max_tokens",
                ..
            }
        ));
    }

    #[test]
    fn synthetic_arguments_done_without_deltas_supplies_the_arguments() {
        let stream =
            join(&FUNCTION_CALL[..1]) + &join(&FUNCTION_CALL[2..]) + &synthetic_completed(1, 1);
        let events = reduce(&stream);
        assert_eq!(
            events[1],
            ReducerEvent::ToolDelta(0, "{\"city\":\"Paris\"}".into())
        );
        assert_eq!(events[2], ReducerEvent::ToolStop(0));
    }

    #[test]
    fn synthetic_message_delivered_only_as_a_completed_item_is_still_rendered() {
        let stream = join(&TEXT_ONLY[9..]);
        let events = reduce(&stream);
        assert_eq!(
            &events[..3],
            &[
                ReducerEvent::TextStart(0),
                ReducerEvent::TextDelta(0, "ok".into()),
                ReducerEvent::TextStop(0),
            ]
        );
    }
}
