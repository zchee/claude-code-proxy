pub mod accumulate;
pub mod model_allowlist;
pub mod reader;
pub mod reasoning_signature;
pub mod reducer;
pub mod request;
pub mod stream;

/// Upstream SSE frames recorded from the Meta Responses API, shared by the
/// translator tests. Frames are copied from the wire record; only the long
/// strings the record itself abbreviates stay abbreviated.
#[cfg(test)]
pub(crate) mod fixtures {
    pub fn frame(event: &str, data: &str) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    pub const DONE: &str = "data: [DONE]\n\n";

    pub const USAGE: &str = r#"{"subscription": {"tier": "<tier_id numeric-string len=17>", "weekly": {"resets_at": 1794787200, "used_percent": 0}, "window": {"resets_at": 1794291326, "used_percent": 1, "window_duration_mins": 300}}, "type": "response.subscription_usage"}"#;

    pub fn usage_frame() -> String {
        frame("response.subscription_usage", USAGE)
    }

    /// The text-only `muse-spark-1.3` stream, up to and including
    /// `response.completed`. Its request had no `include`, so the reasoning
    /// item carries no `encrypted_content`.
    pub const TEXT_ONLY: &[(&str, &str)] = &[
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
    ];

    /// The four function-call frames of the observed tool round.
    pub const FUNCTION_CALL: &[(&str, &str)] = &[
        (
            "response.output_item.added",
            r#"{"item": {"arguments": "", "call_id": "call_01a1c315c9c87310a50529eaae77c51e", "id": "fc_01a1c315c9c87310a50529eaae77c51e", "name": "get_weather", "status": "in_progress", "type": "function_call"}, "output_index": 2, "sequence_number": 10, "type": "response.output_item.added"}"#,
        ),
        (
            "response.function_call_arguments.delta",
            r#"{"delta": "{\"city\":\"Paris\"}", "item_id": "fc_01a1c315c9c87310a50529eaae77c51e", "output_index": 2, "sequence_number": 11, "type": "response.function_call_arguments.delta"}"#,
        ),
        (
            "response.function_call_arguments.done",
            r#"{"arguments": "{\"city\":\"Paris\"}", "item_id": "fc_01a1c315c9c87310a50529eaae77c51e", "name": "get_weather", "output_index": 2, "sequence_number": 12, "type": "response.function_call_arguments.done"}"#,
        ),
        (
            "response.output_item.done",
            r#"{"item": {"arguments": "{\"city\":\"Paris\"}", "call_id": "call_01a1c315c9c87310a50529eaae77c51e", "id": "fc_01a1c315c9c87310a50529eaae77c51e", "name": "get_weather", "status": "completed", "type": "function_call"}, "output_index": 2, "sequence_number": 13, "type": "response.output_item.done"}"#,
        ),
    ];

    /// The four reasoning-summary frames of one observed summary part.
    pub const SUMMARY_PART: &[(&str, &str)] = &[
        (
            "response.reasoning_summary_part.added",
            r#"{"item_id": "rs_6af27444d8ab5b8d5358431e:rs_01a1c3162b7b75ee90fb1351f0f7e8ba", "output_index": 0, "part": {"text": "", "type": "summary_text"}, "sequence_number": 3, "summary_index": 0, "type": "response.reasoning_summary_part.added"}"#,
        ),
        (
            "response.reasoning_summary_text.delta",
            r#"{"delta": "Applying one-sentence arithmetic reasoning to the bat-and-ball problem under output formatting constraints.", "item_id": "rs_6af27444d8ab5b8d5358431e:rs_01a1c3162b7b75ee90fb1351f0f7e8ba", "output_index": 0, "sequence_number": 4, "summary_index": 0, "type": "response.reasoning_summary_text.delta"}"#,
        ),
        (
            "response.reasoning_summary_text.done",
            r#"{"item_id": "rs_6af27444d8ab5b8d5358431e:rs_01a1c3162b7b75ee90fb1351f0f7e8ba", "output_index": 0, "sequence_number": 5, "summary_index": 0, "text": "Applying one-sentence arithmetic reasoning to the bat-and-ball problem under output formatting constraints.", "type": "response.reasoning_summary_text.done"}"#,
        ),
        (
            "response.reasoning_summary_part.done",
            r#"{"item_id": "rs_6af27444d8ab5b8d5358431e:rs_01a1c3162b7b75ee90fb1351f0f7e8ba", "output_index": 0, "part": {"text": "Applying one-sentence arithmetic reasoning to the bat-and-ball problem under output formatting constraints.", "type": "summary_text"}, "sequence_number": 6, "summary_index": 0, "type": "response.reasoning_summary_part.done"}"#,
        ),
    ];

    pub const SUMMARY_ITEM_ID: &str =
        "rs_6af27444d8ab5b8d5358431e:rs_01a1c3162b7b75ee90fb1351f0f7e8ba";
    pub const SUMMARY_TEXT: &str = "Applying one-sentence arithmetic reasoning to the bat-and-ball problem under output formatting constraints.";

    /// The observed `response.incomplete` terminal of a `max_output_tokens: 16`
    /// request.
    pub const INCOMPLETE: &str = r#"{"response": {"background": false, "created_at": 1794274464.0, "error": null, "id": "resp_6af274a0aeedd7aeeddd44e4", "incomplete_details": {"reason": "max_output_tokens"}, "instructions": "<instructions omitted: NoneType>", "max_output_tokens": 16, "model": "muse-spark-1.3", "object": "response", "output": [{"encrypted_content": "<encrypted_content opaque str len=1426>", "id": "rs_6af274a0aeedd7aeeddd44e4:rs_01a1c31794f476dfb01deceb9b9be238", "status": "incomplete", "summary": [], "type": "reasoning"}], "parallel_tool_calls": true, "reasoning": {"effort": "minimal", "summary": "auto"}, "service_tier": "default", "status": "incomplete", "store": false, "temperature": 1.0, "tool_choice": "auto", "tools": "<tools omitted: list>", "top_logprobs": 0, "top_p": 1.0, "truncation": "disabled", "usage": {"input_tokens": 20, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 16, "output_tokens_details": {"reasoning_tokens": 14}, "total_tokens": 36}}, "sequence_number": 4, "type": "response.incomplete"}"#;

    pub fn join(frames: &[(&str, &str)]) -> String {
        frames
            .iter()
            .map(|(event, data)| frame(event, data))
            .collect()
    }

    /// A `response.completed` frame built from the observed one with its
    /// output list emptied, for streams composed in tests.
    pub fn synthetic_completed(input_tokens: u64, output_tokens: u64) -> String {
        frame(
            "response.completed",
            &format!(
                r#"{{"response": {{"id": "resp_6af2741e267d9e80b98c465e", "model": "muse-spark-1.3", "object": "response", "output": [], "status": "completed", "error": null, "incomplete_details": null, "usage": {{"input_tokens": {input_tokens}, "input_tokens_details": {{"cached_tokens": 0}}, "output_tokens": {output_tokens}, "output_tokens_details": {{"reasoning_tokens": 0}}, "total_tokens": {}}}}}, "sequence_number": 99, "type": "response.completed"}}"#,
                input_tokens + output_tokens
            ),
        )
    }

    /// A reasoning `output_item.done` with an encrypted blob, as the record
    /// describes for requests that send `include`.
    pub fn synthetic_reasoning_done(id: &str, output_index: u64, blob: &str) -> String {
        frame(
            "response.output_item.done",
            &format!(
                r#"{{"item": {{"encrypted_content": "{blob}", "id": "{id}", "status": "completed", "summary": [], "type": "reasoning"}}, "output_index": {output_index}, "sequence_number": 7, "type": "response.output_item.done"}}"#
            ),
        )
    }

    pub fn synthetic_reasoning_added(id: &str, output_index: u64) -> String {
        frame(
            "response.output_item.added",
            &format!(
                r#"{{"item": {{"id": "{id}", "status": "in_progress", "summary": [], "type": "reasoning"}}, "output_index": {output_index}, "sequence_number": 2, "type": "response.output_item.added"}}"#
            ),
        )
    }
}
