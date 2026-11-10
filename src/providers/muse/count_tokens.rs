use super::translate::request::{MuseContentPart, MuseInputItem, MuseResponsesRequest};

const MESSAGE_OVERHEAD_TOKENS: u64 = 4;
const TOOL_OVERHEAD_TOKENS: u64 = 4;
/// Images are charged a flat estimate instead of their base64 length.
const IMAGE_TOKEN_ESTIMATE: u64 = 1_024;

/// Estimates the input tokens of a translated request locally. The upstream
/// `/responses/input_tokens` count is not used: for a one-line prompt it
/// reported 171 where the billed input was 14.
pub fn count_tokens(request: &MuseResponsesRequest) -> u64 {
    let instructions = request
        .instructions
        .as_deref()
        .map_or(0, approx_token_count);
    let input: u64 = request.input.iter().map(count_input_item).sum();
    let tools: u64 = request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| {
            approx_token_count(&tool.name)
                + tool.description.as_deref().map_or(0, approx_token_count)
                + approx_token_count(&tool.parameters.to_string())
                + TOOL_OVERHEAD_TOKENS
        })
        .sum();
    (instructions
        + input
        + tools
        + request.input.len() as u64 * MESSAGE_OVERHEAD_TOKENS
        + approx_token_count(&request.model))
    .max(1)
}

fn count_input_item(item: &MuseInputItem) -> u64 {
    match item {
        MuseInputItem::Message { content, .. } => content
            .iter()
            .map(|part| match part {
                MuseContentPart::InputText { text } | MuseContentPart::OutputText { text } => {
                    approx_token_count(text)
                }
                MuseContentPart::InputImage { .. } => IMAGE_TOKEN_ESTIMATE,
            })
            .sum(),
        // The replayed blob is opaque; its length says nothing about how the
        // upstream counts it.
        MuseInputItem::Reasoning { .. } => 0,
        MuseInputItem::FunctionCall {
            name, arguments, ..
        } => approx_token_count(name) + approx_token_count(arguments),
        MuseInputItem::FunctionCallOutput { output, .. } => approx_token_count(output),
    }
}

fn approx_token_count(text: &str) -> u64 {
    let mut count = 0;
    let mut in_word = false;
    for character in text.chars() {
        if character.is_alphanumeric() || character == '-' || character == '_' {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !character.is_whitespace() {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::schema::MessagesRequest;
    use crate::providers::muse::translate::request::translate_request;
    use serde_json::json;

    fn translated(value: serde_json::Value) -> MuseResponsesRequest {
        let request: MessagesRequest = serde_json::from_value(value).expect("valid request");
        translate_request(&request, "muse-spark-1.3", None).expect("translates")
    }

    #[test]
    fn the_observed_one_line_prompt_counts_near_its_billed_input() {
        // The upstream billed 14 input tokens for this prompt.
        let count = count_tokens(&translated(json!({
            "model": "muse-spark-1.3",
            "messages": [{"role": "user", "content": "Reply with the single word ok"}]
        })));
        assert!((6..=40).contains(&count), "{count}");
    }

    #[test]
    fn more_input_counts_more_and_counting_is_deterministic() {
        let short = translated(json!({
            "model": "muse-spark-1.3",
            "messages": [{"role": "user", "content": "hello"}]
        }));
        let long = translated(json!({
            "model": "muse-spark-1.3",
            "system": "Follow all instructions carefully.",
            "tools": [{"name": "get_weather", "description": "Get the weather for a city",
                       "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What colour is this image?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
                ]}
            ]
        }));
        assert!(count_tokens(&long) > count_tokens(&short) + IMAGE_TOKEN_ESTIMATE);
        assert_eq!(count_tokens(&long), count_tokens(&long));
        assert!(count_tokens(&short) > 0);
    }

    #[test]
    fn tool_calls_and_results_are_counted() {
        let without = translated(json!({
            "model": "muse-spark-1.3",
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
        }));
        let with = translated(json!({
            "model": "muse-spark-1.3",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_01a1c315c9c87310a50529eaae77c51e",
                     "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_01a1c315c9c87310a50529eaae77c51e",
                     "content": "{\"temp_c\": 11, \"sky\": \"cloudy\"}"}
                ]}
            ]
        }));
        assert!(count_tokens(&with) > count_tokens(&without));
    }
}
