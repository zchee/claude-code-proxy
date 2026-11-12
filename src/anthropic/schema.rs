use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip)]
    pub bypass_provider_model_override: bool,
    /// The values of every `anthropic-beta` request header, split on commas,
    /// trimmed, empty ones dropped, in the order received. Set by the server,
    /// never read from or written to the body.
    #[serde(skip)]
    pub anthropic_beta: Vec<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: u64,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn anthropic_beta_synthetic_never_reaches_the_body_of_any_provider() {
        let mut body: MessagesRequest = serde_json::from_value(json!({
            "model": "gpt-6-sol",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "anthropic_beta": ["synthetic-body-beta"]
        }))
        .expect("valid request");
        // A body key of the same name is an ordinary extra field, not the
        // header values.
        assert!(body.anthropic_beta.is_empty());
        assert_eq!(body.extra["anthropic_beta"], json!(["synthetic-body-beta"]));

        let before = serde_json::to_value(&body).expect("serialisable");
        body.anthropic_beta = vec![
            "claude-code-20250219".to_string(),
            "oauth-2025-04-20".to_string(),
        ];
        let after = serde_json::to_value(&body).expect("serialisable");
        // Every provider builds its upstream body from this serialisation,
        // so the header values change nothing for them.
        assert_eq!(after, before);
        assert_eq!(
            after,
            json!({
                "model": "gpt-6-sol",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false,
                "anthropic_beta": ["synthetic-body-beta"]
            })
        );
    }
}
