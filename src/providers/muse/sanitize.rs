//! Scrubbing for strings that originate upstream and travel on to clients,
//! the monitor, traffic captures or logs. Key-name based capture redaction
//! does not reach values embedded inside prose, so every such string passes
//! through here first.

use std::sync::LazyLock;

use regex_lite::Regex;

const REDACTED: &str = "[redacted]";
const MAX_TEXT_CHARS: usize = 500;
const MAX_TOKEN_CHARS: usize = 128;

static EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[^\s@]+@[^\s@]+\.[^\s@]+").expect("e-mail pattern is a valid regex")
});

/// Replaces every occurrence of each non-empty secret with "[redacted]".
pub fn redact_secrets(text: &str, secrets: &[&str]) -> String {
    let mut out = text.to_string();
    for secret in secrets.iter().filter(|secret| !secret.is_empty()) {
        if out.contains(secret) {
            out = out.replace(secret, REDACTED);
        }
    }
    out
}

/// For upstream error prose: redact secrets, then redact e-mail shaped
/// substrings, then cut to at most 500 chars on a char boundary.
pub fn sanitize_upstream_text(text: &str, secrets: &[&str]) -> String {
    // Redacting before cutting matters: a secret that straddles the cut would
    // otherwise survive as a prefix that no longer matches the full value.
    let redacted = redact_secrets(text, secrets);
    let redacted = EMAIL.replace_all(&redacted, REDACTED);
    match redacted.char_indices().nth(MAX_TEXT_CHARS) {
        Some((end, _)) => redacted[..end].to_string(),
        None => redacted.into_owned(),
    }
}

/// For upstream error `code` / `param`: Some(value) only when the value is
/// 1..=128 chars of [A-Za-z0-9_.\-\[\]] and contains no secret; else None.
pub fn sanitize_upstream_token(value: &str, secrets: &[&str]) -> Option<String> {
    let well_formed = (1..=MAX_TOKEN_CHARS).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-[]".contains(&byte));
    let carries_secret = secrets
        .iter()
        .any(|secret| !secret.is_empty() && value.contains(secret));
    (well_formed && !carries_secret).then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCESS: &str = "synthetic0access0token0value0abcdefghijklmnopqrstuvwxyz";
    const API_KEY: &str = "synthetic0api0key0value0abcdefghijklmnopqrst";

    #[test]
    fn redact_secrets_replaces_every_occurrence_and_ignores_empty_secrets() {
        let text = format!("first {ACCESS} then {API_KEY} and {ACCESS} again");
        assert_eq!(
            redact_secrets(&text, &[ACCESS, "", API_KEY]),
            "first [redacted] then [redacted] and [redacted] again"
        );
        assert_eq!(redact_secrets("nothing secret", &[""]), "nothing secret");
    }

    #[test]
    fn sanitize_text_keeps_observed_upstream_messages_unchanged() {
        let observed = "The requested model was not found. It may not exist, or this API key may not have access to it. Check the model name; GET /v1/models lists the models this API key can use.";
        assert_eq!(
            sanitize_upstream_text(observed, &[ACCESS, API_KEY]),
            observed
        );
        let effort = "reasoning_effort 'none' is not supported for model 'muse-spark-1.3'. Supported values: [minimal, low, medium, high, xhigh, max]";
        assert_eq!(sanitize_upstream_text(effort, &[ACCESS, API_KEY]), effort);
    }

    #[test]
    fn sanitize_text_synthetic_redacts_tokens_keys_and_email_addresses() {
        let text =
            format!("token {ACCESS} key {API_KEY} owner someone.name+tag@example.co.jp done");
        let out = sanitize_upstream_text(&text, &[ACCESS, API_KEY]);
        assert_eq!(out, "token [redacted] key [redacted] owner [redacted] done");
        assert!(!out.contains('@'));
    }

    #[test]
    fn sanitize_text_synthetic_cuts_long_messages_to_500_chars() {
        let text = "x".repeat(600);
        assert_eq!(sanitize_upstream_text(&text, &[]).chars().count(), 500);
    }

    #[test]
    fn sanitize_text_synthetic_redacts_before_cutting_a_straddling_token() {
        // The token starts at char 490, so a cut at 500 would keep its first
        // ten chars if the cut happened before redaction.
        let text = format!("{}{ACCESS}{}", "y".repeat(490), "z".repeat(100));
        let out = sanitize_upstream_text(&text, &[ACCESS]);
        assert!(out.chars().count() <= 500);
        assert!(!out.contains(&ACCESS[..10]), "{out}");
        assert!(out.starts_with(&"y".repeat(490)));
    }

    #[test]
    fn sanitize_text_synthetic_cuts_on_a_char_boundary() {
        let text = "\u{3042}".repeat(600);
        let out = sanitize_upstream_text(&text, &[]);
        assert_eq!(out.chars().count(), 500);
        assert_eq!(out.len(), 500 * '\u{3042}'.len_utf8());
    }

    #[test]
    fn sanitize_token_keeps_observed_codes_and_params() {
        for value in [
            "invalid_api_key",
            "model_not_found",
            "method_not_allowed",
            "reasoning.effort",
            "model",
            "input[0].content[1]",
        ] {
            assert_eq!(
                sanitize_upstream_token(value, &[ACCESS, API_KEY]).as_deref(),
                Some(value)
            );
        }
    }

    #[test]
    fn sanitize_token_synthetic_drops_malformed_values() {
        let too_long = "a".repeat(129);
        for value in [
            "",
            "has space",
            "semi;colon",
            "someone@example.com",
            "quote\"",
            "\u{3042}",
            too_long.as_str(),
        ] {
            assert_eq!(sanitize_upstream_token(value, &[]), None, "{value:?}");
        }
        assert!(sanitize_upstream_token(&"a".repeat(128), &[]).is_some());
    }

    #[test]
    fn sanitize_token_synthetic_drops_values_carrying_a_secret() {
        assert_eq!(sanitize_upstream_token(API_KEY, &[ACCESS, API_KEY]), None);
        let wrapped = format!("key_{API_KEY}_x");
        assert_eq!(sanitize_upstream_token(&wrapped, &[API_KEY]), None);
        assert_eq!(
            sanitize_upstream_token("invalid_api_key", &[""]).as_deref(),
            Some("invalid_api_key")
        );
    }
}
