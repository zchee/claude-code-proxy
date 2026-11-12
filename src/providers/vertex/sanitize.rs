//! Scrubbing for prose that Vertex AI sends back (HTTP error messages,
//! in-stream errors, `finishMessage`) before it reaches clients, the
//! monitor, traffic captures or logs. Google error messages can quote the
//! request, so besides the access token they may carry a thought signature
//! or inline image data that the proxy sent.

use std::sync::LazyLock;

use regex_lite::{Captures, Regex};

use crate::providers::muse::sanitize::{redact_secrets, sanitize_upstream_text};

/// Shortest run of base64 characters that is treated as encoded data.
pub const MIN_ENCODED_RUN: usize = 64;

/// Runs of the standard and URL-safe base64 alphabets, with padding.
static BASE64_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z0-9+/_-]{64,}={0,2}").expect("base64 run pattern is a valid regex")
});

/// Scrubs one piece of upstream prose, in this order: every occurrence of
/// each secret (pass the access token) becomes `[redacted]`; every run of 64
/// or more base64 characters that mixes upper case, lower case and digits
/// becomes `[redacted len=N]`; e-mail addresses are redacted and the result
/// is cut to 500 characters by `muse::sanitize::sanitize_upstream_text`.
///
/// The mixed-character rule keeps resource names such as
/// `projects/my-project/locations/global/publishers/google/models/gemini-x`,
/// which use the same alphabet but no upper case, and which are the useful
/// part of a not-found message. Encoded data of that length virtually always
/// has all three kinds of character.
///
/// Both replacements happen before the cut, so a value that straddles the
/// 500th character cannot survive as a prefix.
pub fn sanitize_vertex_text(text: &str, secrets: &[&str]) -> String {
    let redacted = redact_secrets(text, secrets);
    let without_data = BASE64_RUN.replace_all(&redacted, |captures: &Captures| {
        let run = &captures[0];
        if looks_encoded(run) {
            format!("[redacted len={}]", run.len())
        } else {
            run.to_string()
        }
    });
    sanitize_upstream_text(&without_data, secrets)
}

fn looks_encoded(run: &str) -> bool {
    let bytes = run.as_bytes();
    bytes.iter().any(u8::is_ascii_uppercase)
        && bytes.iter().any(u8::is_ascii_lowercase)
        && bytes.iter().any(u8::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like a Google OAuth access token: `ya29.` and a long
    /// URL-safe body.
    const SYNTHETIC_ACCESS_TOKEN: &str =
        "ya29.a0Synthetic0Access0Token0Body0With0Mixed0Case0And0Digits0xYz9_Kq-Lm3";

    /// 4 KiB of synthetic bytes, base64-encoded the way an `inlineData.data`
    /// field carries an image.
    fn synthetic_image_base64() -> String {
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        (0..4096usize)
            .map(|index| alphabet[(index * 37 + index / 7) % 64] as char)
            .collect()
    }

    /// A Gemini thought signature is opaque base64; this one is synthetic.
    fn synthetic_signature() -> String {
        "CiQBjz1rX".to_string() + &"Ab3dEf9GhIjK0lMn".repeat(12) + "=="
    }

    #[test]
    fn sanitize_synthetic_echoed_access_token_is_redacted() {
        let text = format!(
            "Request had invalid authentication credentials. Got Bearer {SYNTHETIC_ACCESS_TOKEN} instead."
        );
        assert_eq!(
            sanitize_vertex_text(&text, &[SYNTHETIC_ACCESS_TOKEN]),
            "Request had invalid authentication credentials. Got Bearer [redacted] instead."
        );
    }

    #[test]
    fn sanitize_synthetic_echoed_image_data_is_replaced_with_its_length() {
        let image = synthetic_image_base64();
        let text = format!(
            "Unable to process input image. Please retry or report in https://developers.generativeai.google/guide/troubleshooting. data: \"{image}\""
        );
        let out = sanitize_vertex_text(&text, &[]);
        assert_eq!(
            out,
            "Unable to process input image. Please retry or report in https://developers.generativeai.google/guide/troubleshooting. data: \"[redacted len=4096]\""
        );
    }

    #[test]
    fn sanitize_synthetic_echoed_signature_keeps_the_envelope_prefix() {
        let signature = synthetic_signature();
        let text = format!(
            "Invalid value at 'contents[1].parts[0].thought_signature' (TYPE_BYTES), \"ccp:vertex:v1:gemini-3.5-flash:{signature}\""
        );
        let out = sanitize_vertex_text(&text, &[]);
        assert_eq!(
            out,
            format!(
                "Invalid value at 'contents[1].parts[0].thought_signature' (TYPE_BYTES), \"ccp:vertex:v1:gemini-3.5-flash:[redacted len={}]\"",
                signature.len()
            )
        );
        assert!(!out.contains(&signature[..16]), "{out}");
    }

    #[test]
    fn sanitize_synthetic_resource_names_are_kept() {
        let text = "Publisher Model `projects/synthetic-project-123/locations/global/publishers/google/models/gemini-3.5-flash-lite` was not found or your project does not have access to it.";
        assert_eq!(sanitize_vertex_text(text, &[]), text);
    }

    #[test]
    fn sanitize_synthetic_lowercase_hex_digests_are_kept() {
        let text = format!("digest {}", "0123456789abcdef".repeat(4));
        assert_eq!(sanitize_vertex_text(&text, &[]), text);
    }

    #[test]
    fn sanitize_synthetic_length_boundary_is_64_characters() {
        let unit = "Ab3dEf9GhIjK0lMn";
        let sixty_three = unit.repeat(4)[..63].to_string();
        let sixty_four = unit.repeat(4);
        assert_eq!(sanitize_vertex_text(&sixty_three, &[]), sixty_three);
        assert_eq!(sanitize_vertex_text(&sixty_four, &[]), "[redacted len=64]");
        // Padding belongs to the run.
        assert_eq!(
            sanitize_vertex_text(&format!("{sixty_four}=="), &[]),
            "[redacted len=66]"
        );
    }

    #[test]
    fn sanitize_synthetic_data_straddling_the_cut_leaves_no_prefix() {
        let image = synthetic_image_base64();
        // Spaces keep the prefix out of the run: the data starts at char 480
        // and runs far past the 500-char cut.
        let prefix = "y ".repeat(240);
        let text = format!("{prefix}{image}");
        let out = sanitize_vertex_text(&text, &[]);
        assert_eq!(out, format!("{prefix}[redacted len=4096]"));
        assert!(!out.contains(&image[..8]), "{out}");
    }

    #[test]
    fn sanitize_synthetic_long_prose_is_cut_to_500_chars_and_email_redacted() {
        let text = format!(
            "{} principal synthetic.person@example.com lacks permission",
            "word ".repeat(30)
        );
        let out = sanitize_vertex_text(&text, &[]);
        assert!(out.contains("principal [redacted] lacks"), "{out}");
        assert!(!out.contains('@'), "{out}");
        assert_eq!(
            sanitize_vertex_text(&"long prose ".repeat(100), &[])
                .chars()
                .count(),
            500
        );
    }

    #[test]
    fn sanitize_synthetic_token_is_redacted_before_the_data_rule_sees_it() {
        // The token is long enough to be a base64 run; it must still come out
        // as the plain secret marker, not as a length.
        let text = format!("token {SYNTHETIC_ACCESS_TOKEN} end");
        assert_eq!(
            sanitize_vertex_text(&text, &[SYNTHETIC_ACCESS_TOKEN]),
            "token [redacted] end"
        );
    }
}
