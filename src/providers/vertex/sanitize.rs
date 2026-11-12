//! Scrubbing for prose that Vertex AI sends back (HTTP error messages,
//! in-stream errors, `finishMessage`) before it reaches clients, the
//! monitor, traffic captures or logs. Google error messages can quote the
//! request, so besides the access token they may carry a thought signature
//! or inline image data that the proxy sent.

use std::sync::LazyLock;

use regex_lite::{Captures, Regex};

use crate::providers::muse::sanitize::sanitize_upstream_text;

/// Shortest run of base64 characters that is treated as encoded data.
pub const MIN_ENCODED_RUN: usize = 64;

const REDACTED: &str = "[redacted]";

/// Runs of the standard and URL-safe base64 alphabets, with padding.
static BASE64_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z0-9+/_-]{64,}={0,2}").expect("base64 run pattern is a valid regex")
});

/// A Google OAuth access token: `ya29.` followed by the token characters
/// of RFC 6750. A period may separate parts of the token (`ya29.c.…`), but
/// one that ends it is left alone, so a sentence-ending period survives.
static ACCESS_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"ya29\.[A-Za-z0-9_~+/-]+(?:\.[A-Za-z0-9_~+/-]+)*=*")
        .expect("access token pattern is a valid regex")
});

/// A whole base64 run that is a Google resource name. The run can begin in
/// the middle of a URL (after the host's last `.`), so short lower-case
/// segments such as `com/v1/` may come first; then
/// `projects/<id>/locations/<id>`, any number of `/<collection>/<id>` pairs
/// and at most one trailing collection. Every segment is shorter than an
/// encoded run, so data cannot hide in an ID position.
static RESOURCE_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^/?(?:[a-z0-9-]{1,63}/)*projects/[A-Za-z0-9_-]{1,63}/locations/[A-Za-z0-9_-]{1,63}(?:/[a-z][A-Za-z0-9]{0,62}/[A-Za-z0-9_-]{1,63})*(?:/[a-z][A-Za-z0-9]{0,62})?/?$",
    )
    .expect("resource name pattern is a valid regex")
});

/// Scrubs one piece of upstream prose, in this order:
///
/// 1. every occurrence of each secret (pass the access token) becomes
///    `[redacted]`, whether it appears verbatim or with any of its bytes
///    percent-encoded, and so does anything shaped like a Google access
///    token (`ya29.` and token characters), listed or not;
/// 2. every run of 64 or more base64 or base64url characters becomes
///    `[redacted len=N]`, unless the whole run is a Google resource name
///    such as `projects/my-project/locations/global/publishers/google/models/gemini-x`,
///    which is the useful part of a not-found message;
/// 3. e-mail addresses are redacted and the result is cut to 500
///    characters by `muse::sanitize::sanitize_upstream_text`.
///
/// Every replacement happens before the cut, so a value that straddles the
/// 500th character cannot survive as a prefix. Encoded data is redacted
/// whatever mix of character classes it has: an upper-case-only or
/// hexadecimal value is as opaque as any other.
pub fn sanitize_vertex_text(text: &str, secrets: &[&str]) -> String {
    let without_secrets = redact_secrets_in_any_encoding(text, secrets);
    let without_tokens = ACCESS_TOKEN.replace_all(&without_secrets, REDACTED);
    let without_data = BASE64_RUN.replace_all(&without_tokens, |captures: &Captures| {
        let run = &captures[0];
        if RESOURCE_NAME.is_match(run) {
            run.to_string()
        } else {
            format!("[redacted len={}]", run.len())
        }
    });
    sanitize_upstream_text(&without_data, secrets)
}

/// Replaces every occurrence of each non-empty secret with `[redacted]`.
/// Each byte of the secret may appear as itself or percent-encoded with
/// either case of hex digit (`/`, `%2F` or `%2f`), so a secret quoted from a
/// URL or a form body is found too. Matches start and end on character
/// boundaries.
fn redact_secrets_in_any_encoding(text: &str, secrets: &[&str]) -> String {
    let secrets: Vec<&[u8]> = secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .map(|secret| secret.as_bytes())
        .collect();
    if secrets.is_empty() {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut at = 0;
    while at < bytes.len() {
        let end = text
            .is_char_boundary(at)
            .then(|| {
                secrets
                    .iter()
                    .find_map(|secret| match_end(bytes, at, secret))
            })
            .flatten()
            .filter(|end| text.is_char_boundary(*end));
        match end {
            Some(end) => {
                out.push_str(&text[copied..at]);
                out.push_str(REDACTED);
                copied = end;
                at = end;
            }
            None => at += 1,
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// Where a match of `secret` that starts at `start` ends, or `None`.
fn match_end(text: &[u8], start: usize, secret: &[u8]) -> Option<usize> {
    let mut at = start;
    for &expected in secret {
        match text.get(at) {
            Some(&byte) if byte == expected => at += 1,
            Some(b'%') if text.get(at + 1..at + 3).and_then(percent_decoded) == Some(expected) => {
                at += 3;
            }
            _ => return None,
        }
    }
    Some(at)
}

/// The byte two hex digits stand for.
fn percent_decoded(digits: &[u8]) -> Option<u8> {
    let [high, low] = digits else {
        return None;
    };
    let value = |digit: u8| char::from(digit).to_digit(16);
    u8::try_from(value(*high)? * 16 + value(*low)?).ok()
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
    fn sanitize_synthetic_encoded_runs_are_redacted_whatever_their_character_classes() {
        // Upper case only: "ABC" repeated, base64-encoded.
        let upper = "QUJD".repeat(32);
        // Letters of both cases, no digit.
        let digit_free = "AbCdEfGhIjKlMnOp".repeat(4);
        // Lower case and digits only: a hexadecimal digest.
        let hex = "0123456789abcdef".repeat(4);
        // Lower case only, URL-safe alphabet.
        let lower = "abcdefgh_ijklmno-".repeat(4);
        for run in [&upper, &digit_free, &hex, &lower] {
            let text = format!("Invalid value \"{run}\" here");
            assert_eq!(
                sanitize_vertex_text(&text, &[]),
                format!("Invalid value \"[redacted len={}]\" here", run.len()),
                "{run}"
            );
        }
    }

    #[test]
    fn sanitize_synthetic_resource_name_inside_a_url_is_kept() {
        let text = "Not found: https://aiplatform.googleapis.com/v1/projects/synthetic-project-123/locations/us-central1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent";
        assert_eq!(sanitize_vertex_text(text, &[]), text);
        let text = "Collection projects/synthetic-project-123/locations/europe-west4/publishers/google/models is empty";
        assert_eq!(sanitize_vertex_text(text, &[]), text);
    }

    #[test]
    fn sanitize_synthetic_data_in_a_resource_name_position_is_redacted() {
        // A resource-name prefix does not exempt data glued to it: the run
        // is only kept when every segment fits the resource-name grammar.
        let blob = "Ab3dEf9GhIjK0lMn".repeat(5);
        let run = format!("projects/synthetic-project/locations/global/cachedContents/{blob}");
        let text = format!("Unknown {run} here");
        assert_eq!(
            sanitize_vertex_text(&text, &[]),
            format!("Unknown [redacted len={}] here", run.len())
        );
        let run = format!("{blob}/projects/synthetic-project/locations/global");
        assert_eq!(
            sanitize_vertex_text(&run, &[]),
            format!("[redacted len={}]", run.len())
        );
    }

    #[test]
    fn sanitize_synthetic_unlisted_access_token_shapes_are_redacted() {
        // Neither token is in the secret list. The first is shorter than an
        // encoded run, the second has only lower case and digits.
        let short = "ya29.c.b0Synthetic-Short_Token";
        let lower = format!("ya29.a0{}", "synthetic0lower0case0".repeat(4));
        for token in [short, lower.as_str()] {
            let text = format!("Bearer {token}. Retry later.");
            assert_eq!(
                sanitize_vertex_text(&text, &[]),
                "Bearer [redacted]. Retry later.",
                "{token}"
            );
        }
        // The prefix alone is not a token.
        let prose = "tokens start with ya29. and nothing else";
        assert_eq!(sanitize_vertex_text(prose, &[]), prose);
    }

    #[test]
    fn sanitize_synthetic_percent_encoded_secrets_are_redacted() {
        let secret = "1//synthetic+secret/value=";
        for encoded in [
            // As a form body or query string writes it.
            "1%2F%2Fsynthetic%2Bsecret%2Fvalue%3D",
            // Lower-case hex digits.
            "1%2f%2fsynthetic%2bsecret%2fvalue%3d",
            // Unreserved bytes encoded as well.
            "%31%2F%2F%73ynthetic%2Bsecret%2Fvalue%3D",
        ] {
            let text = format!("refresh_token={encoded}&grant_type=refresh_token");
            assert_eq!(
                sanitize_vertex_text(&text, &[secret]),
                "refresh_token=[redacted]&grant_type=refresh_token",
                "{encoded}"
            );
        }
        let encoded_token = SYNTHETIC_ACCESS_TOKEN
            .replace('.', "%2E")
            .replace('_', "%5F");
        let text = format!("access_token={encoded_token}");
        assert_eq!(
            sanitize_vertex_text(&text, &[SYNTHETIC_ACCESS_TOKEN]),
            "access_token=[redacted]"
        );
        // A percent sign that does not encode the next secret byte breaks
        // the match.
        let text = "1%2G/synthetic";
        assert_eq!(sanitize_vertex_text(text, &[secret]), text);
    }

    #[test]
    fn sanitize_synthetic_secret_next_to_multibyte_text_is_redacted() {
        let secret = "synthetic-secret-ü-value";
        let text = format!("é{secret}ü and é{}ü", secret.replace('ü', "%C3%BC"));
        assert_eq!(
            sanitize_vertex_text(&text, &[secret]),
            "é[redacted]ü and é[redacted]ü"
        );
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
