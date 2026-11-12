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

/// Most input bytes scanned. The result is cut to 500 characters anyway, so
/// a longer field only costs time; everything past this point is dropped.
pub const MAX_SCANNED_BYTES: usize = 64 * 1024;

const REDACTED: &str = "[redacted]";

/// Runs of the standard and URL-safe base64 alphabets, with padding. A
/// percent escape (`%2F`) counts as one character of the run, so data that
/// was percent-encoded once is a run too.
static BASE64_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:[A-Za-z0-9+/_-]|%[0-9A-Fa-f]{2}){64,}(?:=|%3[Dd]){0,2}")
        .expect("base64 run pattern is a valid regex")
});

/// A Google OAuth access token: `ya29.` followed by the token characters
/// of RFC 6750, any of them percent-encoded once (`ya29%2Ec%2E…`). A period
/// may separate parts of the token (`ya29.c.…`), but one that ends it is
/// left alone, so a sentence-ending period survives.
static ACCESS_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"ya29(?:\.|%2[Ee])(?:[A-Za-z0-9_~+/-]|%[0-9A-Fa-f]{2})+(?:\.(?:[A-Za-z0-9_~+/-]|%[0-9A-Fa-f]{2})+)*=*",
    )
    .expect("access token pattern is a valid regex")
});

/// A whole base64 run that is a Google resource name. The run can begin in
/// the middle of a URL (after the host's last `.`), so short lower-case
/// segments such as `com/v1/` may come first; then
/// `projects/<id>/locations/<id>`, any number of `/<collection>/<id>` pairs
/// and at most one trailing collection, every segment at most 63
/// characters. Data written to fit this grammar is kept like a real name.
static RESOURCE_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^/?(?:[a-z0-9-]{1,63}/)*projects/[A-Za-z0-9_-]{1,63}/locations/[A-Za-z0-9_-]{1,63}(?:/[a-z][A-Za-z0-9]{0,62}/[A-Za-z0-9_-]{1,63})*(?:/[a-z][A-Za-z0-9]{0,62})?/?$",
    )
    .expect("resource name pattern is a valid regex")
});

/// Scrubs one piece of upstream prose, in this order:
///
/// 1. only the first [`MAX_SCANNED_BYTES`] are kept, cut on a character
///    boundary; when the cut falls inside a run of token or base64
///    characters (letters, digits, `+/_-=%.~`), that partial run is dropped
///    too, so a shortened secret or data run cannot survive as a prefix;
/// 2. every occurrence of each secret (pass the access token) becomes
///    `[redacted]`, and so does anything shaped like a Google access token
///    (`ya29.` and token characters), listed or not;
/// 3. every run of 64 or more base64 or base64url characters becomes
///    `[redacted len=N]`, N being its length in the text, unless the whole
///    run is a Google resource name such as
///    `projects/my-project/locations/global/publishers/google/models/gemini-x`,
///    which is the useful part of a not-found message; a run that contains a
///    percent escape is never treated as a resource name;
/// 4. e-mail addresses are redacted and the result is cut to 500
///    characters by `muse::sanitize::sanitize_upstream_text`.
///
/// Every replacement happens before the cut, so a value that straddles the
/// 500th character cannot survive as a prefix. Encoded data is redacted
/// whatever mix of character classes it has: an upper-case-only or
/// hexadecimal value is as opaque as any other.
///
/// Percent-encoding is undone one level only. A secret is found verbatim
/// and with any of its bytes percent-encoded once, in either hex case; a
/// `%` inside a secret is matched only as itself, so a secret that
/// contains `%` is found verbatim but not where that `%` was encoded. In
/// the token and run rules a percent escape counts as one character.
/// Out of scope: double encoding (`%252F`), and separators such as spaces,
/// line breaks or `...` inserted inside a secret, a token or a run, which
/// break the match.
pub fn sanitize_vertex_text(text: &str, secrets: &[&str]) -> String {
    let bounded = bounded_prefix(text);
    let without_secrets = redact_secrets_in_any_encoding(bounded, secrets);
    let without_tokens = ACCESS_TOKEN.replace_all(&without_secrets, REDACTED);
    let without_data = BASE64_RUN.replace_all(&without_tokens, |captures: &Captures| {
        let run = &captures[0];
        if !run.contains('%') && RESOURCE_NAME.is_match(run) {
            run.to_string()
        } else {
            format!("[redacted len={}]", run.len())
        }
    });
    sanitize_upstream_text(&without_data, secrets)
}

/// A character that the secret, token and run rules can match as part of a
/// longer value.
fn is_run_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "+/_-=%.~".contains(c)
}

/// The first [`MAX_SCANNED_BYTES`] of `text`, cut on a character boundary.
/// When the cut splits a run of [`is_run_char`] characters, the part of the
/// run before the cut is dropped as well.
fn bounded_prefix(text: &str) -> &str {
    if text.len() <= MAX_SCANNED_BYTES {
        return text;
    }
    let mut end = MAX_SCANNED_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let (head, rest) = text.split_at(end);
    if rest.starts_with(is_run_char) {
        head.trim_end_matches(is_run_char)
    } else {
        head
    }
}

/// Replaces every occurrence of each non-empty secret with `[redacted]`.
/// Each byte of the secret may appear as itself or percent-encoded with
/// either case of hex digit (`/`, `%2F` or `%2f`), so a secret quoted from a
/// URL or a form body is found too. Where several secrets match at one
/// position the longest match wins, so a secret that begins with another
/// one is redacted whole. Matches start and end on character boundaries.
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
        let end = if text.is_char_boundary(at) {
            secrets
                .iter()
                .filter_map(|secret| match_end(bytes, at, secret))
                .filter(|end| text.is_char_boundary(*end))
                .max()
        } else {
            None
        };
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

    #[test]
    fn sanitize_synthetic_percent_encoded_unlisted_token_is_redacted() {
        // Not in the secret list, and the periods arrive as `%2E`.
        assert_eq!(
            sanitize_vertex_text("access_token=ya29%2Ec%2Exxxx", &[]),
            "access_token=[redacted]"
        );
        assert_eq!(
            sanitize_vertex_text("Bearer ya29%2ec.xxxx. Retry later.", &[]),
            "Bearer [redacted]. Retry later."
        );
    }

    #[test]
    fn sanitize_synthetic_run_broken_only_by_percent_escapes_is_redacted() {
        // "QUJD/" repeated, with every `/` percent-encoded: 160 characters
        // of base64 once decoded, 224 bytes as written.
        let text = "QUJD%2F".repeat(32);
        assert_eq!(sanitize_vertex_text(&text, &[]), "[redacted len=224]");
        // Encoded padding belongs to the run.
        let padded = format!("{}%3D%3D end", "Ab3dEf9GhIjK0lMn".repeat(4));
        assert_eq!(sanitize_vertex_text(&padded, &[]), "[redacted len=70] end");
    }

    #[test]
    fn sanitize_synthetic_run_with_an_escape_is_not_a_resource_name() {
        let text =
            "projects/p%2Flocations/global/publishers/google/models/gemini-3-flash-preview-x";
        assert_eq!(sanitize_vertex_text(text, &[]), "[redacted len=79]");
    }

    #[test]
    fn sanitize_synthetic_longest_listed_secret_wins() {
        // One secret is a prefix of the other; the order of the list must
        // not leave a suffix of the longer one behind.
        assert_eq!(
            sanitize_vertex_text("abcdef", &["abc", "abcdef"]),
            "[redacted]"
        );
        assert_eq!(
            sanitize_vertex_text("abcdef", &["abcdef", "abc"]),
            "[redacted]"
        );
        assert_eq!(
            sanitize_vertex_text("abc abcdef abcd", &["abc", "abcdef"]),
            "[redacted] [redacted] [redacted]d"
        );
    }

    #[test]
    fn sanitize_synthetic_secret_with_a_percent_sign_is_matched_verbatim_only() {
        // The `%` of the secret is matched only as itself, so its encoded
        // form `%25` is not found.
        assert_eq!(
            sanitize_vertex_text("%2541 and %41", &["%41"]),
            "%2541 and [redacted]"
        );
    }

    #[test]
    fn sanitize_synthetic_multi_megabyte_field_is_bounded() {
        // 8 MiB of base64 in one run: the run passes the 64 KiB bound, so
        // what is left of it is dropped.
        let run = format!("Invalid data: \"{}\"", "QUJD".repeat(2 * 1024 * 1024));
        assert_eq!(sanitize_vertex_text(&run, &[]), "Invalid data: \"");
        // 20 MiB of words: the usual 500-character cut.
        let prose = "word ".repeat(4 * 1024 * 1024);
        assert_eq!(sanitize_vertex_text(&prose, &[]), "word ".repeat(100));
        // 16 MiB of runs one character short of the data rule.
        let near_miss = format!("{} ", "A".repeat(63)).repeat(256 * 1024);
        assert_eq!(
            sanitize_vertex_text(&near_miss, &[]),
            format!("{} ", "A".repeat(63)).repeat(7) + &"A".repeat(52)
        );
    }

    #[test]
    fn sanitize_synthetic_run_straddling_the_scan_bound_leaves_no_fragment() {
        // The first run shrinks to its marker, which brings the end of the
        // scanned text inside the 500-character output.
        let blob = "Ab3dEf9GhIjK0lMn".repeat(64);
        let text = format!("{} {blob}", "A".repeat(65000));
        assert_eq!(sanitize_vertex_text(&text, &[]), "[redacted len=65000] ");
        // A short listed secret across the bound is dropped, not shortened.
        let secret = "GOCSPX-synthetic-short";
        let text = format!("{} {secret} tail", "A".repeat(65525));
        assert_eq!(
            sanitize_vertex_text(&text, &[secret]),
            "[redacted len=65525] "
        );
        // A run that ends exactly at the bound is complete and redacted.
        let text = format!(
            "{} {} tail",
            "A".repeat(65455),
            "Ab3dEf9GhIjK0lMn".repeat(5)
        );
        assert_eq!(
            sanitize_vertex_text(&text, &[]),
            "[redacted len=65455] [redacted len=80]"
        );
        // A two-byte character across the bound is cut before it.
        let text = format!("{} {}\u{e9}xxxxxxxxxx", "A".repeat(65000), " ".repeat(534));
        assert_eq!(
            sanitize_vertex_text(&text, &[]),
            format!("[redacted len=65000] {}", " ".repeat(479))
        );
    }
}
