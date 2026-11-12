//! Carries Gemini thought signatures through Anthropic thinking blocks.
//!
//! Gemini 3 and later reject a replayed `functionCall` part that lacks the
//! `thoughtSignature` the model attached to it. The proxy keeps no state, so
//! the signature travels to the client inside the `signature` of a thinking
//! block, as `ccp:vertex:v1:<model>:<signature>`, and comes back with the
//! history on the next request. The model ID is part of the envelope so a
//! signature is never sent to a different model after the client switches.
//! An envelope with an empty signature marks a thinking block that had none.

/// Prefix of every envelope this module produces.
pub const ENVELOPE_PREFIX: &str = "ccp:vertex:v1:";

/// The value Gemini accepts in place of a thought signature for a
/// `functionCall` part the API did not generate. Google documents it as a
/// last resort that can lower response quality.
pub const SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";

/// Largest signature carried in an envelope. A larger one is dropped and the
/// next request falls back to [`SKIP_THOUGHT_SIGNATURE_VALIDATOR`].
pub const MAX_SIGNATURE_BYTES: usize = 1024 * 1024;

/// The parts of a decoded envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThoughtSignature<'a> {
    /// The upstream model ID that produced the signature.
    pub model: &'a str,
    /// The opaque signature; empty when the thinking block had none.
    pub signature: &'a str,
}

/// Whether `value` is an envelope produced by this module. Traffic captures
/// redact these values whole.
pub fn is_vertex_thought_signature(value: &str) -> bool {
    value.starts_with(ENVELOPE_PREFIX)
}

/// Wraps `signature` from `model` in an envelope.
///
/// Returns `None` when `model` is empty or contains `:` (it would break the
/// separator), or when `signature` is over [`MAX_SIGNATURE_BYTES`]. An empty
/// `signature` is allowed and yields the "no signature" envelope.
pub fn encode_envelope(model: &str, signature: &str) -> Option<String> {
    if model.is_empty() || model.contains(':') || signature.len() > MAX_SIGNATURE_BYTES {
        return None;
    }
    Some(format!("{ENVELOPE_PREFIX}{model}:{signature}"))
}

/// Reads an envelope produced by [`encode_envelope`]. Any other value,
/// including a signature from another provider or from Anthropic, yields
/// `None`.
pub fn decode_envelope(value: &str) -> Option<ThoughtSignature<'_>> {
    let payload = value.strip_prefix(ENVELOPE_PREFIX)?;
    let (model, signature) = payload.split_once(':')?;
    if model.is_empty() || signature.len() > MAX_SIGNATURE_BYTES {
        return None;
    }
    Some(ThoughtSignature { model, signature })
}

/// The signature inside `value` when it is an envelope from `model` that
/// carries one; `None` for any other value, another model's envelope, or the
/// "no signature" envelope.
pub fn signature_for_model<'a>(value: &'a str, model: &str) -> Option<&'a str> {
    decode_envelope(value)
        .filter(|decoded| decoded.model == model && !decoded.signature.is_empty())
        .map(|decoded| decoded.signature)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thought signatures are opaque base64 strings; this one is synthetic.
    const SYNTHETIC_SIGNATURE: &str = "CiQBjz1rX3VzZXJfc2lnbmF0dXJlX2J5dGVzX2Zvcl90ZXN0cw==";

    #[test]
    fn round_trip_keeps_model_and_signature() {
        let envelope = encode_envelope("gemini-3.5-flash", SYNTHETIC_SIGNATURE).unwrap();
        assert_eq!(
            envelope,
            format!("ccp:vertex:v1:gemini-3.5-flash:{SYNTHETIC_SIGNATURE}")
        );
        assert!(is_vertex_thought_signature(&envelope));
        assert_eq!(
            decode_envelope(&envelope),
            Some(ThoughtSignature {
                model: "gemini-3.5-flash",
                signature: SYNTHETIC_SIGNATURE,
            })
        );
        assert_eq!(
            signature_for_model(&envelope, "gemini-3.5-flash"),
            Some(SYNTHETIC_SIGNATURE)
        );
    }

    #[test]
    fn signature_from_another_model_is_not_used() {
        let envelope = encode_envelope("gemini-4-argon", SYNTHETIC_SIGNATURE).unwrap();
        assert_eq!(signature_for_model(&envelope, "gemini-3.8-flash"), None);
        assert_eq!(signature_for_model(&envelope, "gemini-4-argon-x"), None);
        assert_eq!(signature_for_model(&envelope, "gemini-4"), None);
    }

    #[test]
    fn empty_signature_envelope_marks_a_block_without_a_signature() {
        let envelope = encode_envelope("gemini-3.5-flash", "").unwrap();
        assert_eq!(envelope, "ccp:vertex:v1:gemini-3.5-flash:");
        assert!(is_vertex_thought_signature(&envelope));
        assert_eq!(
            decode_envelope(&envelope),
            Some(ThoughtSignature {
                model: "gemini-3.5-flash",
                signature: "",
            })
        );
        assert_eq!(signature_for_model(&envelope, "gemini-3.5-flash"), None);
    }

    #[test]
    fn oversized_signatures_are_not_carried() {
        let exact = "A".repeat(MAX_SIGNATURE_BYTES);
        let envelope = encode_envelope("gemini-3.5-flash", &exact).expect("the limit is inclusive");
        assert_eq!(
            signature_for_model(&envelope, "gemini-3.5-flash"),
            Some(exact.as_str())
        );

        let over = "A".repeat(MAX_SIGNATURE_BYTES + 1);
        assert_eq!(encode_envelope("gemini-3.5-flash", &over), None);
        assert_eq!(
            decode_envelope(&format!("{ENVELOPE_PREFIX}gemini-3.5-flash:{over}")),
            None
        );
    }

    #[test]
    fn models_that_would_break_the_separator_are_not_encoded() {
        assert_eq!(encode_envelope("", SYNTHETIC_SIGNATURE), None);
        assert_eq!(encode_envelope("gemini:3", SYNTHETIC_SIGNATURE), None);
    }

    #[test]
    fn other_providers_and_anthropic_signatures_are_not_envelopes() {
        // Codex and Muse keep their prefix in the clear; Kimi and the
        // OpenCode Chat Completions path base64url-encode theirs; a real
        // Anthropic signature is an opaque base64 string.
        let foreign = [
            "ccp:codex:v1:cnNfMQ:gAAAAABopaque",
            "ccp:muse:v1:cnNfMQ:gAAAAABopaque",
            "Y2NwOm9wZW5jb2RlOnYxOm1zZ18xOjA",
            "ErUBCkYIBRgCIkA8YWZbVvLuJ5TkyQ6tOl0p3RVb6m0QZ1qA",
            "",
            "ccp:vertex:v1:",
            "ccp:vertex:v1:no-separator",
            "ccp:vertex:v1::signature-without-model",
            "ccp:vertex:v2:gemini-3.5-flash:sig",
        ];
        for value in foreign {
            assert_eq!(decode_envelope(value), None, "{value:?}");
            assert_eq!(signature_for_model(value, "gemini-3.5-flash"), None);
        }
        assert!(!is_vertex_thought_signature("ccp:muse:v1:x:y"));
        assert!(!is_vertex_thought_signature("ErUBCkYIBRgCIkA"));
    }

    #[test]
    fn signature_may_contain_colons() {
        let envelope = encode_envelope("gemini-3.5-flash", "a:b:c").unwrap();
        assert_eq!(
            signature_for_model(&envelope, "gemini-3.5-flash"),
            Some("a:b:c")
        );
    }
}
