//! Carries Gemini `thoughtSignature`s through Anthropic thinking blocks.
//!
//! Gemini rejects a replayed `functionCall` part that does not carry the
//! `thoughtSignature` the model returned with it. The proxy keeps no state,
//! so the signature travels to the client inside the `signature` of an
//! Anthropic thinking block, as `ccp:google-ai-plan:v1:<model>:<signature>`,
//! and comes back with the history. On replay the thinking block is not sent
//! as a thought part: its signature is set on the next `functionCall` part of
//! the same model turn. The prefix differs from the Vertex AI one, so a
//! signature is never replayed to the other provider, and the model ID keeps
//! it from reaching another model. An envelope with an empty signature marks
//! a thinking block with no signature to replay: Google sent none, or the one
//! it sent was over the 1 MiB signature limit.

use crate::providers::vertex::translate::thought_signature::{
    decode_envelope_with, encode_envelope_with, signature_for_model_with,
};

pub use crate::providers::vertex::translate::thought_signature::{
    MAX_SIGNATURE_BYTES, SKIP_THOUGHT_SIGNATURE_VALIDATOR, ThoughtSignature,
};

/// Prefix of every envelope this module produces.
pub const ENVELOPE_PREFIX: &str = "ccp:google-ai-plan:v1:";

/// Whether `value` is an envelope produced by this module. Traffic captures
/// redact these values whole.
pub fn is_google_ai_plan_thought_signature(value: &str) -> bool {
    value.starts_with(ENVELOPE_PREFIX)
}

/// Wraps `signature` from `model` in an envelope.
///
/// Returns `None` when `model` is empty or contains `:`, or when `signature`
/// is over [`MAX_SIGNATURE_BYTES`]. An empty `signature` yields the "no
/// signature" envelope.
pub fn encode_envelope(model: &str, signature: &str) -> Option<String> {
    encode_envelope_with(ENVELOPE_PREFIX, model, signature)
}

/// Reads an envelope produced by [`encode_envelope`]. Any other value,
/// including a Vertex AI envelope or an Anthropic signature, yields `None`.
pub fn decode_envelope(value: &str) -> Option<ThoughtSignature<'_>> {
    decode_envelope_with(ENVELOPE_PREFIX, value)
}

/// The signature inside `value` when it is an envelope from `model` that
/// carries one; `None` for any other value.
pub fn signature_for_model<'a>(value: &'a str, model: &str) -> Option<&'a str> {
    signature_for_model_with(ENVELOPE_PREFIX, value, model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::vertex::translate::thought_signature as vertex;

    /// Thought signatures are opaque base64 strings; this one is synthetic.
    const SYNTHETIC_SIGNATURE: &str = "c3ludGhldGljLWludGVyYWN0aW9ucy1zaWduYXR1cmU=";

    #[test]
    fn round_trip_keeps_model_and_signature() {
        let envelope = encode_envelope("gemini-4-hydrogen", SYNTHETIC_SIGNATURE)
            .expect("a plain model ID is encoded");
        assert_eq!(
            envelope,
            format!("ccp:google-ai-plan:v1:gemini-4-hydrogen:{SYNTHETIC_SIGNATURE}")
        );
        assert!(is_google_ai_plan_thought_signature(&envelope));
        assert_eq!(
            decode_envelope(&envelope),
            Some(ThoughtSignature {
                model: "gemini-4-hydrogen",
                signature: SYNTHETIC_SIGNATURE,
            })
        );
        assert_eq!(
            signature_for_model(&envelope, "gemini-4-hydrogen"),
            Some(SYNTHETIC_SIGNATURE)
        );
        assert_eq!(signature_for_model(&envelope, "gemini-4-neon"), None);
    }

    #[test]
    fn vertex_and_plan_envelopes_never_cross() {
        let plan = encode_envelope("gemini-4-neon", SYNTHETIC_SIGNATURE).expect("encoded");
        let from_vertex =
            vertex::encode_envelope("gemini-4-neon", SYNTHETIC_SIGNATURE).expect("encoded");
        assert_eq!(signature_for_model(&from_vertex, "gemini-4-neon"), None);
        assert_eq!(decode_envelope(&from_vertex), None);
        assert!(!is_google_ai_plan_thought_signature(&from_vertex));
        assert_eq!(vertex::signature_for_model(&plan, "gemini-4-neon"), None);
        assert_eq!(vertex::decode_envelope(&plan), None);
        assert!(!vertex::is_vertex_thought_signature(&plan));
    }

    #[test]
    fn foreign_and_malformed_values_are_not_envelopes() {
        let foreign = [
            "ccp:codex:v1:cnNfMQ:gAAAAABopaque",
            "ccp:muse:v1:cnNfMQ:gAAAAABopaque",
            "ErUBCkYIBRgCIkA8YWZbVvLuJ5TkyQ6tOl0p3RVb6m0QZ1qA",
            "",
            "ccp:google-ai-plan:v1:",
            "ccp:google-ai-plan:v1:no-separator",
            "ccp:google-ai-plan:v1::signature-without-model",
            "ccp:google-ai-plan:v2:gemini-4-neon:sig",
        ];
        for value in foreign {
            assert_eq!(decode_envelope(value), None, "{value:?}");
            assert_eq!(
                signature_for_model(value, "gemini-4-neon"),
                None,
                "{value:?}"
            );
        }
    }

    #[test]
    fn empty_and_oversized_signatures() {
        let empty = encode_envelope("gemini-4-neon", "").expect("empty is allowed");
        assert_eq!(empty, "ccp:google-ai-plan:v1:gemini-4-neon:");
        assert_eq!(signature_for_model(&empty, "gemini-4-neon"), None);
        let over = "A".repeat(MAX_SIGNATURE_BYTES + 1);
        assert_eq!(encode_envelope("gemini-4-neon", &over), None);
        assert_eq!(encode_envelope("gemini:4", SYNTHETIC_SIGNATURE), None);
    }
}
