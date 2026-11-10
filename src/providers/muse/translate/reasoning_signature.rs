use base64::Engine;

const PREFIX: &str = "ccp:muse:v1:";
const MAX_ID_BYTES: usize = 4 * 1024;
const MAX_ENCRYPTED_CONTENT_BYTES: usize = 8 * 1024 * 1024;

/// Whether `signature` was produced by this proxy's Muse reasoning replay
/// encoder.
///
/// These signatures embed the opaque `encrypted_content` replay blob, so
/// traffic captures redact the whole value.
pub(crate) fn is_muse_reasoning_signature(signature: &str) -> bool {
    signature.starts_with(PREFIX)
}

/// The parts of an upstream reasoning item needed to send it back on the
/// next turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningReplay {
    pub id: String,
    pub encrypted_content: String,
}

/// Encodes a reasoning item as an Anthropic thinking signature. The format is
/// an encoding, not encryption: `ccp:muse:v1:<base64url id>:<encrypted_content>`.
/// Returns `None` when either part is empty or over the size limits, in which
/// case the thinking block goes out without a signature.
pub fn encode_reasoning_signature(replay: &ReasoningReplay) -> Option<String> {
    if replay.id.is_empty()
        || replay.id.len() > MAX_ID_BYTES
        || replay.encrypted_content.is_empty()
        || replay.encrypted_content.len() > MAX_ENCRYPTED_CONTENT_BYTES
    {
        return None;
    }
    let encoded_id = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(replay.id.as_bytes());
    Some(format!("{PREFIX}{encoded_id}:{}", replay.encrypted_content))
}

/// Decodes a signature produced by [`encode_reasoning_signature`]. Any other
/// signature, including one from another provider, yields `None`.
pub fn decode_reasoning_signature(signature: &str) -> Option<ReasoningReplay> {
    let payload = signature.strip_prefix(PREFIX)?;
    if payload.is_empty() || payload.len() > max_payload_len() {
        return None;
    }
    let (encoded_id, encrypted_content) = payload.split_once(':')?;
    if encoded_id.is_empty()
        || encoded_id.len() > encoded_id_len_limit()
        || encrypted_content.is_empty()
        || encrypted_content.len() > MAX_ENCRYPTED_CONTENT_BYTES
    {
        return None;
    }
    let id = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded_id)
        .ok()?;
    if id.is_empty() || id.len() > MAX_ID_BYTES {
        return None;
    }
    Some(ReasoningReplay {
        id: String::from_utf8(id).ok()?,
        encrypted_content: encrypted_content.to_string(),
    })
}

fn encoded_id_len_limit() -> usize {
    MAX_ID_BYTES.div_ceil(3) * 4
}

fn max_payload_len() -> usize {
    encoded_id_len_limit() + 1 + MAX_ENCRYPTED_CONTENT_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    // The id format observed on the wire is `rs_<response suffix>:rs_<32 hex>`;
    // it contains a colon, which the base64 encoding keeps out of the
    // separator position.
    const OBSERVED_ID: &str = "rs_6af274a0aeedd7aeeddd44e4:rs_01a1c31794f476dfb01deceb9b9be238";

    #[test]
    fn round_trip_preserves_an_observed_reasoning_id_with_a_colon() {
        let replay = ReasoningReplay {
            id: OBSERVED_ID.to_string(),
            encrypted_content: "gAAAA-opaque:with=colon".to_string(),
        };
        let signature = encode_reasoning_signature(&replay).expect("encodable");
        assert!(signature.starts_with(PREFIX), "{signature}");
        assert!(is_muse_reasoning_signature(&signature));
        assert_eq!(decode_reasoning_signature(&signature), Some(replay));
    }

    #[test]
    fn foreign_prefixes_are_rejected() {
        let replay = ReasoningReplay {
            id: OBSERVED_ID.to_string(),
            encrypted_content: "blob".to_string(),
        };
        let muse = encode_reasoning_signature(&replay).expect("encodable");
        let codex = muse.replacen("ccp:muse:v1:", "ccp:codex:v1:", 1);
        for foreign in [codex.as_str(), "ErUBCkYIBRgCIkA-real-anthropic", ""] {
            assert_eq!(decode_reasoning_signature(foreign), None, "{foreign}");
            assert!(!is_muse_reasoning_signature(foreign), "{foreign}");
        }
    }

    #[test]
    fn malformed_payloads_are_rejected() {
        let cases = [
            "ccp:muse:v1:",
            "ccp:muse:v1:no-separator",
            "ccp:muse:v1::blob",
            "ccp:muse:v1:cnNfMQ:",
            "ccp:muse:v1:***:blob",
            // Valid base64 that decodes to bytes that are not UTF-8.
            "ccp:muse:v1:__8:blob",
        ];
        for signature in cases {
            assert_eq!(decode_reasoning_signature(signature), None, "{signature}");
        }
    }

    #[test]
    fn oversized_parts_are_not_encoded_or_decoded() {
        let big_blob = "A".repeat(MAX_ENCRYPTED_CONTENT_BYTES + 1);
        assert_eq!(
            encode_reasoning_signature(&ReasoningReplay {
                id: OBSERVED_ID.to_string(),
                encrypted_content: big_blob.clone(),
            }),
            None
        );
        assert_eq!(
            encode_reasoning_signature(&ReasoningReplay {
                id: "r".repeat(MAX_ID_BYTES + 1),
                encrypted_content: "blob".to_string(),
            }),
            None
        );
        assert_eq!(
            decode_reasoning_signature(&format!("{PREFIX}cnNfMQ:{big_blob}")),
            None
        );
        let exact = ReasoningReplay {
            id: "r".repeat(MAX_ID_BYTES),
            encrypted_content: "A".repeat(MAX_ENCRYPTED_CONTENT_BYTES),
        };
        let signature = encode_reasoning_signature(&exact).expect("limits are inclusive");
        assert_eq!(decode_reasoning_signature(&signature), Some(exact));
    }

    #[test]
    fn empty_parts_are_not_encoded() {
        for (id, blob) in [("", "blob"), (OBSERVED_ID, "")] {
            let replay = ReasoningReplay {
                id: id.to_string(),
                encrypted_content: blob.to_string(),
            };
            assert_eq!(encode_reasoning_signature(&replay), None);
        }
    }
}
