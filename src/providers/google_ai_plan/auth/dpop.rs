//! DPoP (RFC 9449) proofs for Google's token endpoint.
//!
//! The key pair is a P-256 key created on this machine. It is stored as an
//! unencrypted PKCS#8 document in the credential, next to the refresh token
//! it is bound to.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngCore;
use ring::rand::SystemRandom;
use ring::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Why a DPoP key could not be created, read or used.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DpopError {
    /// The system random number generator failed.
    #[error("could not create a DPoP key")]
    Generate,
    /// The stored key is not a P-256 PKCS#8 document.
    #[error("the stored DPoP key cannot be read")]
    UnreadableKey,
    /// Signing failed.
    #[error("could not sign a DPoP proof")]
    Sign,
}

/// The public half of the key as a JWK. Only `kty`, `crv`, `x` and `y`:
/// Google rejects a JWK with other members such as `ext` or `key_ops`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicJwk {
    /// Always `EC`.
    pub kty: String,
    /// Always `P-256`.
    pub crv: String,
    /// The point's x coordinate, base64url without padding.
    pub x: String,
    /// The point's y coordinate, base64url without padding.
    pub y: String,
}

/// The claims of one proof besides `htm` and `htu`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofClaims {
    /// The proof's unique id.
    pub jti: String,
    /// The server's latest `DPoP-Nonce`, when one is known.
    pub nonce: Option<String>,
    /// The issue time in Unix seconds; `None` takes the local clock.
    pub iat: Option<i64>,
}

impl ProofClaims {
    /// Claims with a fresh random `jti`, no nonce, and the local clock.
    pub fn random() -> Self {
        Self {
            jti: random_jti(),
            nonce: None,
            iat: None,
        }
    }
}

/// The DPoP key pair.
pub struct DpopKey {
    pair: EcdsaKeyPair,
    pkcs8: Vec<u8>,
}

impl std::fmt::Debug for DpopKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DpopKey").finish_non_exhaustive()
    }
}

#[derive(Serialize)]
struct Header<'a> {
    typ: &'static str,
    alg: &'static str,
    jwk: &'a PublicJwk,
}

#[derive(Serialize)]
struct Payload<'a> {
    jti: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    nonce: Option<&'a str>,
    htm: &'static str,
    htu: &'a str,
    iat: i64,
}

impl DpopKey {
    /// Creates a new key pair.
    pub fn generate() -> Result<Self, DpopError> {
        let document =
            EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &SystemRandom::new())
                .map_err(|_| DpopError::Generate)?;
        Self::from_pkcs8(document.as_ref()).map_err(|_| DpopError::Generate)
    }

    /// Reads a key written by [`DpopKey::to_stored`].
    pub fn from_stored(stored: &str) -> Result<Self, DpopError> {
        let pkcs8 = URL_SAFE_NO_PAD
            .decode(stored.trim())
            .map_err(|_| DpopError::UnreadableKey)?;
        Self::from_pkcs8(&pkcs8)
    }

    fn from_pkcs8(pkcs8: &[u8]) -> Result<Self, DpopError> {
        let pair = EcdsaKeyPair::from_pkcs8(
            &ECDSA_P256_SHA256_FIXED_SIGNING,
            pkcs8,
            &SystemRandom::new(),
        )
        .map_err(|_| DpopError::UnreadableKey)?;
        // An uncompressed SEC 1 point: 0x04 || x || y, 32 bytes each.
        let point = pair.public_key().as_ref();
        if point.len() != 65 || point[0] != 0x04 {
            return Err(DpopError::UnreadableKey);
        }
        Ok(Self {
            pair,
            pkcs8: pkcs8.to_vec(),
        })
    }

    /// The key as stored in the credential: the PKCS#8 document,
    /// base64url without padding.
    pub fn to_stored(&self) -> String {
        URL_SAFE_NO_PAD.encode(&self.pkcs8)
    }

    /// The public key as the JWK the proof header carries.
    pub fn public_jwk(&self) -> PublicJwk {
        let point = self.pair.public_key().as_ref();
        PublicJwk {
            kty: "EC".to_string(),
            crv: "P-256".to_string(),
            x: URL_SAFE_NO_PAD.encode(&point[1..33]),
            y: URL_SAFE_NO_PAD.encode(&point[33..65]),
        }
    }

    /// A compact proof JWT for a `POST` to `htu`, the URL actually called.
    pub fn proof(&self, htu: &str, claims: &ProofClaims) -> Result<String, DpopError> {
        let jwk = self.public_jwk();
        let header = Header {
            typ: "dpop+jwt",
            alg: "ES256",
            jwk: &jwk,
        };
        let payload = Payload {
            jti: &claims.jti,
            nonce: claims.nonce.as_deref(),
            htm: "POST",
            htu,
            iat: claims.iat.unwrap_or_else(now_secs),
        };
        let input = format!("{}.{}", encode_json(&header)?, encode_json(&payload)?);
        // The FIXED algorithm signs as raw R || S (64 bytes), the JWS form
        // of ES256; Google rejects DER signatures.
        let signature = self
            .pair
            .sign(&SystemRandom::new(), input.as_bytes())
            .map_err(|_| DpopError::Sign)?;
        Ok(format!(
            "{input}.{}",
            URL_SAFE_NO_PAD.encode(signature.as_ref())
        ))
    }
}

fn encode_json<T: Serialize>(value: &T) -> Result<String, DpopError> {
    let json = serde_json::to_vec(value).map_err(|_| DpopError::Sign)?;
    Ok(URL_SAFE_NO_PAD.encode(json))
}

/// The `jti` of the code-exchange proof: base64url(SHA-256(code)). Google
/// rejects a random `jti` there, which also burns the single-use code.
pub fn code_exchange_jti(code: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(code.as_bytes()))
}

/// A random `jti` for a refresh proof.
pub fn random_jti() -> String {
    let mut bytes = [0_u8; 24];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn now_secs() -> i64 {
    (super::now_ms() / 1000) as i64
}

#[cfg(test)]
mod tests {
    use super::super::test_support::proof_part;
    use super::*;
    use ring::signature::{ECDSA_P256_SHA256_FIXED, UnparsedPublicKey};

    const HTU: &str = "https://oauth2.googleapis.com/token";

    /// Checks `proof`'s signature against the point rebuilt from the JWK in
    /// its own header, as Google does.
    fn verify_with_embedded_jwk(proof: &str) -> Result<(), String> {
        let (input, signature) = proof.rsplit_once('.').ok_or("not a compact JWT")?;
        let signature = URL_SAFE_NO_PAD
            .decode(signature)
            .map_err(|e| format!("signature is not base64url: {e}"))?;
        if signature.len() != 64 {
            return Err(format!(
                "signature is {} bytes, not a raw 64-byte R||S",
                signature.len()
            ));
        }
        let header = proof_part(proof, 0);
        let coordinate = |name: &str| {
            URL_SAFE_NO_PAD
                .decode(header["jwk"][name].as_str().unwrap_or_default())
                .map_err(|e| format!("jwk.{name}: {e}"))
        };
        let mut point = vec![0x04];
        point.extend(coordinate("x")?);
        point.extend(coordinate("y")?);
        UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, point)
            .verify(input.as_bytes(), &signature)
            .map_err(|_| "signature does not verify against the embedded JWK".to_string())
    }

    #[test]
    fn proof_has_the_dpop_header_and_claims_and_verifies_against_its_jwk() {
        let key = DpopKey::generate().unwrap();
        let claims = ProofClaims {
            jti: "synthetic-jti".to_string(),
            nonce: Some("synthetic-nonce".to_string()),
            iat: Some(1_795_608_000),
        };
        let proof = key.proof(HTU, &claims).unwrap();
        println!("proof header: {}", proof_part(&proof, 0));
        println!("proof claims: {}", proof_part(&proof, 1));

        let jwk = key.public_jwk();
        assert_eq!(
            proof_part(&proof, 0),
            serde_json::json!({
                "typ": "dpop+jwt",
                "alg": "ES256",
                "jwk": { "kty": "EC", "crv": "P-256", "x": jwk.x, "y": jwk.y },
            })
        );
        assert_eq!(
            proof_part(&proof, 1),
            serde_json::json!({
                "jti": "synthetic-jti",
                "nonce": "synthetic-nonce",
                "htm": "POST",
                "htu": HTU,
                "iat": 1_795_608_000,
            })
        );
        // The header members come in the order RFC 9449's examples use.
        let header_json = String::from_utf8(
            URL_SAFE_NO_PAD
                .decode(proof.split('.').next().unwrap())
                .unwrap(),
        )
        .unwrap();
        assert!(
            header_json.starts_with(
                r#"{"typ":"dpop+jwt","alg":"ES256","jwk":{"kty":"EC","crv":"P-256","x":"#
            ),
            "{header_json}"
        );
        verify_with_embedded_jwk(&proof).unwrap();
    }

    #[test]
    fn proof_without_nonce_or_iat_omits_the_nonce_and_takes_the_local_clock() {
        let key = DpopKey::generate().unwrap();
        let before = now_secs();
        let proof = key
            .proof("http://127.0.0.1:9/token", &ProofClaims::random())
            .unwrap();
        let claims = proof_part(&proof, 1);
        println!("proof claims: {claims}");
        assert!(claims.get("nonce").is_none(), "{claims}");
        assert_eq!(claims["htu"], "http://127.0.0.1:9/token");
        let iat = claims["iat"].as_i64().unwrap();
        assert!(before <= iat && iat <= now_secs(), "iat {iat}");
        assert_eq!(
            URL_SAFE_NO_PAD
                .decode(claims["jti"].as_str().unwrap())
                .unwrap()
                .len(),
            24
        );
        verify_with_embedded_jwk(&proof).unwrap();
    }

    #[test]
    fn a_tampered_proof_does_not_verify() {
        let key = DpopKey::generate().unwrap();
        let other = DpopKey::generate().unwrap();
        let proof = key.proof(HTU, &ProofClaims::random()).unwrap();
        let foreign = other.proof(HTU, &ProofClaims::random()).unwrap();
        let (_, foreign_signature) = foreign.rsplit_once('.').unwrap();
        let (input, _) = proof.rsplit_once('.').unwrap();
        let swapped = format!("{input}.{foreign_signature}");
        assert!(verify_with_embedded_jwk(&swapped).is_err());
    }

    #[test]
    fn public_jwk_has_only_the_public_members() {
        let jwk = serde_json::to_value(DpopKey::generate().unwrap().public_jwk()).unwrap();
        let mut members: Vec<_> = jwk.as_object().unwrap().keys().cloned().collect();
        members.sort();
        assert_eq!(members, ["crv", "kty", "x", "y"]);
        for coordinate in ["x", "y"] {
            let bytes = URL_SAFE_NO_PAD
                .decode(jwk[coordinate].as_str().unwrap())
                .unwrap();
            assert_eq!(bytes.len(), 32, "{coordinate}");
        }
    }

    #[test]
    fn stored_key_round_trips_to_the_same_key() {
        let key = DpopKey::generate().unwrap();
        let restored = DpopKey::from_stored(&key.to_stored()).unwrap();
        assert_eq!(restored.public_jwk(), key.public_jwk());
        assert_eq!(restored.to_stored(), key.to_stored());
        let proof = restored.proof(HTU, &ProofClaims::random()).unwrap();
        assert_eq!(proof_part(&proof, 0)["jwk"]["x"], key.public_jwk().x);
        verify_with_embedded_jwk(&proof).unwrap();
    }

    #[test]
    fn unreadable_stored_keys_are_rejected() {
        for (name, stored) in [
            ("empty", String::new()),
            ("not base64url", "not base64url!".to_string()),
            ("not PKCS#8", URL_SAFE_NO_PAD.encode(b"synthetic bytes")),
            (
                "truncated PKCS#8",
                URL_SAFE_NO_PAD.encode(
                    &URL_SAFE_NO_PAD
                        .decode(DpopKey::generate().unwrap().to_stored())
                        .unwrap()[..40],
                ),
            ),
        ] {
            assert_eq!(
                DpopKey::from_stored(&stored).unwrap_err(),
                DpopError::UnreadableKey,
                "{name}"
            );
        }
    }

    #[test]
    fn code_exchange_jti_is_the_base64url_sha256_of_the_code() {
        // Expected values computed outside this code, with Python's hashlib.
        assert_eq!(
            code_exchange_jti("abc"),
            "ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0"
        );
        assert_eq!(
            code_exchange_jti("4/0Aabc"),
            "M0wg5hNYWyb1souX0gifr6laC81x-17RZLCgty8r5Ww"
        );
    }

    #[test]
    fn random_jtis_differ() {
        assert_ne!(random_jti(), random_jti());
    }

    #[test]
    fn debug_output_hides_the_private_key() {
        let key = DpopKey::generate().unwrap();
        let printed = format!("{key:?}");
        assert!(!printed.contains(&key.to_stored()), "{printed}");
        assert_eq!(printed, "DpopKey { .. }");
    }
}
