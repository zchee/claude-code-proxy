//! The Vertex AI models this proxy advertises and the rules that map a
//! requested model name to an upstream model ID and wire format.

/// Prefix that routes any model name to this provider.
pub const MODEL_PREFIX: &str = "vertex/";

/// Publisher prefix accepted in front of a Gemini model ID
/// (`vertex/google/gemini-…`).
const GOOGLE_PUBLISHER_PREFIX: &str = "google/";

/// Longest model ID accepted. The ID becomes a URL path segment and part of
/// the thought-signature envelope, so it is bounded.
const MAX_MODEL_ID_BYTES: usize = 256;

/// A Gemini model registered with this proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// The model ID Vertex AI expects in the request URL.
    pub id: &'static str,
    /// Whether the model accepts `thinkingLevel: MINIMAL`. Models that do not
    /// get `LOW` as their smallest thinking level.
    pub supports_minimal: bool,
}

/// The Gemini models on Vertex AI's model pages that this proxy registers.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "gemini-4-argon",
        supports_minimal: false,
    },
    ModelSpec {
        id: "gemini-4-carbon",
        supports_minimal: false,
    },
    ModelSpec {
        id: "gemini-3.8-flash",
        supports_minimal: false,
    },
    ModelSpec {
        id: "gemini-3.7-flash",
        supports_minimal: false,
    },
    ModelSpec {
        id: "gemini-3.5-flash",
        supports_minimal: true,
    },
    ModelSpec {
        id: "gemini-3.5-flash-lite",
        supports_minimal: true,
    },
    ModelSpec {
        id: "gemini-3.1-flash-lite",
        supports_minimal: true,
    },
    ModelSpec {
        id: "gemini-3.1-pro-preview",
        supports_minimal: false,
    },
    ModelSpec {
        id: "gemini-3-flash-preview",
        supports_minimal: true,
    },
];

/// The upstream API dialect a resolved model is reached through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    /// `publishers/google/models/{model}:streamGenerateContent`.
    GenerateContent,
}

/// The outcome of [`resolve`]: what to send upstream for a requested model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// The upstream model ID, without the `vertex/` or publisher prefix. It is
    /// also the model name recorded in thought-signature envelopes.
    pub id: String,
    /// The upstream API dialect.
    pub format: WireFormat,
    /// Whether the model accepts `thinkingLevel: MINIMAL`. Unregistered IDs
    /// get `false`, because `LOW` is accepted by every Gemini model that
    /// takes `thinkingLevel` while `MINIMAL` is not.
    pub supports_minimal: bool,
    /// Whether the ID is in [`MODELS`].
    pub registered: bool,
}

/// A model name this provider cannot serve.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    /// The name does not match any accepted form.
    #[error(
        "unsupported Vertex AI model {requested:?}; use a Gemini model ID (for example \
         gemini-3.5-flash), vertex/<gemini-model-id> or vertex/google/<gemini-model-id>"
    )]
    Unsupported {
        /// The model name as requested.
        requested: String,
    },
    /// The model ID contains characters that cannot be placed in the request
    /// URL.
    #[error(
        "invalid Vertex AI model ID {requested:?}; a model ID starts with a letter or digit \
         and contains only letters, digits, '.', '_', '-' and '@'"
    )]
    InvalidId {
        /// The model name as requested.
        requested: String,
    },
}

/// Maps a requested model name to the upstream model.
///
/// `vertex/` is removed first. A name without `/` and a `google/gemini-…`
/// name are Gemini models; an ID that is not in [`MODELS`] is still accepted
/// with conservative defaults so new models work before they are registered.
///
/// # Errors
///
/// Returns [`ModelError::Unsupported`] for any other form and
/// [`ModelError::InvalidId`] when the ID could not be placed safely in the
/// request URL.
pub fn resolve(raw: &str) -> Result<ResolvedModel, ModelError> {
    let name = raw.strip_prefix(MODEL_PREFIX).unwrap_or(raw);
    let id = match name.strip_prefix(GOOGLE_PUBLISHER_PREFIX) {
        Some(id) if id.starts_with("gemini-") && !id.contains('/') => id,
        Some(_) => {
            return Err(ModelError::Unsupported {
                requested: raw.to_string(),
            });
        }
        None if name.contains('/') => {
            return Err(ModelError::Unsupported {
                requested: raw.to_string(),
            });
        }
        None => name,
    };
    if !is_valid_model_id(id) {
        return Err(ModelError::InvalidId {
            requested: raw.to_string(),
        });
    }
    let spec = MODELS.iter().find(|spec| spec.id == id);
    Ok(ResolvedModel {
        id: id.to_string(),
        format: WireFormat::GenerateContent,
        supports_minimal: spec.is_some_and(|spec| spec.supports_minimal),
        registered: spec.is_some(),
    })
}

/// Every model name this provider lists: each registered ID bare and with the
/// `vertex/` prefix, sorted.
pub fn advertised_models() -> Vec<String> {
    let mut result = Vec::with_capacity(MODELS.len() * 2);
    for model in MODELS {
        result.push(model.id.to_string());
        result.push(format!("{MODEL_PREFIX}{}", model.id));
    }
    result.sort_unstable();
    result
}

fn is_valid_model_id(id: &str) -> bool {
    let mut bytes = id.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    id.len() <= MAX_MODEL_ID_BYTES
        && first.is_ascii_alphanumeric()
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || b"._-@".contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registered models and whether each accepts `MINIMAL`, as listed on
    /// the Vertex AI model pages: Gemini 4 and 3.7 / 3.8 Flash take LOW to
    /// HIGH only, and 3.1 Pro Preview takes LOW / MEDIUM / HIGH.
    const EXPECTED: &[(&str, bool)] = &[
        ("gemini-4-argon", false),
        ("gemini-4-carbon", false),
        ("gemini-3.8-flash", false),
        ("gemini-3.7-flash", false),
        ("gemini-3.5-flash", true),
        ("gemini-3.5-flash-lite", true),
        ("gemini-3.1-flash-lite", true),
        ("gemini-3.1-pro-preview", false),
        ("gemini-3-flash-preview", true),
    ];

    #[test]
    fn every_registered_model_resolves_bare_and_prefixed() {
        assert_eq!(MODELS.len(), EXPECTED.len());
        for (id, supports_minimal) in EXPECTED {
            for raw in [
                id.to_string(),
                format!("vertex/{id}"),
                format!("vertex/google/{id}"),
            ] {
                let resolved = resolve(&raw).unwrap_or_else(|error| panic!("{raw}: {error}"));
                assert_eq!(
                    resolved,
                    ResolvedModel {
                        id: id.to_string(),
                        format: WireFormat::GenerateContent,
                        supports_minimal: *supports_minimal,
                        registered: true,
                    },
                    "{raw}"
                );
            }
        }
    }

    #[test]
    fn unregistered_ids_get_conservative_defaults() {
        let tests = [
            ("vertex/gemini-9-future", "gemini-9-future"),
            ("gemini-2.5-pro", "gemini-2.5-pro"),
            ("vertex/google/gemini-3.6-flash", "gemini-3.6-flash"),
            ("vertex/gemini-3.5-flash@001", "gemini-3.5-flash@001"),
        ];
        for (raw, id) in tests {
            let resolved = resolve(raw).unwrap_or_else(|error| panic!("{raw}: {error}"));
            assert_eq!(resolved.id, id, "{raw}");
            assert!(!resolved.registered, "{raw}");
            assert!(!resolved.supports_minimal, "{raw}");
            assert_eq!(resolved.format, WireFormat::GenerateContent, "{raw}");
        }
    }

    #[test]
    fn other_publishers_and_endpoints_are_rejected() {
        for raw in [
            "vertex/anthropic/claude-opus-5-5",
            "vertex/xai/grok-4.7",
            "vertex/endpoint/my-endpoint",
            "vertex/google/gemma-4-26b-a4b-it-maas",
            "vertex/google/gemini-3.5-flash/extra",
        ] {
            assert_eq!(
                resolve(raw),
                Err(ModelError::Unsupported {
                    requested: raw.to_string()
                }),
                "{raw}"
            );
        }
        let message = resolve("vertex/xai/grok-4.7").unwrap_err().to_string();
        assert!(message.contains("vertex/<gemini-model-id>"), "{message}");
    }

    #[test]
    fn ids_that_would_break_the_url_or_envelope_are_rejected() {
        for raw in [
            "vertex/",
            "",
            "vertex/gemini-3.5-flash?alt=json",
            "vertex/gemini-3.5-flash:generateContent",
            "vertex/gemini 3",
            "vertex/-gemini",
            "vertex/gemini-3.5-flash#x",
        ] {
            assert_eq!(
                resolve(raw),
                Err(ModelError::InvalidId {
                    requested: raw.to_string()
                }),
                "{raw:?}"
            );
        }
        let long = format!("vertex/{}", "a".repeat(MAX_MODEL_ID_BYTES + 1));
        assert!(matches!(resolve(&long), Err(ModelError::InvalidId { .. })));
    }

    #[test]
    fn advertised_models_are_the_nine_ids_bare_and_prefixed() {
        let advertised = advertised_models();
        assert_eq!(advertised.len(), 18);
        for (id, _) in EXPECTED {
            assert!(advertised.contains(&id.to_string()), "{id}");
            assert!(advertised.contains(&format!("vertex/{id}")), "{id}");
        }
        let mut sorted = advertised.clone();
        sorted.sort_unstable();
        assert_eq!(advertised, sorted);
    }
}
