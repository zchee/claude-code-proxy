//! The models of the Google AI Ultra plan and the thinking levels each one
//! accepts.
//!
//! The plan has no model list endpoint, so the list is fixed. The bare
//! Gemini IDs already route to the Vertex AI provider, so every name of this
//! provider carries [`MODEL_PREFIX`].

use serde::Serialize;

/// Prefix that routes a model name to this provider.
pub const MODEL_PREFIX: &str = "google-ai-plan/";

/// Context window of every plan model, in tokens.
pub const CONTEXT_WINDOW_TOKENS: u64 = 1_048_576;

/// Largest output of every plan model, in tokens.
pub const MAX_OUTPUT_TOKENS: u32 = 65_536;

/// Longest model ID accepted after the prefix. The ID is sent in the body
/// and recorded in every thought-signature envelope, so it is bounded.
const MAX_MODEL_ID_BYTES: usize = 256;

/// `generation_config.thinking_level`, lowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// Minimal.
    Minimal,
    /// Low.
    Low,
    /// Medium.
    Medium,
    /// High.
    High,
    /// Extra high.
    Xhigh,
}

/// The inclusive range of thinking levels a model accepts. None of the plan
/// models can turn thinking off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingRange {
    /// The lowest accepted level.
    pub min: ThinkingLevel,
    /// The highest accepted level.
    pub max: ThinkingLevel,
}

impl ThinkingRange {
    /// `level` moved to the nearest accepted level. The accepted levels are
    /// contiguous, so only a level outside the range changes.
    pub fn clamp(self, level: ThinkingLevel) -> ThinkingLevel {
        level.clamp(self.min, self.max)
    }

    /// Whether the model accepts `level` unchanged.
    pub fn contains(self, level: ThinkingLevel) -> bool {
        (self.min..=self.max).contains(&level)
    }
}

/// Gemini 4 Hydrogen and Neon.
const LOW_TO_XHIGH: ThinkingRange = ThinkingRange {
    min: ThinkingLevel::Low,
    max: ThinkingLevel::Xhigh,
};

/// Gemini 4 Carbon, and any model outside the list.
const LOW_TO_HIGH: ThinkingRange = ThinkingRange {
    min: ThinkingLevel::Low,
    max: ThinkingLevel::High,
};

/// Gemini 3.5 Flash-Lite.
const MINIMAL_TO_HIGH: ThinkingRange = ThinkingRange {
    min: ThinkingLevel::Minimal,
    max: ThinkingLevel::High,
};

/// A model the plan includes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// The model ID the Interactions API expects.
    pub id: &'static str,
    /// The display name.
    pub display_name: &'static str,
    /// The thinking levels the model accepts.
    pub thinking: ThinkingRange,
}

/// The models of the Google AI Ultra plan.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "gemini-4-hydrogen",
        display_name: "Gemini 4 Hydrogen",
        thinking: LOW_TO_XHIGH,
    },
    ModelSpec {
        id: "gemini-4-neon",
        display_name: "Gemini 4 Neon",
        thinking: LOW_TO_XHIGH,
    },
    ModelSpec {
        id: "gemini-4-carbon",
        display_name: "Gemini 4 Carbon",
        thinking: LOW_TO_HIGH,
    },
    ModelSpec {
        id: "gemini-3.5-flash-lite",
        display_name: "Gemini 3.5 Flash-Lite",
        thinking: MINIMAL_TO_HIGH,
    },
];

/// A requested model name resolved to what the request needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// The model ID sent upstream and recorded in signature envelopes.
    pub upstream_model: String,
    /// The thinking levels the model accepts.
    pub thinking: ThinkingRange,
    /// Whether the ID is in [`MODELS`].
    pub registered: bool,
}

/// Why a model name cannot be resolved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    /// The name does not start with [`MODEL_PREFIX`].
    #[error(
        "model {requested:?} is not a Google AI Ultra plan model: names start with {MODEL_PREFIX}"
    )]
    NotThisProvider {
        /// The name as requested.
        requested: String,
    },
    /// Nothing follows the prefix.
    #[error("model {requested:?} names no model after {MODEL_PREFIX}")]
    EmptyId {
        /// The name as requested.
        requested: String,
    },
    /// The ID after the prefix is too long or has characters a Gemini model
    /// ID does not use.
    #[error(
        "model {requested:?} is not a valid Gemini model ID: use letters, digits, '.', '_' and '-', at most {MAX_MODEL_ID_BYTES} characters"
    )]
    InvalidId {
        /// The name as requested.
        requested: String,
    },
}

/// The prefixed names of every model in [`MODELS`].
pub fn advertised_models() -> Vec<String> {
    MODELS
        .iter()
        .map(|model| format!("{MODEL_PREFIX}{}", model.id))
        .collect()
}

/// Resolves a requested name such as `google-ai-plan/gemini-4-neon`.
///
/// A prefixed ID outside [`MODELS`] is still resolved, with the levels
/// `low..high` that every Gemini 4 model accepts: Google answers for it, and
/// the error classifier explains a model the plan does not include.
///
/// # Errors
///
/// [`ModelError`] when the prefix is missing, nothing follows it, or the ID
/// is not shaped like a Gemini model ID (`:` would break the signature
/// envelope).
pub fn resolve(requested: &str) -> Result<ResolvedModel, ModelError> {
    let Some(id) = requested.strip_prefix(MODEL_PREFIX) else {
        return Err(ModelError::NotThisProvider {
            requested: requested.to_string(),
        });
    };
    if id.is_empty() {
        return Err(ModelError::EmptyId {
            requested: requested.to_string(),
        });
    }
    let well_formed = id.len() <= MAX_MODEL_ID_BYTES
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte));
    if !well_formed {
        return Err(ModelError::InvalidId {
            requested: requested.to_string(),
        });
    }
    let spec = MODELS.iter().find(|model| model.id == id);
    Ok(ResolvedModel {
        upstream_model: id.to_string(),
        thinking: spec.map_or(LOW_TO_HIGH, |model| model.thinking),
        registered: spec.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertised_models_are_the_four_prefixed_plan_models() {
        assert_eq!(
            advertised_models(),
            vec![
                "google-ai-plan/gemini-4-hydrogen",
                "google-ai-plan/gemini-4-neon",
                "google-ai-plan/gemini-4-carbon",
                "google-ai-plan/gemini-3.5-flash-lite",
            ]
        );
        for name in advertised_models() {
            let model = resolve(&name).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(model.registered, "{name} should be registered");
        }
    }

    #[test]
    fn each_model_resolves_to_its_documented_thinking_levels() {
        use ThinkingLevel::*;
        let tests = [
            ("google-ai-plan/gemini-4-hydrogen", Low, Xhigh, true),
            ("google-ai-plan/gemini-4-neon", Low, Xhigh, true),
            ("google-ai-plan/gemini-4-carbon", Low, High, true),
            ("google-ai-plan/gemini-3.5-flash-lite", Minimal, High, true),
            ("google-ai-plan/gemini-4-argon", Low, High, false),
        ];
        for (name, min, max, registered) in tests {
            let model = resolve(name).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                model,
                ResolvedModel {
                    upstream_model: name.trim_start_matches(MODEL_PREFIX).to_string(),
                    thinking: ThinkingRange { min, max },
                    registered,
                },
                "{name}"
            );
        }
    }

    #[test]
    fn clamping_moves_only_levels_outside_the_range() {
        use ThinkingLevel::*;
        let all = [Minimal, Low, Medium, High, Xhigh];
        let tests = [
            (LOW_TO_XHIGH, [Low, Low, Medium, High, Xhigh]),
            (LOW_TO_HIGH, [Low, Low, Medium, High, High]),
            (MINIMAL_TO_HIGH, [Minimal, Low, Medium, High, High]),
        ];
        for (range, expected) in tests {
            for (level, want) in all.into_iter().zip(expected) {
                assert_eq!(range.clamp(level), want, "{range:?} clamps {level:?}");
                assert!(range.contains(want), "{range:?} accepts {want:?}");
            }
        }
    }

    #[test]
    fn malformed_names_are_rejected() {
        let tests = [
            ("gemini-4-neon", "not a Google AI Ultra plan model"),
            ("vertex/gemini-4-neon", "not a Google AI Ultra plan model"),
            ("google-ai-plan/", "names no model"),
            ("google-ai-plan/gemini:4", "not a valid Gemini model ID"),
            ("google-ai-plan/gemini 4", "not a valid Gemini model ID"),
            ("google-ai-plan/a/b", "not a valid Gemini model ID"),
        ];
        for (name, expected) in tests {
            let error = resolve(name).expect_err(name).to_string();
            assert!(error.contains(expected), "{name}: {error}");
        }
        let long = format!("{MODEL_PREFIX}{}", "a".repeat(MAX_MODEL_ID_BYTES + 1));
        assert!(matches!(resolve(&long), Err(ModelError::InvalidId { .. })));
        let limit = format!("{MODEL_PREFIX}{}", "a".repeat(MAX_MODEL_ID_BYTES));
        assert!(resolve(&limit).is_ok(), "the length limit is inclusive");
    }

    #[test]
    fn levels_serialize_as_the_wire_names() {
        let wire = serde_json::to_value([
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Xhigh,
        ])
        .expect("levels serialize");
        assert_eq!(
            wire,
            serde_json::json!(["minimal", "low", "medium", "high", "xhigh"])
        );
    }
}
