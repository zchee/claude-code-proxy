//! The Vertex AI models this proxy knows and the rules that map a requested
//! model name to a model family, the upstream model string, the wire format,
//! the API version and the rule that picks the location.

use crate::config::{
    VERTEX_LOCATION_OVERRIDES_ENV, VERTEX_LOCATION_OVERRIDES_KEY, VertexLocationOverrides,
};

/// Prefix that routes any model name to this provider.
pub const MODEL_PREFIX: &str = "vertex/";

/// Prefix, after `vertex/`, of a deployed endpoint (`vertex/endpoint/<name>`).
const ENDPOINT_PREFIX: &str = "endpoint/";

/// Suffix of the Model Garden managed open models (`…-maas`).
const OPEN_MODEL_SUFFIX: &str = "-maas";

/// Name of the managed open models in messages. They have no family yet.
pub const OPEN_MODEL_FAMILY_LABEL: &str =
    "Model Garden managed open model (<publisher>/<model-id>-maas)";

/// Name of the deployed endpoints in messages. They have no family yet.
pub const DEPLOYED_ENDPOINT_FAMILY_LABEL: &str = "deployed endpoint (endpoint/<name>)";

/// The locations Google documents for every Mistral model.
pub const MISTRAL_LOCATIONS: &[&str] = &["us-central1", "europe-west4"];

/// Longest publisher accepted in a model name.
const MAX_PUBLISHER_BYTES: usize = 64;

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

/// A group of models that share one API on Vertex AI and one handler in
/// this proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// Google Gemini, through `generateContent`.
    Gemini,
    /// Anthropic Claude, through Anthropic Messages on `rawPredict`.
    Claude,
    /// xAI Grok, through the OpenAI-compatible Responses API under
    /// `endpoints/openapi`.
    Grok,
    /// OpenAI GPT, through the Responses API under the publisher.
    OpenAi,
    /// Meta Muse, through the Responses API under the publisher.
    MetaMuse,
    /// Mistral, through Mistral's chat format on `rawPredict`.
    Mistral,
}

impl Family {
    /// Every family.
    pub const ALL: [Family; 6] = [
        Family::Gemini,
        Family::Claude,
        Family::Grok,
        Family::OpenAi,
        Family::MetaMuse,
        Family::Mistral,
    ];

    /// The family's name in messages.
    pub const fn label(self) -> &'static str {
        match self {
            Family::Gemini => "Gemini",
            Family::Claude => "Anthropic Claude",
            Family::Grok => "xAI Grok",
            Family::OpenAi => "OpenAI",
            Family::MetaMuse => "Meta Muse",
            Family::Mistral => "Mistral",
        }
    }

    /// The upstream API dialect of every model of the family.
    pub const fn wire(self) -> WireFormat {
        match self {
            Family::Gemini => WireFormat::GenerateContent,
            Family::Claude => WireFormat::AnthropicMessages,
            Family::Grok => WireFormat::Responses {
                route: ResponsesRoute::OpenApi,
            },
            Family::OpenAi | Family::MetaMuse => WireFormat::Responses {
                route: ResponsesRoute::Publisher,
            },
            Family::Mistral => WireFormat::MistralChat,
        }
    }

    /// Whether this version of the proxy sends requests for the family.
    /// Only such families are advertised, and a model of any other family
    /// is answered with a 400 that says so. The commit that adds a family's
    /// handler turns this on for it.
    pub const fn is_served(self) -> bool {
        matches!(
            self,
            Family::Gemini
                | Family::Claude
                | Family::Grok
                | Family::OpenAi
                | Family::MetaMuse
                | Family::Mistral
        )
    }

    /// The location rule of a model of the family that is not in the table.
    pub const fn default_location(self) -> LocationRule {
        match self {
            Family::MetaMuse => LocationRule::Fixed("global"),
            Family::Mistral => LocationRule::Required {
                allowed: MISTRAL_LOCATIONS,
            },
            Family::Gemini | Family::Claude | Family::Grok | Family::OpenAi => {
                LocationRule::Inherit
            }
        }
    }

    /// The upstream model string for model ID `id` of `publisher`: Grok
    /// wants the publisher in front (`xai/grok-4.7`), every other family the
    /// bare ID.
    fn upstream_model(self, publisher: &str, id: &str) -> String {
        match self {
            Family::Grok => format!("{publisher}/{id}"),
            Family::Gemini
            | Family::Claude
            | Family::OpenAi
            | Family::MetaMuse
            | Family::Mistral => id.to_string(),
        }
    }

    /// The names of the family's registered models as they would be listed:
    /// each Gemini ID bare and with `vertex/`, and every other model as
    /// `vertex/<publisher>/<model-id>`. Not sorted.
    pub fn model_names(self) -> Vec<String> {
        match self {
            Family::Gemini => MODELS
                .iter()
                .flat_map(|model| [model.id.to_string(), format!("{MODEL_PREFIX}{}", model.id)])
                .collect(),
            family => PARTNER_MODELS
                .iter()
                .filter(|model| model.family == family)
                .map(|model| format!("{MODEL_PREFIX}{}/{}", model.publisher, model.id))
                .collect(),
        }
    }
}

/// The upstream API dialect a resolved model is reached through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    /// `publishers/google/models/{model}:streamGenerateContent`.
    GenerateContent,
    /// Anthropic Messages on
    /// `publishers/anthropic/models/{model}:rawPredict` / `:streamRawPredict`.
    AnthropicMessages,
    /// The OpenAI-compatible Responses API.
    Responses {
        /// Where the `responses` path sits.
        route: ResponsesRoute,
    },
    /// Mistral's chat format on
    /// `publishers/mistralai/models/{model}:rawPredict` / `:streamRawPredict`.
    MistralChat,
}

/// Where a Responses request is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesRoute {
    /// `endpoints/openapi/responses`, with the publisher in the body's
    /// `model`.
    OpenApi,
    /// `publishers/<publisher>/v1/responses`, with the bare model ID.
    Publisher,
}

/// The Vertex AI REST API version in the request path. It belongs to the
/// model, not to the user's settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiVersion {
    /// `v1`.
    V1,
    /// `v1beta1`.
    V1Beta1,
}

impl ApiVersion {
    /// The path segment.
    pub const fn as_str(self) -> &'static str {
        match self {
            ApiVersion::V1 => "v1",
            ApiVersion::V1Beta1 => "v1beta1",
        }
    }
}

/// How the location of a model's requests is chosen when no override names
/// the model itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocationRule {
    /// A publisher override, else the configured location.
    Inherit,
    /// Always this location, whatever the publisher override or the
    /// configured location say: the only one Google documents for the
    /// model.
    Fixed(&'static str),
    /// A publisher override, else the configured location when it is one
    /// of `allowed`; otherwise the request fails, because the configured
    /// location's default (`global`) does not serve the model.
    Required {
        /// The locations Google documents for the model.
        allowed: &'static [&'static str],
    },
}

/// A partner model registered with this proxy as
/// `vertex/<publisher>/<id>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartnerModel {
    /// The Vertex AI publisher.
    pub publisher: &'static str,
    /// The model ID, as written after the publisher in the model name.
    pub id: &'static str,
    /// The family whose handler serves it.
    pub family: Family,
    /// The API version in the request path.
    pub api_version: ApiVersion,
    /// How its location is chosen.
    pub location: LocationRule,
}

/// A row with the family's defaults: `v1` and the family's location rule.
const fn partner(family: Family, publisher: &'static str, id: &'static str) -> PartnerModel {
    PartnerModel {
        publisher,
        id,
        family,
        api_version: ApiVersion::V1,
        location: family.default_location(),
    }
}

/// The partner models on Vertex AI's model pages that this proxy registers.
/// IDs not listed still resolve through the fallback patterns of
/// [`resolve`].
pub const PARTNER_MODELS: &[PartnerModel] = &[
    partner(Family::Claude, "anthropic", "claude-fable-5-2"),
    partner(Family::Claude, "anthropic", "claude-fable-5-1"),
    partner(Family::Claude, "anthropic", "claude-fable-5"),
    partner(Family::Claude, "anthropic", "claude-opus-5-5"),
    partner(Family::Claude, "anthropic", "claude-opus-5"),
    partner(Family::Claude, "anthropic", "claude-opus-4-8"),
    partner(Family::Claude, "anthropic", "claude-opus-4-7"),
    partner(Family::Claude, "anthropic", "claude-sonnet-5-5"),
    partner(Family::Claude, "anthropic", "claude-sonnet-5"),
    partner(Family::Claude, "anthropic", "claude-haiku-5-5"),
    partner(Family::Claude, "anthropic", "claude-opus-4-6"),
    partner(Family::Claude, "anthropic", "claude-sonnet-4-6"),
    partner(Family::Claude, "anthropic", "claude-opus-4-5@20251101"),
    partner(Family::Claude, "anthropic", "claude-haiku-4-5@20251001"),
    partner(Family::Grok, "xai", "grok-4.7"),
    partner(Family::Grok, "xai", "grok-4.6"),
    partner(Family::Grok, "xai", "grok-4.20-reasoning"),
    partner(Family::Grok, "xai", "grok-4.20-non-reasoning"),
    partner(Family::Grok, "xai", "grok-4.3"),
    partner(Family::MetaMuse, "meta", "muse-spark-1.3"),
    partner(Family::MetaMuse, "meta", "muse-wonder-1.0"),
    partner(Family::MetaMuse, "meta", "muse-ember-1.0"),
    partner(Family::OpenAi, "openai", "gpt-6-astra"),
    partner(Family::OpenAi, "openai", "gpt-6.1-sol"),
    partner(Family::OpenAi, "openai", "gpt-6.1-terra"),
    partner(Family::OpenAi, "openai", "gpt-6-luna"),
    partner(Family::Mistral, "mistralai", "mistral-medium-3"),
    partner(Family::Mistral, "mistralai", "mistral-small-2503"),
    partner(Family::Mistral, "mistralai", "codestral-2"),
];

/// The outcome of [`resolve`]: what to send upstream for a requested model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// The family whose handler serves the model.
    pub family: Family,
    /// The upstream API dialect, the family's.
    pub wire: WireFormat,
    /// The Vertex AI publisher (`google` for Gemini).
    pub publisher: String,
    /// The model ID as named after the publisher, without `vertex/` and the
    /// publisher. With the publisher it is the model's override key.
    pub model_id: String,
    /// The model string Vertex AI expects, in the URL or in the body's
    /// `model`, depending on the wire format. For Gemini it is also the
    /// model name recorded in thought-signature envelopes.
    pub upstream_model: String,
    /// The API version in the request path.
    pub api_version: ApiVersion,
    /// How the location of the model's requests is chosen.
    pub location: LocationRule,
    /// Whether the model accepts `thinkingLevel: MINIMAL`. Only Gemini
    /// models read it. Unregistered Gemini IDs get `false`, because `LOW`
    /// is accepted by every Gemini model that takes `thinkingLevel` while
    /// `MINIMAL` is not.
    pub supports_minimal: bool,
    /// Whether the ID is in [`MODELS`] or [`PARTNER_MODELS`].
    pub registered: bool,
}

impl ResolvedModel {
    /// The key a location override names this model by:
    /// `<publisher>/<model-id>`.
    pub fn override_key(&self) -> String {
        format!("{}/{}", self.publisher, self.model_id)
    }

    /// The location for this model's requests.
    ///
    /// An override naming the model wins. Then a [`LocationRule::Fixed`]
    /// location, which an override naming only the publisher does not
    /// move: Muse and the Llama models share the publisher `meta` but not
    /// their locations. Then an override naming the publisher; then
    /// `configured`, which for [`LocationRule::Required`] must be one of the
    /// rule's locations.
    ///
    /// # Errors
    ///
    /// [`LocationRule::Required`] without an override naming the model or
    /// its publisher, and with a `configured` location outside the rule's
    /// list, gives [`LocationRequired`].
    pub fn location<'a>(
        &self,
        overrides: &'a VertexLocationOverrides,
        configured: &'a str,
    ) -> Result<&'a str, LocationRequired> {
        if let Some(location) = overrides.get(&self.override_key()) {
            return Ok(location);
        }
        let by_publisher = overrides.get(&self.publisher);
        match self.location {
            LocationRule::Fixed(location) => Ok(location),
            LocationRule::Inherit => Ok(by_publisher.unwrap_or(configured)),
            LocationRule::Required { allowed } => match by_publisher {
                Some(location) => Ok(location),
                None if allowed.contains(&configured) => Ok(configured),
                None => Err(LocationRequired {
                    publisher: self.publisher.clone(),
                    configured: configured.to_string(),
                    allowed,
                }),
            },
        }
    }
}

/// A model that is served only in some locations has neither an override
/// nor a configured location among them.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "Vertex AI models of publisher {publisher} are not served in location {configured}: set \
     {VERTEX_LOCATION_OVERRIDES_ENV}={publisher}=<location>, \"{publisher}\": \"<location>\" in \
     {VERTEX_LOCATION_OVERRIDES_KEY} in config.json, or the location itself \
     (CCP_VERTEX_LOCATION, GOOGLE_CLOUD_LOCATION or vertex.location), with {} as <location>",
    .allowed.join(" or ")
)]
pub struct LocationRequired {
    /// The model's publisher, the override key to set.
    pub publisher: String,
    /// The configured location, which does not serve the model.
    pub configured: String,
    /// The locations Google documents for the model.
    pub allowed: &'static [&'static str],
}

/// A model name this provider cannot serve.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    /// The name does not match any accepted form.
    #[error(
        "unsupported Vertex AI model {requested:?}; use a Gemini model ID (for example \
         gemini-3.5-flash), vertex/<gemini-model-id> or vertex/google/<gemini-model-id>, or \
         vertex/<publisher>/<model-id> with publisher anthropic, xai, mistralai, openai (gpt-* \
         models) or meta (muse-* models)"
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
    /// The name is valid, but this version of the proxy does not send
    /// requests for its family yet.
    #[error(
        "this version of claude-code-proxy does not support the {family} model family on \
         Vertex AI (requested {requested:?}); supported families: {}",
        served_family_labels()
    )]
    FamilyNotSupported {
        /// The model name as requested.
        requested: String,
        /// The family's name, from [`Family::label`] or one of the labels of
        /// the forms that have no family yet.
        family: &'static str,
    },
}

fn served_family_labels() -> String {
    let labels: Vec<&str> = Family::ALL
        .into_iter()
        .filter(|family| family.is_served())
        .map(Family::label)
        .collect();
    labels.join(", ")
}

/// Maps a requested model name to the upstream model.
///
/// `vertex/` is removed first, then the first rule that matches decides:
///
/// 1. a name without `/` is a Gemini model ID;
/// 2. `endpoint/<name>` is a deployed endpoint, not supported yet;
/// 3. `<publisher>/<model-id>` in [`PARTNER_MODELS`] takes that row;
/// 4. otherwise by pattern: `google/gemini-*` is Gemini, `anthropic/*`
///    Claude, `xai/*` Grok, `mistralai/*` Mistral, `openai/gpt-*` OpenAI and
///    `meta/muse-*` Muse unless they end in `-maas`, and any
///    `<publisher>/<model-id>-maas` a managed open model, not supported yet.
///
/// An ID found only by pattern is accepted with its family's defaults so
/// new models work before they are registered.
///
/// # Errors
///
/// Returns [`ModelError::Unsupported`] for any other form,
/// [`ModelError::InvalidId`] when the ID could not be placed safely in the
/// request URL, and [`ModelError::FamilyNotSupported`] for open models and
/// deployed endpoints.
pub fn resolve(raw: &str) -> Result<ResolvedModel, ModelError> {
    let name = raw.strip_prefix(MODEL_PREFIX).unwrap_or(raw);
    let unsupported = || ModelError::Unsupported {
        requested: raw.to_string(),
    };
    let invalid = || ModelError::InvalidId {
        requested: raw.to_string(),
    };
    let not_supported = |family| ModelError::FamilyNotSupported {
        requested: raw.to_string(),
        family,
    };
    let Some((publisher, id)) = name.split_once('/') else {
        return gemini(name).ok_or_else(invalid);
    };
    if name.starts_with(ENDPOINT_PREFIX) {
        return Err(not_supported(DEPLOYED_ENDPOINT_FAMILY_LABEL));
    }
    if let Some(row) = PARTNER_MODELS
        .iter()
        .find(|row| row.publisher == publisher && row.id == id)
    {
        return Ok(ResolvedModel {
            family: row.family,
            wire: row.family.wire(),
            publisher: row.publisher.to_string(),
            model_id: row.id.to_string(),
            upstream_model: row.family.upstream_model(row.publisher, row.id),
            api_version: row.api_version,
            location: row.location,
            supports_minimal: false,
            registered: true,
        });
    }
    if id.contains('/') {
        return Err(unsupported());
    }
    let open_model = id.ends_with(OPEN_MODEL_SUFFIX);
    let family = match publisher {
        "google" if id.starts_with("gemini-") => {
            return gemini(id).ok_or_else(invalid);
        }
        "anthropic" => Family::Claude,
        "xai" => Family::Grok,
        "mistralai" => Family::Mistral,
        "openai" if id.starts_with("gpt-") && !open_model => Family::OpenAi,
        "meta" if id.starts_with("muse-") && !open_model => Family::MetaMuse,
        _ if open_model && is_valid_publisher(publisher) => {
            return Err(if is_valid_model_id(id) {
                not_supported(OPEN_MODEL_FAMILY_LABEL)
            } else {
                invalid()
            });
        }
        _ => return Err(unsupported()),
    };
    if !is_valid_model_id(id) {
        return Err(invalid());
    }
    Ok(ResolvedModel {
        family,
        wire: family.wire(),
        publisher: publisher.to_string(),
        model_id: id.to_string(),
        upstream_model: family.upstream_model(publisher, id),
        api_version: ApiVersion::V1,
        location: family.default_location(),
        supports_minimal: false,
        registered: false,
    })
}

/// A Gemini model, or `None` when the ID could not be placed in the URL.
fn gemini(id: &str) -> Option<ResolvedModel> {
    if !is_valid_model_id(id) {
        return None;
    }
    let spec = MODELS.iter().find(|spec| spec.id == id);
    Some(ResolvedModel {
        family: Family::Gemini,
        wire: WireFormat::GenerateContent,
        publisher: "google".to_string(),
        model_id: id.to_string(),
        upstream_model: id.to_string(),
        api_version: ApiVersion::V1,
        location: LocationRule::Inherit,
        supports_minimal: spec.is_some_and(|spec| spec.supports_minimal),
        registered: spec.is_some(),
    })
}

/// Every model name this provider lists: the names of every family this
/// version serves (see [`Family::is_served`] and [`Family::model_names`]),
/// sorted.
pub fn advertised_models() -> Vec<String> {
    let mut result: Vec<String> = Family::ALL
        .into_iter()
        .filter(|family| family.is_served())
        .flat_map(Family::model_names)
        .collect();
    result.sort_unstable();
    result
}

fn is_valid_publisher(publisher: &str) -> bool {
    let mut bytes = publisher.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    publisher.len() <= MAX_PUBLISHER_BYTES
        && (first.is_ascii_lowercase() || first.is_ascii_digit())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
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
                        family: Family::Gemini,
                        wire: WireFormat::GenerateContent,
                        publisher: "google".to_string(),
                        model_id: id.to_string(),
                        upstream_model: id.to_string(),
                        api_version: ApiVersion::V1,
                        location: LocationRule::Inherit,
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
            assert_eq!(resolved.upstream_model, id, "{raw}");
            assert!(!resolved.registered, "{raw}");
            assert!(!resolved.supports_minimal, "{raw}");
            assert_eq!(resolved.wire, WireFormat::GenerateContent, "{raw}");
        }
    }

    #[test]
    fn other_publishers_and_endpoints_are_rejected() {
        for raw in [
            "vertex/ai21/jamba-1.5-large",
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
        for (raw, family) in [
            (
                "vertex/endpoint/my-endpoint",
                DEPLOYED_ENDPOINT_FAMILY_LABEL,
            ),
            (
                "vertex/google/gemma-4-26b-a4b-it-maas",
                OPEN_MODEL_FAMILY_LABEL,
            ),
        ] {
            assert_eq!(
                resolve(raw),
                Err(ModelError::FamilyNotSupported {
                    requested: raw.to_string(),
                    family,
                }),
                "{raw}"
            );
        }
        let message = resolve("vertex/ai21/jamba-1.5-large")
            .unwrap_err()
            .to_string();
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
        assert_eq!(advertised.len(), 47);
        for (id, _) in EXPECTED {
            assert!(advertised.contains(&id.to_string()), "{id}");
            assert!(advertised.contains(&format!("vertex/{id}")), "{id}");
        }
        let mut sorted = advertised.clone();
        sorted.sort_unstable();
        assert_eq!(advertised, sorted);
    }
    const OPENAPI_RESPONSES: WireFormat = WireFormat::Responses {
        route: ResponsesRoute::OpenApi,
    };
    const PUBLISHER_RESPONSES: WireFormat = WireFormat::Responses {
        route: ResponsesRoute::Publisher,
    };
    const MISTRAL_REQUIRED: LocationRule = LocationRule::Required {
        allowed: &["us-central1", "europe-west4"],
    };

    /// The partner rows as Google's model pages and the Vertex AI
    /// partner-model guides give them: the name the proxy accepts, the
    /// family, the wire format, the string sent upstream, the API version
    /// and the location rule. Grok is the only family whose upstream string
    /// carries the publisher (`xai/<id>`, as the Grok Responses examples
    /// send it); Muse is documented in `global` only; Mistral is documented
    /// in us-central1 and europe-west4 only, so it has no usable default.
    const EXPECTED_PARTNERS: &[(&str, Family, WireFormat, &str, &str, LocationRule)] = &[
        (
            "vertex/anthropic/claude-fable-5-2",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-fable-5-2",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-fable-5-1",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-fable-5-1",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-fable-5",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-fable-5",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-opus-5-5",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-opus-5-5",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-opus-5",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-opus-5",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-opus-4-8",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-opus-4-8",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-opus-4-7",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-opus-4-7",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-sonnet-5-5",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-sonnet-5-5",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-sonnet-5",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-sonnet-5",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-haiku-5-5",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-haiku-5-5",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-opus-4-6",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-opus-4-6",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-sonnet-4-6",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-sonnet-4-6",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-opus-4-5@20251101",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-opus-4-5@20251101",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/anthropic/claude-haiku-4-5@20251001",
            Family::Claude,
            WireFormat::AnthropicMessages,
            "claude-haiku-4-5@20251001",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/xai/grok-4.7",
            Family::Grok,
            OPENAPI_RESPONSES,
            "xai/grok-4.7",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/xai/grok-4.6",
            Family::Grok,
            OPENAPI_RESPONSES,
            "xai/grok-4.6",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/xai/grok-4.20-reasoning",
            Family::Grok,
            OPENAPI_RESPONSES,
            "xai/grok-4.20-reasoning",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/xai/grok-4.20-non-reasoning",
            Family::Grok,
            OPENAPI_RESPONSES,
            "xai/grok-4.20-non-reasoning",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/xai/grok-4.3",
            Family::Grok,
            OPENAPI_RESPONSES,
            "xai/grok-4.3",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/meta/muse-spark-1.3",
            Family::MetaMuse,
            PUBLISHER_RESPONSES,
            "muse-spark-1.3",
            "v1",
            LocationRule::Fixed("global"),
        ),
        (
            "vertex/meta/muse-wonder-1.0",
            Family::MetaMuse,
            PUBLISHER_RESPONSES,
            "muse-wonder-1.0",
            "v1",
            LocationRule::Fixed("global"),
        ),
        (
            "vertex/meta/muse-ember-1.0",
            Family::MetaMuse,
            PUBLISHER_RESPONSES,
            "muse-ember-1.0",
            "v1",
            LocationRule::Fixed("global"),
        ),
        (
            "vertex/openai/gpt-6-astra",
            Family::OpenAi,
            PUBLISHER_RESPONSES,
            "gpt-6-astra",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/openai/gpt-6.1-sol",
            Family::OpenAi,
            PUBLISHER_RESPONSES,
            "gpt-6.1-sol",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/openai/gpt-6.1-terra",
            Family::OpenAi,
            PUBLISHER_RESPONSES,
            "gpt-6.1-terra",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/openai/gpt-6-luna",
            Family::OpenAi,
            PUBLISHER_RESPONSES,
            "gpt-6-luna",
            "v1",
            LocationRule::Inherit,
        ),
        (
            "vertex/mistralai/mistral-medium-3",
            Family::Mistral,
            WireFormat::MistralChat,
            "mistral-medium-3",
            "v1",
            MISTRAL_REQUIRED,
        ),
        (
            "vertex/mistralai/mistral-small-2503",
            Family::Mistral,
            WireFormat::MistralChat,
            "mistral-small-2503",
            "v1",
            MISTRAL_REQUIRED,
        ),
        (
            "vertex/mistralai/codestral-2",
            Family::Mistral,
            WireFormat::MistralChat,
            "codestral-2",
            "v1",
            MISTRAL_REQUIRED,
        ),
    ];

    #[test]
    fn partner_rows_resolve_to_the_documented_wire_upstream_version_and_location() {
        assert_eq!(PARTNER_MODELS.len(), EXPECTED_PARTNERS.len());
        for (raw, family, wire, upstream, version, location) in EXPECTED_PARTNERS {
            let resolved = resolve(raw).unwrap_or_else(|error| panic!("{raw}: {error}"));
            let (publisher, model_id) = raw
                .strip_prefix("vertex/")
                .and_then(|name| name.split_once('/'))
                .unwrap();
            assert_eq!(resolved.family, *family, "{raw}");
            assert_eq!(resolved.wire, *wire, "{raw}");
            assert_eq!(resolved.wire, family.wire(), "{raw}");
            assert_eq!(resolved.publisher, publisher, "{raw}");
            assert_eq!(resolved.model_id, model_id, "{raw}");
            assert_eq!(resolved.upstream_model, *upstream, "{raw}");
            assert_eq!(resolved.api_version.as_str(), *version, "{raw}");
            assert_eq!(resolved.location, *location, "{raw}");
            assert_eq!(
                resolved.override_key(),
                raw.strip_prefix("vertex/").unwrap()
            );
            assert!(resolved.registered, "{raw}");
            assert!(!resolved.supports_minimal, "{raw}");
        }
    }

    #[test]
    fn partner_rows_have_valid_unique_ids() {
        for (index, row) in PARTNER_MODELS.iter().enumerate() {
            assert!(is_valid_publisher(row.publisher), "{}", row.publisher);
            assert!(is_valid_model_id(row.id), "{}", row.id);
            assert!(
                !PARTNER_MODELS[..index]
                    .iter()
                    .any(|seen| seen.publisher == row.publisher && seen.id == row.id),
                "{}/{} is listed twice",
                row.publisher,
                row.id
            );
        }
    }

    #[test]
    fn unlisted_ids_synthetic_follow_the_fallback_patterns() {
        // Name, family, upstream string, location rule.
        let tests: [(&str, Family, &str, LocationRule); 8] = [
            (
                "vertex/anthropic/claude-synthetic-9",
                Family::Claude,
                "claude-synthetic-9",
                LocationRule::Inherit,
            ),
            (
                "vertex/anthropic/claude-sonnet-4-5@20250929",
                Family::Claude,
                "claude-sonnet-4-5@20250929",
                LocationRule::Inherit,
            ),
            (
                "vertex/xai/grok-synthetic-9",
                Family::Grok,
                "xai/grok-synthetic-9",
                LocationRule::Inherit,
            ),
            (
                "vertex/mistralai/mistral-ocr-2505",
                Family::Mistral,
                "mistral-ocr-2505",
                MISTRAL_REQUIRED,
            ),
            (
                "vertex/openai/gpt-synthetic-7",
                Family::OpenAi,
                "gpt-synthetic-7",
                LocationRule::Inherit,
            ),
            (
                "vertex/meta/muse-synthetic-2.0",
                Family::MetaMuse,
                "muse-synthetic-2.0",
                LocationRule::Fixed("global"),
            ),
            (
                "vertex/google/gemini-synthetic-9",
                Family::Gemini,
                "gemini-synthetic-9",
                LocationRule::Inherit,
            ),
            (
                "anthropic/claude-synthetic-9",
                Family::Claude,
                "claude-synthetic-9",
                LocationRule::Inherit,
            ),
        ];
        for (raw, family, upstream, location) in tests {
            let resolved = resolve(raw).unwrap_or_else(|error| panic!("{raw}: {error}"));
            assert_eq!(resolved.family, family, "{raw}");
            assert_eq!(resolved.wire, family.wire(), "{raw}");
            assert_eq!(resolved.upstream_model, upstream, "{raw}");
            assert_eq!(resolved.api_version, ApiVersion::V1, "{raw}");
            assert_eq!(resolved.location, location, "{raw}");
            assert!(!resolved.registered, "{raw}");
        }
    }

    #[test]
    fn open_models_and_deployed_endpoints_synthetic_are_refused_by_this_version() {
        for (raw, family) in [
            ("vertex/openai/gpt-oss-120b-maas", OPEN_MODEL_FAMILY_LABEL),
            (
                "vertex/meta/llama-4-maverick-17b-128e-instruct-maas",
                OPEN_MODEL_FAMILY_LABEL,
            ),
            ("vertex/meta/muse-synthetic-maas", OPEN_MODEL_FAMILY_LABEL),
            ("vertex/zai-org/glm-5.2-maas", OPEN_MODEL_FAMILY_LABEL),
            (
                "vertex/deepseek-ai/deepseek-v4.1-flash-maas",
                OPEN_MODEL_FAMILY_LABEL,
            ),
            (
                "vertex/endpoint/synthetic-endpoint",
                DEPLOYED_ENDPOINT_FAMILY_LABEL,
            ),
            ("vertex/endpoint/a/b", DEPLOYED_ENDPOINT_FAMILY_LABEL),
        ] {
            let error = resolve(raw).unwrap_err();
            assert_eq!(
                error,
                ModelError::FamilyNotSupported {
                    requested: raw.to_string(),
                    family,
                },
                "{raw}"
            );
            let message = error.to_string();
            assert!(
                message.starts_with("this version of claude-code-proxy does not support the "),
                "{message}"
            );
            assert!(
                message.ends_with(
                    "supported families: Gemini, Anthropic Claude, xAI Grok, OpenAI, Meta Muse, Mistral"
                ),
                "{message}"
            );
        }
        // An open-model ID that cannot go into the request is still invalid.
        assert_eq!(
            resolve("vertex/zai-org/glm 5-maas"),
            Err(ModelError::InvalidId {
                requested: "vertex/zai-org/glm 5-maas".to_string()
            })
        );
    }

    #[test]
    fn unknown_publishers_synthetic_get_a_400_naming_the_accepted_forms() {
        for raw in [
            "vertex/ai21/jamba-1.5-mini",
            "vertex/openai/o9-synthetic",
            "vertex/meta/llama-synthetic",
            "vertex/google/gemma-synthetic",
            "vertex/Synthetic/model-maas",
            "vertex//gemini-3.5-flash",
            "vertex/xai/grok/extra",
        ] {
            let error = resolve(raw).unwrap_err();
            assert_eq!(
                error,
                ModelError::Unsupported {
                    requested: raw.to_string()
                },
                "{raw}"
            );
            let message = error.to_string();
            for form in [
                "gemini-3.5-flash",
                "vertex/<gemini-model-id>",
                "vertex/google/<gemini-model-id>",
                "vertex/<publisher>/<model-id>",
                "anthropic",
                "xai",
                "mistralai",
                "openai (gpt-* models)",
                "meta (muse-* models)",
            ] {
                assert!(message.contains(form), "{form}: {message}");
            }
        }
        for raw in [
            "vertex/anthropic/claude:rawPredict",
            "vertex/xai/grok 4",
            "vertex/mistralai/",
            "vertex/meta/muse-x?y",
        ] {
            assert_eq!(
                resolve(raw),
                Err(ModelError::InvalidId {
                    requested: raw.to_string()
                }),
                "{raw}"
            );
        }
    }

    fn overrides(raw: &str) -> VertexLocationOverrides {
        let config_dir = tempfile::TempDir::new().unwrap();
        let env = std::collections::HashMap::from([(
            VERTEX_LOCATION_OVERRIDES_ENV.to_string(),
            raw.to_string(),
        )]);
        crate::config::resolve_vertex_config(&env, config_dir.path())
            .location_overrides()
            .unwrap_or_else(|error| panic!("{raw}: {error}"))
    }

    fn location_of(raw: &str, overrides: &VertexLocationOverrides, configured: &str) -> String {
        resolve(raw)
            .unwrap()
            .location(overrides, configured)
            .unwrap_or_else(|error| panic!("{raw}: {error}"))
            .to_string()
    }

    #[test]
    fn location_synthetic_takes_model_key_then_fixed_then_publisher_key_then_configured() {
        let none = VertexLocationOverrides::default();
        let publisher = overrides("anthropic=eu,google=us-central1,meta=us-east5,xai=us");
        let both = overrides(
            "anthropic/claude-opus-4-6=us-east5,anthropic=eu,\
             meta/muse-spark-1.3=us,meta=us-east5,\
             google/gemini-3.5-flash=global,google=us-central1",
        );
        // Name, overrides, expected location with `asia-northeast1`
        // configured.
        let tests: [(&str, &VertexLocationOverrides, &str); 20] = [
            ("vertex/anthropic/claude-opus-5-5", &none, "asia-northeast1"),
            ("gemini-3.5-flash", &none, "asia-northeast1"),
            ("vertex/xai/grok-4.7", &none, "asia-northeast1"),
            ("vertex/openai/gpt-6-luna", &none, "asia-northeast1"),
            ("vertex/meta/muse-spark-1.3", &none, "global"),
            ("vertex/anthropic/claude-opus-5-5", &publisher, "eu"),
            ("vertex/anthropic/claude-synthetic-9", &publisher, "eu"),
            ("vertex/xai/grok-4.7", &publisher, "us"),
            ("gemini-3.5-flash", &publisher, "us-central1"),
            ("vertex/google/gemini-3.5-flash", &publisher, "us-central1"),
            ("vertex/openai/gpt-6-luna", &publisher, "asia-northeast1"),
            // A publisher key does not move a fixed location: `meta` also
            // publishes models that live elsewhere.
            ("vertex/meta/muse-spark-1.3", &publisher, "global"),
            ("vertex/meta/muse-synthetic-2.0", &publisher, "global"),
            ("vertex/anthropic/claude-opus-4-6", &both, "us-east5"),
            ("vertex/anthropic/claude-opus-5-5", &both, "eu"),
            // A model key does move it.
            ("vertex/meta/muse-spark-1.3", &both, "us"),
            ("vertex/meta/muse-wonder-1.0", &both, "global"),
            ("gemini-3.5-flash", &both, "global"),
            ("vertex/gemini-3.5-flash", &both, "global"),
            ("gemini-3.8-flash", &both, "us-central1"),
        ];
        for (raw, overrides, expected) in tests {
            assert_eq!(
                location_of(raw, overrides, "asia-northeast1"),
                expected,
                "{raw} with {overrides:?}"
            );
        }
    }

    #[test]
    fn mistral_synthetic_takes_an_allowed_configured_location_or_an_override() {
        let none = VertexLocationOverrides::default();
        let models = [
            "vertex/mistralai/mistral-medium-3",
            "vertex/mistralai/codestral-2",
            "vertex/mistralai/mistral-ocr-2505",
        ];
        // A configured location Google serves Mistral in needs no override.
        for configured in ["us-central1", "europe-west4"] {
            for raw in models {
                assert_eq!(location_of(raw, &none, configured), configured, "{raw}");
            }
        }
        // Any other configured location, the default included, is refused
        // with a message naming every setting and the allowed locations.
        for configured in ["global", "us", "asia-northeast1"] {
            for raw in models {
                let error = resolve(raw)
                    .unwrap()
                    .location(&none, configured)
                    .unwrap_err();
                assert_eq!(
                    error,
                    LocationRequired {
                        publisher: "mistralai".to_string(),
                        configured: configured.to_string(),
                        allowed: &["us-central1", "europe-west4"],
                    },
                    "{raw} {configured}"
                );
                assert_eq!(
                    error.to_string(),
                    format!(
                        "Vertex AI models of publisher mistralai are not served in location {configured}: set CCP_VERTEX_LOCATION_OVERRIDES=mistralai=<location>, \"mistralai\": \"<location>\" in vertex.locationOverrides in config.json, or the location itself (CCP_VERTEX_LOCATION, GOOGLE_CLOUD_LOCATION or vertex.location), with us-central1 or europe-west4 as <location>"
                    )
                );
            }
        }
        // Overrides come before the configured location, allowed or not.
        let set = overrides("mistralai/codestral-2=us-central1,mistralai=europe-west4");
        for configured in ["global", "us-central1"] {
            assert_eq!(
                location_of("vertex/mistralai/codestral-2", &set, configured),
                "us-central1"
            );
            assert_eq!(
                location_of("vertex/mistralai/mistral-medium-3", &set, configured),
                "europe-west4"
            );
        }
        let model_only = overrides("mistralai/codestral-2=us-central1");
        assert_eq!(
            location_of("vertex/mistralai/codestral-2", &model_only, "global"),
            "us-central1"
        );
        assert!(
            resolve("vertex/mistralai/mistral-small-2503")
                .unwrap()
                .location(&model_only, "global")
                .is_err()
        );
        assert_eq!(
            location_of(
                "vertex/mistralai/mistral-small-2503",
                &model_only,
                "europe-west4"
            ),
            "europe-west4"
        );
    }

    /// The Claude names of the Vertex AI model list, each under
    /// `vertex/anthropic/`.
    const CLAUDE_NAMES: [&str; 14] = [
        "vertex/anthropic/claude-fable-5-2",
        "vertex/anthropic/claude-fable-5-1",
        "vertex/anthropic/claude-fable-5",
        "vertex/anthropic/claude-opus-5-5",
        "vertex/anthropic/claude-opus-5",
        "vertex/anthropic/claude-opus-4-8",
        "vertex/anthropic/claude-opus-4-7",
        "vertex/anthropic/claude-sonnet-5-5",
        "vertex/anthropic/claude-sonnet-5",
        "vertex/anthropic/claude-haiku-5-5",
        "vertex/anthropic/claude-opus-4-6",
        "vertex/anthropic/claude-sonnet-4-6",
        "vertex/anthropic/claude-opus-4-5@20251101",
        "vertex/anthropic/claude-haiku-4-5@20251001",
    ];

    #[test]
    fn advertised_models_are_exactly_the_served_family_names() {
        let mut expected: Vec<String> = EXPECTED
            .iter()
            .flat_map(|(id, _)| [id.to_string(), format!("vertex/{id}")])
            .chain(CLAUDE_NAMES.iter().map(|name| name.to_string()))
            .chain(
                [
                    "vertex/xai/grok-4.7",
                    "vertex/xai/grok-4.6",
                    "vertex/xai/grok-4.20-reasoning",
                    "vertex/xai/grok-4.20-non-reasoning",
                    "vertex/xai/grok-4.3",
                    "vertex/meta/muse-spark-1.3",
                    "vertex/meta/muse-wonder-1.0",
                    "vertex/meta/muse-ember-1.0",
                    "vertex/openai/gpt-6-astra",
                    "vertex/openai/gpt-6.1-sol",
                    "vertex/openai/gpt-6.1-terra",
                    "vertex/openai/gpt-6-luna",
                    "vertex/mistralai/mistral-medium-3",
                    "vertex/mistralai/mistral-small-2503",
                    "vertex/mistralai/codestral-2",
                ]
                .map(str::to_string),
            )
            .collect();
        expected.sort_unstable();
        assert_eq!(advertised_models(), expected);
        // Only Gemini IDs are advertised bare; a partner name always carries
        // `vertex/`, so bare claude-*, gpt-*, grok-*, muse-* and mistral-*
        // stay with the providers that own them.
        for name in advertised_models() {
            assert!(
                name.starts_with("gemini-") || name.starts_with("vertex/"),
                "{name}"
            );
        }
        assert_eq!(
            Family::ALL
                .into_iter()
                .filter(|family| family.is_served())
                .collect::<Vec<_>>(),
            vec![
                Family::Gemini,
                Family::Claude,
                Family::Grok,
                Family::OpenAi,
                Family::MetaMuse,
                Family::Mistral
            ]
        );
        let mut claude = Family::Claude.model_names();
        claude.sort_unstable();
        let mut listed = CLAUDE_NAMES.map(str::to_string).to_vec();
        listed.sort_unstable();
        assert_eq!(claude, listed);
    }

    #[test]
    fn family_model_names_list_every_registered_name_of_the_family() {
        assert_eq!(
            Family::Grok.model_names(),
            vec![
                "vertex/xai/grok-4.7",
                "vertex/xai/grok-4.6",
                "vertex/xai/grok-4.20-reasoning",
                "vertex/xai/grok-4.20-non-reasoning",
                "vertex/xai/grok-4.3",
            ]
        );
        assert_eq!(
            Family::MetaMuse.model_names(),
            vec![
                "vertex/meta/muse-spark-1.3",
                "vertex/meta/muse-wonder-1.0",
                "vertex/meta/muse-ember-1.0",
            ]
        );
        assert_eq!(
            Family::OpenAi.model_names(),
            vec![
                "vertex/openai/gpt-6-astra",
                "vertex/openai/gpt-6.1-sol",
                "vertex/openai/gpt-6.1-terra",
                "vertex/openai/gpt-6-luna",
            ]
        );
        assert_eq!(
            Family::Mistral.model_names(),
            vec![
                "vertex/mistralai/mistral-medium-3",
                "vertex/mistralai/mistral-small-2503",
                "vertex/mistralai/codestral-2",
            ]
        );
        assert_eq!(Family::Claude.model_names().len(), 14);
        assert_eq!(Family::Gemini.model_names().len(), 18);
        let every: usize = Family::ALL
            .into_iter()
            .map(|family| family.model_names().len())
            .sum();
        assert_eq!(every, 18 + EXPECTED_PARTNERS.len());
        for (raw, family, ..) in EXPECTED_PARTNERS {
            assert!(family.model_names().contains(&raw.to_string()), "{raw}");
        }
    }
}
