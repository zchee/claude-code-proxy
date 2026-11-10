/// The Muse models this proxy serves. The `-contributor` variants and the
/// hidden image model in the upstream catalog are deliberately absent.
pub const MUSE_MODELS: &[&str] = &[
    "muse-spark-1.3",
    "muse-spark-1.2",
    "muse-wonder-1.0",
    "muse-wonder-full-1.0",
    "muse-ember-1.0",
];

/// The catalog lists no `max` effort for this model; `xhigh` is its highest.
const MODEL_WITHOUT_MAX_EFFORT: &str = "muse-spark-1.2";

pub fn assert_allowed_model(model: &str) -> anyhow::Result<()> {
    if MUSE_MODELS.contains(&model) {
        Ok(())
    } else {
        anyhow::bail!("unsupported Muse model: {model}")
    }
}

/// Maps a client effort (`low|medium|high|xhigh|max`, as validated by
/// `translate_shared::read_effort`) to the value the model accepts. The
/// upstream rejects `none` and `ultra`, and `minimal` is never derived from a
/// client effort, so neither can come out of this function.
pub fn map_effort(model: &str, effort: &str) -> &'static str {
    match effort {
        "low" => "low",
        "medium" => "medium",
        "xhigh" => "xhigh",
        "max" if model == MODEL_WITHOUT_MAX_EFFORT => "xhigh",
        "max" => "max",
        _ => "high",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_public_models_are_allowed() {
        for model in MUSE_MODELS {
            assert!(assert_allowed_model(model).is_ok(), "{model}");
        }
        assert_eq!(MUSE_MODELS.len(), 5);
    }

    #[test]
    fn contributor_hidden_and_unknown_models_are_rejected() {
        for model in [
            "muse-spark-1.3-contributor",
            "muse-wonder-full-1.0-contributor",
            "muse-image-1.0",
            "muse-spark",
            "MUSE-SPARK-1.3",
            "",
        ] {
            assert!(assert_allowed_model(model).is_err(), "{model}");
        }
    }

    #[test]
    fn effort_maps_per_model() {
        for model in MUSE_MODELS {
            let max = if *model == "muse-spark-1.2" {
                "xhigh"
            } else {
                "max"
            };
            let cases = [
                ("low", "low"),
                ("medium", "medium"),
                ("high", "high"),
                ("xhigh", "xhigh"),
                ("max", max),
            ];
            for (effort, expected) in cases {
                assert_eq!(map_effort(model, effort), expected, "{model} {effort}");
            }
        }
    }
}
