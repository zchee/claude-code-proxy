//! Error responses of the Google AI Ultra plan Cloud Code endpoints
//! (`v1internal:streamGenerateContent`, `v1internal:loadCodeAssist` and
//! `v1internal:onboardUser`), turned into Anthropic errors.
//!
//! The message is Google's own text, or a status-only fallback text when the
//! body carries none (the client passes an empty body when it was too long,
//! broke off or came too late), plus a hint where that text does not say
//! what to do (sign in again, upgrade, switch model). An error inside a
//! started stream is worded by the stream translator instead. Only 429s need
//! a retry decision: a per-minute limit is retried, while a daily limit or a
//! model the plan does not include is final. Claude Code retries a 429 on
//! its own, so a final one is answered with `x-should-retry: false`.
//!
//! Bodies come in three shapes: a JSON object, a JSON array of
//! `google.rpc.Status` objects (401 and gateway 403s), and an SSE
//! `event: error` frame (some errors of streaming requests).

use std::sync::LazyLock;

use axum::http::StatusCode;
use regex_lite::Regex;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::Value;

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::providers::vertex::sanitize::sanitize_vertex_text;

const PLANS_URL: &str = "https://one.google.com/ai";

const SIGN_IN_AGAIN: &str = "Run `claude-code-proxy google-ai-plan auth login` to sign in again.";

const CLIENT_NOT_ALLOWED: &str = "The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.";

/// "You've reached today's AI Ultra limit for …".
static DAILY_MESSAGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)reached today's .*? limit").expect("daily limit pattern is a valid regex")
});

/// "Rate limit exceeded for model … (limit: 0 tokens per day)".
static PER_DAY_MESSAGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)rate limit exceeded(?: for model \S+)? \(limit: (\d+) (?:tokens|requests) per day\)",
    )
    .expect("per-day limit pattern is a valid regex")
});

static NOT_FOR_PER_USER_QUOTA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)not supported for per-user quota")
        .expect("per-user quota pattern is a valid regex")
});

static INSUFFICIENT_SCOPES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)insufficient authentication scopes").expect("scope pattern is a valid regex")
});

static CLIENT_NOT_AUTHORIZED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)oauth client is not authorized").expect("client pattern is a valid regex")
});

/// A non-2xx response, decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedError {
    /// The status to answer the client with.
    pub status: StatusCode,
    /// The Anthropic error type.
    pub kind: ProviderErrorKind,
    /// `<upstream status> <Google's message> <hint>`. Google's message has
    /// already been through [`sanitize_vertex_text`] (access tokens, long
    /// base64 runs and e-mail addresses removed, cut to 500 characters);
    /// the caller may redact the request's own secrets on top.
    pub message: String,
    /// Google's `Retry-After` in seconds, only for a 429 the client may
    /// retry.
    pub retry_after: Option<String>,
    /// `false` when the client must not retry: answer with
    /// `x-should-retry: false`.
    pub should_retry: bool,
}

impl ClassifiedError {
    /// The error in the form the server answers with. The caller adds the
    /// `x-should-retry: false` header when [`Self::should_retry`] is false.
    pub fn into_provider_error(self) -> ProviderError {
        let mut error = ProviderError::new(self.status, self.kind, self.message);
        error.retry_after = self.retry_after;
        error
    }
}

/// What the body says, as far as the classification uses it.
#[derive(Debug, Default, PartialEq)]
struct ErrorBody {
    message: String,
    /// `ErrorInfo.reason`, such as `ACCESS_TOKEN_SCOPE_INSUFFICIENT`.
    reason: Option<String>,
    /// The `QuotaFailure` violation of a per-day quota: `Some(None)` when it
    /// has no readable limit.
    per_day_quota: Option<Option<f64>>,
    /// `retryDelay` of a `google.rpc.RetryInfo` detail, in whole seconds,
    /// used as the retry hint when the `Retry-After` header is absent.
    retry_delay_secs: Option<u64>,
}

/// Which kind of 429 a response is.
enum RateLimit {
    PerMinute,
    NotOnPlan,
    Daily { reset_in_message: bool },
}

/// Classifies a non-2xx answer for `model_id` (the upstream ID, without the
/// provider prefix).
pub fn classify_error(
    status: StatusCode,
    headers: &HeaderMap,
    body: &str,
    model_id: &str,
) -> ClassifiedError {
    classify(status, headers, body, model_id, false)
}

/// Classifies a non-2xx answer from the project-discovery endpoints
/// (`loadCodeAssist` / `onboardUser`). A generic permission failure there
/// says nothing about the account's plan, so it keeps the upstream message
/// without the "upgrade your plan" advice that a generation 403 carries; the
/// specific scope and client reasons still apply.
pub fn classify_discovery_error(
    status: StatusCode,
    headers: &HeaderMap,
    body: &str,
    model_id: &str,
) -> ClassifiedError {
    classify(status, headers, body, model_id, true)
}

fn classify(
    status: StatusCode,
    headers: &HeaderMap,
    body: &str,
    model_id: &str,
    discovery: bool,
) -> ClassifiedError {
    let error = parse_error_body(body);
    let upstream = status.as_u16();
    let message = |hint: Option<String>| {
        let text = sanitize_vertex_text(&error.message, &[]);
        let text = if text.trim().is_empty() {
            format!("Google AI Ultra plan returned HTTP {upstream}")
        } else {
            text
        };
        match hint {
            Some(hint) => format!("{upstream} {text} {hint}"),
            None => format!("{upstream} {text}"),
        }
    };

    if upstream == 429 {
        let (hint, retry_after, should_retry) = match classify_rate_limit(&error) {
            RateLimit::PerMinute => (
                None,
                retry_after_secs(headers)
                    .or_else(|| error.retry_delay_secs.map(|secs| secs.to_string())),
                true,
            ),
            RateLimit::NotOnPlan => (Some(not_on_plan(model_id)), None, false),
            RateLimit::Daily { reset_in_message } => {
                let reset = if reset_in_message {
                    String::new()
                } else {
                    format_reset(headers, error.retry_delay_secs)
                };
                let hint = format!(
                    "Daily limit reached for {model_id}{reset}: switch to another model or provider."
                );
                (Some(hint), None, false)
            }
        };
        return ClassifiedError {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: ProviderErrorKind::RateLimit,
            message: message(hint),
            retry_after,
            should_retry,
        };
    }

    let (kind, answer) = match upstream {
        401 => (ProviderErrorKind::Authentication, StatusCode::UNAUTHORIZED),
        402 | 403 => (ProviderErrorKind::Permission, StatusCode::FORBIDDEN),
        400..=499 => (ProviderErrorKind::InvalidRequest, status),
        _ => (ProviderErrorKind::Api, StatusCode::BAD_GATEWAY),
    };
    ClassifiedError {
        status: answer,
        kind,
        message: message(hint_for(upstream, &error, model_id, discovery)),
        retry_after: None,
        should_retry: true,
    }
}

fn classify_rate_limit(error: &ErrorBody) -> RateLimit {
    let daily_message = DAILY_MESSAGE.is_match(&error.message);
    let per_day_message = PER_DAY_MESSAGE
        .captures(&error.message)
        .and_then(|captures| captures.get(1))
        .map(|limit| limit.as_str().to_string());
    if !daily_message && per_day_message.is_none() && error.per_day_quota.is_none() {
        return RateLimit::PerMinute;
    }
    // A daily limit of zero means the plan does not include the model.
    if error.per_day_quota == Some(Some(0.0)) || per_day_message.as_deref() == Some("0") {
        return RateLimit::NotOnPlan;
    }
    RateLimit::Daily {
        reset_in_message: daily_message,
    }
}

fn hint_for(status: u16, error: &ErrorBody, model_id: &str, discovery: bool) -> Option<String> {
    if status == 400 && NOT_FOR_PER_USER_QUOTA.is_match(&error.message) {
        return Some(not_on_plan(model_id));
    }
    if status == 401 {
        return Some(SIGN_IN_AGAIN.to_string());
    }
    let reason = error.reason.as_deref();
    // Flat error bodies carry no `ErrorInfo`, so the message is matched too.
    if status == 403
        && (reason == Some("ACCESS_TOKEN_SCOPE_INSUFFICIENT")
            || INSUFFICIENT_SCOPES.is_match(&error.message))
    {
        return Some(SIGN_IN_AGAIN.to_string());
    }
    // The client's API is disabled or Google blocked the client; the user
    // cannot fix that with their own account.
    if status == 403
        && (reason == Some("SERVICE_DISABLED") || CLIENT_NOT_AUTHORIZED.is_match(&error.message))
    {
        return Some(CLIENT_NOT_ALLOWED.to_string());
    }
    // A discovery 403 reports a plain permission failure; only a generation
    // 402/403 tells the user their plan is the problem.
    if !discovery && (status == 402 || status == 403) {
        return Some(format!(
            "Your Google account does not have an eligible Google AI Ultra plan. Upgrade at {PLANS_URL}, or switch to another model or provider."
        ));
    }
    None
}

fn not_on_plan(model_id: &str) -> String {
    format!("{model_id} is not available on the Google AI Ultra plan. Choose another model.")
}

/// `Retry-After` when it is a whole number of seconds.
fn retry_after_secs(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    value.parse::<u64>().ok().map(|secs| secs.to_string())
}

/// `" (resets in 5h 18m)"` from `Retry-After` seconds, falling back to the
/// `google.rpc.RetryInfo.retryDelay` seconds, or `""` when neither is present.
fn format_reset(headers: &HeaderMap, retry_delay_secs: Option<u64>) -> String {
    let seconds = headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .or_else(|| {
            retry_delay_secs
                .filter(|secs| *secs > 0)
                .map(|secs| secs as f64)
        });
    let Some(seconds) = seconds else {
        return String::new();
    };
    // Bounded by the filter above and by f64's range; the cast saturates.
    let total_minutes = (seconds / 60.0).ceil() as u64;
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    if hours > 0 {
        format!(" (resets in {hours}h {minutes}m)")
    } else {
        format!(" (resets in {minutes}m)")
    }
}

/// Reads message, `ErrorInfo.reason` and the per-day quota from any of the
/// three body shapes; falls back to the raw text.
fn parse_error_body(body: &str) -> ErrorBody {
    let trimmed = body.trim();
    let fallback = || ErrorBody {
        message: trimmed.to_string(),
        ..ErrorBody::default()
    };
    let json = if trimmed.starts_with(['[', '{']) {
        Some(trimmed)
    } else {
        trimmed
            .lines()
            .find_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
    };
    let Some(parsed) = json.and_then(|json| serde_json::from_str::<Value>(json).ok()) else {
        return fallback();
    };
    let first = match &parsed {
        Value::Array(items) => items.first(),
        other => Some(other),
    };
    let Some(error) = first.and_then(|first| first.get("error")) else {
        return fallback();
    };
    let Some(message) = error.get("message").and_then(Value::as_str) else {
        return fallback();
    };
    let details = error
        .get("details")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let detail_of = |suffix: &str| {
        details.iter().find(|detail| {
            detail
                .get("@type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.ends_with(suffix))
        })
    };
    let reason = detail_of("google.rpc.ErrorInfo")
        .and_then(|detail| detail.get("reason"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let per_day = detail_of("google.rpc.QuotaFailure")
        .and_then(|detail| detail.get("violations"))
        .and_then(Value::as_array)
        .and_then(|violations| {
            violations.iter().find(|violation| {
                violation
                    .get("quotaId")
                    .and_then(Value::as_str)
                    .is_some_and(|id| id.contains("PerDay"))
            })
        });
    // `quotaValue` is an int64, which JSON carries as a string.
    let per_day_quota = per_day.map(|violation| {
        match violation.get("quotaValue") {
            Some(Value::Number(value)) => value.as_f64(),
            Some(Value::String(value)) => value.trim().parse::<f64>().ok(),
            _ => None,
        }
        .filter(|limit| limit.is_finite())
    });
    // `google.rpc.RetryInfo.retryDelay` is a protobuf Duration string such as
    // `"12s"` or `"2.540s"`; the shared parser rounds it up to whole seconds.
    let retry_delay_secs = detail_of("google.rpc.RetryInfo")
        .and_then(|detail| detail.get("retryDelay"))
        .and_then(Value::as_str)
        .and_then(crate::providers::vertex::client::retry_delay_secs);
    ErrorBody {
        message: message.to_string(),
        reason,
        per_day_quota,
        retry_delay_secs,
    }
}

#[cfg(test)]
mod tests {
    use reqwest::header::HeaderValue;
    use serde_json::json;

    use super::*;

    const MODEL: &str = "gemini-4-hydrogen";

    fn daily_limit_body() -> String {
        json!({"error":{
            "code":"too_many_requests",
            "message":"You've reached today's AI Ultra limit for gemini-4-hydrogen in third-party apps. Your daily limit refreshes in 5 hours, 18 minutes. Until then, you can switch to other models."
        }})
        .to_string()
    }

    fn per_day_quota_body(quota_value: Value) -> String {
        json!({"error":{
            "message":"Resource exhausted.",
            "details":[{
                "@type":"type.googleapis.com/google.rpc.QuotaFailure",
                "violations":[{"quotaId":"GenerateRequestsPerDayPerUserPerModel","quotaValue":quota_value}]
            }]
        }})
        .to_string()
    }

    fn zero_limit_body() -> String {
        json!({"error":{"message":"Rate limit exceeded for model gemini-4-argon (limit: 0 tokens per day)"}})
            .to_string()
    }

    fn headers(retry_after: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(value) = retry_after {
            headers.insert(
                RETRY_AFTER,
                HeaderValue::from_str(value).expect("valid header"),
            );
        }
        headers
    }

    fn classify(status: u16, body: &str, retry_after: Option<&str>) -> ClassifiedError {
        classify_error(
            StatusCode::from_u16(status).expect("valid status"),
            &headers(retry_after),
            body,
            MODEL,
        )
    }

    #[test]
    fn a_discovery_403_keeps_the_permission_error_without_plan_upgrade_advice() {
        let body = json!({"error":{
            "code":403,
            "message":"Permission denied on resource project."
        }})
        .to_string();
        let discovery =
            classify_discovery_error(StatusCode::FORBIDDEN, &headers(None), &body, MODEL);
        assert_eq!(
            discovery.message,
            "403 Permission denied on resource project."
        );
        assert_eq!(discovery.kind, ProviderErrorKind::Permission);
        // The same body on the generation path still points at the plan.
        assert!(classify(403, &body, None).message.contains(PLANS_URL));
    }

    #[test]
    fn daily_limit_is_final_and_keeps_googles_text_which_already_says_when_it_resets() {
        assert_eq!(
            classify(429, &daily_limit_body(), Some("19104")),
            ClassifiedError {
                status: StatusCode::TOO_MANY_REQUESTS,
                kind: ProviderErrorKind::RateLimit,
                message: "429 You've reached today's AI Ultra limit for gemini-4-hydrogen in third-party apps. Your daily limit refreshes in 5 hours, 18 minutes. Until then, you can switch to other models. Daily limit reached for gemini-4-hydrogen: switch to another model or provider.".to_string(),
                retry_after: None,
                should_retry: false,
            }
        );
    }

    #[test]
    fn per_day_quota_violation_gets_the_reset_time_from_retry_after() {
        let tests = [
            (json!("1000000"), Some("19104"), " (resets in 5h 19m)"),
            (json!("1000000"), Some("7199"), " (resets in 2h 0m)"),
            (json!(1000000), Some("90"), " (resets in 2m)"),
            (json!("1000000"), None, ""),
            (json!("1000000"), Some("soon"), ""),
            (json!(null), Some("60"), " (resets in 1m)"),
        ];
        for (quota, retry_after, reset) in tests {
            let error = classify(429, &per_day_quota_body(quota.clone()), retry_after);
            assert_eq!(
                error.message,
                format!(
                    "429 Resource exhausted. Daily limit reached for gemini-4-hydrogen{reset}: switch to another model or provider."
                ),
                "quota {quota} retry-after {retry_after:?}"
            );
            assert!(!error.should_retry, "quota {quota}");
            assert_eq!(error.retry_after, None, "a final 429 sends no Retry-After");
        }
    }

    #[test]
    fn a_daily_limit_of_zero_is_a_model_the_plan_does_not_include() {
        let hint =
            "gemini-4-hydrogen is not available on the Google AI Ultra plan. Choose another model.";
        for quota in [json!("0"), json!(0)] {
            let error = classify(429, &per_day_quota_body(quota.clone()), None);
            assert_eq!(
                error.message,
                format!("429 Resource exhausted. {hint}"),
                "{quota}"
            );
            assert!(!error.should_retry, "{quota}");
        }
        let error = classify(429, &zero_limit_body(), Some("30"));
        assert_eq!(
            error.message,
            format!(
                "429 Rate limit exceeded for model gemini-4-argon (limit: 0 tokens per day) {hint}"
            )
        );
        assert!(!error.should_retry);
        assert_eq!(error.retry_after, None);
    }

    #[test]
    fn a_per_minute_429_stays_retryable_with_its_retry_after() {
        let body = json!({"error":{"message":"Too many requests."}}).to_string();
        assert_eq!(
            classify(429, &body, Some("7")),
            ClassifiedError {
                status: StatusCode::TOO_MANY_REQUESTS,
                kind: ProviderErrorKind::RateLimit,
                message: "429 Too many requests.".to_string(),
                retry_after: Some("7".to_string()),
                should_retry: true,
            }
        );
        assert_eq!(
            classify(429, &body, Some("Wed, 21 Oct 2026 07:28:00 GMT")).retry_after,
            None
        );
        assert_eq!(classify(429, &body, None).retry_after, None);
    }

    #[test]
    fn a_per_minute_429_falls_back_to_the_retry_info_delay_without_a_header() {
        let retry_info = |delay: &str| {
            json!({"error":{
                "message":"Too many requests.",
                "details":[{
                    "@type":"type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay":delay
                }]
            }})
            .to_string()
        };
        assert_eq!(
            classify(429, &retry_info("12s"), None).retry_after,
            Some("12".to_string())
        );
        // A fractional delay rounds up to whole seconds.
        assert_eq!(
            classify(429, &retry_info("2.540s"), None).retry_after,
            Some("3".to_string())
        );
        // A present Retry-After header is used ahead of the RetryInfo delay.
        assert_eq!(
            classify(429, &retry_info("12s"), Some("7")).retry_after,
            Some("7".to_string())
        );
    }

    #[test]
    fn models_outside_the_plan_are_reported() {
        let body = json!({"error":{"message":"Model gemini-4-argon is not supported for per-user quota requests.","code":"invalid_request"}}).to_string();
        let error = classify_error(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            &body,
            "gemini-4-argon",
        );
        assert_eq!(
            error.message,
            "400 Model gemini-4-argon is not supported for per-user quota requests. gemini-4-argon is not available on the Google AI Ultra plan. Choose another model."
        );
        assert_eq!(
            (error.status, error.kind),
            (StatusCode::BAD_REQUEST, ProviderErrorKind::InvalidRequest)
        );
    }

    #[test]
    fn a_401_sent_as_a_json_array_asks_to_sign_in_again() {
        let body = json!([{"error":{"code":401,"message":"Request had invalid authentication credentials.","status":"UNAUTHENTICATED"}}]).to_string();
        let error = classify(401, &body, None);
        assert_eq!(
            error.message,
            "401 Request had invalid authentication credentials. Run `claude-code-proxy google-ai-plan auth login` to sign in again."
        );
        assert_eq!(
            (error.status, error.kind),
            (StatusCode::UNAUTHORIZED, ProviderErrorKind::Authentication)
        );
    }

    #[test]
    fn scope_disabled_service_and_plan_errors_are_told_apart() {
        let error_info = |reason: &str| {
            json!([{"error":{"code":403,"message":"Permission denied.","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":reason}]}}]).to_string()
        };
        let tests = [
            (
                403,
                error_info("ACCESS_TOKEN_SCOPE_INSUFFICIENT"),
                "403 Permission denied. Run `claude-code-proxy google-ai-plan auth login` to sign in again.",
            ),
            (
                403,
                json!({"error":{"message":"Request had insufficient authentication scopes."}})
                    .to_string(),
                "403 Request had insufficient authentication scopes. Run `claude-code-proxy google-ai-plan auth login` to sign in again.",
            ),
            (
                403,
                error_info("SERVICE_DISABLED"),
                "403 Permission denied. The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.",
            ),
            (
                403,
                json!({"error":{"message":"The OAuth client is not authorized to use this API."}})
                    .to_string(),
                "403 The OAuth client is not authorized to use this API. The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.",
            ),
            (
                403,
                json!({"error":{"message":"Permission denied."}}).to_string(),
                "403 Permission denied. Your Google account does not have an eligible Google AI Ultra plan. Upgrade at https://one.google.com/ai, or switch to another model or provider.",
            ),
            (
                402,
                json!({"error":{"message":"Payment required."}}).to_string(),
                "402 Payment required. Your Google account does not have an eligible Google AI Ultra plan. Upgrade at https://one.google.com/ai, or switch to another model or provider.",
            ),
        ];
        for (status, body, expected) in tests {
            let error = classify(status, &body, None);
            assert_eq!(error.message, expected, "{body}");
            assert_eq!(
                (error.status, error.kind),
                (StatusCode::FORBIDDEN, ProviderErrorKind::Permission),
                "{body}"
            );
            assert!(error.should_retry, "{body}");
        }
    }

    #[test]
    fn the_message_is_read_from_an_sse_error_frame() {
        let body = "event: error\ndata: {\"error\":{\"message\":\"Function call is missing a thought_signature.\",\"code\":\"invalid_request\"},\"event_type\":\"error\"}\n\n";
        assert_eq!(
            classify(400, body, None),
            ClassifiedError {
                status: StatusCode::BAD_REQUEST,
                kind: ProviderErrorKind::InvalidRequest,
                message: "400 Function call is missing a thought_signature.".to_string(),
                retry_after: None,
                should_retry: true,
            }
        );
    }

    #[test]
    fn other_statuses_and_unreadable_bodies() {
        let tests = [
            (
                404,
                "{\"error\":{\"message\":\"Not found.\"}}",
                StatusCode::NOT_FOUND,
                ProviderErrorKind::InvalidRequest,
                "404 Not found.",
            ),
            (
                413,
                "too big",
                StatusCode::PAYLOAD_TOO_LARGE,
                ProviderErrorKind::InvalidRequest,
                "413 too big",
            ),
            (
                500,
                "<html>oops</html>",
                StatusCode::BAD_GATEWAY,
                ProviderErrorKind::Api,
                "500 <html>oops</html>",
            ),
            (
                503,
                "",
                StatusCode::BAD_GATEWAY,
                ProviderErrorKind::Api,
                "503 Google AI Ultra plan returned HTTP 503",
            ),
            (
                500,
                "{\"error\":{\"code\":500}}",
                StatusCode::BAD_GATEWAY,
                ProviderErrorKind::Api,
                "500 {\"error\":{\"code\":500}}",
            ),
        ];
        for (status, body, answer, kind, message) in tests {
            let error = classify(status, body, None);
            assert_eq!(
                (error.status, error.kind, error.message.as_str()),
                (answer, kind, message),
                "{status} {body}"
            );
        }
    }

    #[test]
    fn googles_text_is_sanitised_before_the_hint() {
        let token = format!("ya29.{}", "a1B2c3".repeat(20));
        let body = json!({"error":{"message":format!("Bad token {token} sent")}}).to_string();
        let error = classify(401, &body, None);
        assert!(!error.message.contains(&token), "{}", error.message);
        assert!(error.message.ends_with(SIGN_IN_AGAIN), "{}", error.message);
    }

    #[test]
    fn into_provider_error_keeps_status_type_and_retry_after() {
        let body = json!({"error":{"message":"Too many requests."}}).to_string();
        let error = classify(429, &body, Some("3")).into_provider_error();
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.error_type(), "rate_limit_error");
        assert_eq!(error.retry_after.as_deref(), Some("3"));
        assert_eq!(error.message, "429 Too many requests.");
    }
}
