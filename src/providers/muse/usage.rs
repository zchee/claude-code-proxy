/// One metered window of the subscription allowance.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct UsageWindow {
    pub used_percent: i64,
    pub resets_at: i64, // Unix seconds
    #[serde(default)]
    pub window_duration_mins: Option<i64>,
}

/// The allowance as reported by the key mint (`subs_usage`) and by the
/// `response.subscription_usage` stream event (`subscription`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct SubscriptionUsage {
    #[serde(default)]
    pub window: Option<UsageWindow>,
    #[serde(default)]
    pub weekly: Option<UsageWindow>,
}

/// Identifies the login a request ran under: SHA-256 of the access token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoginId(pub [u8; 32]);

/// Receives allowance updates observed on a response stream.
pub trait UsageSink: Send + Sync {
    fn record_usage(&self, login: LoginId, usage: SubscriptionUsage);
}
