//! `claude-code-proxy muse auth ...`. Runs in its own process with blocking
//! HTTP and shares only the stored file and the mint types with the server.

use std::time::{SystemTime, UNIX_EPOCH};

use jiff::Timestamp;
use jiff::tz::TimeZone;

use super::constants::LOGIN_COMMAND;
use super::device;
use super::mint::{self, MintError, MintTarget};
use super::token_store::{MuseTokenStore, StoredAuth, file_store};
use crate::auth::AuthStorage;
use crate::provider::CliHandlers;
use crate::providers::muse::usage::UsageWindow;

pub struct MuseCli;
pub static MUSE_CLI: MuseCli = MuseCli;

impl CliHandlers for MuseCli {
    fn login(&self) -> anyhow::Result<()> {
        // Meta offers only the device-code flow for this client.
        self.device()
    }

    fn device(&self) -> anyhow::Result<()> {
        let store = file_store();
        let auth = device::device_login(&store)?;
        println!("{}", login_summary(&auth, &store.auth_path()));
        Ok(())
    }

    fn status(&self) -> anyhow::Result<()> {
        let target = MintTarget::from_config()?;
        let report = status_report(
            &file_store(),
            &mint::blocking_client()?,
            &target,
            now_ms(),
            &TimeZone::system(),
        )?;
        println!("{report}");
        Ok(())
    }

    fn logout(&self) -> anyhow::Result<()> {
        println!("{}", logout_with(&file_store())?);
        Ok(())
    }
}

fn login_summary(auth: &StoredAuth, path: &str) -> String {
    let mut lines = vec![format!("Muse authentication saved in {path}")];
    if let Some(name) = &auth.user_full_name {
        lines.push(format!("User: {name}"));
    }
    lines.push(format!("Subscription active: {}", auth.is_subs_active));
    if let Some(tier) = &auth.subs_tier_name {
        lines.push(format!("Tier: {tier}"));
    }
    lines.join("\n")
}

/// Mints once with the stored login (the server's in-memory state is not
/// visible from here) and describes the account and its allowance.
pub fn status_report<S: AuthStorage<StoredAuth>>(
    store: &MuseTokenStore<S>,
    client: &reqwest::blocking::Client,
    target: &MintTarget,
    now_ms: u64,
    time_zone: &TimeZone,
) -> anyhow::Result<String> {
    let path = store.auth_path();
    let Some(auth) = store.load_auth()? else {
        anyhow::bail!("Not authenticated with Muse. Run: {LOGIN_COMMAND}");
    };
    let minted = mint::mint_blocking(client, target, &auth.access_token).map_err(|error| {
        match error {
            MintError::Unauthorized => anyhow::anyhow!(
                "Muse login stored in {path} is no longer valid (upstream HTTP 401). Run: {LOGIN_COMMAND}"
            ),
            other => anyhow::anyhow!("{}", other.message_with_status()),
        }
    })?;

    let mut lines = vec![format!("Auth path: {path}"), "Authenticated: true".into()];
    if let Some(name) = minted
        .user_full_name
        .as_ref()
        .or(auth.user_full_name.as_ref())
    {
        lines.push(format!("User: {name}"));
    }
    lines.push(format!("Subscription active: {}", minted.is_subs_active));
    if let Some(tier) = &minted.subs_tier_name {
        lines.push(format!("Tier: {tier}"));
    }
    if let Some(usage) = &minted.subs_usage {
        if let Some(window) = &usage.window {
            lines.push(usage_line(&window_label(window), window, time_zone));
        }
        if let Some(weekly) = &usage.weekly {
            lines.push(usage_line("Weekly", weekly, time_zone));
        }
    }
    lines.push(format!(
        "Logged in: {} ago",
        human_duration(now_ms.saturating_sub(auth.obtained_at_ms) / 1000)
    ));
    Ok(lines.join("\n"))
}

/// Removes only the proxy's own file. The upstream logout is never called:
/// the API key is shared with the Muse CLI, and revoking it would sign that
/// out as well.
pub fn logout_with<S: AuthStorage<StoredAuth>>(
    store: &MuseTokenStore<S>,
) -> anyhow::Result<String> {
    store.clear_auth()?;
    Ok(format!(
        "Muse proxy credentials removed from {}. The Muse CLI's own login and the API key it shares with the proxy were not touched.",
        store.auth_path()
    ))
}

fn window_label(window: &UsageWindow) -> String {
    match window.window_duration_mins {
        Some(minutes) if minutes > 0 && minutes % 60 == 0 => {
            format!("{}-hour window", minutes / 60)
        }
        Some(minutes) if minutes > 0 => format!("{minutes}-minute window"),
        _ => "Window".into(),
    }
}

fn usage_line(label: &str, window: &UsageWindow, time_zone: &TimeZone) -> String {
    let resets = Timestamp::from_second(window.resets_at)
        .map(|timestamp| {
            timestamp
                .to_zoned(time_zone.clone())
                .strftime("%Y-%m-%d %H:%M:%S %Z")
                .to_string()
        })
        .unwrap_or_else(|_| window.resets_at.to_string());
    format!("{label}: {}% used, resets {resets}", window.used_percent)
}

fn human_duration(total_secs: u64) -> String {
    let days = total_secs / 86_400;
    let hours = total_secs % 86_400 / 3_600;
    let minutes = total_secs % 3_600 / 60;
    match (days, hours) {
        (0, 0) => format!("{minutes}m"),
        (0, _) => format!("{hours}h {minutes}m"),
        _ => format!("{days}d {hours}h"),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::muse::auth::mint::tests::{observed_mint_body, spawn_mint, target};
    use crate::providers::muse::auth::token_store::store_at;
    use http::StatusCode;

    const ACCESS_TOKEN: &str = "synthetic-oauth-access-token-274";
    const OBTAINED_AT_MS: u64 = 1_794_274_189_000;

    fn stored() -> StoredAuth {
        StoredAuth {
            access_token: ACCESS_TOKEN.into(),
            obtained_at_ms: OBTAINED_AT_MS,
            client_id: "1031625952748946".into(),
            oauth_host: "https://auth.meta.com".into(),
            user_full_name: Some("Stored Name".into()),
            is_subs_active: true,
            subs_tier_name: Some("Muse Code Max Usage".into()),
        }
    }

    fn blocking_client() -> reqwest::blocking::Client {
        mint::blocking_client_builder(super::super::constants::MINT_TIMEOUT)
            .no_proxy()
            .build()
            .unwrap()
    }

    fn with_mint(
        status: StatusCode,
        body: String,
    ) -> (tokio::runtime::Runtime, String, mint::tests::Log) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (base, log) = runtime.block_on(spawn_mint(status, vec![], body));
        (runtime, base, log)
    }

    #[test]
    fn status_reports_account_tier_allowance_and_login_age() {
        let (_runtime, base, log) = with_mint(StatusCode::OK, observed_mint_body().to_string());
        let temp = tempfile::TempDir::new().unwrap();
        let store = store_at(temp.path().join("auth.json").to_string_lossy().into_owned());
        store.save_auth(stored()).unwrap();
        let report = status_report(
            &store,
            &blocking_client(),
            &target(&base),
            OBTAINED_AT_MS + 3_725_000,
            &TimeZone::UTC,
        )
        .unwrap();
        assert_eq!(
            report,
            format!(
                "Auth path: {}\nAuthenticated: true\nUser: Synthetic Person\nSubscription active: true\nTier: Muse Code Max Usage\n5-hour window: 1% used, resets 2026-11-10 06:15:26 UTC\nWeekly: 0% used, resets 2026-11-16 00:00:00 UTC\nLogged in: 1h 2m ago",
                store.auth_path()
            )
        );
        assert!(!report.contains("example.com"));
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn status_with_a_refused_login_asks_for_a_new_login() {
        let (_runtime, base, _) = with_mint(
            StatusCode::UNAUTHORIZED,
            r#"{"title":"Authentication Error","detail":"Authentication Error","status":401}"#
                .into(),
        );
        let temp = tempfile::TempDir::new().unwrap();
        let store = store_at(temp.path().join("auth.json").to_string_lossy().into_owned());
        store.save_auth(stored()).unwrap();
        let error = status_report(
            &store,
            &blocking_client(),
            &target(&base),
            OBTAINED_AT_MS,
            &TimeZone::UTC,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no longer valid"), "{error}");
        assert!(
            error.contains("claude-code-proxy muse auth login"),
            "{error}"
        );
        assert!(!error.contains(ACCESS_TOKEN));
    }

    #[test]
    fn status_without_a_login_does_not_call_upstream() {
        let (_runtime, base, log) = with_mint(StatusCode::OK, observed_mint_body().to_string());
        let temp = tempfile::TempDir::new().unwrap();
        let store = store_at(temp.path().join("auth.json").to_string_lossy().into_owned());
        let error = status_report(
            &store,
            &blocking_client(),
            &target(&base),
            OBTAINED_AT_MS,
            &TimeZone::UTC,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            error,
            "Not authenticated with Muse. Run: claude-code-proxy muse auth login"
        );
        assert!(log.lock().unwrap().is_empty());
    }

    #[test]
    fn logout_removes_only_the_local_file() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("muse").join("auth.json");
        let store = store_at(path.to_string_lossy().into_owned());
        store.save_auth(stored()).unwrap();
        let message = logout_with(&store).unwrap();
        assert!(!path.exists());
        assert!(message.contains("not touched"), "{message}");
        // Logging out twice is not an error.
        logout_with(&store).unwrap();
    }

    #[test]
    fn login_summary_lists_the_saved_display_fields() {
        assert_eq!(
            login_summary(&stored(), "/tmp/muse/auth.json"),
            "Muse authentication saved in /tmp/muse/auth.json\nUser: Stored Name\nSubscription active: true\nTier: Muse Code Max Usage"
        );
    }

    #[test]
    fn durations_and_window_labels_read_naturally() {
        assert_eq!(human_duration(59), "0m");
        assert_eq!(human_duration(3_725), "1h 2m");
        assert_eq!(human_duration(2 * 86_400 + 3 * 3_600), "2d 3h");
        let window = |minutes| UsageWindow {
            used_percent: 0,
            resets_at: 0,
            window_duration_mins: minutes,
        };
        assert_eq!(window_label(&window(Some(300))), "5-hour window");
        assert_eq!(window_label(&window(Some(90))), "90-minute window");
        assert_eq!(window_label(&window(None)), "Window");
    }
}
