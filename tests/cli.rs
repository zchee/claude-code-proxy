use assert_cmd::Command;
use predicates::str::contains;
use std::env;
#[cfg(unix)]
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::{Child, ExitStatus, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;

#[test]
fn version_aliases_print_expected_version() -> Result<(), Box<dyn std::error::Error>> {
    let expected = format!("claude-code-proxy {}", env!("CARGO_PKG_VERSION"));

    for arg in ["--version", "-v", "version"] {
        let mut cmd = Command::cargo_bin("claude-code-proxy")?;
        cmd.arg(arg)
            .assert()
            .success()
            .stdout(contains(expected.clone()));
    }
    Ok(())
}

#[test]
fn models_prints_all_providers() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.arg("models");
    let out = String::from_utf8(cmd.output()?.stdout)?;
    assert!(out.contains("codex:"));
    assert!(out.contains("kimi:"));
    assert!(out.contains("opencode:"));
    assert!(out.contains("cursor:"));

    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.args(["models", "--full"]);
    cmd.output()?;
    Ok(())
}

#[test]
fn help_describes_visible_commands_and_hides_demo() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.arg("--help");
    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout)?;
    for description in [
        "Print version information",
        "Start the proxy server and monitor",
        "List supported provider models",
        "Manage Codex authentication",
        "Manage Kimi authentication",
        "Manage Cursor authentication",
        "Manage Grok authentication",
    ] {
        assert!(stdout.contains(description), "missing: {description}");
    }
    assert!(!stdout.contains("demo"));
    assert!(!stdout.contains("mock data and no proxy server"));
    Ok(())
}

#[test]
fn invalid_command_exits_two() -> Result<(), Box<dyn std::error::Error>> {
    Command::cargo_bin("claude-code-proxy")?
        .arg("definitely-not-a-command")
        .assert()
        .failure()
        .code(2);
    Ok(())
}

#[test]
fn unsupported_provider_auth_command_exits_two() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.args(["cursor", "auth", "device"]);
    let output = cmd.output()?;
    assert_eq!(output.status.code(), Some(2));
    let out = String::from_utf8(output.stderr)?;
    assert!(out.contains("not yet implemented") || out.contains("unsupported"));
    Ok(())
}

#[test]
fn provider_logout_without_auth_is_success() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.args(["kimi", "auth", "logout"]);
    cmd.env("CCP_CONFIG_DIR", temp.path());
    // Logout clears the legacy path as well as the primary credential path.
    cmd.env("HOME", temp.path());
    cmd.env("USERPROFILE", temp.path());
    cmd.assert().success();
    Ok(())
}

#[test]
fn models_output_is_stable_order() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.args(["models", "--full"]);
    let output = cmd.output()?;
    let out = String::from_utf8(output.stdout)?;
    let codex_pos = out.find("codex:").unwrap_or(0);
    let kimi_pos = out.find("kimi:").unwrap_or(0);
    let cursor_pos = out.find("cursor:").unwrap_or(0);
    assert!(codex_pos < kimi_pos);
    assert!(kimi_pos < cursor_pos);
    Ok(())
}

#[cfg(unix)]
struct ChildGuard(Child);

#[cfg(unix)]
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[cfg(unix)]
fn wait_for_service(
    child: &mut ChildGuard,
    port: u16,
) -> Result<TcpStream, Box<dyn std::error::Error>> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(stream) => return Ok(stream),
            Err(error) if Instant::now() < deadline => {
                if let Some(status) = child.0.try_wait()? {
                    let mut stderr = String::new();
                    if let Some(mut pipe) = child.0.stderr.take() {
                        pipe.read_to_string(&mut stderr)?;
                    }
                    return Err(format!("service exited with {status}: {stderr}").into());
                }
                let _ = error;
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(unix)]
fn send_signal(child: &ChildGuard, signal: &str) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new("kill")
        .args([signal, &child.0.id().to_string()])
        .status()?;
    if !status.success() {
        return Err(format!("kill {signal} failed with {status}").into());
    }
    Ok(())
}

#[cfg(unix)]
fn wait_for_exit(
    child: &mut ChildGuard,
    timeout: Duration,
) -> Result<ExitStatus, Box<dyn std::error::Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.0.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            return Err("plain service did not exit after the second signal".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(unix)]
fn plain_service_exits_on_second_signal(signal: &str) -> Result<(), Box<dyn std::error::Error>> {
    let upstream = TcpListener::bind("127.0.0.1:0")?;
    let upstream_url = format!("http://{}", upstream.local_addr()?);
    let (accepted_tx, accepted_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let fixture = thread::spawn(move || {
        let (stream, _) = upstream.accept().unwrap();
        accepted_tx.send(()).unwrap();
        let _ = release_rx.recv();
        drop(stream);
    });

    let config = TempDir::new()?;
    let auth_dir = config.path().join("kimi");
    std::fs::create_dir_all(&auth_dir)?;
    std::fs::write(
        auth_dir.join("auth.json"),
        r#"{"access":"test","refresh":"test","expires":4102444800000,"scope":"openid","userId":"test"}"#,
    )?;
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_claude-code-proxy"))
        .args(["serve", "--no-monitor", "--port", &port.to_string()])
        .env("CCP_CONFIG_DIR", config.path())
        .env("CCP_KIMI_BASE_URL", upstream_url)
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut child = ChildGuard(child);
    let mut downstream = wait_for_service(&mut child, port)?;
    let body = br#"{"model":"kimi-for-coding","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}"#;
    write!(
        downstream,
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )?;
    downstream.write_all(body)?;
    accepted_rx.recv_timeout(Duration::from_secs(20))?;

    send_signal(&child, signal)?;
    thread::sleep(Duration::from_millis(200));
    assert!(child.0.try_wait()?.is_none());
    send_signal(&child, signal)?;
    let status = wait_for_exit(&mut child, Duration::from_secs(2));
    let _ = release_tx.send(());
    fixture.join().unwrap();

    assert_eq!(status?.code(), Some(130));
    Ok(())
}

#[cfg(unix)]
#[test]
fn plain_service_exits_on_second_ctrl_c() -> Result<(), Box<dyn std::error::Error>> {
    plain_service_exits_on_second_signal("-INT")
}

#[cfg(unix)]
#[test]
fn plain_service_exits_on_second_sigterm() -> Result<(), Box<dyn std::error::Error>> {
    plain_service_exits_on_second_signal("-TERM")
}

#[test]
fn kimi_auth_status_reads_stored_auth() -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let auth_dir = temp.path().join("kimi");
    std::fs::create_dir_all(&auth_dir)?;
    std::fs::write(
        auth_dir.join("auth.json"),
        r#"{"access":"a","refresh":"r","expires":4102444800000,"scope":"openid","userId":"u"}"#,
    )?;
    let mut cmd = Command::cargo_bin("claude-code-proxy")?;
    cmd.args(["kimi", "auth", "status"]);
    cmd.env("CCP_CONFIG_DIR", temp.path());
    cmd.assert().success().stdout(contains("User: u"));
    Ok(())
}

/// Local stand-in for the Muse key mint and OAuth host. Every request is
/// counted; the mint answers with `status` and `body`.
struct MuseFixture {
    base: String,
    requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

fn muse_fixture(status: u16, body: &'static str) -> MuseFixture {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = requests.clone();
    let app = axum::Router::new().fallback(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        async move {
            (
                axum::http::StatusCode::from_u16(status).unwrap(),
                [("content-type", "application/json")],
                body,
            )
        }
    });
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    MuseFixture {
        base: format!("http://{}", receiver.recv().unwrap()),
        requests,
    }
}

const MUSE_OBSERVED_MINT: &str = r#"{"api_key":"synthetic-minted-api-key-48-chars-0000000000000","base_url":"https://api.meta.ai/v1","has_payment_method":false,"require_payment":false,"is_subs_active":true,"can_subscribe":false,"show_subs_upsell":true,"user_full_name":"Synthetic Person","user_email":"synthetic.person@example.com","user_avatar_url":null,"payment_method":null,"action_url":null,"subs_tier_id":"12345678901234567","subs_tier_name":"Muse Code Max Usage","is_subs_upgrade_available":false,"subs_usage":{"window":{"used_percent":1,"window_duration_mins":300,"resets_at":1794291326},"weekly":{"used_percent":0,"resets_at":1794787200},"tier":"12345678901234567"}}"#;

const MUSE_STORED_LOGIN: &str = r#"{"access_token":"synthetic-oauth-access-token","obtained_at_ms":1794274189000,"client_id":"1031625952748946","oauth_host":"https://auth.meta.com","user_full_name":"Stored Name","is_subs_active":true,"subs_tier_name":"Muse Code Max Usage"}"#;

fn muse_command(config: &TempDir, fixture: &MuseFixture, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("claude-code-proxy").unwrap();
    cmd.args(args)
        .env("CCP_CONFIG_DIR", config.path())
        .env("HOME", config.path())
        .env("USERPROFILE", config.path())
        .env("CCP_MUSE_MINT_BASE_URL", &fixture.base)
        .env("CCP_MUSE_OAUTH_HOST", &fixture.base)
        .env("CCP_MUSE_BASE_URL", format!("{}/v1", fixture.base))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost");
    cmd
}

fn store_muse_login(config: &TempDir) -> std::path::PathBuf {
    let dir = config.path().join("muse");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("auth.json");
    std::fs::write(&path, MUSE_STORED_LOGIN).unwrap();
    path
}

fn muse_requests(fixture: &MuseFixture) -> usize {
    fixture.requests.load(std::sync::atomic::Ordering::SeqCst)
}

#[test]
fn models_lists_muse_after_grok_with_its_five_models() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::cargo_bin("claude-code-proxy")?
        .args(["models"])
        .output()?;
    let out = String::from_utf8(output.stdout)?;
    let position = |label: &str| {
        out.find(&format!("\n{label}:"))
            .or_else(|| out.starts_with(&format!("{label}:")).then_some(0))
    };
    let order: Vec<_> = ["codex", "kimi", "grok", "muse", "opencode", "cursor"]
        .iter()
        .map(|label| position(label).unwrap_or_else(|| panic!("missing {label}: {out}")))
        .collect();
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{out}");
    let muse = out.lines().find(|line| line.starts_with("muse:")).unwrap();
    assert_eq!(
        muse,
        "muse: muse-ember-1.0, muse-spark-1.2, muse-spark-1.3, muse-wonder-1.0, muse-wonder-full-1.0"
    );
    Ok(())
}

#[test]
fn muse_auth_status_reports_the_stored_login() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(200, MUSE_OBSERVED_MINT);
    let config = TempDir::new()?;
    let path = store_muse_login(&config);
    let output = muse_command(&config, &fixture, &["muse", "auth", "status"]).output()?;
    let stdout = String::from_utf8(output.stdout)?;
    assert!(output.status.success(), "{stdout}");
    for expected in [
        format!("Auth path: {}", path.display()),
        "Authenticated: true".into(),
        "User: Synthetic Person".into(),
        "Subscription active: true".into(),
        "Tier: Muse Code Max Usage".into(),
        "5-hour window: 1% used, resets ".into(),
        "Weekly: 0% used, resets ".into(),
        "Logged in: ".into(),
    ] {
        assert!(stdout.contains(&expected), "missing {expected:?}: {stdout}");
    }
    assert!(!stdout.contains("synthetic.person@example.com"));
    assert!(!stdout.contains("synthetic-minted-api-key"));
    assert_eq!(muse_requests(&fixture), 1);
    Ok(())
}

#[test]
fn muse_auth_status_without_a_login_exits_one_without_calling_upstream()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(200, MUSE_OBSERVED_MINT);
    let config = TempDir::new()?;
    let output = muse_command(&config, &fixture, &["muse", "auth", "status"]).output()?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(output.stdout)?.trim(),
        "Not authenticated"
    );
    assert!(String::from_utf8(output.stderr)?.contains("claude-code-proxy muse auth login"));
    assert_eq!(muse_requests(&fixture), 0);
    Ok(())
}

#[test]
fn muse_auth_status_with_a_refused_login_exits_one_and_asks_for_a_new_one()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(
        401,
        r#"{"title":"Authentication Error","detail":"Authentication Error","status":401}"#,
    );
    let config = TempDir::new()?;
    store_muse_login(&config);
    let output = muse_command(&config, &fixture, &["muse", "auth", "status"]).output()?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(output.stdout)?.trim(),
        "Not authenticated"
    );
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("Muse login is no longer valid. Run: claude-code-proxy muse auth login"),
        "{stderr}"
    );
    assert!(!stderr.contains("synthetic-oauth-access-token"), "{stderr}");
    assert_eq!(muse_requests(&fixture), 1);
    Ok(())
}

#[test]
fn muse_auth_status_with_an_unavailable_mint_exits_two_with_its_own_message()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(503, "{}");
    let config = TempDir::new()?;
    store_muse_login(&config);
    let output = muse_command(&config, &fixture, &["muse", "auth", "status"]).output()?;
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        String::from_utf8(output.stdout)?.trim(),
        "Muse key service is unavailable (upstream HTTP 503)"
    );
    assert_eq!(muse_requests(&fixture), 1);
    Ok(())
}

#[test]
fn muse_auth_synthetic_unsafe_settings_stop_the_cli_before_any_request()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(200, MUSE_OBSERVED_MINT);
    let config = TempDir::new()?;
    store_muse_login(&config);
    for (args, setting, value) in [
        (
            &["muse", "auth", "status"][..],
            "CCP_MUSE_MINT_BASE_URL",
            "http://relay.example",
        ),
        (
            &["muse", "auth", "status"][..],
            "CCP_MUSE_BASE_URL",
            "https://user:synthetic-pass@api.meta.ai/v1",
        ),
        (
            &["muse", "auth", "login"][..],
            "CCP_MUSE_OAUTH_HOST",
            "http://auth.relay.example",
        ),
        (
            &["muse", "auth", "status"][..],
            "CCP_MUSE_USER_AGENT",
            "synthetic-pass\r\nX-Injected: 1",
        ),
        (
            &["muse", "auth", "login"][..],
            "CCP_MUSE_USER_AGENT",
            "synthetic-pass\u{1b}[2J",
        ),
    ] {
        let output = muse_command(&config, &fixture, args)
            .env(setting, value)
            .output()?;
        assert!(!output.status.success(), "{setting}");
        let text = String::from_utf8(output.stdout)? + &String::from_utf8(output.stderr)?;
        assert!(
            text.contains(&format!("Invalid Muse setting {setting}: ")),
            "{setting}: {text}"
        );
        assert!(!text.contains("synthetic-pass"), "{text}");
    }
    assert_eq!(muse_requests(&fixture), 0);
    Ok(())
}

#[cfg(unix)]
#[test]
fn muse_server_synthetic_unsafe_settings_fail_the_request_with_their_name()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(200, MUSE_OBSERVED_MINT);
    let config = TempDir::new()?;
    store_muse_login(&config);
    for (setting, value, expected) in [
        (
            "CCP_MUSE_MINT_BASE_URL",
            "http://relay.example",
            "Invalid Muse setting CCP_MUSE_MINT_BASE_URL: must use https (plain http only for a loopback host)",
        ),
        (
            "CCP_MUSE_USER_AGENT",
            "synthetic-pass\r\nX-Injected: 1",
            "Invalid Muse setting CCP_MUSE_USER_AGENT: must contain only printable ASCII characters",
        ),
    ] {
        let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
        let child = std::process::Command::new(env!("CARGO_BIN_EXE_claude-code-proxy"))
            .args(["serve", "--no-monitor", "--port", &port.to_string()])
            .env("CCP_CONFIG_DIR", config.path())
            .env("HOME", config.path())
            .env("USERPROFILE", config.path())
            .env("CCP_MUSE_MINT_BASE_URL", &fixture.base)
            .env("CCP_MUSE_OAUTH_HOST", &fixture.base)
            .env("CCP_MUSE_BASE_URL", format!("{}/v1", fixture.base))
            .env(setting, value)
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("no_proxy", "127.0.0.1,localhost")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut child = ChildGuard(child);
        let mut downstream = wait_for_service(&mut child, port)?;
        let body = br#"{"model":"muse-spark-1.3","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}"#;
        write!(
            downstream,
            "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )?;
        downstream.write_all(body)?;
        downstream.set_read_timeout(Some(Duration::from_secs(20)))?;
        let mut response = String::new();
        downstream.read_to_string(&mut response)?;
        assert!(
            response.starts_with("HTTP/1.1 500"),
            "{setting}: {response}"
        );
        assert!(response.contains(expected), "{setting}: {response}");
        assert!(!response.contains("synthetic-pass"), "{response}");
    }
    assert_eq!(muse_requests(&fixture), 0);
    Ok(())
}

#[test]
fn muse_auth_synthetic_padded_endpoint_settings_reach_the_configured_hosts()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(200, MUSE_OBSERVED_MINT);
    let config = TempDir::new()?;
    store_muse_login(&config);
    let output = muse_command(&config, &fixture, &["muse", "auth", "status"])
        .env("CCP_MUSE_MINT_BASE_URL", format!(" {} ", fixture.base))
        .env("CCP_MUSE_BASE_URL", format!("\t{}/v1 ", fixture.base))
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("Authenticated: true"), "{stdout}");
    assert_eq!(muse_requests(&fixture), 1);

    // The fixture answers the device authorization with the mint body, so
    // the login stops there; reaching it proves the OAuth URL was built.
    let output = muse_command(&config, &fixture, &["muse", "auth", "login"])
        .env("CCP_MUSE_OAUTH_HOST", format!(" {}\n", fixture.base))
        .output()?;
    assert!(!output.status.success());
    let text = String::from_utf8(output.stdout)? + &String::from_utf8(output.stderr)?;
    assert!(
        text.contains("Muse device authorization response is not valid"),
        "{text}"
    );
    assert_eq!(muse_requests(&fixture), 2);
    Ok(())
}

/// Mint answering with the observed key and a Model API answering each
/// request with the next of `answers` (status, content type, body).
#[cfg(unix)]
fn muse_hostile_upstream(answers: Vec<(u16, &'static str, String)>) -> String {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    let queue = Arc::new(Mutex::new(VecDeque::from(answers)));
    let app = axum::Router::new()
        .route(
            "/v1/responses",
            axum::routing::post(move || {
                let next = queue.lock().unwrap().pop_front();
                async move {
                    let (status, content_type, body) =
                        next.unwrap_or((500, "application/json", "{}".into()));
                    (
                        axum::http::StatusCode::from_u16(status).unwrap(),
                        [("content-type", content_type)],
                        body,
                    )
                }
            }),
        )
        .route(
            "/muse-code/key",
            axum::routing::post(|| async {
                ([("content-type", "application/json")], MUSE_OBSERVED_MINT)
            }),
        );
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    format!("http://{}", receiver.recv().unwrap())
}

/// One request on its own connection; returns the raw HTTP response.
#[cfg(unix)]
fn post_raw(port: u16, body: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )?;
    stream.set_read_timeout(Some(Duration::from_secs(20)))?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

/// Every file under `dir`, concatenated.
#[cfg(unix)]
fn all_files_text(dir: &std::path::Path) -> String {
    let mut text = String::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                text.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
                text.push('\n');
            }
        }
    }
    text
}

#[cfg(unix)]
#[test]
fn muse_server_synthetic_hostile_upstream_failures_leak_nothing_into_logs_or_captures()
-> Result<(), Box<dyn std::error::Error>> {
    const ACCESS_TOKEN: &str = "synthetic-oauth-access-token";
    const API_KEY: &str = "synthetic-minted-api-key-48-chars-0000000000000";
    const EMAIL: &str = "synthetic.person@example.com";
    let text = format!("key {API_KEY} token {ACCESS_TOKEN} owner {EMAIL}");
    let http_error = serde_json::json!({
        "error": {"code": API_KEY, "message": text, "param": EMAIL, "type": "invalid_request_error"},
        "detail": format!("detail {ACCESS_TOKEN} {EMAIL}"),
        "title": format!("title {API_KEY}")
    })
    .to_string();
    let failed = serde_json::json!({
        "type": "response.failed",
        "response": {"status": "failed", "error": {"code": API_KEY, "message": text, "param": EMAIL}}
    });
    let failed = format!("event: response.failed\ndata: {failed}\n\ndata: [DONE]\n\n");
    let error =
        serde_json::json!({"type": "error", "code": API_KEY, "message": text, "param": EMAIL});
    let error = format!("event: error\ndata: {error}\n\ndata: [DONE]\n\n");
    let sse = "text/event-stream";
    let json = "application/json";
    let base = muse_hostile_upstream(vec![
        (400, json, http_error),
        (200, sse, failed.clone()),
        (200, sse, error.clone()),
        (200, sse, failed),
        (200, sse, error),
    ]);

    let config = TempDir::new()?;
    let state = TempDir::new()?;
    store_muse_login(&config);
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_claude-code-proxy"))
        .args(["serve", "--no-monitor", "--port", &port.to_string()])
        .env("CCP_CONFIG_DIR", config.path())
        .env("HOME", config.path())
        .env("USERPROFILE", config.path())
        .env("XDG_STATE_HOME", state.path())
        .env("CCP_TRAFFIC_LOG", "1")
        .env("CCP_MUSE_MINT_BASE_URL", &base)
        .env("CCP_MUSE_OAUTH_HOST", &base)
        .env("CCP_MUSE_BASE_URL", format!("{base}/v1"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut child = ChildGuard(child);
    drop(wait_for_service(&mut child, port)?);

    let mut downstream = String::new();
    for stream in [false, false, false, true, true] {
        let body = serde_json::json!({
            "model": "muse-spark-1.3",
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]
        })
        .to_string();
        downstream.push_str(&post_raw(port, &body)?);
    }
    // The reader may log just after the client side of a stream ends.
    thread::sleep(Duration::from_millis(500));
    child.0.kill()?;
    child.0.wait()?;
    let mut stderr = String::new();
    if let Some(mut pipe) = child.0.stderr.take() {
        pipe.read_to_string(&mut stderr)?;
    }

    let state_dir = state.path().join("claude-code-proxy");
    let log = std::fs::read_to_string(state_dir.join("proxy.log"))?;
    // The scan below is only meaningful if the failures were recorded.
    assert_eq!(log.matches("\"request_failed\"").count(), 3, "{log}");
    assert_eq!(log.matches("\"muse_stream_failed\"").count(), 4, "{log}");
    assert!(
        log.contains("key [redacted] token [redacted] owner [redacted]"),
        "{log}"
    );
    let errors = std::fs::read_dir(state_dir.join("errors"))?.count();
    assert_eq!(errors, 3);
    let files = all_files_text(&state_dir);
    for (place, text) in [
        ("state files", &files),
        ("stderr", &stderr),
        ("responses", &downstream),
    ] {
        for secret in [API_KEY, ACCESS_TOKEN, EMAIL, "@example.com"] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
    Ok(())
}

#[test]
fn muse_auth_logout_removes_the_file_and_calls_nothing_upstream()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = muse_fixture(200, MUSE_OBSERVED_MINT);
    let config = TempDir::new()?;
    let path = store_muse_login(&config);
    let output = muse_command(&config, &fixture, &["muse", "auth", "logout"]).output()?;
    assert!(output.status.success());
    assert!(!path.exists());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("not touched"), "{stdout}");
    // Logging out again is not an error either.
    muse_command(&config, &fixture, &["muse", "auth", "logout"])
        .assert()
        .success();
    assert_eq!(muse_requests(&fixture), 0);
    Ok(())
}

#[test]
fn help_lists_the_muse_command_and_its_auth_actions() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::cargo_bin("claude-code-proxy")?
        .arg("--help")
        .output()?;
    assert!(String::from_utf8(output.stdout)?.contains("Manage Meta Muse Code authentication"));
    let output = Command::cargo_bin("claude-code-proxy")?
        .args(["muse", "auth", "--help"])
        .output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    for action in ["login", "device", "status", "logout"] {
        assert!(stdout.contains(action), "missing {action}: {stdout}");
    }
    Ok(())
}

/// Shaped like a Google OAuth access token.
const VERTEX_ACCESS_TOKEN: &str =
    "ya29.synthetic-access-token-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ";
const VERTEX_CLIENT_SECRET: &str = "synthetic-client-secret-d41d8cd98f00b204";
const VERTEX_REFRESH_TOKEN: &str = "1//synthetic-refresh-token-e9800998ecf8427e";
/// A Gemini thought signature: opaque base64, long enough to be redacted.
const VERTEX_SIGNATURE: &str =
    "CiQBVKhc7synthetic0signature0AAAAbbbbCCCCddddEEEEffff0123456789GGGGhhhhIIIIjjjjKKKK==";
const VERTEX_IMAGE_DATA: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg";
const VERTEX_EMAIL: &str = "synthetic.person@example.com";
/// Body text of a synthetic service account private key.
const VERTEX_PRIVATE_KEY_BODY: &str = "synthetic0private0key0body";

/// Variables that would let a Vertex command find the user's real settings
/// or credentials.
const VERTEX_REAL_ENV: &[&str] = &[
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "GOOGLE_CLOUD_UNIVERSE_DOMAIN",
    "CLOUDSDK_CONFIG",
    "CCP_VERTEX_PROJECT",
    "CCP_VERTEX_LOCATION",
    "CCP_VERTEX_LOCATION_OVERRIDES",
    "CCP_VERTEX_BASE_URL",
];

/// A loopback address nothing listens on, so a connection fails at once.
fn closed_loopback_address() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    address.to_string()
}

/// Settings for a command that may look up Google credentials: every home,
/// config and state directory is `home`, and the metadata server is a closed
/// loopback port, so the command can neither read the user's files nor
/// contact Google. Apply after removing [`VERTEX_REAL_ENV`].
fn vertex_isolated_env(home: &std::path::Path) -> Vec<(&'static str, String)> {
    let home_text = home.display().to_string();
    vec![
        ("HOME", home_text.clone()),
        ("USERPROFILE", home_text.clone()),
        ("APPDATA", home_text.clone()),
        ("XDG_STATE_HOME", home.join("state").display().to_string()),
        ("CCP_CONFIG_DIR", home.join("config").display().to_string()),
        ("GCE_METADATA_HOST", closed_loopback_address()),
        ("NO_PROXY", "127.0.0.1,localhost".to_string()),
        ("no_proxy", "127.0.0.1,localhost".to_string()),
    ]
}

fn vertex_command(home: &TempDir, args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("claude-code-proxy").unwrap();
    cmd.args(args);
    for name in VERTEX_REAL_ENV {
        cmd.env_remove(name);
    }
    for (name, value) in vertex_isolated_env(home.path()) {
        cmd.env(name, value);
    }
    cmd.timeout(std::time::Duration::from_secs(90));
    cmd
}

/// Loopback stand-in for Google's OAuth token endpoint (`POST /token`, which
/// answers with [`VERTEX_ACCESS_TOKEN`]) and for Vertex AI (every other path,
/// answered with the next of `answers`: status, content type, body).
struct VertexFixture {
    base: String,
    token_requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

fn vertex_fixture(answers: Vec<(u16, &'static str, String)>) -> VertexFixture {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    let queue = Arc::new(Mutex::new(VecDeque::from(answers)));
    let token_requests = Arc::new(AtomicUsize::new(0));
    let counted = token_requests.clone();
    let token_body = serde_json::json!({
        "access_token": VERTEX_ACCESS_TOKEN,
        "token_type": "Bearer",
        "expires_in": 3600
    })
    .to_string();
    let app = axum::Router::new()
        .route(
            "/token",
            axum::routing::post(move || {
                counted.fetch_add(1, Ordering::SeqCst);
                let body = token_body.clone();
                async move { ([("content-type", "application/json")], body) }
            }),
        )
        .fallback(move || {
            let next = queue.lock().unwrap().pop_front();
            async move {
                let (status, content_type, body) = next.unwrap_or((
                    500,
                    "application/json",
                    r#"{"error":{"code":500,"message":"no scripted answer left","status":"INTERNAL"}}"#
                        .to_string(),
                ));
                (
                    axum::http::StatusCode::from_u16(status).unwrap(),
                    [("content-type", content_type)],
                    body,
                )
            }
        });
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, app).await.unwrap();
        });
    });
    VertexFixture {
        base: format!("http://{}", receiver.recv().unwrap()),
        token_requests,
    }
}

/// A user credential in gcloud's format whose token endpoint is the
/// fixture, so the credential library refreshes against loopback.
fn vertex_user_credentials(fixture: &VertexFixture) -> String {
    serde_json::json!({
        "type": "authorized_user",
        "client_id": "synthetic-client-id.apps.googleusercontent.com",
        "client_secret": VERTEX_CLIENT_SECRET,
        "refresh_token": VERTEX_REFRESH_TOKEN,
        "token_uri": format!("{}/token", fixture.base),
        "universe_domain": "googleapis.com"
    })
    .to_string()
}

/// [`vertex_user_credentials`] in a file of its own inside `home`.
fn write_vertex_credentials(home: &TempDir, fixture: &VertexFixture) -> std::path::PathBuf {
    let path = home.path().join("synthetic-adc.json");
    std::fs::write(&path, vertex_user_credentials(fixture)).unwrap();
    path
}

/// The file inside `home` that [`vertex_server`] sends the server's stderr
/// to. A file rather than a pipe, so a server that writes a lot never
/// stalls on a pipe that nobody reads while the test runs.
#[cfg(unix)]
fn vertex_server_stderr(home: &TempDir) -> std::path::PathBuf {
    home.path().join("server-stderr.log")
}

/// `claude-code-proxy serve` on `port` with the Vertex settings isolated in
/// `home`, the project set and Vertex AI pointed at `fixture`. `extra_env`
/// adds variables, such as `GOOGLE_APPLICATION_CREDENTIALS`. The server's
/// stderr goes to [`vertex_server_stderr`].
#[cfg(unix)]
fn vertex_server(
    home: &TempDir,
    fixture: &VertexFixture,
    port: u16,
    extra_env: &[(&str, &std::ffi::OsStr)],
) -> Result<ChildGuard, Box<dyn std::error::Error>> {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_claude-code-proxy"));
    command.args(["serve", "--no-monitor", "--port", &port.to_string()]);
    for name in VERTEX_REAL_ENV {
        command.env_remove(name);
    }
    for (name, value) in vertex_isolated_env(home.path()) {
        command.env(name, value);
    }
    for (name, value) in extra_env {
        command.env(name, value);
    }
    let stderr = std::fs::File::create(vertex_server_stderr(home))?;
    let child = command
        .env("CCP_VERTEX_PROJECT", "synthetic-project")
        .env("CCP_VERTEX_BASE_URL", &fixture.base)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .spawn()?;
    let mut child = ChildGuard(child);
    if let Err(error) = wait_for_service(&mut child, port) {
        let stderr = std::fs::read_to_string(vertex_server_stderr(home)).unwrap_or_default();
        return Err(format!("{error}; server stderr: {stderr}").into());
    }
    Ok(child)
}

#[test]
fn vertex_auth_login_and_device_point_at_gcloud() -> Result<(), Box<dyn std::error::Error>> {
    let home = TempDir::new()?;
    for action in ["login", "device"] {
        let output = vertex_command(&home, &["vertex", "auth", action]).output()?;
        assert_eq!(output.status.code(), Some(2), "{action}");
        let stderr = String::from_utf8(output.stderr)?;
        assert!(
            stderr.contains(
                "Vertex AI uses Google Application Default Credentials, which the proxy does not store. Run `gcloud auth application-default login`, or set GOOGLE_APPLICATION_CREDENTIALS to a service account key file"
            ),
            "{action}: {stderr}"
        );
    }
    Ok(())
}

#[test]
fn vertex_auth_logout_says_gcloud_manages_the_credentials() -> Result<(), Box<dyn std::error::Error>>
{
    let home = TempDir::new()?;
    let output = vertex_command(&home, &["vertex", "auth", "logout"]).output()?;
    // The handler's error leaves `main` as an error: exit 1.
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "Vertex AI credentials are managed by gcloud, not by the proxy. Run `gcloud auth application-default revoke` to remove them"
        ),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn vertex_auth_status_without_a_project_exits_two_and_names_the_settings()
-> Result<(), Box<dyn std::error::Error>> {
    let home = TempDir::new()?;
    let output = vertex_command(&home, &["vertex", "auth", "status"]).output()?;
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains(
            "Vertex project is not set: set CCP_VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT or vertex.project in config.json"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("Location: global (default)"), "{stdout}");
    Ok(())
}

#[test]
fn vertex_auth_status_without_credentials_exits_one_with_the_gcloud_command()
-> Result<(), Box<dyn std::error::Error>> {
    let home = TempDir::new()?;
    let output = vertex_command(&home, &["vertex", "auth", "status"])
        .env("CCP_VERTEX_PROJECT", "synthetic-project")
        .output()?;
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("Project: synthetic-project (CCP_VERTEX_PROJECT)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Credential type: metadata server"),
        "{stdout}"
    );
    assert!(stdout.contains("Access token obtained: false"), "{stdout}");
    assert!(stdout.contains("Not authenticated"), "{stdout}");
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("gcloud auth application-default login"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn vertex_auth_status_synthetic_credentials_file_gets_a_token_and_prints_no_secret()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = vertex_fixture(Vec::new());
    let home = TempDir::new()?;
    let credentials = write_vertex_credentials(&home, &fixture);
    let output = vertex_command(&home, &["vertex", "auth", "status"])
        .env("GOOGLE_APPLICATION_CREDENTIALS", &credentials)
        .env("CCP_VERTEX_PROJECT", "synthetic-project")
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8(output.stderr)?;
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains(&format!(
            "Credentials: {} (GOOGLE_APPLICATION_CREDENTIALS)",
            credentials.display()
        )),
        "{stdout}"
    );
    assert!(
        stdout.contains("Credential type: authorized_user"),
        "{stdout}"
    );
    assert!(stdout.contains("Access token obtained: true"), "{stdout}");
    assert!(
        stdout.contains("Endpoint: https://aiplatform.googleapis.com (derived from the location)"),
        "{stdout}"
    );
    for secret in [
        VERTEX_ACCESS_TOKEN,
        VERTEX_CLIENT_SECRET,
        VERTEX_REFRESH_TOKEN,
    ] {
        assert!(!stdout.contains(secret), "stdout leaked {secret}: {stdout}");
        assert!(!stderr.contains(secret), "stderr leaked {secret}: {stderr}");
    }
    assert!(
        fixture
            .token_requests
            .load(std::sync::atomic::Ordering::SeqCst)
            >= 1
    );
    Ok(())
}

#[test]
fn help_lists_the_vertex_command_and_its_auth_actions() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::cargo_bin("claude-code-proxy")?
        .arg("--help")
        .output()?;
    assert!(
        String::from_utf8(output.stdout)?.contains("Manage Google Cloud Vertex AI credentials")
    );
    let output = Command::cargo_bin("claude-code-proxy")?
        .args(["vertex", "auth", "--help"])
        .output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    for action in ["login", "device", "status", "logout"] {
        assert!(stdout.contains(action), "missing {action}: {stdout}");
    }
    Ok(())
}

#[test]
fn models_lists_vertex_after_opencode_with_its_served_models()
-> Result<(), Box<dyn std::error::Error>> {
    let output = Command::cargo_bin("claude-code-proxy")?
        .args(["models"])
        .output()?;
    let out = String::from_utf8(output.stdout)?;
    let position = |label: &str| {
        out.find(&format!("\n{label}:"))
            .or_else(|| out.starts_with(&format!("{label}:")).then_some(0))
    };
    let order: Vec<_> = [
        "codex", "kimi", "grok", "muse", "opencode", "vertex", "cursor",
    ]
    .iter()
    .map(|label| position(label).unwrap_or_else(|| panic!("missing {label}: {out}")))
    .collect();
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{out}");
    let vertex = out
        .lines()
        .find(|line| line.starts_with("vertex:"))
        .unwrap();
    let models: Vec<&str> = vertex
        .trim_start_matches("vertex:")
        .split(',')
        .map(str::trim)
        .collect();
    assert_eq!(models.len(), 47, "{vertex}");
    for model in [
        "gemini-4-argon",
        "gemini-4-carbon",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
    ] {
        assert!(models.contains(&model), "missing {model}: {vertex}");
        let prefixed = format!("vertex/{model}");
        assert!(
            models.contains(&prefixed.as_str()),
            "missing {prefixed}: {vertex}"
        );
    }
    for model in [
        "vertex/anthropic/claude-opus-5-5",
        "vertex/anthropic/claude-opus-4-5@20251101",
        "vertex/anthropic/claude-haiku-4-5@20251001",
    ] {
        assert!(models.contains(&model), "missing {model}: {vertex}");
    }
    // Claude appears only with its prefix.
    assert!(
        !models.iter().any(|model| model.starts_with("claude-")),
        "{vertex}"
    );
    Ok(())
}

/// The credential library picks its credential once, when it is built:
/// without a file it settles on the metadata server. A request that could
/// not get a token must therefore keep nothing, so that credentials created
/// afterwards (here `gcloud auth application-default login`'s default file)
/// are used by the next request without restarting the proxy.
#[cfg(unix)]
#[test]
fn vertex_server_synthetic_credentials_created_after_a_failed_request_are_used_without_a_restart()
-> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::Ordering;
    let text = serde_json::json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello from Vertex"}]}}]
    });
    let finish = serde_json::json!({
        "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]
    });
    let answer = format!("data: {text}\n\ndata: {finish}\n\n");
    let fixture = vertex_fixture(vec![
        (200, "text/event-stream", answer.clone()),
        (200, "text/event-stream", answer),
    ]);
    let home = TempDir::new()?;
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    // No GOOGLE_APPLICATION_CREDENTIALS and no default file: the library
    // falls back to the metadata server, which is a closed loopback port.
    let _child = vertex_server(&home, &fixture, port, &[])?;
    let body = serde_json::json!({
        "model": "gemini-3.5-flash",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}]
    })
    .to_string();

    let first = post_raw(port, &body)?;
    assert!(first.starts_with("HTTP/1.1 401"), "{first}");
    assert!(first.contains("authentication_error"), "{first}");
    assert!(
        first.contains("Google access token could not be obtained"),
        "{first}"
    );
    assert!(
        first.contains("gcloud auth application-default login"),
        "{first}"
    );
    assert_eq!(fixture.token_requests.load(Ordering::SeqCst), 0);

    let default_file = home
        .path()
        .join(".config")
        .join("gcloud")
        .join("application_default_credentials.json");
    std::fs::create_dir_all(default_file.parent().unwrap())?;
    std::fs::write(&default_file, vertex_user_credentials(&fixture))?;

    let second = post_raw(port, &body)?;
    assert!(second.starts_with("HTTP/1.1 200"), "{second}");
    assert!(second.contains("Hello from Vertex"), "{second}");
    let fetched = fixture.token_requests.load(Ordering::SeqCst);
    assert!(fetched >= 1, "no token was fetched from the new file");

    // From now on the credentials are kept: the next request uses the
    // cached token instead of loading them again.
    let third = post_raw(port, &body)?;
    assert!(third.starts_with("HTTP/1.1 200"), "{third}");
    assert_eq!(
        fixture.token_requests.load(Ordering::SeqCst),
        fetched,
        "the credentials were loaded again"
    );
    Ok(())
}

/// A service account key file whose private key cannot sign. The library
/// signs service account tokens itself, so it fails while building the
/// token, without any network request. Its error must not carry the key
/// into a response, the log, an error record, a capture or stderr.
#[cfg(unix)]
#[test]
fn vertex_server_synthetic_unusable_service_account_key_leaks_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::Ordering;
    let fixture = vertex_fixture(Vec::new());
    let home = TempDir::new()?;
    // Valid base64 that does not decode to a usable key.
    let line = format!("{}0123456789AB", VERTEX_PRIVATE_KEY_BODY.repeat(2));
    let private_key = format!(
        "-----BEGIN PRIVATE KEY-----\n{line}\n{line}\n{line}\n{line}\n-----END PRIVATE KEY-----\n"
    );
    let key_file = home.path().join("synthetic-service-account.json");
    std::fs::write(
        &key_file,
        serde_json::json!({
            "type": "service_account",
            "project_id": "synthetic-project",
            "private_key_id": "synthetic-private-key-id",
            "private_key": private_key,
            "client_email": "synthetic@synthetic-project.iam.gserviceaccount.com",
            "client_id": "000000000000000000000",
            "token_uri": format!("{}/token", fixture.base),
            "universe_domain": "googleapis.com"
        })
        .to_string(),
    )?;
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let mut child = vertex_server(
        &home,
        &fixture,
        port,
        &[
            ("GOOGLE_APPLICATION_CREDENTIALS", key_file.as_os_str()),
            ("CCP_TRAFFIC_LOG", std::ffi::OsStr::new("1")),
        ],
    )?;
    let mut downstream = String::new();
    for stream in [false, true] {
        let body = serde_json::json!({
            "model": "gemini-3.5-flash",
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]
        })
        .to_string();
        let response = post_raw(port, &body)?;
        assert!(response.starts_with("HTTP/1.1 401"), "{response}");
        assert!(
            response.contains("Google access token could not be obtained"),
            "{response}"
        );
        downstream.push_str(&response);
    }
    thread::sleep(Duration::from_millis(500));
    child.0.kill()?;
    child.0.wait()?;
    let stderr = std::fs::read_to_string(vertex_server_stderr(&home))?;

    // The library never asked the token endpoint: the failure is local.
    assert_eq!(fixture.token_requests.load(Ordering::SeqCst), 0);
    let state_dir = home.path().join("state").join("claude-code-proxy");
    let log = std::fs::read_to_string(state_dir.join("proxy.log"))?;
    assert!(log.contains("\"request_failed\""), "{log}");
    let files = all_files_text(&state_dir);
    for (place, text) in [
        ("state files", &files),
        ("stderr", &stderr),
        ("responses", &downstream),
    ] {
        for secret in [
            VERTEX_PRIVATE_KEY_BODY,
            line.as_str(),
            "PRIVATE KEY",
            "synthetic-private-key-id",
        ] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn vertex_server_synthetic_echoed_secrets_leak_nothing_into_logs_or_captures()
-> Result<(), Box<dyn std::error::Error>> {
    let echo = format!(
        "token {VERTEX_ACCESS_TOKEN} signature {VERTEX_SIGNATURE} image {VERTEX_IMAGE_DATA} owner {VERTEX_EMAIL}"
    );
    let http_error = serde_json::json!({
        "error": {"code": 400, "message": echo, "status": "INVALID_ARGUMENT",
                  "details": [{"@type": "type.googleapis.com/google.rpc.DebugInfo", "detail": echo}]}
    })
    .to_string();
    let partial = serde_json::json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "partial"}]}}]
    });
    let stream_error = serde_json::json!({
        "error": {"code": 500, "message": echo, "status": "INTERNAL"}
    });
    let stream_error = format!("data: {partial}\n\ndata: {stream_error}\n\n");
    let blocked = serde_json::json!({
        "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "SAFETY", "finishMessage": echo}]
    });
    let blocked = format!("data: {partial}\n\ndata: {blocked}\n\n");
    let sse = "text/event-stream";
    let fixture = vertex_fixture(vec![
        (400, "application/json", http_error),
        (200, sse, stream_error.clone()),
        (200, sse, stream_error),
        (200, sse, blocked.clone()),
        (200, sse, blocked),
    ]);

    let home = TempDir::new()?;
    let credentials = write_vertex_credentials(&home, &fixture);
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_claude-code-proxy"));
    command.args(["serve", "--no-monitor", "--port", &port.to_string()]);
    for name in VERTEX_REAL_ENV {
        command.env_remove(name);
    }
    for (name, value) in vertex_isolated_env(home.path()) {
        command.env(name, value);
    }
    let child = command
        .env("GOOGLE_APPLICATION_CREDENTIALS", &credentials)
        .env("CCP_VERTEX_PROJECT", "synthetic-project")
        .env("CCP_VERTEX_BASE_URL", &fixture.base)
        .env("CCP_TRAFFIC_LOG", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut child = ChildGuard(child);
    drop(wait_for_service(&mut child, port)?);

    // The request carries the same image and a signed tool call, so the
    // request captures are checked as well as the error paths.
    let envelope = format!("ccp:vertex:v1:gemini-3.5-flash:{VERTEX_SIGNATURE}");
    let mut downstream = String::new();
    for stream in [false, false, true, false, true] {
        let body = serde_json::json!({
            "model": "gemini-3.5-flash",
            "max_tokens": 64,
            "stream": stream,
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": VERTEX_IMAGE_DATA}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": envelope},
                    {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "fn a() {}"}
                ]}
            ]
        })
        .to_string();
        downstream.push_str(&post_raw(port, &body)?);
    }
    thread::sleep(Duration::from_millis(500));
    child.0.kill()?;
    child.0.wait()?;
    let mut stderr = String::new();
    if let Some(mut pipe) = child.0.stderr.take() {
        pipe.read_to_string(&mut stderr)?;
    }

    // The scan below is only meaningful if every path was exercised.
    assert_eq!(
        downstream.matches("HTTP/1.1 200").count(),
        2,
        "{downstream}"
    );
    assert!(downstream.contains("event: error"), "{downstream}");
    assert!(downstream.contains("finishReason SAFETY"), "{downstream}");
    assert!(downstream.contains("[redacted]"), "{downstream}");
    let state_dir = home.path().join("state").join("claude-code-proxy");
    let log = std::fs::read_to_string(state_dir.join("proxy.log"))?;
    assert_eq!(log.matches("\"request_failed\"").count(), 3, "{log}");
    assert!(log.contains("[redacted]"), "{log}");
    let errors = std::fs::read_dir(state_dir.join("errors"))?.count();
    assert_eq!(errors, 3);
    let mut names = Vec::new();
    let mut pending = vec![state_dir.join("traffic")];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                names.push(name.to_string());
            }
        }
    }
    for expected in [
        "020-upstream-request.json",
        "061-vertex-buffered-summary.json",
        "061-vertex-stream-summary.json",
    ] {
        assert!(
            names.iter().any(|name| name.ends_with(expected)),
            "missing {expected}: {names:?}"
        );
    }
    assert!(
        fixture
            .token_requests
            .load(std::sync::atomic::Ordering::SeqCst)
            >= 1
    );

    let files = all_files_text(&state_dir);
    for (place, text) in [
        ("state files", &files),
        ("stderr", &stderr),
        ("responses", &downstream),
    ] {
        for secret in [
            VERTEX_ACCESS_TOKEN,
            VERTEX_CLIENT_SECRET,
            VERTEX_REFRESH_TOKEN,
            VERTEX_SIGNATURE,
            VERTEX_IMAGE_DATA,
            VERTEX_EMAIL,
            "@example.com",
        ] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
    Ok(())
}

/// Every name Vertex lists, as Google's model pages spell them: the Gemini
/// IDs bare and prefixed, and each partner model as
/// `vertex/<publisher>/<model-id>`.
#[test]
fn models_lists_every_vertex_name_of_every_served_family() -> Result<(), Box<dyn std::error::Error>>
{
    let output = Command::cargo_bin("claude-code-proxy")?
        .args(["models"])
        .output()?;
    let out = String::from_utf8(output.stdout)?;
    let vertex = out
        .lines()
        .find(|line| line.starts_with("vertex:"))
        .ok_or_else(|| format!("no vertex line: {out}"))?;
    let mut listed: Vec<&str> = vertex
        .trim_start_matches("vertex:")
        .split(',')
        .map(str::trim)
        .collect();
    listed.sort_unstable();
    let gemini = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-4-argon",
        "gemini-4-carbon",
    ];
    let partners = [
        "anthropic/claude-fable-5-2",
        "anthropic/claude-fable-5-1",
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-5@20251101",
        "anthropic/claude-haiku-4-5@20251001",
        "xai/grok-4.7",
        "xai/grok-4.6",
        "xai/grok-4.20-reasoning",
        "xai/grok-4.20-non-reasoning",
        "xai/grok-4.3",
        "meta/muse-spark-1.3",
        "meta/muse-wonder-1.0",
        "meta/muse-ember-1.0",
        "openai/gpt-6-astra",
        "openai/gpt-6.1-sol",
        "openai/gpt-6.1-terra",
        "openai/gpt-6-luna",
        "mistralai/mistral-medium-3",
        "mistralai/mistral-small-2503",
        "mistralai/codestral-2",
    ];
    let mut expected: Vec<String> = gemini.iter().map(|model| (*model).to_string()).collect();
    expected.extend(gemini.iter().map(|model| format!("vertex/{model}")));
    expected.extend(partners.iter().map(|model| format!("vertex/{model}")));
    expected.sort_unstable();
    assert_eq!(expected.len(), 47);
    assert_eq!(listed, expected, "{vertex}");
    Ok(())
}

/// One partner family per wire (Claude's Messages pass-through, Grok's
/// Responses, Mistral's Chat Completions) whose upstream echoes the access
/// token and an email address in an HTTP error body, in a failed buffered
/// response and in a failed stream. Nothing may reach a response, the log,
/// an error record, a capture or stderr.
#[cfg(unix)]
#[test]
fn vertex_server_synthetic_partner_errors_echoing_the_token_leak_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let echo = format!("saw token {VERTEX_ACCESS_TOKEN} for {VERTEX_EMAIL}");
    let sse_frame = |value: serde_json::Value| format!("data: {value}\n\n");
    let claude_start = serde_json::json!({"type": "message_start", "message": {
        "id": "msg_vrtx_01Synthetic", "type": "message", "role": "assistant",
        "model": "claude-opus-5-5", "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 12, "output_tokens": 1}
    }});
    let claude_error =
        serde_json::json!({"type": "error", "error": {"type": "api_error", "message": echo}});
    let responses_failed = serde_json::json!({"type": "response.failed", "response": {
        "status": "failed", "error": {"code": "server_error", "message": echo}
    }});
    let responses_created = serde_json::json!({
        "type": "response.created", "response": {"id": "resp_synthetic", "status": "in_progress"}
    });
    let mistral_partial = serde_json::json!({
        "id": "synthetic-mistral", "object": "chat.completion.chunk", "model": "mistral-medium-3",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}, "finish_reason": null}]
    });
    let mistral_error = serde_json::json!({"error": {"message": echo}});
    let json = "application/json";
    let sse = "text/event-stream";
    let fixture = vertex_fixture(vec![
        (
            400,
            json,
            serde_json::json!({"type": "error", "error": {"type": "invalid_request_error", "message": echo}})
                .to_string(),
        ),
        (
            200,
            sse,
            format!("event: message_start\ndata: {claude_start}\n\nevent: error\ndata: {claude_error}\n\n"),
        ),
        (
            400,
            json,
            serde_json::json!({"error": {"message": echo, "type": "invalid_request_error", "code": null}})
                .to_string(),
        ),
        (200, sse, sse_frame(responses_failed.clone())),
        (
            200,
            sse,
            format!("{}{}", sse_frame(responses_created), sse_frame(responses_failed)),
        ),
        (
            400,
            json,
            serde_json::json!({"object": "error", "message": echo, "type": "invalid_request_error", "code": null})
                .to_string(),
        ),
        (200, sse, sse_frame(mistral_error.clone())),
        (
            200,
            sse,
            format!("{}{}", sse_frame(mistral_partial), sse_frame(mistral_error)),
        ),
    ]);

    let home = TempDir::new()?;
    let credentials = write_vertex_credentials(&home, &fixture);
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let mut child = vertex_server(
        &home,
        &fixture,
        port,
        &[
            ("GOOGLE_APPLICATION_CREDENTIALS", credentials.as_os_str()),
            ("CCP_TRAFFIC_LOG", std::ffi::OsStr::new("1")),
            (
                "CCP_VERTEX_LOCATION_OVERRIDES",
                std::ffi::OsStr::new("mistralai=us-central1"),
            ),
        ],
    )?;
    let requests = [
        ("vertex/anthropic/claude-opus-5-5", false),
        ("vertex/anthropic/claude-opus-5-5", true),
        ("vertex/xai/grok-4.7", false),
        ("vertex/xai/grok-4.7", false),
        ("vertex/xai/grok-4.7", true),
        ("vertex/mistralai/mistral-medium-3", false),
        ("vertex/mistralai/mistral-medium-3", false),
        ("vertex/mistralai/mistral-medium-3", true),
    ];
    let mut downstream = String::new();
    for (model, stream) in requests {
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]
        })
        .to_string();
        let response = post_raw(port, &body)?;
        if stream {
            assert!(response.starts_with("HTTP/1.1 200"), "{model}: {response}");
            assert!(response.contains("event: error"), "{model}: {response}");
        } else {
            assert!(!response.starts_with("HTTP/1.1 200"), "{model}: {response}");
            assert!(response.contains("[redacted]"), "{model}: {response}");
        }
        downstream.push_str(&response);
    }
    thread::sleep(Duration::from_millis(500));
    child.0.kill()?;
    child.0.wait()?;
    let stderr = std::fs::read_to_string(vertex_server_stderr(&home))?;

    // The scan below is only meaningful if every path was exercised: five
    // answers that are not 200 and three streams that end in an error.
    assert_eq!(
        downstream.matches("HTTP/1.1 200").count(),
        3,
        "{downstream}"
    );
    let state_dir = home.path().join("state").join("claude-code-proxy");
    let log = std::fs::read_to_string(state_dir.join("proxy.log"))?;
    assert_eq!(log.matches("\"request_failed\"").count(), 5, "{log}");
    assert!(log.contains("[redacted]"), "{log}");
    let errors = std::fs::read_dir(state_dir.join("errors"))?.count();
    assert_eq!(errors, 5);
    let mut names = Vec::new();
    let mut pending = vec![state_dir.join("traffic")];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                names.push(name.to_string());
            }
        }
    }
    for expected in [
        "020-upstream-request.json",
        "060-vertex-claude-messages-stream-error.json",
        "061-vertex-claude-messages-stream-summary.json",
        "060-vertex-responses-stream-error.json",
        "061-vertex-responses-stream-summary.json",
        "061-vertex-mistral-stream-summary.json",
        "061-vertex-buffered-summary.json",
    ] {
        assert!(
            names.iter().any(|name| name.ends_with(expected)),
            "missing {expected}: {names:?}"
        );
    }
    assert!(
        fixture
            .token_requests
            .load(std::sync::atomic::Ordering::SeqCst)
            >= 1
    );

    let files = all_files_text(&state_dir);
    for (place, text) in [
        ("state files", &files),
        ("stderr", &stderr),
        ("responses", &downstream),
    ] {
        for secret in [
            VERTEX_ACCESS_TOKEN,
            VERTEX_CLIENT_SECRET,
            VERTEX_REFRESH_TOKEN,
            VERTEX_EMAIL,
            "@example.com",
        ] {
            assert!(!text.contains(secret), "{place} leaked {secret}: {text}");
        }
    }
    Ok(())
}

/// `vertex auth status` with a usable credential and `extra`; returns the
/// exit code, stdout and stderr.
fn vertex_status_with(
    home: &TempDir,
    fixture: &VertexFixture,
    extra: &[(&str, &str)],
) -> Result<(Option<i32>, String, String), Box<dyn std::error::Error>> {
    let credentials = write_vertex_credentials(home, fixture);
    let mut command = vertex_command(home, &["vertex", "auth", "status"]);
    command
        .env("GOOGLE_APPLICATION_CREDENTIALS", &credentials)
        .env("CCP_VERTEX_PROJECT", "synthetic-project");
    for (name, value) in extra {
        command.env(name, value);
    }
    let output = command.output()?;
    Ok((
        output.status.code(),
        String::from_utf8(output.stdout)?,
        String::from_utf8(output.stderr)?,
    ))
}

#[test]
fn vertex_auth_status_synthetic_prints_the_location_overrides_with_their_source()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = vertex_fixture(Vec::new());

    let home = TempDir::new()?;
    let (code, stdout, stderr) = vertex_status_with(&home, &fixture, &[])?;
    assert_eq!(code, Some(0), "{stdout}\n{stderr}");
    assert!(
        stdout
            .lines()
            .any(|line| line == "Location overrides: none"),
        "{stdout}"
    );

    let home = TempDir::new()?;
    let (code, stdout, stderr) = vertex_status_with(
        &home,
        &fixture,
        &[(
            "CCP_VERTEX_LOCATION_OVERRIDES",
            "mistralai=europe-west4, anthropic/claude-opus-4-6=us-east5",
        )],
    )?;
    assert_eq!(code, Some(0), "{stdout}\n{stderr}");
    assert!(
        stdout.lines().any(|line| line
            == "Location overrides: mistralai=europe-west4, anthropic/claude-opus-4-6=us-east5 (CCP_VERTEX_LOCATION_OVERRIDES)"),
        "{stdout}"
    );

    // From config.json in the isolated config directory.
    let home = TempDir::new()?;
    let config_dir = home.path().join("config");
    std::fs::create_dir_all(&config_dir)?;
    std::fs::write(
        config_dir.join("config.json"),
        serde_json::json!({"vertex": {"locationOverrides": {"mistralai": "us-central1"}}})
            .to_string(),
    )?;
    let (code, stdout, stderr) = vertex_status_with(&home, &fixture, &[])?;
    assert_eq!(code, Some(0), "{stdout}\n{stderr}");
    assert!(
        stdout.lines().any(|line| line
            == "Location overrides: mistralai=us-central1 (vertex.locationOverrides in config.json)"),
        "{stdout}"
    );
    Ok(())
}

#[test]
fn vertex_auth_status_synthetic_malformed_location_override_exits_two_naming_the_setting()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = vertex_fixture(Vec::new());
    let home = TempDir::new()?;
    let (code, stdout, stderr) = vertex_status_with(
        &home,
        &fixture,
        &[("CCP_VERTEX_LOCATION_OVERRIDES", "mistralai")],
    )?;
    let expected = "Invalid Vertex setting CCP_VERTEX_LOCATION_OVERRIDES: entry \"mistralai\" has no location; an entry is <key>=<location> in the variable and \"<key>\": \"<location>\" in config.json";
    // A credential that works does not hide the setting: no request could
    // be sent with it.
    assert_eq!(code, Some(2), "{stdout}\n{stderr}");
    assert!(
        stdout
            .lines()
            .any(|line| line == format!("Location overrides: {expected}")),
        "{stdout}"
    );
    assert!(stdout.contains("Access token obtained: true"), "{stdout}");
    assert!(
        format!("{stdout}\n{stderr}").matches(expected).count() >= 2,
        "the failure itself names the setting too:\n{stdout}\n{stderr}"
    );
    Ok(())
}

/// The upstream echoes the access token where no `message` key holds it: in
/// text the model writes, in a Responses error's `param` and in a Mistral
/// error's `detail`. The client gets what the model wrote, but no file the
/// proxy writes (captures, log, error records) may hold the token.
#[cfg(unix)]
#[test]
fn vertex_server_synthetic_token_outside_message_fields_stays_out_of_every_file()
-> Result<(), Box<dyn std::error::Error>> {
    let echo = format!("the token is {VERTEX_ACCESS_TOKEN}");
    let sse_frame = |value: serde_json::Value| format!("data: {value}\n\n");
    let claude_events = [
        serde_json::json!({"type": "message_start", "message": {
            "id": "msg_vrtx_01Synthetic", "type": "message", "role": "assistant",
            "model": "claude-opus-5-5", "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 12, "output_tokens": 1}
        }}),
        serde_json::json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": echo}}),
        serde_json::json!({"type": "content_block_stop", "index": 0}),
        serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 5}}),
        serde_json::json!({"type": "message_stop"}),
    ];
    let claude_stream: String = claude_events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
    let claude_message = serde_json::json!({
        "id": "msg_vrtx_01Synthetic", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": echo}], "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 727, "output_tokens": 308}
    })
    .to_string();
    let responses_text: String = [
        serde_json::json!({"type": "response.created", "response": {"id": "resp_synthetic", "status": "in_progress"}}),
        serde_json::json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_up", "role": "assistant", "content": []}}),
        serde_json::json!({"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_up", "delta": echo}),
        serde_json::json!({"type": "response.completed", "response": {
            "id": "resp_synthetic", "status": "completed", "incomplete_details": null,
            "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        }}),
    ]
    .into_iter()
    .map(sse_frame)
    .collect();
    let responses_param = sse_frame(serde_json::json!({"type": "response.failed", "response": {
        "id": "resp_synthetic", "status": "failed",
        "error": {"code": "invalid_request_error", "message": "Invalid parameter", "param": echo}
    }}));
    let mistral_chunk = |delta: serde_json::Value, finish: Option<&str>| {
        serde_json::json!({
            "id": "synthetic-mistral", "object": "chat.completion.chunk", "model": "mistral-medium-3",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
    };
    let mistral_text = format!(
        "{}{}",
        sse_frame(mistral_chunk(
            serde_json::json!({"role": "assistant", "content": echo}),
            None
        )),
        sse_frame(mistral_chunk(serde_json::json!({}), Some("stop")))
    );
    let mistral_detail = sse_frame(serde_json::json!({
        "error": {"message": "Invalid request", "detail": echo}
    }));
    let json = "application/json";
    let sse = "text/event-stream";
    let fixture = vertex_fixture(vec![
        (200, sse, claude_stream),
        (200, json, claude_message),
        (200, sse, responses_text.clone()),
        (200, sse, responses_text),
        (200, sse, responses_param.clone()),
        (200, sse, responses_param),
        (200, sse, mistral_text.clone()),
        (200, sse, mistral_text),
        (200, sse, mistral_detail.clone()),
        (200, sse, mistral_detail),
    ]);

    let home = TempDir::new()?;
    let credentials = write_vertex_credentials(&home, &fixture);
    let port = TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let mut child = vertex_server(
        &home,
        &fixture,
        port,
        &[
            ("GOOGLE_APPLICATION_CREDENTIALS", credentials.as_os_str()),
            ("CCP_TRAFFIC_LOG", std::ffi::OsStr::new("1")),
            (
                "CCP_VERTEX_LOCATION_OVERRIDES",
                std::ffi::OsStr::new("mistralai=us-central1"),
            ),
        ],
    )?;
    // (model, stream, whether the model's text, and so the token, reaches
    // the client)
    let requests = [
        ("vertex/anthropic/claude-opus-5-5", true, true),
        ("vertex/anthropic/claude-opus-5-5", false, true),
        ("vertex/xai/grok-4.7", true, true),
        ("vertex/xai/grok-4.7", false, true),
        ("vertex/xai/grok-4.7", true, false),
        ("vertex/xai/grok-4.7", false, false),
        ("vertex/mistralai/mistral-medium-3", true, true),
        ("vertex/mistralai/mistral-medium-3", false, true),
        ("vertex/mistralai/mistral-medium-3", true, false),
        ("vertex/mistralai/mistral-medium-3", false, false),
    ];
    for (model, stream, answered) in requests {
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}]
        })
        .to_string();
        let response = post_raw(port, &body)?;
        if answered {
            // The echo did reach the proxy: the model's own text is the
            // client's to see.
            assert!(response.starts_with("HTTP/1.1 200"), "{model}: {response}");
            assert!(
                response.contains(VERTEX_ACCESS_TOKEN),
                "{model} stream={stream}: {response}"
            );
        } else if stream {
            assert!(response.starts_with("HTTP/1.1 200"), "{model}: {response}");
            assert!(response.contains("event: error"), "{model}: {response}");
        } else {
            assert!(response.starts_with("HTTP/1.1 502"), "{model}: {response}");
        }
        if !answered {
            assert!(
                !response.contains(VERTEX_ACCESS_TOKEN),
                "{model} stream={stream}: {response}"
            );
        }
    }
    thread::sleep(Duration::from_millis(500));
    child.0.kill()?;
    child.0.wait()?;

    let state_dir = home.path().join("state").join("claude-code-proxy");
    let mut names = Vec::new();
    let mut pending = vec![state_dir.join("traffic")];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                names.push(name.to_string());
            }
        }
    }
    // The scan below is only meaningful if the upstream events were
    // captured on every path.
    for expected in [
        "061-vertex-claude-messages-stream-summary.json",
        "032-upstream-response-body.json",
        "061-vertex-responses-stream-summary.json",
        "061-vertex-mistral-stream-summary.json",
        "061-vertex-buffered-summary.json",
    ] {
        assert!(
            names.iter().any(|name| name.ends_with(expected)),
            "missing {expected}: {names:?}"
        );
    }
    let files = all_files_text(&state_dir);
    assert!(files.contains("[redacted]"), "{files}");
    assert!(
        !files.contains(VERTEX_ACCESS_TOKEN),
        "a file the proxy wrote holds the token: {files}"
    );
    Ok(())
}
