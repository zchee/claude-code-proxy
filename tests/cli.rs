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
