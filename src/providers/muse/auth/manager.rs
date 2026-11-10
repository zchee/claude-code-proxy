//! Server-side owner of the Muse API key. The key lives only in memory and is
//! minted on demand from the stored OAuth access token. There is no OAuth
//! refresh: once the mint answers 401 the login is unusable until the stored
//! access token changes.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::{BoxFuture, Shared};
use sha2::{Digest, Sha256};
use tokio::task::AbortHandle;
use tokio::time::Instant;

use super::constants::{LOGIN_COMMAND, MINT_TIMEOUT};
use super::mint::{self, MintError, MintTarget, MintedKey};
use super::token_store::{FileTokenStore, file_store};
use crate::logging::Logger;
use crate::providers::muse::usage::{LoginId, SubscriptionUsage, UsageSink};

/// Longest a caller waits for a key, counted from when it asked. It sits just
/// outside the mint timeout so a stalled mint reports its own failure first.
pub const CALLER_DEADLINE: Duration = Duration::from_secs(21);

/// Allowance figures older than this are not used to derive `retry-after`.
const USAGE_FRESHNESS: Duration = Duration::from_secs(600);

/// Mints a key for an access token. The production implementation is
/// [`HttpMinter`]; the trait exists so the single-flight logic can be driven
/// under tokio's paused clock, which cannot be combined with real sockets.
#[async_trait]
pub trait KeyMinter: Send + Sync {
    async fn mint(&self, access_token: &str) -> Result<MintedKey, MintError>;
}

pub struct HttpMinter {
    client: reqwest::Client,
    target: MintTarget,
}

impl HttpMinter {
    pub fn new(target: MintTarget, timeout: Duration) -> anyhow::Result<Self> {
        Ok(Self::with_client(
            mint::async_client_builder(timeout).build()?,
            target,
        ))
    }

    pub fn with_client(client: reqwest::Client, target: MintTarget) -> Self {
        Self { client, target }
    }
}

#[async_trait]
impl KeyMinter for HttpMinter {
    async fn mint(&self, access_token: &str) -> Result<MintedKey, MintError> {
        mint::mint(&self.client, &self.target, access_token).await
    }
}

/// The key a request runs with. A rejected key is identified by `login` and
/// `generation`, never by its bytes: the mint returns the same key string
/// every time.
#[derive(Clone, PartialEq, Eq)]
pub struct ActiveKey {
    pub login: LoginId,
    pub generation: u64,
    pub api_key: String,
    pub base_url: String,
    /// The access token the key was minted from, kept so upstream text can
    /// be scrubbed of both secrets.
    pub access_token: String,
}

impl ActiveKey {
    /// Values that must never leave the proxy in upstream-derived text.
    pub fn secrets(&self) -> [&str; 2] {
        [self.access_token.as_str(), self.api_key.as_str()]
    }
}

impl std::fmt::Debug for ActiveKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveKey")
            .field("login", &self.login)
            .field("generation", &self.generation)
            .field("api_key", &"[redacted]")
            .field("base_url", &self.base_url)
            .field("access_token", &"[redacted]")
            .finish()
    }
}

/// Why no key could be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthFailure {
    /// No stored login.
    NotAuthenticated,
    /// The mint failed, or failed earlier for this login (`Unauthorized`),
    /// or the caller's deadline passed (`Transient` without a status).
    Mint(MintError),
}

impl AuthFailure {
    pub fn upstream_status(&self) -> Option<http::StatusCode> {
        match self {
            AuthFailure::NotAuthenticated => None,
            AuthFailure::Mint(error) => error.upstream_status(),
        }
    }

    /// User-facing message, without the upstream status suffix.
    pub fn message(&self) -> String {
        match self {
            AuthFailure::NotAuthenticated => {
                format!("Not authenticated with Muse. Run: {LOGIN_COMMAND}")
            }
            AuthFailure::Mint(error) => error.message(),
        }
    }
}

/// SHA-256 of the access token; identifies a login without keeping the
/// token as a map key.
pub fn login_id(access_token: &str) -> LoginId {
    LoginId(Sha256::digest(access_token.as_bytes()).into())
}

#[derive(Clone)]
enum MintOutcome {
    Done {
        login: LoginId,
        result: Result<ActiveKey, MintError>,
    },
    /// Cancelled or superseded by a login change; waiters start over.
    Aborted,
}

type SharedMint = Shared<BoxFuture<'static, MintOutcome>>;

struct Inflight {
    login: LoginId,
    seq: u64,
    future: SharedMint,
    abort: AbortHandle,
}

struct UsageRecord {
    usage: SubscriptionUsage,
    received_at: Instant,
}

#[derive(Default)]
struct State {
    login: Option<LoginId>,
    obtained_at_ms: u64,
    key: Option<ActiveKey>,
    unusable: bool,
    subs_active: Option<bool>,
    inflight: Option<Inflight>,
    usage: Option<UsageRecord>,
    /// Only ever increases, also across login changes.
    last_generation: u64,
    last_mint_seq: u64,
    mint_attempts: u64,
}

impl State {
    fn forget_login(&mut self) {
        self.key = None;
        self.unusable = false;
        self.subs_active = None;
        self.usage = None;
        if let Some(inflight) = self.inflight.take() {
            inflight.abort.abort();
        }
    }
}

pub struct MuseAuthManager {
    store: FileTokenStore,
    minter: Arc<dyn KeyMinter>,
    mint_timeout: Duration,
    caller_deadline: Duration,
    state: Arc<Mutex<State>>,
    log: Option<Logger>,
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

impl MuseAuthManager {
    /// The server's manager: the configured mint, the stored login file.
    pub fn from_config() -> anyhow::Result<Self> {
        Self::new(file_store(), MintTarget::from_config()?)
    }

    pub fn new(store: FileTokenStore, target: MintTarget) -> anyhow::Result<Self> {
        let minter = Arc::new(HttpMinter::new(target, MINT_TIMEOUT)?);
        let mut manager = Self::with_minter(store, minter, MINT_TIMEOUT, CALLER_DEADLINE);
        manager.log = Some(crate::logging::create_logger("muse"));
        Ok(manager)
    }

    /// Manager with an explicit minter and timeouts and without logging.
    pub fn with_minter(
        store: FileTokenStore,
        minter: Arc<dyn KeyMinter>,
        mint_timeout: Duration,
        caller_deadline: Duration,
    ) -> Self {
        Self {
            store,
            minter,
            mint_timeout,
            caller_deadline,
            state: Arc::default(),
            log: None,
        }
    }

    pub fn store(&self) -> &FileTokenStore {
        &self.store
    }

    /// How many mints this manager has started.
    pub fn mint_attempts(&self) -> u64 {
        lock(&self.state).mint_attempts
    }

    /// The key for the current login, minting one when none is held.
    pub async fn get_key(&self) -> Result<ActiveKey, AuthFailure> {
        self.acquire(None).await
    }

    /// Called after the Model API answered 401 to the key identified by
    /// `login` and `generation`. Returns a key that is newer than the rejected
    /// one, minting at most once however many requests report the same key.
    pub async fn recover(&self, login: LoginId, generation: u64) -> Result<ActiveKey, AuthFailure> {
        self.acquire(Some((login, generation))).await
    }

    /// Whether the most recent mint for the current login reported that the
    /// account has no active subscription.
    pub fn subscription_inactive(&self) -> bool {
        lock(&self.state).subs_active == Some(false)
    }

    /// The most recent allowance figures for the current login.
    pub fn latest_usage(&self) -> Option<SubscriptionUsage> {
        lock(&self.state)
            .usage
            .as_ref()
            .map(|record| record.usage.clone())
    }

    /// Seconds until an exhausted allowance window resets, from figures at
    /// most ten minutes old. `now_unix` is the wall clock in Unix seconds.
    pub fn retry_after_secs(&self, now_unix: i64) -> Option<u64> {
        let state = lock(&self.state);
        let record = state.usage.as_ref()?;
        if record.received_at.elapsed() > USAGE_FRESHNESS {
            return None;
        }
        [record.usage.window.as_ref(), record.usage.weekly.as_ref()]
            .into_iter()
            .flatten()
            .filter(|window| window.used_percent >= 100 && window.resets_at > now_unix)
            .map(|window| window.resets_at.abs_diff(now_unix))
            .max()
    }

    async fn acquire(&self, rejected: Option<(LoginId, u64)>) -> Result<ActiveKey, AuthFailure> {
        let deadline = Instant::now() + self.caller_deadline;
        loop {
            // `timeout_at` returns an already-finished mint without looking at
            // the clock, so a loop that keeps rejoining one must check it here.
            if Instant::now() >= deadline {
                return Err(AuthFailure::Mint(MintError::Transient { status: None }));
            }
            let (login, access_token) = self.sync_login()?;
            let pending = {
                let mut state = lock(&self.state);
                if state.login != Some(login) {
                    None
                } else {
                    if state.unusable {
                        return Err(AuthFailure::Mint(MintError::Unauthorized));
                    }
                    if let Some((rejected_login, rejected_generation)) = rejected
                        && rejected_login == login
                        && state
                            .key
                            .as_ref()
                            .is_some_and(|key| key.generation <= rejected_generation)
                    {
                        state.key = None;
                    }
                    if let Some(key) = &state.key {
                        return Ok(key.clone());
                    }
                    Some(self.join_or_start(&mut state, login, access_token))
                }
            };
            let Some(pending) = pending else { continue };
            let Ok(outcome) = tokio::time::timeout_at(deadline, pending).await else {
                return Err(AuthFailure::Mint(MintError::Transient { status: None }));
            };
            // The login may have changed while waiting; a result minted for
            // another login is never handed out.
            let (current, _) = self.sync_login()?;
            if let MintOutcome::Done {
                login: minted_for,
                result,
            } = outcome
                && minted_for == current
            {
                return result.map_err(AuthFailure::Mint);
            }
        }
    }

    /// Re-reads the stored login. A missing file or a different access token
    /// drops everything held for the previous login, including its mint.
    fn sync_login(&self) -> Result<(LoginId, String), AuthFailure> {
        let stored = self.store.load_auth().ok().flatten();
        let mut state = lock(&self.state);
        let Some(auth) = stored else {
            state.forget_login();
            state.login = None;
            return Err(AuthFailure::NotAuthenticated);
        };
        let login = login_id(&auth.access_token);
        if state.login != Some(login) {
            state.forget_login();
            state.login = Some(login);
            state.obtained_at_ms = auth.obtained_at_ms;
        }
        Ok((login, auth.access_token))
    }

    fn join_or_start(&self, state: &mut State, login: LoginId, access_token: String) -> SharedMint {
        if let Some(inflight) = &state.inflight
            && inflight.login == login
        {
            return inflight.future.clone();
        }
        if let Some(stale) = state.inflight.take() {
            stale.abort.abort();
        }
        state.last_mint_seq += 1;
        state.mint_attempts += 1;
        let seq = state.last_mint_seq;
        // Spawned so the mint completes and records its result even when
        // every caller has gone away.
        let handle = tokio::spawn(mint_task(MintJob {
            state: self.state.clone(),
            minter: self.minter.clone(),
            access_token,
            login,
            seq,
            timeout: self.mint_timeout,
            log: self.log.clone(),
        }));
        let abort = handle.abort_handle();
        let slot = self.state.clone();
        let future = async move {
            match handle.await {
                Ok(outcome) => outcome,
                Err(error) if error.is_panic() => {
                    // The task never reached `finish_mint`, so its slot would
                    // otherwise stay occupied by this finished future forever.
                    let mut state = lock(&slot);
                    if state
                        .inflight
                        .as_ref()
                        .is_some_and(|inflight| inflight.seq == seq)
                    {
                        state.inflight = None;
                    }
                    MintOutcome::Done {
                        login,
                        result: Err(MintError::Transient { status: None }),
                    }
                }
                Err(_) => MintOutcome::Aborted,
            }
        }
        .boxed()
        .shared();
        state.inflight = Some(Inflight {
            login,
            seq,
            future: future.clone(),
            abort,
        });
        future
    }
}

impl UsageSink for MuseAuthManager {
    fn record_usage(&self, login: LoginId, usage: SubscriptionUsage) {
        let mut state = lock(&self.state);
        if state.login == Some(login) {
            state.usage = Some(UsageRecord {
                usage,
                received_at: Instant::now(),
            });
        }
    }
}

struct MintJob {
    state: Arc<Mutex<State>>,
    minter: Arc<dyn KeyMinter>,
    access_token: String,
    login: LoginId,
    seq: u64,
    timeout: Duration,
    log: Option<Logger>,
}

async fn mint_task(job: MintJob) -> MintOutcome {
    let result = tokio::time::timeout(job.timeout, job.minter.mint(&job.access_token))
        .await
        .unwrap_or(Err(MintError::Transient { status: None }));
    let (outcome, first_rejection) = {
        let mut state = lock(&job.state);
        finish_mint(&mut state, job.login, job.seq, job.access_token, result)
    };
    if let (Some(elapsed_secs), Some(log)) = (first_rejection, job.log) {
        // The access token's lifetime is unknown; this line is how it gets
        // observed in practice.
        let mut fields = serde_json::Map::new();
        fields.insert("elapsed_since_login_secs".into(), elapsed_secs.into());
        log.warn(
            "Muse key mint refused the stored login; a new login is required",
            Some(fields),
        );
    }
    outcome
}

/// Records a finished mint. The in-flight slot is cleared only when it still
/// belongs to this mint, and the result is kept only while the login it was
/// minted for is current. Returns the elapsed seconds since login when this
/// is the first 401 for the login.
fn finish_mint(
    state: &mut State,
    login: LoginId,
    seq: u64,
    access_token: String,
    result: Result<MintedKey, MintError>,
) -> (MintOutcome, Option<u64>) {
    if state
        .inflight
        .as_ref()
        .is_some_and(|inflight| inflight.seq == seq)
    {
        state.inflight = None;
    }
    if state.login != Some(login) {
        return (MintOutcome::Aborted, None);
    }
    let mut first_rejection = None;
    let result = match result {
        Ok(minted) => {
            state.last_generation += 1;
            let key = ActiveKey {
                login,
                generation: state.last_generation,
                api_key: minted.api_key,
                base_url: minted.base_url,
                access_token,
            };
            state.key = Some(key.clone());
            state.subs_active = Some(minted.is_subs_active);
            if let Some(usage) = minted.subs_usage {
                state.usage = Some(UsageRecord {
                    usage,
                    received_at: Instant::now(),
                });
            }
            Ok(key)
        }
        Err(MintError::Unauthorized) => {
            if !state.unusable {
                first_rejection = Some(now_ms().saturating_sub(state.obtained_at_ms) / 1000);
            }
            state.unusable = true;
            Err(MintError::Unauthorized)
        }
        Err(other) => Err(other),
    };
    (MintOutcome::Done { login, result }, first_rejection)
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
    use crate::providers::muse::auth::token_store::{StoredAuth, store_at};
    use crate::providers::muse::usage::UsageWindow;
    use futures_util::future::join_all;
    use http::StatusCode;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Semaphore;

    const TOKEN_A: &str = "synthetic-access-token-login-a";
    const TOKEN_B: &str = "synthetic-access-token-login-b";
    /// The mint returns the same key string on every call.
    const SAME_KEY: &str = "synthetic-minted-api-key-identical-on-re-mint";

    type Behaviour =
        Box<dyn Fn(String) -> BoxFuture<'static, Result<MintedKey, MintError>> + Send + Sync>;

    struct FnMinter {
        calls: AtomicUsize,
        tokens: Mutex<Vec<String>>,
        behaviour: Behaviour,
    }

    impl FnMinter {
        fn new(
            behaviour: impl Fn(String) -> BoxFuture<'static, Result<MintedKey, MintError>>
            + Send
            + Sync
            + 'static,
        ) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                tokens: Mutex::new(Vec::new()),
                behaviour: Box::new(behaviour),
            })
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl KeyMinter for FnMinter {
        async fn mint(&self, access_token: &str) -> Result<MintedKey, MintError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.tokens.lock().unwrap().push(access_token.to_string());
            (self.behaviour)(access_token.to_string()).await
        }
    }

    fn observed_usage() -> SubscriptionUsage {
        SubscriptionUsage {
            window: Some(UsageWindow {
                used_percent: 1,
                resets_at: 1794291326,
                window_duration_mins: Some(300),
            }),
            weekly: Some(UsageWindow {
                used_percent: 0,
                resets_at: 1794787200,
                window_duration_mins: None,
            }),
        }
    }

    fn minted(api_key: &str) -> MintedKey {
        MintedKey {
            api_key: api_key.to_string(),
            base_url: "http://127.0.0.1:9/v1".into(),
            is_subs_active: true,
            subs_tier_name: Some("Muse Code Max Usage".into()),
            subs_usage: Some(observed_usage()),
            require_payment: Some(false),
            can_subscribe: Some(false),
            user_full_name: Some("Synthetic Person".into()),
        }
    }

    fn always(result: Result<MintedKey, MintError>) -> Arc<FnMinter> {
        FnMinter::new(move |_| {
            let result = result.clone();
            async move { result }.boxed()
        })
    }

    /// Each mint waits for one permit, so the test decides when it ends.
    fn gated(gate: Arc<Semaphore>, result: Result<MintedKey, MintError>) -> Arc<FnMinter> {
        FnMinter::new(move |_| {
            let gate = gate.clone();
            let result = result.clone();
            async move {
                gate.acquire().await.unwrap().forget();
                result
            }
            .boxed()
        })
    }

    fn stalled() -> Arc<FnMinter> {
        FnMinter::new(|_| futures_util::future::pending().boxed())
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        path: String,
    }

    impl Fixture {
        fn new(token: Option<&str>) -> Self {
            let dir = tempfile::TempDir::new().unwrap();
            let path = dir
                .path()
                .join("muse")
                .join("auth.json")
                .to_string_lossy()
                .into_owned();
            let fixture = Self { _dir: dir, path };
            if let Some(token) = token {
                fixture.login(token);
            }
            fixture
        }

        fn login(&self, token: &str) {
            store_at(self.path.clone())
                .save_auth(StoredAuth {
                    access_token: token.to_string(),
                    obtained_at_ms: 1_794_274_189_000,
                    client_id: "1031625952748946".into(),
                    oauth_host: "https://auth.meta.com".into(),
                    user_full_name: None,
                    is_subs_active: true,
                    subs_tier_name: None,
                })
                .unwrap();
        }

        fn logout(&self) {
            store_at(self.path.clone()).clear_auth().unwrap();
        }

        fn manager(&self, minter: Arc<FnMinter>) -> Arc<MuseAuthManager> {
            Arc::new(MuseAuthManager::with_minter(
                store_at(self.path.clone()),
                minter,
                MINT_TIMEOUT,
                CALLER_DEADLINE,
            ))
        }
    }

    async fn settle() {
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn held_key_is_reused_without_minting() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = always(Ok(minted(SAME_KEY)));
        let manager = fixture.manager(minter.clone());
        let first = manager.get_key().await.unwrap();
        let second = manager.get_key().await.unwrap();
        assert_eq!(first, second);
        assert_eq!(first.generation, 1);
        assert_eq!(first.login, login_id(TOKEN_A));
        assert_eq!(first.api_key, SAME_KEY);
        assert_eq!(first.secrets(), [TOKEN_A, SAME_KEY]);
        assert_eq!(minter.calls(), 1);
        assert_eq!(*minter.tokens.lock().unwrap(), [TOKEN_A]);
    }

    #[tokio::test(start_paused = true)]
    async fn ten_concurrent_callers_share_one_mint() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let gate = Arc::new(Semaphore::new(0));
        let minter = gated(gate.clone(), Ok(minted(SAME_KEY)));
        let manager = fixture.manager(minter.clone());
        let callers: Vec<_> = (0..10)
            .map(|_| {
                let manager = manager.clone();
                tokio::spawn(async move { manager.get_key().await })
            })
            .collect();
        settle().await;
        gate.add_permits(1);
        for caller in join_all(callers).await {
            assert_eq!(caller.unwrap().unwrap().generation, 1);
        }
        assert_eq!(minter.calls(), 1);
        assert_eq!(manager.mint_attempts(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn staggered_401s_for_one_key_mint_once_even_when_the_key_string_repeats() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = always(Ok(minted(SAME_KEY)));
        let manager = fixture.manager(minter.clone());
        let rejected = manager.get_key().await.unwrap();
        assert_eq!(rejected.generation, 1);

        let first = manager
            .recover(rejected.login, rejected.generation)
            .await
            .unwrap();
        assert_eq!(first.generation, 2);
        assert_eq!(first.api_key, rejected.api_key);
        for _ in 0..9 {
            let late = manager
                .recover(rejected.login, rejected.generation)
                .await
                .unwrap();
            assert_eq!(late.generation, 2);
        }
        assert_eq!(minter.calls(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_401s_for_one_key_share_one_re_mint() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let gate = Arc::new(Semaphore::new(1));
        let minter = gated(gate.clone(), Ok(minted(SAME_KEY)));
        let manager = fixture.manager(minter.clone());
        let rejected = manager.get_key().await.unwrap();
        let callers: Vec<_> = (0..10)
            .map(|_| {
                let manager = manager.clone();
                let rejected = rejected.clone();
                tokio::spawn(
                    async move { manager.recover(rejected.login, rejected.generation).await },
                )
            })
            .collect();
        settle().await;
        gate.add_permits(1);
        for caller in join_all(callers).await {
            assert_eq!(caller.unwrap().unwrap().generation, 2);
        }
        assert_eq!(minter.calls(), 2);
    }

    /// First mint succeeds; every later one waits for a permit and then
    /// returns `later`.
    fn first_ok_then_gated(gate: Arc<Semaphore>, later: MintError) -> Arc<FnMinter> {
        let calls = AtomicUsize::new(0);
        FnMinter::new(move |_| {
            let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
            let gate = gate.clone();
            let later = later.clone();
            async move {
                if first {
                    return Ok(minted(SAME_KEY));
                }
                gate.acquire().await.unwrap().forget();
                Err(later)
            }
            .boxed()
        })
    }

    #[tokio::test(start_paused = true)]
    async fn re_mint_answering_401_fails_everyone_once_and_marks_the_login_unusable() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let gate = Arc::new(Semaphore::new(0));
        let minter = first_ok_then_gated(gate.clone(), MintError::Unauthorized);
        let manager = fixture.manager(minter.clone());
        let rejected = manager.get_key().await.unwrap();
        let callers: Vec<_> = (0..10)
            .map(|_| {
                let manager = manager.clone();
                let rejected = rejected.clone();
                tokio::spawn(
                    async move { manager.recover(rejected.login, rejected.generation).await },
                )
            })
            .collect();
        settle().await;
        gate.add_permits(1);
        for caller in join_all(callers).await {
            assert_eq!(
                caller.unwrap().unwrap_err(),
                AuthFailure::Mint(MintError::Unauthorized)
            );
        }
        assert_eq!(minter.calls(), 2);
        assert_eq!(
            manager.get_key().await.unwrap_err(),
            AuthFailure::Mint(MintError::Unauthorized)
        );
        assert_eq!(minter.calls(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn mint_401_stops_upstream_calls_until_the_stored_token_changes() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = FnMinter::new(|token| {
            async move {
                if token == TOKEN_A {
                    Err(MintError::Unauthorized)
                } else {
                    Ok(minted(SAME_KEY))
                }
            }
            .boxed()
        });
        let manager = fixture.manager(minter.clone());
        for _ in 0..6 {
            let failure = manager.get_key().await.unwrap_err();
            assert_eq!(failure, AuthFailure::Mint(MintError::Unauthorized));
            assert_eq!(
                failure.message(),
                "Muse login is no longer valid. Run: claude-code-proxy muse auth login"
            );
            assert_eq!(failure.upstream_status(), Some(StatusCode::UNAUTHORIZED));
        }
        assert_eq!(minter.calls(), 1);

        fixture.login(TOKEN_B);
        let key = manager.get_key().await.unwrap();
        assert_eq!(key.login, login_id(TOKEN_B));
        assert_eq!(minter.calls(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_mint_fails_all_ten_callers_once_after_20_seconds() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = stalled();
        let manager = fixture.manager(minter.clone());
        let started = Instant::now();
        let results = join_all((0..10).map(|_| async {
            let result = manager.get_key().await;
            (result, started.elapsed())
        }))
        .await;
        for (result, elapsed) in results {
            assert_eq!(
                result.unwrap_err(),
                AuthFailure::Mint(MintError::Transient { status: None })
            );
            assert_eq!(elapsed, MINT_TIMEOUT);
            assert!(elapsed <= CALLER_DEADLINE);
        }
        assert_eq!(minter.calls(), 1);

        let failure = manager.get_key().await.unwrap_err();
        assert_eq!(failure.message(), "Muse key service is unreachable");
        assert_eq!(failure.upstream_status(), None);
        assert_eq!(minter.calls(), 2);
    }

    #[test]
    fn synthetic_panicking_minter_fails_callers_and_a_later_call_mints_again() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let calls = AtomicUsize::new(0);
        let minter = FnMinter::new(move |_| {
            let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if first {
                    panic!("synthetic minter failure");
                }
                Ok(minted(SAME_KEY))
            }
            .boxed()
        });
        let manager = Arc::new(MuseAuthManager::with_minter(
            store_at(fixture.path.clone()),
            minter.clone(),
            Duration::from_secs(2),
            Duration::from_secs(3),
        ));
        // The callers run on their own thread: a caller that spins on the
        // finished mint never yields, and the bounded wait below then fails
        // the test instead of hanging it.
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let outcome = runtime.block_on(async {
                let callers = join_all((0..4).map(|_| manager.get_key())).await;
                let later = manager.get_key().await;
                (callers, later)
            });
            let _ = sender.send(outcome);
        });
        let (callers, later) = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("callers return within their deadline");
        for result in callers {
            assert_eq!(
                result.unwrap_err(),
                AuthFailure::Mint(MintError::Transient { status: None })
            );
        }
        assert_eq!(later.unwrap().api_key, SAME_KEY);
        assert_eq!(minter.calls(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn rejected_mint_is_not_remembered() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let calls = AtomicUsize::new(0);
        let minter = FnMinter::new(move |_| {
            let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                if first {
                    Err(MintError::Rejected {
                        status: StatusCode::BAD_REQUEST,
                        detail: "Request body is required but was empty or null".into(),
                    })
                } else {
                    Ok(minted(SAME_KEY))
                }
            }
            .boxed()
        });
        let manager = fixture.manager(minter.clone());
        let failure = manager.get_key().await.unwrap_err();
        assert!(matches!(
            failure,
            AuthFailure::Mint(MintError::Rejected { status, .. }) if status == StatusCode::BAD_REQUEST
        ));
        assert_eq!(manager.get_key().await.unwrap().generation, 1);
        assert_eq!(minter.calls(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn missing_login_file_is_not_authenticated() {
        let fixture = Fixture::new(None);
        let minter = always(Ok(minted(SAME_KEY)));
        let manager = fixture.manager(minter.clone());
        let failure = manager.get_key().await.unwrap_err();
        assert_eq!(failure, AuthFailure::NotAuthenticated);
        assert_eq!(
            failure.message(),
            "Not authenticated with Muse. Run: claude-code-proxy muse auth login"
        );
        assert_eq!(failure.upstream_status(), None);

        fixture.login(TOKEN_A);
        let key = manager.get_key().await.unwrap();
        fixture.logout();
        assert_eq!(
            manager
                .recover(key.login, key.generation)
                .await
                .unwrap_err(),
            AuthFailure::NotAuthenticated
        );
        assert_eq!(minter.calls(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn recover_after_re_login_returns_the_new_login_key() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = FnMinter::new(|token| {
            async move {
                Ok(minted(if token == TOKEN_A {
                    "synthetic-key-a"
                } else {
                    "synthetic-key-b"
                }))
            }
            .boxed()
        });
        let manager = fixture.manager(minter.clone());
        let old = manager.get_key().await.unwrap();
        fixture.login(TOKEN_B);
        let new = manager.recover(old.login, old.generation).await.unwrap();
        assert_eq!(new.login, login_id(TOKEN_B));
        assert_eq!(new.api_key, "synthetic-key-b");
        assert_eq!(minter.calls(), 2);

        // A late 401 for the previous login's key must not discard the new key.
        let again = manager.recover(old.login, old.generation).await.unwrap();
        assert_eq!(again, new);
        assert_eq!(minter.calls(), 2);
    }

    /// Minter whose login-A mints wait for `gate_a` and end with `a_result`,
    /// setting `a_finished` only if they run to completion; login-B mints
    /// answer at once with key b.
    fn a_gated_b_immediate(
        gate_a: Arc<Semaphore>,
        a_finished: Arc<std::sync::atomic::AtomicBool>,
        a_result: Result<MintedKey, MintError>,
    ) -> Arc<FnMinter> {
        FnMinter::new(move |token| {
            let gate_a = gate_a.clone();
            let a_finished = a_finished.clone();
            let a_result = a_result.clone();
            async move {
                if token == TOKEN_A {
                    gate_a.acquire().await.unwrap().forget();
                    a_finished.store(true, Ordering::SeqCst);
                    a_result
                } else {
                    Ok(minted("synthetic-key-b"))
                }
            }
            .boxed()
        })
    }

    #[tokio::test(start_paused = true)]
    async fn re_login_during_a_mint_aborts_it_and_waiters_get_the_new_login_result() {
        for a_result in [Ok(minted("synthetic-key-a")), Err(MintError::Unauthorized)] {
            let fixture = Fixture::new(Some(TOKEN_A));
            let gate_a = Arc::new(Semaphore::new(0));
            let a_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let minter = a_gated_b_immediate(gate_a.clone(), a_finished.clone(), a_result);
            let manager = fixture.manager(minter.clone());
            let waiting_on_a = {
                let manager = manager.clone();
                tokio::spawn(async move { manager.get_key().await })
            };
            settle().await;
            fixture.login(TOKEN_B);
            let b = manager.get_key().await.unwrap();
            assert_eq!(b.api_key, "synthetic-key-b");
            // Releasing A now must have no effect: its mint was aborted.
            gate_a.add_permits(1);
            let a_waiter = waiting_on_a.await.unwrap().unwrap();
            settle().await;
            assert_eq!(a_waiter, b);
            assert!(!a_finished.load(Ordering::SeqCst));
            assert_eq!(minter.calls(), 2);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn mint_finishing_after_a_re_login_is_discarded_by_its_waiters() {
        for a_result in [Ok(minted("synthetic-key-a")), Err(MintError::Unauthorized)] {
            let fixture = Fixture::new(Some(TOKEN_A));
            let gate_a = Arc::new(Semaphore::new(0));
            let a_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let minter = a_gated_b_immediate(gate_a.clone(), a_finished.clone(), a_result);
            let manager = fixture.manager(minter.clone());
            let waiting_on_a = {
                let manager = manager.clone();
                tokio::spawn(async move { manager.get_key().await })
            };
            settle().await;
            // The file changes, but no request has noticed yet when A ends.
            fixture.login(TOKEN_B);
            gate_a.add_permits(1);
            let key = waiting_on_a.await.unwrap().unwrap();
            assert!(a_finished.load(Ordering::SeqCst));
            assert_eq!(key.login, login_id(TOKEN_B));
            assert_eq!(key.api_key, "synthetic-key-b");
            assert_eq!(minter.calls(), 2);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn waiting_across_a_re_login_never_exceeds_21_seconds() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = stalled();
        let manager = fixture.manager(minter.clone());
        let started = Instant::now();
        let waiting_on_a = {
            let manager = manager.clone();
            tokio::spawn(async move {
                let result = manager.get_key().await;
                (result, started.elapsed())
            })
        };
        tokio::time::sleep(Duration::from_secs(15)).await;
        fixture.login(TOKEN_B);
        let b_started = Instant::now();
        let b_result = manager.get_key().await;
        let (a_result, a_elapsed) = waiting_on_a.await.unwrap();
        assert_eq!(
            a_result.unwrap_err(),
            AuthFailure::Mint(MintError::Transient { status: None })
        );
        assert_eq!(a_elapsed, CALLER_DEADLINE);
        assert_eq!(
            b_result.unwrap_err(),
            AuthFailure::Mint(MintError::Transient { status: None })
        );
        assert_eq!(b_started.elapsed(), MINT_TIMEOUT);
        assert_eq!(minter.calls(), 2);
    }

    #[test]
    fn stale_mint_completion_leaves_the_newer_inflight_slot() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let login = login_id(TOKEN_A);
            let pending: SharedMint = futures_util::future::pending().boxed().shared();
            let placeholder = tokio::spawn(futures_util::future::pending::<()>());
            let mut state = State {
                login: Some(login),
                inflight: Some(Inflight {
                    login,
                    seq: 2,
                    future: pending,
                    abort: placeholder.abort_handle(),
                }),
                ..State::default()
            };
            let (outcome, _) = finish_mint(
                &mut state,
                login,
                1,
                TOKEN_A.to_string(),
                Ok(minted(SAME_KEY)),
            );
            assert!(matches!(outcome, MintOutcome::Done { .. }));
            assert_eq!(
                state.inflight.as_ref().map(|inflight| inflight.seq),
                Some(2)
            );
            let (_, _) = finish_mint(
                &mut state,
                login,
                2,
                TOKEN_A.to_string(),
                Ok(minted(SAME_KEY)),
            );
            assert!(state.inflight.is_none());
            placeholder.abort();
        });
    }

    #[tokio::test(start_paused = true)]
    async fn usage_from_a_previous_login_is_ignored() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = FnMinter::new(|token| {
            async move {
                let mut key = minted(&format!("key-for-{token}"));
                key.subs_usage = None;
                Ok(key)
            }
            .boxed()
        });
        let manager = fixture.manager(minter.clone());
        manager.get_key().await.unwrap();
        manager.record_usage(login_id(TOKEN_A), observed_usage());
        assert_eq!(manager.latest_usage(), Some(observed_usage()));

        fixture.login(TOKEN_B);
        manager.get_key().await.unwrap();
        assert_eq!(manager.latest_usage(), None);
        manager.record_usage(login_id(TOKEN_A), observed_usage());
        assert_eq!(manager.latest_usage(), None);
        manager.record_usage(login_id(TOKEN_B), observed_usage());
        assert_eq!(manager.latest_usage(), Some(observed_usage()));
    }

    #[tokio::test(start_paused = true)]
    async fn generation_keeps_increasing_across_login_changes() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let minter = always(Ok(minted(SAME_KEY)));
        let manager = fixture.manager(minter.clone());
        assert_eq!(manager.get_key().await.unwrap().generation, 1);
        fixture.login(TOKEN_B);
        assert_eq!(manager.get_key().await.unwrap().generation, 2);
        fixture.login(TOKEN_A);
        assert_eq!(manager.get_key().await.unwrap().generation, 3);
        fixture.logout();
        assert!(manager.get_key().await.is_err());
        fixture.login(TOKEN_A);
        assert_eq!(manager.get_key().await.unwrap().generation, 4);
    }

    fn exhausted(
        window_percent: i64,
        window_reset: i64,
        weekly_percent: i64,
        weekly_reset: i64,
    ) -> SubscriptionUsage {
        SubscriptionUsage {
            window: Some(UsageWindow {
                used_percent: window_percent,
                resets_at: window_reset,
                window_duration_mins: Some(300),
            }),
            weekly: Some(UsageWindow {
                used_percent: weekly_percent,
                resets_at: weekly_reset,
                window_duration_mins: None,
            }),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retry_after_synthetic_comes_from_fresh_exhausted_windows_only() {
        let now = 1794280000;
        let fixture = Fixture::new(Some(TOKEN_A));
        let manager = fixture.manager(always(Ok(minted(SAME_KEY))));
        let login = manager.get_key().await.unwrap().login;

        // The observed figures (1 % and 0 %) are not exhausted.
        assert_eq!(manager.retry_after_secs(now), None);

        manager.record_usage(login, exhausted(100, now + 600, 0, now + 9000));
        assert_eq!(manager.retry_after_secs(now), Some(600));

        // Both exhausted: the later reset is the one that frees the account.
        manager.record_usage(login, exhausted(100, now + 600, 100, now + 9000));
        assert_eq!(manager.retry_after_secs(now), Some(9000));

        // A reset in the past does not count.
        manager.record_usage(login, exhausted(100, now - 1, 0, now + 9000));
        assert_eq!(manager.retry_after_secs(now), None);

        manager.record_usage(login, exhausted(100, now + 600, 0, now + 9000));
        tokio::time::advance(Duration::from_secs(600)).await;
        assert_eq!(manager.retry_after_secs(now), Some(600));
        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(manager.retry_after_secs(now), None);
    }

    #[tokio::test(start_paused = true)]
    async fn inactive_subscription_is_reported_from_the_latest_mint() {
        let fixture = Fixture::new(Some(TOKEN_A));
        let mut inactive = minted(SAME_KEY);
        inactive.is_subs_active = false;
        let manager = fixture.manager(always(Ok(inactive)));
        assert!(!manager.subscription_inactive());
        manager.get_key().await.unwrap();
        assert!(manager.subscription_inactive());
    }

    #[test]
    fn active_key_debug_hides_both_secrets() {
        let key = ActiveKey {
            login: login_id(TOKEN_A),
            generation: 1,
            api_key: SAME_KEY.into(),
            base_url: "https://api.meta.ai/v1".into(),
            access_token: TOKEN_A.into(),
        };
        let rendered = format!("{key:?}");
        assert!(!rendered.contains(SAME_KEY), "{rendered}");
        assert!(!rendered.contains(TOKEN_A), "{rendered}");
    }

    #[tokio::test]
    async fn ten_concurrent_callers_cause_one_http_mint() {
        use crate::providers::muse::auth::mint::tests::{observed_mint_body, spawn_mint, target};
        let fixture = Fixture::new(Some(TOKEN_A));
        let (base, log) =
            spawn_mint(StatusCode::OK, vec![], observed_mint_body().to_string()).await;
        let minter = Arc::new(HttpMinter::with_client(
            mint::async_client_builder(MINT_TIMEOUT)
                .no_proxy()
                .build()
                .unwrap(),
            target(&base),
        ));
        let manager = MuseAuthManager::with_minter(
            store_at(fixture.path.clone()),
            minter,
            MINT_TIMEOUT,
            CALLER_DEADLINE,
        );
        let results = join_all((0..10).map(|_| manager.get_key())).await;
        for result in results {
            let key = result.unwrap();
            assert_eq!(key.generation, 1);
            assert_eq!(key.api_key, mint::tests::API_KEY);
        }
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(
            log[0].headers["authorization"],
            format!("Bearer {TOKEN_A}").as_str()
        );
    }
}
