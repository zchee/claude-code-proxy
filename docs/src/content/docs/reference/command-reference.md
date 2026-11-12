---
title: Command reference
description: Canonical claude-code-proxy command syntax for serving, monitoring, listing models, version output, and provider authentication.
---

Running `claude-code-proxy` without a subcommand is equivalent to `claude-code-proxy serve`.

## Global version commands

```sh
claude-code-proxy --version
claude-code-proxy -v
claude-code-proxy version
```

Each prints `claude-code-proxy <version>`.

## `serve`

```sh
claude-code-proxy serve [--port <PORT>] [--no-monitor]
```

Starts the local HTTP proxy and blocks until shutdown.

| Option | Behavior |
| --- | --- |
| `--port <PORT>` | Overrides `PORT`, `config.json`, and the default for this invocation. |
| `--no-monitor` | Uses plain output even when stdout is a terminal. |

The bind address comes from `CCP_BIND_ADDRESS` or `bindAddress`. Interactive stdout opens the monitor unless `--no-monitor` is present. Non-terminal stdout uses plain mode.

Plain mode continues collecting monitor history and supports separate dashboards. SIGTERM (Unix) and Ctrl-C request graceful service shutdown.

## `monitor`

```sh
claude-code-proxy monitor [--url <URL>]
```

Attach a read-only dashboard to a running proxy. The default URL is `http://127.0.0.1:<configured-port>`. No provider login is needed in the dashboard process. `q` and `Ctrl-C` detach without stopping the proxy; multiple dashboards are supported. After a connection failure, the dashboard retains its last snapshot and retries. The proxy accepts monitor reads only from loopback peers; use an SSH port forward for another machine.

## `demo`

```sh
claude-code-proxy demo
```

Opens the monitor with deterministic simulated traffic. It does not bind a network port or contact providers.

## `models`

```sh
claude-code-proxy models [--full]
```

Prints supported IDs grouped by provider. The default output compacts Cursor's runtime catalog. `--full` prints every Cursor alias.

## Provider authentication

The command shape is:

```text
claude-code-proxy <provider> auth <action>
```

| Provider | `login` | `device` | `status` | `logout` |
| --- | --- | --- | --- | --- |
| `codex` | Browser PKCE | Device code | Account, expiry, storage | Delete proxy credential |
| `kimi` | Device code | Unsupported | User, expiry, scope, storage | Delete proxy credential |
| `grok` | Browser PKCE | Device code | Expiry and storage | Delete proxy credential |
| `muse` | Device code | Device code | User, subscription, tier, usage windows, login age | Delete proxy credential only |
| `vertex` | Prints the gcloud command | Prints the gcloud command | Credential source and type, project, location, endpoint, location overrides, quota project, whether a token was obtained | Prints the gcloud command; deletes nothing |
| `cursor` | Browser polling flow | Unsupported | Source, claims, expiry | Delete proxy credential |

Examples:

```sh
claude-code-proxy codex auth login
claude-code-proxy grok auth device
claude-code-proxy kimi auth status
claude-code-proxy cursor auth logout
```

A missing credential makes `auth status` exit with status 1. For Muse, a stored login that Meta no longer accepts also exits with status 1 and prints the `claude-code-proxy muse auth login` hint. For Vertex AI, a missing or invalid project, location, base URL, or location override setting exits with status 2, even when credentials are also missing. Other provider command failures exit with status 2. Successful commands exit with status 0.

Logout removes the local proxy-owned credential. It does not call the provider to revoke a refresh token. `muse auth logout` also leaves the Muse CLI's login and the API key it shares with the proxy untouched. `muse auth status` contacts Meta's key service to report the current subscription and usage.

The proxy stores no Google credentials. `vertex auth login` and `vertex auth device` print `gcloud auth application-default login` and exit with status 2; `vertex auth logout` points at `gcloud auth application-default revoke` and exits with status 1. `vertex auth status` requests one access token through Application Default Credentials and never prints it. See [Vertex AI](/providers/vertex/#vertex-auth-status).

## Development commands

From a source checkout:

```sh
cargo run -- serve
cargo test --all
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
just check
just docs
```

`just docs` installs the locked documentation dependencies and starts the Astro development server on an available local port.
