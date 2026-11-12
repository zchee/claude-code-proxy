---
title: Troubleshooting
description: Diagnose listener, authentication, model routing, streaming, context, Cursor bundle, rate-limit, logging, and traffic-capture problems.
---

## Proxy is unreachable

Check the process and liveness route:

```sh
curl http://127.0.0.1:18765/healthz
```

A healthy listener returns `{"ok":true}`. Confirm `ANTHROPIC_BASE_URL` uses the same address and port. `claude-code-proxy serve --port 11435` and `PORT=11435` change the listener port. `--port` wins for that command.

## Authentication error

Check the provider selected by the model, then inspect its credential:

```sh
claude-code-proxy codex auth status
claude-code-proxy kimi auth status
claude-code-proxy grok auth status
claude-code-proxy muse auth status
claude-code-proxy vertex auth status
claude-code-proxy cursor auth status
```

Use that provider's login command when credentials are missing or expired. Codex requires ChatGPT subscription auth, not an OpenAI API key. Each provider uses proxy-owned credentials, except Vertex AI, which uses Google Application Default Credentials.
Muse logins do not refresh. When requests fail with `Muse login is no longer valid` or
`Muse rejected the API key`, run `claude-code-proxy muse auth login` again; a
running proxy picks up the new login without a restart. `muse auth status`
checks the login against Meta and also shows the subscription state and how
much of the 5-hour and weekly usage windows is used. A 403 from Muse links to
Meta's Accounts Center when the subscription is inactive. An HTTP 500 with
`Muse transport is unavailable: ...` means the proxy could not set up its Muse
HTTP clients, which it does on the first Muse request rather than at startup;
the message names the cause.
OpenCode Go has no `auth status` command; configure its API key through
`CCP_OPENCODE_API_KEY`, `OPENCODE_API_KEY`, or `opencode.apiKey` in
`config.json`.
Vertex AI has no login in the proxy. `vertex auth status` shows the credential
file and type, the project, location and endpoint with the setting each came
from, the quota project, and whether an access token was obtained. An HTTP 401
that starts with `Google credentials are not available` or `Google access token
could not be obtained` means Application Default Credentials found nothing
usable: run `gcloud auth application-default login` and retry, or set
`GOOGLE_APPLICATION_CREDENTIALS` to a key file and restart the proxy so it sees
the variable. Until a request has obtained an access token, every request looks
for the credentials again; after that, the proxy keeps them until it exits, so a
later login needs a restart. When gcloud runs with
`CLOUDSDK_CONFIG`, point `GOOGLE_APPLICATION_CREDENTIALS` at
`$CLOUDSDK_CONFIG/application_default_credentials.json`, because the lookup does
not follow that variable. An HTTP 500 with `Invalid Vertex configuration` names
the missing or invalid setting, usually the project.

## Vertex AI returns 401, 403, 404, or 429

Each message ends with `(upstream HTTP <status>)` and names what to check:

- **401**: Google rejected the access token. Run `gcloud auth application-default login` again, or check the key file named by `GOOGLE_APPLICATION_CREDENTIALS`, then restart the proxy: it keeps the credentials from the first Vertex AI request that obtained an access token.
- **403**: the account needs `roles/aiplatform.user` on the project, the Vertex AI API (`aiplatform.googleapis.com`) must be enabled there, and user credentials may need a quota project. Set one with `gcloud auth application-default set-quota-project PROJECT_ID`.
- **404**: the model is not available in that project and location. Check the model ID, then try `global` or a region where the model is offered, through `CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION`, or `vertex.location`.
- **429**: the project's quota or Google's shared capacity for the model in that location is used up. The response carries `retry-after` when Google sends a delay.

See [Vertex AI errors](/providers/vertex/#errors) for the full table.

## Model returns HTTP 400

Run:

```sh
claude-code-proxy models
claude-code-proxy models --full
```

An unknown local ID returns a catalog in the error. A known ID can still be rejected upstream when the account, subscription, or region lacks access. Cursor's prefixed form, such as `cursor:gpt-5.5`, forces Cursor routing and avoids cross-provider collisions.

## Background requests fail

Set `ANTHROPIC_SMALL_FAST_MODEL` to a concrete routable ID. Claude Code sends title and small background tasks through that model independently of the main model.

## A tool runs twice

Set `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1` for Claude Code. Retrying a partially completed stream as non-streaming can duplicate tool calls.

## Session reaches context limits

`[1m]` changes Claude Code's local compaction threshold and does not enlarge upstream context. Use a safe `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, remove `[1m]`, or compact earlier. For ChatGPT GPT-5.6 subscription models, `272000` is the documented setup value.

## Codex WebSocket fails

For a non-TUN local HTTP proxy, set both destination-scheme variables before starting the proxy:

| Variable | Value |
| --- | --- |
| `HTTP_PROXY` | `http://127.0.0.1:7890` |
| `HTTPS_PROXY` | `http://127.0.0.1:7890` |

Set them through the operating system, service manager, or shell, then start `claude-code-proxy serve` in the same environment.

The default `wss://chatgpt.com` connection uses `HTTPS_PROXY`; a working proxy should show `CONNECT chatgpt.com:443`. The `http://` value is normal: it describes how to reach the proxy, while `HTTPS_PROXY` describes which destinations use it. Restart claude-code-proxy after changing these variables because the client and pooled WebSocket route are created at startup.

Check `NO_PROXY` when the proxy sees no request. Proxy connection, authentication, or CONNECT failure is returned as an error and never retried directly. Environment variables are supported; OS proxy settings and PAC/WPAD discovery are not automatic.

Use HTTP SSE to isolate transport behavior:

```sh
CCP_CODEX_TRANSPORT=http claude-code-proxy serve
```

`auto` falls back only when WebSocket setup fails before sending the request. It does not replay an in-flight request.

## Cursor bundle cannot be found

Cursor needs the installed Cursor Agent JavaScript bundle for protobuf schemas. Confirm Cursor Agent is installed, then set:

```sh
CCP_CURSOR_AGENT_BUNDLE=/path/to/cursor-agent/index.js \
  claude-code-proxy serve
```

## Rate limited

Upstream limits are shared with other clients on the same account. Codex limit responses and Kimi HTTP 429 responses surface as HTTP 429 with `retry-after`. Muse HTTP 429 responses carry `retry-after` when Muse sends one as whole seconds, or when the latest usage report shows a full window, in which case it is the time until that window resets; either value is capped at 86400 seconds. Vertex AI HTTP 429 responses carry `retry-after` from Google's `RetryInfo` or a whole-second `Retry-After` header, capped at 86400 seconds. Wait for the indicated interval or reduce concurrent traffic.

## Find the complete error

The monitor shows request detail. `proxy.log` contains structured JSONL events, and `errors/` stores complete redacted failed-response payloads. A `request_failed` event includes the error file path.

For a source checkout, run the isolated helper:

```sh
./scripts/debug-proxy
```

It prints a random local URL, a sourceable `client.env`, and artifact paths. Keep the terminal open, reproduce once, then press Ctrl-C. Captures contain sensitive prompt and tool content.

## Focused traffic capture

```sh
CCP_LOG_VERBOSE=1 \
CCP_TRAFFIC_LOG=1 \
  claude-code-proxy serve --no-monitor
```

Traffic capture writes the inbound request, translated upstream request, upstream headers and events, and downstream events in emission order. Known credentials are redacted, but prompts and tool content are preserved. Disable capture and delete artifacts after diagnosis.
