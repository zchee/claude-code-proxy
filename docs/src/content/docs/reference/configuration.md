---
title: Configuration
description: Canonical claude-code-proxy configuration keys, environment variables, defaults, precedence, boolean parsing, and example config.json.
---

Proxy settings come from environment variables or `config.json`. Precedence is **environment variable, config file, built-in default**. The `serve --port` option takes precedence over all port settings.

These settings configure the proxy process. Claude Code client settings such as `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, and `CLAUDE_CODE_*` are documented separately in [Configure Claude Code](/using/configure-claude-code/).

## Example `config.json`

```json
{
  "bindAddress": "127.0.0.1",
  "port": 18765,
  "aliasProvider": "codex",
  "autoReviewModel": "gpt-5.6-terra",
  "codex": {
    "originator": "claude-code-proxy",
    "userAgent": "claude-code-proxy/0.1.24",
    "model": "gpt-6-sol",
    "effort": "high",
    "reasoningSummary": "auto",
    "serviceTier": "fast",
    "baseUrl": "https://chatgpt.com/backend-api/codex/responses",
    "transport": "auto",
    "headerTimeoutMs": 300000,
    "previousResponseId": false,
    "serverCompaction": false,
    "responsesApi": false,
    "fullLane": false,
    "imagesApi": false,
    "imagesBaseUrl": "https://chatgpt.com/backend-api/codex"
  },
  "kimi": {
    "userAgent": "KimiCLI/1.37.0",
    "oauthHost": "https://auth.kimi.com",
    "baseUrl": "https://api.kimi.com/coding/v1"
  },
  "grok": {
    "baseUrl": "https://cli-chat-proxy.grok.com/v1",
    "clientVersion": "1.0.13"
  },
  "muse": {
    "baseUrl": "https://api.meta.ai/v1",
    "oauthHost": "https://auth.meta.com",
    "mintBaseUrl": "https://api.meta.ai",
    "userAgent": "claude-code-proxy/0.1.48"
  },
  "opencode": {
    "apiKey": "YOUR_OPENCODE_GO_API_KEY",
    "baseUrl": "https://opencode.ai/zen/go/v1"
  },
  "vertex": {
    "project": "YOUR_GOOGLE_CLOUD_PROJECT_ID",
    "location": "global",
    "locationOverrides": {
      "mistralai": "us-central1"
    }
  },
  "cursor": {
    "baseUrl": "https://api2.cursor.sh",
    "clientVersion": "0.48.5",
    "agentBundle": "/path/to/cursor-agent/index.js"
  },
  "log": {
    "stderr": false,
    "verbose": false
  }
}
```

All keys are optional. An unreadable file, malformed JSON, or incompatible field type causes the file configuration to be ignored and built-in or environment values to apply.

## Core and diagnostics

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_BIND_ADDRESS` | `bindAddress` | `127.0.0.1` | Listener IP address. |
| `PORT` | `port` | `18765` | Listener port. |
| `CCP_CONFIG_DIR` | none | Platform config directory | Replaces the configuration and file-backed auth root. |
| `CCP_ALIAS_PROVIDER` | `aliasProvider` | `codex` | Routes recognized Anthropic-style aliases through `codex` or `kimi`. |
| `CCP_AUTO_REVIEW_MODEL` | `autoReviewModel` | `gpt-6-luna` for Codex | Routes Claude Code's non-streaming, tool-free Bash security-review classifier through a registered model. |
| `CCP_LOG_STDERR` | `log.stderr` | `false` | Mirrors logs to stderr when present in the environment, regardless of its value. |
| `CCP_LOG_VERBOSE` | `log.verbose` | `false` | Preserves full string fields in structured logs when present, regardless of its value. |
| `CCP_TRAFFIC_LOG` | none | `false` | Enables full request captures for `1`, `true`, or `yes`. |
| `XDG_STATE_HOME` | none | `~/.local/state` | State base on macOS and Linux. |

`CCP_CONFIG_DIR` affects `config.json` and file-backed provider auth. It does not relocate the state directory.

Auto-review routing resolves the classifier's requested model first, not the main conversation's model. Recognized Claude aliases follow Codex/Kimi session affinity or `aliasProvider`; Grok and other providers do not establish alias affinity. Codex auto-review classifier requests use `gpt-6-luna` by default. Requests routed through other providers retain their requested model. `CCP_AUTO_REVIEW_MODEL` or `autoReviewModel` selects an explicit registered model for all detected classifier requests without changing the session's provider affinity. Normal messages, streaming requests, tool-using requests, and token counting retain their requested model.

## Outbound proxies

Outbound HTTP requests and Codex WebSocket setup inherit standard proxy environment variables when each provider client is created:

| Environment | Applies to | Purpose |
| --- | --- | --- |
| `HTTP_PROXY` / `http_proxy` | `http://` and `ws://` destinations | Routes plain HTTP and WebSocket Upgrade requests through the configured proxy. |
| `HTTPS_PROXY` / `https_proxy` | `https://` and `wss://` destinations | Routes TLS destinations through the configured proxy, normally with HTTP CONNECT. |
| `ALL_PROXY` / `all_proxy` | Any destination without a scheme-specific proxy | Provides the fallback proxy. |
| `NO_PROXY` / `no_proxy` | Matching destination hosts and IPs | Bypasses proxy routing. Supports comma-separated domains, subdomains, IP addresses, CIDR ranges, and `*`. |

On case-sensitive platforms, uppercase names are checked before lowercase names when both spellings exist. Windows treats environment names as case-insensitive, so each uppercase/lowercase pair identifies one variable. Set these variables before starting claude-code-proxy; changing them requires a restart because clients and pooled WebSocket connections retain their startup route. For CGI safety, proxy environment variables are ignored when `REQUEST_METHOD` is present. The variable name describes the **destination** scheme, so this default-WSS configuration is valid even though the local proxy URL uses `http://`:

| Variable | Value |
| --- | --- |
| `HTTP_PROXY` | `http://127.0.0.1:7890` |
| `HTTPS_PROXY` | `http://127.0.0.1:7890` |

After setting both variables through the operating system, service manager, or shell, start `claude-code-proxy serve` in the same environment.

Proxy URLs may use `http`, `https`, `socks4`, `socks4a`, `socks5`, or `socks5h`. HTTP proxy URLs can contain percent-encoded Basic credentials, for example `http://user:password@127.0.0.1:7890`; SOCKS5 and SOCKS5H URLs can contain username/password credentials for the SOCKS handshake. SOCKS4 and SOCKS4A are supported without URL credentials. Prefer a secret-management mechanism when available because environment variables may be visible to other local processes. WSS certificate verification uses both bundled public roots and the platform native root store, including locally installed enterprise proxy CAs. Malformed or unsupported proxy URLs fail provider startup rather than being ignored. Proxy failures do not silently retry with a direct connection; only `NO_PROXY` selects direct routing. OS proxy settings, PAC files, and WPAD are not read automatically.

## Codex

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_CODEX_MODEL` | `codex.model` | unset | Forces every Codex request to one upstream model. |
| `CCP_CODEX_EFFORT` | `codex.effort` | unset | Forces `none`, `low`, `medium`, `high`, `xhigh`, or `max`. |
| `CCP_COMPACT_EFFORT` | none | `low` | Caps Codex reasoning effort for Claude Code summary compaction requests. `off` disables the cap and `none` removes reasoning. |
| `CCP_CODEX_REASONING_SUMMARY` | `codex.reasoningSummary` | unset | Overrides summary mode. `off` and `none` suppress summaries. |
| `CCP_CODEX_SERVICE_TIER` | `codex.serviceTier` | unset | Forces `fast`, `priority`, `ultrafast`, or `flex` on `/v1/messages` requests. Fast is sent as `priority`. Ultrafast is sent as `priority` to models that do not offer it. |
| `CCP_CODEX_BASE_URL` | `codex.baseUrl` | ChatGPT Codex Responses URL | Changes the Codex endpoint. |
| `CCP_CODEX_TRANSPORT` | `codex.transport` | `auto` | Selects `websocket`, `http`, or `auto`. `auto` falls back to HTTP SSE only when WebSocket setup fails before a request is sent. |
| `CCP_CODEX_HEADER_TIMEOUT_MS` | `codex.headerTimeoutMs` | `300000` | Sets how long an HTTP-transport request waits for the Codex response headers. Values below `1000` are ignored. |
| `CCP_CODEX_PREVIOUS_RESPONSE_ID` | `codex.previousResponseId` | `false` | Enables append-only WebSocket continuation for `1`, `true`, or `yes`. |
| `CCP_CODEX_SERVER_COMPACTION` | `codex.serverCompaction` | `false` | Enables or disables native compaction for standard boolean words. |
| `CCP_CODEX_RESPONSES_API` | `codex.responsesApi` | `false` | Enables `/v1/responses` and `/v1/chat/completions` for every registered provider. Accepts `1`, `true`, or `yes`. |
| `CCP_CODEX_FULL_LANE` | `codex.fullLane` | `false` | Enables parallel tool calls on the six [supported full-lane models](/providers/codex/#responses-lanes-and-parallel-tools). Luna and catalog-only models keep their default lane. Accepts `1`, `true`, or `yes`. |
| `CCP_CODEX_IMAGES_API` | `codex.imagesApi` | `false` | Enables `/v1/images/generations` and `/v1/images/edits` for `1`, `true`, or `yes`. |
| `CCP_CODEX_IMAGES_BASE_URL` | `codex.imagesBaseUrl` | `https://chatgpt.com/backend-api/codex` | Sets the trusted Codex Images API root; production use is restricted to HTTPS `chatgpt.com/backend-api/codex`. |
| `CCP_CODEX_ORIGINATOR` | `codex.originator` | `claude-code-proxy` | Changes the Codex `originator` header. |
| `CCP_CODEX_USER_AGENT` | `codex.userAgent` | `claude-code-proxy/<version>` | Changes the Codex user-agent. |

`CLAUDE_CODE_PROXY_CODEX_BASE_URL` remains an accepted fallback for the Codex base URL. `CCP_CODEX_BASE_URL` takes precedence.

## Kimi

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_KIMI_OAUTH_HOST` | `kimi.oauthHost` | `https://auth.kimi.com` | Changes the OAuth host. |
| `CCP_KIMI_BASE_URL` | `kimi.baseUrl` | `https://api.kimi.com/coding/v1` | Changes the API base URL. |
| `CCP_KIMI_USER_AGENT` | `kimi.userAgent` | `KimiCLI/1.37.0` | Changes the Kimi user-agent. |

## Grok

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_GROK_BASE_URL` | `grok.baseUrl` | `https://cli-chat-proxy.grok.com/v1` | Changes the Responses API base URL. |
| `CCP_GROK_CLIENT_VERSION` | `grok.clientVersion` | `1.0.13` | Changes the Grok client version header. |
| `CCP_GROK_TOOL_IMAGE` | none | `omit` | Selects `omit`, `reattach`, `inline`, or `reject` image handling. |
| `CCP_GROK_HOSTED_SEARCH` | none | off | Set to `1`, `on`, or `true` to let hosted search tools replace the caller's own search tools and force them on an explicit search turn. |
| `CCP_GROK_SEARCH_BLOCKS` | none | `text` | Selects how a hosted search is reported: `text` for a text block, `native` for `server_tool_use` plus a `*_tool_result` block. |

## Muse

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_MUSE_BASE_URL` | `muse.baseUrl` | `https://api.meta.ai/v1` | Changes the Responses API base URL. |
| `CCP_MUSE_OAUTH_HOST` | `muse.oauthHost` | `https://auth.meta.com` | Changes the device-login host. |
| `CCP_MUSE_MINT_BASE_URL` | `muse.mintBaseUrl` | `https://api.meta.ai` | Changes the host of the API key service, which lives at the host root rather than under `/v1`. |
| `CCP_MUSE_USER_AGENT` | `muse.userAgent` | `claude-code-proxy/<version>` | Changes the user-agent of every Muse request. The value must be 1 to 256 printable ASCII characters; an empty value counts as not set, since Meta's login host rejects requests without one. |

The three URL settings must use `https` (plain `http` only for a loopback host) and must not contain a user name or password. For all four settings an empty value counts as not set; a non-empty invalid value is an error that names the setting, not a fallback to the default. The key service also reports a base URL. The proxy uses only its path, and only when it is HTTPS and has the same host as `muse.baseUrl`; the scheme, host, and port stay those of `muse.baseUrl`. Otherwise it keeps `muse.baseUrl`.

## OpenCode Go

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_OPENCODE_API_KEY` | `opencode.apiKey` | unset | OpenCode Go API key; takes precedence over `OPENCODE_API_KEY` and config. |
| `OPENCODE_API_KEY` | `opencode.apiKey` | unset | Fallback API-key variable accepted by the proxy when the CCP-specific variable is unset. |
| `CCP_OPENCODE_BASE_URL` | `opencode.baseUrl` | `https://opencode.ai/zen/go/v1` | Changes the OpenCode Go API base URL. |

## LM Studio

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_LMSTUDIO_ENABLED` | `lmstudio.enabled` | `true` | `0`, `false`, `no`, or `off` turns the provider off. |
| `CCP_LMSTUDIO_BASE_URL` | `lmstudio.baseUrl` | `http://127.0.0.1:1234` | The LM Studio server: a scheme, a host, and a port, `https` unless the host is a loopback address. |
| `CCP_LMSTUDIO_API_KEY`, then `LM_API_TOKEN` | `lmstudio.apiKey` | unset | Token for a server with "Require Authentication" on. |

The model list is read from the server on every listing. While a server on a loopback address is stopped, the list comes from `lms ls`, found on `PATH` or in `~/.lmstudio/bin`. `claude-code-proxy models` reads these settings each time it runs; the proxy server reads them at startup. See [LM Studio](/providers/lmstudio/).

## Vertex AI

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_VERTEX_PROJECT` | `vertex.project` | unset | Google Cloud project that requests go to and are billed to. Required; there is no default. Takes precedence over `GOOGLE_CLOUD_PROJECT` and config. |
| `GOOGLE_CLOUD_PROJECT` | `vertex.project` | unset | Standard Google Cloud project variable, used when `CCP_VERTEX_PROJECT` is unset. |
| `CCP_VERTEX_LOCATION` | `vertex.location` | `global` | Vertex AI location, such as `global`, `us`, `eu`, or `us-central1`. Takes precedence over `GOOGLE_CLOUD_LOCATION` and config. |
| `GOOGLE_CLOUD_LOCATION` | `vertex.location` | `global` | Standard Google Cloud location variable, used when `CCP_VERTEX_LOCATION` is unset. |
| `CCP_VERTEX_BASE_URL` | `vertex.baseUrl` | Derived from the location | Replaces `https://<host>`, for a relay or a local test server. Its path prefix is kept and `/v1/projects/...` is appended, or `/v1beta1/projects/...` for the Llama 4 managed open models. |
| `CCP_VERTEX_LOCATION_OVERRIDES` | `vertex.locationOverrides` | unset | Locations for single publishers or models, as `<key>=<location>` entries separated by commas in the variable and an object of `"<key>": "<location>"` members in config. A key is `<publisher>` or `<publisher>/<model-id>`, such as `mistralai` or `anthropic/claude-opus-4-6`; Gemini uses the publisher `google`. The variable replaces the config value as a whole. |
| none | `vertex.endpoints` | unset | Endpoints you deployed, as an object of `"<name>": {"endpoint", "location", "model"}` members, each with optional `project`, `host`, and `tools`. A request names one as `vertex/endpoint/<name>`. See [Deployed endpoints](/providers/vertex/#deployed-endpoints). |

The precedence is `CCP_VERTEX_*`, then `GOOGLE_CLOUD_PROJECT` or `GOOGLE_CLOUD_LOCATION`, then `config.json`. An empty or blank value counts as not set. A project may contain letters, digits, `.`, `_`, `:`, and `-`, starting with a letter or digit; a location may contain lowercase letters, digits, and `-`. The base URL receives the Google access token, so it must use `https` (plain `http` only for a loopback host) and must not contain a user name or password. A non-empty invalid value is an error that names the setting, not a fallback to the default. The startup summary lists a project or location taken from the Google variables as `vertex.project (Google env)` or `vertex.location (Google env)`.

A location override names the model's location in place of `vertex.location`. An override that names the model wins over everything else; then Meta Muse models keep their fixed `global` and the Llama 4 managed open models their fixed `us-east5`, so a `meta` override moves neither; then an override that names the publisher applies; then the configured location. Mistral needs `us-central1` or `europe-west4` from an override or the configured location, and is a 400 otherwise. An override key that is malformed or listed twice, the key `endpoint`, an entry without a valid location, or a `vertex.locationOverrides` value that is not an object makes every Vertex AI request fail with a 500 naming the setting. The startup summary lists `vertex.locationOverrides (env)` or the file's entries. See [Location overrides](/providers/vertex/#location-overrides).

A `vertex.endpoints` entry needs `endpoint` (the endpoint ID), `location` (its region, never `global`), and `model` (the value the serving container accepts). `project` replaces the Vertex project for that endpoint, which must still be set. `host` is the endpoint's dedicated URL or DNS name; it receives the access token, so it must be an `https` host under `prediction.vertexai.goog` on the default port, or a loopback host, with no path, user name, or password. Without `host`, the proxy reads the endpoint once with `endpoints.get`. `"tools": false` refuses requests that carry tools. An invalid entry, or an unknown key in it, fails only requests to that endpoint, with a 500 naming `vertex.endpoints` and the endpoint. The startup summary lists the endpoint names.

The server reads these settings on the first Vertex AI request and keeps them until it exits, so restart the proxy after changing them. When they are invalid, for example because no project is set, nothing is kept, and the next request reads `config.json` again. A request that fails only because no access token could be obtained keeps the settings and leaves the credentials to be looked up again; see [Vertex AI](/providers/vertex/#project-and-location).

Credentials are not configured here. The Google library reads `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_QUOTA_PROJECT` itself; see [Vertex AI](/providers/vertex/#credentials).

## Cursor Agent

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_CURSOR_BASE_URL` | `cursor.baseUrl` | `https://api2.cursor.sh` | Changes the Cursor API base URL. |
| `CCP_CURSOR_CLIENT_VERSION` | `cursor.clientVersion` | `0.48.5` | Changes Cursor client version headers. |
| `CCP_CURSOR_AGENT_BUNDLE` | `cursor.agentBundle` | Auto-detected | Points to Cursor Agent's bundled `index.js` protobuf schemas. |
| `CCP_CURSOR_AUTH_TOKEN` | none | unset | Uses a bearer token instead of proxy-owned Cursor auth storage. |

## Shared compatibility fallbacks

`CCP_USER_AGENT` is the fallback when a Codex or Kimi provider-specific user-agent is absent. Prefer `CCP_CODEX_USER_AGENT` or `CCP_KIMI_USER_AGENT` in durable configuration.

Endpoint and client identity overrides can break provider compatibility. Use defaults unless a controlled integration or focused diagnostic requires an override.
