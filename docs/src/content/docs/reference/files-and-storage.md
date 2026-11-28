---
title: Files and storage
description: Canonical claude-code-proxy configuration, credential, device ID, log, error, traffic-capture, and service-log paths on macOS, Linux, and Windows.
---

claude-code-proxy separates configuration and credentials from runtime state.

## Directory roots

| Platform | Configuration root | State root |
| --- | --- | --- |
| macOS | `~/.config/claude-code-proxy` | `${XDG_STATE_HOME:-~/.local/state}/claude-code-proxy` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/claude-code-proxy` | `${XDG_STATE_HOME:-~/.local/state}/claude-code-proxy` |
| Windows | `%APPDATA%/claude-code-proxy` | `%LOCALAPPDATA%/claude-code-proxy` |

Windows falls back to `%USERPROFILE%/AppData/Roaming` and `%USERPROFILE%/AppData/Local` when the corresponding environment variable is absent.

`CCP_CONFIG_DIR` replaces the configuration root for the current process. It does not change the state root.

## Configuration

`config.json` lives directly under the configuration root. See [Configuration](/reference/configuration/) for its schema and precedence.

## Provider credentials

On macOS, Codex and Cursor use Keychain services:

- `claude-code-proxy.codex`
- `claude-code-proxy.cursor`

Kimi, Grok, Muse, and the Google AI Ultra plan use `<configuration-root>/<provider>/auth.json` on every platform. The Muse file holds the Meta login token, the display name, the subscription state, and the tier name; the API key and e-mail address are not written to disk. Codex and Cursor use the same file layout on Linux and Windows. File-backed credentials are written with restrictive permissions where supported.

The Google AI Ultra plan stores access and refresh tokens, access-token expiry, the OAuth client ID, and, once discovered and saved, the Cloud AI Companion project ID in `<configuration-root>/google-ai-plan/auth.json`. When saving the project ID fails, the storage error is logged and the ID is kept in memory only; a save skipped because the credential changed logs nothing. It does not store the OAuth client secret. With no proxy sign-in, the provider can read an existing Antigravity CLI sign-in from `~/.gemini/jetski-standalone-oauth-token` or, on macOS, from Keychain; a token refresh or a project ID save then writes a copy of that sign-in to the proxy's `auth.json`, which is used from then on instead of the native one. Setting `CCP_CONFIG_DIR` disables both native lookups. Proxy logout writes `<configuration-root>/google-ai-plan/logged-out` to suppress native fallback; proxy login tries to remove the marker and still succeeds when it cannot, in which case native fallback stays off. Neither command modifies Antigravity's files or Keychain entry. See [Signing in again](/providers/google-ai-plan/#signing-in-again) for older proxy credentials.

When `CCP_CONFIG_DIR` is set, file-backed provider credentials use
`<CCP_CONFIG_DIR>/<provider>/auth.json`, including Codex and Cursor on macOS.
`CCP_CURSOR_AUTH_TOKEN` bypasses Cursor's local credential store for that
process.

OpenCode Go is the exception: it reads its API key from
`CCP_OPENCODE_API_KEY`, `OPENCODE_API_KEY`, or `opencode.apiKey` in
`config.json` and does not create a provider auth store.

Vertex AI does not store credentials either. The Google library reads Application
Default Credentials from `GOOGLE_APPLICATION_CREDENTIALS`, from
`~/.config/gcloud/application_default_credentials.json`
(`%APPDATA%\gcloud\application_default_credentials.json` on Windows), or from
the metadata server, and keeps the access token in memory only. It does not
follow `CLOUDSDK_CONFIG`.

Proxy-owned credentials are separate from native provider credentials. See the exceptions above for Antigravity's native credential fallback and Vertex AI's use of Application Default Credentials.

## Kimi device ID

Kimi stores a persistent UUID at `<configuration-root>/kimi/device_id` for file-backed setups. It is bound into the Kimi token and must remain paired with that login.

## Structured log

`proxy.log` lives under the state root. It uses JSON Lines and rotates at 20 MiB. Known credential keys, including authorization, access tokens, refresh tokens, ID tokens, and account headers, are redacted before writing.

A Homebrew service also writes `service.log` under the state root.

## Failed responses

`errors/` under the state root contains JSON files for failed proxy responses. A `request_failed` log event includes `errorFile`, which points to the complete redacted payload.

Error payloads are safer to share than raw traffic captures, but inspect their prompt-derived content and paths before publishing them.

## Traffic captures

Set `CCP_TRAFFIC_LOG=1` to create captures under `traffic/` in the state root. Requests are grouped by Claude Code session and request sequence. A request directory can include:

- inbound Anthropic request and metadata
- translated upstream request
- upstream URL and redacted headers
- raw or decoded upstream events
- translated downstream events and response
- transport or reducer error details

Event filenames use monotonic sequence numbers so lexical order preserves emission order.

<div class="security-callout">
<strong>Traffic captures preserve content.</strong> Header and token redaction does not remove prompts, source code, tool definitions, tool inputs, tool results, or provider output. Image payloads are the exception: base64 image data and image data URLs are replaced with a length marker, as are encrypted reasoning blobs and the proxy's reasoning signatures. Treat the entire directory as sensitive user data.
</div>

Enable capture only for a focused reproduction, keep the directory local, and delete it after the investigation.
