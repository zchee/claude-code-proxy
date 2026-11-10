---
title: Muse
description: Configure Meta Muse Code device-code login, the five Muse models, reasoning effort, tools, images, the API key shared with the Muse CLI, and provider settings.
---

Muse uses Meta's Responses endpoint at `https://api.meta.ai/v1/responses`.

<div class="security-callout">
<strong>Unofficial client.</strong> The proxy is not a Meta client. It was checked against one account with an active Muse Code subscription; accounts without a subscription, with usage-based billing, or managed by an organization may behave differently. Review the terms for your account before using it.
</div>

## Account and authentication

Use a **Meta account with an active Muse Code subscription**. Meta offers only a device-code flow for this client, so `login` and `device` run the same flow:

```sh
claude-code-proxy muse auth login
claude-code-proxy muse auth status
```

The login prints a URL and a code. Open the URL on any device, enter the code, and approve. The proxy prints the URL only when it is HTTPS on `auth.meta.com` or on the configured login host; any other URL stops the login with an error instead of being shown. The proxy then obtains the account's API key once to confirm that the login works, and saves the login only if that succeeds.

The proxy stores the Meta login token in `<configuration-root>/muse/auth.json`, together with the display name, whether the subscription is active, and the tier name. The API key itself and the account's e-mail address are not stored. A running server obtains the key on its first Muse request, not at startup, and keeps it in memory only.

`muse auth status` checks the login against Meta's key service and prints:

- the auth file path
- the account's display name
- whether the subscription is active, and the tier name
- usage of the 5-hour window and the weekly window, with their reset times in local time
- how long ago the login was made

### Logins do not refresh

Meta returns no refresh token for this client, and how long a login stays valid is not known. When the login stops working, requests fail with:

```text
Muse login is no longer valid. Run: claude-code-proxy muse auth login (upstream HTTP 401)
```

`muse auth status` then prints `Not authenticated`, prints `Muse login is no longer valid. Run: claude-code-proxy muse auth login` on stderr, and exits with status 1, as it does when no login is stored. Other failures of the key service exit with status 2 and their own message, such as `Muse key service is unavailable (upstream HTTP 503)`.

Run `claude-code-proxy muse auth login` again. After that error the proxy stops asking Meta for a key with the old login; it picks up a new login from the auth file on the next request, without a restart.

### The API key is shared with the Muse CLI

Meta issues one API key per account. The proxy and the Muse CLI receive the same key, and logging in to the proxy does not log the Muse CLI out.

`claude-code-proxy muse auth logout` deletes only the proxy's auth file. It does not call Meta's logout and does not touch the Muse CLI's login or the shared key. A running server notices the deleted file on its next request and answers it with the login hint.

Logging out in the Muse CLI may revoke the shared key and stop the proxy from working until you log in to the proxy again. This has not been confirmed.

## Models

The registered IDs are `muse-spark-1.3`, `muse-spark-1.2`, `muse-wonder-1.0`, `muse-wonder-full-1.0`, and `muse-ember-1.0`. The `-contributor` variants and the hidden image model in Meta's catalog are not registered for Muse. OpenCode Go registers `muse-spark-1.3-contributor` and `muse-spark-1.2-contributor` as its own models, so those IDs route to OpenCode Go and use its API key, not your Muse login. Use the same Muse ID for `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`.

```sh
ANTHROPIC_MODEL=muse-spark-1.3 \
ANTHROPIC_SMALL_FAST_MODEL=muse-spark-1.3 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1007997 \
  claude --model muse-spark-1.3
```

| Model | Context | Max output | Input | Effort |
| --- | --- | --- | --- | --- |
| `muse-spark-1.3` | 1,007,997 | 128,000 | Text, images | `low` to `max` |
| `muse-spark-1.2` | 1,007,997 | 128,000 | Text, images | `low` to `xhigh`; `max` is sent as `xhigh` |
| `muse-wonder-1.0` | 1,007,997 | 128,000 | Text, images | `low` to `max` |
| `muse-wonder-full-1.0` | 1,007,997 | 128,000 | Text, images | `low` to `max` |
| `muse-ember-1.0` | 1,007,997 | 128,000 | Text, images | `low` to `max` |

Every Muse model has a 1,007,997 token context window. Claude Code treats unknown model IDs as 200,000 tokens, so set `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1007997` when using them. `max_tokens` above 128,000 is lowered to 128,000.

## Reasoning and tools

Claude Code's `/effort` setting maps to Muse reasoning effort:

- `low`, `medium`, `high`, and `xhigh` are sent as they are.
- `max` is sent as `max`, except on `muse-spark-1.2`, which has no `max` level and receives `xhigh`.
- Without an effort setting, the proxy omits effort and Muse uses its default, `high`.
- When thinking is disabled and no effort is set, the proxy sends `minimal`, the lowest level Muse accepts. Muse rejects `none`.

Reasoning summaries appear as Claude Code thinking blocks. Muse sends a summary only for prompts that need one, so short replies often have none. Each thinking block carries a signature that holds Muse's encrypted reasoning, and the proxy sends that reasoning back to Muse on the next turn, as the Muse CLI does. Thinking blocks from other providers are dropped from the history sent to Muse. The proxy does not claim that sending reasoning back improves answers.

Claude function tools, tool choice, tool calls, and tool results map to Responses function calls. Anthropic server tools, such as `web_search_20250305`, are rejected with HTTP 400 because Muse has no equivalent through this proxy.

The proxy derives a prompt-cache key from Claude Code's session ID, so repeated turns of a session can reuse cached input. Only a hash of the session ID is sent.

## Images

Base64 images in user messages are sent as `input_image` data URLs. Images inside tool results follow the tool result in a separate user message, because Muse tool outputs are text. The proxy does not resize or filter images. Very small images can be misread; a 16x16 test image was described incorrectly, while a 256x256 one was read correctly.

## OpenAI-compatible APIs

When the OpenAI routes are enabled, Muse models work with both `POST /v1/chat/completions` and `POST /v1/responses`. Text, images, reasoning effort, function tools, tool results, token limits, streaming, and usage use the standard shape for the chosen route. `reasoning_effort` and `reasoning.effort` accept `low`, `medium`, `high`, `xhigh`, and `max`. These routes do not return thinking signatures, so reasoning is not sent back to Muse on later turns.

## Errors

The proxy chooses the error type from Muse's HTTP status. When Muse answered with a status, the message ends with `(upstream HTTP <status>)`, even when the proxy answers with a different status.

| Situation | Proxy answer |
| --- | --- |
| Muse HTTP clients could not be set up, which happens on the first Muse request rather than at startup | 500, `Muse transport is unavailable: ...` with the cause |
| A Muse URL setting is invalid (see [Configuration](#configuration)) | 500, `Muse transport is unavailable: Invalid Muse setting <setting>: <reason>` |
| No auth file | 401, `Not authenticated with Muse. Run: claude-code-proxy muse auth login` |
| Login no longer valid | 401, `Muse login is no longer valid. Run: claude-code-proxy muse auth login` |
| API key rejected after one retry with a new key | 401, `Muse rejected the API key. Run: claude-code-proxy muse auth login` |
| Key service unavailable or unreachable | 503 |
| Muse rejected the request (400, 404) | Same status, with Muse's message |
| Not permitted (402, 403) | 403. When the subscription is inactive, the message links to `https://accountscenter.meta.com/muse_code/` |
| Rate limited (429) | 429, with `retry-after` when Muse sends one as whole seconds or when a recent usage report shows an exhausted window; either way at most 86400 seconds |
| Stream stopped before it finished | Stream error event, or 502 for a non-streaming request, `Muse stream ended before completion` |

Only an API key rejection is retried, once. Rate limits and server errors are returned to Claude Code, which decides whether to retry. The proxy puts no timeout on a Muse response, because a long prompt can take more than a minute to start streaming. The body of an error response is the exception: if it has not arrived within 5 seconds, the proxy answers from the status alone, as it does for an error without a message (for example `Muse API request failed (upstream HTTP 503)`).

Muse error text is checked before it reaches Claude Code, logs, or traffic captures: the login token, the API key, and e-mail addresses are replaced with `[redacted]`, and the text is cut to 500 characters.

## Configuration

- `CCP_MUSE_BASE_URL` or `muse.baseUrl` changes the Responses API base URL.
- `CCP_MUSE_OAUTH_HOST` or `muse.oauthHost` changes the device-login host.
- `CCP_MUSE_MINT_BASE_URL` or `muse.mintBaseUrl` changes the host of the API key service.

These three hosts receive the Meta login token or the API key, so each value must be an `https` URL without a user name or password. Plain `http` is accepted only for a loopback host (`127.0.0.0/8`, `::1`, or `localhost`). An empty value counts as not set. A non-empty invalid value is not replaced by the default: the server fails Muse requests and the `muse auth` commands stop with an error that names the setting, without showing the value:

```text
Invalid Muse setting CCP_MUSE_BASE_URL: must use https (plain http only for a loopback host)
Invalid Muse setting muse.mintBaseUrl in config.json: must not contain a user name or password
```

- `CCP_MUSE_USER_AGENT` or `muse.userAgent` changes the user-agent. It is sent with the login token and the API key, so the value must be 1 to 256 printable ASCII characters (no control characters). An empty value counts as not set, because Meta's login host rejects requests without a user-agent. A non-empty invalid value is an error that names the setting, as for the URL settings above.

The proxy identifies itself as `claude-code-proxy/<version>` and does not present itself as the Muse CLI. See [Configuration](/reference/configuration/) for defaults.

## Limitations and troubleshooting

- Subscription usage is not shown in the monitor yet. Use `muse auth status`.
- Token counts are local estimates. Meta's token-count endpoint is not used, because its counts did not match billed input.
- Only the account checked during development has been tested. Behaviour without a subscription, after the usage allowance runs out, or under heavy load has not been observed, so those errors pass Muse's status and message through as described above.

If requests fail after a successful login, run `muse auth status` first. It contacts Meta directly and shows whether the login and subscription are still valid. Inspect the failed request in the monitor, and use the structured log or error capture for the full redacted response.
