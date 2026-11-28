---
title: Google AI Ultra plan
description: Use Gemini on the quota of your own Google AI Ultra subscription with a browser sign-in, the four plan models and their thinking levels, errors, settings, and the OAuth client the sign-in uses.
---

The Google AI Ultra plan provider sends requests to Gemini on the quota of your own Google AI Ultra subscription. It signs in with your Google account and calls Google's Interactions API:

```text
POST https://generativelanguage.googleapis.com/v1beta/interactions:createPerUserQuota
```

Every request carries your access token as a Bearer token and asks for an event stream. Nothing is billed to a Google Cloud project; for pay-as-you-go Gemini on your own project, use [Vertex AI](/providers/vertex/) instead.

<div class="security-callout">
<strong>Unofficial client.</strong> The proxy is not a Google client. It signs in with an OAuth client that the pi project registered with Google for this plan (see <a href="#the-oauth-client">The OAuth client</a>). Usage counts against your plan's limits, and Google decides which apps and accounts may use it. Review the terms for your account before using it.
</div>

## Sign in

Use a Google account with an active Google AI Ultra plan:

```sh
claude-code-proxy google-ai-plan auth login
claude-code-proxy google-ai-plan auth status
```

`login` prints a Google sign-in URL and tries to open it in your browser. Google sends the browser back to a listener the proxy opens on `127.0.0.1`, on a free port. When the browser runs on another machine, the redirect cannot reach that listener: copy the URL the browser ended on from its address bar and paste it into the terminal. The proxy waits up to 10 minutes for either, long enough to finish a plan upgrade in the browser.

The sign-in uses PKCE, a `state` and an OpenID `nonce`, and binds the tokens to a DPoP key the proxy creates for that sign-in. It fails when Google does not grant the plan's scope (`generative-language.peruserquota`):

```text
Google did not grant access to the Google AI Ultra plan. Try signing in again.
```

`login` and `device` run the same browser sign-in; there is no device-code sign-in.

### What is stored

The sign-in is saved in `<configuration-root>/google-ai-plan/auth.json`. It holds the access token, the refresh token, the access token's expiry, the OAuth client ID the tokens were issued to, the DPoP private key, and the latest DPoP nonce from Google. The client secret is not stored.

The proxy refreshes the access token, with a DPoP proof, when it is less than 5 minutes from expiry. Concurrent requests share one refresh.

### Signing in again

Google's grant for this sign-in expires after 7 days, according to the pi project's sign-in code, so expect to sign in again about once a week. When Google refuses the refresh token (`invalid_grant` or `unauthorized_client`), requests fail with:

```text
Your Google AI Ultra plan sign-in is no longer valid. Run `claude-code-proxy google-ai-plan auth login` to sign in again.
```

Run `login` again. A running server reads the new sign-in from the file on its next request, without a restart.

The tokens belong to the OAuth client that issued them. After you switch between the built-in client and your own (see [Use your own client](#use-your-own-client)), requests fail with `The stored Google AI Ultra plan sign-in belongs to a different OAuth client than the one in use.` until you sign in again.

### `google-ai-plan auth status`

`status` reads the stored sign-in and prints:

- the auth file path
- `Authenticated: true`, only for a sign-in issued to the OAuth client in use
- `OAuth client: built-in` or `OAuth client: configured`; the client secret is never printed
- when the access token expires, or that it has expired and is refreshed on the next request

It does not contact Google. Without a sign-in it prints the auth file path and `Not authenticated`, and exits with status 1. An auth file that cannot be read or parsed counts as no sign-in. For a sign-in issued to another client than the one in use, it prints the path and the `different OAuth client` message and exits with status 2; an invalid setting also exits with status 2 and its message.

`logout` deletes the auth file. It does not revoke the grant at Google.

## Models

The plan has no model list endpoint, so the proxy lists four models. Every name starts with `google-ai-plan/`, because the bare Gemini IDs route to Vertex AI.

| Model | Thinking levels |
| --- | --- |
| `google-ai-plan/gemini-4-hydrogen` | `low` to `xhigh` |
| `google-ai-plan/gemini-4-neon` | `low` to `xhigh` |
| `google-ai-plan/gemini-4-carbon` | `low` to `high` |
| `google-ai-plan/gemini-3.5-flash-lite` | `minimal` to `high` |

Every model has a 1,048,576 token context window and at most 65,536 output tokens. `max_tokens` above 65,536 is lowered to 65,536; without `max_tokens` the proxy sends none and Google uses its default. Claude Code treats unknown model IDs as 200,000 tokens, so set `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576` when using them:

```sh
ANTHROPIC_MODEL=google-ai-plan/gemini-4-hydrogen \
ANTHROPIC_SMALL_FAST_MODEL=google-ai-plan/gemini-3.5-flash-lite \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
  claude
```

Any other `google-ai-plan/<model-id>` is sent as well, with the levels `low` to `high`, so a model Google adds to the plan works before the proxy lists it. A model the plan does not include gets Google's answer, explained as in [Errors](#errors). The ID after the prefix may contain only ASCII letters, digits, `.`, `_`, and `-`, and at most 256 bytes; any other name, and the prefix alone, is rejected with HTTP 400 before any setting or credential is read.

## Thinking

None of the models can turn thinking off. Claude Code's `/effort` setting and its thinking setting choose `generation_config.thinking_level`, moved to the nearest level the model accepts:

| Claude Code | Level asked for |
| --- | --- |
| effort `low`, `medium`, or `high` | the same level |
| effort `xhigh` or `max` | `xhigh` |
| no effort, thinking on | `high` |
| no effort, thinking off or absent | `minimal` |

For example, effort `max` on `gemini-4-carbon` is sent as `high`, and thinking off on `gemini-4-hydrogen` is sent as `low`. Thinking is on for any thinking type other than `disabled`; `budget_tokens` is not used.

Thought summaries appear as Claude Code thinking blocks only when thinking was on for the request. Each thinking block carries a signature envelope in its `signature` field, as `ccp:google-ai-plan:v1:<model>:<signature>`, whether or not thinking was on, and the proxy sends Google's signature back with the history on the next request. When Google sent no signature for a thought, the envelope is empty after the model (`ccp:google-ai-plan:v1:<model>:`) and nothing is sent back for it. Gemini rejects a replayed function call without a signed thought before it; when the history has none from the same model, for example after switching models or providers, the proxy sends Google's placeholder `skip_thought_signature_validator`. Signatures from Vertex AI and other providers are never sent here.

## Requests

Every request is stateless (`store: false`) and carries the whole conversation. Claude function tools become Interactions function tools, with `input_schema` sent as the parameters. `tool_choice` is sent only together with tools: `auto`, `any`, and `none` are sent as they are, and `tool` with a name is sent as `any` with only that tool declared. A `tool` choice naming a tool the request does not declare is rejected with HTTP 400. Without tools, `tool_choice` `none` is dropped and any other choice is rejected with HTTP 400 (`tool_choice requires at least one tool`). Base64 images in user messages and in tool results are sent as image content with their media type.

What the Interactions API has no counterpart for:

| Request field | What happens |
| --- | --- |
| `temperature`, `top_p`, `top_k` | Dropped |
| `stop_sequences` | Absent, `null`, or empty: nothing is sent. Non-empty: HTTP 400, `stop_sequences is not supported by the Google AI Ultra plan` |
| `max_tokens` of 0 | HTTP 400, `max_tokens must be at least 1` |
| Image URLs | HTTP 400; send the image as base64 data |
| Anthropic server tools, such as `web_search_20250305` | HTTP 400 |

Token counts (`/v1/messages/count_tokens`) are local estimates; nothing is sent to Google.

## Stop reasons and usage

The stream ends with `interaction.completed`. Its status decides the stop reason:

| Status | Stop reason |
| --- | --- |
| `completed` | `end_turn` |
| `requires_action` | `tool_use` |
| `incomplete` | `max_tokens` |
| anything else | an error |

The proxy stops reading at `interaction.completed`. A stream that closes before it, or that carries an `error` event, ends with one Anthropic `error` event and no `message_stop`; a non-streaming request gets an error status instead (502 for a stream that closed early). The proxy does not send the request again; Claude Code decides whether to retry. The proxy log records each streaming response that ends in an error as `google_ai_plan_stream_failed`, with the request ID, the model, the kind of failure, and the bytes read, but none of Google's text.

Usage comes from `interaction.usage`: `input_tokens` is `total_input_tokens` minus `total_cached_tokens`, `cache_read_input_tokens` is `total_cached_tokens`, and `output_tokens` is `total_output_tokens` plus `total_thought_tokens`. The `input_tokens` in `message_start` is the proxy's local estimate, because it is sent before Google reports usage.

## Errors

An error from Google is passed on as `<upstream status> <Google's message> <hint>`. Google's text is cleaned first: the access token, e-mail addresses, and long runs of base64 data are replaced, and the text is cut to 500 characters.

| Situation | Proxy answer |
| --- | --- |
| Per-minute limit (429) | 429 `rate_limit_error` with Google's message, and `retry-after` when Google sends whole seconds. Claude Code retries it |
| Daily limit (429) | 429 `rate_limit_error`, ending `Daily limit reached for <model>: switch to another model or provider.`, with `x-should-retry: false`. When Google's message does not say when the limit resets and its reply has a usable `Retry-After`, the time is added after the model, as `(resets in 5h 18m)` or, under an hour, `(resets in 42m)` |
| Model not on the plan (429 with a daily limit of 0, or 400 `not supported for per-user quota`) | Same status, ending `<model> is not available on the Google AI Ultra plan. Choose another model.`; the 429 carries `x-should-retry: false` |
| Sign in again (Google 401, or 403 for missing scopes) | 401 `authentication_error` or 403 `permission_error`, ending ``Run `claude-code-proxy google-ai-plan auth login` to sign in again.`` |
| No eligible plan (Google 402 or 403) | 403 `permission_error`, ending `Your Google account does not have an eligible Google AI Ultra plan. Upgrade at https://one.google.com/ai, or switch to another model or provider.` |
| Client not allowed (Google 403 `SERVICE_DISABLED`, or `OAuth client is not authorized`) | 403 `permission_error`, ending ``The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.`` See [The OAuth client](#the-oauth-client) |
| Other Google 4xx | Same status, `invalid_request_error` |
| Google 5xx, or no connection | 502 `api_error`; `Google AI Ultra plan is unreachable` when no status arrived |
| No response headers within 300 seconds | 504 `api_error` |
| No sign-in (including an auth file that cannot be read or parsed), a sign-in for another client, or a refused refresh token | 401 `authentication_error` naming `claude-code-proxy google-ai-plan auth login` |
| The token endpoint refused the refresh for another reason, or could not be reached | 502 `api_error`, `Google token refresh failed: <status> <error> <description>` with the refresh token and the client secret replaced by `[redacted]` |
| The refreshed sign-in could not be written to the auth file | 500 `api_error` |
| A setting is invalid (see [Settings](#settings)) | 500 `api_error` naming the setting |

The proxy does not retry 429s or 5xx itself. `x-should-retry: false` tells Claude Code not to retry a limit that a backoff does not lift; it is sent on `/v1/messages` only.

### Timeouts

The proxy waits at most 10 seconds to connect and 300 seconds for the response headers. Once the headers have arrived there is no timeout: a turn may stream for many minutes, and the model may stay silent while it thinks, so a stalled upstream holds the request until the client gives up. Vertex AI behaves the same.

## OpenAI-compatible routes

When the OpenAI routes are enabled, the four models work on `POST /v1/chat/completions` and `POST /v1/responses`, streaming or not. The responses name the model as `google-ai-plan/<model-id>`, so a client that sends the name back reaches this provider again. `reasoning_effort` and `reasoning.effort` choose the thinking level as in [Thinking](#thinking), but these routes return no thinking text and no thought signature. `parallel_tool_calls` is accepted but not enforced.

An HTTP error from Google before the response starts is passed on with its status and message, without `x-should-retry`. An error that arrives inside a stream that has started is a stream error on the streaming routes and a 502 on the non-streaming ones.

## Settings

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_GOOGLE_AI_PLAN_CLIENT_ID` | `googleAiPlan.clientId` | the built-in client | Your own OAuth client ID. Set it together with the secret |
| `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` | `googleAiPlan.clientSecret` | the built-in client | Your own OAuth client secret. Set it together with the ID |
| `CCP_GOOGLE_AI_PLAN_BASE_URL` | `googleAiPlan.baseUrl` | `https://generativelanguage.googleapis.com/v1beta` | The Gemini API base URL, for a relay or a local test server. `/interactions:createPerUserQuota` is appended |
| `CCP_GOOGLE_AI_PLAN_USER_AGENT` | `googleAiPlan.userAgent` | `claude-code-proxy/<version>` | The `User-Agent` of every model call: 1 to 256 printable ASCII characters |

Environment variables take precedence over `config.json`, one setting at a time, and an empty or blank value counts as not set. The base URL receives the access token, so it must use `https` (plain `http` only for a loopback host) and must not contain a user name, a password, a query, or a fragment. A client ID without a secret, or a secret without an ID, is an error. A non-empty invalid value is an error that names the setting, not a fallback to the default:

```text
Invalid Google AI Ultra plan setting CCP_GOOGLE_AI_PLAN_BASE_URL: must use https (plain http only for a loopback host)
```

The server reads these settings at startup. With an invalid setting the provider is still listed, and every request whose model name and body are valid fails with a 500 and that message; a malformed request still gets its 400 first. `google-ai-plan auth login` and `auth status` stop with the message, while `auth logout` works regardless. The `auth` commands read the settings each time they run.

## The OAuth client

Google approves OAuth clients for the Google AI Ultra plan one app at a time. The proxy ships a built-in client: the `Desktop app` client the pi project registered with Google for this plan. pi's README says Google approved that client for pi only and that forks and other apps must register their own Google OAuth client. Google's consent screen therefore names pi's app, not claude-code-proxy.

Google or the pi project can revoke or restrict the built-in client at any time. Requests then fail with a 403 that ends ``The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.``

The secret of an installed app's client cannot be kept secret, and Google treats it as non-confidential; the sign-in is protected by the loopback redirect, PKCE, and DPoP. The proxy still keeps it out of logs, captures, and `status` output.

### Use your own client

1. In your own Google Cloud project, create an OAuth client of type `Desktop app`.
2. Have Google approve it for the Google AI Ultra plan's scope, `https://www.googleapis.com/auth/generative-language.peruserquota`.
3. Configure both values, in the environment or in `config.json`:

   ```json
   {
     "googleAiPlan": {
       "clientId": "YOUR_CLIENT_ID.apps.googleusercontent.com",
       "clientSecret": "YOUR_CLIENT_SECRET"
     }
   }
   ```

4. Sign in again with `claude-code-proxy google-ai-plan auth login`. `status` then prints `OAuth client: configured`.

Tokens from one client do not work with another, so switching clients in either direction needs a new sign-in.
