---
title: Google AI Ultra plan
description: Use Gemini on the quota of your own Google AI Ultra subscription with a browser sign-in, the four plan models and their thinking levels, errors, settings, and the OAuth client the sign-in uses.
---

Use Gemini on the quota of your own Google AI Ultra subscription. The proxy calls Cloud Code PA at `https://daily-cloudcode-pa.googleapis.com`. If that host is unreachable or returns a 5xx status, the proxy tries `https://cloudcode-pa.googleapis.com` once for that call.

- `POST /v1internal:streamGenerateContent?alt=sse` streams the model response, even when the client requests a buffered reply.
- `POST /v1internal:loadCodeAssist` discovers the Cloud AI Companion project for the sign-in. If an otherwise usable eligibility answer names no project, `POST /v1internal:onboardUser` attempts to obtain one. The proxy caches the project in memory for that credential and tries to save it in the proxy's auth file so later requests can reuse it; a failed save is logged and the request goes on. When the sign-in in use is a native Antigravity CLI sign-in, that save, like a token refresh, writes a copy of the sign-in to the proxy's auth file, and the proxy uses that copy from then on, whatever later happens to the native sign-in.

These calls use a Bearer access token. The discovered project is not a billing project you configure; for pay-as-you-go Gemini on your own Google Cloud project, use [Vertex AI](/providers/vertex/) instead.

<div class="security-callout">
<strong>Unofficial client.</strong> The proxy is not a Google client. By default it uses the OAuth client of Google's Antigravity CLI (see <a href="#the-oauth-client">The OAuth client</a>). Usage counts against your plan's limits, and Google decides which apps and accounts may use it. Review the terms for your account before using it.
</div>

## Sign in

Use a Google account with an active Google AI Ultra plan:

```sh
claude-code-proxy google-ai-plan auth login
claude-code-proxy google-ai-plan auth status
```

`login` prints a Google sign-in URL and tries to open it in your browser. Google sends the browser back to a listener the proxy opens on `127.0.0.1`, on a free port. When the browser runs on another machine, the redirect cannot reach that listener: copy the URL the browser ended on from its address bar and paste it into the terminal. The proxy waits up to 10 minutes for the redirect or pasted URL.

The browser sign-in uses the OAuth authorization code flow with PKCE and `state`. An `id_token` is optional; when returned, the proxy checks its issuer, audience, expiry, and nonce. `login` and `device` run the same browser sign-in; there is no device-code sign-in.

### What is stored

The proxy uses its own sign-in first. With no proxy sign-in and `CCP_CONFIG_DIR` unset, it can reuse an existing native Antigravity CLI (`agy`) sign-in issued to the built-in OAuth client. See [Files and storage](/reference/files-and-storage/#provider-credentials) for both credential locations.

The proxy refreshes the access token when it is within 5 minutes of expiry. Concurrent requests share one refresh.

### Signing in again

When Google refuses the refresh token (`invalid_grant` or `unauthorized_client`), requests fail with:

```text
Your Google AI Ultra plan sign-in is no longer valid. Run `claude-code-proxy google-ai-plan auth login` to sign in again.
```

Run `login` again. A running server reads the new sign-in on its next request, without a restart. Older `auth.json` files containing a DPoP-bound sign-in also require `claude-code-proxy google-ai-plan auth login`; the current sign-in does not use DPoP and cannot refresh those tokens.

The tokens belong to the OAuth client that issued them. After you switch between the built-in client and your own (see [Use your own client](#use-your-own-client)), requests fail with `The stored Google AI Ultra plan sign-in belongs to a different OAuth client than the one in use.` until you sign in again.

### `google-ai-plan auth status`

`status` reads the stored sign-in and prints:

- the auth file path
- `Authenticated: true`, only for a sign-in issued to the OAuth client in use
- `OAuth client: built-in` or `OAuth client: configured`; the client secret is never printed
- when the access token expires, or that it has expired and is refreshed on the next request

It does not contact Google. An unreadable or malformed proxy auth file does not count as a sign-in. If no sign-in is found, `status` prints the auth file path and `Not authenticated` and exits with status 1. A sign-in issued to another client or an invalid setting causes an error and exit status 2.

`logout` deletes the proxy's auth file and writes a `logged-out` marker that disables native fallback for this configuration root. If it cannot write the marker, `logout` fails with an error instead of reporting success. The next proxy `login` tries to remove the marker; if that fails, `login` still succeeds and native fallback stays off. `logout` leaves the native Antigravity credential untouched and does not revoke the grant at Google.

## Models

The plan has no model list endpoint, so the proxy lists four models. Every name starts with `google-ai-plan/`, because the bare Gemini IDs route to Vertex AI.

| Model | Upstream thinking levels |
| --- | --- |
| `google-ai-plan/gemini-4-hydrogen` | `LOW` to `HIGH` |
| `google-ai-plan/gemini-4-neon` | `LOW` to `HIGH` |
| `google-ai-plan/gemini-4-carbon` | `LOW` to `HIGH` |
| `google-ai-plan/gemini-3.5-flash-lite` | `MINIMAL` to `HIGH` |

Every model has a 1,048,576 token context window and at most 65,536 output tokens. `max_tokens` above 65,536 is lowered to 65,536 (a value above 4,294,967,295 is rejected); without `max_tokens` the proxy sends none and Google uses its default. Claude Code treats unknown model IDs as 200,000 tokens, so set `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576` when using them:

```sh
ANTHROPIC_MODEL=google-ai-plan/gemini-4-hydrogen \
ANTHROPIC_SMALL_FAST_MODEL=google-ai-plan/gemini-3.5-flash-lite \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576 \
  claude
```

Any other `google-ai-plan/<model-id>` is sent as well, with the levels `low` to `high`, so a model Google adds to the plan works before the proxy lists it. A model the plan does not include gets Google's answer, explained as in [Errors](#errors). After the proxy removes the `[1m]` suffix (in any case), the ID after the prefix may contain only ASCII letters, digits, `.`, `_`, and `-`, and at most 256 bytes; an ID that does not, and the prefix alone, is rejected with HTTP 400 before the credential is read or an invalid setting is reported.

## Thinking

None of the models can turn thinking off. Claude Code's `/effort` setting and its thinking setting choose `generationConfig.thinkingConfig.thinkingLevel`, clamped to the model's supported range:

| Claude Code | Requested upstream level before clamping |
| --- | --- |
| effort `low`, `medium`, or `high` | `LOW`, `MEDIUM`, or `HIGH` |
| effort `xhigh` or `max` | `HIGH` (both are input aliases) |
| no effort, thinking on | `HIGH` |
| no effort, thinking off or absent | `MINIMAL` |

For example, with no effort and thinking off, `gemini-4-hydrogen` still receives `LOW`. Thinking is on for any thinking object whose type is not `disabled`; `budget_tokens` is not used.

Thought summaries appear as Claude Code thinking text only when thinking was on for the request. The proxy can still emit a thinking block to carry a signature when thinking text is off. Claude Code records each block's `signature` field as `ccp:google-ai-plan:v1:<model>:<signature>`; an empty suffix means there is no signature to replay: Google sent none, or the one it sent was over 1 MiB. The proxy attaches a non-empty signature from the same model to the next function call in the replayed history. When there is no signature for the first function call of a model turn, the proxy uses Google's placeholder `skip_thought_signature_validator`. It does not replay signatures from other models or providers.

## Requests

Every request carries the whole conversation. Claude function tools become Gemini function declarations, with `input_schema` sent as `parametersJsonSchema`. With tools, `tool_choice` `auto`, `any`, and `none` map to `AUTO`, `ANY`, and `NONE`. A named `tool` choice sets `ANY` with only that name allowed; an undeclared name is rejected with HTTP 400. Without tools, `tool_choice` `none` is dropped and any other choice is rejected with HTTP 400 (`tool_choice requires at least one tool`). Base64 images in user messages and tool results are sent inline with their media type.

Request limits:

| Request field | What happens |
| --- | --- |
| `temperature`, `top_p`, `top_k` | Dropped |
| `stop_sequences` | Absent, `null`, or empty: nothing is sent. Non-empty: HTTP 400, `stop_sequences is not supported by the Google AI Ultra plan` |
| `max_tokens` of 0 | HTTP 400, `max_tokens must be at least 1` |
| Image URLs | HTTP 400; send the image as base64 data |
| Anthropic server tools, such as `web_search_20250305` | HTTP 400 |

Token counts (`/v1/messages/count_tokens`) are local estimates; nothing is sent to Google.

## Stop reasons and usage

The model's `finishReason` determines the Anthropic stop reason:

| `finishReason` | Stop reason |
| --- | --- |
| `STOP` with a function call | `tool_use` |
| `STOP` without a function call | `end_turn` |
| `MAX_TOKENS` | `max_tokens` |
| Any other value | An error |

After a successful `finishReason`, once the proxy starts waiting for trailing `usageMetadata`, it reads for up to 2 seconds to collect it, for both streaming and buffered replies. It finishes sooner if Google closes the stream. A stream that closes before a `finishReason`, or that carries an upstream error, ends with one Anthropic `error` event and no `message_stop`; a buffered request gets an error status instead (502 for a stream that closed early). Both reply modes reject an upstream frame over 16 MiB and translated content over 16 MiB; a buffered reply also fails when the whole upstream body exceeds 16 MiB, which a stream of many small frames can stay under. The proxy does not resend a call after its stream has started. It logs streaming failures as `google_ai_plan_stream_failed`, with the request ID, model, kind of failure, the error (which can name an upstream error code, block reason, or finish reason), and bytes read, but not Google's error or finish message.

Usage comes from `usageMetadata`: `input_tokens` is `promptTokenCount` minus `cachedContentTokenCount`, `cache_read_input_tokens` is `cachedContentTokenCount`, and `output_tokens` is `candidatesTokenCount` plus `thoughtsTokenCount`. The `input_tokens` in `message_start` is the proxy's local estimate, because it is sent before Google reports usage.

## Errors

A non-2xx response to a Cloud Code generation or discovery call is passed on as `<upstream status> <Google's message> <hint>`; when the error body is empty, too long, cut off, or late, `Google AI Ultra plan returned HTTP <upstream status>` takes the place of Google's message. Google's text is cleaned first: the access token, e-mail addresses, and long runs of base64 data are replaced, and the text is cut to 500 characters.

| Situation | Proxy answer |
| --- | --- |
| Per-minute limit (429) | 429 `rate_limit_error` with Google's message and `Retry-After` in seconds when available. Claude Code can retry it |
| Daily limit (429) | 429 `rate_limit_error`, ending `Daily limit reached for <model>: switch to another model or provider.`, with `x-should-retry: false`. Unless Google's message is worded `reached today's … limit`, the proxy adds a positive delay from `Retry-After` or a usable `RetryInfo.retryDelay` (rounded up to whole seconds, at most one day), when one is present, as `(resets in 5h 18m)` or `(resets in 42m)` |
| Model not on the plan | Google's status, ending `<model> is not available on the Google AI Ultra plan. Choose another model.`; a 429 carries `x-should-retry: false` |
| Sign in again (Google 401, or 403 for missing scopes) | 401 `authentication_error` or 403 `permission_error`, ending ``Run `claude-code-proxy google-ai-plan auth login` to sign in again.`` |
| Another Google 402 or 403 from a model call, not matched by the sign-in or client rows | 403 `permission_error` with a suggestion to upgrade the plan or switch model or provider. The proxy infers this from the status alone; Google's answer need not be about the plan |
| Client not allowed (Google 403 `SERVICE_DISABLED`, or `OAuth client is not authorized`) | 403 `permission_error`, ending ``The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.`` See [The OAuth client](#the-oauth-client) |
| Any other Google 402 or 403 from project discovery or onboarding | 403 `permission_error` without an upgrade suggestion |
| Other Google 4xx | Same status, `invalid_request_error` |
| Invalid, cut off, or incomplete project discovery response | 502 `api_error` |
| Google 5xx, or no connection | 502 `api_error`; `Google AI Ultra plan is unreachable` when no status arrived |
| No response headers within 300 seconds, or a successful project discovery body not read within 300 more seconds | 504 `api_error`, saying no response headers arrived in both cases |
| No usable sign-in (an auth file that cannot be read or parsed counts as none, so an eligible native sign-in is used instead), a sign-in for another client, or a refused refresh token | 401 `authentication_error` naming `claude-code-proxy google-ai-plan auth login` |
| The token endpoint refused the refresh for another reason | 502 `api_error`, `Google token refresh failed: <status>`, followed by Google's `error` and `error_description` when present (for a reply that is not a JSON object, `HTTP <status>` and the start of the body or a fixed description), with the refresh token and the client secret replaced by `[redacted]` |
| The token endpoint could not be reached, or its reply could not be read | 502 `api_error`, `Google token request failed:` followed by the connection error |
| The refreshed sign-in could not be written to the auth file, or another writer replaced the stored sign-in during each of three refreshes of one request and the sign-in stored last still needs refreshing (a fresh one is used; a sign-out gets the 401 above) | 500 `api_error`; the request can be retried |
| A setting is invalid (see [Settings](#settings)) | 500 `api_error` naming the setting |

For a per-minute 429, the proxy's `Retry-After` is Google's `Retry-After` when that is a whole number of seconds, otherwise `google.rpc.RetryInfo.retryDelay` from the response body when it holds a usable delay (rounded up to whole seconds, at most one day). A daily-limit 429 uses these values only for the reset estimate in its message, and a 429 for a model not on the plan uses neither. The proxy does not retry 429s. `x-should-retry: false` tells Claude Code not to retry a limit that a backoff does not lift; it is sent on `/v1/messages` only.

### Timeouts

For each Cloud Code generation or discovery attempt, the proxy waits at most 10 seconds to connect and 300 seconds for response headers. The body of a successful project discovery response gets 300 more seconds, and an error body 10 seconds. There is no streaming idle timeout before `finishReason`; a stalled upstream can hold the request until the client gives up. After `finishReason`, the [trailing usage window](#stop-reasons-and-usage) applies.

## OpenAI-compatible routes

When the OpenAI routes are enabled, the four models work on `POST /v1/chat/completions` and `POST /v1/responses`, streaming or not. The responses name the model as `google-ai-plan/<model-id>`, so a client that sends the name back reaches this provider again. `reasoning_effort` and `reasoning.effort` choose the thinking level as in [Thinking](#thinking), but these routes return no thinking text and no thought signature. `parallel_tool_calls` is accepted but not enforced.

A provider error returned before generation starts gets the status and message `/v1/messages` answers with (see [Errors](#errors)), without `x-should-retry`. An error that arrives inside a stream that has started is a stream error on the streaming routes and a 502 on the non-streaming ones.

## Settings

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_GOOGLE_AI_PLAN_CLIENT_ID` | `googleAiPlan.clientId` | the built-in client | Your own OAuth client ID. Set it together with the secret |
| `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` | `googleAiPlan.clientSecret` | the built-in client | Your own OAuth client secret. Set it together with the ID |
| `CCP_GOOGLE_AI_PLAN_BASE_URL` | `googleAiPlan.baseUrl` | `https://daily-cloudcode-pa.googleapis.com` | The Cloud Code PA base URL. The default has the fallback described above; a different base URL does not |
| `CCP_GOOGLE_AI_PLAN_USER_AGENT` | `googleAiPlan.userAgent` | Antigravity CLI user agent for the current OS and architecture | The HTTP `User-Agent` on Cloud Code PA calls: 1 to 256 printable ASCII characters |

Environment variables take precedence over `config.json`, one setting at a time, and an empty or blank value counts as not set. The base URL receives the access token, so it must use `https` (plain `http` only for a loopback host) and must not contain a user name, a password, a query, or a fragment. A client ID without a secret, or a secret without an ID, is an error. A `config.json` that cannot be read or parsed, for example because a setting is not a string, is ignored as a whole, and the environment and the defaults apply. A non-empty invalid value, from the environment or from a `config.json` that can be read and parsed, is an error that names the setting, not a fallback to the default:

```text
Invalid Google AI Ultra plan setting CCP_GOOGLE_AI_PLAN_BASE_URL: must use https (plain http only for a loopback host)
```

The server reads these settings at startup. With an invalid setting the provider is still listed, and every model call whose model name and body are valid fails with a 500 and that message; a malformed request still gets its 400 first. Token counting checks only the model name first, so a body that passes the shared request checks but that a model call would reject gets the 500 there. `google-ai-plan auth login` and `auth status` stop with the message, while `auth logout` works regardless. The `auth` commands read the settings each time they run.

## The OAuth client

The built-in `Desktop app` OAuth client belongs to Google's Antigravity CLI. The proxy uses that client identity but is not the official client. Google controls which clients and accounts can use the service and can restrict access at any time.

When Cloud Code PA answers 403 with `SERVICE_DISABLED` as the `ErrorInfo` reason, or with `OAuth client is not authorized` in its message, and the 403 is not an insufficient-scope error (that check comes first), requests fail with a 403 that ends ``The Google OAuth client this proxy signs in with is not allowed to use the Google AI Ultra plan. Set `CCP_GOOGLE_AI_PLAN_CLIENT_ID` and `CCP_GOOGLE_AI_PLAN_CLIENT_SECRET` to a client Google has approved.`` When the token endpoint refuses a refresh with `unauthorized_client`, requests fail with the 401 in [Signing in again](#signing-in-again) instead.

Google treats an installed app's client secret as non-confidential. The sign-in uses the loopback redirect and PKCE; the proxy keeps the client secret out of logs, captures, and `status` output.

### Use your own client

1. In your own Google Cloud project, create an OAuth client of type `Desktop app`. It must be allowed by Google to use Cloud Code PA with the plan; creating it does not grant access.
2. Configure both values, in the environment or in `config.json`:

   ```json
   {
     "googleAiPlan": {
       "clientId": "YOUR_CLIENT_ID.apps.googleusercontent.com",
       "clientSecret": "YOUR_CLIENT_SECRET"
     }
   }
   ```

3. Sign in again with `claude-code-proxy google-ai-plan auth login`. `status` then prints `OAuth client: configured`.

Tokens from one client do not work with another, so switching clients in either direction needs a new sign-in.
