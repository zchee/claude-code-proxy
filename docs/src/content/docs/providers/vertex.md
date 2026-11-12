---
title: Vertex AI
description: Use Gemini models on Google Cloud Vertex AI with Application Default Credentials, project and location settings, thinking levels, tools, thought signatures, errors, and limits.
---

The Vertex AI provider sends Gemini requests to Google Cloud's `streamGenerateContent` method:

```text
https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/publishers/google/models/<model>:streamGenerateContent?alt=sse
```

<div class="security-callout">
<strong>Requests are billed to your Google Cloud project.</strong> Vertex AI has no subscription. Every request is charged at Vertex AI's pay-as-you-go rates to the project the proxy is configured with, using your own Google credentials with the `cloud-platform` scope. A listener that other people can reach can spend that project's money.
</div>

Only Gemini models are supported. Claude, partner models, open models from Model Garden, and deployed endpoints on Vertex AI are not.

## Prerequisites

You need a Google Cloud project with billing enabled, the Vertex AI API enabled in it, and an account that holds the role `roles/aiplatform.user` on that project.

Enable the API once per project:

```sh
gcloud services enable aiplatform.googleapis.com --project=PROJECT_ID
```

Grant the role to the account that will make the requests. This needs permission to change the project's IAM policy; if you do not have it, ask a project admin to run the command:

```sh
gcloud projects add-iam-policy-binding PROJECT_ID --member=user:USER_EMAIL --role=roles/aiplatform.user
```

## Credentials

The proxy uses Google Application Default Credentials (ADC). It stores no Google credentials and has no login flow of its own. The `google-cloud-auth` library finds the credential, refreshes the access token in memory, and builds the request headers.

Create user credentials with gcloud:

```sh
gcloud auth application-default login
```

The library looks for a credential in this order:

1. The file named by `GOOGLE_APPLICATION_CREDENTIALS`, for example a service account key. When this variable is set, even to an empty value, it is the only file tried.
2. `~/.config/gcloud/application_default_credentials.json` (`%APPDATA%\gcloud\application_default_credentials.json` on Windows), which `gcloud auth application-default login` writes.
3. The metadata server, which exists only on Google Cloud (Compute Engine, GKE, Cloud Run).

The credentials file is trusted input: the library sends its refresh token or key material to the token endpoint and other URLs the file names, and the proxy does not check those endpoints, so whoever can write the file decides where that material goes.

The lookup does not follow `CLOUDSDK_CONFIG`. If you run gcloud with `CLOUDSDK_CONFIG` pointing at another directory, `gcloud auth application-default login` writes the credentials there, and the proxy does not find them. Point `GOOGLE_APPLICATION_CREDENTIALS` at that file instead:

```sh
export GOOGLE_APPLICATION_CREDENTIALS="$CLOUDSDK_CONFIG/application_default_credentials.json"
```

### Quota project

The library sends the quota project in the `x-goog-user-project` header. The proxy does not add one itself; it sends what the library produces:

- `GOOGLE_CLOUD_QUOTA_PROJECT`, when set, wins for every credential type.
- Otherwise the file's `quota_project_id` is used for `authorized_user`, `external_account_authorized_user`, and `impersonated_service_account` credentials.
- For `service_account` and `external_account` credentials the library ignores the file's `quota_project_id`; set `GOOGLE_CLOUD_QUOTA_PROJECT` if you need one.

To store a quota project in your user credentials:

```sh
gcloud auth application-default set-quota-project PROJECT_ID
```

### `vertex auth status`

```sh
claude-code-proxy vertex auth status
```

The command requests one access token and prints:

- `Credentials:` the file it read and whether it came from `GOOGLE_APPLICATION_CREDENTIALS` or the default location, or that the library falls back to the metadata server
- `Credential type:` the file's `type`, or `metadata server`
- a `Note:` line when `CLOUDSDK_CONFIG` points elsewhere and `GOOGLE_APPLICATION_CREDENTIALS` is not set, with the `GOOGLE_APPLICATION_CREDENTIALS=...` value to set
- `Project:`, `Location:`, and `Endpoint:` with the setting each value came from
- `Quota project:` and its source, or `none`
- `Access token obtained: true`, or `false` with the reason

It reads only the `type` and `quota_project_id` fields of the credentials file and never prints a token, key, or secret. Without a credentials file it contacts the metadata server and can wait up to 30 seconds for an answer.

| Exit status | Meaning |
| --- | --- |
| 0 | A project is set, every setting is valid, and an access token was obtained. |
| 1 | No credential was found or no token was obtained. After the report it prints `Not authenticated`, and the gcloud hint on stderr. |
| 2 | A Vertex setting is missing or invalid. After the report it prints the error, which names the setting. This wins when credentials are also missing. |

`vertex auth login` and `vertex auth device` print the `gcloud auth application-default login` hint on stderr and exit with status 2. `vertex auth logout` removes nothing: it fails with a message that gcloud manages the credentials and points at `gcloud auth application-default revoke`, and exits with status 1.

## Project and location

The project has no default, because a guessed project would bill the wrong one. Each value comes from the first source that sets it to something other than blank:

| Value | Sources, in order | Default |
| --- | --- | --- |
| Project | `CCP_VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `vertex.project` in `config.json` | none |
| Location | `CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION`, `vertex.location` in `config.json` | `global` |
| Base URL | `CCP_VERTEX_BASE_URL`, `vertex.baseUrl` in `config.json` | derived from the location |

Reading the Google Cloud variables lets a shell already set up for gcloud or the Gemini CLI work unchanged. The `CCP_` variables point the proxy at another project or location without changing them. When the project comes from `GOOGLE_CLOUD_PROJECT`, the startup summary lists `vertex.project (Google env)`, so a project picked up from the shell is visible.

Set the project for one shell:

```sh
export CCP_VERTEX_PROJECT=PROJECT_ID
```

The location selects the host:

| Location | Host |
| --- | --- |
| `global` | `aiplatform.googleapis.com` |
| `us`, `eu` | `aiplatform.us.rep.googleapis.com`, `aiplatform.eu.rep.googleapis.com` |
| any region, such as `us-central1` | `us-central1-aiplatform.googleapis.com` |

A project ID may contain letters, digits, `.`, `_`, `:`, and `-`, starting with a letter or digit. A location may contain lowercase letters, digits, and `-`. A value that does not fit is an error naming its source, for example:

```text
Invalid Vertex setting CCP_VERTEX_LOCATION: must be a Vertex AI location such as global, us-central1 or europe-west4: lowercase letters, digits and '-'
```

The base URL receives the access token, so it must use `https` (plain `http` only for a loopback host) and must not contain a user name or password. It replaces `https://<host>` and keeps its own path prefix; `/v1/projects/...` is appended after it, and its query and fragment are dropped.

The server reads these settings on the first Vertex AI request and keeps them until it exits, so restart the proxy after changing the environment or `config.json`. When the settings are invalid, for example because no project is set, nothing is kept, and the next request reads `config.json` again.

Credentials are kept only once they have produced an access token. Until then, every request looks for them again, so after a 401 from a proxy that has not obtained a token yet, running `gcloud auth application-default login` or fixing the file named by `GOOGLE_APPLICATION_CREDENTIALS` takes effect on the next request. Once a token has been obtained, the proxy keeps those credentials until it exits: restart it after a new login or after changing the credentials file. Setting or changing `GOOGLE_APPLICATION_CREDENTIALS` itself always needs a restart, because the proxy reads its environment only when it starts.

## Start Claude Code

After the credentials and project are set, start the proxy in the same shell:

```sh
claude-code-proxy serve
```

Then start Claude Code against it from another terminal:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gemini-4-argon \
ANTHROPIC_SMALL_FAST_MODEL=gemini-3.5-flash-lite \
  claude
```

## Models

The registered IDs are `gemini-4-argon`, `gemini-4-carbon`, `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`, and `gemini-3-flash-preview`. Each is accepted bare, as `vertex/<model-id>`, and as `vertex/google/<model-id>`.

| Model | Lowest thinking level |
| --- | --- |
| `gemini-4-argon` | `LOW` |
| `gemini-4-carbon` | `LOW` |
| `gemini-3.8-flash` | `LOW` |
| `gemini-3.7-flash` | `LOW` |
| `gemini-3.5-flash` | `MINIMAL` |
| `gemini-3.5-flash-lite` | `MINIMAL` |
| `gemini-3.1-flash-lite` | `MINIMAL` |
| `gemini-3.1-pro-preview` | `LOW` |
| `gemini-3-flash-preview` | `MINIMAL` |

`vertex/<model-id>` also reaches a Gemini model that is not registered, so a new model works before the proxy lists it. A bare ID that is not registered does not route to Vertex AI; only the `vertex/` form does. An unregistered ID gets `LOW` as its lowest thinking level, because every Gemini model that takes a thinking level accepts `LOW`. A model ID starts with a letter or digit and contains only letters, digits, `.`, `_`, `-`, and `@`, up to 256 bytes.

Any other publisher, such as `vertex/anthropic/<model-id>`, is rejected with HTTP 400:

```text
unsupported Vertex AI model "vertex/anthropic/claude-x"; use a Gemini model ID (for example gemini-3.5-flash), vertex/<gemini-model-id> or vertex/google/<gemini-model-id>
```

Whether a model is offered in your location, and whether your project may use it, is decided by Google. A model that is not offered answers with 404 (see [Errors](#errors)).

`max_tokens` is sent as `maxOutputTokens`. Without `max_tokens` the proxy sends 32,000, and values above 65,536 are lowered to 65,536. `temperature`, `top_p`, `top_k`, and `stop_sequences` are sent as they are.

## Thinking

Claude Code's `/effort` setting and its thinking setting map to Gemini's `thinkingLevel`:

| Claude Code | `thinkingLevel` | Thought summaries |
| --- | --- | --- |
| effort `low` | `LOW` | when thinking is on |
| effort `medium` | `MEDIUM` | when thinking is on |
| effort `high`, `xhigh`, or `max` | `HIGH` | when thinking is on |
| no effort, thinking on | not sent; the model uses its default | yes |
| no effort, thinking off or absent | the model's lowest level: `MINIMAL`, or `LOW` | no |

Thinking is on for any thinking type other than `disabled`. `budget_tokens` is not used, and `thinkingBudget` is never sent, because Gemini 3 deprecates it and Gemini 4 rejects it. When thinking is off, the proxy sends the lowest level the model accepts rather than omitting the setting. An effort value other than the five above is rejected with HTTP 400.

Thought summaries appear as Claude Code thinking blocks only when thinking was on for the request; otherwise they are dropped. A model may answer without thinking, and then sends no summary.

## Tools and thought signatures

Claude function tools become Gemini function declarations. Each tool's `input_schema` is sent unchanged as `parametersJsonSchema`. A tool name must start with a letter or `_`, contain only letters, digits, `_`, `.`, `:`, and `-`, and have at most 128 characters; any other name is rejected with HTTP 400 naming the tool. Anthropic server tools, such as `web_search_20250305`, are rejected with HTTP 400.

| `tool_choice` | Gemini function-calling mode |
| --- | --- |
| `auto` | `AUTO` |
| `any` | `ANY` |
| `none` | `NONE` |
| `tool` with a name | `ANY`, limited to that function |

`disable_parallel_tool_use` has no Gemini equivalent and is ignored. Function calls and results are sent without IDs; each result is matched to its call by order. Results come first in a user turn, in the order of the calls in the preceding model turn, followed by text and images. A tool result whose `is_error` is true is sent under `error` instead of `output`.

Gemini 3 and later reject a function call in the history that lacks the thought signature the model attached to it. The proxy keeps no state, so the signature travels inside the `signature` field of a Claude Code thinking block, as `ccp:vertex:v1:<model>:<signature>`, and returns with the history on the next request. When a function call arrives without an open thinking block, the proxy adds an empty thinking block to carry its signature.

On the next request the proxy sends each signature back on the function call it came with. The thinking text itself is never sent. When the first function call of a model turn has no signature from the same model, for example after switching from another provider or model, when the thinking block was dropped from the history, or when the signature was over 1 MiB, the proxy sends Google's documented placeholder `skip_thought_signature_validator`. Google describes it as a last resort that can lower answer quality. Signatures on text parts are dropped.

## Images

Base64 images in user messages are sent as inline data with their media type. Images inside a tool result stay with that result, inside its function response, so parallel results keep their own images. Image URLs are rejected with HTTP 400: `Vertex AI Gemini does not accept image URLs; send the image as base64 data`.

## Stop reasons and usage

`STOP` ends the turn with `end_turn`, or with `tool_use` when the response contained a function call. `MAX_TOKENS` ends it with `max_tokens`; function calls sent before it are complete, because each one arrives whole. Any other `finishReason` is an error (see [Errors](#errors)), even when tool calls were already sent.

Usage comes from Gemini's `usageMetadata`: `input_tokens` is `promptTokenCount` minus `cachedContentTokenCount`, `cache_read_input_tokens` is `cachedContentTokenCount`, and `output_tokens` is `candidatesTokenCount` plus `thoughtsTokenCount`. The `input_tokens` in `message_start` is the proxy's local estimate, because it is sent before Gemini reports usage.

## OpenAI-compatible APIs

When the OpenAI routes are enabled, Vertex AI models work with both `POST /v1/chat/completions` and `POST /v1/responses`. Text, images, function tools, tool results, token limits, streaming, and usage use the standard shape for the chosen route. `reasoning_effort` and `reasoning.effort` accept `low`, `medium`, `high`, `xhigh`, and `max`, mapped as in [Thinking](#thinking). These routes return no thinking signatures, so a function call replayed through them carries the `skip_thought_signature_validator` placeholder. `parallel_tool_calls` is accepted and has no effect.

## Errors

The proxy chooses the error type from the HTTP status of Vertex AI. When Vertex AI answered the request with an error status, the message ends with `(upstream HTTP <status>)`, even when the proxy answers with a different status. Google's error text is cleaned first: the access token, e-mail addresses, and long runs of base64 data (a thought signature or image the request carried) are replaced, and the text is cut to 500 characters.

| Situation | Proxy answer |
| --- | --- |
| Project missing or a setting invalid | 500 `api_error`, `Invalid Vertex configuration: ...` naming the setting |
| HTTP client could not be set up, which happens on the first Vertex AI request rather than at startup | 500 `api_error`, `Vertex AI transport is unavailable: ...` with the cause |
| No credential, or no access token (including no answer within 30 seconds) | 401 `authentication_error`, `Google credentials are not available: ...` or `Google access token could not be obtained: ...`, followed by the `gcloud auth application-default login` hint |
| Vertex AI 401 | 401 `authentication_error`, `Vertex AI rejected the Google access token: ...` with the same hint |
| Vertex AI 403 | 403 `permission_error`. The message asks you to check the role `roles/aiplatform.user` on the project, that `aiplatform.googleapis.com` is enabled there, and, for user credentials, that a quota project is set with `gcloud auth application-default set-quota-project <project>` |
| Vertex AI 404 | 404 `invalid_request_error`, `Model <model> is not available in project <project> / location <location>: ...`, naming the three location settings |
| Vertex AI 429 | 429 `rate_limit_error`. The message names the model and location and suggests retrying later or another location. `retry-after` is set from Google's `RetryInfo`, or from a whole-second `Retry-After` header, when it is between 0 and 86400 seconds |
| Other Vertex AI 4xx | Same status, `invalid_request_error`, with Google's message |
| Vertex AI 5xx, or no connection | 502 `api_error`; `Vertex AI is unreachable` when no status arrived |
| No response headers within 300 seconds | 504 `api_error` |
| Error inside a 200 stream | A blocked prompt (`blockReason`), a stop other than `STOP` or `MAX_TOKENS` (`finishReason`, with Google's `finishMessage`), a stream that ends without a `finishReason`, or an `error` chunk. Streaming requests get an `error` event. Non-streaming requests get the matching status: 429 for rate limits, 401 and 403 for credential errors, the in-stream 4xx code or 400 for invalid requests and blocked prompts, and 502 for the rest |

What to check for each status:

- **401**: the credential is expired or revoked. Run `gcloud auth application-default login` again, or check the key file named by `GOOGLE_APPLICATION_CREDENTIALS`, then restart the proxy, because it keeps the credentials from the first request that obtained an access token. If no request has obtained one since the proxy started, no restart is needed: the next request looks for the credentials again.
- **403**: the account lacks `roles/aiplatform.user` on the project, the Vertex AI API is not enabled in it, or user credentials have no quota project.
- **404**: the model ID is wrong, or the model is not offered in that location. Try `global`, or a region where the model page lists it.
- **429**: the project's quota or Google's shared capacity for the model in that location is used up. Wait for `retry-after`, or set another location.

The proxy does not retry and does not fall back to another model; Claude Code decides whether to retry. Non-streaming requests are also sent to Vertex AI as streams and assembled by the proxy, so both kinds end the same way. There is no limit on the time between chunks of a stream, because a model can think for a long time before it sends text. The body of an error response is read for at most 10 seconds and 64 KiB; past that, the proxy answers from the status alone.

## Configuration

| Environment | Config key | Purpose |
| --- | --- | --- |
| `CCP_VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT` | `vertex.project` | Google Cloud project. Required. |
| `CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION` | `vertex.location` | Vertex AI location. Default `global`. |
| `CCP_VERTEX_BASE_URL` | `vertex.baseUrl` | Replaces the host derived from the location, for a relay or a local test server. |
| `GOOGLE_APPLICATION_CREDENTIALS` | none | Credentials file, read by the Google library. |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | none | Quota project, read by the Google library. |

An empty value counts as not set. See [Configuration](/reference/configuration/#vertex-ai) for the full table.

## Implemented from documentation only

No request to Vertex AI was observed while this provider was built. The request and response formats come from Google's `v1` API reference and the Gemini model pages. The following points are assumptions; each one shows a specific symptom if it is wrong. A [traffic capture](/reference/files-and-storage/#traffic-captures) of the failing request helps when reporting one.

| Assumption | Symptom if wrong |
| --- | --- |
| Gemini matches function results to calls by order, without IDs. | HTTP 400 on turns with parallel tool calls. |
| Gemini accepts Claude Code's tool schemas, including MCP tools, unchanged. | Every request in a session with a certain tool fails with HTTP 400 naming a schema field. |
| Every registered model accepts images inside a function response. | HTTP 400 mentioning `function_response.parts` after a tool returns an image. |
| `candidatesTokenCount` excludes thinking tokens, so the proxy adds `thoughtsTokenCount` to the output count. | Output token counts are higher than Google bills. |
| The `us` and `eu` multi-region hosts serve Gemini. | HTTP 404 with those locations. Use `global` or a region, or set `CCP_VERTEX_BASE_URL`. |
| User credentials without a quota project are accepted. | HTTP 403. Run `gcloud auth application-default set-quota-project PROJECT_ID`. |
| Claude Code keeps empty thinking blocks that carry only a signature. | No error; the proxy sends the placeholder signature, and answer quality may drop. |
| Error bodies carry `RetryInfo`, and errors inside a stream arrive as a chunk with an `error` key. | 429 responses have no `retry-after`. An unknown chunk shape ends the stream with an error. |

## Limitations

- Only Gemini on Vertex AI is supported. Claude, partner and open models, and deployed endpoints are not, and the proxy does not enable models or create endpoints.
- Gemini's built-in tools (Google Search, code execution, URL context) and Anthropic server tools are not available. Structured output (`output_config.format`), `cache_control`, and `metadata` are ignored.
- PDF and other document blocks are dropped. Image URLs are rejected. Responses that contain images, files, or code-execution parts end with an error.
- `thinkingBudget` is never sent, so models that take only `thinkingBudget`, such as Gemini 2.5, are not supported.
- Safety settings are Google's defaults; the proxy sends none.
- Token counts for `/v1/messages/count_tokens` are local estimates; Vertex AI's `countTokens` is not used.
- Express mode (API keys), the Interactions API, explicit context caches, and Provisioned Throughput are not supported.

If requests fail, run `claude-code-proxy vertex auth status` first. It shows which credential, project, location, endpoint, and quota project the proxy resolves, and whether a token can be obtained.
