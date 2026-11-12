---
title: Vertex AI
description: Use Gemini, Anthropic Claude, xAI Grok, OpenAI, Meta Muse, and Mistral models on Google Cloud Vertex AI with Application Default Credentials, project, location and per-model location settings, errors, and limits.
---

The Vertex AI provider sends requests to Google Cloud Vertex AI for six model families. Each family uses the API Google documents for it:

| Family | Model names | Upstream API |
| --- | --- | --- |
| [Gemini](#gemini) | `gemini-*` IDs, bare or as `vertex/<model-id>` or `vertex/google/<model-id>` | `publishers/google/models/<model>:streamGenerateContent` |
| [Anthropic Claude](#anthropic-claude) | `vertex/anthropic/<model-id>` | Anthropic Messages on `publishers/anthropic/models/<model>:streamRawPredict` or `:rawPredict` |
| [xAI Grok](#xai-grok) | `vertex/xai/<model-id>` | OpenAI Responses on `endpoints/openapi/responses` |
| [OpenAI](#openai) | `vertex/openai/gpt-<…>` | OpenAI Responses on `publishers/openai/v1/responses` |
| [Meta Muse](#meta-muse) | `vertex/meta/muse-<…>` | OpenAI Responses on `publishers/meta/v1/responses` |
| [Mistral](#mistral) | `vertex/mistralai/<model-id>` | Mistral's chat format on `publishers/mistralai/models/<model>:streamRawPredict` |

Every path follows `https://<host>/v1/projects/<project>/locations/<location>/`. For example, a Gemini request goes to:

```text
https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/publishers/google/models/<model>:streamGenerateContent?alt=sse
```

<div class="security-callout">
<strong>Requests are billed to your Google Cloud project.</strong> Vertex AI has no subscription. Every request is charged at Vertex AI's pay-as-you-go rates to the project the proxy is configured with, using your own Google credentials with the `cloud-platform` scope. This holds for partner models too: Claude, Grok, OpenAI, Muse, and Mistral requests are billed through the same project. A listener that other people can reach can spend that project's money.
</div>

Model Garden's managed open models (`vertex/<publisher>/<model-id>-maas`) and deployed endpoints (`vertex/endpoint/<name>`) are not supported; see [Models](#models).

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

### Partner models in Model Garden

Gemini needs nothing more. Claude, Grok, OpenAI, Muse, and Mistral models must each be enabled in Model Garden before the project can call them. Open the model's page in Model Garden in the Google Cloud console and enable it there. Google's documentation names these requirements:

- Enabling a model needs the role `roles/consumerprocurement.entitlementManager`. Calling it needs the permission `aiplatform.endpoints.predict`, which `roles/aiplatform.user` includes.
- Organization policies must allow the Consumer Procurement service (`cloudcommerceconsumerprocurement.googleapis.com`) and the model.
- The OpenAI models are in Private Preview. They also need Google's preview onboarding and allowlisting for the project.

Grant the enabling role the same way as the request role:

```sh
gcloud projects add-iam-policy-binding PROJECT_ID --member=user:USER_EMAIL --role=roles/consumerprocurement.entitlementManager
```

The proxy does not enable models and cannot tell whether one is enabled. A request for a model that is not enabled gets Google's error with its status; [Errors](#errors) lists what to check for 403 and 404.

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

The credentials file is trusted input: the token endpoint and other URLs the file names receive the credentials the library sends them, which is the refresh token and client secret for user credentials, or an assertion signed with the key for a service account (the key itself signs locally and is never sent). The proxy does not check those endpoints, so whoever can write the file decides where those credentials go.

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

The command does not read the [location overrides](#location-overrides) and does not check that a partner model is enabled. A malformed override is reported by the first Vertex AI request instead.

| Exit status | Meaning |
| --- | --- |
| 0 | A project is set, the project, location, and base URL settings are valid, and an access token was obtained. |
| 1 | No credential was found or no token was obtained. After the report it prints `Not authenticated`, and the gcloud hint on stderr. |
| 2 | A Vertex setting is missing or invalid. After the report it prints the error, which names the setting. This wins when credentials are also missing. |

`vertex auth login` and `vertex auth device` print the `gcloud auth application-default login` hint on stderr and exit with status 2. `vertex auth logout` removes nothing: it fails with a message that gcloud manages the credentials and points at `gcloud auth application-default revoke`, and exits with status 1.

## Project and location

The project has no default, because a guessed project would bill the wrong one. Each value comes from the first source that sets it to something other than blank:

| Value | Sources, in order | Default |
| --- | --- | --- |
| Project | `CCP_VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `vertex.project` in `config.json` | none |
| Location | `CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION`, `vertex.location` in `config.json` | `global` |
| Location overrides | `CCP_VERTEX_LOCATION_OVERRIDES`, `vertex.locationOverrides` in `config.json` | none |
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

The base URL receives the access token, so it must use `https` (plain `http` only for a loopback host) and must not contain a user name or password. It replaces `https://<host>` and keeps its own path prefix; `/v1/projects/...` is appended after it, and its query and fragment are dropped. A base URL replaces the host for every model, whatever location the model's request uses.

The server reads these settings on the first Vertex AI request and keeps them until it exits, so restart the proxy after changing the environment or `config.json`. When the settings are invalid, for example because no project is set or an override is malformed, nothing is kept, and the next request reads `config.json` again.

Credentials are kept only once they have produced an access token. Until then, every request looks for them again, so after a 401 from a proxy that has not obtained a token yet, running `gcloud auth application-default login` or fixing the file named by `GOOGLE_APPLICATION_CREDENTIALS` takes effect on the next request. Once a token has been obtained, the proxy keeps those credentials until it exits: restart it after a new login or after changing the credentials file. Setting or changing `GOOGLE_APPLICATION_CREDENTIALS` itself always needs a restart, because the proxy reads its environment only when it starts.

### Location overrides

Google serves each model family in different locations, so one location rarely fits every model. A location override sets the location for one publisher or one model. The key is `<publisher>` or `<publisher>/<model-id>`, with the model ID written as in the model name after `vertex/<publisher>/`, including any `@<version>` suffix. Gemini models use the publisher `google`.

In the environment, entries are `<key>=<location>`, separated by commas:

```sh
export CCP_VERTEX_LOCATION_OVERRIDES=anthropic=global,mistralai=europe-west4
```

In `config.json`, the overrides are an object:

```json
{
  "vertex": {
    "project": "YOUR_GOOGLE_CLOUD_PROJECT_ID",
    "location": "us-central1",
    "locationOverrides": {
      "anthropic": "global",
      "anthropic/claude-opus-4-6": "us-east5",
      "mistralai": "europe-west4"
    }
  }
}
```

`CCP_VERTEX_LOCATION_OVERRIDES` replaces the file's overrides as a whole; the two are not merged. A blank value, or one with only empty entries, counts as not set. The startup summary lists `vertex.locationOverrides (env)` when the variable is used, and `vertex.locationOverrides: <key>=<location>, ...` when the file is.

For each request the location comes from the first rule that applies:

1. An override that names the model, such as `anthropic/claude-opus-4-6`.
2. A fixed location: Meta Muse models always use `global`.
3. An override that names the publisher, such as `anthropic`.
4. The configured location (`CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION`, `vertex.location`). For Mistral it must be `us-central1` or `europe-west4`; any other value is a 400 (see [Mistral](#mistral)).

A publisher override therefore does not move a Muse model; only an override naming the model does.

An override is checked on the first Vertex AI request, before credentials are looked up. A key that is not `<publisher>` or `<publisher>/<model-id>`, a key listed twice, the key `endpoint`, a location that fails the location check, an entry without a location, or a `vertex.locationOverrides` value that is not an object is an error. Every Vertex AI request then fails with HTTP 500, for every family, Gemini included:

```text
Invalid Vertex configuration: Invalid Vertex setting CCP_VERTEX_LOCATION_OVERRIDES: entry "mistralai" has no location; an entry is <key>=<location> in the variable and "<key>": "<location>" in config.json
```

When the overrides come from the file, the message names `vertex.locationOverrides in config.json` instead. A key that fails the check is referred to by its position, such as `entry 2`, rather than quoted.

## Start Claude Code

After the credentials and project are set, start the proxy in the same shell:

```sh
claude-code-proxy serve
```

Then start Claude Code against it from another terminal. With Gemini:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gemini-4-argon \
ANTHROPIC_SMALL_FAST_MODEL=gemini-3.5-flash-lite \
  claude
```

With Claude on Vertex AI, after enabling both models in Model Garden:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=vertex/anthropic/claude-opus-5-5 \
ANTHROPIC_SMALL_FAST_MODEL=vertex/anthropic/claude-haiku-5-5 \
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
  claude
```

`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` is explained in [Claude beta headers](#claude-beta-headers). Inside a session, `/model vertex/xai/grok-4.7` and the other names below switch families; see [Thinking blocks from other providers](#thinking-blocks-from-other-providers) before switching to Claude in the middle of a conversation.

### Try each family

Each command sends one short request through a running proxy. A JSON object with `"type":"message"` means the model is enabled, reachable in its location, and answering. Mistral needs a location first, for example by starting the proxy with `CCP_VERTEX_LOCATION_OVERRIDES=mistralai=us-central1`.

Claude:

```sh
curl -sS http://127.0.0.1:18765/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"vertex/anthropic/claude-haiku-5-5","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: ok"}]}'
```

Grok:

```sh
curl -sS http://127.0.0.1:18765/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"vertex/xai/grok-4.7","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: ok"}]}'
```

OpenAI, when the project has Private Preview access:

```sh
curl -sS http://127.0.0.1:18765/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"vertex/openai/gpt-6-luna","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: ok"}]}'
```

Muse:

```sh
curl -sS http://127.0.0.1:18765/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"vertex/meta/muse-spark-1.3","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: ok"}]}'
```

Mistral:

```sh
curl -sS http://127.0.0.1:18765/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"vertex/mistralai/mistral-medium-3","max_tokens":64,"messages":[{"role":"user","content":"Reply with the single word: ok"}]}'
```

## Models

`claude-code-proxy models` lists every registered name on its `vertex:` line:

| Family | Registered names |
| --- | --- |
| Gemini | `gemini-4-argon`, `gemini-4-carbon`, `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, each also as `vertex/<model-id>` |
| Anthropic Claude | `vertex/anthropic/claude-fable-5-2`, `vertex/anthropic/claude-fable-5-1`, `vertex/anthropic/claude-fable-5`, `vertex/anthropic/claude-opus-5-5`, `vertex/anthropic/claude-opus-5`, `vertex/anthropic/claude-opus-4-8`, `vertex/anthropic/claude-opus-4-7`, `vertex/anthropic/claude-sonnet-5-5`, `vertex/anthropic/claude-sonnet-5`, `vertex/anthropic/claude-haiku-5-5`, `vertex/anthropic/claude-opus-4-6`, `vertex/anthropic/claude-sonnet-4-6`, `vertex/anthropic/claude-opus-4-5@20251101`, `vertex/anthropic/claude-haiku-4-5@20251001` |
| xAI Grok | `vertex/xai/grok-4.7`, `vertex/xai/grok-4.6`, `vertex/xai/grok-4.20-reasoning`, `vertex/xai/grok-4.20-non-reasoning`, `vertex/xai/grok-4.3` |
| OpenAI | `vertex/openai/gpt-6-astra`, `vertex/openai/gpt-6.1-sol`, `vertex/openai/gpt-6.1-terra`, `vertex/openai/gpt-6-luna` |
| Meta Muse | `vertex/meta/muse-spark-1.3`, `vertex/meta/muse-wonder-1.0`, `vertex/meta/muse-ember-1.0` |
| Mistral | `vertex/mistralai/mistral-medium-3`, `vertex/mistralai/mistral-small-2503`, `vertex/mistralai/codestral-2` |

Only Gemini IDs are accepted bare. A bare `claude-*`, `gpt-*`, `grok-*`, or `muse-*` ID keeps its own provider and never reaches Vertex AI: `claude-opus-5-5` is a Codex alias, `gpt-6.1-sol` is Codex, `grok-4.7` is Grok, and `muse-spark-1.3` is Muse. The `vertex/<publisher>/` prefix is what selects Vertex AI. A trailing `[1m]` is removed before routing, as for every model.

An ID that is not registered still reaches Vertex AI when it matches its family's form, so a new model works before the proxy lists it:

| Form | Family |
| --- | --- |
| `vertex/<model-id>`, `vertex/google/gemini-<…>` | Gemini |
| `vertex/anthropic/<model-id>` | Anthropic Claude |
| `vertex/xai/<model-id>` | xAI Grok |
| `vertex/openai/gpt-<…>`, unless it ends in `-maas` | OpenAI |
| `vertex/meta/muse-<…>`, unless it ends in `-maas` | Meta Muse |
| `vertex/mistralai/<model-id>` | Mistral |

A name without a publisher is always Gemini: `vertex/gpt-6.1-sol` is sent to Gemini and fails upstream, while `vertex/openai/gpt-6.1-sol` reaches OpenAI. An unregistered ID gets its family's defaults: Gemini's lowest thinking level `LOW` (see [Thinking](#thinking)) and the family's location rule. A model ID starts with a letter or digit and contains only letters, digits, `.`, `_`, `-`, and `@`, up to 256 bytes.

Managed open models and deployed endpoints are recognised and rejected with HTTP 400 before any setting or credential is read:

```text
this version of claude-code-proxy does not support the Model Garden managed open model (<publisher>/<model-id>-maas) model family on Vertex AI (requested "vertex/openai/gpt-oss-120b-maas"); supported families: Gemini, Anthropic Claude, xAI Grok, OpenAI, Meta Muse, Mistral
```

A deployed endpoint gets the same message with `deployed endpoint (endpoint/<name>)` as the family. Any other form, such as `vertex/ai21/<model-id>` or `vertex/meta/llama-<…>`, gets HTTP 400 with a message listing the accepted forms.

Whether a model is offered in your location, and whether your project may use it, is decided by Google. A model that is not offered or not enabled answers with Google's error (see [Errors](#errors)).

`/v1/messages/count_tokens` is answered with the proxy's local estimate for every family; nothing is sent to Vertex AI.

## Gemini

Gemini requests are translated to `streamGenerateContent`. The registered Gemini models and their lowest thinking levels:

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

A bare ID that is not registered does not route to Vertex AI; only the `vertex/` form does. An unregistered ID gets `LOW` as its lowest thinking level, because every Gemini model that takes a thinking level accepts `LOW`. Gemini uses the configured location, or an override for `google` or `google/<model-id>`.

`max_tokens` is sent as `maxOutputTokens`. Without `max_tokens` the proxy sends 32,000, and values above 65,536 are lowered to 65,536. `temperature`, `top_p`, `top_k`, and `stop_sequences` are sent as they are.

### Thinking

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

### Tools and thought signatures

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

### Images

Base64 images in user messages are sent as inline data with their media type. Images inside a tool result stay with that result, inside its function response, so parallel results keep their own images. Image URLs are rejected with HTTP 400: `Vertex AI Gemini does not accept image URLs; send the image as base64 data`.

### Stop reasons and usage

`STOP` ends the turn with `end_turn`, or with `tool_use` when the response contained a function call. `MAX_TOKENS` ends it with `max_tokens`; function calls sent before it are complete, because each one arrives whole. Any other `finishReason` is an error (see [Errors](#errors)), even when tool calls were already sent.

Usage comes from Gemini's `usageMetadata`: `input_tokens` is `promptTokenCount` minus `cachedContentTokenCount`, `cache_read_input_tokens` is `cachedContentTokenCount`, and `output_tokens` is `candidatesTokenCount` plus `thoughtsTokenCount`. The `input_tokens` in `message_start` is the proxy's local estimate, because it is sent before Gemini reports usage.

## Anthropic Claude

Claude requests are passed through as Anthropic Messages. A streaming request goes to `:streamRawPredict` and a non-streaming one to `:rawPredict`:

```text
https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/publishers/anthropic/models/<model>:streamRawPredict
```

The model ID goes into the URL as written, including an `@<date>` suffix such as `claude-opus-4-5@20251101`. Every registered Claude model is generally available on Vertex AI, according to Google's model pages.

### Claude location

Claude uses the configured location, or an override for `anthropic` or `anthropic/<model-id>`. According to Google and Anthropic, Claude 4.7 and later (Fable, Opus 4.7, 4.8, 5 and 5.5, Sonnet 5 and 5.5, Haiku 5.5) are served only from `global` and the `us` and `eu` multi-regions, and Opus 4.6, Sonnet 4.6, Opus 4.5, and Haiku 4.5 from `global` and some regions such as `us-east5`, not from `us` or `eu`. The proxy does not check the combination; a location Google does not serve for the model gets Google's error.

When the configured location is a region for Gemini, send Claude to `global`:

```sh
export CCP_VERTEX_LOCATION_OVERRIDES=anthropic=global
```

### Claude request and response

The proxy changes only these parts of the request:

- `model` is removed, because the URL names the model.
- `anthropic_version: "vertex-2023-10-16"` is added when the request has none. A value the client sends, such as `2026-10-21`, is kept.
- `max_tokens: 32000` is added when the request has none.
- Thinking blocks Claude cannot verify are removed (see [Thinking blocks from other providers](#thinking-blocks-from-other-providers)).
- The `anthropic-beta` headers are filtered and joined (see [Claude beta headers](#claude-beta-headers)).

Everything else, including `stream`, `system`, `tools`, `tool_choice`, `thinking`, `output_config` with the effort, `cache_control`, and `metadata`, is sent as the client sent it. Claude Code's thinking and `/effort` settings therefore reach Claude unchanged.

A streaming response is forwarded byte for byte. The proxy checks it as it passes: an upstream `error` event, a stream that ends before `message_stop`, or any event after `message_stop`, `ping` included, ends the stream with an `api_error` event, `Vertex AI Messages stream is invalid`. Google's own error text is not forwarded to the client; the [traffic capture](/reference/files-and-storage/#traffic-captures) records it. The chunk that carries `message_stop` is held until Vertex AI closes the stream or 2 seconds pass, so the end of a turn can arrive up to 2 seconds late. More than 16 MiB after `message_stop` is an error.

A non-streaming response is read whole, up to 16 MiB, and returned unchanged once it parses as JSON. A larger body is a 502, `Vertex AI Claude response exceeds 16 MiB`, and a body that is not JSON is a 502, `Vertex AI returned a Claude response that is not JSON`.

`/v1/messages/count_tokens` is the proxy's local estimate; Vertex AI's Claude token-counting endpoint is not used.

### Claude beta headers

Google's Claude reads only one `anthropic-beta` header and ignores any others. The proxy collects the values of every `anthropic-beta` header the client sent, keeps those on the list below, removes duplicates, and sends them in one header, separated by commas. When no value is left, no header is sent.

A value that is not on the list is dropped silently: the request is not refused, and the proxy writes one debug log line per request (`Vertex AI Claude request: beta values outside the Vertex list were not sent`) with the dropped values. The list is the one Claude Code 2.1.322 applies itself when it talks to Vertex AI directly:

- `claude-code-20250219`
- `interleaved-thinking-2025-05-14`
- `context-1m-2025-08-07`
- `context-management-2025-06-27`
- `structured-outputs-2025-12-15`
- `web-search-2025-03-05`
- `effort-2025-11-24`
- `tool-search-tool-2025-10-19`
- `afk-mode-2026-01-31`
- `dangerous-tool-use-2026-09-03`
- `fallback-credit-2026-06-01`
- `thinking-token-count-2026-05-13`
- `thinking-binding-controls-2026-08-01`
- `thinking-display-updates-2026-08-18`

Behind the proxy, Claude Code believes it talks to Anthropic and sends its full set, including values such as `oauth-2025-04-20` that are not on the list. Only the header is filtered: request fields that belong to a dropped beta are still sent, and Vertex AI may refuse them. Starting Claude Code with `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` makes it send fewer betas and the fields that go with them.

A body field named `anthropic_beta` is not read and is sent as received. On the [OpenAI-compatible routes](#openai-compatible-apis) no `anthropic-beta` header is sent.

### Thinking blocks from other providers

Claude verifies the signature of every thinking block in the history. Before sending, the proxy removes the thinking blocks Claude cannot verify:

- blocks whose signature this proxy made, which starts with `ccp:` or decodes from base64url to bytes starting with `ccp:`. Codex, Muse, Kimi, OpenCode Go, and Vertex AI Gemini produce these.
- blocks whose signature is missing, empty, or blank, which other providers behind the proxy produce.

Thinking blocks with Anthropic's signature and `redacted_thinking` blocks are sent unchanged. An assistant message that held only removed blocks is left out.

When a conversation switches to Claude from another provider, for example with `/model`, the earlier thinking is therefore lost; text, tool calls, and tool results remain. If Claude refuses such a history, for example because a turn with thinking enabled must start with a thinking block, start a new conversation with the Claude model.

### Features Vertex AI does not accept

Google and Anthropic document that Claude on Vertex AI does not support the Files API, URL sources for images and documents, code execution, web fetch, the advisor tool, the MCP connector, the Message Batches API, and server-side `fallbacks`. Only the basic web search tool `web_search_20250305` is available, without dynamic filtering. Claude 4.7 and later models reject `thinking.type: "enabled"` with `budget_tokens` with a 400; they take adaptive thinking. A request is limited to 30 MB, an image to 5 MB, and a request to 100 images.

The proxy checks none of these. A request that uses one gets Vertex AI's error with its status (see [Errors](#errors)).

## Grok, OpenAI, and Muse

xAI Grok, OpenAI, and Meta Muse models are reached through the OpenAI Responses API. The proxy translates the Anthropic request with the same Responses translator as OpenCode Go, then edits it for Vertex AI:

| Request part | xAI Grok | OpenAI | Meta Muse |
| --- | --- | --- | --- |
| URL | `endpoints/openapi/responses` | `publishers/openai/v1/responses` | `publishers/meta/v1/responses` |
| `model` | `xai/<model-id>` | the bare model ID | the bare model ID |
| `store` | `false` | `false` | `false` |
| `reasoning.effort` | not sent | Claude Code's effort | not sent |
| `reasoning.summary`, `include`, `text.verbosity`, `prompt_cache_key` | not sent | not sent | not sent |
| `max_output_tokens` | the request's `max_tokens` | the request's `max_tokens`, at most 128,000 | the request's `max_tokens` |
| Thinking blocks in the history | not sent | not sent | not sent |
| Hosted web search tool | refused with 400 | refused with 400 | refused with 400 |

Instructions, messages, images, function tools, `tool_choice`, `parallel_tool_calls`, and structured output (`text.format`) are sent as the translator produces them. Like OpenCode Go, the translator also rewrites the description and arguments of Claude Code's Read tool and normalises tool schemas.

The upstream is always asked for a stream. A non-streaming client gets the stream assembled into one message.

### No reasoning across turns

No thinking or `redacted_thinking` block from the history is sent. The translator would send a proxy-signed reasoning block back as encrypted reasoning, and nothing in the signature says which provider, publisher, model, or project made it, so the proxy sends none. Reasoning does not continue from one turn to the next; visible text, tool calls, and tool results do. Reasoning summaries are not requested, so Claude Code shows thinking only if a publisher sends summary text on its own.

### Hosted web search

The translator turns Anthropic's `web_search_20250305` tool into a hosted `web_search` tool, which would run on the publisher's side. The proxy refuses any tool that is not a function, before any setting or credential is read:

```text
Vertex AI does not serve the hosted web_search tool for xAI Grok models; remove the web search tool from the request
```

The message names `OpenAI` or `Meta Muse` for those families. Remove the web search tool from Claude Code's tool set for these models.

### Strict stream checks

Google documents the Responses streams of these publishers only by example. The proxy checks every event before it is translated and stops the stream with an error rather than dropping content it does not recognise. This is a deliberate choice: on a live service, the first symptom of an event the proxy does not know is a stream that ends with `Vertex AI Responses stream is invalid`.

The stream fails when an event:

- is not JSON, or has no string `type`.
- has a `type` the translator does not handle and the proxy has not classified as harmless. The accepted types are the ones the translator handles, such as `response.created`, `response.output_item.added`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`, `response.failed`, and `error`, plus the progress events `response.queued`, `response.content_part.added`, `response.content_part.done`, `response.output_text.done`, `response.reasoning_summary_part.done`, and `response.reasoning_summary_text.done`. `ping`, `response.refusal.*`, and `response.reasoning_text.*` are not accepted.
- is a `.done` snapshot or a `response.output_item.done` that carries text no delta delivered, because the translator would drop that text.
- arrives after `response.completed`, `response.incomplete`, or `response.done`.

After the completion event the proxy reads until Vertex AI closes the stream or 2 seconds pass, and accepts at most 16 MiB more. A single event is limited to 1 MiB.

A streaming client gets an `api_error` event with `Vertex AI Responses stream is invalid`. A non-streaming client gets a 502 whose message adds the reason, such as `Vertex AI Responses stream is invalid: unlisted_event_type`. The [traffic capture](/reference/files-and-storage/#traffic-captures) records the event and the reason in `060-vertex-responses-stream-error`. Report the capture when this happens, so the event can be added to the accepted list.

### xAI Grok

| Item | Detail |
| --- | --- |
| Names | `vertex/xai/grok-4.7`, `vertex/xai/grok-4.6`, `vertex/xai/grok-4.20-reasoning`, `vertex/xai/grok-4.20-non-reasoning`, `vertex/xai/grok-4.3`, and any `vertex/xai/<model-id>` |
| Status | Generally available; `grok-4.3` is in Preview, according to Google's model pages |
| Location | The configured location, or an override for `xai` or `xai/<model-id>`. Google documents `global` for every model, and `us` for `grok-4.6` and `grok-4.7` |
| Effort | Not sent. Google documents that Grok reasoning models do not take `reasoning_effort` on Chat Completions and does not establish an effort control on Responses, so Claude Code's `/effort` has no effect |

Grok's stored responses on Vertex AI (`store: true`, `previous_response_id`) are a Preview behind an organization policy, according to Google; the proxy does not use them.

### OpenAI

| Item | Detail |
| --- | --- |
| Names | `vertex/openai/gpt-6-astra`, `vertex/openai/gpt-6.1-sol`, `vertex/openai/gpt-6.1-terra`, `vertex/openai/gpt-6-luna`, and any `vertex/openai/gpt-<…>` that does not end in `-maas` |
| Status | Private Preview, according to Google. The project needs preview access and allowlisting besides Model Garden enablement |
| Location | The configured location, or an override for `openai` or `openai/<model-id>`. Google documents `global` and `us` |
| Effort | Sent as `reasoning.effort`: `low`, `medium`, `high`, `xhigh`, or `max`, as Claude Code's effort maps through the translator. Google documents adjustable effort with `medium` as its example |
| Limits | Google documents 922,000 input tokens and 128,000 output tokens; `max_output_tokens` is lowered to 128,000 |

Google documents that reasoning summaries and hosted tools (web search, file search, code interpreter, vector stores, remote MCP servers) are not available for these models. Google also documents encrypted reasoning that carries across turns; this version does not use it.

### Meta Muse

| Item | Detail |
| --- | --- |
| Names | `vertex/meta/muse-spark-1.3`, `vertex/meta/muse-wonder-1.0`, `vertex/meta/muse-ember-1.0`, and any `vertex/meta/muse-<…>` that does not end in `-maas` |
| Status | Preview, according to Google's model pages |
| Location | Always `global`, the only location Google documents. The configured location and a `meta` override do not change it; only an override naming the model, such as `meta/muse-spark-1.3`, does |
| Effort | Not sent; Google does not document an effort control for Muse |
| Limits | Google documents 1,048,576 input tokens. `store: true` is not supported, which is why the proxy always sends `false` |

These are Meta's models on your Google Cloud project, billed by Google. They are separate from the [Muse provider](/providers/muse/), which uses a Muse Code subscription; the bare `muse-*` IDs belong to that provider.

## Mistral

Mistral requests are sent in Mistral's Chat-Completions-shaped format, always as a stream:

```text
https://us-central1-aiplatform.googleapis.com/v1/projects/<project>/locations/us-central1/publishers/mistralai/models/<model>:streamRawPredict
```

| Item | Detail |
| --- | --- |
| Names | `vertex/mistralai/mistral-medium-3`, `vertex/mistralai/mistral-small-2503`, `vertex/mistralai/codestral-2`, and any `vertex/mistralai/<model-id>` |
| Status | Generally available, according to Google's model pages |
| Location | `us-central1` or `europe-west4`, the locations Google documents for Mistral |
| Effort | Not sent |

### Mistral location

Google serves Mistral only in regions, and the default location `global` is not one of them. The location comes from an override naming the model, then an override for `mistralai`, then the configured location when it is `us-central1` or `europe-west4`. Otherwise the request fails with HTTP 400 before credentials are looked up:

```text
Vertex AI models of publisher mistralai are not served in location global: set CCP_VERTEX_LOCATION_OVERRIDES=mistralai=<location>, "mistralai": "<location>" in vertex.locationOverrides in config.json, or the location itself (CCP_VERTEX_LOCATION, GOOGLE_CLOUD_LOCATION or vertex.location), with us-central1 or europe-west4 as <location>
```

Set the override, which leaves the other families in their location:

```sh
export CCP_VERTEX_LOCATION_OVERRIDES=mistralai=us-central1
```

### Mistral request and response

The request comes from the same Chat Completions translator as OpenCode Go, with these changes:

- `stream_options` and `reasoning_effort` are not sent; Google does not document them for Mistral.
- `stream` is always `true`.
- `model` is the bare model ID. Google's Mistral page says to leave an `@<version>` suffix out of the body, so the proxy removes it there and keeps it in the URL.
- Thinking blocks in the history are not sent, and a turn that held only thinking is left out. Reasoning does not carry across turns.
- `parallel_tool_calls` and `tool_choice` are sent only when the client set `tool_choice`.

Google documents function calling for `mistral-medium-3`. The proxy sends tools to every Mistral model.

A `finish_reason` completes the message, with or without `data: [DONE]`. After it the proxy reads until Vertex AI closes the stream or 2 seconds pass, and a late `[DONE]` starts the 2 seconds again. Content after `finish_reason` and any event after `[DONE]` are errors. A streaming client gets an `api_error` event with `Vertex AI Mistral stream is invalid`; a non-streaming client gets a 502 with the reason, such as `Vertex AI Mistral event after [DONE]`. A single event is limited to 1 MiB.

## OpenAI-compatible APIs

When the OpenAI routes are enabled, every Vertex AI model works with both `POST /v1/chat/completions` and `POST /v1/responses`. Text, images, function tools, tool results, token limits, streaming, and usage use the standard shape for the chosen route. `reasoning_effort` and `reasoning.effort` accept `low`, `medium`, `high`, `xhigh`, and `max` for every family. `parallel_tool_calls` is accepted.

| Family | On the OpenAI-compatible routes |
| --- | --- |
| Gemini | Effort maps as in [Thinking](#thinking). These routes return no thinking signatures, so a function call replayed through them carries the `skip_thought_signature_validator` placeholder. `parallel_tool_calls` has no effect. |
| Anthropic Claude | The request becomes an Anthropic Messages request and is always streamed from `:streamRawPredict`. Effort is sent as `output_config.effort`. No `anthropic-beta` header is sent. |
| xAI Grok, Meta Muse | Effort is not sent. |
| OpenAI | Effort is sent as `reasoning.effort`. |
| Mistral | Effort is not sent. |

## Errors

The proxy chooses the error type from the HTTP status of Vertex AI. When Vertex AI answered the request with an error status, the message ends with `(upstream HTTP <status>)`, even when the proxy answers with a different status. The message is taken from `error.message` in the body, where Google, Anthropic, and OpenAI all put it; when the body has no such field, the body text itself is used. Google's error text is cleaned first: the access token, e-mail addresses, and long runs of base64 data (a thought signature or image the request carried) are replaced, and the text is cut to 500 characters.

| Situation | Proxy answer |
| --- | --- |
| Project missing or a setting invalid, including a malformed location override | 500 `api_error`, `Invalid Vertex configuration: ...` naming the setting |
| HTTP client could not be set up, which happens on the first Vertex AI request rather than at startup | 500 `api_error`, `Vertex AI transport is unavailable: ...` with the cause |
| Managed open model, deployed endpoint, or a name in no accepted form | 400 `invalid_request_error`, before any setting or credential is read (see [Models](#models)) |
| Hosted web search tool for Grok, OpenAI, or Muse | 400 `invalid_request_error`, `Vertex AI does not serve the hosted web_search tool for <family> models; ...` |
| Mistral without a Mistral location | 400 `invalid_request_error`, `Vertex AI models of publisher mistralai are not served in location <location>: ...` naming the settings |
| No credential, or no access token (including no answer within 30 seconds) | 401 `authentication_error`, `Google credentials are not available: ...` or `Google access token could not be obtained: ...`, followed by the `gcloud auth application-default login` hint |
| Vertex AI 401 | 401 `authentication_error`, `Vertex AI rejected the Google access token: ...` with the same hint |
| Vertex AI 403 | 403 `permission_error`. The message asks you to check the role `roles/aiplatform.user` on the project, that `aiplatform.googleapis.com` is enabled there, and, for user credentials, that a quota project is set with `gcloud auth application-default set-quota-project <project>` |
| Vertex AI 404 | 404 `invalid_request_error`, `Model <model> is not available in project <project> / location <location>: ...`, naming the three location settings |
| Vertex AI 429 | 429 `rate_limit_error`. The message names the model and location and suggests retrying later or another location. `retry-after` is set from Google's `RetryInfo`, or from a whole-second `Retry-After` header, when it is between 0 and 86400 seconds |
| Other Vertex AI 4xx, such as a Claude feature Vertex AI does not accept | Same status, `invalid_request_error`, with the upstream message |
| Vertex AI 5xx, or no connection | 502 `api_error`; `Vertex AI is unreachable` when no status arrived |
| No response headers within 300 seconds | 504 `api_error` |
| Error inside a Gemini 200 stream | A blocked prompt (`blockReason`), a stop other than `STOP` or `MAX_TOKENS` (`finishReason`, with Google's `finishMessage`), a stream that ends without a `finishReason`, an `error` chunk, response content over 16 MiB, or an event or part with none of the documented fields. Streaming requests get an `error` event. Non-streaming requests get the matching status: 429 for rate limits, 401 and 403 for credential errors, the in-stream 4xx code or 400 for invalid requests and blocked prompts, and 502 for the rest |
| Error inside a Claude, Grok, OpenAI, Muse, or Mistral 200 stream | Streaming requests get an `api_error` event: `Vertex AI Messages stream is invalid` for Claude, `Vertex AI Responses stream is invalid` for Grok, OpenAI, and Muse, `Vertex AI Mistral stream is invalid` for Mistral. Non-streaming requests get a 502 |

What to check for each status:

- **401**: the credential is expired or revoked. Run `gcloud auth application-default login` again, or check the key file named by `GOOGLE_APPLICATION_CREDENTIALS`, then restart the proxy, because it keeps the credentials from the first request that obtained an access token. If no request has obtained one since the proxy started, no restart is needed: the next request looks for the credentials again.
- **403**: the account lacks `roles/aiplatform.user` on the project, the Vertex AI API is not enabled in it, or user credentials have no quota project. For a partner model, also check that the model is enabled in Model Garden for this project, that organization policies allow it, and, for OpenAI, that the project has Private Preview access.
- **404**: the model ID is wrong, the model is not offered in that location, or a partner model is not enabled in Model Garden. The location in the message is the one the request used, which may come from a [location override](#location-overrides) rather than from the three settings it names. Try `global`, or a location the model page lists; see the location notes for [Claude](#claude-location), [Grok](#xai-grok), [OpenAI](#openai), [Muse](#meta-muse), and [Mistral](#mistral-location).
- **429**: the project's quota or Google's shared capacity for the model in that location is used up. Wait for `retry-after`, or set another location.

The proxy does not retry and does not fall back to another model; Claude Code decides whether to retry. Non-streaming requests are also sent to Vertex AI as streams and assembled by the proxy, except non-streaming Claude requests, which use `:rawPredict`. There is no limit on the time between chunks of a stream, because a model can think for a long time before it sends text. The body of an error response is read for at most 10 seconds and 64 KiB; past that, the proxy answers from the status alone.

## Traffic captures

With [traffic capture](/reference/files-and-storage/#traffic-captures) on, Vertex AI requests add these files to the request directory:

| Path | Files |
| --- | --- |
| Gemini stream | `061-vertex-stream-summary` |
| Claude stream | `060-vertex-claude-messages-stream-error`, `061-vertex-claude-messages-stream-summary` |
| Claude non-streaming | `032-upstream-response-body` and `051-downstream-response`, as JSON |
| Grok, OpenAI, Muse stream | `060-vertex-responses-stream-error`, `061-vertex-responses-stream-summary` |
| Mistral stream | `061-vertex-mistral-stream-summary` |
| Non-streaming Gemini, Grok, OpenAI, Muse, Mistral | `061-vertex-buffered-summary` and `051-downstream-response` |

The `060-...-stream-error` file is written only when the stream failed.

## Configuration

| Environment | Config key | Purpose |
| --- | --- | --- |
| `CCP_VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT` | `vertex.project` | Google Cloud project. Required. |
| `CCP_VERTEX_LOCATION`, `GOOGLE_CLOUD_LOCATION` | `vertex.location` | Vertex AI location. Default `global`. |
| `CCP_VERTEX_LOCATION_OVERRIDES` | `vertex.locationOverrides` | Locations for single publishers or models (see [Location overrides](#location-overrides)). |
| `CCP_VERTEX_BASE_URL` | `vertex.baseUrl` | Replaces the host derived from the location, for a relay or a local test server. |
| `GOOGLE_APPLICATION_CREDENTIALS` | none | Credentials file, read by the Google library. |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | none | Quota project, read by the Google library. |

An empty value counts as not set. See [Configuration](/reference/configuration/#vertex-ai) for the full table.

## Implemented from documentation only

No request to Vertex AI was observed while this provider was built. The request and response formats come from Google's `v1` API reference, Google's and Anthropic's model pages, and the beta list of Claude Code 2.1.322. The following points are assumptions; each one shows a specific symptom if it is wrong. A [traffic capture](/reference/files-and-storage/#traffic-captures) of the failing request helps when reporting one.

| Family | Assumption | Symptom if wrong |
| --- | --- | --- |
| Gemini | Gemini matches function results to calls by order, without IDs. | HTTP 400 on turns with parallel tool calls. |
| Gemini | Gemini accepts Claude Code's tool schemas, including MCP tools, unchanged. | Every request in a session with a certain tool fails with HTTP 400 naming a schema field. |
| Gemini | Every registered model accepts images inside a function response. | HTTP 400 mentioning `function_response.parts` after a tool returns an image. |
| Gemini | `candidatesTokenCount` excludes thinking tokens, so the proxy adds `thoughtsTokenCount` to the output count. | Output token counts are higher than Google bills. |
| Gemini | The `us` and `eu` multi-region hosts serve Gemini. | HTTP 404 with those locations. Use `global` or a region, or set `CCP_VERTEX_BASE_URL`. |
| All | User credentials without a quota project are accepted. | HTTP 403. Run `gcloud auth application-default set-quota-project PROJECT_ID`. |
| Gemini | Claude Code keeps empty thinking blocks that carry only a signature. | No error; the proxy sends the placeholder signature, and answer quality may drop. |
| Gemini | Error bodies carry `RetryInfo`, and errors inside a stream arrive as a chunk with an `error` key. | 429 responses have no `retry-after`. An unknown chunk shape ends the stream with an error. |
| All partner families | Error bodies carry their text at `error.message`. | Error messages show the raw body instead of Google's sentence. The status mapping is unchanged. |
| Claude | The Claude stream on Vertex AI has the same events as Anthropic's. | Broken display or usage in Claude Code, or `Vertex AI Messages stream is invalid` when `message_stop` never arrives. |
| Claude | Vertex AI accepts the request fields Claude Code sends when it believes it talks to Anthropic. | Requests that use a certain feature fail with HTTP 400 naming the field. `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` is expected to reduce them. |
| Claude | Vertex AI accepts every value on the beta list. | HTTP 400 naming a beta. |
| Claude | Claude accepts a history whose thinking blocks from other providers were removed. | HTTP 400 saying a turn must start with a thinking block. Start a new conversation. |
| Grok, OpenAI, Muse | The publishers stream only the Responses events the proxy accepts. | The stream ends with `Vertex AI Responses stream is invalid`; `060-vertex-responses-stream-error` names the reason and the capture holds the event. |
| Grok, OpenAI, Muse | The fields the proxy leaves out (effort for Grok and Muse, reasoning summaries, encrypted reasoning) are not needed. | No error. Effort has no effect, no thinking appears, and reasoning does not carry across turns. |
| Grok, OpenAI, Muse | The publishers accept the translator's `tool_choice` and `parallel_tool_calls`, and OpenAI accepts the effort values `xhigh` and `max`. | HTTP 400 naming the field. |
| Grok, OpenAI, Muse | The translator's rewrite of the Read tool and its schema normalisation suit these publishers, as they do for OpenCode Go. | Read calls with wrong arguments, or HTTP 400 about a tool schema. |
| OpenAI | A project with Private Preview access reaches the registered models. | HTTP 403 or 404. |
| Mistral | Mistral accepts `parallel_tool_calls` and `max_tokens`, and its stream ends with a `finish_reason`. | HTTP 400 naming the field, or `Vertex AI Mistral stream is invalid`. |
| Mistral | `mistral-small-2503` and `codestral-2` take tools, as `mistral-medium-3` does. | HTTP 400, or the model never calls a tool. |

## Limitations

- Model Garden's managed open models (`vertex/<publisher>/<model-id>-maas`) and deployed endpoints (`vertex/endpoint/<name>`) are refused with HTTP 400. The proxy does not enable models or create endpoints.
- Gemini's built-in tools (Google Search, code execution, URL context) and Anthropic server tools are not available for Gemini. Structured output (`output_config.format`), `cache_control`, and `metadata` are ignored for Gemini.
- For Gemini, PDF and other document blocks are dropped and image URLs are rejected. Gemini responses that contain images, files, or code-execution parts end with an error.
- `thinkingBudget` is never sent, so models that take only `thinkingBudget`, such as Gemini 2.5, are not supported.
- Safety settings are Google's defaults; the proxy sends none.
- Claude features that Vertex AI does not accept are not checked by the proxy; the upstream error is returned. `anthropic-beta` values outside the Claude Code list are dropped without an error.
- Grok, OpenAI, Muse, and Mistral do not carry reasoning across turns. Hosted tools are refused. Stored responses (`store: true`, `previous_response_id`) and OpenAI's `/responses/compact` are not used.
- Grok, OpenAI, Muse, and Mistral streams end with an error on any event the proxy does not recognise, and accept events up to 1 MiB each.
- Token counts for `/v1/messages/count_tokens` are local estimates for every family; Vertex AI's token-counting endpoints are not used.
- `vertex auth status` does not read the location overrides.
- Express mode (API keys), the Interactions API, explicit context caches, and Provisioned Throughput are not supported.

If requests fail, run `claude-code-proxy vertex auth status` first. It shows which credential, project, location, endpoint, and quota project the proxy resolves, and whether a token can be obtained.
