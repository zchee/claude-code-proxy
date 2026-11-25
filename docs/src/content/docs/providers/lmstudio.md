---
title: LM Studio
description: List and call the models downloaded into LM Studio through its Anthropic-compatible local server.
---

LM Studio 0.4.1 and later serve an Anthropic-compatible `POST /v1/messages`
from their local server. The proxy sends Claude Code's requests there
unchanged, including streaming and tool use, so nothing is translated. The
provider is on by default and expects the server at `http://127.0.0.1:1234`.

## Start the server

Start LM Studio's server from the app's Developer page or from the terminal:

```sh
lms server start
```

The proxy needs no login. When "Require Authentication" is on in LM Studio's
server settings, give the proxy an API token from those settings:

```sh
export CCP_LMSTUDIO_API_KEY=YOUR_LM_STUDIO_TOKEN
claude-code-proxy serve
```

The proxy also reads `LM_API_TOKEN`, the variable LM Studio's own
documentation uses.

## Models

Every LM Studio model is named `lmstudio/<model key>`, where the key is the one
`lms ls` and `GET /api/v1/models` show:

```sh
claude-code-proxy models
```

```text
lmstudio: lmstudio/gemma-4-12b-obliterated, lmstudio/qwen3.8-27b-obliterated, ...
```

The list is read from LM Studio each time `claude-code-proxy models` runs and
each time a client calls the proxy's `GET /v1/models`, so a model downloaded
while the proxy runs shows up without a restart. Embedding models are left
out because they cannot answer a Messages request.

When the server on this machine is not running, the proxy lists the LLMs on
this machine's disk instead, as `lms ls --json --llm` reports them, and says
that the server has to be started before they can be used:

```text
lmstudio: lmstudio/gemma-4-12b-obliterated, lmstudio/qwen3.8-27b-obliterated, ...
  (listed from this machine's disk: LM Studio server is not reachable at http://127.0.0.1:1234; start it with `lms server start` before using them)
```

The disk list leaves out models on other devices reached through LM Link. It
is used only when the base URL is a loopback address and the `lms` CLI is on
`PATH` or in `~/.lmstudio/bin`. A server that answers with an error, such as
a missing token, is reported as that error rather than replaced by the disk
list. When neither source answers, the line says why:

```text
lmstudio: (no models: LM Studio server is not reachable at http://127.0.0.1:1234; start it with `lms server start` (could not run ...))
```

Every name that starts with `lmstudio/` routes to LM Studio, listed or not. The
model key, without the prefix, is what LM Studio receives. Claude Code's
`/model` picker only shows gateway models whose IDs begin with `claude` or
`anthropic`, so select an LM Studio model with `--model` or the environment:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:18765 ANTHROPIC_AUTH_TOKEN=unused
export ANTHROPIC_MODEL=lmstudio/qwen3.8-27b-obliterated
export ANTHROPIC_SMALL_FAST_MODEL=lmstudio/qwen3.8-27b-obliterated
claude
```

## Loading and context length

With "Just-in-Time model loading" on, LM Studio loads a model on its first
request, so that request waits for the load. LM Studio then uses the model's
default context length, which can be far smaller than Claude Code's system
prompt and tool definitions. Load the model yourself with a larger context
before starting Claude Code:

```sh
lms load qwen3.8-27b-obliterated --context-length 65536
```

Pick a model trained for tool use; `GET /api/v1/models` reports it as
`capabilities.trained_for_tool_use`.

## Token counting

LM Studio has no `/v1/messages/count_tokens`. The proxy answers that request
itself with an estimate and does not send it to LM Studio.

## Configuration

| Environment | Config key | Default | Purpose |
| --- | --- | --- | --- |
| `CCP_LMSTUDIO_ENABLED` | `lmstudio.enabled` | `true` | `0`, `false`, `no`, or `off` turns the provider off; it is then neither listed nor routed. |
| `CCP_LMSTUDIO_BASE_URL` | `lmstudio.baseUrl` | `http://127.0.0.1:1234` | The LM Studio server. |
| `CCP_LMSTUDIO_API_KEY`, then `LM_API_TOKEN` | `lmstudio.apiKey` | unset | Token sent as `Authorization: Bearer` when the server requires authentication. |

The base URL must be a scheme, a host, and a port only, without `/v1`, and must
use `https` unless the host is a loopback address, since it receives the
token. An empty value counts as not set. An invalid value does not hide the
provider: `claude-code-proxy models` and every LM Studio request report the
setting by name.

## Errors

- **`(no models: ... requires authentication (HTTP 401) ...)`**: the server
  requires a token. Set `CCP_LMSTUDIO_API_KEY`.
- **`(no models: ... has no /api/v1/models ...)`**: the LM Studio version is
  older than 0.4.1. Update LM Studio.
- **502 `Could not reach LM Studio at ...`**: the server stopped. Start it
  again with `lms server start`.
- **400 `No models loaded ...`**: LM Studio could not load the model, for
  example because the key is wrong or Just-in-Time loading is off. Check the
  key with `lms ls`, or load the model with `lms load`.
