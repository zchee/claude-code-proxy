---
title: Models and routing
description: Understand provider model patterns, aliases, Codex fast mode, context hints, small models, model listing, and Claude Code gateway discovery.
---

The model ID in each request selects its provider. One proxy listener can serve every provider through Anthropic Messages and the enabled OpenAI-compatible routes.

## Routing patterns

| Pattern | Provider |
| --- | --- |
| Registered `gpt-*` IDs and their `-fast` and `-ultrafast` forms | Codex |
| `kimi-for-coding`, `kimi-k2.6`, `k2.6` | Kimi |
| `grok-composer-2.5-fast`, `grok-4.5`, `grok-4.6`, `grok-4.7`, `grok-4.7-build-fast`, `grok-4.7-fast` | Grok |
| `muse-spark-1.3`, `muse-spark-1.2`, `muse-wonder-1.0`, `muse-wonder-full-1.0`, `muse-ember-1.0` | Muse |
| Non-conflicting registered OpenCode Go IDs and every `opencode-go/<model-id>` | OpenCode Go |
| `gemini-4-argon`, `gemini-4-carbon`, `gemini-4-hydrogen`, `gemini-4-neon`, `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, the `-fast` forms of the four Gemini 4 IDs, and every `vertex/<model-id>` or `vertex/<publisher>/<model-id>` | Vertex AI |
| Every `google-ai-plan/<model-id>`; listed: `gemini-4-hydrogen`, `gemini-4-neon`, `gemini-4-carbon`, `gemini-3.5-flash-lite` | Google AI Ultra plan |
| `cursor`, Cursor legacy aliases, `cursor:<id>`, `cursor-plan:<id>`, `cursor-ask:<id>` | Cursor Agent |
| Every `lmstudio/<model key>`, listed from the running LM Studio server | LM Studio |
| Anthropic-style aliases such as `haiku`, `sonnet`, `opus`, `fable`, and registered `claude-*` aliases | The `aliasProvider`, Codex by default |

`muse-spark-1.3-contributor` and `muse-spark-1.2-contributor` are OpenCode Go models and route to OpenCode Go, not to Muse.

Every name that starts with `vertex/` routes to Vertex AI, and the Vertex AI provider decides the model family from the rest of the name:

| Name | Vertex AI family |
| --- | --- |
| `vertex/<model-id>`, `vertex/google/gemini-<…>` | Gemini |
| `vertex/anthropic/<model-id>`, such as `vertex/anthropic/claude-opus-5-5` | Anthropic Claude |
| `vertex/xai/<model-id>`, such as `vertex/xai/grok-4.7` | xAI Grok |
| `vertex/openai/gpt-<…>`, such as `vertex/openai/gpt-6.1-sol` | OpenAI |
| `vertex/meta/muse-<…>`, such as `vertex/meta/muse-spark-1.3` | Meta Muse |
| `vertex/mistralai/<model-id>`, such as `vertex/mistralai/mistral-medium-3` | Mistral |
| `vertex/<publisher>/<model-id>-maas`, such as `vertex/zai-org/glm-5.2-maas` or `vertex/openai/gpt-oss-120b-maas` | Model Garden managed open model |
| `vertex/endpoint/<name>`, where `<name>` is a key of `vertex.endpoints` in `config.json` | An endpoint you deployed |

Unregistered IDs in these forms are sent too, so a new model works before the proxy lists it. Every other `vertex/<publisher>/<model-id>`, and a `vertex/endpoint/<name>` that names no configured endpoint, return HTTP 400. A bare ID that is not registered, such as a new `gemini-*` model, is not routed to Vertex AI; use the `vertex/` form.

Every name that starts with `google-ai-plan/` routes to the Google AI Ultra plan, listed or not, so the same Gemini model is reached on the plan's quota as `google-ai-plan/gemini-4-hydrogen` and on a Google Cloud project as `gemini-4-hydrogen` or `vertex/gemini-4-hydrogen`. See [Google AI Ultra plan models](/providers/google-ai-plan/#models).

Only Gemini IDs reach Vertex AI without the prefix. Bare `claude-*`, `gpt-*`, `grok-*`, and `muse-*` IDs keep their own providers: `claude-opus-5-5` is an Anthropic-style alias that follows the alias provider setting (Codex by default; see [Alias routing](#alias-routing)), `gpt-6.1-sol` is Codex, `grok-4.7` is Grok, and `muse-spark-1.3` is Muse. A bare open-model ID such as `glm-5.2-maas` does not reach Vertex AI either. The same model on Vertex AI is reached only as `vertex/<publisher>/<model-id>`, billed to your Google Cloud project. See [Vertex AI models](/providers/vertex/#models).

An unknown ID returns HTTP 400 with the supported provider catalog. There is no implicit fallback for arbitrary model names.

## Prefer the live catalog

The model list changes faster than documentation. Ask the installed CLI:

```sh
claude-code-proxy models
claude-code-proxy models --full
```

The compact command abbreviates Cursor's dynamic aliases. `--full` prints every alias discovered through the installed Cursor Agent catalog.

The HTTP equivalent is:

```sh
curl http://127.0.0.1:18765/v1/models
```

## Codex fast mode

Every registered Codex model also has a local `-fast` form. The proxy removes `-fast` from the upstream model and requests the priority service tier.

Models that offer the ultrafast tier also have a local `-ultrafast` form, which requests the ultrafast service tier. Today those are `gpt-6-astra-ultrafast`, `gpt-6.1-sol-ultrafast`, `gpt-6.2-astra-ultrafast`, and `gpt-6.2-sol-ultrafast`. An `-ultrafast` name on any other model is not a registered ID.

For Claude Code requests on `/v1/messages`, a configured `codex.serviceTier` or `CCP_CODEX_SERVICE_TIER` override wins over both suffixes. When the forced tier is `ultrafast` and the model does not offer it, the proxy requests priority instead.

## Gemini fast mode

`gemini-4-argon`, `gemini-4-carbon`, `gemini-4-hydrogen`, and `gemini-4-neon` on Vertex AI also have a local `-fast` form, bare or under `vertex/`. The proxy removes `-fast` from the upstream model and asks Vertex AI for Priority PayGo, which is billed at a higher rate. `-fast` on any other Gemini ID is not a registered ID. See [Priority PayGo](/providers/vertex/#priority-paygo).

## The `[1m]` hint

A trailing `[1m]` affects Claude Code's local compaction policy. The proxy strips it before matching and forwarding the model. Use it only when the upstream model and account can accept the resulting context, and set a safe `CLAUDE_CODE_AUTO_COMPACT_WINDOW` when the real limit is below one million tokens.

## Main and small models

Set both model variables to routable IDs:

```sh
ANTHROPIC_MODEL=gpt-6-sol[1m]
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna[1m]
```

Claude Code sends background work to its small/fast model. Its built-in Haiku ID can route through `aliasProvider`, but a concrete provider ID keeps the behavior explicit.

## Claude Code `/model`

When `ANTHROPIC_BASE_URL` already points to the proxy, these can change the request model:

- `claude --model <id>` at launch
- Claude Code's `/model` command in a session
- `ANTHROPIC_MODEL` for a new process

A model change can switch upstream providers because routing is per request. It does not change `ANTHROPIC_BASE_URL` or move the process back to direct Anthropic.

## Gateway model discovery

Enable Claude Code discovery at process start:

```sh
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 \
ANTHROPIC_BASE_URL=http://127.0.0.1:18765 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_MODEL=gpt-6-sol[1m] \
ANTHROPIC_SMALL_FAST_MODEL=gpt-6-luna[1m] \
  claude
```

`/v1/models` includes raw provider IDs and Anthropic-style aliases. Claude Code's gateway picker filters for IDs beginning with `claude` or `anthropic`, so configured aliases are the entries most likely to appear. `vertex/anthropic/...` names begin with `vertex`, so the picker does not list them. Raw IDs remain usable through `--model`, `/model`, and environment variables.

## Alias routing

`CCP_ALIAS_PROVIDER=kimi` or `"aliasProvider": "kimi"` routes recognized Anthropic-style aliases to Kimi. Accepted values are `codex` and `kimi`. Explicit provider IDs always use their provider.
