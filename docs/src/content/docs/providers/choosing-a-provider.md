---
title: Choosing a provider
description: Compare claude-code-proxy providers by account, protocol, models, reasoning, multimodal input, tools, and operational tradeoffs.
---

One `serve` process supports every provider. Choose based on the account you have, model access, and the capabilities your work needs.

| Provider | Account | Upstream protocol | Model selection | Notable capabilities |
| --- | --- | --- | --- | --- |
| [Codex](/providers/codex/) | ChatGPT Plus or Pro | OpenAI Responses over WebSocket or HTTP SSE | Named Codex catalog, `-fast` and `-ultrafast` variants | Function tools, image input, hosted web search, reasoning summaries, optional native Responses route |
| [Kimi](/providers/kimi/) | kimi.com with Kimi Code access | OpenAI-style chat completions | `kimi-for-coding` and aliases | Function tools, reasoning, image and video input |
| [Grok](/providers/grok/) | grok.com | Responses API | `grok-composer-2.5-fast`, `grok-4.5`, `grok-4.6`, `grok-4.7`, `grok-4.7-build-fast`, `grok-4.7-fast` | Function tools, reasoning, web search, X search, citations |
| [Muse](/providers/muse/) | Meta account with a Muse Code subscription | Meta Responses API | `muse-spark-1.3`, `muse-spark-1.2`, `muse-wonder-1.0`, `muse-wonder-full-1.0`, `muse-ember-1.0` | Function tools, reasoning summaries, image input, 1,007,997 token context |
| [OpenCode Go](/providers/opencode-go/) | OpenCode Go subscription | OpenAI-compatible Chat Completions, OpenAI Responses, or Anthropic-compatible Messages | Non-conflicting bare IDs and `opencode-go/<model-id>` forms | Curated models tested and benchmarked for coding-agent use |
| [Vertex AI](/providers/vertex/) | Google Cloud project with billing and the Vertex AI API enabled; partner and open models enabled in Model Garden | Vertex AI `streamGenerateContent` (Gemini), Anthropic Messages on `rawPredict` (Claude), OpenAI Responses (Grok, OpenAI, Muse), Mistral chat on `rawPredict`, OpenAI Chat Completions (managed open models) | Registered Gemini IDs, bare or as `vertex/<model-id>`; Claude, Grok, OpenAI, Muse, and Mistral as `vertex/<publisher>/<model-id>`; managed open models as `vertex/<publisher>/<model-id>-maas` | Function tools, Gemini thinking levels, Claude passthrough, image input, per-model locations, Google Application Default Credentials, pay-as-you-go billing |
| [Google AI Ultra plan](/providers/google-ai-plan/) | Google account with a Google AI Ultra plan | Cloud Code PA (`v1internal:streamGenerateContent`) | `google-ai-plan/<model-id>`: `gemini-4-hydrogen`, `gemini-4-neon`, `gemini-4-carbon`, `gemini-3.5-flash-lite` | Function tools, Gemini thinking levels up to `HIGH`, image input, browser sign-in, 1,048,576 token context |
| [Cursor Agent](/providers/cursor-agent/) | Cursor account | HTTP/2 Connect stream | Cursor modes and `cursor:<model-id>` prefixes | Dynamic model catalog, effort variants, images, plan and ask modes, session continuation |
| [LM Studio](/providers/lmstudio/) | None; models run on this machine | Anthropic-compatible Messages, forwarded unchanged | `lmstudio/<model key>`, listed live from LM Studio | Every downloaded LLM, tools and streaming as the model supports them, no subscription |

## Practical guidance

- Start with **Codex** when you have a ChatGPT subscription and want the most developed Claude Code translation path.
- Choose **Kimi** for the Kimi Code model and multimodal coding input.
- Choose **Grok** for Grok models and hosted web or X search.
- Choose **Muse** when you have a Muse Code subscription and want Meta's Muse models. Logins do not refresh, so plan for an occasional new login.
- Choose **OpenCode Go** when you have a Go subscription and want its documented model catalog in Claude Code.
- Choose **Vertex AI** when you have a Google Cloud project and want Gemini, Claude, Grok, OpenAI, Muse, Mistral, or Model Garden's managed open models, such as gpt-oss, GLM, Gemma, Llama 4, and DeepSeek, billed to it. The proxy uses your gcloud or service account credentials and stores none of its own. Partner and open models must be enabled in Model Garden, and some are served only in certain locations.
- Choose **Google AI Ultra plan** when you have a Google AI Ultra subscription and want Gemini on its quota rather than billed to a Google Cloud project. See the [sign-in options and unofficial client notice](/providers/google-ai-plan/).
- Choose **Cursor Agent** when you want Cursor's model catalog and agent modes. It depends on an installed Cursor Agent bundle for protobuf schemas.

## Shared behavior

All providers route by model ID, stream responses, handle Claude Code tool definitions, refresh tokens when supported, and report failures through the same Anthropic-shaped API. For each provider's credential source, see [Files and storage](/reference/files-and-storage/#provider-credentials).

<div class="security-callout">
<strong>Account policy matters.</strong> Provider subscriptions, model access, regional availability, rate limits, and rules for unofficial clients can change. Review the terms for your account before using a provider through the proxy.
</div>

Use `claude-code-proxy models` for the current compact catalog and `claude-code-proxy models --full` for all advertised aliases. The [Models and routing](/using/models-and-routing/) page explains aliases and discovery.
