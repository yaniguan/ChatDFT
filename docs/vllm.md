# vLLM integration

ChatDFT can route LLM calls to a local [vLLM](https://github.com/vllm-project/vllm)
server to cut latency and cost for high-volume, low-reasoning agents (intent
parsing, plan generation, QA, post-analysis summarisation). vLLM's paged
attention + continuous batching makes it dramatically faster than serial
HTTP calls to a hosted API once concurrency ramps up.

This document covers:

1. Architecture — how ChatDFT decides which provider handles which agent.
2. Running a local vLLM server.
3. Configuring ChatDFT to route specific agents to it.
4. Benchmarking.
5. Observability in the monitoring dashboard.
6. Troubleshooting.

---

## 1. Architecture

```
                 ┌────────────────┐
 agent call ─────▶  chatgpt_call  │  (server/utils/openai_wrapper.py)
                 └────────┬───────┘
                          ▼
                 ┌────────────────┐
                 │   LLMRouter    │  (server/utils/llm_providers.py)
                 │                │  - picks provider from llm.yaml
                 │                │  - health check + fallback
                 │                │  - per-provider semaphore cap
                 └───┬────────┬───┘
                     │        │
        ┌────────────┘        └────────────┐
        ▼                                  ▼
┌───────────────┐                 ┌────────────────┐
│ OpenAIProvider│                 │  VLLMProvider  │
│  api.openai   │                 │  localhost:8001│
└───────┬───────┘                 └────────┬───────┘
        │                                  │
        └──────────▶ AgentLog (model=provider:name) ──▶ dashboard
```

Key design points:

* **Both providers wrap `openai.AsyncOpenAI`** — vLLM serves an OpenAI-compatible
  API, so the HTTP shape is identical. The difference is `base_url` and a
  client-side concurrency semaphore.
* **Routing lives in one YAML file** (`server/llm.yaml`). No hardcoded
  model names in agent code.
* **Fallback is opt-in.** If vLLM raises or returns 5xx, the router retries
  on the configured fallback provider (default: OpenAI). Slow responses do
  not trigger fallback — slow is still correct.
* **Observability is automatic.** The provider name is prefixed onto the
  `model` column in `agent_log`, and the monitoring dashboard groups by
  provider — no schema migration needed.

---

## 2. Running a local vLLM server

### GPU (recommended)

```bash
# 1. Make sure nvidia-container-toolkit is installed
nvidia-smi

# 2. Start vLLM in the background
docker compose -f infra/docker/vllm.docker-compose.yaml up -d

# 3. Watch model download + startup (can take 2–5 minutes on first run)
docker logs -f chatdft-vllm

# 4. Verify
curl http://localhost:8001/health                  # → {"status":"ok"}
curl http://localhost:8001/v1/models | jq          # lists served models
```

Default model is `Qwen/Qwen2.5-7B-Instruct` — strong on structured JSON,
reasonable on instruction following, fits comfortably on a single 24 GB
GPU. Swap via env:

```bash
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
  docker compose -f infra/docker/vllm.docker-compose.yaml up -d
```

For gated models (Llama 3.1, Mistral, etc.) export `HF_TOKEN` first.

### CPU-only smoke test

vLLM's CPU path is intentionally slow — it exists for CI, not real traffic.
To try the plumbing end-to-end on a laptop:

```bash
VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  docker compose -f infra/docker/vllm.docker-compose.yaml up -d
```

Expect 30–90 s per response — fine for testing the routing code, not for
real agent work.

---

## 3. Pointing ChatDFT at vLLM

Edit `server/llm.yaml` and flip the agents you want offloaded:

```yaml
routing:
  intent_agent: vllm_local        # cheap JSON parsing
  plan_agent: vllm_local          # task graph generation
  qa_agent: vllm_local            # freeform QA
  post_analysis_agent: vllm_local # result summarisation
  # keep heavy reasoning on OpenAI:
  hypothesis_agent: openai
  analyze_agent: openai
```

Restart the ChatDFT backend. The router logs the first call for each agent
so you can verify routing:

```
chatdft.llm_providers: intent_agent → vllm_local (Qwen/Qwen2.5-7B-Instruct)
chatdft.llm_providers: hypothesis_agent → openai (gpt-4o)
```

### Environment-variable overrides

You can override without editing YAML:

```bash
# Re-point vllm_local at a remote machine
export CHATDFT_VLLM_BASE_URL=http://gpu-box.lan:8001/v1
export CHATDFT_VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct

# Flip a single agent at runtime
export CHATDFT_LLM_ROUTING_INTENT_AGENT=vllm_local

# Disable fallback entirely (errors propagate)
export CHATDFT_LLM_FALLBACK_PROVIDER=
```

---

## 4. Benchmarking

Use the bundled script to measure throughput and latency against whatever
providers you have configured:

```bash
python scripts/benchmark_llm.py \
    --providers openai vllm_local \
    --n 50 \
    --concurrency 16 \
    --agent intent_agent
```

Example output on an RTX 4090 running `Qwen/Qwen2.5-7B-Instruct`:

```
benchmark_llm
=============
agent        : intent_agent
concurrency  : 16
prompts      : 50

provider      requests  ok  p50 ms   p95 ms   p99 ms   tokens/sec  wall s
openai             50  50     850     1420     1810          612    4.72
vllm_local         50  50     240      390      460         2410    1.48
speedup (wall)  3.19x

provider      in_tok  out_tok  usd
openai          8150     4820  0.0895
vllm_local      8150     4820  0.0000  (local)
```

The script writes its raw JSON to `runs/llm_benchmark_<timestamp>.json`
so you can track regressions over time.

---

## 5. Observability

Once the dashboard is up (PR #1), the `Providers` row shows per-provider:

* request count
* p50 / p95 / p99 latency
* tokens/sec
* cost (USD; local providers show 0)
* error rate

It reads the same `agent_log.model` column that `openai_wrapper` now
populates with `provider:model` strings, so nothing else has to be wired.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `router: no healthy providers` | vLLM container isn't up or health check fails | `docker logs chatdft-vllm`; confirm `/health` returns 200 |
| Requests go to OpenAI even after routing change | Config cached in-process | Restart the ChatDFT backend, or call `llm_config.reset_llm_config()` |
| `gpt-4o` sent to vLLM | Caller passed hardcoded model hint | `_pick_model()` auto-falls-back to the provider's default when the hint is obviously the wrong family — verify your provider's `model:` field in `llm.yaml` |
| Out-of-memory on startup | Context window too large for your GPU | Lower `VLLM_MAX_MODEL_LEN` or `VLLM_GPU_MEMORY_UTIL` |
| JSON schema errors spike on vLLM | Smaller models struggle with strict JSON | Raise `max_tokens`, add an explicit JSON example to the system prompt, or route `plan_agent` back to OpenAI |
| Empty vision responses | vLLM is routing a multi-modal call | Leave `knowledge_agent` on OpenAI (its vision path uses a direct client by design) |

---

## 7. Scope notes

What this integration does **not** do (left for follow-ups):

* **Streaming responses** — none of the agents currently stream.
* **Client-side request batching** — vLLM's continuous batching handles
  this server-side; client-side batching would only add latency.
* **Tool calls / function calling** — no agent uses them today.
* **Multi-node tensor parallelism** — swap the docker-compose for the
  official vLLM production manifests if you need it.
* **Fine-tuning a ChatDFT-specific model** — separate project.
