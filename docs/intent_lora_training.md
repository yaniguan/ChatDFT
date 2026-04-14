# Intent agent LoRA training — runbook

End-to-end guide for training and evaluating the Qwen2.5-7B LoRA that
replaces the GPT-4o-mini intent parser. This doc assumes you've already
run Phase 1 (data generation) successfully — if `artifacts/sft_v2/` does
not exist, regenerate it with:

```bash
DATABASE_URL=postgresql+asyncpg://yaniguan@localhost/chatdft_ase \
  python -m scripts.export_sft_dataset --out artifacts/sft_v2
```

You have **two** training paths. Pick one:

* **Path A — Modal (recommended if you don't have a GPU)**: one
  command from your Mac, no Docker, no axolotl install, no GPU env
  setup. See [Path A — one-command remote training via Modal](#path-a--one-command-remote-training-via-modal).

* **Path B — your own GPU box**: full control, no cloud costs per
  run, but you have to set up the axolotl env yourself. See
  [Path B — training on a GPU workstation](#path-b--training-on-a-gpu-workstation).

## The 30-second version

```bash
# Path A — Modal
pip install modal && modal setup               # one time
modal run scripts/train_modal.py --overfit     # ~25 min sanity check
modal run scripts/train_modal.py               # ~90 min real run

# Path B — your own GPU
python -m scripts.verify_sft_dataset artifacts/sft_v2   # on the GPU box
AXOLOTL_CONFIG=configs/qwen_lora_overfit.yaml \
  bash scripts/train_qwen_lora.sh                       # overfit check
bash scripts/train_qwen_lora.sh                         # real run
```

---

## Path A — one-command remote training via Modal

Modal rents GPUs by the minute and handles the Docker image, the
package install, the CUDA toolchain, and weight persistence. You run
**one command from your Mac** and get back a trained adapter + the
LoRA's eval scores next to the GPT-4o-mini baseline.

### Prerequisites

```bash
pip install modal        # Python SDK
modal setup              # sign in + authorize a workspace
```

### Every run — export dataset first

The SFT corpus lives in PostgreSQL and has to be exported to local
JSONL before Modal can mount it:

```bash
DATABASE_URL=postgresql+asyncpg://yaniguan@localhost/chatdft_ase \
  python -m scripts.export_sft_dataset --out artifacts/sft_v2

python -m scripts.verify_sft_dataset artifacts/sft_v2
# → must print "PASS" (warnings ok)
```

### Launch

```bash
# Full pipeline: overfit sanity check → real run → eval → compare to baseline
modal run scripts/train_modal.py --overfit       # ~25 min, ~$0.45
modal run scripts/train_modal.py                 # ~90 min, ~$1.50 + eval ~$0.05
```

Other flags:

```bash
modal run scripts/train_modal.py --preprocess-only   # tokenize + cache, no training
modal run scripts/train_modal.py --train-only        # skip eval pass
modal run scripts/train_modal.py --eval-only         # re-score an already-trained adapter
modal run scripts/train_modal.py --config configs/custom.yaml
```

### What the Modal run does

1. `_preflight()` — verifies `artifacts/sft_v2/train.jsonl` exists
   and the config file is present. Fails fast with a fix hint.
2. `train_lora.remote()` — boots a Modal container (winglian/axolotl
   image, L4 24 GB GPU), pulls Qwen2.5-7B weights from HuggingFace
   (first run only, then cached in a `modal.Volume`), runs
   `axolotl preprocess` + `axolotl train`. Commits the adapter to a
   persistent output volume.
3. `eval_lora.remote()` — loads the just-trained adapter with
   `peft.PeftModel.from_pretrained`, runs zero-temperature inference
   on all 30 hand-labeled eval cases, parses the JSON, scores via the
   harness's pure `score_case` + `aggregate` functions.
4. Local entrypoint prints a side-by-side delta against
   `artifacts/baseline_gpt4o_mini.json` with a ship-gate pass/fail
   indicator. Also writes `artifacts/lora_eval.json` for future diffs.

The canonical system prompt is imported from `server/chat/intent_prompt.py`
on both ends — single source of truth, zero train/serve skew.

### Cost reference

| run | GPU wall time | Modal cost |
|---|---|---|
| `--overfit` (10 epochs × 1736 rows) | ~25 min on L4 | **~$0.45** |
| production (`qwen_lora.yaml`, 3 epochs × 1736 rows) | ~90 min on L4 | **~$1.50** |
| `eval_lora` | ~3 min on L4 | **~$0.05** |
| Qwen-7B first download (one-time, then cached) | ~5 min | ~$0.10 |

Override the GPU class if you want a different speed/cost tradeoff:

```bash
# Cheaper — 16 GB A10G, needs micro_batch_size=1
# (edit configs/qwen_lora.yaml before launching)
modal run scripts/train_modal.py --gpu A10G

# Faster — A100 40 GB, ~2.5x speed for ~3x cost
modal run scripts/train_modal.py --gpu A100-40GB
```

### Pulling the adapter back to your Mac

After training, the LoRA weights live on a persistent Modal Volume:

```bash
modal volume get chatdft-intent-lora-output \
  artifacts/qwen_lora_out ./artifacts/qwen_lora_out
```

You can then either:
- Merge + serve via vLLM from a GPU workstation (see Path B step 6).
- Skip merging entirely and re-run `modal run scripts/train_modal.py
  --eval-only` as many times as you want — the adapter stays on the
  volume and the eval is cheap.

### Troubleshooting

- **"artifacts/sft_v2/train.jsonl not found"**: you forgot to export
  the dataset. See the Every-run section above.
- **"Modal: ImagePullError"**: the axolotl image tag may have rotated.
  Edit `_image = Image.from_registry("winglian/axolotl:main-latest"...)`
  in `scripts/train_modal.py` and pin to a specific date tag.
- **CUDA OOM**: drop `micro_batch_size` to 1 in the config, bump
  `gradient_accumulation_steps` to 32 to keep effective batch = 32.
- **eval loads model slow**: that's normal, the 15 GB base model
  loads in ~45 s on first call. Subsequent calls are fast because
  the HF volume is warm.

---

## Path B — training on a GPU workstation

Local path, nothing pays per minute, but you own the env setup.

### Short version

```bash
# local (Mac, llm-agent env)
python -m scripts.verify_sft_dataset artifacts/sft_v2
# → must report PASS before you rent a GPU

# remote GPU (24 GB+ VRAM, e.g. 3090/4090/L4/A100)
bash scripts/train_qwen_lora.sh              # real run
# OR
AXOLOTL_CONFIG=configs/qwen_lora_overfit.yaml \
  bash scripts/train_qwen_lora.sh            # overfit sanity check first
```

## Prerequisites

| item | version | where |
|---|---|---|
| SFT corpus | `artifacts/sft_v2/` | generated by `scripts/generate_intent_pairs.py` |
| axolotl | ≥ 0.5.0 | fresh conda env on the GPU box (NOT llm-agent) |
| CUDA | 12.1+ | matches bundled torch in axolotl |
| VRAM | ≥ 24 GB | for `micro_batch_size=2, sequence_len=4096` |
| HF cache space | ~20 GB | Qwen2.5-7B weights + tokenizer |

## 1. Pre-flight verification (local, ~5 seconds)

Run the pure-Python verifier **before** you rent GPU time. This catches:

* malformed JSON in any record
* role-sequence bugs (`system/user/assistant` in the wrong order)
* assistant content that doesn't pass `IntentSchema`
* catastrophic length outliers (>8000 chars, would blow `sequence_len`)
* area imbalance beyond 5× ratio

```bash
python -m scripts.verify_sft_dataset artifacts/sft_v2
```

A clean run looks like:

```
══════════════════════════════════════════════════════════════
  SFT dataset verification: artifacts/sft_v2
══════════════════════════════════════════════════════════════
train.jsonl: 1736 records, 0 errors
val.jsonl:   205 records, 0 errors
...
PASS. Dataset is ready to train.
```

If this says `FAIL`, **do not** proceed to the GPU — fix the data first.

## 2. Remote GPU setup (one-time)

axolotl pins torch/transformers versions that conflict with the
FastAPI/sqlalchemy stack, so it needs its own env:

```bash
# On the GPU box
conda create -n axolotl python=3.10 -y
conda activate axolotl
pip install packaging ninja
pip install "axolotl[flash-attn,deepspeed]"
axolotl --version   # verify
```

Sync the repo + dataset:

```bash
# From local
rsync -azP \
  --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='artifacts/qwen_lora_*_out' \
  /Users/yaniguan/Documents/04_vibecode/ChatDFT/ \
  gpu-box:/path/to/ChatDFT/
```

## 3. Pipeline validation: the overfit config (~30 min)

**Do this first, before committing to a real run.** The overfit config is
designed to drive train loss to near-zero on the 1736-row corpus. If it
doesn't, something in the chat_template rendering, tokenization, or loss
masking is broken, and you need to fix that before wasting hours on a
real run.

```bash
# On the GPU box
cd /path/to/ChatDFT
conda activate axolotl
AXOLOTL_CONFIG=configs/qwen_lora_overfit.yaml \
  bash scripts/train_qwen_lora.sh
```

Watch the log:

* **Healthy**: `train/loss` drops from ~2.0 → < 0.1 by epoch 3, then
  plateaus near 0.05. `eval/loss` diverges from `train/loss` after epoch
  2-3 — this is OVERFITTING and that's the goal.
* **Broken (chat_template)**: `train/loss` plateaus above 0.5 and won't
  budge. Check `axolotl preprocess` output — look at one rendered
  example and confirm the assistant span is actually being unmasked for
  loss computation.
* **Broken (precision)**: `train/loss` goes NaN at step 1. Switch to
  `load_in_4bit: true` + `qlora` adapter OR drop `learning_rate` to 5e-5.
* **Broken (OOM)**: set `micro_batch_size: 1` and bump
  `gradient_accumulation_steps` to 32.

If the overfit run succeeds, move to the real run.

## 4. Real training run (~2-4 hours on one 24 GB card)

```bash
# Uses the default configs/qwen_lora.yaml
bash scripts/train_qwen_lora.sh
```

TensorBoard:

```bash
tensorboard --logdir artifacts/qwen_lora_out
# → localhost:6006
```

Watch for:

* `train/loss` decreasing smoothly (no NaN spikes)
* `eval/loss` tracking `train/loss` closely for epoch 1-2, then lagging
  — healthy generalization
* If `eval/loss` starts climbing after epoch 2, that's overfitting
  despite the regularization — reduce `num_epochs` to 2 or bump
  `lora_dropout` to 0.1

## 5. Evaluate the trained adapter

Two evaluations matter. Run both.

### 5a. Against the 30 hand-labeled eval set (compare to baseline)

The baseline GPT-4o-mini numbers from 2026-04-11 are saved in
`artifacts/baseline_gpt4o_mini.json`. The LoRA has to beat these:

| metric | GPT-4o-mini baseline |
|---|---|
| field_accuracy | **0.787** |
| critical_em_rate | **0.467** |
| area accuracy | 0.900 |
| stage accuracy | 0.733 |

Steps:

1. Merge LoRA into base or use PEFT loader
2. Serve via vLLM:
   ```bash
   docker compose -f infra/docker/vllm.docker-compose.yaml up -d
   # with CHATDFT_VLLM_MODEL=/path/to/merged/qwen_lora_merged
   ```
3. Flip routing:
   ```bash
   export CHATDFT_LLM_ROUTING_INTENT_AGENT=vllm_local
   ```
4. Restart `server.main:app` so routing takes effect.
5. Re-run the harness:
   ```bash
   python -m tests.intent_eval.harness \
     --predictor live \
     --show-failures \
     --min-field-accuracy 0.85 \
     --min-critical-em 0.55 \
     > artifacts/lora_eval.log
   ```
6. Diff vs baseline:
   ```bash
   diff artifacts/baseline_gpt4o_mini.json artifacts/lora_eval.log
   ```

**Ship threshold**: LoRA must match or beat baseline on `field_accuracy`
AND `critical_em_rate`. Ideally it also lifts `stage` accuracy (which
was the weakest dimension in the baseline at 0.733).

### 5b. Against the held-out SFT validation set

```bash
python -m scripts.eval_lora_against_val \
  --adapter artifacts/qwen_lora_out \
  --val artifacts/sft_v2/val.jsonl
```

(Script doesn't exist yet — write it when you have a trained adapter.
It should load the val.jsonl records, run each user payload through the
adapter, and compute exact-match on the assistant JSON.)

## 6. Ship

1. Merge the adapter:
   ```bash
   axolotl merge-lora configs/qwen_lora.yaml \
     --lora-model-dir artifacts/qwen_lora_out
   ```
2. Update `server/llm.yaml` — change `routing.intent_agent` from
   `openai` to `vllm_local`.
3. Update `infra/docker/vllm.docker-compose.yaml` to point at the merged
   model weights.
4. Keep `fallback_provider: openai` so a vLLM outage does not blackhole
   the pipeline.
5. Ship behind a feature flag: shadow-mode the LoRA for 10% of traffic
   first, alert on >15% disagreement with the GPT-4o-mini baseline,
   then flip to 100% when clean for a week.

## Baseline reference (2026-04-11)

Captured on the 30-case `tests/intent_eval/eval_set.jsonl` against the
then-current production path (GPT-4o-mini via `server/llm.yaml` default
routing, IntentSchema validation enabled, retry loop active):

```
n_cases:           30
field_accuracy:    0.787
critical_em_rate:  0.467
per-field:
  stage:                      0.733
  area:                       0.900
  substrate:                  0.720
  facet:                      0.941
  reactant:                   0.714
  product:                    0.562     ← weakest
  conditions.pH:              1.000
  conditions.potential_V_vs_RHE: 1.000
  conditions.temperature:     1.000
  conditions.pressure:        1.000
  conditions.electrolyte:     0.000     ← baseline misses electrolyte on the 1 case that has it
```

Area confusion matrix (rows = gold, columns = predicted):

| | electro | thermal | homo | photo | hetero |
|---|---:|---:|---:|---:|---:|
| **electro** | 10 | 0 | 0 | 0 | 0 |
| **thermal** | 0 | **3** | 0 | 0 | **3** |
| **homo** | 0 | 0 | 3 | 0 | 0 |
| **photo** | 0 | 0 | 0 | 3 | 0 |
| **hetero** | 0 | 0 | 0 | 0 | 8 |

The failure pattern to beat: **thermal_catalysis ↔ heterogeneous_catalysis
confusion** (3 of 6 thermal cases classified as heterogeneous). That's
the area dimension where the LoRA needs to improve most.
