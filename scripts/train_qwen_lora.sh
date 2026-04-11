#!/usr/bin/env bash
# scripts/train_qwen_lora.sh
#
# Thin wrapper around `axolotl` that trains the intent-agent student LoRA
# on Qwen2.5-7B-Instruct. See configs/qwen_lora.yaml for the per-knob
# rationale; this script just verifies prerequisites and invokes axolotl.
#
# Why a wrapper at all (and not "just call axolotl"):
# * Catches the most common configuration mistakes BEFORE you wait two
#   minutes for HuggingFace downloads to start.
# * Pins the config path so you can't accidentally train against a stale
#   YAML living somewhere else.
# * Makes the train command discoverable from `ls scripts/`.
#
# Usage
# -----
#   bash scripts/train_qwen_lora.sh                # full training run
#   bash scripts/train_qwen_lora.sh --preprocess   # tokenize + cache only
#   bash scripts/train_qwen_lora.sh --resume PATH  # resume from a checkpoint
#
# Environment
# -----------
#   AXOLOTL_CONFIG    override the YAML path (default: configs/qwen_lora.yaml)
#   HF_HOME           override the HuggingFace cache root
#   WANDB_PROJECT     enable Weights & Biases logging if set

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${AXOLOTL_CONFIG:-configs/qwen_lora.yaml}"
TRAIN_JSONL="artifacts/sft_v2/train.jsonl"
VAL_JSONL="artifacts/sft_v2/val.jsonl"

# ── 0. help shortcut (must work even without axolotl installed) ──────────
case "${1:-}" in
    -h|--help)
        sed -n '1,30p' "$0"
        exit 0
        ;;
esac

# ── 1. preflight: tools ───────────────────────────────────────────────────
if ! command -v axolotl >/dev/null 2>&1; then
    echo "ERROR: 'axolotl' not on PATH." >&2
    echo >&2
    echo "Install in a fresh env (do NOT install into the llm-agent env —" >&2
    echo "axolotl pins torch/transformers versions that conflict with the" >&2
    echo "FastAPI/sqlalchemy stack):" >&2
    echo >&2
    echo "  conda create -n axolotl python=3.10 -y && conda activate axolotl" >&2
    echo "  pip install packaging ninja" >&2
    echo "  pip install 'axolotl[flash-attn,deepspeed]'" >&2
    exit 1
fi

# ── 2. preflight: config + dataset ────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: axolotl config not found at $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$TRAIN_JSONL" ]]; then
    echo "ERROR: training data not found at $TRAIN_JSONL" >&2
    echo >&2
    echo "Generate it first with:" >&2
    echo "  python -m scripts.generate_intent_pairs --n-per-stratum 8" >&2
    echo "  python -m scripts.export_sft_dataset --out artifacts/sft_v2" >&2
    exit 1
fi

N_TRAIN=$(wc -l < "$TRAIN_JSONL" | tr -d ' ')
N_VAL=0
if [[ -f "$VAL_JSONL" ]]; then
    N_VAL=$(wc -l < "$VAL_JSONL" | tr -d ' ')
fi

echo "─── ChatDFT intent-agent LoRA training ───"
echo "  config:    $CONFIG"
echo "  train:     $TRAIN_JSONL ($N_TRAIN examples)"
echo "  val:       $VAL_JSONL ($N_VAL examples)"
echo "  axolotl:   $(axolotl --version 2>&1 | head -1 || echo unknown)"
echo

if (( N_TRAIN < 500 )); then
    echo "WARNING: only $N_TRAIN training examples." >&2
    echo "  LoRA tends to underfit Qwen-7B below ~2k examples on a structured" >&2
    echo "  output task. Generate more pairs first:" >&2
    echo "    python -m scripts.generate_intent_pairs --n-per-stratum 30" >&2
    echo >&2
fi

# ── 3. dispatch ───────────────────────────────────────────────────────────
case "${1:-train}" in
    --preprocess|preprocess)
        echo "→ axolotl preprocess $CONFIG"
        axolotl preprocess "$CONFIG"
        ;;
    --resume)
        if [[ -z "${2:-}" ]]; then
            echo "ERROR: --resume requires a checkpoint path" >&2
            exit 2
        fi
        echo "→ axolotl train $CONFIG --resume_from_checkpoint $2"
        axolotl train "$CONFIG" --resume_from_checkpoint "$2"
        ;;
    train|"")
        echo "→ axolotl preprocess $CONFIG"
        axolotl preprocess "$CONFIG"
        echo
        echo "→ axolotl train $CONFIG"
        axolotl train "$CONFIG"
        ;;
    -h|--help)
        sed -n '1,30p' "$0"
        exit 0
        ;;
    *)
        echo "ERROR: unknown subcommand: ${1}" >&2
        echo "Try: $0 --help" >&2
        exit 2
        ;;
esac

OUT_DIR="artifacts/qwen_lora_out"
echo
echo "─── done ───"
echo "  output adapter:  $OUT_DIR"
echo
echo "Next steps:"
echo "  1. Eval against the intent regression set:"
echo "       python -m tests.intent_eval.harness \\"
echo "         --predictor live --min-field-accuracy 0.85"
echo "     (with the LoRA-served Qwen pointed at via CHATDFT_LLM_ROUTING_INTENT_AGENT=vllm_local)"
echo
echo "  2. Merge the adapter for production serving:"
echo "       axolotl merge-lora $CONFIG --lora-model-dir $OUT_DIR"
echo
echo "  3. Update server/llm.yaml routing.intent_agent: vllm_local"
