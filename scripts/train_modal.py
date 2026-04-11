#!/usr/bin/env python3
# scripts/train_modal.py
# -*- coding: utf-8 -*-
"""
One-command remote LoRA training for the intent agent, via Modal.

Why this exists
---------------
The intent-agent LoRA needs a 24 GB+ GPU (Qwen2.5-7B-Instruct at
``micro_batch_size=2, sequence_len=4096`` with flash-attention). Most
laptops don't have that. Modal lets you rent the GPU by the minute
without managing Docker images or provisioning; this script is the
"press one button" wrapper for our specific training + eval workflow.

Prerequisites (one-time)
------------------------
::

    pip install modal
    modal setup   # sign in and authorize

Prerequisites (every run)
-------------------------
The SFT corpus must exist locally under ``artifacts/sft_v2/``.
Regenerate with::

    DATABASE_URL=postgresql+asyncpg://yaniguan@localhost/chatdft_ase \\
      python -m scripts.export_sft_dataset --out artifacts/sft_v2

(The dataset is gitignored since it's mechanically regenerable from the
``intent_pair`` table — see ``artifacts/README.md``.)

Usage
-----
::

    # Full pipeline: train → eval → print delta vs baseline
    modal run scripts/train_modal.py

    # Pipeline sanity check first (overfit 10 epochs on 1736 rows, ~25 min)
    modal run scripts/train_modal.py --overfit

    # Train only, skip the eval pass
    modal run scripts/train_modal.py --train-only

    # Re-run eval against an already-trained adapter
    modal run scripts/train_modal.py --eval-only

    # Override the GPU class
    modal run scripts/train_modal.py --gpu A100-40GB

Pulling the adapter back to your Mac
------------------------------------
The trained LoRA weights live on a persistent Modal Volume named
``chatdft-intent-lora-output``. After the run finishes::

    modal volume get chatdft-intent-lora-output artifacts/qwen_lora_out .

Cost
----
At default settings (L4 GPU, 3 epochs on ~1700 examples):
  * L4 @ ~$1.05/h × ~1.5 h ≈ **$1.50 per full run**
  * Plus the first-run 15 GB Qwen-7B download (cached in a Volume
    afterward, free on subsequent runs)

Overfit config is faster: ~25 min ≈ ~$0.45.

Eval step (``eval_lora``) takes ~3 min on L4, ≈ $0.05.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "chatdft-intent-lora"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REMOTE_ROOT = "/workspace/chatdft"

# L4 is a sweet spot: 24 GB VRAM (enough for 7B LoRA at micro_batch_size=2),
# ~$1/h on Modal, available in every region. Override via --gpu at the CLI.
DEFAULT_GPU = "L4"

# Hard wall-clock cap per Modal function. A 3-epoch real run takes ~90 min
# on L4; 6 hours leaves ample headroom for dataset growth.
TRAIN_TIMEOUT_S = 6 * 60 * 60
EVAL_TIMEOUT_S = 30 * 60


app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Start from axolotl's official image — it already has pinned torch, flash
# attention, deepspeed, bitsandbytes, and the ``axolotl`` CLI on PATH. Cold
# starts are ~30 s instead of ~15 min for a from-scratch build.
_image = (
    modal.Image.from_registry(
        "winglian/axolotl:main-latest",
        add_python="3.10",
    )
    .pip_install(
        # Extra deps we need for eval_lora() that aren't in the axolotl image
        "pydantic>=2.0",
        "peft>=0.11",
    )
    # ── local code + data mounts ────────────────────────────────────────
    # Training: axolotl needs the configs and the JSONL corpus.
    .add_local_dir("./configs", f"{REMOTE_ROOT}/configs")
    .add_local_dir("./artifacts/sft_v2", f"{REMOTE_ROOT}/artifacts/sft_v2")
    # Eval: the harness + eval set + canonical schema validator + prompt.
    .add_local_dir("./tests/intent_eval", f"{REMOTE_ROOT}/tests/intent_eval")
    .add_local_file(
        "./server/chat/intent_schema.py",
        f"{REMOTE_ROOT}/server/chat/intent_schema.py",
    )
    .add_local_file(
        "./server/chat/intent_prompt.py",
        f"{REMOTE_ROOT}/server/chat/intent_prompt.py",
    )
    .add_local_file(
        "./server/__init__.py",
        f"{REMOTE_ROOT}/server/__init__.py",
    )
    .add_local_file(
        "./tests/intent_eval/__init__.py",
        f"{REMOTE_ROOT}/tests/intent_eval/__init__.py",
    )
    # No need for tests/__init__.py or server/chat/__init__.py — Python 3.3+
    # namespace packages handle sub-directories without an __init__.py
    # as long as the containing package isn't a regular package. Verified
    # locally that ``from server.chat.intent_prompt import ...`` resolves
    # correctly under this setup.
)


# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
# HF cache — avoids re-downloading 15 GB of Qwen-7B weights on every run.
hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)

# Training output — adapter weights, checkpoints, training logs. Pullable
# via `modal volume get chatdft-intent-lora-output ./artifacts/qwen_lora_out .`
output_vol = modal.Volume.from_name(
    f"{APP_NAME}-output", create_if_missing=True
)


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------

@app.function(
    image=_image,
    gpu=DEFAULT_GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{REMOTE_ROOT}/artifacts/qwen_lora_out": output_vol,
    },
    timeout=TRAIN_TIMEOUT_S,
)
def train_lora(
    config_rel_path: str,
    preprocess_only: bool = False,
) -> dict:
    """Run ``axolotl preprocess`` + ``axolotl train``."""
    import os
    import subprocess

    os.chdir(REMOTE_ROOT)

    print(f"→ axolotl preprocess {config_rel_path}")
    subprocess.check_call(["axolotl", "preprocess", config_rel_path])

    if preprocess_only:
        return {"status": "preprocessed_only", "config": config_rel_path}

    print(f"→ axolotl train {config_rel_path}")
    subprocess.check_call(["axolotl", "train", config_rel_path])

    # Commit so the adapter survives across function invocations and is
    # visible to eval_lora + the local `modal volume get` command.
    output_vol.commit()

    return {
        "status": "trained",
        "config": config_rel_path,
        "adapter_dir": f"{REMOTE_ROOT}/artifacts/qwen_lora_out",
    }


@app.function(
    image=_image,
    gpu=DEFAULT_GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{REMOTE_ROOT}/artifacts/qwen_lora_out": output_vol,
    },
    timeout=EVAL_TIMEOUT_S,
)
def eval_lora() -> dict:
    """
    Load the trained LoRA and score it against the 30-case eval set.

    Runs pure ``transformers`` + ``peft`` inference, no HTTP layer. Each
    query gets the same system prompt + user payload shape the production
    ``intent_agent`` sends to its LLM. Scoring uses the harness's pure
    ``score_case`` + ``aggregate`` functions — the same code path the
    ``--predictor live`` CLI uses against the FastAPI server.

    Returns the ``AggregateReport.to_dict()`` so the local entrypoint can
    diff it against ``artifacts/baseline_gpt4o_mini.json``.
    """
    import json as _json
    import sys as _sys

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _sys.path.insert(0, REMOTE_ROOT)
    from server.chat.intent_prompt import INTENT_SYSTEM_PROMPT
    from tests.intent_eval.harness import (  # noqa: E402
        CaseResult,
        aggregate,
        load_eval_set,
        score_case,
    )

    adapter_dir = f"{REMOTE_ROOT}/artifacts/qwen_lora_out"

    print(f"→ loading base model {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"→ loading LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    eval_path = Path(f"{REMOTE_ROOT}/tests/intent_eval/eval_set.jsonl")
    cases = load_eval_set(eval_path)
    print(f"→ running inference on {len(cases)} eval cases")

    results = []
    for i, case in enumerate(cases):
        user_payload = _json.dumps(
            {
                "query": case.query,
                "guided": {},
                "fewshots_hint": [],
                "rag_hint": "",
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Tolerant JSON extraction — LoRA output can still have stray text.
        predicted: dict = {}
        try:
            start = completion.find("{")
            end = completion.rfind("}")
            if start >= 0 and end > start:
                predicted = _json.loads(completion[start : end + 1])
        except (_json.JSONDecodeError, ValueError) as e:
            predicted = {"_parse_error": str(e)}

        field_scores, critical_em = score_case(predicted, case.gold)
        results.append(
            CaseResult(
                id=case.id,
                query=case.query,
                gold=case.gold,
                predicted=predicted,
                field_scores=field_scores,
                critical_em=critical_em,
            )
        )
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(cases)}] done")

    report = aggregate(results)
    print(f"→ aggregate: field_accuracy={report.field_accuracy:.4f} "
          f"critical_em={report.critical_em_rate:.4f}")
    return report.to_dict()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _preflight() -> None:
    """Bail early if the local side isn't ready."""
    errors: list[str] = []

    if not Path("artifacts/sft_v2/train.jsonl").exists():
        errors.append(
            "artifacts/sft_v2/train.jsonl not found. Regenerate with:\n"
            "    DATABASE_URL=postgresql+asyncpg://yaniguan@localhost/chatdft_ase \\\n"
            "      python -m scripts.export_sft_dataset --out artifacts/sft_v2"
        )

    if not Path("artifacts/sft_v2/val.jsonl").exists():
        errors.append("artifacts/sft_v2/val.jsonl not found (see above)")

    if errors:
        raise SystemExit("\n\n❌ preflight failed:\n\n" + "\n\n".join(errors))


def _print_delta_vs_baseline(report: dict) -> None:
    """Pretty-print the LoRA report alongside the saved GPT-4o-mini baseline."""
    baseline_path = Path("artifacts/baseline_gpt4o_mini.json")
    if not baseline_path.exists():
        print("(no baseline to compare against — "
              "artifacts/baseline_gpt4o_mini.json missing)")
        return

    baseline = json.loads(baseline_path.read_text())

    def _row(label: str, key: str, fmt: str = ".4f") -> str:
        a = baseline.get(key)
        b = report.get(key)
        if a is None or b is None:
            return f"  {label:24s} —"
        delta = b - a
        sign = "+" if delta >= 0 else ""
        return f"  {label:24s} {a:{fmt}} → {b:{fmt}}   ({sign}{delta:{fmt}})"

    print()
    print("─── LoRA vs GPT-4o-mini baseline ─────────────────────────")
    print(_row("field_accuracy", "field_accuracy"))
    print(_row("critical_em_rate", "critical_em_rate"))

    pf_a = baseline.get("per_field_accuracy") or {}
    pf_b = report.get("per_field_accuracy") or {}
    for key in sorted(set(pf_a) | set(pf_b)):
        a = pf_a.get(key, 0.0)
        b = pf_b.get(key, 0.0)
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"  {key:24s} {a:.4f} → {b:.4f}   ({sign}{delta:.4f})")

    # Ship gate
    thresh = (baseline.get("ship_thresholds") or {})
    field_gate = thresh.get("field_accuracy_gte", 0.0)
    em_gate = thresh.get("critical_em_rate_gte", 0.0)
    gated = (
        report["field_accuracy"] >= field_gate
        and report["critical_em_rate"] >= em_gate
    )
    print()
    if gated:
        print("✅ meets ship thresholds — ready to route intent_agent to vllm_local")
    else:
        print("❌ below ship thresholds — iterate before flipping routing")
        print(f"   need field_accuracy ≥ {field_gate}, critical_em ≥ {em_gate}")


@app.local_entrypoint()
def main(
    config: str = "configs/qwen_lora.yaml",
    overfit: bool = False,
    train_only: bool = False,
    eval_only: bool = False,
    preprocess_only: bool = False,
):
    """
    Entry point called by ``modal run scripts/train_modal.py [--flag ...]``.
    """
    _preflight()

    if overfit:
        config = "configs/qwen_lora_overfit.yaml"

    if not Path(config).exists():
        raise SystemExit(f"❌ config not found: {config}")

    n_train = sum(1 for _ in open("artifacts/sft_v2/train.jsonl"))
    n_val = (
        sum(1 for _ in open("artifacts/sft_v2/val.jsonl"))
        if Path("artifacts/sft_v2/val.jsonl").exists()
        else 0
    )

    print("─── ChatDFT intent LoRA on Modal ─────────────────────────")
    print(f"  config:          {config}")
    print(f"  base model:      {BASE_MODEL}")
    print(f"  train samples:   {n_train}")
    print(f"  val samples:     {n_val}")
    print(f"  GPU:             {DEFAULT_GPU}")
    print(f"  train only:      {train_only}")
    print(f"  eval only:       {eval_only}")
    print(f"  preprocess only: {preprocess_only}")
    print()

    # ── train ────────────────────────────────────────────────────────────
    if not eval_only:
        print("─── remote: train_lora ───────────────────────────────────")
        train_result = train_lora.remote(
            config, preprocess_only=preprocess_only
        )
        print(f"✓ train: {train_result}")
        if preprocess_only:
            return

    # ── eval ─────────────────────────────────────────────────────────────
    if not train_only and not preprocess_only:
        print()
        print("─── remote: eval_lora ────────────────────────────────────")
        report = eval_lora.remote()
        print(f"✓ eval: {json.dumps(report, indent=2)}")

        _print_delta_vs_baseline(report)

        # Persist the LoRA eval alongside the baseline for future diffing.
        out_path = Path("artifacts/lora_eval.json")
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print()
        print(f"→ saved eval report to {out_path}")

    print()
    print("─── done ─────────────────────────────────────────────────")
    print("To pull the trained LoRA adapter down to your Mac:")
    print(f"  modal volume get {APP_NAME}-output "
          f"artifacts/qwen_lora_out ./artifacts/qwen_lora_out")
    print()
    print("To serve it locally via vLLM:")
    print("  # merge the adapter into the base (run on the GPU box):")
    print(f"  axolotl merge-lora {config} "
          "--lora-model-dir artifacts/qwen_lora_out")
    print("  # then point vllm at the merged output dir")
