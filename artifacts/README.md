# artifacts/

Outputs from the intent-agent training pipeline. Files in this directory
split into two groups:

## Checked-in (small, specific, worth preserving)

- **`baseline_gpt4o_mini.json`** — canonical baseline measurement of the
  production GPT-4o-mini intent agent against
  `tests/intent_eval/eval_set.jsonl`. The number the fine-tuned Qwen-7B
  LoRA has to beat. Regenerating it costs ~$0.10 in gpt-4o-mini credits
  plus ~2 min of wall time — see `docs/intent_lora_training.md` §5a.

## Regenerable (gitignored; each rebuild is idempotent from the DB)

- **`sft_v1/`**, **`sft_v2/`** — SFT JSONL corpora exported from
  `intent_pair`. Regenerate any time with:

  ```bash
  python -m scripts.export_sft_dataset --out artifacts/sft_v2
  ```

- **`qwen_lora_out/`**, **`qwen_lora_overfit_out/`** — LoRA adapter
  weights + training logs. Regenerate by running
  `bash scripts/train_qwen_lora.sh` on a GPU box. Each ~1 GB.

- **`qwen_lora_prepared/`**, **`qwen_lora_overfit_prepared/`** —
  axolotl's tokenized dataset cache. Deleted by axolotl between runs
  anyway, so don't hand-edit.

- **`*.log`** — raw stdout/stderr captures from eval or training runs.
  Useful for post-mortem but not durable enough to check in.

## Why this split

The baseline JSON is a **measurement** that took API credits to produce
and is the reference for every future comparison — it's worth its 1 KB
in git history. Everything else can be mechanically rebuilt from
`intent_pair` + the base Qwen-7B weights, so paying git LFS or repo
bloat costs for them is a bad trade.

If you need the old corpus or adapter for a historical comparison,
tag the commit in git and regenerate from that tag.
