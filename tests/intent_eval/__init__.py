"""Intent agent evaluation harness.

The contents of ``eval_set.jsonl`` are the frozen regression set used to
gate changes to the intent prompt and to compare candidate models
(GPT-4o-mini vs fine-tuned Qwen-7B vs DPO checkpoint, etc.). Add new
hand-labeled rows freely; never delete existing ones without bumping a
version line.
"""
