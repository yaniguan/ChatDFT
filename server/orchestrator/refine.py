# server/orchestrator/refine.py
"""
LLM-driven hypothesis refinement.

Inputs:
  - current hypothesis (markdown + structured graph)
  - new completed results (DFT outputs since last refine)
  - reward signals attached to those results
  - per-agent budget status (so the LLM knows which kinds are still allowed)

Output (strict JSON):
  {
    "updated_hypothesis_md": "...",   # short delta — appended to existing MD
    "actions": [ ProposedAction-shaped dicts ],
    "stop_reason": null | "..."
  }

Validation is delegated to :mod:`server.orchestrator.actions`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from server.orchestrator.actions import (
    ProposedAction,
    llm_action_schema_prompt,
    validate_action_batch,
)
from server.orchestrator.state import OrchestrationState

log = logging.getLogger(__name__)


# Hard stop: never let the prompt grow past this many chars
_MAX_HYP_CHARS = 1500
_MAX_RESULT_CHARS = 2000
_MAX_REWARD_CHARS = 800


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _format_results(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results[-10:]:           # only the last 10 to keep prompt bounded
        lines.append(
            f"- task#{r.get('task_id', '?')} {r.get('agent', '?')} → "
            f"{r.get('result_type', 'unknown')}: "
            f"{r.get('species', '')} on {r.get('surface', '')} "
            f"= {r.get('value', '?')} eV (converged={r.get('converged', '?')})"
        )
    return _truncate("\n".join(lines), _MAX_RESULT_CHARS) or "(no results yet)"


def _format_rewards(rewards: List[Dict[str, Any]]) -> str:
    if not rewards:
        return "(no reward signals yet)"
    lines = [
        f"- {r.get('species','?')}/{r.get('surface','?')}: r={r.get('reward','?'):.2f}  ({r.get('details','')})"
        for r in rewards[-8:]
    ]
    return _truncate("\n".join(lines), _MAX_REWARD_CHARS)


def _format_budgets(state: OrchestrationState) -> str:
    lines = []
    ema = state.reward_ema()
    for name, b in state.budgets.items():
        flags = []
        if b.max_rounds is not None:
            flags.append(f"rounds {b.rounds_used}/{b.max_rounds}")
        if b.use_reward_gate:
            flags.append(f"reward_gate(ema≥{b.reward_ema_threshold} & streak≥{b.streak_threshold}); "
                         f"current ema={ema:.2f} streak={b.no_new_streak}")
        status = "EXHAUSTED" if b.is_exhausted(ema) else "OPEN"
        lines.append(f"  - {name}: {status}  [{', '.join(flags) or 'uncapped'}]")
    return "\n".join(lines)


def _build_prompt(state: OrchestrationState) -> str:
    return f"""You are the ChatDFT closed-loop refiner. Given the current hypothesis,
the latest DFT results, and reward signals comparing predictions to actuals,
propose at most {len(state.budgets)} bounded follow-up actions that will most
efficiently sharpen, extend, or falsify the hypothesis.

CURRENT HYPOTHESIS (markdown, possibly truncated):
{_truncate(state.hypothesis_md, _MAX_HYP_CHARS) or "(empty)"}

REACTION NETWORK:
intermediates = {state.hypothesis_graph.get('intermediates', [])}
elementary_steps = {state.hypothesis_graph.get('reaction_network', [])}
ts_candidates = {state.hypothesis_graph.get('ts_candidates', [])}

RECENT RESULTS:
{_format_results(state.completed_results)}

RECENT REWARD SIGNALS (r ∈ [-1, 1]; +1 = matched prediction):
{_format_rewards(state.reward_history)}

CURRENT BUDGET STATUS (do NOT propose actions whose primary agent is EXHAUSTED):
{_format_budgets(state)}
  → kind "extend" / "scan" / "challenge" mostly drive the structure agent
  → kind "verify" with reconverge / functional drives the parameter agent

LOOP STATE: iteration={state.iteration}/{state.max_iterations}, \
confidence={state.confidence():.2f}, no_new_streak={state.no_new_actions_streak}

{llm_action_schema_prompt()}

Decision rules:
- If reward EMA is high (≥0.8) and you've already extended the network, prefer
  CHALLENGE actions (try to falsify). Don't keep proposing more EXTEND.
- If the latest result contradicted the prediction (reward < 0), VERIFY first.
- If the hypothesis claims "X depends on Y" but you have <3 data points along Y,
  propose a SCAN.
- Be ruthless about cost: each action costs at least 1 HPC job.
- Output empty actions list and set stop_reason="hypothesis sufficiently supported"
  if there's nothing genuinely useful to add.

Return ONLY the JSON object — no preamble.
"""


async def refine_hypothesis(
    state: OrchestrationState,
) -> Tuple[str, List[ProposedAction], List[Dict[str, Any]], str]:
    """
    Call the LLM to propose follow-up actions.

    Returns
    -------
    (updated_hypothesis_md, accepted_actions, rejected_action_dicts, stop_reason)

    ``stop_reason`` may be a non-empty string if the LLM decided to stop.
    Validation rejections are returned in ``rejected_action_dicts`` for audit.
    """
    from server.utils.openai_wrapper import chatgpt_call

    prompt = _build_prompt(state)
    messages = [
        {"role": "system", "content": "You output strict JSON only. No prose."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = await chatgpt_call(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=1200,
            json_mode=True,
            agent_name="orchestrator.refine",
            session_id=state.session_id,
        )
    except Exception as exc:        # pragma: no cover — defensive
        log.exception("refine LLM call crashed")
        return ("", [], [], f"llm_error: {exc}")

    if "error" in resp and not resp.get("choices"):
        return ("", [], [], f"llm_error: {resp['error']}")

    try:
        text = resp["choices"][0]["message"]["content"]
        payload = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        log.warning("refine: bad LLM JSON: %s", exc)
        return ("", [], [], f"llm_bad_json: {exc}")

    updated_md = str(payload.get("updated_hypothesis_md", "")).strip()
    raw_actions = payload.get("actions") or []
    if not isinstance(raw_actions, list):
        raw_actions = []

    accepted, errors = validate_action_batch(raw_actions)

    rejected: List[Dict[str, Any]] = []
    if errors:
        # Group errors back to their source actions (best-effort)
        rejected = [{"raw": raw_actions, "errors": [str(e) for e in errors]}]
        log.info("refine: %d action(s) rejected by validator: %s",
                 len(errors), [str(e) for e in errors])

    stop_reason = str(payload.get("stop_reason") or "").strip()
    if not accepted and not stop_reason:
        # LLM gave us nothing actionable AND no explicit reason — synthetic
        stop_reason = "llm_returned_no_actions"

    return updated_md, accepted, rejected, stop_reason
