# server/chat/analyze_agent.py
# -*- coding: utf-8 -*-
"""
Analyze Agent — the "PhD advisor" layer.

Role
----
After DFT calculations complete (or partially complete), this agent:
1. Loads all DFTResult rows for the session.
2. Retrieves literature context via RAG for comparison.
3. Asks the LLM to reason like a senior computational chemist.
4. Returns:
   - Scientific conclusions  (findings with evidence + confidence)
   - Gaps                    (what's missing for publication)
   - Suggestions             (next calculations, sorted by priority)
   - Publication checklist   (what you have / what you still need)
   - Next workflow tasks      (feeds back into plan_agent)

Endpoints
---------
POST /chat/analyze
    Body: {session_id, focus?, result_ids?}
    Returns: {ok, summary_md, conclusions, gaps, suggestions,
              publication_checklist, next_tasks}

POST /chat/analyze/result
    Body: {session_id, task_id, result_type, species, value, extra?, job_uid?}
    Registers a single DFT result (called by post_analysis_agent).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy imports (to avoid circular deps)
# ---------------------------------------------------------------------------
try:
    from server.utils.openai_wrapper import chatgpt_call
except Exception:
    async def chatgpt_call(messages, **kw):  # type: ignore
        return {}

try:
    from server.utils.rag_utils import rag_context, log_agent_call
except Exception:
    async def rag_context(query, **kw):  # type: ignore
        return ""
    async def log_agent_call(*a, **kw):  # type: ignore
        pass

try:
    from server.db import (
        AsyncSessionLocal, ChatMessage, DFTResult, WorkflowTask,
        ReactionSystem, MechanismGraph,
    )
    _DB_OK = True
except Exception:
    _DB_OK = False
    AsyncSessionLocal = None


# ===========================================================================
# Helpers
# ===========================================================================

async def _load_session_context(session_id: int) -> Dict[str, Any]:
    """
    Pull intent, hypothesis, plan, and DFT results for a session.
    Returns a context dict for the LLM prompt.
    """
    ctx: Dict[str, Any] = {
        "session_id": session_id,
        "intent": None,
        "hypothesis_md": None,
        "plan_tasks": [],
        "dft_results": [],
        "workflow_tasks": [],
    }

    if not _DB_OK or AsyncSessionLocal is None:
        return ctx

    from sqlalchemy import select, desc

    try:
        async with AsyncSessionLocal() as s:
            # Latest messages of each key type
            for msg_type in ("intent", "hypothesis", "plan", "rxn_network"):
                stmt = (
                    select(ChatMessage)
                    .where(
                        ChatMessage.session_id == session_id,
                        ChatMessage.msg_type == msg_type,
                    )
                    .order_by(desc(ChatMessage.created_at))
                    .limit(1)
                )
                res = await s.execute(stmt)
                msg = res.scalar_one_or_none()
                if msg:
                    try:
                        ctx[msg_type] = json.loads(msg.content)
                    except Exception:
                        ctx[msg_type] = msg.content

            # DFT results
            dft_stmt = (
                select(DFTResult)
                .where(DFTResult.session_id == session_id)
                .order_by(DFTResult.created_at)
            )
            dft_res = await s.execute(dft_stmt)
            ctx["dft_results"] = [
                {
                    "type": r.result_type,
                    "species": r.species,
                    "surface": r.surface,
                    "site": r.site,
                    "value": r.value,
                    "unit": r.unit,
                    "converged": r.converged,
                    "extra": r.extra or {},
                    "warnings": r.warnings or [],
                }
                for r in dft_res.scalars().all()
            ]

            # Workflow task summary
            wf_stmt = (
                select(WorkflowTask)
                .where(WorkflowTask.session_id == session_id)
                .order_by(WorkflowTask.step_order)
            )
            wf_res = await s.execute(wf_stmt)
            ctx["workflow_tasks"] = [
                {
                    "name": t.name,
                    "task_type": t.task_type,
                    "status": t.status,
                    "agent": t.agent,
                }
                for t in wf_res.scalars().all()
            ]

    except Exception as e:
        log.error("_load_session_context failed: %s", e, exc_info=True)

    return ctx


def _build_analysis_prompt(ctx: Dict[str, Any], focus: str, rag_text: str) -> str:
    """
    Construct the analysis prompt from session context.
    The prompt asks the LLM to reason like a senior computational chemist.
    """
    intent_str = json.dumps(ctx.get("intent") or {}, indent=2)[:800]
    hyp_str    = str(ctx.get("hypothesis_md") or "Not yet generated.")[:600]
    dft_str    = json.dumps(ctx.get("dft_results") or [], indent=2)[:2000]
    tasks_str  = json.dumps(ctx.get("workflow_tasks") or [], indent=2)[:800]

    return f"""
You are a senior computational catalysis scientist reviewing the progress of a DFT study.

## Study context
{intent_str}

## Working hypothesis
{hyp_str}

## Calculated DFT results so far
{dft_str}

## Workflow task status
{tasks_str}

## Literature context (from RAG)
{rag_text[:2000] if rag_text else "Not available."}

## Analysis focus
{focus}

---
Your task: analyse the results as if writing the discussion section of a Nature Catalysis paper.

Return strict JSON with this schema:
{{
  "summary_md": "3-5 sentence markdown paragraph summarising what has been found",
  "conclusions": [
    {{
      "finding": "Concise scientific statement",
      "evidence": "Which DFT numbers support this",
      "confidence": 0.0-1.0,
      "comparison_with_literature": "brief note, or null"
    }}
  ],
  "gaps": [
    "Missing: TS for step X → Y",
    "Missing: potential-dependent analysis (GC-DFT)",
    "..."
  ],
  "suggestions": [
    {{
      "action": "short_action_key",
      "description": "Human-readable description",
      "priority": "critical|high|medium|low",
      "rationale": "Why this matters scientifically",
      "calc_type": "neb|gcdft|adsorption|microkinetics|water_cluster|surface_doping|pcet|dos|bader",
      "estimated_cost": "cheap|moderate|expensive"
    }}
  ],
  "publication_checklist": {{
    "status": "incomplete|near_complete|publishable",
    "present": ["ΔG diagram", "adsorption energies", "..."],
    "missing": ["micro-kinetic model", "solvent effect", "..."],
    "nice_to_have": ["PCET correction", "surface coverage study", "..."]
  }},
  "next_tasks": [
    {{
      "name": "Task name for plan_agent",
      "agent": "neb.ci_neb|structure.adsorption|gcdft.potential_sweep|...",
      "priority": 1,
      "depends_on": [],
      "params": {{}}
    }}
  ]
}}
"""


def _json_from_response(raw: Any) -> Dict:
    """Extract JSON dict from an OpenAI response object."""
    import re
    try:
        if isinstance(raw, dict):
            content = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "{}")
            )
        else:
            content = str(raw)

        content = re.sub(r"```(?:json)?", "", content).strip("` \n")
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


async def _save_analysis_message(session_id: int, content: str) -> Optional[int]:
    if not _DB_OK or AsyncSessionLocal is None:
        return None
    try:
        async with AsyncSessionLocal() as s:
            msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=content,
                msg_type="analysis",
            )
            s.add(msg)
            await s.commit()
            await s.refresh(msg)
            return msg.id
    except Exception as e:
        log.error("_save_analysis_message failed: %s", e)
        return None


# ===========================================================================
# Endpoints
# ===========================================================================

@router.post("/chat/analyze")
async def analyze(request: Request):
    """
    Main analysis endpoint.
    Loads all available DFT results for the session, runs LLM analysis,
    and returns structured scientific conclusions + next-step suggestions.
    """
    t0 = time.time()
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"}, status_code=400)

    session_id: Optional[int] = body.get("session_id")
    focus: str = body.get("focus", "overall progress and next steps")
    result_ids: Optional[List[int]] = body.get("result_ids")   # optional filter

    # 1. Load session context
    ctx = await _load_session_context(session_id) if session_id else {}

    if result_ids and ctx.get("dft_results"):
        # If caller wants to restrict analysis to specific result rows:
        ctx["dft_results"] = [
            r for r in ctx["dft_results"]
            # Note: DFTResult doesn't store id in dict above; add it in _load_session_context if needed
        ]

    # 2. Build RAG query from intent
    intent_obj = ctx.get("intent") or {}
    if isinstance(intent_obj, str):
        rag_q = intent_obj[:200]
    else:
        surface  = intent_obj.get("substrate") or intent_obj.get("system", {}).get("material", "")
        reaction = intent_obj.get("task") or intent_obj.get("problem_type", "")
        rag_q    = f"{reaction} on {surface} DFT mechanism"

    rag_text = await rag_context(
        query=rag_q,
        session_id=session_id,
        top_k=6,
        include_history=False,   # exclude chat history here; only literature
    )

    # 3. Build prompt + call LLM
    prompt = _build_analysis_prompt(ctx, focus, rag_text)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior computational catalysis scientist. "
                "You think quantitatively (in eV), cite literature trends, "
                "and give precise actionable recommendations. "
                "Always return valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    latency_start = time.time()
    raw = await chatgpt_call(messages, model="gpt-4o", temperature=0.1, max_tokens=3000)
    latency_ms = int((time.time() - latency_start) * 1000)

    parsed = _json_from_response(raw)

    # 4. Log the call
    usage = {}
    if isinstance(raw, dict):
        usage = raw.get("usage", {})

    asyncio.create_task(log_agent_call(
        agent_name="analyze_agent",
        call_type="llm",
        model="gpt-4o",
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=latency_ms,
        success=bool(parsed),
        input_preview=prompt[:500],
        output_preview=json.dumps(parsed)[:500],
        session_id=session_id,
    ))

    # 5. Persist as ChatMessage
    if parsed and session_id:
        asyncio.create_task(
            _save_analysis_message(session_id, json.dumps(parsed))
        )

    # 6. Build response
    if not parsed:
        return JSONResponse(
            {"ok": False, "error": "LLM returned empty analysis. Check AgentLog for details."},
            status_code=502,
        )

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "summary_md": parsed.get("summary_md", ""),
        "conclusions": parsed.get("conclusions", []),
        "gaps": parsed.get("gaps", []),
        "suggestions": parsed.get("suggestions", []),
        "publication_checklist": parsed.get("publication_checklist", {}),
        "next_tasks": parsed.get("next_tasks", []),
        "n_dft_results_used": len(ctx.get("dft_results", [])),
        "duration_s": round(time.time() - t0, 2),
    })


@router.post("/chat/analyze/result")
async def register_result(request: Request):
    """
    Register a single DFT result for a session.
    Called by post_analysis_agent after parsing VASP/QE output.

    Body: {
        session_id: int,
        task_id: int,
        result_type: str,   # "adsorption_energy" | "activation_barrier" | ...
        species: str,       # "C4H9*"
        surface: str,       # "Pt(111)"
        value: float,       # eV
        unit: str,          # default "eV"
        site: str,          # "bridge"
        extra: dict,        # {"zpe": 0.12, "ts_correction": -0.03}
        job_uid: str,
        converged: bool,
        warnings: list[str]
    }
    """
    if not _DB_OK or AsyncSessionLocal is None:
        return JSONResponse({"ok": False, "error": "DB not available"}, status_code=503)

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)

    required = ("session_id", "task_id", "result_type", "value")
    missing  = [f for f in required if f not in body]
    if missing:
        return JSONResponse(
            {"ok": False, "error": f"Missing fields: {missing}"},
            status_code=400,
        )

    try:
        async with AsyncSessionLocal() as s:
            row = DFTResult(
                session_id  = body["session_id"],
                task_id     = body["task_id"],
                result_type = body["result_type"],
                species     = body.get("species"),
                surface     = body.get("surface"),
                site        = body.get("site"),
                value       = float(body["value"]),
                unit        = body.get("unit", "eV"),
                extra       = body.get("extra", {}),
                job_uid     = body.get("job_uid"),
                converged   = body.get("converged", True),
                warnings    = body.get("warnings", []),
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return JSONResponse({"ok": True, "result_id": row.id})
    except Exception as e:
        log.error("register_result failed: %s", e, exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/chat/analyze/summary/{session_id}")
async def get_analysis_summary(session_id: int):
    """
    Quick summary endpoint: returns the latest analysis message for a session,
    plus a count of converged/failed DFT results.
    """
    if not _DB_OK or AsyncSessionLocal is None:
        return JSONResponse({"ok": False, "error": "DB not available"}, status_code=503)

    from sqlalchemy import select, desc, func

    try:
        async with AsyncSessionLocal() as s:
            # Latest analysis message
            stmt = (
                select(ChatMessage)
                .where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.msg_type == "analysis",
                )
                .order_by(desc(ChatMessage.created_at))
                .limit(1)
            )
            res = await s.execute(stmt)
            msg = res.scalar_one_or_none()

            # DFT result stats
            count_stmt = (
                select(DFTResult.converged, func.count(DFTResult.id))
                .where(DFTResult.session_id == session_id)
                .group_by(DFTResult.converged)
            )
            count_res = await s.execute(count_stmt)
            counts = {str(row[0]): row[1] for row in count_res.fetchall()}

        analysis = None
        if msg:
            try:
                analysis = json.loads(msg.content)
            except Exception:
                analysis = {"summary_md": msg.content}

        return JSONResponse({
            "ok": True,
            "session_id": session_id,
            "has_analysis": msg is not None,
            "analysis": analysis,
            "dft_stats": {
                "converged": counts.get("True", 0),
                "failed": counts.get("False", 0),
                "total": sum(counts.values()),
            },
        })
    except Exception as e:
        log.error("get_analysis_summary failed: %s", e, exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
