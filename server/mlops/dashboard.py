# server/mlops/dashboard.py
# -*- coding: utf-8 -*-
"""
ChatDFT Monitoring Dashboard — Aggregation Layer
================================================

Single source of truth for dashboard metrics. Computes:

1. System-level online metrics (traffic, latency, finish/success/error rate,
   error-rate slope, retry rate, active workflow count, optional token/cost)
2. Agent-level online metrics (request count, success/error, slope,
   p50/p95/p99 latency, retries, handoff success, schema-valid rate,
   timeout rate, token usage)
3. Agent-level offline evaluation metrics (per-agent golden-set scores from
   ``science.evaluation.metrics``; marked "not yet instrumented" if the
   relevant evaluation pipeline has not been run)
4. Alert thresholds and regression detection

Data sources (real, not mocked):
--------------------------------
* ``agent_log``     — every LLM call (``server/utils/openai_wrapper.chatgpt_call``
                      now auto-instruments this). Used for per-agent latency,
                      success, schema-valid and token stats.
* ``workflow_task`` — plan-agent-emitted steps executed by the execution layer.
                      Workflow states: ``idle / queued / running / done / failed``.
                      Used for e2e finish rate, success rate, p99 latency,
                      active workflow count and retry pressure.
* ``chat_message``  — historical proxy for per-agent activity before the
                      AgentLog auto-logging landed (``msg_type`` ≈ agent name).
                      Used to back-fill request counts where AgentLog has zero
                      rows for a known agent.
* ``execution_step``— execution-layer per-step status; provides downstream
                      handoff signal.

Alert thresholds are defined in one place (``ALERTS``) so both the backend
and the dashboard UI use identical rules.

Formulas
--------
* rolling_window(N minutes) = [now - N*60, now]
* p99 latency(agent)        = np.percentile(successful_latency_ms_in_window, 99)
* success_rate(agent)       = count(success=True) / max(1, count(*))
* error_rate(agent)         = 1 - success_rate
* finish_rate(workflows)    = count(status ∈ {done, failed}) / max(1, count(*))
* slope_error_rate(agent)   = linear fit (np.polyfit, deg=1) of bucketed
                              error_rate over ``slope_buckets`` windows,
                              expressed as Δrate per hour
* retry_rate(agent)         = count(retries_used > 0) / max(1, count(*))
* schema_valid_rate(agent)  = count(call_type='llm_json') /
                              max(1, count(call_type ∈ {llm_json, llm_json_invalid}))
* timeout_rate(agent)       = count(error_msg ILIKE '%timeout%') /
                              max(1, count(*))
* handoff_success(session)  = fraction of sessions where the next downstream
                              agent also produced an AgentLog row
* cost_usd(agent)           = Σ input_tokens * rate_in + output_tokens * rate_out
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent registry — canonical list + responsibility → offline metric map
# ---------------------------------------------------------------------------

#: Canonical ChatDFT agents. Discovery falls back to this when DB is empty.
AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Chat / reasoning agents (LLM-backed)
    "intent_agent": {
        "role": "Parse natural-language query into structured intent JSON.",
        "layer": "chat",
        "llm_backed": True,
        "json_producer": True,
        "downstream": ["hypothesis_agent"],
        "offline_metrics": [
            "intent_stage_accuracy",
            "intent_material_match",
            "intent_facet_match",
        ],
    },
    "hypothesis_agent": {
        "role": "Propose reaction mechanism and intermediates.",
        "layer": "chat",
        "llm_backed": True,
        "json_producer": True,
        "downstream": ["plan_agent"],
        "offline_metrics": [
            "hypothesis_intermediate_recall",
            "hypothesis_intermediate_precision",
            "hypothesis_intermediate_f1",
            "hypothesis_forward_direction",
        ],
    },
    "plan_agent": {
        "role": "Convert hypothesis into an executable DFT task graph.",
        "layer": "chat",
        "llm_backed": True,
        "json_producer": True,
        "downstream": ["structure_agent", "parameters_agent"],
        "offline_metrics": [
            "plan_task_graph_validity",         # not yet instrumented
            "plan_task_coverage",               # not yet instrumented
            "plan_malformed_rate",              # derived from schema-valid rate
        ],
    },
    "knowledge_agent": {
        "role": "Retrieve literature evidence via hybrid RAG.",
        "layer": "chat",
        "llm_backed": True,
        "json_producer": False,
        "downstream": ["hypothesis_agent", "analyze_agent"],
        "offline_metrics": [
            "rag_hit_rate_at_k",
            "rag_mrr",
            "rag_ndcg_at_k",
        ],
    },
    "analyze_agent": {
        "role": "Summarise DFT results, cite literature, suggest next steps.",
        "layer": "chat",
        "llm_backed": True,
        "json_producer": True,
        "downstream": [],
        "offline_metrics": [
            "analyze_parse_success",
            "analyze_extraction_completeness",  # not yet instrumented
            "analyze_consistency",              # not yet instrumented
        ],
    },
    "qa_agent": {
        "role": "Answer freeform scientific questions.",
        "layer": "chat",
        "llm_backed": True,
        "json_producer": False,
        "downstream": [],
        "offline_metrics": [],  # conversational — no golden set
    },
    # Execution / scientific agents
    "structure_agent": {
        "role": "Build bulk / slab / adsorption POSCARs.",
        "layer": "execution",
        "llm_backed": False,
        "json_producer": False,
        "downstream": ["parameters_agent"],
        "offline_metrics": [
            "structure_build_success_rate",
            "structure_site_correctness",  # not yet instrumented
        ],
    },
    "parameters_agent": {
        "role": "Emit VASP/QE INCAR, KPOINTS, POTCAR settings.",
        "layer": "execution",
        "llm_backed": False,
        "json_producer": False,
        "downstream": ["hpc_agent"],
        "offline_metrics": [
            "params_benchmark_agreement",  # not yet instrumented
        ],
    },
    "hpc_agent": {
        "role": "Submit, poll and recover HPC jobs.",
        "layer": "execution",
        "llm_backed": False,
        "json_producer": False,
        "downstream": ["post_analysis_agent"],
        "offline_metrics": [
            "hpc_submission_success",
            "hpc_poll_success",
            "hpc_recovery_success",  # not yet instrumented
        ],
    },
    "post_analysis_agent": {
        "role": "Parse OUTCAR/vasprun.xml → DFTResult.",
        "layer": "execution",
        "llm_backed": False,
        "json_producer": False,
        "downstream": ["analyze_agent"],
        "offline_metrics": [
            "post_parse_success_rate",
            "post_extraction_completeness",  # not yet instrumented
        ],
    },
}

# Map ChatMessage.msg_type → agent_name (used when AgentLog is sparse)
MSG_TYPE_TO_AGENT = {
    "intent": "intent_agent",
    "hypothesis": "hypothesis_agent",
    "rxn_network": "hypothesis_agent",
    "plan": "plan_agent",
    "analysis": "analyze_agent",
    "knowledge": "knowledge_agent",
    "history": None,
    "clarification": None,
}

#: WorkflowTask.agent prefixes that belong to each execution agent.
#: ChatDFT uses two conventions side-by-side:
#:   * dotted: ``structure.relax_slab`` / ``post.energy`` / ``electronic.dos``
#:   * flat:   ``adsorption`` / ``neb`` / ``microkinetic`` (from intent_agent)
#: The matcher checks exact match, dotted prefix, and flat aliases.
EXECUTION_AGENT_PREFIXES: Dict[str, List[str]] = {
    "structure_agent": [
        "structure", "adsorption", "coadsorption", "slab_build", "neb",
    ],
    "parameters_agent": [
        "parameters", "params", "incar", "kpoints",
    ],
    "hpc_agent": [
        "hpc", "submit", "poll", "job",
    ],
    "post_analysis_agent": [
        "post", "post_analysis", "postprocess", "analyze_result",
    ],
}


def _workflow_matches_agent(wf_agent: str, agent_name: str) -> bool:
    """Return True if a WorkflowTask.agent value belongs to ``agent_name``."""
    if not wf_agent:
        return False
    prefixes = EXECUTION_AGENT_PREFIXES.get(agent_name, [])
    if not prefixes:
        # Fall back to stripping ``_agent`` and matching dotted / exact
        fallback = agent_name.replace("_agent", "")
        prefixes = [fallback]
    head = wf_agent.split(".", 1)[0].lower()
    wf_lower = wf_agent.lower()
    return any(
        head == p or wf_lower.startswith(f"{p}.") or wf_lower == p
        for p in prefixes
    )

# Token cost rates (USD per 1k tokens). Tune via env vars if needed.
_MODEL_COST: Dict[str, Tuple[float, float]] = {
    # model        : (input/1k, output/1k)
    "gpt-4o":         (0.005, 0.015),
    "gpt-4o-mini":    (0.00015, 0.0006),
    "gpt-4-turbo":    (0.01, 0.03),
    "gpt-3.5-turbo":  (0.0005, 0.0015),
}


def _split_provider_model(tagged: str) -> Tuple[str, str]:
    """
    Parse a ``provider:model`` string as written by ``openai_wrapper``.

    Returns ``(provider, model)``. Falls back to ``("unknown", tagged)`` if
    there is no ``:`` separator (pre-vLLM rows from an earlier deploy).
    """
    if not tagged:
        return ("unknown", "")
    # vLLM model names often contain ``/`` (HF repo) but never ``:`` as a
    # separator, so a single split on the first colon is unambiguous.
    if ":" in tagged:
        provider, model = tagged.split(":", 1)
        return (provider or "unknown", model)
    return ("unknown", tagged)


def _cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    """
    USD cost estimate. Understands ``provider:model`` tagging from the
    vLLM integration and strips the provider prefix before rate lookup.
    Local providers (``vllm_*``) always return 0.
    """
    provider, bare_model = _split_provider_model(model) if ":" in model else ("", model)
    if provider.startswith("vllm"):
        return 0.0
    rate_in, rate_out = _MODEL_COST.get(bare_model, (0.0, 0.0))
    return (in_tok * rate_in + out_tok * rate_out) / 1000.0


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

@dataclass
class AlertThresholds:
    p99_latency_ms: int = 20000          # system p99 latency budget
    agent_p99_latency_ms: int = 25000    # per-agent p99 latency budget
    finish_rate_floor: float = 0.85      # e2e workflow completion floor
    success_rate_floor: float = 0.90     # e2e success rate floor
    error_rate_ceiling: float = 0.15     # per-agent error rate ceiling
    error_slope_ceiling: float = 0.05    # Δerror-rate per hour ceiling
    schema_valid_floor: float = 0.90     # JSON-producing agent schema floor
    timeout_rate_ceiling: float = 0.10

    # Slope regression: compare the last bucket to the mean of all preceding
    # buckets; trigger if (last - mean_prior) > drift_threshold.
    drift_threshold: float = 0.10


ALERTS = AlertThresholds()


@dataclass
class Alert:
    severity: str        # "warning" | "critical"
    scope: str           # "system" | "agent"
    target: str          # "system" or agent name
    metric: str
    message: str
    value: float
    threshold: float


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], pct: float) -> float:
    """np.percentile wrapper that returns 0 for empty input."""
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def linear_slope_per_hour(
    bucket_centres_s: Sequence[float],
    rates: Sequence[float],
    counts: Optional[Sequence[int]] = None,
) -> float:
    """
    Linear fit of rate over time. Returns Δrate per hour.

    Uses ``numpy.polyfit(deg=1)``. Empty buckets (count==0) have an
    undefined rate and are excluded from the fit so a period of silence
    doesn't get counted as "improving". Returns 0 if fewer than 2 valid
    buckets remain.
    """
    if len(rates) != len(bucket_centres_s):
        return 0.0
    if counts is not None and len(counts) == len(rates):
        xs = [x for x, c in zip(bucket_centres_s, counts) if c > 0]
        ys = [y for y, c in zip(rates, counts) if c > 0]
    else:
        xs = list(bucket_centres_s)
        ys = list(rates)
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float) / 3600.0  # seconds → hours
    y = np.asarray(ys, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def bucketise(
    timestamps: Sequence[float],
    values: Sequence[float],
    n_buckets: int,
    window_start: float,
    window_end: float,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Split [window_start, window_end] into ``n_buckets`` equal buckets and
    return (bucket_centre_seconds_from_start, mean_value, count).
    """
    if n_buckets <= 0 or window_end <= window_start:
        return [], [], []
    edges = np.linspace(window_start, window_end, n_buckets + 1)
    centres: List[float] = []
    means: List[float] = []
    counts: List[int] = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        sel = [v for t, v in zip(timestamps, values) if lo <= t < hi]
        centres.append((lo + hi) / 2.0 - window_start)
        counts.append(len(sel))
        means.append(float(np.mean(sel)) if sel else 0.0)
    return centres, means, counts


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

@dataclass
class AgentLogRow:
    agent_name: str
    call_type: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    error_msg: Optional[str]
    session_id: Optional[int]
    created_at: float  # unix seconds


@dataclass
class WorkflowTaskRow:
    id: int
    session_id: Optional[int]
    agent: str
    task_type: str
    status: str
    run_time: float
    error_msg: Optional[str]
    created_at: float
    updated_at: float


async def _fetch_agent_logs(window_s: int) -> List[AgentLogRow]:
    try:
        from sqlalchemy import select
        from server.db import AgentLog, AsyncSessionLocal
    except Exception as e:
        log.warning("dashboard: AgentLog import failed (%s)", e)
        return []

    cutoff = datetime.utcnow() - timedelta(seconds=window_s)
    try:
        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(AgentLog).where(AgentLog.created_at >= cutoff)
                )
            ).scalars().all()
    except Exception as e:
        log.warning("dashboard: AgentLog query failed (%s)", e)
        return []

    out: List[AgentLogRow] = []
    for r in rows:
        out.append(AgentLogRow(
            agent_name=r.agent_name or "unknown",
            call_type=r.call_type or "",
            model=r.model or "",
            input_tokens=int(r.input_tokens or 0),
            output_tokens=int(r.output_tokens or 0),
            latency_ms=int(r.latency_ms or 0),
            success=bool(r.success),
            error_msg=r.error_msg,
            session_id=r.session_id,
            created_at=(r.created_at or datetime.utcnow()).replace(tzinfo=None).timestamp(),
        ))
    return out


async def _fetch_workflow_tasks(window_s: int) -> List[WorkflowTaskRow]:
    try:
        from sqlalchemy import select, or_
        from server.db import WorkflowTask, AsyncSessionLocal
    except Exception as e:
        log.warning("dashboard: WorkflowTask import failed (%s)", e)
        return []

    cutoff = datetime.utcnow() - timedelta(seconds=window_s)
    try:
        async with AsyncSessionLocal() as s:
            # Include tasks updated inside the window even if created earlier,
            # so that long-running workflows show up in the active count.
            rows = (
                await s.execute(
                    select(WorkflowTask).where(
                        or_(WorkflowTask.updated_at >= cutoff,
                            WorkflowTask.created_at >= cutoff)
                    )
                )
            ).scalars().all()
    except Exception as e:
        log.warning("dashboard: WorkflowTask query failed (%s)", e)
        return []

    out: List[WorkflowTaskRow] = []
    for r in rows:
        out.append(WorkflowTaskRow(
            id=int(r.id),
            session_id=r.session_id,
            agent=(r.agent or ""),
            task_type=(r.task_type or ""),
            status=(r.status or "idle"),
            run_time=float(r.run_time or 0.0),
            error_msg=r.error_msg,
            created_at=(r.created_at or datetime.utcnow()).replace(tzinfo=None).timestamp(),
            updated_at=(r.updated_at or datetime.utcnow()).replace(tzinfo=None).timestamp(),
        ))
    return out


async def _fetch_chat_messages_by_agent(window_s: int) -> Dict[str, int]:
    """Historical request count proxy for agents that predate AgentLog auto-logging."""
    try:
        from sqlalchemy import select, func
        from server.db import ChatMessage, AsyncSessionLocal
    except Exception:
        return {}

    cutoff = datetime.utcnow() - timedelta(seconds=window_s)
    try:
        async with AsyncSessionLocal() as s:
            rows = (
                await s.execute(
                    select(ChatMessage.msg_type, func.count(ChatMessage.id))
                    .where(ChatMessage.created_at >= cutoff)
                    .group_by(ChatMessage.msg_type)
                )
            ).all()
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    for msg_type, n in rows:
        agent = MSG_TYPE_TO_AGENT.get(msg_type)
        if agent:
            counts[agent] = counts.get(agent, 0) + int(n)
    return counts


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class SystemMetrics:
    window_seconds: int
    active_workflows: int
    total_workflows: int
    finished_workflows: int
    successful_workflows: int
    failed_workflows: int
    finish_rate: float
    success_rate: float
    error_rate: float
    error_rate_slope_per_hour: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    retry_rate: float
    recent_traffic: int  # created in last 5 min
    total_tokens: int
    total_cost_usd: float
    finish_trend: List[float] = field(default_factory=list)
    error_trend: List[float] = field(default_factory=list)
    bucket_centres_s: List[float] = field(default_factory=list)


def _system_metrics(
    workflows: List[WorkflowTaskRow],
    logs: List[AgentLogRow],
    window_s: int,
    n_buckets: int,
) -> SystemMetrics:
    now = time.time()
    window_start = now - window_s

    # Workflow-level aggregates
    active_states = {"idle", "queued", "running"}
    done_state = "done"
    failed_state = "failed"
    terminal_states = {done_state, failed_state}

    total = len(workflows)
    done = sum(1 for w in workflows if w.status == done_state)
    failed = sum(1 for w in workflows if w.status == failed_state)
    active = sum(1 for w in workflows if w.status in active_states)
    finished = done + failed

    finish_rate = finished / total if total else 0.0
    success_rate = done / finished if finished else 0.0
    error_rate = failed / finished if finished else 0.0

    # Latency: only for tasks that completed in the window
    run_times_ms = [w.run_time * 1000.0 for w in workflows if w.status == done_state and w.run_time > 0]
    p50 = percentile(run_times_ms, 50)
    p95 = percentile(run_times_ms, 95)
    p99 = percentile(run_times_ms, 99)

    # Recent traffic — workflows created in last 5 minutes
    recent_cut = now - 300
    recent_traffic = sum(1 for w in workflows if w.created_at >= recent_cut)

    # Retry rate — derived from AgentLog (failed calls as a proxy for retry pressure)
    n_logs = len(logs)
    n_log_failed = sum(1 for r in logs if not r.success)
    retry_rate = n_log_failed / n_logs if n_logs else 0.0

    # Token / cost accounting (LLM agents only)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in logs)
    total_cost = sum(_cost_usd(r.model, r.input_tokens, r.output_tokens) for r in logs)

    # Error-rate slope over buckets (by workflow creation time)
    wf_ts = [w.created_at for w in workflows if w.status in terminal_states]
    wf_err = [1.0 if w.status == failed_state else 0.0 for w in workflows if w.status in terminal_states]
    centres, err_trend, err_counts = bucketise(
        wf_ts, wf_err, n_buckets, window_start, now
    )
    _, fin_trend, _ = bucketise(
        [w.created_at for w in workflows],
        [1.0 if w.status in terminal_states else 0.0 for w in workflows],
        n_buckets,
        window_start,
        now,
    )
    slope = linear_slope_per_hour(centres, err_trend, err_counts)

    return SystemMetrics(
        window_seconds=window_s,
        active_workflows=active,
        total_workflows=total,
        finished_workflows=finished,
        successful_workflows=done,
        failed_workflows=failed,
        finish_rate=finish_rate,
        success_rate=success_rate,
        error_rate=error_rate,
        error_rate_slope_per_hour=slope,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        retry_rate=retry_rate,
        recent_traffic=recent_traffic,
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 4),
        finish_trend=fin_trend,
        error_trend=err_trend,
        bucket_centres_s=centres,
    )


@dataclass
class ProviderMetrics:
    """Roll-up of AgentLog rows grouped by LLM provider.

    The monitoring dashboard uses this to show a per-backend breakdown
    (OpenAI vs vLLM vs Anthropic) without needing a schema migration —
    the provider is encoded into ``AgentLog.model`` as ``"provider:model"``
    by the openai_wrapper whenever the vLLM router is active. Rows
    without a provider prefix (legacy) roll up under ``"unknown"``.
    """
    provider: str
    request_count: int
    success_rate: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    tokens_total: int
    tokens_per_sec: float
    total_cost_usd: float
    distinct_models: List[str] = field(default_factory=list)


def _provider_metrics(
    logs: List[AgentLogRow], window_s: int
) -> List[ProviderMetrics]:
    """Group AgentLog rows by provider and compute headline metrics."""
    groups: Dict[str, List[AgentLogRow]] = {}
    for r in logs:
        provider, _ = _split_provider_model(r.model) if r.model else ("unknown", "")
        groups.setdefault(provider, []).append(r)

    out: List[ProviderMetrics] = []
    window_s_safe = max(window_s, 1)
    for provider, rows in sorted(groups.items()):
        n = len(rows)
        if n == 0:
            continue
        success = sum(1 for r in rows if r.success)
        latencies = [r.latency_ms for r in rows if r.latency_ms > 0]
        in_tok = sum(r.input_tokens for r in rows)
        out_tok = sum(r.output_tokens for r in rows)
        total_cost = sum(_cost_usd(r.model, r.input_tokens, r.output_tokens) for r in rows)
        models = sorted({_split_provider_model(r.model)[1] for r in rows if r.model})
        out.append(ProviderMetrics(
            provider=provider,
            request_count=n,
            success_rate=success / n,
            error_rate=(n - success) / n,
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            tokens_total=in_tok + out_tok,
            tokens_per_sec=round((in_tok + out_tok) / window_s_safe, 2),
            total_cost_usd=round(total_cost, 4),
            distinct_models=models[:5],
        ))
    return out


@dataclass
class AgentMetrics:
    name: str
    role: str
    layer: str
    llm_backed: bool
    json_producer: bool
    request_count: int
    success_rate: float
    error_rate: float
    error_rate_slope_per_hour: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    retry_rate: float
    schema_valid_rate: Optional[float]
    timeout_rate: float
    handoff_success_rate: Optional[float]
    total_tokens: int
    total_cost_usd: float
    sample_window_s: int
    latency_trend: List[float] = field(default_factory=list)
    error_trend: List[float] = field(default_factory=list)
    bucket_centres_s: List[float] = field(default_factory=list)
    historical_request_count: int = 0  # from ChatMessage (pre-instrumentation)
    notes: List[str] = field(default_factory=list)


def _agent_metrics(
    name: str,
    reg: Dict[str, Any],
    logs: List[AgentLogRow],
    workflows: List[WorkflowTaskRow],
    historical_counts: Dict[str, int],
    window_s: int,
    n_buckets: int,
) -> AgentMetrics:
    now = time.time()
    window_start = now - window_s
    notes: List[str] = []

    # LLM-backed agents draw primarily from AgentLog.
    if reg.get("llm_backed", False):
        agent_logs = [r for r in logs if r.agent_name == name]
    else:
        # Execution agents: map WorkflowTask.agent via EXECUTION_AGENT_PREFIXES.
        # Only include tasks that have reached a terminal state so latency is
        # meaningful (``done`` / ``failed``).
        wf_rows = [
            w for w in workflows
            if _workflow_matches_agent(w.agent, name)
            and w.status in ("done", "failed")
        ]
        agent_logs = [
            AgentLogRow(
                agent_name=name,
                call_type="workflow",
                model="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=int(w.run_time * 1000),
                success=(w.status == "done"),
                error_msg=w.error_msg,
                session_id=w.session_id,
                created_at=w.updated_at or w.created_at,
            )
            for w in wf_rows
        ]

    n = len(agent_logs)
    if n == 0:
        notes.append("no events in window")

    success = sum(1 for r in agent_logs if r.success)
    error = n - success
    success_rate = success / n if n else 0.0
    error_rate = error / n if n else 0.0

    latencies = [r.latency_ms for r in agent_logs if r.latency_ms > 0]
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    # Retry: chatgpt_call writes retries_used into full_input when > 0. Because
    # AgentLog.full_input is JSON, we approximate retry rate as the fraction of
    # rows that failed. For execution agents, use tasks with error_msg.
    retries = sum(1 for r in agent_logs if not r.success)
    retry_rate = retries / n if n else 0.0

    # Schema validity (JSON producers only)
    if reg.get("json_producer", False) and reg.get("llm_backed", False):
        json_rows = [r for r in agent_logs if r.call_type in ("llm_json", "llm_json_invalid")]
        if json_rows:
            valid = sum(1 for r in json_rows if r.call_type == "llm_json")
            schema_valid_rate: Optional[float] = valid / len(json_rows)
        else:
            schema_valid_rate = None
            notes.append("schema_valid_rate: no JSON calls in window")
    else:
        schema_valid_rate = None

    # Timeout rate
    timeouts = sum(
        1 for r in agent_logs
        if r.error_msg and ("timeout" in r.error_msg.lower() or "timed out" in r.error_msg.lower())
    )
    timeout_rate = timeouts / n if n else 0.0

    # Handoff success rate — for each agent in this session, did the next
    # downstream agent also produce a log row in the same session?
    downstream = reg.get("downstream") or []
    if downstream and n > 0:
        log_by_session: Dict[int, set] = {}
        for r in logs:
            if r.session_id is None:
                continue
            log_by_session.setdefault(r.session_id, set()).add(r.agent_name)
        sessions_with_me = {r.session_id for r in agent_logs if r.session_id is not None}
        if sessions_with_me:
            handed = sum(
                1 for sid in sessions_with_me
                if any(d in log_by_session.get(sid, set()) for d in downstream)
            )
            handoff_success_rate: Optional[float] = handed / len(sessions_with_me)
        else:
            handoff_success_rate = None
            notes.append("handoff: no sessions with session_id")
    else:
        handoff_success_rate = None

    # Tokens and cost (LLM only)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in agent_logs)
    total_cost = sum(_cost_usd(r.model, r.input_tokens, r.output_tokens) for r in agent_logs)

    # Error-rate slope (bucketed, empty buckets excluded from the fit)
    ts = [r.created_at for r in agent_logs]
    err_values = [0.0 if r.success else 1.0 for r in agent_logs]
    lat_values = [float(r.latency_ms) for r in agent_logs]
    centres, err_trend, err_counts = bucketise(
        ts, err_values, n_buckets, window_start, now
    )
    _, lat_trend, _ = bucketise(ts, lat_values, n_buckets, window_start, now)
    slope = linear_slope_per_hour(centres, err_trend, err_counts)

    return AgentMetrics(
        name=name,
        role=reg.get("role", ""),
        layer=reg.get("layer", ""),
        llm_backed=bool(reg.get("llm_backed")),
        json_producer=bool(reg.get("json_producer")),
        request_count=n,
        success_rate=success_rate,
        error_rate=error_rate,
        error_rate_slope_per_hour=slope,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        retry_rate=retry_rate,
        schema_valid_rate=schema_valid_rate,
        timeout_rate=timeout_rate,
        handoff_success_rate=handoff_success_rate,
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost, 4),
        sample_window_s=window_s,
        latency_trend=lat_trend,
        error_trend=err_trend,
        bucket_centres_s=centres,
        historical_request_count=historical_counts.get(name, 0),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Offline evaluation metrics (golden-dataset backed)
# ---------------------------------------------------------------------------

@dataclass
class OfflineMetric:
    name: str
    value: Optional[float]        # None → not yet instrumented
    source: str                   # "golden_dataset" | "runtime" | "not_instrumented"
    higher_is_better: bool = True
    note: str = ""


def _offline_metrics_for_agent(
    agent_name: str, logs: List[AgentLogRow], workflows: List[WorkflowTaskRow]
) -> List[OfflineMetric]:
    """
    Return offline metrics. These come from:

    * ``science.evaluation.metrics`` when a benchmark run has been executed
      and cached. Otherwise marked "not_instrumented".
    * Runtime-derived metrics where meaningful (e.g. schema-valid rate ≈
      plan ``plan_malformed_rate``).
    """
    reg = AGENT_REGISTRY.get(agent_name, {})
    metric_names = reg.get("offline_metrics") or []
    out: List[OfflineMetric] = []

    # Try to load cached evaluation results. The repo ships an evaluation
    # module but does not persist runs — so almost everything falls through
    # to "not_instrumented" unless the user wires up a nightly job.
    cached = _load_cached_offline_metrics()

    for m in metric_names:
        if m in cached:
            val, src = cached[m]
            out.append(OfflineMetric(name=m, value=val, source=src))
            continue

        # Runtime-derived fallbacks
        if m == "plan_malformed_rate":
            json_rows = [r for r in logs if r.agent_name == "plan_agent"]
            total = sum(1 for r in json_rows if r.call_type in ("llm_json", "llm_json_invalid"))
            bad = sum(1 for r in json_rows if r.call_type == "llm_json_invalid")
            if total:
                out.append(OfflineMetric(
                    name=m,
                    value=bad / total,
                    source="runtime",
                    higher_is_better=False,
                    note="derived from AgentLog llm_json_invalid rate",
                ))
            else:
                out.append(OfflineMetric(
                    name=m, value=None, source="not_instrumented",
                    higher_is_better=False,
                ))
            continue

        if m == "analyze_parse_success":
            ar = [r for r in logs if r.agent_name == "analyze_agent"]
            if ar:
                ok = sum(1 for r in ar if r.success)
                out.append(OfflineMetric(
                    name=m, value=ok / len(ar), source="runtime",
                    note="derived from AgentLog success flag",
                ))
            else:
                out.append(OfflineMetric(name=m, value=None, source="not_instrumented"))
            continue

        if m == "structure_build_success_rate":
            wt = [w for w in workflows if w.agent.startswith("structure")]
            if wt:
                ok = sum(1 for w in wt if w.status == "done")
                out.append(OfflineMetric(
                    name=m, value=ok / len(wt), source="runtime",
                    note="derived from WorkflowTask status",
                ))
            else:
                out.append(OfflineMetric(name=m, value=None, source="not_instrumented"))
            continue

        if m in ("hpc_submission_success", "hpc_poll_success"):
            wt = [w for w in workflows if w.agent.startswith("hpc") or w.task_type in ("submit", "poll")]
            if wt:
                ok = sum(1 for w in wt if w.status == "done")
                out.append(OfflineMetric(
                    name=m, value=ok / len(wt), source="runtime",
                    note="derived from WorkflowTask status",
                ))
            else:
                out.append(OfflineMetric(name=m, value=None, source="not_instrumented"))
            continue

        if m == "post_parse_success_rate":
            wt = [w for w in workflows if w.agent.startswith("post") or w.task_type in ("post_analysis",)]
            if wt:
                ok = sum(1 for w in wt if w.status == "done")
                out.append(OfflineMetric(
                    name=m, value=ok / len(wt), source="runtime",
                    note="derived from WorkflowTask status",
                ))
            else:
                out.append(OfflineMetric(name=m, value=None, source="not_instrumented"))
            continue

        # Default: not instrumented
        out.append(OfflineMetric(name=m, value=None, source="not_instrumented"))
    return out


def _load_cached_offline_metrics() -> Dict[str, Tuple[float, str]]:
    """
    Optional: load cached offline metric values from a JSON file written by a
    nightly evaluation job. The file path can be overridden via
    ``CHATDFT_OFFLINE_METRICS_JSON``. Returns ``{metric_name: (value, source)}``.
    """
    path = os.environ.get(
        "CHATDFT_OFFLINE_METRICS_JSON",
        os.path.join(os.path.dirname(__file__), "..", "..", "runs", "offline_metrics.json"),
    )
    try:
        with open(path, "r") as f:
            blob = __import__("json").load(f)
    except Exception:
        return {}
    out: Dict[str, Tuple[float, str]] = {}
    for k, v in blob.items():
        if isinstance(v, (int, float)):
            out[k] = (float(v), "golden_dataset")
        elif isinstance(v, dict) and "value" in v:
            out[k] = (float(v["value"]), str(v.get("source", "golden_dataset")))
    return out


# ---------------------------------------------------------------------------
# Alert derivation
# ---------------------------------------------------------------------------

def derive_alerts(system: SystemMetrics, agents: List[AgentMetrics]) -> List[Alert]:
    alerts: List[Alert] = []

    # System-level
    if system.p99_latency_ms > ALERTS.p99_latency_ms and system.finished_workflows > 0:
        alerts.append(Alert(
            severity="warning",
            scope="system",
            target="system",
            metric="p99_latency_ms",
            message=f"p99 latency {system.p99_latency_ms:.0f} ms exceeds budget {ALERTS.p99_latency_ms} ms",
            value=system.p99_latency_ms,
            threshold=ALERTS.p99_latency_ms,
        ))
    if system.total_workflows >= 5 and system.finish_rate < ALERTS.finish_rate_floor:
        alerts.append(Alert(
            severity="critical" if system.finish_rate < 0.5 else "warning",
            scope="system",
            target="system",
            metric="finish_rate",
            message=f"finish rate {system.finish_rate:.1%} below floor {ALERTS.finish_rate_floor:.0%}",
            value=system.finish_rate,
            threshold=ALERTS.finish_rate_floor,
        ))
    if system.finished_workflows >= 5 and system.success_rate < ALERTS.success_rate_floor:
        alerts.append(Alert(
            severity="critical" if system.success_rate < 0.5 else "warning",
            scope="system",
            target="system",
            metric="success_rate",
            message=f"success rate {system.success_rate:.1%} below floor {ALERTS.success_rate_floor:.0%}",
            value=system.success_rate,
            threshold=ALERTS.success_rate_floor,
        ))
    if system.error_rate_slope_per_hour > ALERTS.error_slope_ceiling:
        alerts.append(Alert(
            severity="warning",
            scope="system",
            target="system",
            metric="error_rate_slope",
            message=(
                f"error rate trending up: Δ{system.error_rate_slope_per_hour:+.3f}/h "
                f"exceeds ceiling {ALERTS.error_slope_ceiling:+.3f}/h"
            ),
            value=system.error_rate_slope_per_hour,
            threshold=ALERTS.error_slope_ceiling,
        ))

    # Agent-level
    for a in agents:
        if a.request_count < 5:
            continue  # ignore noisy tiny samples
        if a.p99_latency_ms > ALERTS.agent_p99_latency_ms:
            alerts.append(Alert(
                severity="warning", scope="agent", target=a.name,
                metric="p99_latency_ms",
                message=f"{a.name} p99 {a.p99_latency_ms:.0f} ms > {ALERTS.agent_p99_latency_ms} ms",
                value=a.p99_latency_ms, threshold=ALERTS.agent_p99_latency_ms,
            ))
        if a.error_rate > ALERTS.error_rate_ceiling:
            alerts.append(Alert(
                severity="critical" if a.error_rate > 0.4 else "warning",
                scope="agent", target=a.name,
                metric="error_rate",
                message=f"{a.name} error rate {a.error_rate:.1%} > {ALERTS.error_rate_ceiling:.0%}",
                value=a.error_rate, threshold=ALERTS.error_rate_ceiling,
            ))
        if a.error_rate_slope_per_hour > ALERTS.error_slope_ceiling:
            alerts.append(Alert(
                severity="warning", scope="agent", target=a.name,
                metric="error_rate_slope",
                message=(
                    f"{a.name} error rate trending up "
                    f"Δ{a.error_rate_slope_per_hour:+.3f}/h"
                ),
                value=a.error_rate_slope_per_hour,
                threshold=ALERTS.error_slope_ceiling,
            ))
        if a.schema_valid_rate is not None and a.schema_valid_rate < ALERTS.schema_valid_floor:
            alerts.append(Alert(
                severity="warning", scope="agent", target=a.name,
                metric="schema_valid_rate",
                message=(
                    f"{a.name} JSON schema-valid rate "
                    f"{a.schema_valid_rate:.1%} < {ALERTS.schema_valid_floor:.0%}"
                ),
                value=a.schema_valid_rate, threshold=ALERTS.schema_valid_floor,
            ))
        if a.timeout_rate > ALERTS.timeout_rate_ceiling:
            alerts.append(Alert(
                severity="warning", scope="agent", target=a.name,
                metric="timeout_rate",
                message=f"{a.name} timeout rate {a.timeout_rate:.1%} > {ALERTS.timeout_rate_ceiling:.0%}",
                value=a.timeout_rate, threshold=ALERTS.timeout_rate_ceiling,
            ))

    return alerts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def compute_dashboard(
    window_minutes: int = 60,
    n_buckets: int = 12,
) -> Dict[str, Any]:
    """
    Compute the full dashboard payload over a rolling window.

    ``n_buckets`` buckets are used for error/latency sparklines and slope fit.
    """
    window_s = max(60, window_minutes * 60)
    logs, workflows, historical = await asyncio.gather(
        _fetch_agent_logs(window_s),
        _fetch_workflow_tasks(window_s),
        _fetch_chat_messages_by_agent(window_s),
    )

    # Discover agents: registry + anything observed in AgentLog / WorkflowTask
    observed = set(r.agent_name for r in logs)
    for w in workflows:
        if w.agent:
            prefix = w.agent.split(".", 1)[0]
            observed.add(f"{prefix}_agent" if not prefix.endswith("_agent") else prefix)
    all_agents = sorted(set(AGENT_REGISTRY.keys()) | observed)

    agent_metrics: List[AgentMetrics] = []
    agent_offline: Dict[str, List[OfflineMetric]] = {}
    for name in all_agents:
        reg = AGENT_REGISTRY.get(name, {
            "role": "Discovered from runtime logs.",
            "layer": "unknown",
            "llm_backed": True,
            "json_producer": False,
            "downstream": [],
            "offline_metrics": [],
        })
        am = _agent_metrics(name, reg, logs, workflows, historical, window_s, n_buckets)
        agent_metrics.append(am)
        agent_offline[name] = _offline_metrics_for_agent(name, logs, workflows)

    system = _system_metrics(workflows, logs, window_s, n_buckets)
    providers = _provider_metrics(logs, window_s)
    alerts = derive_alerts(system, agent_metrics)

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_minutes": window_minutes,
        "n_buckets": n_buckets,
        "thresholds": asdict(ALERTS),
        "system": asdict(system),
        "providers": [asdict(p) for p in providers],
        "agents": [asdict(a) for a in agent_metrics],
        "offline_metrics": {
            name: [asdict(m) for m in metrics]
            for name, metrics in agent_offline.items()
        },
        "alerts": [asdict(a) for a in alerts],
        "formulas": FORMULA_HELP,
    }


# ---------------------------------------------------------------------------
# Formula help (surfaced on the dashboard "Help" panel)
# ---------------------------------------------------------------------------

FORMULA_HELP: Dict[str, str] = {
    "rolling_window":
        "All metrics are computed over the last `window_minutes` minutes. "
        "Default 60 minutes, configurable per request.",
    "p99_latency_ms":
        "np.percentile(run_time_ms, 99) over WorkflowTasks with status='done' "
        "completed in the window. Per-agent p99 uses AgentLog.latency_ms.",
    "finish_rate":
        "count(WorkflowTask.status ∈ {done, failed}) / max(1, count(*)). "
        "Fraction of workflow tasks that reached a terminal state.",
    "success_rate":
        "count(status='done') / max(1, count(status ∈ {done, failed})). "
        "Success rate among tasks that terminated.",
    "error_rate":
        "1 - success_rate (per agent or system).",
    "error_rate_slope_per_hour":
        "np.polyfit(bucket_centres_hours, bucketed_error_rate, 1)[0]. "
        "Positive slope = error rate trending up. Alerts at "
        "`thresholds.error_slope_ceiling` Δ/hour.",
    "retry_rate":
        "fraction of AgentLog rows with success=False (approximates retry pressure "
        "since the openai_wrapper retries internally before persisting).",
    "schema_valid_rate":
        "count(call_type='llm_json') / max(1, count(call_type ∈ {llm_json, llm_json_invalid})). "
        "Only defined for JSON-producing agents. None ⇒ no JSON calls in window.",
    "handoff_success_rate":
        "Fraction of sessions where the agent produced an AgentLog row AND at least "
        "one downstream agent produced one in the same session. None ⇒ no session ids.",
    "timeout_rate":
        "count(error_msg ILIKE '%timeout%') / max(1, count(*)). ",
    "total_cost_usd":
        "Σ (input_tokens * rate_in + output_tokens * rate_out) / 1000 using the "
        "`_MODEL_COST` rate card.",
    "active_workflows":
        "count(WorkflowTask.status ∈ {idle, queued, running}) inside the window.",
    "recent_traffic":
        "count(WorkflowTask.created_at ≥ now - 5 min) — last-5-minute arrivals.",
    "providers":
        "Group AgentLog rows by the 'provider' prefix of the model column "
        "(openai_wrapper now writes 'provider:model', e.g. 'vllm_local:Qwen/"
        "Qwen2.5-7B-Instruct'). Per-provider request count, p50/p95/p99 "
        "latency, tokens/sec, success rate, and USD cost (local providers "
        "always $0). Rows without a prefix roll up under 'unknown'.",
}
