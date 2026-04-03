# server/execution/job_watcher.py
# -*- coding: utf-8 -*-
"""
JobWatcher — the missing bridge between HPC cluster and the ChatDFT feedback loop.

Pipeline once a job finishes
-----------------------------
  HPC cluster (COMPLETED) ──► fetch_outputs()
        ──► PostAnalysisAgent.analyze()
        ──► insert DFTResult rows into PostgreSQL
        ──► POST /chat/hypothesis/feedback   (closes the learning loop)
        ──► WorkflowTask.status = "done"

Usage::

    from server.execution.job_watcher import watch_job

    # Non-blocking: schedules watching in background
    asyncio.create_task(
        watch_job(
            task_id=42,
            job_id="12345",
            job_dir=Path("/tmp/jobs/pt111_relax"),
            hpc=hpc_agent,
            session_id=7,
            species="H",
            surface="Pt(111)",
            poll_interval=60,
        )
    )
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    from server.execution.post_analysis_agent import PostAnalysisAgent
    _HAS_PA = True
except ImportError:
    _HAS_PA = False

try:
    from server.execution.utils.task_status import emit_task_status
    _HAS_TS = True
except ImportError:
    _HAS_TS = False
    async def emit_task_status(*a, **kw): return False  # type: ignore

try:
    from sqlalchemy import select
    from server.db import AsyncSessionLocal, DFTResult, WorkflowTask
    _DB_OK = True
except ImportError:
    _DB_OK = False
    AsyncSessionLocal = None  # type: ignore

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    try:
        import requests as _requests
        _HAS_HTTPX = False
    except ImportError:
        _HAS_HTTPX = False
        _requests = None  # type: ignore

import os
_BACKEND = os.getenv("CHATDFT_BACKEND", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Feedback submission
# ---------------------------------------------------------------------------

async def _post_hypothesis_feedback(
    session_id: int,
    result_type: str,
    species: str,
    surface: str,
    value: float,
    converged: bool = True,
    extra: Optional[Dict] = None,
) -> bool:
    """Call POST /chat/hypothesis/feedback to close the learning loop."""
    payload = {
        "session_id": session_id,
        "result_type": result_type,
        "species": species,
        "surface": surface,
        "value": value,
        "converged": converged,
        "extra": extra or {},
    }
    url = f"{_BACKEND}/chat/hypothesis/feedback"
    try:
        if _HAS_HTTPX:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, json=payload)
                return r.status_code < 300
        elif _requests is not None:
            r = _requests.post(url, json=payload, timeout=30)
            return r.status_code < 300
        else:
            log.warning("job_watcher: no HTTP client available for feedback")
            return False
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("_post_hypothesis_feedback failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# DFT result persistence
# ---------------------------------------------------------------------------

async def _persist_dft_results(
    session_id: int,
    task_id: int,
    records: List[Dict[str, Any]],
) -> List[int]:
    """
    Insert DFTResult rows into PostgreSQL from JobRecord dicts.
    Returns list of inserted ids.
    """
    if not _DB_OK:
        return []

    inserted_ids = []
    try:
        async with AsyncSessionLocal() as db:
            for rec in records:
                label = rec.get("label", "")
                engine = rec.get("engine", "vasp")
                calc = rec.get("calc", "scf")

                # Map calc type → result_type
                if rec.get("E_ads_eV") is not None:
                    rt = "adsorption_energy"
                    val = rec["E_ads_eV"]
                elif rec.get("barrier_eV") is not None:
                    rt = "activation_barrier"
                    val = rec["barrier_eV"]
                elif rec.get("bandgap_eV") is not None:
                    rt = "dos"
                    val = rec["bandgap_eV"]
                elif rec.get("E_eV") is not None:
                    rt = "total_energy"
                    val = rec["E_eV"]
                else:
                    continue   # no numeric result

                converged = not bool(rec.get("debug_issues"))
                warnings = rec.get("debug_issues") or []

                extra = {k: rec[k] for k in (
                    "fermi_eV", "bandgap_eV", "spin_mag", "zpe_eV", "notes", "figs"
                ) if rec.get(k) is not None}

                row = DFTResult(
                    task_id=task_id,
                    session_id=session_id,
                    result_type=rt,
                    species=label,
                    surface="",          # caller should override if known
                    value=float(val),
                    unit="eV",
                    extra=extra,
                    converged=converged,
                    warnings=warnings,
                )
                db.add(row)
                await db.flush()
                inserted_ids.append(row.id)

            await db.commit()
    except Exception as exc:
        log.error("_persist_dft_results failed: %s", exc, exc_info=True)

    return inserted_ids


# ---------------------------------------------------------------------------
# Main watcher coroutine
# ---------------------------------------------------------------------------

async def watch_job(
    *,
    task_id: int,
    job_id: str,
    job_dir: Path,
    hpc,                        # HPCAgent instance
    session_id: int,
    species: str = "",
    surface: str = "",
    poll_interval: int = 60,    # seconds between squeue/qstat polls
    max_wait: int = 86400,      # give up after 24 h
) -> Dict[str, Any]:
    """
    Async coroutine that:
    1. Polls the cluster until the job is COMPLETED or FAILED.
    2. Fetches output files (CONTCAR, OUTCAR, OSZICAR, stdout).
    3. Runs PostAnalysisAgent.analyze() on the local job directory.
    4. Persists DFTResult rows into PostgreSQL.
    5. Calls POST /chat/hypothesis/feedback for each result.
    6. Updates WorkflowTask.status to "done" or "failed".

    Designed to run as a background asyncio Task.
    """
    job_dir = Path(job_dir)
    t_start = time.monotonic()

    await emit_task_status(task_id, "running")

    # ── 1. Poll until terminal state ──────────────────────────────────────
    terminal_status = "FAILED"
    while (time.monotonic() - t_start) < max_wait:
        await asyncio.sleep(poll_interval)
        try:
            status_info = hpc.status(job_id)
            raw_status = (status_info.get("status") or "").upper()
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("watch_job poll error for job %s: %s", job_id, exc)
            continue

        log.debug("job %s → %s (task_id=%d)", job_id, raw_status, task_id)

        if raw_status in ("COMPLETED", "DONE", "CD"):
            terminal_status = "COMPLETED"
            break
        if raw_status in ("FAILED", "CANCELLED", "TIMEOUT", "F", "CA", "TO"):
            terminal_status = "FAILED"
            break
        # RUNNING / PENDING → keep waiting

    elapsed = time.monotonic() - t_start

    # ── 2. Fetch outputs ─────────────────────────────────────────────────
    try:
        hpc.fetch_outputs(
            job_dir,
            filters=["CONTCAR", "OUTCAR", "OSZICAR", "stdout", "vasprun.xml"],
        )
        log.info("watch_job: fetched outputs for task_id=%d", task_id)
    except Exception as exc:
        log.warning("watch_job: fetch_outputs failed for task_id=%d: %s", task_id, exc)

    # ── 3. Post-analysis ──────────────────────────────────────────────────
    records: List[Dict[str, Any]] = []
    if _HAS_PA and terminal_status == "COMPLETED":
        try:
            pa = PostAnalysisAgent()
            result = pa.analyze(job_dir)
            # Load the parsed records from results.json
            import json
            rj = job_dir / "results.json"
            if rj.exists():
                records = json.loads(rj.read_text())
            log.info("watch_job: post-analysis done, %d records (task_id=%d)", len(records), task_id)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("watch_job: post_analysis failed for task_id=%d: %s", task_id, exc)

    # ── 4. Persist DFTResult rows ─────────────────────────────────────────
    inserted_ids = []
    if records:
        inserted_ids = await _persist_dft_results(session_id, task_id, records)
        log.info("watch_job: inserted %d DFTResult rows (task_id=%d)", len(inserted_ids), task_id)

    # ── 5. Fire hypothesis feedback ───────────────────────────────────────
    feedback_sent = 0
    for rec in records:
        try:
            if rec.get("E_ads_eV") is not None:
                ok = await _post_hypothesis_feedback(
                    session_id=session_id,
                    result_type="adsorption_energy",
                    species=species or rec.get("label", ""),
                    surface=surface,
                    value=float(rec["E_ads_eV"]),
                    converged=not bool(rec.get("debug_issues")),
                    extra={"label": rec.get("label"), "calc": rec.get("calc")},
                )
                feedback_sent += int(ok)
            elif rec.get("barrier_eV") is not None:
                ok = await _post_hypothesis_feedback(
                    session_id=session_id,
                    result_type="activation_barrier",
                    species=species or rec.get("label", ""),
                    surface=surface,
                    value=float(rec["barrier_eV"]),
                    converged=True,
                    extra={"label": rec.get("label")},
                )
                feedback_sent += int(ok)
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("watch_job: feedback post failed: %s", exc)

    # ── 6. Update WorkflowTask ────────────────────────────────────────────
    final_status = "done" if terminal_status == "COMPLETED" else "failed"
    output_data = {
        "job_id": job_id,
        "terminal_cluster_status": terminal_status,
        "n_records": len(records),
        "dft_result_ids": inserted_ids,
        "feedback_sent": feedback_sent,
        "elapsed_s": round(elapsed, 1),
    }
    error_msg = None if terminal_status == "COMPLETED" else f"cluster status: {terminal_status}"
    await emit_task_status(
        task_id, final_status,
        output_data=output_data,
        error_msg=error_msg,
        run_time=elapsed,
    )

    log.info(
        "watch_job complete: task_id=%d  cluster=%s  dft_results=%d  feedback=%d  elapsed=%.0fs",
        task_id, terminal_status, len(records), feedback_sent, elapsed,
    )
    return output_data
