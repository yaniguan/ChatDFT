"""
ChatDFT Closed-Loop Orchestrator — Streamlit Page
==================================================

Drives ``/api/orchestrator/*``: start a closed-loop run on a session,
poll its progress, inspect proposed/rejected actions, stop early.

Run::

    streamlit run client/pages/orchestrator.py
    # or via the main ChatDFT app sidebar
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from client.utils import api  # noqa: E402

st.set_page_config(
    page_title="Orchestrator | ChatDFT",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_REASON_LABELS = {
    "max_iterations_reached":     ("Max iterations reached", "#6b7280"),
    "confidence_threshold_reached": ("Confidence threshold reached", "#15803d"),
    "no_new_actions_streak":      ("No new actions (converged)", "#15803d"),
    "all_agent_budgets_exhausted": ("All agent budgets exhausted", "#b45309"),
    "user_stopped":               ("User stopped", "#6b7280"),
    "error":                      ("Errored out", "#b91c1c"),
}

ACTION_KIND_COLOR = {
    "verify":    "#2563eb",   # blue
    "extend":    "#15803d",   # green
    "challenge": "#b45309",   # amber
    "scan":      "#7c3aed",   # purple
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _pill(text: str, *, color: str = "#374151", bg: str = "#f3f4f6") -> str:
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{bg};color:{color};font-size:0.78rem;font-weight:600;'
        f'margin:2px 3px;">{text}</span>'
    )


def _kind_pill(kind: str) -> str:
    color = ACTION_KIND_COLOR.get(kind, "#374151")
    return _pill(kind.upper(), color="white", bg=color)


def _stop_pill(reason: Optional[str]) -> str:
    if not reason:
        return _pill("RUNNING", color="white", bg="#2563eb")
    label, color = STOP_REASON_LABELS.get(reason, (reason, "#6b7280"))
    return _pill(label, color="white", bg=color)


def _fmt_secs(t: Optional[float]) -> str:
    if t is None:
        return "—"
    return time.strftime("%H:%M:%S", time.localtime(t))


def _confidence_bar(c: float) -> str:
    pct = max(0.0, min(1.0, c)) * 100
    color = "#15803d" if c >= 0.85 else "#b45309" if c >= 0.5 else "#b91c1c"
    return (
        f'<div style="background:#e5e7eb;border-radius:6px;width:100%;height:14px;'
        f'overflow:hidden;"><div style="background:{color};width:{pct:.0f}%;'
        f'height:100%"></div></div>'
        f'<div style="font-size:0.8rem;color:#374151;margin-top:2px;">'
        f'{c:.2f} (threshold 0.85)</div>'
    )


# ---------------------------------------------------------------------------
# Sidebar — session picker + start form
# ---------------------------------------------------------------------------

st.title("🔁 Closed-Loop Orchestrator")
st.caption(
    "Drives the iterative refinement loop: results feed back into hypothesis, "
    "which proposes bounded follow-up actions until the run converges or "
    "exhausts its budget."
)

with st.sidebar:
    st.header("Session")
    sess_resp = api.list_sessions()
    sessions = sess_resp.get("sessions") or []
    if not sessions:
        st.warning("No chat sessions found. Create one in the main ChatDFT app first.")
        st.stop()

    session_labels = [f"#{s['id']} — {s.get('name', '?')}" for s in sessions]
    sess_idx = st.selectbox(
        "Pick a session",
        list(range(len(sessions))),
        format_func=lambda i: session_labels[i],
        key="orch_session_idx",
    )
    session_id = int(sessions[sess_idx]["id"])

    st.divider()
    st.header("Start new run")
    auto_submit = st.checkbox(
        "auto_submit", value=False,
        help="Off (default) = prepare jobs only. On = actually submit to HPC.",
    )
    max_iter = st.slider("max_iterations", 1, 20, 10)
    conf_thr = st.slider("confidence_threshold", 0.5, 0.99, 0.85, 0.01)
    no_new_thr = st.slider("no_new_actions_threshold", 1, 5, 2)
    cluster = st.text_input("cluster", value="hoffman2")
    engine = st.text_input("engine", value="vasp")

    if st.button("▶ Start run", type="primary", use_container_width=True):
        resp = api.orchestrator_start(
            session_id,
            max_iterations=max_iter,
            confidence_threshold=conf_thr,
            no_new_actions_threshold=no_new_thr,
            auto_submit=auto_submit,
            cluster=cluster,
            engine=engine,
        )
        if not resp.get("ok"):
            st.error(resp.get("detail") or resp.get("error") or "Failed to start.")
        else:
            st.success(f"Started run #{resp['run_id']}")
            st.session_state["orch_active_run"] = resp["run_id"]
            st.rerun()


# ---------------------------------------------------------------------------
# Run picker — runs for this session
# ---------------------------------------------------------------------------

runs_resp = api.orchestrator_runs(session_id, limit=30)
runs: List[Dict[str, Any]] = runs_resp.get("runs") or []

if not runs:
    st.info(
        "No orchestrator runs for this session yet. "
        "Use the sidebar to start one — the session needs an intent + hypothesis "
        "+ plan (run them via the main ChatDFT app first)."
    )
    st.stop()

active_run_id = st.session_state.get("orch_active_run") or runs[0]["id"]

# Build a compact table for quick selection
runs_df = pd.DataFrame([
    {
        "run_id":    r["id"],
        "status":    r["status"],
        "stop":      r.get("stop_reason") or "",
        "iter":      r["iteration"],
        "conf":      f"{r['confidence']:.2f}",
        "reward_ema": f"{r['reward_ema']:+.2f}",
        "started":   r.get("started_at") or "",
        "ended":     r.get("ended_at") or "",
    }
    for r in runs
])

st.subheader("Runs for this session")
st.dataframe(runs_df, use_container_width=True, hide_index=True)

run_id = st.selectbox(
    "Inspect run",
    [r["id"] for r in runs],
    index=[r["id"] for r in runs].index(active_run_id) if active_run_id in [r["id"] for r in runs] else 0,
    key="orch_inspect_run",
)


# ---------------------------------------------------------------------------
# Run detail — header / live state / iteration trace
# ---------------------------------------------------------------------------

auto_refresh = st.checkbox("Auto-refresh every 5 s (while live)", value=True)

status_resp = api.orchestrator_status(int(run_id))
if not status_resp.get("ok"):
    st.error(f"Failed to load run: {status_resp.get('detail')}")
    st.stop()

state = status_resp.get("state") or {}
trace = status_resp.get("trace_tail") or []
is_live = bool(status_resp.get("live"))

# ── Header row ──────────────────────────────────────────────────────────────
hdr1, hdr2, hdr3 = st.columns([3, 2, 2])
with hdr1:
    st.markdown(f"### Run #{run_id} {_stop_pill(state.get('stop_reason'))}", unsafe_allow_html=True)
    st.markdown(
        " ".join([
            _pill(f"iter {state.get('iteration', 0)} / {state.get('max_iterations', '?')}"),
            _pill(f"auto_submit={state.get('auto_submit', '?')}"),
            _pill(f"engine={state.get('engine', '?')}"),
            _pill(f"cluster={state.get('cluster', '?')}"),
        ]),
        unsafe_allow_html=True,
    )
with hdr2:
    st.markdown("**Confidence**", help="mean reward of last 5 signals, mapped to [0,1]")
    st.markdown(_confidence_bar(float(state.get("confidence", 0.5))), unsafe_allow_html=True)
with hdr3:
    if is_live:
        if st.button("⏹ Stop run", type="secondary", use_container_width=True):
            r = api.orchestrator_stop(int(run_id))
            if r.get("ok"):
                st.success("Stop requested — will pick up at next iteration boundary.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(r.get("detail") or "Failed to stop.")
    else:
        st.metric("Status", "finished" if not is_live else "live",
                  state.get("stop_reason") or "")


# ── Counters row ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Plan tasks",      state.get("n_plan_tasks", 0))
m2.metric("Pending",         state.get("n_pending_tasks", 0))
m3.metric("Completed",       state.get("n_completed_results", 0))
m4.metric("Reward EMA",      f"{state.get('reward_ema', 0.0):+.2f}")
m5.metric("No-new streak",   f"{state.get('no_new_actions_streak', 0)} / {state.get('no_new_actions_threshold', 2)}")


# ── Per-agent budgets ───────────────────────────────────────────────────────
st.subheader("Per-agent budgets")
budgets = state.get("budgets") or {}
if budgets:
    bdf = pd.DataFrame([
        {
            "agent": name,
            "rounds_used": b.get("rounds_used", 0),
            "max_rounds":  b.get("max_rounds")  if b.get("max_rounds") is not None else "—",
            "no_new_streak": b.get("no_new_streak", 0),
            "reward_gate": "✓" if b.get("use_reward_gate") else "—",
            "exhausted":  "✋" if b.get("exhausted") else "open",
        }
        for name, b in budgets.items()
    ])
    st.dataframe(bdf, use_container_width=True, hide_index=True)
else:
    st.caption("(budget telemetry unavailable)")


# ── Iteration trace ─────────────────────────────────────────────────────────
st.subheader("Iteration trace (most recent 10)")

if not trace:
    st.caption("No iterations yet.")
else:
    for entry in trace:
        with st.expander(
            f"Iter {entry['iteration']}  "
            f"· task#{entry.get('executed_task_id', '—')} "
            f"· {entry.get('executed_agent', '—')} "
            f"· {'✅' if entry.get('success') else '❌' if entry.get('success') is False else '⏳'} "
            f"· reward={entry.get('reward')}"
            f"· conf→{entry.get('confidence_after')}"
        ):
            if entry.get("notes"):
                st.caption(f"📝 {entry['notes']}")

            cols = st.columns(2)

            # Accepted actions
            with cols[0]:
                st.markdown("**Proposed actions (accepted)**")
                accepted = entry.get("proposed_actions") or []
                if not accepted:
                    st.caption("(none)")
                for a in accepted:
                    st.markdown(
                        f"{_kind_pill(a.get('kind', '?'))} "
                        f"<code>{a.get('subkind', '?')}</code> · "
                        f"<b>{a.get('target', '?')}</b> · "
                        f"priority={a.get('priority', '?'):.2f} · "
                        f"cost={a.get('cost_estimate', '?')}",
                        unsafe_allow_html=True,
                    )
                    if a.get("rationale"):
                        st.caption(f"💭 {a['rationale']}")
                    if a.get("params"):
                        st.json(a["params"], expanded=False)

            # Rejected actions
            with cols[1]:
                st.markdown("**Rejected actions**")
                rejected = entry.get("rejected_actions") or []
                if not rejected:
                    st.caption("(none)")
                for r in rejected:
                    raw = r.get("raw") or {}
                    if isinstance(raw, list):
                        # batch error: raw is the whole input list
                        st.caption(f"❌ batch error: {r.get('errors')}")
                        continue
                    st.markdown(
                        f"❌ {_kind_pill(raw.get('kind', '?'))} "
                        f"<code>{raw.get('subkind', '?')}</code> · "
                        f"<b>{raw.get('target', '?')}</b>",
                        unsafe_allow_html=True,
                    )
                    for err in r.get("errors", []):
                        st.caption(f"  · {err}")


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh and is_live:
    time.sleep(5)
    st.rerun()
