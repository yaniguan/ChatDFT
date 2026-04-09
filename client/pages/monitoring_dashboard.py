"""
ChatDFT Monitoring Dashboard — Streamlit Page
==============================================

Visualises the payload returned by ``GET /dashboard/overview``:

* System summary row (p99 latency, finish rate, success rate, active, cost...)
* Alert banner (warnings + criticals, with current value vs threshold)
* Per-agent cards with trend sparklines (error rate + latency)
* Offline (golden-dataset) metrics per agent, with "not yet instrumented"
  clearly marked
* Formula/help panel showing exactly how every number is computed

Run
---
    streamlit run client/pages/monitoring_dashboard.py
    # or access via the main ChatDFT app sidebar
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from client.utils import api  # noqa: E402

st.set_page_config(
    page_title="Monitoring | ChatDFT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITY_COLOURS = {"warning": "#b45309", "critical": "#b91c1c"}
BG_COLOURS = {"warning": "#fef3c7", "critical": "#fee2e2"}


def _fmt_ms(x: float) -> str:
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x / 1000:.2f} s"
    return f"{x:.0f} ms"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _fmt_delta(x: float) -> str:
    return f"{x:+.3f}/h"


def _fmt_money(x: float) -> str:
    return f"${x:.3f}"


def _healthy(a: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
    """Evaluate whether an agent card should render green."""
    if a["error_rate"] > thresholds["error_rate_ceiling"]:
        return False
    if a["p99_latency_ms"] > thresholds["agent_p99_latency_ms"]:
        return False
    if a.get("schema_valid_rate") is not None and a["schema_valid_rate"] < thresholds["schema_valid_floor"]:
        return False
    if a["error_rate_slope_per_hour"] > thresholds["error_slope_ceiling"]:
        return False
    return True


def _metric_pill(label: str, value: str, good: bool = True) -> str:
    bg = "#dcfce7" if good else "#fee2e2"
    fg = "#15803d" if good else "#b91c1c"
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 10px;"
        f"border-radius:10px;font-size:0.78rem;font-weight:600;"
        f"margin:0 4px 0 0;display:inline-block'>{label}: {value}</span>"
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Window")
window_minutes = st.sidebar.slider(
    "Rolling window (minutes)",
    min_value=5,
    max_value=360,
    value=60,
    step=5,
)
n_buckets = st.sidebar.slider("Trend buckets", 4, 30, 12)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 30 s", value=False)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.rerun()

if auto_refresh:
    st.sidebar.caption("Auto-refresh active — rerun triggered every 30 s")
    # Streamlit's autorefresh is 3rd-party; fallback to manual rerun hint
    import time as _t
    _t.sleep(30)
    st.rerun()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

st.title("📈 ChatDFT Monitoring Dashboard")
st.caption(
    "Production-style view of agent quality + system health. "
    "All metrics are computed in `server/mlops/dashboard.py` from real "
    "AgentLog, WorkflowTask and ChatMessage rows."
)

with st.spinner("Loading dashboard payload..."):
    data = api.dashboard_overview(window_minutes=window_minutes, n_buckets=n_buckets)

if not data.get("ok", True):
    st.error(f"Dashboard endpoint failed: {data.get('detail') or data.get('error') or data}")
    st.stop()

system = data.get("system", {})
agents = data.get("agents", [])
alerts = data.get("alerts", [])
thresholds = data.get("thresholds", {})
offline = data.get("offline_metrics", {})
formulas = data.get("formulas", {})

st.caption(
    f"Generated at **{data.get('generated_at', '—')}** · "
    f"Window: **{window_minutes} min** · Buckets: **{n_buckets}**"
)


# ---------------------------------------------------------------------------
# Alert banner
# ---------------------------------------------------------------------------

if alerts:
    n_crit = sum(1 for a in alerts if a["severity"] == "critical")
    n_warn = sum(1 for a in alerts if a["severity"] == "warning")
    headline = f"🚨 {n_crit} critical · ⚠️ {n_warn} warning"
    st.error(headline) if n_crit else st.warning(headline)

    for a in alerts:
        colour = SEVERITY_COLOURS.get(a["severity"], "#6b7280")
        bg = BG_COLOURS.get(a["severity"], "#f3f4f6")
        st.markdown(
            f"""
            <div style='background:{bg};border-left:4px solid {colour};
                        padding:8px 14px;margin:4px 0;border-radius:4px'>
                <strong>[{a['severity'].upper()}] {a['scope']}:{a['target']}</strong>
                &nbsp;·&nbsp; metric=<code>{a['metric']}</code>
                <br/><span style='color:#475569'>{a['message']}</span>
                <br/><span style='font-size:0.78rem;color:#64748b'>
                    value = {a['value']:.4f} · threshold = {a['threshold']:.4f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No active alerts — all metrics within thresholds.")


# ---------------------------------------------------------------------------
# System summary
# ---------------------------------------------------------------------------

st.header("System summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "p99 latency",
    _fmt_ms(system.get("p99_latency_ms", 0)),
    delta=None if system.get("p95_latency_ms", 0) == 0 else f"p95 {_fmt_ms(system['p95_latency_ms'])}",
)
c2.metric(
    "Finish rate",
    _fmt_pct(system.get("finish_rate", 0)),
    delta=f"floor {_fmt_pct(thresholds.get('finish_rate_floor', 0))}",
)
c3.metric(
    "Success rate",
    _fmt_pct(system.get("success_rate", 0)),
    delta=f"floor {_fmt_pct(thresholds.get('success_rate_floor', 0))}",
)
c4.metric(
    "Error slope (Δ/h)",
    _fmt_delta(system.get("error_rate_slope_per_hour", 0)),
    delta=f"ceiling {_fmt_delta(thresholds.get('error_slope_ceiling', 0))}",
    delta_color="inverse",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Active workflows", system.get("active_workflows", 0))
c6.metric("Recent traffic (5m)", system.get("recent_traffic", 0))
c7.metric("Retry rate", _fmt_pct(system.get("retry_rate", 0)))
c8.metric("LLM spend", _fmt_money(system.get("total_cost_usd", 0)))

# Trend charts
if system.get("error_trend") and system.get("finish_trend"):
    trend_df = pd.DataFrame(
        {
            "Bucket": list(range(1, len(system["error_trend"]) + 1)),
            "Error rate": system["error_trend"],
            "Finish rate": system["finish_trend"],
        }
    ).set_index("Bucket")
    st.line_chart(trend_df, height=220)


with st.expander("Raw system metrics"):
    st.json(system)


# ---------------------------------------------------------------------------
# Agent cards
# ---------------------------------------------------------------------------

st.header("Agents")

cards_per_row = 2
for row_start in range(0, len(agents), cards_per_row):
    cols = st.columns(cards_per_row)
    for idx, a in enumerate(agents[row_start : row_start + cards_per_row]):
        col = cols[idx]
        healthy = _healthy(a, thresholds)
        border = "#16a34a" if healthy else "#b91c1c"
        with col:
            st.markdown(
                f"""
                <div style='border:1px solid {border};border-left:6px solid {border};
                            border-radius:8px;padding:12px 16px;margin-bottom:8px;
                            background:#ffffff'>
                    <div style='font-weight:700;font-size:1.05rem;color:#111827'>
                        {a['name']} <span style='font-size:0.78rem;color:#6b7280'>
                        · {a['layer']}{' · LLM' if a['llm_backed'] else ''}
                        </span>
                    </div>
                    <div style='font-size:0.82rem;color:#475569;margin-bottom:8px'>
                        {a['role'] or '—'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Requests", a["request_count"])
            mc2.metric("Success", _fmt_pct(a["success_rate"]))
            mc3.metric("p99", _fmt_ms(a["p99_latency_ms"]))

            mc4, mc5, mc6 = st.columns(3)
            mc4.metric("p50", _fmt_ms(a["p50_latency_ms"]))
            mc5.metric("Error slope", _fmt_delta(a["error_rate_slope_per_hour"]))
            mc6.metric("Retry rate", _fmt_pct(a["retry_rate"]))

            extras: List[str] = []
            if a.get("schema_valid_rate") is not None:
                extras.append(
                    _metric_pill(
                        "schema-valid",
                        _fmt_pct(a["schema_valid_rate"]),
                        a["schema_valid_rate"] >= thresholds["schema_valid_floor"],
                    )
                )
            if a.get("handoff_success_rate") is not None:
                extras.append(
                    _metric_pill(
                        "handoff",
                        _fmt_pct(a["handoff_success_rate"]),
                        a["handoff_success_rate"] >= 0.8,
                    )
                )
            if a.get("timeout_rate", 0) > 0:
                extras.append(
                    _metric_pill(
                        "timeouts",
                        _fmt_pct(a["timeout_rate"]),
                        a["timeout_rate"] < thresholds["timeout_rate_ceiling"],
                    )
                )
            if a.get("total_tokens", 0) > 0:
                extras.append(_metric_pill("tokens", f"{a['total_tokens']:,}", True))
            if a.get("total_cost_usd", 0) > 0:
                extras.append(_metric_pill("cost", _fmt_money(a["total_cost_usd"]), True))
            if a.get("historical_request_count", 0) > 0 and a["request_count"] == 0:
                extras.append(
                    _metric_pill(
                        "hist",
                        f"{a['historical_request_count']} (ChatMessage)",
                        False,
                    )
                )
            if extras:
                st.markdown(" ".join(extras), unsafe_allow_html=True)

            if a.get("error_trend") or a.get("latency_trend"):
                trend_df = pd.DataFrame(
                    {
                        "Bucket": list(range(1, len(a["error_trend"]) + 1)),
                        "Error rate": a["error_trend"],
                        "Latency ms": a["latency_trend"],
                    }
                ).set_index("Bucket")
                st.line_chart(trend_df, height=150)

            if a.get("notes"):
                st.caption(" · ".join(a["notes"]))

            # Offline / golden-dataset metrics
            off_list = offline.get(a["name"]) or []
            if off_list:
                rows = []
                for m in off_list:
                    val = m["value"]
                    if val is None:
                        shown = "not yet instrumented"
                    elif m["higher_is_better"]:
                        shown = f"{val:.3f}"
                    else:
                        shown = f"{val:.3f} (lower=better)"
                    rows.append({
                        "Metric": m["name"],
                        "Value": shown,
                        "Source": m["source"],
                        "Note": m.get("note", ""),
                    })
                with st.expander(f"Offline eval — {a['name']}"):
                    st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        use_container_width=True,
                    )


# ---------------------------------------------------------------------------
# Help / formulas panel
# ---------------------------------------------------------------------------

st.header("📖 How metrics are computed")
with st.expander("Formulas", expanded=False):
    for key, desc in formulas.items():
        st.markdown(f"**`{key}`** — {desc}")

with st.expander("Alert thresholds"):
    st.json(thresholds)

st.caption(
    "Data sources: `agent_log`, `workflow_task`, `chat_message`, `execution_step`. "
    "Missing values are shown as `—`. Metrics marked "
    "`not yet instrumented` require wiring up the corresponding benchmark in "
    "`science/evaluation/metrics.py` and writing results to "
    "`runs/offline_metrics.json`."
)
