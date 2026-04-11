"""
Unit tests for the monitoring dashboard aggregator.

Focus: the pure-math helpers and metric derivation from in-memory fixtures.
These tests do NOT hit the database, so they run in <1 s on CI.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.mlops.dashboard import (  # noqa: E402
    AGENT_REGISTRY,
    ALERTS,
    AgentLogRow,
    WorkflowTaskRow,
    _agent_metrics,
    _cost_usd,
    _system_metrics,
    _workflow_matches_agent,
    bucketise,
    derive_alerts,
    linear_slope_per_hour,
    percentile,
)

# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_returns_zero(self):
        assert percentile([], 99) == 0.0

    def test_single_value(self):
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 99) == 42.0

    def test_known_p50_p99(self):
        vals = list(range(1, 101))  # 1..100
        assert percentile(vals, 50) == pytest.approx(50.5, rel=0.02)
        assert percentile(vals, 99) == pytest.approx(99.01, rel=0.02)


# ---------------------------------------------------------------------------
# linear_slope_per_hour
# ---------------------------------------------------------------------------


class TestLinearSlopePerHour:
    def test_flat_series_returns_zero(self):
        xs = [0.0, 1800.0, 3600.0, 5400.0]
        ys = [0.1, 0.1, 0.1, 0.1]
        assert linear_slope_per_hour(xs, ys) == pytest.approx(0.0, abs=1e-9)

    def test_linear_growth_returns_correct_slope(self):
        # rate grows from 0 to 0.4 over 1 hour, linearly → slope = 0.4/h
        xs = [0.0, 900.0, 1800.0, 2700.0, 3600.0]
        ys = [0.0, 0.1, 0.2, 0.3, 0.4]
        slope = linear_slope_per_hour(xs, ys)
        assert slope == pytest.approx(0.4, rel=0.01)

    def test_single_point_returns_zero(self):
        assert linear_slope_per_hour([100.0], [0.5]) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert linear_slope_per_hour([1.0, 2.0], [0.1]) == 0.0

    def test_empty_buckets_excluded_from_fit(self):
        # Bucket 1 has traffic w/ error_rate=0.2, bucket 2 is empty (count=0),
        # bucket 3 has traffic w/ error_rate=0.4. Without exclusion, the empty
        # middle bucket (rate=0) would pull the slope down. With exclusion we
        # fit only the two non-empty points → slope = (0.4-0.2) / (2/3600 h)·3600
        xs = [0.0, 1800.0, 3600.0]
        ys = [0.2, 0.0, 0.4]
        counts = [10, 0, 10]
        slope = linear_slope_per_hour(xs, ys, counts)
        # (0.4 - 0.2) over 1 hour = +0.2/h
        assert slope == pytest.approx(0.2, rel=0.05)

    def test_all_empty_buckets_returns_zero(self):
        assert linear_slope_per_hour([0.0, 100.0], [0.0, 0.0], [0, 0]) == 0.0


# ---------------------------------------------------------------------------
# bucketise
# ---------------------------------------------------------------------------


class TestBucketise:
    def test_buckets_into_correct_counts(self):
        # Half-open intervals: [0,2), [2,4), [4,6), [6,8), [8,10)
        ts = [0.0, 1.0, 2.0, 5.0, 9.9]
        vs = [1.0, 1.0, 1.0, 1.0, 1.0]
        centres, means, counts = bucketise(ts, vs, 5, 0.0, 10.0)
        assert len(centres) == 5
        assert counts == [2, 1, 1, 0, 1]
        # each non-empty bucket has mean 1.0
        assert means == [1.0, 1.0, 1.0, 0.0, 1.0]

    def test_mean_rates_per_bucket(self):
        ts = [0.0, 0.5, 2.0, 3.0]
        vs = [1.0, 0.0, 1.0, 1.0]
        _, means, counts = bucketise(ts, vs, 2, 0.0, 4.0)
        assert counts == [2, 2]
        assert means == [0.5, 1.0]


# ---------------------------------------------------------------------------
# _cost_usd
# ---------------------------------------------------------------------------


class TestWorkflowMatchesAgent:
    def test_dotted_prefix_matches(self):
        assert _workflow_matches_agent("structure.relax_slab", "structure_agent")
        assert _workflow_matches_agent("post.energy", "post_analysis_agent")
        assert _workflow_matches_agent("adsorption.scan", "structure_agent")

    def test_flat_agent_matches(self):
        # intent_agent._expand_workflow_tasks emits flat names like "adsorption"
        assert _workflow_matches_agent("adsorption", "structure_agent")
        assert _workflow_matches_agent("neb", "structure_agent")

    def test_wrong_agent_does_not_match(self):
        assert not _workflow_matches_agent("structure.relax_slab", "parameters_agent")
        assert not _workflow_matches_agent("post.energy", "hpc_agent")

    def test_empty_returns_false(self):
        assert not _workflow_matches_agent("", "structure_agent")
        assert not _workflow_matches_agent("foo.bar", "nonexistent_agent")


class TestCostUsd:
    def test_gpt_4o_cost(self):
        # 1000 input + 1000 output on gpt-4o → 0.005 + 0.015 = $0.020
        assert _cost_usd("gpt-4o", 1000, 1000) == pytest.approx(0.020, rel=1e-6)

    def test_unknown_model_zero(self):
        assert _cost_usd("made-up-model", 10_000, 10_000) == 0.0

    def test_gpt_4o_mini_cheap(self):
        assert _cost_usd("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075, rel=1e-6)


# ---------------------------------------------------------------------------
# _system_metrics — aggregate workflow tasks
# ---------------------------------------------------------------------------


def _wf(id_: int, status: str, run_time: float = 1.0, created_offset: float = -60.0) -> WorkflowTaskRow:
    now = time.time()
    return WorkflowTaskRow(
        id=id_,
        session_id=1,
        agent="structure.relax_slab",
        task_type="slab_build",
        status=status,
        run_time=run_time,
        error_msg=None if status != "failed" else "boom",
        created_at=now + created_offset,
        updated_at=now + created_offset + 10,
    )


def _log(
    agent: str,
    success: bool,
    latency_ms: int = 1000,
    call_type: str = "llm_json",
    model: str = "gpt-4o-mini",
    session_id: int = 1,
    offset: float = -60.0,
) -> AgentLogRow:
    return AgentLogRow(
        agent_name=agent,
        call_type=call_type,
        model=model,
        input_tokens=500,
        output_tokens=200,
        latency_ms=latency_ms,
        success=success,
        error_msg=None if success else "oops",
        session_id=session_id,
        created_at=time.time() + offset,
    )


class TestSystemMetrics:
    def test_empty(self):
        sm = _system_metrics([], [], window_s=3600, n_buckets=6)
        assert sm.total_workflows == 0
        assert sm.finish_rate == 0.0
        assert sm.success_rate == 0.0
        assert sm.p99_latency_ms == 0.0

    def test_mixed_workflow_states(self):
        wf = [
            _wf(1, "done", run_time=0.5),
            _wf(2, "done", run_time=1.5),
            _wf(3, "failed", run_time=0.8),
            _wf(4, "running", run_time=0.0),
            _wf(5, "queued", run_time=0.0),
        ]
        sm = _system_metrics(wf, [], window_s=3600, n_buckets=6)
        assert sm.total_workflows == 5
        assert sm.finished_workflows == 3  # 2 done + 1 failed
        assert sm.successful_workflows == 2
        assert sm.failed_workflows == 1
        assert sm.active_workflows == 2
        assert sm.finish_rate == pytest.approx(3 / 5)
        assert sm.success_rate == pytest.approx(2 / 3)
        assert sm.error_rate == pytest.approx(1 / 3)
        # latency is in ms, only 'done' tasks count
        assert sm.p99_latency_ms > 0

    def test_latency_percentiles_use_done_tasks_only(self):
        wf = [
            _wf(1, "done", run_time=1.0),
            _wf(2, "done", run_time=5.0),
            _wf(3, "failed", run_time=99.0),  # ignored
        ]
        sm = _system_metrics(wf, [], window_s=3600, n_buckets=6)
        # run_time*1000 → 1000, 5000 (2 samples); p99 ≈ 4960
        assert 1000 <= sm.p99_latency_ms <= 5000
        assert sm.p50_latency_ms == pytest.approx(3000.0, rel=0.1)


# ---------------------------------------------------------------------------
# _agent_metrics
# ---------------------------------------------------------------------------


class TestAgentMetrics:
    def test_llm_agent_from_agent_log(self):
        logs = [
            _log("intent_agent", True, latency_ms=500),
            _log("intent_agent", True, latency_ms=600),
            _log("intent_agent", False, latency_ms=300, call_type="llm_json_invalid"),
            _log("hypothesis_agent", True, latency_ms=800),
        ]
        am = _agent_metrics(
            "intent_agent",
            AGENT_REGISTRY["intent_agent"],
            logs,
            workflows=[],
            historical_counts={},
            window_s=3600,
            n_buckets=6,
        )
        assert am.request_count == 3
        assert am.success_rate == pytest.approx(2 / 3)
        assert am.error_rate == pytest.approx(1 / 3)
        assert am.schema_valid_rate == pytest.approx(2 / 3)
        assert am.retry_rate == pytest.approx(1 / 3)
        assert am.total_tokens > 0
        # handoff: there is a hypothesis_agent log in same session
        assert am.handoff_success_rate == pytest.approx(1.0)

    def test_execution_agent_from_workflow_tasks(self):
        wfs = [
            _wf(1, "done", run_time=2.0),
            _wf(2, "done", run_time=3.0),
            _wf(3, "failed", run_time=1.0),
        ]
        am = _agent_metrics(
            "structure_agent",
            AGENT_REGISTRY["structure_agent"],
            logs=[],
            workflows=wfs,
            historical_counts={},
            window_s=3600,
            n_buckets=6,
        )
        assert am.request_count == 3
        assert am.success_rate == pytest.approx(2 / 3)
        # execution agents have no JSON schema
        assert am.schema_valid_rate is None
        # p99 derived from run_time_ms
        assert am.p99_latency_ms > 0

    def test_empty_registers_note(self):
        am = _agent_metrics(
            "intent_agent",
            AGENT_REGISTRY["intent_agent"],
            logs=[],
            workflows=[],
            historical_counts={},
            window_s=3600,
            n_buckets=6,
        )
        assert am.request_count == 0
        assert "no events in window" in am.notes


# ---------------------------------------------------------------------------
# derive_alerts
# ---------------------------------------------------------------------------


class TestDeriveAlerts:
    def test_quiet_system_emits_no_alerts(self):
        wf = [_wf(i, "done", run_time=0.5) for i in range(10)]
        sm = _system_metrics(wf, [], window_s=3600, n_buckets=6)
        # all agents empty
        from server.mlops.dashboard import AgentMetrics

        ams = [
            AgentMetrics(
                name="intent_agent",
                role="",
                layer="chat",
                llm_backed=True,
                json_producer=True,
                request_count=0,
                success_rate=1.0,
                error_rate=0.0,
                error_rate_slope_per_hour=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                retry_rate=0.0,
                schema_valid_rate=None,
                timeout_rate=0.0,
                handoff_success_rate=None,
                total_tokens=0,
                total_cost_usd=0.0,
                sample_window_s=3600,
            ),
        ]
        assert derive_alerts(sm, ams) == []

    def test_finish_rate_alert_fires(self):
        wf = [_wf(i, "failed", run_time=1.0) for i in range(6)] + [
            _wf(i, "running", run_time=0.0) for i in range(10, 14)
        ]
        sm = _system_metrics(wf, [], window_s=3600, n_buckets=6)
        alerts = derive_alerts(sm, [])
        kinds = {(a.scope, a.metric) for a in alerts}
        # finish_rate = 6/10 = 0.6 < 0.85 → alert
        assert ("system", "finish_rate") in kinds

    def test_agent_error_rate_alert_fires(self):
        from server.mlops.dashboard import AgentMetrics

        am = AgentMetrics(
            name="intent_agent",
            role="",
            layer="chat",
            llm_backed=True,
            json_producer=True,
            request_count=10,
            success_rate=0.3,
            error_rate=0.7,
            error_rate_slope_per_hour=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            retry_rate=0.0,
            schema_valid_rate=0.95,
            timeout_rate=0.0,
            handoff_success_rate=None,
            total_tokens=0,
            total_cost_usd=0.0,
            sample_window_s=3600,
        )
        sm = _system_metrics([], [], window_s=3600, n_buckets=6)
        alerts = derive_alerts(sm, [am])
        assert any(a.metric == "error_rate" and a.target == "intent_agent" for a in alerts)

    def test_schema_valid_rate_alert_fires(self):
        from server.mlops.dashboard import AgentMetrics

        am = AgentMetrics(
            name="plan_agent",
            role="",
            layer="chat",
            llm_backed=True,
            json_producer=True,
            request_count=20,
            success_rate=1.0,
            error_rate=0.0,
            error_rate_slope_per_hour=0.0,
            p50_latency_ms=100,
            p95_latency_ms=200,
            p99_latency_ms=300,
            retry_rate=0.0,
            schema_valid_rate=0.5,  # well below floor
            timeout_rate=0.0,
            handoff_success_rate=None,
            total_tokens=0,
            total_cost_usd=0.0,
            sample_window_s=3600,
        )
        sm = _system_metrics([], [], window_s=3600, n_buckets=6)
        alerts = derive_alerts(sm, [am])
        assert any(a.metric == "schema_valid_rate" for a in alerts)


# ---------------------------------------------------------------------------
# Smoke test on compute_dashboard — mocks DB
# ---------------------------------------------------------------------------


class TestComputeDashboardSmoke:
    def test_compute_dashboard_handles_empty_db(self, monkeypatch):
        import asyncio

        from server.mlops import dashboard as _dash

        async def _empty():
            return []

        async def _empty_dict():
            return {}

        monkeypatch.setattr(_dash, "_fetch_agent_logs", lambda window_s: _empty())
        monkeypatch.setattr(_dash, "_fetch_workflow_tasks", lambda window_s: _empty())
        monkeypatch.setattr(_dash, "_fetch_chat_messages_by_agent", lambda window_s: _empty_dict())

        data = asyncio.run(_dash.compute_dashboard(window_minutes=60, n_buckets=6))
        assert data["ok"] is True
        assert "system" in data
        assert "agents" in data
        assert "offline_metrics" in data
        assert "alerts" in data
        assert data["thresholds"]["p99_latency_ms"] == ALERTS.p99_latency_ms
        # every canonical agent should be represented
        names = {a["name"] for a in data["agents"]}
        assert "intent_agent" in names
        assert "hypothesis_agent" in names
        assert "plan_agent" in names
