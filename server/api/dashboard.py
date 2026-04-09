# server/api/dashboard.py
# -*- coding: utf-8 -*-
"""
FastAPI routes that surface the monitoring dashboard payload.

Endpoints
---------
GET /dashboard/overview?window_minutes=60&n_buckets=12
    Full dashboard payload (system + agent metrics + offline metrics + alerts).

GET /dashboard/help
    Static documentation of the metric formulas — the dashboard UI renders
    this in a collapsible "Formulas" panel so there is no ambiguity about
    how the numbers are computed.

GET /dashboard/alerts?window_minutes=60
    Lightweight endpoint for alert-only polling (e.g. Slack webhooks,
    external uptime monitors).
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from server.mlops.dashboard import (
    FORMULA_HELP,
    compute_dashboard,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview")
async def dashboard_overview(
    window_minutes: int = Query(60, ge=1, le=24 * 60, description="Rolling window in minutes"),
    n_buckets: int = Query(12, ge=2, le=60, description="Number of trend/sparkline buckets"),
):
    """Full dashboard payload over a rolling window."""
    try:
        data = await compute_dashboard(window_minutes=window_minutes, n_buckets=n_buckets)
        return JSONResponse(data)
    except Exception as e:  # pragma: no cover — defensive
        return JSONResponse(
            {"ok": False, "error": str(e), "error_type": type(e).__name__},
            status_code=500,
        )


@router.get("/help")
async def dashboard_help():
    """Static metric formulas and definitions."""
    return JSONResponse({"ok": True, "formulas": FORMULA_HELP})


@router.get("/alerts")
async def dashboard_alerts(
    window_minutes: int = Query(60, ge=1, le=24 * 60),
):
    """Return only the alerts list — for lightweight external polling."""
    try:
        data = await compute_dashboard(window_minutes=window_minutes, n_buckets=6)
        return JSONResponse({
            "ok": True,
            "generated_at": data["generated_at"],
            "alerts": data["alerts"],
            "thresholds": data["thresholds"],
        })
    except Exception as e:  # pragma: no cover
        return JSONResponse(
            {"ok": False, "error": str(e), "error_type": type(e).__name__},
            status_code=500,
        )
