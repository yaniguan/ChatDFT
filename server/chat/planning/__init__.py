"""
Planning module — split from plan_agent.py for maintainability.

Backward-compatible: all public functions re-exported.
"""
from server.chat.plan_agent import (
    _extract_intent,
    _mech_guess,
    _mech_seed,
    plan_route,
)
