# server/orchestrator/__init__.py
"""
ChatDFT closed-loop orchestrator.

The orchestrator is the engine that closes the loop:

    query → intent → hypothesis → plan → execute → results
                          ▲                            │
                          └──── refine + propose ──────┘

Public surface:
    - ChatDFTOrchestrator: the iterating engine (loop.py)
    - ProposedAction:      typed, validated action LLM is allowed to propose
    - OrchestrationState:  per-run mutable state (plan + results + rewards)
    - router:              FastAPI router exposing /api/orchestrator/*

Design contract:
    Every action LLM proposes goes through ``actions.validate_action`` —
    no free-form prompts, no schema escape, no cross-session writes.
"""

from server.orchestrator.actions import (
    SUBKINDS,
    ActionKind,
    ProposedAction,
    validate_action,
)
from server.orchestrator.loop import ChatDFTOrchestrator
from server.orchestrator.state import OrchestrationState

# Note: ``server.orchestrator.routes`` is intentionally not re-exported here.
# It depends on a live database; importers that need the FastAPI router
# should ``from server.orchestrator.routes import router`` directly so the
# pure logic above (validators, loop, state) stays import-safe in tests
# that have no DATABASE_URL.

__all__ = [
    "ActionKind",
    "ChatDFTOrchestrator",
    "OrchestrationState",
    "ProposedAction",
    "SUBKINDS",
    "validate_action",
]
