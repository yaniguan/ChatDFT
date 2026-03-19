# server/mechanisms/registry.py
# -*- coding: utf-8 -*-
"""
DEPRECATED static registry — replaced by the dynamic MechanismBuilder.

This module now re-exports REACTION_TYPE_TEMPLATES from builder.py so that
any existing code that does `from server.mechanisms.registry import REGISTRY`
continues to work without crashing (REGISTRY is now an alias for the templates).

New code should use:
    from server.mechanisms.builder import build_mechanism, check_missing_info
"""

from server.mechanisms.builder import REACTION_TYPE_TEMPLATES  # noqa: F401

# Backward-compat alias used by intent_agent / plan_agent
REGISTRY = REACTION_TYPE_TEMPLATES
