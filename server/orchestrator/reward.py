# server/orchestrator/reward.py
"""
Map a freshly executed task's result + the live hypothesis into a reward
signal in [-1, 1].

Wraps :class:`server.execution.agent_coordinator.RewardTracker` so we
reuse the existing reward math (so the EMA + domain bookkeeping stays in
one place).  Adds two responsibilities the tracker doesn't have:

  1. Extract the *predicted* trend / range from the hypothesis graph,
     given a (species, surface) target.
  2. Decide what to do when the result is non-numeric (e.g. only converged
     successfully): emit ``reward=0`` (inconclusive, not negative).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from server.execution.agent_coordinator import RewardTracker
from server.orchestrator.state import OrchestrationState

log = logging.getLogger(__name__)


def _predicted_trend_for(
    hypothesis_graph: Dict[str, Any], species: str, surface: str,
) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    """
    Pull a (predicted_trend, predicted_range) tuple out of the hypothesis
    graph for the given (species, surface) pair.

    The hypothesis_agent stores predictions in graph['predictions'] as a
    list of {species, surface, trend, range_lo, range_hi}.  If absent we
    fall back to None (caller treats reward as inconclusive → 0).
    """
    preds = hypothesis_graph.get("predictions") or []
    species_norm = species.replace("*", "").strip().lower()
    surface_norm = surface.split("(")[0].strip().lower()

    for pred in preds:
        if not isinstance(pred, dict):
            continue
        s = str(pred.get("species", "")).replace("*", "").strip().lower()
        f = str(pred.get("surface", "")).split("(")[0].strip().lower()
        if s == species_norm and (not f or f == surface_norm):
            trend = pred.get("trend")
            lo = pred.get("range_lo")
            hi = pred.get("range_hi")
            rng = (float(lo), float(hi)) if lo is not None and hi is not None else None
            return trend, rng

    return None, None


def compute_reward_for_result(
    *,
    state: OrchestrationState,
    tracker: RewardTracker,
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Compute and record a reward signal for one completed result.

    Returns a serialized signal dict (suitable for ``state.reward_history``)
    or ``None`` if the result lacks the data needed to score it.
    """
    species = str(result.get("species") or "")
    surface = str(result.get("surface") or "")
    value = result.get("value")
    converged = bool(result.get("converged", True))

    if value is None or not species:
        # No numeric prediction-vs-actual possible → inconclusive
        signal_dict = {
            "species": species,
            "surface": surface,
            "reward": 0.0,
            "reaction_type": result.get("reaction_type", "unknown"),
            "details": "inconclusive: value or species missing",
            "converged": converged,
        }
        state.reward_history.append(signal_dict)
        return signal_dict

    if not converged:
        # Failed convergence is a soft negative — distinguish from "wrong"
        signal_dict = {
            "species": species,
            "surface": surface,
            "reward": -0.3,
            "reaction_type": result.get("reaction_type", "unknown"),
            "details": "non-converged",
            "converged": False,
        }
        state.reward_history.append(signal_dict)
        return signal_dict

    trend, rng = _predicted_trend_for(state.hypothesis_graph, species, surface)
    if trend is None:
        # Hypothesis didn't make a prediction here — neutral signal
        signal_dict = {
            "species": species,
            "surface": surface,
            "reward": 0.0,
            "reaction_type": result.get("reaction_type", "unknown"),
            "details": "no prediction in hypothesis to compare against",
            "converged": True,
        }
        state.reward_history.append(signal_dict)
        return signal_dict

    catalyst = surface.split("(")[0] if surface else "unknown"
    rxn_type = result.get("reaction_type") or state.intent.get("problem_type", "unknown") or "unknown"

    signal = tracker.compute_reward(
        predicted_trend=str(trend),
        predicted_range=rng,
        dft_value=float(value),
        reaction_type=str(rxn_type),
        catalyst_class=catalyst,
        species=species,
        surface=surface,
        session_id=state.session_id,
    )
    tracker.record(signal)

    signal_dict = {
        "species": signal.species,
        "surface": signal.surface,
        "reward": signal.reward,
        "reaction_type": signal.reaction_type,
        "predicted_trend": signal.predicted_trend,
        "dft_value": signal.dft_value,
        "details": signal.details,
        "converged": True,
    }
    state.reward_history.append(signal_dict)
    return signal_dict
