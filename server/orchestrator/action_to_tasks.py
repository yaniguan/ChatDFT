# server/orchestrator/action_to_tasks.py
"""
Convert a validated :class:`ProposedAction` into one or more WorkflowTask-shaped
dicts that ``_pipeline()`` (server/execution/agent_routes.py) can execute.

The mapping is intentionally narrow.  Each subkind has exactly one shape; if
you need new behavior, add a subkind to ``actions.SUBKINDS`` and a branch
here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from server.orchestrator.actions import ActionKind, ProposedAction

log = logging.getLogger(__name__)


def _next_task_id(existing_tasks: List[Dict[str, Any]]) -> int:
    if not existing_tasks:
        return 1
    return max(int(t.get("id", 0)) for t in existing_tasks) + 1


def _meta(state_session_id: int, source_action: ProposedAction) -> Dict[str, Any]:
    return {
        "project": f"session-{state_session_id}",
        "run_id": state_session_id,
        "source": "orchestrator",
        "action_kind": source_action.kind,
        "action_subkind": source_action.subkind,
        "action_target": source_action.target,
        "action_rationale": source_action.rationale,
    }


def _make_task(
    *,
    tid: int,
    section: str,
    name: str,
    agent: str,
    description: str,
    payload: Dict[str, Any],
    state_session_id: int,
    source_action: ProposedAction,
    depends_on: Optional[List[int]] = None,
) -> Dict[str, Any]:
    return {
        "id": tid,
        "section": section,
        "name": name,
        "agent": agent,
        "description": description,
        "params": {"form": [], "payload": payload, "endpoint": ""},
        "depends_on": depends_on or [],
        "status": "pending",
        "meta": _meta(state_session_id, source_action),
    }


# ---------------------------------------------------------------------------
# Per-kind handlers
# ---------------------------------------------------------------------------

def _verify_to_tasks(
    action: ProposedAction, *, existing: List[Dict[str, Any]], session_id: int,
) -> List[Dict[str, Any]]:
    tid0 = _next_task_id(existing)
    p = action.params

    if action.subkind == "reconverge":
        # Tighten ENCUT/EDIFF/k-mesh on an existing task's structure
        delta_encut = float(p.get("delta_encut", 50))
        kmesh_factor = float(p.get("kmesh_factor", 1.5))
        return [_make_task(
            tid=tid0,
            section="verify",
            name=f"Reconverge task#{p.get('target_task_id')}",
            agent="run_dft",
            description=f"Reconverge {action.target}: ENCUT+{delta_encut}, k-mesh×{kmesh_factor}",
            payload={
                "target_task_id": int(p["target_task_id"]),
                "incar_overrides": {
                    "ENCUT": f"+{delta_encut}",
                    "EDIFF": float(p.get("edens_factor", 0.1)) * 1e-5,
                },
                "kpoints_factor": kmesh_factor,
            },
            state_session_id=session_id,
            source_action=action,
        )]

    if action.subkind == "replicate":
        return [_make_task(
            tid=tid0,
            section="verify",
            name=f"Replicate task#{p.get('target_task_id')}",
            agent="run_dft",
            description=f"Replicate {action.target} (sanity check, optional jitter)",
            payload={
                "target_task_id": int(p["target_task_id"]),
                "seed_jitter": float(p.get("seed_jitter", 0.0)),
            },
            state_session_id=session_id,
            source_action=action,
        )]

    if action.subkind == "functional":
        return [_make_task(
            tid=tid0,
            section="verify",
            name=f"Functional sensitivity ({p.get('new_xc')})",
            agent="run_dft",
            description=(
                f"Re-run {action.target} with XC={p.get('new_xc')}"
                + (f", +U={p.get('plus_u')}" if p.get("plus_u") else "")
                + (f", vdW={p.get('vdw')}" if p.get("vdw") else "")
            ),
            payload={
                "target_task_id": int(p["target_task_id"]),
                "incar_overrides": {
                    "GGA": p.get("new_xc"),
                    **({"LDAU": True, "LDAUTYPE": 2, "LDAUU": p["plus_u"]}
                       if p.get("plus_u") else {}),
                    **({"IVDW": p["vdw"]} if p.get("vdw") else {}),
                },
            },
            state_session_id=session_id,
            source_action=action,
        )]

    return []


def _extend_to_tasks(
    action: ProposedAction, *, existing: List[Dict[str, Any]], session_id: int,
) -> List[Dict[str, Any]]:
    tid0 = _next_task_id(existing)
    p = action.params

    if action.subkind == "intermediate":
        return [_make_task(
            tid=tid0,
            section="extend",
            name=f"Adsorb {p['species']}",
            agent="structure.relax_adsorbate",
            description=f"New intermediate {p['species']} on {p.get('surface') or action.target}",
            payload={
                "adsorbate": p["species"],
                "surface": p.get("surface") or action.target,
                "site": p.get("site", "auto"),
            },
            state_session_id=session_id,
            source_action=action,
        )]

    if action.subkind == "site":
        return [_make_task(
            tid=tid0,
            section="extend",
            name=f"{p['species']} @ {p['site']}",
            agent="structure.relax_adsorbate",
            description=f"{p['species']} on {p['surface']} at site {p['site']}",
            payload={
                "adsorbate": p["species"],
                "surface": p["surface"],
                "site": p["site"],
            },
            state_session_id=session_id,
            source_action=action,
        )]

    if action.subkind == "facet":
        # one task per facet — but cost_estimate already bounds list length
        out = []
        for i, facet_str in enumerate([p["facet"]]):  # facet is single-string here
            out.append(_make_task(
                tid=tid0 + i,
                section="extend",
                name=f"{p['species']} @ {facet_str}",
                agent="structure.relax_adsorbate",
                description=f"{p['species']} on {p.get('surface', '')}({facet_str})",
                payload={
                    "adsorbate": p["species"],
                    "surface": p.get("surface", ""),
                    "facet": facet_str,
                },
                state_session_id=session_id,
                source_action=action,
            ))
        return out

    if action.subkind == "coadsorbate":
        return [_make_task(
            tid=tid0,
            section="extend",
            name=f"co-ads {p['species_a']}+{p['species_b']}",
            agent="adsorption.co",
            description=f"Co-adsorption of {p['species_a']} + {p['species_b']} on {p['surface']}",
            payload={
                "species_a": p["species_a"],
                "species_b": p["species_b"],
                "surface": p["surface"],
                "site_a": p.get("site_a", "auto"),
                "site_b": p.get("site_b", "auto"),
            },
            state_session_id=session_id,
            source_action=action,
        )]

    return []


def _challenge_to_tasks(
    action: ProposedAction, *, existing: List[Dict[str, Any]], session_id: int,
) -> List[Dict[str, Any]]:
    tid0 = _next_task_id(existing)
    p = action.params

    if action.subkind == "alternative_step":
        return [_make_task(
            tid=tid0,
            section="challenge",
            name="Alternative reaction step",
            agent="neb.run",
            description=(
                f"Compute alternative path: '{p['original_step']}' "
                f"vs '{p['alternative_step']}' on {p.get('surface', '')}"
            ),
            payload={
                "original_step": p["original_step"],
                "alternative_step": p["alternative_step"],
                "surface": p.get("surface", ""),
            },
            state_session_id=session_id,
            source_action=action,
        )]

    if action.subkind == "alternative_ts":
        return [_make_task(
            tid=tid0,
            section="challenge",
            name=f"Alt-TS for {p['step']}",
            agent="neb.run",
            description=f"Alternative TS variant '{p['ts_variant']}' for step {p['step']}",
            payload={
                "step": p["step"],
                "ts_variant": p["ts_variant"],
                "surface": p.get("surface", ""),
            },
            state_session_id=session_id,
            source_action=action,
        )]

    if action.subkind == "counterexample":
        return [_make_task(
            tid=tid0,
            section="challenge",
            name=f"Counterexample {p['species']} / {p['surface']}",
            agent="structure.relax_adsorbate",
            description=(
                f"Counterexample: predict {p.get('expected_trend','?')} "
                f"({p.get('expected_value','?')} eV)"
            ),
            payload={
                "adsorbate": p["species"],
                "surface": p["surface"],
                "expected_value": p.get("expected_value"),
                "expected_trend": p.get("expected_trend"),
            },
            state_session_id=session_id,
            source_action=action,
        )]

    return []


def _scan_to_tasks(
    action: ProposedAction, *, existing: List[Dict[str, Any]], session_id: int,
) -> List[Dict[str, Any]]:
    tid0 = _next_task_id(existing)
    p = action.params

    if action.subkind == "coverage":
        out = []
        for i, cov in enumerate(p["values"]):
            out.append(_make_task(
                tid=tid0 + i,
                section="scan",
                name=f"coverage {cov}",
                agent="adsorption.scan",
                description=f"{p['species']} on {p['surface']} @ θ={cov}",
                payload={
                    "adsorbate": p["species"],
                    "surface": p["surface"],
                    "coverage": float(cov),
                },
                state_session_id=session_id,
                source_action=action,
            ))
        return out

    if action.subkind == "facet":
        out = []
        for i, facet in enumerate(p["facets"]):
            out.append(_make_task(
                tid=tid0 + i,
                section="scan",
                name=f"facet {facet}",
                agent="structure.relax_adsorbate",
                description=f"{p['species']} on {p.get('surface_template','')}({facet})",
                payload={
                    "adsorbate": p["species"],
                    "facet": facet,
                    "surface_template": p.get("surface_template", ""),
                },
                state_session_id=session_id,
                source_action=action,
            ))
        return out

    if action.subkind == "composition":
        out = []
        for i, comp in enumerate(p["compositions"]):
            out.append(_make_task(
                tid=tid0 + i,
                section="scan",
                name=f"composition {comp}",
                agent="structure.relax_adsorbate",
                description=f"Composition sweep on {p['base_surface']}: {comp}",
                payload={
                    "base_surface": p["base_surface"],
                    "composition": comp,
                    "adsorbate": p.get("species"),
                },
                state_session_id=session_id,
                source_action=action,
            ))
        return out

    if action.subkind == "field":
        out = []
        for i, fval in enumerate(p["fields"]):
            out.append(_make_task(
                tid=tid0 + i,
                section="scan",
                name=f"field {fval}{p.get('units','V/Å')}",
                agent="run_dft",
                description=f"{p['species']} on {p['surface']} under field {fval}",
                payload={
                    "adsorbate": p["species"],
                    "surface": p["surface"],
                    "field": float(fval),
                    "field_units": p.get("units", "V/Å"),
                    "incar_overrides": {"EFIELD": float(fval)},
                },
                state_session_id=session_id,
                source_action=action,
            ))
        return out

    return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_HANDLERS = {
    ActionKind.VERIFY.value: _verify_to_tasks,
    ActionKind.EXTEND.value: _extend_to_tasks,
    ActionKind.CHALLENGE.value: _challenge_to_tasks,
    ActionKind.SCAN.value: _scan_to_tasks,
}


def action_to_tasks(
    action: ProposedAction,
    *,
    existing_tasks: List[Dict[str, Any]],
    session_id: int,
) -> List[Dict[str, Any]]:
    """Convert a validated action to ≥0 WorkflowTask-shaped dicts."""
    handler = _HANDLERS.get(action.kind)
    if handler is None:
        log.warning("action_to_tasks: unknown kind %s", action.kind)
        return []
    tasks = handler(action, existing=existing_tasks, session_id=session_id)
    action.spawned_task_ids = [t["id"] for t in tasks]
    return tasks


def primary_agent_for(kind: str) -> str:
    """
    Map an action kind to the primary budgeted agent it stresses.
    Used by stop_conditions to decrement the right counter.
    """
    if kind == ActionKind.VERIFY.value:
        return "parameter"      # verify mostly tweaks INCAR / k-points
    return "structure"          # extend/challenge/scan all build new structures
