# server/chat/intent_schema.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def intent_system_prompt() -> str:
    return (
        "You are an AI-for-Science intent parser for computational catalysis. "
        "Return a STRICT JSON object (no code fences, no prose) with EXACT keys:\n"
        "{stage, area, task, substrate, facet, adsorbates, conditions, metrics, "
        "reaction_network, deliverables, hypothesis, tags, constraints, summary}.\n"
        "Schema:\n"
        "- stage:str  (e.g., 'catalysis')\n"
        "- area:str   (e.g., 'electro')\n"
        "- task:str   (short)\n"
        "- substrate:str | null  (e.g., 'Cu(111)')\n"
        "- facet:str | null      (e.g., 'Cu(111)')\n"
        "- adsorbates:[str]\n"
        "- conditions:{pH:number|null, potential_V_vs_RHE:number|null, solvent:str|null, temperature:number|null, electrolyte:str|null}\n"
        "- metrics:[{name:str, unit?:str, note?:str}]\n"
        "- reaction_network:{intermediates:[str], steps:[{reactants:[str], products:[str], kind?:str}], coads_pairs:[str]}\n"
        "- deliverables:{target_products:[str], figures?:[str]}\n"
        "- hypothesis:str\n"
        "- tags:[str]\n"
        "- constraints:{notes?:str}\n"
        "- summary:str\n"
        "Important: Use English only in all fields. If the input is non-English, translate silently and output English.\n"
        "All lists must be present (use [] if unknown). Fill best-effort defaults from context."
    )


def intent_to_v2(intent: dict, user_text: str) -> dict:
    it = intent or {}
    stage = (it.get("stage") or "catalysis").strip()
    area = (it.get("area") or "electro").strip()
    task = (it.get("task") or "").strip()
    substrate = (it.get("facet") or it.get("substrate") or "").strip()
    ads_list = it.get("adsorbates") or []
    rn = it.get("reaction_network") or {}
    inters = rn.get("intermediates") or []

    def _norm_ads(x):
        if not isinstance(x, str):
            return None
        s = x.strip()
        return s[:-1] if s.endswith("*") else s

    adsorbate = None
    if ads_list:
        adsorbate = _norm_ads(ads_list[0])
    elif inters:
        adsorbate = _norm_ads(inters[0])

    sc_match = re.search(r"(\d+)\s*[x×]\s*(\d+)(?:\s*[x×]\s*(\d+))?", user_text, flags=re.I)
    supercell = None
    if sc_match:
        supercell = f"{sc_match.group(1)}x{sc_match.group(2)}" + (f"x{sc_match.group(3)}" if sc_match.group(3) else "")
    vac_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:Å|Angstrom|A)\b", user_text, flags=re.I)
    vacuum = f"{vac_match.group(1)} Å" if vac_match else None

    tl = (user_text or "").lower() + " " + task.lower()
    intent_type = "workflow_planning"
    if re.search(r"adsorb|adsorption", tl) or adsorbate:
        intent_type = "adsorption_energy_calculation"
    elif re.search(r"\bdos\b|density of states|electronic", tl):
        intent_type = "dos_calculation"
    elif re.search(r"neb|transition state|\bts\b", tl):
        intent_type = "neb_calculation"

    domain = "catalysis" if stage == "catalysis" or area in ("electro", "thermal") else (stage or "materials")

    surf_txt = substrate or "the surface"
    if intent_type == "adsorption_energy_calculation":
        ad = adsorbate or "adsorbate"
        workflow = [
            f"Build slab model of {surf_txt}",
            f"Place {ad} molecule on high-symmetry adsorption sites (atop, bridge, hollow)",
            "Relax the structures",
            "Compute adsorption energies",
        ]
    elif intent_type == "dos_calculation":
        workflow = [
            f"Build/relax structure of {surf_txt}",
            "Generate DOS input parameters",
            "Run static calculation",
            "Post-process DOS",
        ]
    elif intent_type == "neb_calculation":
        workflow = [
            f"Build initial/final states for {surf_txt}",
            "Interpolate images",
            "Run NEB",
            "Extract barrier",
        ]
    else:
        workflow = ["Draft workflow steps"]

    def _is_magnetic(s: str) -> bool:
        return any(x in (s or "") for x in ["Fe", "Co", "Ni", "Mn", "Cr"])

    params = {"exchange_correlation": "PBE", "spin_polarization": bool(_is_magnetic(substrate))}

    if intent_type == "adsorption_energy_calculation":
        deliverables = ["Adsorption energy values", "Optimized geometries"]
    elif intent_type == "dos_calculation":
        deliverables = ["DOS plot", "Fermi level"]
    elif intent_type == "neb_calculation":
        deliverables = ["Energy barrier", "Optimized path"]
    else:
        deliverables = ["Structured plan"]

    next_step = ["POSCAR Builder", "INCAR Copilot", "Job Submission", "Post-analysis"]

    return {
        "intent_type": intent_type,
        "domain": domain,
        "target_system": {"surface": substrate or None, "supercell": supercell, "vacuum": vacuum},
        "adsorbate": adsorbate,
        "workflow": workflow,
        "parameters": params,
        "deliverables": deliverables,
        "next_step": next_step,
    }


def intent_unique(intent: dict, v2: dict, user_text: str) -> dict:
    import uuid
    it = intent or {}
    v2 = v2 or {}
    intent_type = v2.get("intent_type") or "workflow_planning"
    domain = v2.get("domain") or (it.get("stage") or it.get("area") or "catalysis")
    intent_id = f"{intent_type[:10].replace('_','')}_{str(uuid.uuid4())[:8]}"

    system = (v2.get("target_system") or {}) if isinstance(v2.get("target_system"), dict) else {}
    surface = system.get("surface") or it.get("facet") or it.get("substrate")
    adsorbates: List[str] = []
    if isinstance(v2.get("adsorbate"), str) and v2.get("adsorbate"):
        adsorbates.append(v2.get("adsorbate"))
    for x in (it.get("adsorbates") or []):
        s = str(x).strip(); s = s[:-1] if s.endswith("*") else s
        if s and s not in adsorbates:
            adsorbates.append(s)

    tl = (user_text or "") + " " + (it.get("task") or "")
    def rq_ads():
        ad = adsorbates[0] if adsorbates else "the adsorbate"
        surf = surface or "the surface"
        return f"What is the stability of {ad} adsorption on {surf}, and which adsorption site is most favorable?"
    def rq_dos():
        surf = surface or "the material"
        return f"What are the electronic states and density of states features of {surf} around the Fermi level?"
    def rq_neb():
        surf = surface or "the surface"
        return f"What is the activation barrier and minimum energy path for the elementary step on {surf}?"
    if "adsorb" in tl.lower():
        research_question = rq_ads()
        operation = "adsorption_energy_calculation"
    elif re.search(r"\bdos\b|density of states|electronic", tl, flags=re.I):
        research_question = rq_dos()
        operation = "dos_calculation"
    elif re.search(r"neb|transition state|\bts\b", tl, flags=re.I):
        research_question = rq_neb()
        operation = "neb_calculation"
    else:
        operation = intent_type
        research_question = (it.get("summary") or it.get("task") or "What scientific objective is being pursued?").rstrip(".") + "."

    tags = [str(x) for x in (it.get("tags") or [])]
    area = (it.get("area") or "").lower()
    ctx: List[str] = []
    if any(t.lower().startswith("co2rr") for t in tags) or "co2" in tl.lower():
        ctx.append("CO2 reduction (CO2RR)")
    if any("her" in t.lower() for t in tags):
        ctx.append("Hydrogen evolution reaction (HER)")
    if any("orr" in t.lower() for t in tags):
        ctx.append("Oxygen reduction reaction (ORR)")
    if area == "electro":
        ctx.append("Electrocatalysis")
    if area == "thermal":
        ctx.append("Thermocatalysis")
    scientific_context = ", ".join(dict.fromkeys(ctx)) or "General catalysis study"

    workflow_expectation = v2.get("workflow") if isinstance(v2.get("workflow"), list) else []
    params = v2.get("parameters") if isinstance(v2.get("parameters"), dict) else {}
    deliverables_expected = v2.get("deliverables") if isinstance(v2.get("deliverables"), list) else []

    handoff_notes = {
        "to_hypothesis_agent": "Suggest mechanistic hypotheses and expected trends based on operation and system.",
        "to_plan_agent": "Generate executable workflow for structure building, input generation, HPC submission, and post-analysis.",
    }

    return {
        "intent_id": intent_id,
        "domain": domain,
        "research_question": research_question,
        "scientific_context": scientific_context,
        "system": {"surface": surface, "supercell": system.get("supercell"), "vacuum": system.get("vacuum")},
        "adsorbates": adsorbates,
        "operation": operation,
        "workflow_expectation": workflow_expectation,
        "parameters": {"functional": params.get("exchange_correlation") or "PBE", "spin_polarization": params.get("spin_polarization", False)},
        "deliverables_expected": deliverables_expected,
        "handoff_notes": handoff_notes,
    }

