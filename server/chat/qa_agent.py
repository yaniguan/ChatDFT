# server/chat/qa_agent.py
# -*- coding: utf-8 -*-
"""
QA & Benchmarking Hub — the intelligent "second opinion" layer.

Endpoints:
  POST /chat/qa/functional    — functional / parameter recommendations
  POST /chat/qa/surface       — surface stability sanity check
  POST /chat/qa/debug         — analyse a VASP job directory (OUTCAR debug)
  POST /chat/qa/free_energy   — compute ΔG diagram from session DFT results
  POST /chat/qa/microkinetics — run microkinetic model from session barriers

All endpoints accept {session_id?, ...} and return machine-readable JSON
plus a markdown summary suitable for rendering in the chat window.
"""
from __future__ import annotations

import json, os, re
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request

router = APIRouter()

# ── Lazy imports (avoid circular deps) ──────────────────────────────────

def _get_outcar_debugger() -> Any:
    from server.utils.outcar_debugger import (
        debug_job, recommend_functional, check_surface_stability,
        FUNCTIONAL_ADVICE, SURFACE_STABILITY,
    )
    return debug_job, recommend_functional, check_surface_stability

def _get_thermo() -> Any:
    from server.utils.thermo_utils import (
        build_free_energy_diagram, plot_free_energy_diagram,
        run_microkinetics_from_diagram, get_known_pathway,
        suggest_competing_pathways, KNOWN_PATHWAYS,
        apply_thermo_corrections,
    )
    return (build_free_energy_diagram, plot_free_energy_diagram,
            run_microkinetics_from_diagram, get_known_pathway,
            suggest_competing_pathways, KNOWN_PATHWAYS, apply_thermo_corrections)

def _get_llm() -> Any:
    from server.utils.openai_wrapper import chatgpt_call
    return chatgpt_call

# ── LLM enhancer ─────────────────────────────────────────────────────────

async def _llm_enhance(prompt_json: Dict, system_msg: str) -> str:
    """Ask the LLM to enrich a structured recommendation with context."""
    try:
        llm = _get_llm()
        return await llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": json.dumps(prompt_json, ensure_ascii=False)},
            ],
            model="gpt-4o-mini", temperature=0.2, max_tokens=600,
        )
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────────────────
# 1.  POST /chat/qa/functional
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/qa/functional")
async def qa_functional(request: Request) -> Dict[str, Any]:
    """
    Recommend DFT functional / INCAR settings for a given system.

    Body: {session_id?, system: str, adsorbate?: str, reaction?: str, incar?: dict}

    Examples:
      {"system": "CO adsorption on Pt(111)"}
        → "PBE overestimates CO binding. Use BEEF-vdW or +0.2 eV correction."
      {"system": "Fe2O3 catalyst for OER"}
        → "Add DFT+U (LDAUU=3.3 for Fe)."
    """
    data = await request.json()
    system_desc = (
        data.get("system") or
        data.get("adsorbate") or
        data.get("reaction") or
        data.get("query") or ""
    )

    debug_job, recommend_functional, check_surface_stability = _get_outcar_debugger()
    recs = recommend_functional(system_desc)

    # Extract material + facet for surface check
    mat_m   = re.search(r"\b([A-Z][a-z]?)(?:\((\d+)\))?", system_desc)
    mat     = mat_m.group(1) if mat_m else ""
    facet   = mat_m.group(2) if (mat_m and mat_m.group(2)) else "111"
    surf_warn = check_surface_stability(mat, facet) if mat else None

    # LLM enhancement
    llm_text = ""
    if recs or surf_warn:
        llm_text = await _llm_enhance(
            {"system": system_desc, "recommendations": recs, "surface_warning": surf_warn},
            system_msg=(
                "You are a DFT methods expert. Given the system description and "
                "structured recommendations, write a concise (≤150 words) "
                "markdown advisory for a graduate student. Lead with the most "
                "important recommendation and include one specific INCAR example."
            ),
        )

    return {
        "ok": True,
        "system": system_desc,
        "functional_recommendations": recs,
        "surface_stability": surf_warn,
        "advisory_md": llm_text or _fallback_functional_md(recs, surf_warn),
    }


def _fallback_functional_md(recs: List, surf_warn: Optional[Dict]) -> str:
    parts = []
    if surf_warn:
        parts.append(f"⚠️ **Surface stability:** {surf_warn['warning']}\n\n"
                     f"💡 {surf_warn['mitigation']}")
    for r in recs:
        parts.append(f"**{r['topic'].replace('_',' ').title()}:** {r['issue']}\n\n"
                     + "\n".join(f"- {x}" for x in r["recommendations"]))
    return "\n\n".join(parts) if parts else "No specific issues detected for this system."


# ─────────────────────────────────────────────────────────────────────────
# 2.  POST /chat/qa/surface
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/qa/surface")
async def qa_surface(request: Request) -> Dict[str, Any]:
    """
    Surface stability sanity check.

    Body: {material: str, facet: str, conditions?: dict}

    Returns warnings about known reconstructions and condition-dependent
    surface terminations.
    """
    data     = await request.json()
    material = data.get("material") or data.get("catalyst") or ""
    facet    = str(data.get("facet") or "111").strip()
    conds    = data.get("conditions") or {}

    debug_job, recommend_functional, check_surface_stability = _get_outcar_debugger()
    stability = check_surface_stability(material, facet)

    # Additional LLM check for condition-dependent stability
    llm_note = ""
    if material:
        llm_note = await _llm_enhance(
            {
                "material": material, "facet": facet, "conditions": conds,
                "known_stability_warning": stability,
                "task": (
                    "As a surface science expert: "
                    "1) Is this surface stable under the given conditions (T, P, potential)? "
                    "2) What termination/reconstruction should the DFT model use? "
                    "3) Any surface energy or Pourbaix diagram consideration? "
                    "Return ≤100 words markdown."
                ),
            },
            system_msg="You are a surface science and DFT expert. Be specific and cite examples.",
        )

    return {
        "ok": True,
        "material": material, "facet": facet,
        "stability_warning": stability,
        "llm_note": llm_note,
        "status": "warning" if stability else "ok",
    }


# ─────────────────────────────────────────────────────────────────────────
# 3.  POST /chat/qa/debug
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/qa/debug")
async def qa_debug(request: Request) -> Dict[str, Any]:
    """
    Intelligent VASP job debugger.

    Body: {job_dir: str, session_id?: int}

    Analyses OUTCAR / OSZICAR / INCAR and returns:
      - converged: bool
      - issues: [{code, severity, description, fixes}]
      - incar_patch: {KEY: value} dict — drop into INCAR to fix
      - summary: human-readable diagnosis
      - llm_advice: LLM-enhanced paragraph
    """
    data    = await request.json()
    job_dir = data.get("job_dir") or ""

    if not job_dir:
        return {"ok": False, "detail": "job_dir required"}

    from pathlib import Path
    if not Path(job_dir).is_dir():
        return {"ok": False, "detail": f"Directory not found: {job_dir}"}

    debug_job, recommend_functional, check_surface_stability = _get_outcar_debugger()
    report = debug_job(job_dir)

    # LLM enhancement for non-trivial issues
    llm_advice = ""
    if report.issues:
        llm_advice = await _llm_enhance(
            {
                "job_dir": job_dir,
                "converged": report.converged,
                "elec_converged": report.elec_conv,
                "issues": [{"code": i.code, "desc": i.description,
                             "fixes": i.fixes[:3]} for i in report.issues],
                "incar_patch": report.incar_patch,
            },
            system_msg=(
                "You are a VASP expert reviewing a failed DFT job. "
                "Explain the diagnosis in plain language (2-3 sentences), "
                "then give the top 3 INCAR lines to add/change. "
                "Format as markdown with an INCAR code block."
            ),
        )

    result = report.to_dict()
    result["ok"] = True
    result["llm_advice"] = llm_advice or _fallback_debug_md(report)
    return result


def _fallback_debug_md(report) -> str:
    if not report.issues:
        return "✅ Job appears healthy."
    lines = [f"**Diagnosis — {report.job_dir}**\n"]
    for i in report.issues:
        icon = "🔴" if i.severity == "critical" else ("🟡" if i.severity == "warning" else "🔵")
        lines.append(f"{icon} **{i.code}**: {i.description}")
        if i.fixes:
            lines.append("```\n" + "\n".join(i.fixes[:4]) + "\n```")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# 4.  POST /chat/qa/free_energy
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/qa/free_energy")
async def qa_free_energy(request: Request) -> Dict[str, Any]:
    """
    Build ΔG diagrams from session DFT results.

    Body:
      {
        session_id: int,
        reaction: "CO2RR" | "HER" | "OER" | "NRR" | ...,
        temperature: 298.15,    # K
        potential_V: 0.0,       # V vs RHE
        use_known_pathway: true,  # use literature reference values
        # OR provide custom data:
        pathways: [
          {
            name: "carboxyl",
            intermediates: ["CO2+*", "COOH*", "CO*", "CO(g)"],
            E_dft: [0.0, 0.22, -0.56, 0.10],   # relative DFT energies
            ec_steps: [1,2], n_e: [1,1,0,0]
          },
          ...
        ]
      }
    """
    data       = await request.json()
    session_id = data.get("session_id")
    reaction   = (data.get("reaction") or "").upper()
    T          = float(data.get("temperature", 298.15))
    U          = float(data.get("potential_V", 0.0))

    (build_fe, plot_fe, run_mk, get_known, suggest_pw,
     KNOWN_PW, apply_thermo) = _get_thermo()

    diagrams_data = []
    plot_b64      = None
    diagram_objs  = []

    # --- Use known literature pathways ---
    if data.get("use_known_pathway", True) and reaction:
        keys = suggest_pw(reaction)
        if not keys:
            keys = [k for k in KNOWN_PW if reaction.split("RR")[0] in k.upper()]
        for key in keys:
            diag = get_known(key, T=T, U=U)
            if diag:
                diagram_objs.append(diag)
                diagrams_data.append(diag.to_dict())

    # --- Use custom pathways from request ---
    for pw in (data.get("pathways") or []):
        try:
            # Apply ZPE+entropy corrections if E_dft provided
            E_dft  = pw.get("E_dft") or pw.get("G_relative") or []
            inters = pw.get("intermediates") or []
            # Apply thermo corrections to each intermediate
            G_vals = []
            for i, (sp, e) in enumerate(zip(inters, E_dft)):
                corr = apply_thermo(float(e), sp, T=T)
                # Make relative to first
                G_vals.append(corr.G - (apply_thermo(float(E_dft[0]), inters[0], T=T).G
                                        if i > 0 else 0.0))
            if G_vals:
                G_vals[0] = 0.0  # anchor at zero
            diag = build_fe(
                pathway_name    = pw.get("name") or f"pathway_{len(diagram_objs)+1}",
                intermediates   = inters,
                G_relative      = G_vals,
                ec_step_indices = pw.get("ec_steps"),
                electrons_per_step = pw.get("n_e"),
                T=T, potential_V=U,
            )
            diagram_objs.append(diag)
            diagrams_data.append(diag.to_dict())
        except (ValueError, KeyError, TypeError) as e:
            pass

    # --- Load DFT results from DB and build pathway ---
    if session_id and not diagrams_data:
        try:
            db_results = await _load_dft_results(session_id)
            if db_results:
                inters = [r["species"] for r in db_results]
                E_raw  = [float(r["value"]) for r in db_results]
                E_rel  = [e - E_raw[0] for e in E_raw]
                diag   = build_fe(
                    pathway_name    = f"session-{session_id} pathway",
                    intermediates   = inters,
                    G_relative      = E_rel,
                    T=T, potential_V=U,
                )
                diagram_objs.append(diag)
                diagrams_data.append(diag.to_dict())
        except Exception:
            pass

    # --- Plot ---
    if diagram_objs:
        title = f"{reaction} Free Energy Diagram" if reaction else "ΔG Diagram"
        try:
            plot_b64 = plot_fe(diagram_objs, title=title)
        except Exception:
            pass

    # --- LLM interpretation ---
    llm_text = ""
    if diagrams_data:
        llm_text = await _llm_enhance(
            {"diagrams": [{"pathway": d["pathway"],
                           "U_limiting": d.get("U_limiting_V"),
                           "overpotential": d.get("overpotential_V"),
                           "rds": d["steps"][d["rds_index"]]["label"] if d["rds_index"] >= 0 else "?"}
                          for d in diagrams_data],
             "potential_V": U, "T_K": T},
            system_msg=(
                "You are a computational electrochemist. Given the free energy diagram data, "
                "write a ≤120-word markdown interpretation: identify the rate-determining step, "
                "compare limiting potentials between pathways (if >1), and suggest the "
                "next DFT calculation to reduce uncertainty."
            ),
        )

    return {
        "ok": True,
        "reaction": reaction, "T_K": T, "U_V_RHE": U,
        "diagrams": diagrams_data,
        "plot_png_b64": plot_b64,   # render with <img src="data:image/png;base64,...">
        "interpretation_md": llm_text,
    }


async def _load_dft_results(session_id: int) -> List[Dict]:
    """Load DFTResult rows from DB ordered by task_id (reaction coordinate)."""
    try:
        from server.db import AsyncSessionLocal, DFTResult
        from sqlalchemy import select
        async with AsyncSessionLocal() as s:
            rows = (await s.execute(
                select(DFTResult)
                .where(DFTResult.session_id == session_id)
                .order_by(DFTResult.task_id)
            )).scalars().all()
            return [{"species": r.species, "value": r.value_ev,
                     "task_id": r.task_id, "type": r.result_type}
                    for r in rows if r.value_ev is not None]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────
# 5.  POST /chat/qa/microkinetics
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/qa/microkinetics")
async def qa_microkinetics(request: Request) -> Dict[str, Any]:
    """
    Run a microkinetic model from session barriers or provided data.

    Body:
      {
        session_id?: int,
        reaction: "CO2RR",
        temperature: 298.15,
        potential_V: -0.8,
        use_known_pathway: true,
        # OR provide barriers explicitly:
        steps: ["CO2→COOH*", "COOH*→CO*", "CO*→CO"],
        Ea_eV: [0.22, 0.78, 0.10],
        delta_G_eV: [0.22, -0.78, 0.66]
      }
    """
    data       = await request.json()
    T          = float(data.get("temperature", 298.15))
    U          = float(data.get("potential_V", 0.0))
    reaction   = (data.get("reaction") or "").upper()

    (build_fe, plot_fe, run_mk, get_known, suggest_pw,
     KNOWN_PW, apply_thermo) = _get_thermo()

    mk_result = None

    # Try known pathway first
    if data.get("use_known_pathway", True) and reaction:
        keys = suggest_pw(reaction)
        if keys:
            diag = get_known(keys[0], T=T, U=U)
            if diag:
                mk_result = run_mk(diag, T=T)

    # Override with explicit steps if provided
    steps   = data.get("steps")
    Ea_fwd  = data.get("Ea_eV")
    dG_list = data.get("delta_G_eV")
    if steps and Ea_fwd and dG_list:
        from server.utils.thermo_utils import solve_microkinetics
        mk_result = solve_microkinetics(
            step_labels=steps, Ea_fwd_eV=Ea_fwd, delta_G_eV=dG_list, T=T
        )

    if mk_result is None:
        return {"ok": False, "detail": "No pathway data available for microkinetic modeling."}

    # LLM interpretation
    llm_text = await _llm_enhance(
        {"microkinetics": mk_result, "reaction": reaction, "T_K": T, "U_V": U},
        system_msg=(
            "You are a kinetics expert. Given the microkinetic result (TOF, coverages, "
            "rate-controlling step), write a ≤100-word markdown summary: "
            "report TOF in scientific notation, explain the rate-controlling step, "
            "and suggest one experiment to validate."
        ),
    )

    return {
        "ok": True,
        "reaction": reaction, "T_K": T, "U_V_RHE": U,
        "microkinetics": mk_result,
        "interpretation_md": llm_text,
    }


# ─────────────────────────────────────────────────────────────────────────
# 6.  POST /chat/qa/neb_prep
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/qa/neb_prep")
async def qa_neb_prep(request: Request) -> Dict[str, Any]:
    """
    Suggest NEB pathway(s) and generate input-file checklist.

    Body: {session_id?, reaction: str, surface: str, intermediates?: [...]}

    Returns:
      - competing_pathways: e.g. ["carboxyl (COOH*)", "formate (HCOO*)"] for CO2RR
      - neb_pairs: list of (IS, FS) pairs to run CI-NEB
      - incar_neb: recommended INCAR settings for NEB
      - checklist_md: human-readable checklist
    """
    data       = await request.json()
    reaction   = (data.get("reaction") or "HER").upper()
    surface    = data.get("surface") or data.get("catalyst") or "Cu(111)"
    inters_in  = data.get("intermediates") or []

    (build_fe, plot_fe, run_mk, get_known, suggest_pw,
     KNOWN_PW, apply_thermo) = _get_thermo()

    pw_keys    = suggest_pw(reaction)
    pathways   = []
    neb_pairs  = []

    for key in pw_keys:
        pw = KNOWN_PW.get(key)
        if pw is None:
            continue
        inters = pw["intermediates"]
        pathways.append({
            "key": key,
            "description": pw.get("description", key),
            "intermediates": inters,
        })
        # Adjacent pairs → NEB jobs
        for i in range(len(inters) - 1):
            neb_pairs.append({
                "IS": inters[i],
                "FS": inters[i+1],
                "pathway": key,
                "priority": "high" if i == 0 else "medium",
            })

    # If no known pathway, use provided intermediates
    if not pathways and inters_in:
        for i in range(len(inters_in) - 1):
            neb_pairs.append({"IS": inters_in[i], "FS": inters_in[i+1],
                               "pathway": "custom", "priority": "high"})

    incar_neb = {
        "ICHAIN":  "0   # NEB",
        "IMAGES":  "7   # number of NEB images (increase to 11 for accuracy)",
        "SPRING":  "-5  # spring constant (eV/Å²)",
        "LCLIMB":  ".TRUE.  # climbing image NEB (CI-NEB)",
        "IBRION":  "3",
        "POTIM":   "0.0",
        "SMASS":   "0",
        "EDIFF":   "1E-5",
        "EDIFFG":  "-0.05  # force criterion (eV/Å); tighten to -0.02 for publication",
        "NSW":     "500",
        "ALGO":    "Fast",
    }

    # LLM enrichment
    llm_text = await _llm_enhance(
        {
            "reaction": reaction, "surface": surface,
            "competing_pathways": [p["description"] for p in pathways],
            "n_neb_jobs": len(neb_pairs),
        },
        system_msg=(
            "You are a DFT NEB expert. Given the reaction and competing pathways, "
            "write a ≤120-word markdown guide: which pathway to run first, why, "
            "typical barrier values from literature, and one pitfall to avoid. "
            "Include a note on image count selection."
        ),
    )

    checklist_md = _neb_checklist_md(reaction, surface, pathways, neb_pairs, incar_neb)

    return {
        "ok": True,
        "reaction": reaction, "surface": surface,
        "competing_pathways": pathways,
        "neb_pairs": neb_pairs,
        "incar_neb": incar_neb,
        "llm_guide": llm_text,
        "checklist_md": checklist_md,
    }


def _neb_checklist_md(reaction, surface, pathways, neb_pairs, incar_neb) -> str:
    lines = [
        f"## NEB Preparation Checklist — {reaction} on {surface}\n",
        "### Competing Pathways",
    ]
    for i, pw in enumerate(pathways, 1):
        lines.append(f"{i}. **{pw['description']}**  \n"
                     f"   Intermediates: {' → '.join(pw['intermediates'])}")

    lines += ["\n### NEB Jobs to Submit", "| # | IS | FS | Pathway | Priority |",
              "|---|----|----|---------|----------|"]
    for i, p in enumerate(neb_pairs, 1):
        lines.append(f"| {i} | {p['IS']} | {p['FS']} | {p['pathway']} | {p['priority']} |")

    lines += [
        "\n### INCAR Settings for NEB",
        "```fortran",
        *[f"{k} = {v}" for k, v in incar_neb.items()],
        "```",
        "\n### Workflow Steps",
        "1. Relax IS and FS structures to EDIFFG = -0.02 eV/Å",
        "2. Generate NEB images with `nebmake.pl IS FS N` (VTST scripts) or ASE NEB",
        "3. Run CI-NEB; verify max-force < 0.05 eV/Å on climbing image",
        "4. Confirm exactly **1 imaginary frequency** on the TS (IBRION=5)",
        "5. Extract barrier: Ea = E_TS - E_IS (eV)",
    ]
    return "\n".join(lines)
