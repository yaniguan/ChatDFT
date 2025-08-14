# server/chat/hypothesis_agent.py
# -*- coding: utf-8 -*-
"""Advanced hypothesis-generation agent (LLM-first + validated graph)

Returns
-------
{
  "ok": true,
  "result_md": "<markdown hypothesis>",
  "fields": {...},             # echo of inputs for traceability
  "graph": {                   # structured network the planner/executor can consume
    "system": {"catalyst":"Pt","facet":"111"},
    "reaction_network": [{"lhs":["H+","e-","*"],"rhs":["H*"]}, ...],
    "intermediates": ["*","H*","OH*"],
    "coads_pairs": [["H*","H*"],["H*","OH*"]],
    "ts_edges": ["H* + H* -> H2(g) + 2*"],
    "provenance": {"source":"llm|fallback","model":"...", "warnings":[...]}
  }
}
"""
from __future__ import annotations

import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, Request
from .contracts import HypothesisBundle, RunEvent

# === NEW: mechanism registry (optional import; stays noop if missing) ===
try:
    from server.mechanisms.registry import REGISTRY
except Exception:
    REGISTRY = {}

# 通用保存
async def _save_artifact(session_id: int | None, msg_type: str, content):
    if not session_id:
        return
    from server.db_last import AsyncSessionLocal, ChatMessage
    import json as _json
    txt = content if isinstance(content, str) else _json.dumps(content, ensure_ascii=False)
    async with AsyncSessionLocal() as s:
        m = ChatMessage(session_id=session_id, role="assistant", msg_type=msg_type, content=txt)
        s.add(m)
        await s.commit()
router = APIRouter()

# ============================ small utils ============================

def _s(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) and x.strip() else "-"

def _join(items: List[Optional[str]]) -> str:
    return ", ".join([i.strip() for i in items if isinstance(i, str) and i.strip()]) or "-"

def _conditions_line(cond: Dict[str, Any]) -> str:
    return _join([
        f"pH={cond['pH']}"                   if cond.get("pH") else None,
        f"U={cond['potential']}"             if cond.get("potential") else None,
        f"electrolyte={cond['electrolyte']}" if cond.get("electrolyte") else None,
        f"solvent={cond['solvent']}"         if cond.get("solvent") else None,
        f"T={cond['temperature']}K"          if cond.get("temperature") else None,
    ])

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        start = s.find("{"); end = s.rfind("}")
        if start >= 0 and end >= 0:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except Exception:
        return {}

def _canonical_species(x: str) -> str:
    if not isinstance(x, str): return ""
    return x.strip().replace(" ", "")

def _norm_step(s: str) -> Dict[str, Any]:
    # "A + B* -> C* + D(g)" -> {"lhs":[...], "rhs":[...]}
    if not isinstance(s, str) or "->" not in s:
        z = _canonical_species(s) if isinstance(s, str) else ""
        return {"lhs":[z] if z else [], "rhs":[]}
    lhs, rhs = s.split("->", 1)
    L = [_canonical_species(z) for z in lhs.split("+")]
    R = [_canonical_species(z) for z in rhs.split("+")]
    L = [z for z in L if z]
    R = [z for z in R if z]
    return {"lhs":L, "rhs":R}

def _unique_seq(xs: List[Any]) -> List[Any]:
    out, seen = [], set()
    for x in xs:
        k = json.dumps(x, sort_keys=True) if not isinstance(x, str) else x
        if k not in seen:
            out.append(x); seen.add(k)
    return out

# -------------------- validation & auto-repair --------------------

def _validate_graph(h: Dict[str, Any], intent: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (fixed_graph, warnings)."""
    warnings: List[str] = []

    sys = intent.get("system") or {}
    metal = (sys.get("material") or intent.get("catalyst") or "Pt")
    facet = (sys.get("facet") or intent.get("facet") or "111")
    metal = str(metal).strip()
    facet = str(facet).strip()

    rn_in = h.get("reaction_network") or []
    inter_in = h.get("intermediates") or []
    coads_in = h.get("coads_pairs") or []
    ts_in = h.get("ts_edges") or []

    # normalize steps
    rn = []
    for item in rn_in:
        if isinstance(item, dict) and item.get("lhs") is not None and item.get("rhs") is not None:
            d = {"lhs":[_canonical_species(z) for z in (item["lhs"] or [])],
                 "rhs":[_canonical_species(z) for z in (item["rhs"] or [])]}
        else:
            d = _norm_step(item if isinstance(item, str) else str(item))
        # surface sanity: ensure at least one '*' on both sides if any adsorbate present
        has_surface = any(("*" in z) for z in d["lhs"] + d["rhs"])
        if has_surface and not any((z=="*" or z.endswith("*")) for z in d["lhs"]):
            d["lhs"].append("*")
        if has_surface and not any((z=="*" or z.endswith("*")) for z in d["rhs"]):
            d["rhs"].append("*")
        if d["lhs"] or d["rhs"]:
            rn.append(d)

    # intermediates
    inter = []
    for x in inter_in:
        x = _canonical_species(x)
        if not x: continue
        if x.endswith("*") or x.endswith("(g)") or x == "*":
            inter.append(x)
        else:
            inter.append(x + "*")
    inter = _unique_seq(inter)

    # co-ads pairs
    coads = []
    for pair in coads_in:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2: continue
        a, b = _canonical_species(pair[0]), _canonical_species(pair[1])
        if not a or not b: continue
        if not (a.endswith("*") or a=="*"): continue
        if not (b.endswith("*") or b=="*"): continue
        coads.append(tuple(sorted([a,b])))
    coads = sorted(_unique_seq(coads))

    # ts edges
    ts_edges = []
    for s in ts_in:
        if not isinstance(s, str) or "->" not in s: continue
        ts_edges.append(_canonical_species(s.replace("->"," -> ")).replace(" -> ", " -> "))
    ts_edges = _unique_seq(ts_edges)

    if not rn and (inter or coads or ts_edges):
        warnings.append("reaction_network empty; keeping intermediates/coads/ts only.")

    fixed = {
        "system": {"catalyst": metal, "facet": facet},
        "reaction_network": rn,
        "intermediates": inter,
        "coads_pairs": coads,
        "ts_edges": ts_edges,
        "provenance": h.get("provenance", {})
    }
    return fixed, warnings

# ============================ LLM calls ============================

_HAS_LLM = False
def _llm_available() -> bool:
    global _HAS_LLM
    if _HAS_LLM: return True
    if not os.getenv("OPENAI_API_KEY"): return False
    try:
        from server.utils.openai_wrapper import chatgpt_call  # noqa
        _HAS_LLM = True
        return True
    except Exception:
        return False

async def _llm_markdown(fields: Dict[str, Any]) -> Optional[str]:
    if not _llm_available(): return None
    from server.utils.openai_wrapper import chatgpt_call  # type: ignore
    sys = (
        "You are a senior computational chemist. Given the structured JSON 'fields', "
        "write a concise markdown hypothesis exactly in this layout:\n\n"
        "**Conditions:** <one-liner>\n\n"
        "**Hypothesis:** <single sentence>\n\n"
        "**Why it may be true (mechanistic rationale):**\n"
        "- <bullet 1>\n- <bullet 2>\n- <bullet 3>\n\n"
        "**What to compute next:**\n"
        "- <DFT task 1>\n- <DFT task 2>\n- <DFT task 3>\n\n"
        "**Optional experimental validation:**\n"
        "- <suggestion> (0-2 bullets)"
    )
    cond = _conditions_line((fields.get("intent") or {}).get("conditions") or {})
    sys2 = f"{sys}\n\nNote: Summarize conditions as: {cond if cond!='-' else 'N/A'}."
    user = "FIELDS:\n" + json.dumps(fields, ensure_ascii=False, indent=2)
    try:
        txt = await chatgpt_call(
            [{"role":"system","content": sys2},
             {"role":"user","content": user}],
            temperature=0.3, max_tokens=600
        )
        return txt.strip()
    except Exception:
        return None

async def _llm_graph(intent: Dict[str, Any], hint: str="", seed: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, Any]]:
    """Ask LLM for strict JSON graph, with optional seed suggestions."""
    if not _llm_available(): return None
    from server.utils.openai_wrapper import chatgpt_call  # type: ignore
    sys = (
        "You are an expert in heterogeneous and homogeneous catalysis. "
        "Produce STRICT JSON describing a reaction network for the given intent. "
        "Use '*' for adsorbates and '(g)' for gas when applicable. "
        "Schema:\n"
        "{\n"
        '  "reaction_network": ["A* + B -> C* + D(g)", ...],\n'
        '  "intermediates": ["H*","OH*","*"],\n'
        '  "coads_pairs": [["H*","H*"],["H*","OH*"]],\n'
        '  "ts_edges": ["H* + H* -> H2(g) + 2*"]\n'
        "}\n"
        "Return ONLY JSON."
    )
    payload = {"intent": intent, "hint": hint or ""}
    if seed:
        payload["seed"] = seed  # 作为“建议起点”，LLM可在此基础上细化或修正
    try:
        raw = await chatgpt_call(
            [{"role":"system","content": sys},
             {"role":"user","content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.2, max_tokens=1200
        )
        data = _safe_json(raw)
        if data and any(k in data for k in ("reaction_network","intermediates","coads_pairs","ts_edges")):
            return data
        return None
    except Exception:
        return None

# ============================ Mechanism expansion (NEW) ============================

# 机制别名规则（与 intent_agent 保持一致风格，轻量）
_MECH_ALIASES = [
    (r"\bco2rr\b|\bco2\b.*\breduc", ["CO2RR_CO_path","CO2RR_HCOO_path","CO2RR_to_ethanol_CO_coupling"]),
    (r"\bnrr\b|\bn2\b.*\breduc",    ["NRR_distal","NRR_alternating","NRR_dissociative"]),
    (r"\borr\b|\boxygen\s+reduction", ["ORR_4e"]),
    (r"\boer\b|\boxygen\s+evolution", ["OER_lattice_oxo_skeleton"]),
    (r"\bher\b|\bhydrogen\s+evolution", ["HER_VHT"]),
    (r"\bno3rr\b|\bnitrate\b.*\breduc", ["NO3RR_to_NH3_skeleton"]),
    (r"\bmsr\b|\bmethane\s+steam\s+reform", ["MSR_basic"]),
    (r"\bhaber\b|\bnh3\b.*\bsynth", ["Haber_Bosch_Fe"]),
    (r"\bco\s+oxidation\b", ["CO_oxidation_LH","CO_oxidation_MvK"]),
    (r"\bisomeriz", ["Hydroisomerization_zeolite"]),
    (r"\balkylation\b", ["Alkylation_acid"]),
    (r"\bdehydration\b|\bto\s+olefin\b", ["Alcohol_dehydration"]),
    (r"\bwilkinson\b|\brhcl\(pph3\)3\b|\balkene\s+hydrogenation", ["Wilkinson_hydrogenation"]),
    (r"\bhydroformylation\b|\brh\(pph3\)3cl\b|\bhco\(co\)4\b", ["Hydroformylation_Rh"]),
    (r"\bheck\b", ["Heck_Pd"]),
    (r"\bsuzuki\b", ["Suzuki_Pd"]),
    (r"\bsonogashira\b", ["Sonogashira_Pd_Cu"]),
    (r"\bepoxidation\b|\bsharpless\b|\bjacobsen\b", ["Epoxidation_Sharpless"]),
    (r"\bnoyori\b|\bknowles\b|\basymmetric\b.*\bhydrogenation", ["Asymmetric_Hydrogenation_Noyori"]),
    (r"\bphotocatalysis\b|\bphoto\s+water\s+split", ["Photocatalytic_water_splitting"]),
    (r"\bphotothermal\b.*co2", ["Photothermal_CO2RR_skeleton"]),
    (r"\bphotothermal\b.*methane|\bphotothermal\b.*ch4", ["Photothermal_methane_conversion"]),
]

def _to_list(x):
    if x is None: return []
    if isinstance(x,(list,tuple)): return list(x)
    return [x]

def _mech_guess_from_intent(intent: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    # 1) tags 直指
    for t in _to_list(intent.get("tags")):
        if isinstance(t, str) and t in REGISTRY and t not in keys:
            keys.append(t)
    # 2) 任务/反应名关键词
    text = " ".join([
        str(intent.get("task") or ""),
        str(intent.get("reaction") or ""),
        str((intent.get("deliverables") or {}).get("target_products") or "")
    ]).lower()
    if re.search(r"\bmethanol\b|\bch3oh\b", text):
        for k in ["CO2RR_CO_path","CO2RR_HCOO_path"]:
            if k in REGISTRY and k not in keys: keys.append(k)
    if re.search(r"\bethanol\b|\bch3ch2oh\b", text):
        if "CO2RR_to_ethanol_CO_coupling" in REGISTRY and "CO2RR_to_ethanol_CO_coupling" not in keys:
            keys.append("CO2RR_to_ethanol_CO_coupling")
    for pat, cand in _MECH_ALIASES:
        if re.search(pat, text):
            for k in cand:
                if k in REGISTRY and k not in keys:
                    keys.append(k)
    return keys[:4]

def _apply_variant(entry: Dict[str, Any], substrate: Optional[str], facet: Optional[str]) -> Dict[str, Any]:
    vs = entry.get("variants") or {}
    if facet and facet in vs: return vs[facet] or {}
    if substrate and substrate in vs: return vs[substrate] or {}
    return {}

def _expand_mech(keys: List[str], substrate: Optional[str], facet: Optional[str]) -> Dict[str, Any]:
    """合并多个机制为 seed graph：返回与 _llm_graph 相同 schema 的 JSON。"""
    inters, steps, coads, ts = [], [], [], []
    for k in keys:
        base = REGISTRY.get(k) or {}
        var = _apply_variant(base, substrate, facet)
        inters += _to_list(base.get("intermediates")) + _to_list(var.get("intermediates"))
        # steps（统一成字符串 'A + B -> C + D'，便于与 LLM 输出合并）
        for st in _to_list(base.get("steps")) + _to_list(var.get("steps")):
            if isinstance(st, dict):
                lhs = " + ".join(_to_list(st.get("r") or st.get("reactants") or []))
                rhs = " + ".join(_to_list(st.get("p") or st.get("products")  or []))
                if lhs or rhs:
                    steps.append(f"{lhs} -> {rhs}".strip())
            elif isinstance(st, str):
                steps.append(st)
        for pair in _to_list(base.get("coads")) + _to_list(var.get("coads")):
            if isinstance(pair,(list,tuple)) and len(pair)>=2:
                a,b = str(pair[0]), str(pair[1])
                coads.append([a,b])
    # 从 steps 推断 coads
    for st in steps:
        if "->" in st:
            L = st.split("->",1)[0]
            ads = [z.strip() for z in re.split(r"[+]", L) if z.strip().endswith("*")]
            if len(ads)>=2: coads.append([ads[0], ads[1]])
    # 去重
    def _uniq(seq):
        seen=set(); out=[]
        for x in seq:
            j = json.dumps(x, sort_keys=True) if isinstance(x, (dict,list)) else str(x)
            if j not in seen:
                seen.add(j); out.append(x)
        return out
    seed = {
        "reaction_network": _uniq(steps),
        "intermediates": _uniq(inters),
        "coads_pairs": _uniq(coads),
        "ts_edges": ts
    }
    return seed

def _merge_graph_like(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """合并两个 graph JSON（同 schema），列表合并去重。"""
    out = {}
    for k in ("reaction_network","intermediates","coads_pairs","ts_edges"):
        la = _to_list(a.get(k)); lb = _to_list(b.get(k))
        # 规范 steps 为字符串；coads 为二维；其余维持
        if k == "reaction_network":
            la = [x if isinstance(x,str) else f"{' + '.join(x.get('lhs',[]))} -> {' + '.join(x.get('rhs',[]))}" for x in la]
            lb = [x if isinstance(x,str) else f"{' + '.join(x.get('lhs',[]))} -> {' + '.join(x.get('rhs',[]))}" for x in lb]
        out[k] = _unique_seq(la + lb)
    return out

# ============================ Fallback seeds (legacy) ============================

def _seed_graph(intent: Dict[str, Any]) -> Dict[str, Any]:
    txt = (intent.get("reaction") or intent.get("problem_type") or intent.get("task") or "").lower()
    # HER seed
    if "her" in txt or "hydrogen evolution" in txt:
        rn = [
            "H+ + e- + * -> H*",
            "H* + H* -> H2(g) + 2*",
            "H* + H+ + e- -> H2(g) + *"
        ]
        inter = ["*","H*","H2O*","OH*"]
        coads = [["H*","H*"],["H*","OH*"]]
        ts = ["H* + H* -> H2(g) + 2*","H* + H+ -> H2(g) + *"]
        return {"reaction_network": rn, "intermediates": inter, "coads_pairs": coads, "ts_edges": ts}
    # CO2RR minimal seed
    if "co2rr" in txt or "co2 reduction" in txt or "co2->" in txt or "co2 to" in txt:
        rn = [
            "CO2(g) + * -> CO2*",
            "CO2* + H+ + e- -> COOH*",
            "COOH* + H+ + e- -> CO* + H2O(g) + *",
            "CO* -> CO(g) + *"
        ]
        inter = ["*","CO2*","COOH*","CO*","H*","OH*"]
        coads = [["CO*","H*"],["CO*","OH*"]]
        ts = ["COOH* -> CO* + OH*"]
        return {"reaction_network": rn, "intermediates": inter, "coads_pairs": coads, "ts_edges": ts}
    # generic
    return {"reaction_network": [], "intermediates": ["*"], "coads_pairs": [], "ts_edges": []}

# ============================ Template markdown fallback ============================

def _template_markdown(intent: Dict[str, Any]) -> str:
    dom   = _s(intent.get("domain"))
    rxn   = _s(intent.get("reaction") or intent.get("problem_type") or intent.get("task"))
    cat   = _s((intent.get("system") or {}).get("catalyst") or intent.get("catalyst"))
    facet = _s((intent.get("system") or {}).get("facet") or intent.get("facet"))
    cat_line = f"{cat}({facet})" if (cat != "-" and facet != "-") else cat
    cond_md = _conditions_line((intent.get("conditions") or {}))

    if "her" in (rxn.lower()):
        return (
            f"**Conditions:** {cond_md}\n\n"
            f"**Hypothesis:** {cat_line} can catalyse HER efficiently via a Volmer–Heyrovsky–Tafel manifold with near-zero ΔG_H* on key sites.\n\n"
            f"**Why it may be true (mechanistic rationale):**\n"
            f"- Volmer PCET barrier is lowered in the presence of adsorbed water.\n"
            f"- {cat_line} exposes sites with optimal H binding.\n"
            f"- Co-adsorbed OH* can assist proton shuttling.\n\n"
            f"**What to compute next:**\n"
            f"- Adsorption energies: H*, OH*, H2O*.\n"
            f"- Co-adsorption stability: H*+H*, H*+OH*.\n"
            f"- NEB barriers for Volmer/Heyrovsky/Tafel.\n"
            f"- Bader charges & PDOS under potential.\n"
        )
    return (
        f"**Conditions:** {cond_md}\n\n"
        f"**Hypothesis:** The chosen model is suitable to probe the target reactivity/physics.\n\n"
        f"**What to compute next:**\n"
        f"- Build & relax slab/supercell.\n"
        f"- Compute adsorption energetics of key intermediates.\n"
        f"- If needed, find transition states (NEB/CI-NEB).\n"
    )

# ============================ FastAPI endpoint ============================

@router.post("/chat/hypothesis")
async def hypothesis_create(request: Request):
    data: Dict[str, Any] = (await request.json()) or {}
    session_id = data.get("session_id")
    # extract intent/knowledge/history (keep API compatible)
    intent: Dict[str, Any] = data.get("intent") or data.get("fields") or {}
    knowledge: Dict[str, Any] = data.get("knowledge") or {}
    history: List[Dict[str, Any]] = data.get("history") or []
    hint: str = data.get("hint") or data.get("hypothesis") or ""

    fields = {"intent": intent, "knowledge": knowledge, "history": history}

    # === NEW step 0: mechanism-based seed from registry ===
    mech_keys = _mech_guess_from_intent(intent) if REGISTRY else []
    mech_seed = _expand_mech(
        mech_keys,
        substrate=intent.get("substrate") or (intent.get("system") or {}).get("catalyst"),
        facet=intent.get("facet") or (intent.get("system") or {}).get("facet"),
    ) if mech_keys else {}

    # 1) try LLM graph (strict JSON) with seed
    llm_graph_raw = await _llm_graph(intent, hint, seed=mech_seed if mech_seed else None)

    # 1.5) choose base graph before validation: merge LLM with seed if both exist
    raw_graph_for_validation: Dict[str, Any]
    provenance = {"provenance":{"source":"llm", **({"model":"gpt-4o-mini"} if os.getenv("OPENAI_API_KEY") else {})}}
    if llm_graph_raw and mech_seed:
        merged = _merge_graph_like(mech_seed, llm_graph_raw)
        raw_graph_for_validation = {**provenance, **merged}
    elif llm_graph_raw:
        raw_graph_for_validation = {**provenance, **llm_graph_raw}
    elif mech_seed:
        raw_graph_for_validation = {"provenance":{"source":"mechanism_seed"}, **mech_seed}
    else:
        # fallback: legacy seeds
        seeded = _seed_graph(intent)
        raw_graph_for_validation = {"provenance":{"source":"fallback"}, **seeded}

    # validate/repair
    graph_fixed, warns = _validate_graph(raw_graph_for_validation, intent)
    graph_fixed["provenance"]["warnings"] = _unique_seq(list(graph_fixed["provenance"].get("warnings", [])) + (warns or []))
    if mech_keys:
        # 把使用到的机制键记录在 provenance
        graph_fixed["provenance"]["mechanisms"] = mech_keys

    # 2) markdown hypothesis（与原逻辑一致）
    md = await _llm_markdown({"intent": intent, "knowledge": knowledge, "history": history})
    if not md:
        md = _template_markdown(intent)

    bundle = HypothesisBundle(
        md=md,
        steps=data.get("steps", []),
        intermediates=data.get("intermediates", []),
        coads=[tuple(x) for x in (data.get("coads", []) or []) if isinstance(x,(list,tuple)) and len(x)==2],
        ts=data.get("ts", []),
        confidence=float(data.get("confidence", 0.0) or 0.0),
    )

    # —— 保持原 DB 交互：保存 hypothesis（markdown）
    await _save_artifact(session_id, "hypothesis", md)

    # —— 保持原 DB 交互：保存 rxn_network（结构化）
    rxn_payload = {
        "elementary_steps": graph_fixed.get("reaction_network") or bundle.steps,
        "intermediates": graph_fixed.get("intermediates") or bundle.intermediates,
        "coads_pairs": graph_fixed.get("coads_pairs") or bundle.coads,
        "ts_candidates": graph_fixed.get("ts_edges") or bundle.ts,
        "system": graph_fixed.get("system"),
        "provenance": graph_fixed.get("provenance"),
    }
    await _save_artifact(session_id, "rxn_network", rxn_payload)

    return {"ok": True, "hypothesis": bundle.model_dump(), "graph": graph_fixed, "fields": fields}

@router.post("/chat/hypothesis/ingest_event")
async def hypothesis_ingest_event(request: Request):
    evt = RunEvent(**(await request.json()))
    # TODO：可将 evt 持久化到 hypothesis_evidence 表，后续联动 refine
    return {"ok": True}