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
from fastapi.responses import JSONResponse
from .contracts import HypothesisBundle, RunEvent

# === Dynamic mechanism builder (replaces static REGISTRY) ===
try:
    from server.mechanisms.builder import build_mechanism as _build_mechanism
    _HAS_BUILDER = True
except ImportError:
    _HAS_BUILDER = False
    _build_mechanism = None  # type: ignore

REGISTRY: dict = {}  # kept for backward-compat; no longer used by endpoint

# 通用保存
async def _save_artifact(session_id: int | None, msg_type: str, content) -> Any:
    if not session_id:
        return
    from server.db import AsyncSessionLocal, ChatMessage
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

def _safe_json(s) -> Dict[str, Any]:
    # Handle dict responses from chatgpt_call (extract text content)
    if isinstance(s, dict):
        choices = s.get("choices") or []
        if choices:
            s = choices[0].get("message", {}).get("content", "")
        else:
            return s  # Already a dict, maybe it IS the JSON
    if not isinstance(s, str):
        return {}
    try:
        start = s.find("{"); end = s.rfind("}")
        if start >= 0 and end >= 0:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
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
    except ImportError:
        return False

async def _llm_markdown(fields: Dict[str, Any]) -> Optional[str]:
    if not _llm_available(): return None
    from server.utils.openai_wrapper import chatgpt_call  # type: ignore

    intent = fields.get("intent") or {}
    reactant, product = _extract_reactant_product(intent)
    direction_note = ""
    if reactant and product:
        direction_note = (
            f"\nCRITICAL: The reaction is {reactant} → {product} (FORWARD direction only). "
            f"In the elementary steps section, list steps starting from {reactant} and "
            f"ending at {product}. Never list steps in reverse order."
        )

    sys = (
        "You are a senior computational chemist. Given the structured JSON 'fields', "
        "write a concise markdown hypothesis exactly in this layout:\n\n"
        "**Conditions:** <one-liner>\n\n"
        "**Hypothesis:** <single sentence>\n\n"
        "**Why it may be true (mechanistic rationale):**\n"
        "- <bullet 1>\n- <bullet 2>\n- <bullet 3>\n\n"
        "**Elementary steps (forward direction):**\n"
        "- <step 1: reactant(s) → intermediate(s)>\n- ...\n\n"
        "**What to compute next:**\n"
        "- <DFT task 1>\n- <DFT task 2>\n- <DFT task 3>\n\n"
        "**Optional experimental validation:**\n"
        f"- <suggestion> (0-2 bullets){direction_note}"
    )
    cond = _conditions_line((fields.get("intent") or {}).get("conditions") or {})
    sys2 = f"{sys}\n\nNote: Summarize conditions as: {cond if cond!='-' else 'N/A'}."
    user = "FIELDS:\n" + json.dumps(fields, ensure_ascii=False, indent=2)
    try:
        resp = await chatgpt_call(
            [{"role":"system","content": sys2},
             {"role":"user","content": user}],
            temperature=0.3, max_tokens=600, json_mode=False
        )
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            txt = choices[0]["message"]["content"] if choices else ""
        else:
            txt = str(resp)
        return txt.strip() if isinstance(txt, str) else ""
    except (ValueError, KeyError, TypeError, IndexError):
        return None

async def _llm_graph(intent: Dict[str, Any], hint: str="", seed: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, Any]]:
    """Ask LLM for strict JSON graph, with optional seed suggestions."""
    if not _llm_available(): return None
    from server.utils.openai_wrapper import chatgpt_call  # type: ignore

    # Extract reactant/product for explicit forward-direction enforcement
    reactant, product = _extract_reactant_product(intent)
    direction_note = ""
    if reactant and product:
        direction_note = (
            f"\nCRITICAL: The reaction proceeds FORWARD: {reactant} → {product}. "
            f"The FIRST step MUST start with {reactant} as a reactant (not as a product). "
            f"Steps must be listed in forward causal order. "
            f"NEVER list {product} → {reactant} or any reverse-direction step as the first entry."
        )
        # For CO2RR specifically, ensure CO2 appears explicitly
        if "CO2" in reactant.upper() or "co2rr" in str(intent.get("tags", [])).lower():
            direction_note += (
                f"\nFor CO2RR: explicitly include CO2(g) + * -> CO2* as the very FIRST step. "
                f"Then COOH*, CO*, CHO*, etc. in forward order toward {product}."
            )

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
        f"Return ONLY JSON.{direction_note}"
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

def _to_list(x) -> List:
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
    def _uniq(seq) -> Any:
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

    rxn_lower = rxn.lower()
    if "her" in rxn_lower:
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

    # Extract key species from intent for specific templates
    sys_info = intent.get("system") or {}
    mol_list = sys_info.get("molecule") or []
    reactant = mol_list[0] if mol_list else (intent.get("reactant") or "substrate")
    product  = mol_list[-1] if len(mol_list) > 1 else (intent.get("product") or "product")

    if any(k in rxn_lower for k in ["dehydrog", "c-h activ", "alkane", "butane", "propane"]):
        return (
            f"**Conditions:** {cond_md}\n\n"
            f"**Hypothesis:** {cat_line} facilitates the dehydrogenation of {reactant} to {product} "
            f"via sequential C–H bond activation, with {cat_line} providing moderate adsorption of "
            f"alkyl intermediates and fast recombinative H₂ desorption.\n\n"
            f"**Why it may be true (mechanistic rationale):**\n"
            f"- {cat_line} (Group IB metal) binds alkyl intermediates weakly enough to avoid "
            f"deep dehydrogenation / coking, but strongly enough to activate C–H bonds.\n"
            f"- Primary and secondary C–H bonds in {reactant} differ in activation barrier "
            f"(sec < pri for 1-butene selectivity; pri < sec for 2-butene).\n"
            f"- Two sequential β-hydride eliminations: {reactant}* → C₄H₉* + H* → {product}* + 2H*.\n"
            f"- The rate-limiting step is likely the first C–H bond cleavage (highest barrier).\n"
            f"- Electrochemical promotion (if GC-DFT) can tune ΔG through the potential-dependent "
            f"CHE correction, opening a selectivity window.\n\n"
            f"**Key unknowns to resolve:**\n"
            f"- Adsorption geometry of {reactant}* on {cat_line}: flat (di-σ) vs. end-on.\n"
            f"- Relative stability of 1-butenyl vs. 2-butenyl radical intermediates.\n"
            f"- H* coverage at reaction conditions and its effect on recombination.\n"
            f"- Whether surface reconstruction (e.g., Ag surface oxide) occurs under O-containing "
            f"co-adsorbates.\n\n"
            f"**What to compute next:**\n"
            f"1. Relax {cat_line} slab (4×4, 4 layers, 15 Å vacuum).\n"
            f"2. Adsorption: {reactant}*, {product}*, C₄H₉* (1-butenyl & 2-butenyl), H* "
            f"(top/bridge/hollow sites, GCDFT for potential dependence).\n"
            f"3. Gas-phase thermochemistry: {reactant}(g), {product}(g), H₂(g) with ZPE+TS corrections.\n"
            f"4. NEB/CI-NEB: IS = {reactant}* → TS → FS = C₄H₉* + H* (×2 for both butenyl isomers).\n"
            f"5. NEB: C₄H₉* → {product}* + H*.\n"
            f"6. ZPE frequency calculations for all surface intermediates.\n"
            f"7. ΔG free-energy diagram at T=500 K and applied potential (GC-DFT).\n"
            f"8. Microkinetic model: TOF and selectivity vs. T and coverage.\n"
        )

    if any(k in rxn_lower for k in ["co2rr", "co2 reduc"]):
        return (
            f"**Conditions:** {cond_md}\n\n"
            f"**Hypothesis:** {cat_line} selectively reduces CO₂ to valuable C₁/C₂ products "
            f"via adsorbed COOH*/CO* intermediates under electrochemical driving force.\n\n"
            f"**What to compute next:**\n"
            f"- Adsorption: CO₂*, COOH*, CO*, CHO*, OCCHO*, H*.\n"
            f"- NEB barriers for CO₂ activation and key PCET steps.\n"
            f"- GC-DFT: potential-dependent ΔG diagram.\n"
            f"- Microkinetic model for TOF vs. potential.\n"
        )

    # Generic fallback with more context than the bare minimum
    return (
        f"**Conditions:** {cond_md}\n\n"
        f"**Hypothesis:** {cat_line} can facilitate the conversion of {reactant} to {product} "
        f"via surface-mediated intermediates. The key elementary steps involve adsorption, "
        f"bond activation, and desorption of products.\n\n"
        f"**What to compute next:**\n"
        f"- Build & relax {cat_line} slab (4×4, 4 layers).\n"
        f"- Adsorption energetics: key intermediates + reactant/product.\n"
        f"- Gas-phase thermochemistry with ZPE+entropy corrections.\n"
        f"- Transition states (NEB/CI-NEB) for rate-limiting steps.\n"
        f"- ΔG free-energy diagram + microkinetic model.\n"
    )

# ============================ Dynamic mechanism seed (builder) ============================

# Maps common reaction shortnames to (reactant, product) molecules
_RXN_MOLECULE_MAP = {
    "her":            ("H+",   "H2"),
    "co2rr":          ("CO2",  "CO"),
    "orr":            ("O2",   "H2O"),
    "oer":            ("H2O",  "O2"),
    "nrr":            ("N2",   "NH3"),
    "no3rr":          ("NO3-", "NH3"),
    "msr":            ("CH4",  "CO"),
    "haber":          ("N2",   "NH3"),
    "dehydrogenation":("C4H10","C4H8"),
    "c-h activation": ("C4H10","C4H8"),
    "alkane dehydrogenation": ("C4H10","C4H8"),
    "propane dehydrogenation":("C3H8","C3H6"),
    "methane activation":     ("CH4", "CH3"),
}


def _extract_reactant_product(intent: Dict[str, Any]) -> Tuple[str, str]:
    """
    Robustly extract (reactant, product) from any intent format.
    Checks top-level fields, system.molecule list, and reaction_network.steps.
    """
    # 1) Explicit top-level fields
    reactant = intent.get("reactant") or ""
    product  = intent.get("product")  or ""
    if reactant and product:
        return reactant, product

    # 2) system.molecule list: first = reactant, last = product
    mol_list = (intent.get("system") or {}).get("molecule") or []
    if not isinstance(mol_list, list):
        mol_list = [mol_list] if mol_list else []
    if len(mol_list) >= 2:
        if not reactant: reactant = str(mol_list[0])
        if not product:  product  = str(mol_list[-1])
    elif len(mol_list) == 1:
        if not reactant: reactant = str(mol_list[0])

    # 3) reaction_network.steps: parse "A → B" or "A -> B"
    rn = intent.get("reaction_network") or {}
    steps = rn.get("steps") or []
    if steps and (not reactant or not product):
        first_step = str(steps[0]) if steps else ""
        for sep in ["→", "->", "⟶"]:
            if sep in first_step:
                parts = first_step.split(sep, 1)
                lhs = parts[0].strip().replace("*", "").strip()
                rhs = parts[1].strip().replace("*", "").replace("+ H*","").strip()
                if lhs and not reactant: reactant = lhs
                if rhs and not product:  product  = rhs
                break

    # 4) _RXN_MOLECULE_MAP shortcut from task text
    if not reactant or not product:
        task_text = (intent.get("task") or intent.get("reaction") or "").lower()
        for key, (r0, p0) in _RXN_MOLECULE_MAP.items():
            if key in task_text:
                if not reactant: reactant = r0
                if not product:  product  = p0
                break

    return reactant, product


async def _get_mech_seed(intent: Dict[str, Any], session_id=None) -> Dict[str, Any]:
    """Call build_mechanism() and convert MechanismResult to hypothesis seed format."""
    if not _HAS_BUILDER or _build_mechanism is None:
        return {}
    try:
        sys_info = intent.get("system") or {}
        catalyst = (sys_info.get("catalyst") or sys_info.get("material") or
                    intent.get("catalyst") or "")
        facet    = sys_info.get("facet") or intent.get("facet") or "111"
        surface  = f"{catalyst}({facet})" if catalyst else None

        # Determine reaction domain
        rxn_text = " ".join([
            str(intent.get("domain") or ""),
            str(intent.get("area") or ""),
            str(intent.get("task") or ""),
            str(intent.get("reaction") or intent.get("problem_type") or ""),
            " ".join((intent.get("tags") or [])),
        ]).lower()

        domain = intent.get("domain") or ""
        if not domain:
            if any(k in rxn_text for k in ["co2rr", "her", "orr", "oer", "nrr", "no3rr"]):
                domain = "electrochemical"
            elif any(k in rxn_text for k in ["electroch", "potential", "pcet", "electrocatalys"]):
                domain = "electrochemical"
            elif any(k in rxn_text for k in ["dehydrog", "steam reform", "hydrogenation",
                                               "oxidation", "c-h activ", "thermal", "heterogen"]):
                domain = "thermal"
            elif "photo" in rxn_text:
                domain = "photocatalytic"
            else:
                domain = "thermal"  # safer default than electrochemical for unknown reactions

        reactant, product = _extract_reactant_product(intent)

        if not reactant or not product:
            return {}

        result = await _build_mechanism(
            domain=domain,
            reactant=reactant,
            product=product,
            surface=surface,
            conditions=intent.get("conditions"),
            session_id=session_id,
        )

        # Convert steps: [{"r":[...], "p":[...]}] -> "A + B -> C + D" strings
        rn = []
        for step in result.steps:
            if isinstance(step, dict):
                lhs = " + ".join(step.get("r") or [])
                rhs = " + ".join(step.get("p") or [])
                if lhs or rhs:
                    rn.append(f"{lhs} -> {rhs}")

        return {
            "reaction_network": rn,
            "intermediates":    result.intermediates,
            "coads_pairs":      result.coads,
            "ts_edges":         result.ts_candidates,
            "provenance":       {**result.provenance, "builder_name": result.name},
        }
    except Exception:
        return {}


# ============================ FastAPI endpoint ============================

@router.post("/chat/hypothesis")
async def hypothesis_create(request: Request):
    try:
        return await _hypothesis_create_impl(request)
    except Exception as e:
        import logging
        logging.getLogger("chatdft").exception("Hypothesis endpoint crashed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _hypothesis_create_impl(request: Request) -> None:
    data: Dict[str, Any] = (await request.json()) or {}
    session_id = data.get("session_id")
    # extract intent/knowledge/history (keep API compatible)
    intent: Dict[str, Any] = data.get("intent") or data.get("fields") or {}
    knowledge: Dict[str, Any] = data.get("knowledge") or {}
    history: List[Dict[str, Any]] = data.get("history") or []
    hint: str = data.get("hint") or data.get("hypothesis") or ""

    fields = {"intent": intent, "knowledge": knowledge, "history": history}

    # === step 0: dynamic mechanism seed from builder ===
    mech_seed = await _get_mech_seed(intent, session_id)

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
    if mech_seed.get("provenance", {}).get("builder_name"):
        graph_fixed["provenance"]["builder_name"] = mech_seed["provenance"]["builder_name"]

    # Auto-generate TS edges for dehydrogenation if LLM/builder didn't produce them
    if not graph_fixed.get("ts_edges"):
        _intent_text = " ".join([
            str(intent.get("task", "")), str(intent.get("area", "")),
            str(intent.get("problem_type", "")), " ".join(intent.get("tags") or [])
        ]).lower()
        _is_dehyd = any(k in _intent_text for k in ["dehydrog", "c-h activ", "butane", "c4h10", "alkane"])
        if _is_dehyd:
            # Derive TS edges from reaction_network elementary steps (C-H bond breaking steps)
            _rn_steps = graph_fixed.get("reaction_network") or []
            _ts_auto = []
            for _step in _rn_steps:
                _s = " + ".join(_step.get("lhs", [])) + " -> " + " + ".join(_step.get("rhs", [])) if isinstance(_step, dict) else _step
                # C-H activation steps: CₙHₓ* -> CₙHₓ₋₁* + H*
                if re.search(r"C\d+H\d+\*\s*[-+][^>]*->\s*C\d+H\d+\*.*H\*", _s, re.IGNORECASE) or \
                   re.search(r"C\d+H\d+\*\s*->\s*C\d+H\d+\*", _s, re.IGNORECASE):
                    _ts_auto.append(_s)
            # Fallback canonical TS for butane dehydrogenation
            if not _ts_auto:
                _inter = graph_fixed.get("intermediates") or []
                _c4_inter = [i for i in _inter if re.match(r"C4H\d+\*", i)]
                if len(_c4_inter) >= 2:
                    _ts_auto = [f"{_c4_inter[i]}* -> {_c4_inter[i+1]} + H*" if not _c4_inter[i].endswith("*")
                                else f"{_c4_inter[i]} -> {_c4_inter[i+1]} + H*"
                                for i in range(len(_c4_inter)-1)]
                else:
                    _ts_auto = ["C4H10* -> C4H9* + H*", "C4H9* -> C4H8* + H*"]
            graph_fixed["ts_edges"] = _ts_auto

    # 2) markdown hypothesis（与原逻辑一致）
    md = await _llm_markdown({"intent": intent, "knowledge": knowledge, "history": history})
    if not md:
        md = _template_markdown(intent)

    # Populate bundle from graph_fixed (not request body) so plan agent gets correct intermediates
    _rn = graph_fixed.get("reaction_network") or []
    _steps_str = []
    for _s in _rn:
        if isinstance(_s, str):
            _steps_str.append(_s)
        elif isinstance(_s, dict):
            _lhs = " + ".join(_s.get("lhs") or [])
            _rhs = " + ".join(_s.get("rhs") or [])
            _steps_str.append(f"{_lhs} -> {_rhs}")
    bundle = HypothesisBundle(
        md=md,
        steps=_steps_str or data.get("steps", []),
        intermediates=graph_fixed.get("intermediates") or data.get("intermediates", []),
        coads=[tuple(x) for x in (graph_fixed.get("coads_pairs") or data.get("coads", []) or []) if isinstance(x,(list,tuple)) and len(x)==2],
        ts=graph_fixed.get("ts_edges") or data.get("ts", []),
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
async def hypothesis_ingest_event(request: Request) -> Dict[str, Any]:
    evt = RunEvent(**(await request.json()))
    # TODO：可将 evt 持久化到 hypothesis_evidence 表，后续联动 refine
    return {"ok": True}


@router.post("/chat/hypothesis/feedback")
async def hypothesis_feedback(request: Request) -> None:
    """
    Receive execution results from low-level agents and feed them back to the
    hypothesis/plan for refinement.

    Expected body:
    {
      "session_id": int,
      "task_id": str,          // WorkflowTask id that just finished
      "result_type": str,      // "adsorption_energy" | "activation_barrier" | "debug" | ...
      "species": str,          // "CO*", "COOH*", ...
      "surface": str,          // "Pt(111)"
      "value": float,          // primary result in eV
      "converged": bool,
      "extra": {}              // additional structured data
    }

    The endpoint:
    1. Persists the result to DFTResult table
    2. Calls the LLM to interpret the result in the context of the current hypothesis
    3. Returns an updated hypothesis fragment + suggested plan adjustments
    """
    body = await request.json()
    session_id  = body.get("session_id")
    result_type = body.get("result_type", "unknown")
    species     = body.get("species", "")
    surface     = body.get("surface", "")
    value       = body.get("value")
    converged   = bool(body.get("converged", True))
    extra       = body.get("extra") or {}
    task_id     = body.get("task_id")

    # 1) Persist to DFTResult
    try:
        from server.db import AsyncSessionLocal, DFTResult
        from sqlalchemy import select
        async with AsyncSessionLocal() as db:
            row = DFTResult(
                session_id=session_id,
                result_type=result_type,
                species=species,
                surface=surface,
                value=float(value) if value is not None else None,
                converged=converged,
                extra=extra,
            )
            db.add(row)
            await db.commit()
    except Exception:
        pass  # don't break feedback on DB error

    # 2) Load current hypothesis context
    hypothesis_ctx = ""
    rxn_network_ctx = ""
    try:
        from server.db import AsyncSessionLocal, ChatMessage
        from sqlalchemy import select
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id,
                       ChatMessage.msg_type.in_(["hypothesis", "rxn_network"]))
                .order_by(ChatMessage.id.desc())
                .limit(3)
            )).scalars().all()
            for r in rows:
                if r.msg_type == "hypothesis" and not hypothesis_ctx:
                    hypothesis_ctx = r.content[:1200]
                elif r.msg_type == "rxn_network" and not rxn_network_ctx:
                    rxn_network_ctx = r.content[:800]
    except Exception:
        pass

    # 3) LLM interpretation of the new result
    from server.utils.openai_wrapper import LLMAgent
    prompt = f"""You are a DFT computational chemistry assistant. A new calculation result
has arrived. Interpret it in the context of the current hypothesis and suggest any
refinements to the research plan.

CURRENT HYPOTHESIS (excerpt):
{hypothesis_ctx or "(none yet)"}

REACTION NETWORK (excerpt):
{rxn_network_ctx or "(none yet)"}

NEW RESULT:
- Type: {result_type}
- Species: {species}  Surface: {surface}
- Value: {value} eV   Converged: {converged}
- Extra: {json.dumps(extra, ensure_ascii=False)[:400]}

Provide:
1. A one-paragraph interpretation of what this result means physically/chemically.
2. Any flag or warning if the value seems anomalous (e.g. implausibly large binding).
3. 1–3 concrete follow-up recommendations (e.g. "check coverage dependence", "rerun with DFT+U", "this step may be rate-limiting").

Be concise (< 200 words total). Use Markdown bullet points for #3."""

    interpretation = ""
    suggestions: List[str] = []
    try:
        raw = LLMAgent.chat(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=400,
        )
        interpretation = raw.strip()
        # Parse bullet points from the response for structured suggestions
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith(("- ", "* ", "• ")) or (len(line) > 2 and line[0].isdigit() and line[1] in ".):"):
                suggestions.append(re.sub(r"^[\-\*•\d\.\):\s]+", "", line).strip())
    except (ValueError, KeyError, TypeError) as e:
        interpretation = f"(LLM unavailable: {e})"

    # 4) Save interpretation back to chat
    await _save_artifact(session_id, "analysis",
                         f"## Feedback: {result_type} — {species} / {surface}\n\n{interpretation}")

    return {
        "ok": True,
        "session_id": session_id,
        "result_type": result_type,
        "species": species,
        "value": value,
        "converged": converged,
        "interpretation": interpretation,
        "suggestions": suggestions[:3],
    }