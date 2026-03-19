# server/chat/plan_agent.py
# -*- coding: utf-8 -*-
"""
Plan Agent — LLM + RAG + hypothesis 优先 + 轻量种子兜底 (+ Mechanism Registry)

Endpoints
---------
POST /chat/plan
  - 输入: {session_id?, intent, hypothesis(str|dict)?, knowledge?, history?, query?}
  - 流程:
      1) RAG 历史/知识上下文
      2) 优先采用 hypothesis 的结构化反应网络 / external graph
      3) 若缺失则 LLM 生成（注入 RAG/knowledge/history 作为 hint）
      4) 合并 mechanism registry 的 seed（如命中），质量清洗 + 限流
      5) 产出 tasks + rxn_network（steps/inter/coads/ts）+ workflow（便于 HPC 页面提交）
  - 返回: {ok, steps, intermediates, coads, ts, tasks, workflow, confidence, ...}

POST /chat/execute
  - 选择性执行部分 tasks（示例执行层，安全容错）
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import asyncio
import tempfile
import json
import os
import re
import unicodedata
from datetime import datetime

from fastapi import APIRouter, FastAPI, Request

# ---------- LLM & RAG ----------
try:
    from server.utils.openai_wrapper import chatgpt_call  # async
except Exception:
    async def chatgpt_call(messages, **kw):
        return json.dumps({"steps": [], "intermediates": [], "coads": [], "ts": []})

try:
    from server.utils.rag_utils import rag_context  # async
except Exception:
    async def rag_context(query: str, session_id: int | None = None, top_k: int = 8) -> str:
        return ""

# ---------- Dynamic Mechanism Builder ----------
try:
    from server.mechanisms.builder import build_mechanism as _build_mechanism
    _HAS_BUILDER = True
except Exception:
    _HAS_BUILDER = False
    _build_mechanism = None  # type: ignore

REGISTRY: dict = {}  # kept for backward-compat; no longer used by endpoint

router = APIRouter()

# ---------- 轻量持久化（失败静默） ----------
async def _save_artifact(session_id: int | None, msg_type: str, content: Any):
    if not session_id:
        return
    txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    # 优先 async 会话
    try:
        from server.db_last import AsyncSessionLocal, ChatMessage  # type: ignore
        async with AsyncSessionLocal() as s:                       # type: ignore
            m = ChatMessage(session_id=session_id, role="assistant", msg_type=msg_type, content=txt)
            s.add(m)
            await s.commit()
        return
    except Exception:
        pass
    # 回退到 sync 会话（放到线程池）
    try:
        from server.db import SessionLocal, ChatMessage  # type: ignore
        def _save_sync():
            s = SessionLocal()
            try:
                m = ChatMessage(session_id=session_id, role="assistant", msg_type=msg_type, content=txt)
                s.add(m)
                s.commit()
            finally:
                s.close()
        await asyncio.to_thread(_save_sync)
    except Exception:
        # 静默失败，避免 500
        return

# ---------- 小工具 ----------
def _slug(s) -> str:
    s = str(s or "").strip()
    try:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "job"

# SMILES map for gas-phase molecules used in plan task generation
_MOL_SMILES: Dict[str, str] = {
    # C4 dehydrogenation intermediates
    "C4H10":    "CCCC",           # n-butane
    "n-butane": "CCCC",
    "C4H9":     "CCC[CH2]",       # 1-butyl radical (primary)
    "C4H9-1":   "CCC[CH2]",       # 1-butyl
    "C4H9-2":   "CC[CH]C",        # 2-butyl radical (secondary)
    "1-C4H9":   "CCC[CH2]",
    "2-C4H9":   "CC[CH]C",
    "C4H8":     "C=CCC",          # 1-butene
    "C4H8-1":   "C=CCC",          # 1-butene
    "C4H8-2":   "CC=CC",          # 2-butene (trans)
    "1-butene":  "C=CCC",
    "2-butene":  "CC=CC",
    "1-C4H8":   "C=CCC",
    "2-C4H8":   "CC=CC",
    "C4H6":     "C=CC=C",         # 1,3-butadiene
    # Small alkanes/alkenes
    "C3H8":  "CCC",
    "C3H6":  "C=CC",
    "C2H6":  "CC",
    "C2H4":  "C=C",
    "CH4":   "C",
    # Common gases
    "CO2":   "O=C=O",
    "CO":    "[C-]#[O+]",
    "H2O":   "O",
    "NH3":   "N",
    "H2":    "[HH]",
    "N2":    "N#N",
    "O2":    "O=O",
    # Oxygenates
    "CH3OH":    "CO",
    "C2H5OH":   "CCO",
    "HCOOH":    "OC=O",
    "CH3COOH":  "CC(=O)O",
    "acetone":  "CC(=O)C",
}

# ========================= Tunables =========================
DEFAULTS = dict(
    USE_SEED_POLICY="auto",     # "auto" | "never" | "always"
    CONF_THRESHOLD=0.55,        # < this -> light merge with seed
    LIMITS={"inter": 40, "coads": 80, "ts": 40},
    STRICT=True                 # ban charged species in intermediates (electrochem steps still allowed)
)

# ========================= Intent parsing（简化兜底） =========================
def _extract_intent(body: Dict[str, Any]) -> Dict[str, Any]:
    intent = body.get("intent") or {}
    text   = (body.get("query") or body.get("text") or "").strip().lower()

    def pick_rxn(t: str) -> str:
        if any(k in t for k in ["co2rr","co2 reduction","co2->","co2 to"]): return "CO2RR"
        if any(k in t for k in ["oer","oxygen evolution"]): return "OER"
        if any(k in t for k in ["orr","oxygen reduction"]): return "ORR"
        if any(k in t for k in ["nrr","nitrogen reduction","nh3 synthesis"]): return "NRR"
        if any(k in t for k in ["msr","steam reform","ch4 reform"]): return "MSR"
        if any(k in t for k in ["dehydrog","c4h10","butane","c-h activ","alkane dehydrog"]): return "DEHYDROGENATION"
        return "HER"

    def pick_cat(t: str) -> str:
        for m in ["pt","cu","ni","co","fe","ag","au","pd","rh","ir","ru"]:
            if m in t: return m.upper()
        return "PT"

    def pick_facet(t: str) -> str:
        for f in ["111","100","110","211","533"]:
            if f in t: return f
        return "111"

    if not intent and text:
        rxn = pick_rxn(text)
        intent = {
            "domain": "catalysis",
            "problem_type": rxn,
            "system": {"material": pick_cat(text), "catalyst": pick_cat(text), "facet": pick_facet(text)},
        }

    # normalize
    sys = intent.get("system") or {}
    intent["system"] = {
        "material": (sys.get("material") or sys.get("catalyst") or "PT").upper(),
        "catalyst": (sys.get("catalyst") or sys.get("material") or "PT").upper(),
        "facet": sys.get("facet") or intent.get("facet") or "111",
    }
    intent["problem_type"] = intent.get("problem_type") or intent.get("reaction") or "HER"
    return intent

# ========================= Mechanism utils (NEW) =========================
# 轻量 alias 规则（与 intent/hypothesis 保持一致）
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

def _mech_guess(intent: Dict[str, Any], query: str) -> List[str]:
    text = " ".join([
        query or "",
        str(intent.get("task") or ""),
        str(intent.get("reaction") or intent.get("problem_type") or ""),
        " ".join(_to_list((intent.get("deliverables") or {}).get("target_products")))
    ]).lower()
    keys: List[str] = []
    # tags 直接命中
    for t in _to_list(intent.get("tags")):
        if isinstance(t, str) and t in REGISTRY and t not in keys:
            keys.append(t)
    # 产物线索
    if re.search(r"\bmethanol\b|\bch3oh\b", text):
        for k in ["CO2RR_CO_path","CO2RR_HCOO_path"]:
            if k in REGISTRY and k not in keys: keys.append(k)
    if re.search(r"\bethanol\b|\bch3ch2oh\b", text):
        if "CO2RR_to_ethanol_CO_coupling" in REGISTRY and "CO2RR_to_ethanol_CO_coupling" not in keys:
            keys.append("CO2RR_to_ethanol_CO_coupling")
    # alias
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

def _mech_seed(intent: Dict[str, Any], mech_keys: List[str]) -> Dict[str, Any]:
    """从 REGISTRY 合并生成 seed: {steps, intermediates, coads, ts}"""
    inters, steps, coads = [], [], []
    substrate = (intent.get("system") or {}).get("catalyst") or intent.get("substrate")
    facet     = (intent.get("system") or {}).get("facet") or intent.get("facet")
    for k in mech_keys:
        base = REGISTRY.get(k) or {}
        var  = _apply_variant(base, substrate, facet)
        inters += _to_list(base.get("intermediates")) + _to_list(var.get("intermediates"))
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
                coads.append((str(pair[0]), str(pair[1])))
    # 从 steps 左侧推断共吸附
    for st in steps:
        if "->" in st:
            L = st.split("->",1)[0]
            ads = [z.strip() for z in re.split(r"[+]", L) if z.strip().endswith("*")]
            if len(ads)>=2:
                coads.append((ads[0], ads[1]))
    # 去重
    def _uniq(seq):
        seen=set(); out=[]
        for x in seq:
            j = json.dumps(x, sort_keys=True) if isinstance(x,(dict,list,tuple)) else str(x)
            if j not in seen: seen.add(j); out.append(x)
        return out
    seed = {
        "steps": _uniq(steps),
        "intermediates": _uniq(inters),
        "coads": _uniq(coads),
        "ts": []
    }
    return seed

# ========================= Dynamic mechanism seed (builder) =========================

_RXN_MOLECULE_MAP = {
    "her":   ("H+",   "H2"),
    "co2rr": ("CO2",  "CO"),
    "orr":   ("O2",   "H2O"),
    "oer":   ("H2O",  "O2"),
    "nrr":   ("N2",   "NH3"),
    "no3rr": ("NO3-", "NH3"),
    "msr":   ("CH4",  "CO"),
    "haber": ("N2",   "NH3"),
}

async def _get_mech_seed_builder(
    intent: Dict[str, Any],
    query: str = "",
    session_id=None,
) -> Dict[str, Any]:
    """
    Call build_mechanism() and return seed in plan_agent format:
    {steps: [...str], intermediates: [...str], coads: [...tuple], ts: [...str], _name: str}
    Returns {} on any failure so callers fall back to legacy seeds.
    """
    if not _HAS_BUILDER or _build_mechanism is None:
        return {}
    try:
        sys_info = intent.get("system") or {}
        catalyst = (sys_info.get("catalyst") or sys_info.get("material") or
                    intent.get("catalyst") or "")
        facet    = sys_info.get("facet") or intent.get("facet") or "111"
        surface  = f"{catalyst}({facet})" if catalyst else None

        rxn_text = " ".join([
            query or "",
            str(intent.get("domain") or ""),
            str(intent.get("task") or ""),
            str(intent.get("reaction") or intent.get("problem_type") or ""),
        ]).lower()

        domain = intent.get("domain") or ""
        if not domain:
            if any(k in rxn_text for k in ["co2rr", "her", "orr", "oer", "nrr", "no3rr", "electroch"]):
                domain = "electrochemical"
            elif any(k in rxn_text for k in ["dehydrog", "steam reform", "hydrogenation", "oxidation"]):
                domain = "thermal"
            elif "photo" in rxn_text:
                domain = "photocatalytic"
            else:
                domain = "electrochemical"

        reactant = intent.get("reactant") or ""
        product  = intent.get("product") or ""
        rxn_key  = (intent.get("reaction") or intent.get("problem_type") or "").lower().strip()
        if not reactant or not product:
            r0, p0 = _RXN_MOLECULE_MAP.get(rxn_key, ("", ""))
            if not reactant: reactant = r0
            if not product:  product  = p0

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
        steps = []
        for step in result.steps:
            if isinstance(step, dict):
                lhs = " + ".join(step.get("r") or [])
                rhs = " + ".join(step.get("p") or [])
                if lhs or rhs:
                    steps.append(f"{lhs} -> {rhs}")

        coads = [(a, b) for a, b in result.coads
                 if isinstance(a, str) and isinstance(b, str)]

        return {
            "steps":        steps,
            "intermediates": result.intermediates,
            "coads":         coads,
            "ts":            result.ts_candidates,
            "_name":         result.name,
        }
    except Exception:
        return {}


# ========================= Seeds（原来的简单反应集，作为最终兜底） =========================
def _seed_for(reaction: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    r = (reaction or "").upper()
    if "CO2RR" in r:
        steps = ["CO2(g) + * -> CO2*", "CO2* + H+ + e- -> COOH*", "COOH* + H+ + e- -> CO* + H2O(g) + *", "CO* -> CO(g) + *"]
        inter = ["*","CO2*","COOH*","CO*","H*","OH*","CO(g)","H2O(g)"]
        pairs = [("CO*","H*"),("CO*","OH*")]
        return steps, inter, sorted({tuple(sorted(p)) for p in pairs})
    if "MSR" in r:
        steps = ["CH4* + * -> CH3* + H*", "H2O* + * -> OH* + H*", "C* + O* -> CO*", "CO* -> CO(g) + *", "H* + H* -> H2(g) + *"]
        inter = ["CH4*","CH3*","H*","H2O*","OH*","C*","O*","CO*","CO(g)","H2(g)","*"]
        pairs = [("CO*","O*"),("CO*","OH*"),("CH3*","H*")]
        return steps, inter, sorted({tuple(sorted(p)) for p in pairs})
    if any(k in r for k in ["DEHYDROG", "C4H10", "BUTANE", "C-H ACTIV", "ALKANE"]):
        steps = ["C4H10(g) + * -> C4H10*", "C4H10* + * -> C4H9* + H*", "C4H9* + * -> C4H8* + H*", "C4H8* -> C4H8(g) + *", "H* + H* -> H2(g) + 2*"]
        inter = ["C4H10(g)","C4H10*","C4H9*","C4H8*","C4H8(g)","H*","H2(g)","*"]
        pairs = [("C4H10*","H*"),("C4H9*","H*"),("C4H8*","H*")]
        return steps, inter, sorted({tuple(sorted(p)) for p in pairs})
    # HER 默认
    steps = ["H+ + e- + * -> H*", "H* + H+ + e- -> H2(g) + *", "H* + H* -> H2(g) + 2*"]
    inter = ["*","H*","H2(g)","H2O*","OH*"]
    pairs = [("H*","H*"),("H*","OH*")]
    return steps, inter, sorted({tuple(sorted(p)) for p in pairs})

# ========================= Hypothesis → graph 适配 =========================
def _graph_from_hyp(hyp: Dict[str, Any]) -> Dict[str, Any]:
    """把 hypothesis 的结构化输出转成 external_graph 形态。"""
    if not isinstance(hyp, dict):
        return {}
    rn  = hyp.get("reaction_network") or hyp.get("steps") or []
    inter = hyp.get("intermediates") or []
    ts    = hyp.get("ts_targets") or hyp.get("ts") or []
    coads = hyp.get("coadsorption_targets") or hyp.get("coads_pairs") or hyp.get("coads") or []
    out: Dict[str, Any] = {}
    if rn:    out["reaction_network"] = rn
    if inter: out["intermediates"] = inter
    if ts:    out["ts_edges"] = ts
    if coads: out["coads_pairs"] = coads
    return out

# ========================= Cleaning & EC-safe balance =========================
def _normalize_species(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^\*([A-Za-z].*)$", r"\1*", s)  # *CO -> CO*
    s = s.replace("**","*")
    return s

_IGNORE_TOKENS = {"*", "e-", "e⁻", "e–"}

def _neutral_expr(expr: str) -> str:
    # map H+->H, OH- -> OH, ignore (g)/(l)/(aq)
    x = expr.replace("(g)","").replace("(l)","").replace("(aq)","")
    x = x.replace("H+","H").replace("OH-","OH")
    return x

def _elem_count(expr: str) -> Dict[str,int]:
    expr = _neutral_expr(expr).replace("*","")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", expr)
    out: Dict[str,int] = {}
    for el, n in tokens:
        out[el] = out.get(el,0) + (int(n or 1))
    return out

def _mass_balanced(step: str) -> bool:
    """Electrochem-safe: allow steps with e-/H+/OH- by neutral mapping; '*' ignored."""
    if "->" not in step: return False
    L, R = [t.strip() for t in step.split("->",1)]
    Lp = [p.strip() for p in L.split("+")]
    Rp = [p.strip() for p in R.split("+")]
    # relaxed if explicit e-/H+/OH-
    relaxed = any(p in {"e-","e⁻","e–"} for p in Lp+Rp) or any(x in step for x in ["H+","OH-"])
    def side_count(parts):
        tot: Dict[str,int] = {}
        for sp in parts:
            spn = _normalize_species(sp)
            if spn in _IGNORE_TOKENS: continue
            c = _elem_count(spn)
            for k,v in c.items(): tot[k] = tot.get(k,0)+v
        return tot
    try:
        ok = side_count(Lp) == side_count(Rp)
        return ok or relaxed
    except Exception:
        return relaxed

def _ok_intermediate(s: str, strict: bool = True) -> bool:
    s = _normalize_species(s)
    if not (s.endswith("*") or s.endswith("(g)") or s=="*"): return False
    if strict and any(ch in s for ch in ["^","+","--","++"]): return False
    return True

def _uniq_limit(items: List[Any], limit: int) -> List[Any]:
    out, seen = [], set()
    for x in items:
        k = json.dumps(x, sort_keys=True) if not isinstance(x, str) else x
        if k not in seen:
            out.append(x); seen.add(k)
            if len(out) >= limit: break
    return out

def _clean_all(steps: List[str], inter: List[str], limits: Dict[str,int], strict: bool) -> Tuple[List[str], List[str]]:
    steps = [_normalize_species(s) for s in (steps or []) if isinstance(s, str)]
    steps = [s for s in steps if "->" in s and _mass_balanced(s)]
    inter = [_normalize_species(s) for s in (inter or []) if isinstance(s, str)]
    inter = [s for s in inter if _ok_intermediate(s, strict=strict)]
    steps = _uniq_limit(steps, limits["ts"])
    inter = _uniq_limit(inter, limits["inter"])
    return steps, inter

# ========================= LLM generate + score =========================
async def _llm_generate(intent: Dict[str, Any], hint: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    sys = (
        "You are a senior researcher in heterogeneous (electro)catalysis. "
        "Return STRICT JSON with keys: steps, intermediates, coads, ts. "
        "Use '*' for adsorbates, '(g)' for gas. Mass-balance elements ignoring electrons and explicit charges."
    )
    user = {"intent": intent, "hint": hint, "seed_hint": seed}
    raw = await chatgpt_call(
        [{"role":"system","content":sys},{"role":"user","content":json.dumps(user, ensure_ascii=False)}],
        model="gpt-4o-mini", temperature=0.1, max_tokens=1800
    )
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    return {
        "steps": data.get("steps", []),
        "intermediates": data.get("intermediates", []),
        "coads": data.get("coads", []),
        "ts": data.get("ts", []),
    }

async def _llm_confidence(intent: Dict[str, Any], steps: List[str], inter: List[str]) -> float:
    try:
        prompt = {
            "intent": intent,
            "steps": steps[:20],
            "intermediates": inter[:30],
            "question": "Rate typicality/reasonableness 0.0-1.0 (float only)."
        }
        raw = await chatgpt_call(
            [{"role":"system","content":"Return ONLY a float between 0 and 1."},
             {"role":"user","content":json.dumps(prompt, ensure_ascii=False)}],
            model="gpt-4o-mini", temperature=0.0, max_tokens=10
        )
        m = re.search(r"(?:0?\.\d+|1(?:\.0+)?)", raw)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0

# ========================= Build tasks =========================
def _build_tasks(intent: Dict[str, Any],
                 steps: List[str], inter: List[str],
                 coads_pairs: List[Tuple[str,str]], ts_edges: List[str],
                 session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    sys_info = intent.get("system") or {}
    catalyst = (sys_info.get("catalyst") or sys_info.get("material") or
                intent.get("catalyst") or "Pt")
    facet    = sys_info.get("facet") or intent.get("facet") or "111"
    tasks: List[Dict[str, Any]] = []
    tid = 1
    # maps group-number → list of task IDs in that group (for dependency wiring)
    _group_ids: Dict[int, List[int]] = {}

    def _field(key, label, ftype="text", value="", **kw):
        d = {"key": key, "label": label, "type": ftype, "value": value}
        d.update({k:v for k,v in kw.items() if v is not None})
        return d

    def _task(section, name, agent, desc, form=None, payload=None, group=0,
              endpoint=None, depends_on_groups: List[int] = None):
        nonlocal tid
        # Build depends_on from previous groups
        dep_ids: List[int] = []
        for g in (depends_on_groups or []):
            dep_ids.extend(_group_ids.get(g, []))
        t = {
            "id": tid, "section": section, "name": name, "agent": agent, "description": desc,
            "params": {"form": form or [], "payload": payload or {}, "endpoint": endpoint or ""},
            "depends_on": dep_ids,
            "meta": {"parallel_group": group, "action_endpoint": endpoint,
                     "project": f"session-{session_id}" if session_id else None,
                     "run_id": session_id}
        }
        _group_ids.setdefault(group, []).append(tid)
        tid += 1
        tasks.append(t)

    # Detect reaction type from intent for context-aware task generation
    task_text = (intent.get("task") or intent.get("reaction") or "").lower()
    tags_text = " ".join(intent.get("tags") or []).lower()
    area_text = (intent.get("area") or "").lower()
    is_electrochemical = any(k in (area_text + tags_text + task_text)
                             for k in ["electro", "pcet", "potential", "gcdft", "gc-dft"])
    prob_type = (intent.get("problem_type") or "").upper()
    is_dehydrog = ("DEHYDROG" in prob_type) or any(k in (task_text + tags_text)
                      for k in ["dehydrog", "c-h activ", "alkane", "butane", "propane", "ch activ"])
    needs_gcdft = is_electrochemical or is_dehydrog  # dehydrog can also be electrochemically promoted

    # Extract actual molecules from intent for adsorbate list
    mol_list = sys_info.get("molecule") or []
    if not isinstance(mol_list, list):
        mol_list = [mol_list] if mol_list else []
    rn = intent.get("reaction_network") or {}
    rn_intermediates = rn.get("intermediates") or []

    # ── Infer full gas-phase molecule list ──────────────────────────────────
    # Collects from: mol_list, reaction-network intermediates, dehydrogenation chain
    def _clean_mol(s: str) -> str:
        return s.replace("*","").replace("(g)","").replace("(l)","").strip()

    # Canonical isomer expansion: C4H9 → both radicals; C4H8 / butene → both alkene forms
    _ISOMER_MAP: Dict[str, List[str]] = {
        "C4H9":    ["C4H9-1", "C4H9-2"],
        "C4H8":    ["C4H8-1", "C4H8-2"],
        "BUTENE":  ["C4H8-1", "C4H8-2"],
        "1-BUTENE":["C4H8-1"],
        "2-BUTENE":["C4H8-2"],
        "1-C4H9":  ["C4H9-1"],
        "2-C4H9":  ["C4H9-2"],
    }
    def _expand_isomers(name: str) -> List[str]:
        key = _clean_mol(name).upper()
        return _ISOMER_MAP.get(key, [_clean_mol(name)])

    # Seed from mol_list (explicit intent)
    _mol_seed: List[str] = []
    for m in mol_list:
        _mol_seed += _expand_isomers(m)

    # Pull gas-phase species from reaction-network intermediates (strip *)
    for m in (inter + rn_intermediates):
        mc = _clean_mol(m)
        if mc and mc not in ("H", "H2", "*", ""):
            _mol_seed += _expand_isomers(mc)

    # For dehydrogenation: if only reactant given, infer the full dehydrogenation chain
    if is_dehydrog:
        # Typical C4 chain: C4H10 → C4H9 (2 forms) → C4H8 (2 forms)
        _c4_chain = ["C4H10", "C4H9-1", "C4H9-2", "C4H8-1", "C4H8-2"]
        has_c4 = any("C4H" in m.upper() for m in _mol_seed)
        if has_c4 or any("C4H" in m.upper() for m in mol_list):
            for m in _c4_chain:
                if m not in _mol_seed:
                    _mol_seed.append(m)
        # General CnHm chain inference from first molecule
        if mol_list:
            rct = _clean_mol(mol_list[0]).upper()
            match = re.match(r"([A-Z]+)(\d+)H(\d+)", rct)
            if match and "C4" not in rct:  # already handled above
                elem, cn, hn = match.group(1), int(match.group(2)), int(match.group(3))
                for h in range(hn, hn - 4, -1):
                    cand = f"{elem}{cn}H{h}"
                    if cand not in _mol_seed:
                        _mol_seed.append(cand)

    # Deduplicate, skip trivial/surface-only species
    _skip = {"H", "H2", "2H", "2*", "*", ""}
    seen_mols: set = set()
    mol_list_full: List[str] = []
    for m in _mol_seed:
        mc = _clean_mol(m)
        if mc and mc.upper() not in _skip and mc not in seen_mols:
            seen_mols.add(mc)
            mol_list_full.append(mc)

    # Fall back to original mol_list if derivation gave nothing
    if not mol_list_full:
        mol_list_full = [_clean_mol(m) for m in mol_list if _clean_mol(m)]

    # ─────────────────────────────────────────────────────────────────────────
    # Build adsorbates list from inter (from hypothesis graph) + fallback
    # ─────────────────────────────────────────────────────────────────────────
    adsorbates = [s for s in inter if s.endswith("*") and s != "*"]
    if not adsorbates:
        fallback = []
        for mol in mol_list:
            ads = mol if mol.endswith("*") else mol + "*"
            if ads not in fallback: fallback.append(ads)
        for sp in rn_intermediates:
            ads = sp if sp.endswith("*") else sp + "*"
            if ads not in fallback and sp not in ("*", ""): fallback.append(ads)
        if "H*" not in fallback: fallback.append("H*")
        adsorbates = fallback or ["H*"]

    # Auto-generate TS edges for dehydrogenation if none provided
    if is_dehydrog and not ts_edges and mol_list:
        reactant = mol_list[0] if mol_list else "C4H10"
        product  = mol_list[-1] if len(mol_list) > 1 else "C4H8"
        m = re.match(r"([A-Z][a-z]?)(\d+)H(\d+)", reactant)
        if m:
            elem, cn, hn = m.group(1), int(m.group(2)), int(m.group(3))
            inter1 = f"{elem}{cn}H{hn-1}*"
        else:
            inter1 = "C4H9*"
        ts_edges = [
            f"{reactant}* -> {inter1} + H*",
            f"{inter1} -> {product}* + H*",
            "H* + H* -> H2(g) + 2*",
        ]

    reaction_label = (intent.get("problem_type") or intent.get("task") or
                      (f"{mol_list[0]}→{mol_list[-1]}" if len(mol_list) >= 2 else ""))
    t_default = 500.0 if is_dehydrog else 298.15

    # =========================================================================
    # SECTION 1: Surface  — ONE compound task with 4 sub-steps in the UI
    # sub-step 1: build model  sub-step 2: generate params  sub-step 3: submit HPC  sub-step 4: retrieve
    # =========================================================================
    _task(
        "1. Surface",
        f"1.1 {catalyst}({facet}) surface",
        "structure.build_surface",
        (f"Build {catalyst}({facet}) {4}×{4}×{3} slab with FixAtoms on bottom layer, "
         f"relax with VASP (ISIF=2, EDIFFG=-0.02 eV/Å), retrieve CONTCAR + OUTCAR."),
        form=[
            _field("element","Element","text", catalyst,
                   help="e.g. Pt, Ag, Cu — crystal system auto-detected"),
            _field("surface_type","Surface","select", facet,
                   options=["111","100","110","211","0001","10m10","443"]),
            _field("nx","Nx","number",4, step=1, min_value=1, max_value=8),
            _field("ny","Ny","number",4, step=1, min_value=1, max_value=8),
            _field("nlayers","Layers","number",3, step=1, min_value=2, max_value=8),
            _field("vacuum","Vacuum (Å)","number",10.0, step=1.0, min_value=8.0, max_value=25.0),
            _field("fix_bottom","Fix bottom layer","checkbox",True),
            _field("encut","ENCUT (eV)","number",400, step=50),
            _field("kpoints","K-points","text","4x4x1"),
            _field("ediffg","EDIFFG (eV/Å)","number",-0.02),
        ],
        payload={"element": catalyst, "surface_type": facet, "nx": 4, "ny": 4,
                 "nlayers": 3, "vacuum": 10.0},
        group=1,
        endpoint="/agent/structure/build_surface",
        depends_on_groups=[],
    )
    # Embed calc metadata for the UI 4-step card
    tasks[-1]["calc_types"] = ["geo"]
    tasks[-1]["retrieve_files"] = ["CONTCAR", "OUTCAR", "stdout", "OSZICAR"]

    # =========================================================================
    # SECTION 2: Gas-phase Molecules
    # Hierarchical: 2.1 C4H10 | 2.2 C4H8: 2.2.1 1-butene  2.2.2 2-butene
    #                          | 2.3 C4H9: 2.3.1 primary    2.3.2 secondary
    # Each task: PubChem SMILES fetch → geo opt → freq → ZPE + Gibbs
    # =========================================================================
    # Group isomers: base formula → [(isomer_name, smiles), ...]
    _MOL_BASE: Dict[str, str] = {
        "C4H10": "C4H10", "N-BUTANE": "C4H10",
        "C4H9-1": "C4H9", "C4H9-2": "C4H9",
        "C4H8-1": "C4H8", "C4H8-2": "C4H8",
        "C4H6": "C4H6",
    }
    _MOL_LABELS: Dict[str, str] = {
        "C4H10":  "n-butane (CCCC)",
        "C4H9-1": "1-butyl radical (CCC[CH2])",
        "C4H9-2": "2-butyl radical (CC[CH]C)",
        "C4H8-1": "1-butene (C=CCC)",
        "C4H8-2": "2-butene (CC=CC)",
        "C4H6":   "1,3-butadiene (C=CC=C)",
    }

    # Build grouped structure: {base: [(mol_clean, smiles, label), ...]}
    _mol_groups: Dict[str, List[tuple]] = {}
    for m in mol_list_full:
        smiles = (_MOL_SMILES.get(m) or _MOL_SMILES.get(m.upper())
                  or _MOL_SMILES.get(m.replace("-1","").replace("-2",""))
                  or m)
        base = _MOL_BASE.get(m.upper(), m)
        label = _MOL_LABELS.get(m, m)
        _mol_groups.setdefault(base, []).append((m, smiles, label))

    _mol_group_start = 20
    mol_gas_group_ids: Dict[str, int] = {}
    _mg_counter = _mol_group_start
    _sec2_idx = 0
    for _base, _isomers in _mol_groups.items():
        _sec2_idx += 1
        for _iso_idx, (mol_clean, smiles, mol_label) in enumerate(_isomers):
            # Numbering: single isomer → "2.1 C4H10", multiple → "2.2.1 1-butene"
            if len(_isomers) == 1:
                task_name = f"2.{_sec2_idx} {_base} — {mol_label}"
            else:
                task_name = f"2.{_sec2_idx}.{_iso_idx+1} {mol_label}"
            mg = _mg_counter
            _mg_counter += 1
            _task(
                "2. Gas-phase Molecules",
                task_name,
                "structure.build_molecule",
                (f"PubChem fetch via SMILES ({smiles}). "
                 f"Geo opt (EDIFFG=-0.01 eV/Å) → freq (IBRION=5) "
                 f"→ ZPE + entropy + Gibbs free energy at T={t_default:.0f} K."),
                form=[
                    _field("smiles","SMILES","text", smiles,
                           help=f"SMILES for {mol_clean}"),
                    _field("label","Label","text", mol_clean),
                    _field("cell_size","Cell size (Å)","number",20.0),
                    _field("ediffg","EDIFFG (eV/Å)","number",-0.01),
                    _field("temperature","T (K)","number",t_default, step=50),
                ],
                payload={"smiles": smiles, "label": mol_clean, "cell_size": 20.0},
                group=mg,
                endpoint="/agent/structure/build_molecule",
                depends_on_groups=[],
            )
            tasks[-1]["calc_types"] = ["geo", "freq"]
            tasks[-1]["retrieve_files"] = ["CONTCAR", "OUTCAR", "stdout"]
            mol_gas_group_ids[mol_clean] = mg

    # =========================================================================
    # SECTION 3: Adsorption — grouped by species, multiple configs per species
    # 3.1 C4H10 → 3.1.1 conf1, 3.1.2 conf2 ...
    # 3.2 C4H9-1, 3.3 C4H9-2, 3.4 C4H8-1, 3.5 C4H8-2
    # Each config: geo opt → freq → (lsol for electrochemical)
    # =========================================================================
    _ads_start = 40
    ads_opt_group_ids: Dict[str, int] = {}
    _n_configs_per_sp = 3     # default configs per species

    # Build ordered adsorbate list: gas-phase molecules are the surface adsorbates
    _ads_species: List[str] = list(mol_list_full)  # already deduplicated + isomer-expanded
    # Add hypothesis-derived adsorbates only if NOT already covered by an expanded isomer
    _base_set = {_MOL_BASE.get(m.upper(), m) for m in _ads_species}
    for sp in adsorbates:
        sp_c = sp.rstrip("*")
        if not sp_c or sp_c.upper() in ("H", "H2", ""):
            continue
        sp_base = _MOL_BASE.get(sp_c.upper(), sp_c)
        # Skip if an isomer-expanded version is already in the list
        if sp_c not in _ads_species and sp_base not in _base_set:
            _ads_species.append(sp_c)
    # Always include H* for dehydrogenation (Langmuir-Hinshelwood H recombination)
    if is_dehydrog and "H" not in _ads_species:
        _ads_species.append("H")

    _ag_counter = _ads_start
    for si, sp_clean in enumerate(_ads_species[:8]):
        for ci in range(_n_configs_per_sp):
            ag = _ag_counter
            _ag_counter += 1
            conf_label = f"conf{ci+1}"
            lsol_types = (["geo", "freq", "lsol"] if is_electrochemical else ["geo", "freq"])
            _task(
                "3. Adsorption",
                f"3.{si+1}.{ci+1} {sp_clean}*/Pt({facet}) — {conf_label}",
                "structure.generate_configs",
                (f"Place {sp_clean} on {catalyst}({facet}) config {ci+1}, "
                 f"relax (EDIFFG=-0.02 eV/Å), freq (IBRION=5) → E_ads + Gibbs."
                 + (f" Then VASPsol (lsol) for implicit solvation correction." if is_electrochemical else "")),
                form=[
                    _field("adsorbate","Adsorbate","text", sp_clean),
                    _field("config_index","Config index","number", ci,
                           step=1, min_value=0, max_value=7),
                    _field("height","Height (Å)","number",2.0, step=0.1),
                    _field("sites","Sites","select","all",
                           options=["all","top","bridge","hollow_fcc","hollow_hcp"]),
                    _field("ediffg","EDIFFG (eV/Å)","number",-0.02),
                    _field("temperature","T (K)","number",t_default, step=50),
                ] + ([_field("lsol","VASPsol (lsol)","checkbox",True)] if is_electrochemical else []),
                payload={"adsorbate": sp_clean, "max_configs": _n_configs_per_sp,
                         "element": catalyst, "surface_type": facet,
                         "config_index": ci},
                group=ag,
                endpoint="/agent/structure/generate_configs",
                depends_on_groups=[1],
            )
            tasks[-1]["calc_types"] = lsol_types
            tasks[-1]["retrieve_files"] = ["CONTCAR", "OUTCAR", "stdout"]
            tasks[-1]["species"] = sp_clean
            tasks[-1]["config_index"] = ci
        ads_opt_group_ids[sp_clean] = _ads_start + si * _n_configs_per_sp

    # =========================================================================
    # SECTION 4: NEB / Transition States
    # =========================================================================
    _neb_start = 130
    for ni, s in enumerate(ts_edges[:6]):
        ng = _neb_start + ni
        parts = re.split(r"\s*->\s*", s)
        is_str = parts[0].strip() if parts else s
        fs_str = parts[1].strip() if len(parts) > 1 else ""
        _task(
            "4. NEB",
            f"4.{ni+1} TS: {s[:55]}",
            "adsorption.co",
            (f"Build IS ({is_str}) + FS ({fs_str}), run CI-NEB (7 images, LCLIMB=True, "
             f"SPRING=-5.0), validate TS with IBRION=5 (expect 1 imaginary mode)."),
            form=[
                _field("IS","IS","text", is_str),
                _field("FS","FS","text", fs_str),
                _field("n_images","Images","number",7, step=1, min_value=3, max_value=15),
                _field("climbing","CI-NEB","checkbox",True),
                _field("ediffg","EDIFFG (eV/Å)","number",-0.05),
                _field("kpoints","K-points","text","4x4x1"),
            ],
            payload={"step": s, "n_images": 7, "lclimb": True,
                     "element": catalyst, "surface_type": facet},
            group=ng,
            endpoint="/agent/adsorption.co",
            depends_on_groups=list(ads_opt_group_ids.values())[:4] or [_ads_start],
        )
        tasks[-1]["calc_types"] = ["neb", "freq"]
        tasks[-1]["retrieve_files"] = ["CONTCAR", "OUTCAR", "stdout", "MODECAR"]

    # =========================================================================
    # SECTION 5: Electronic Structure
    # Generated dynamically based on intent["electronic_calcs"] list.
    # If not specified, show the full menu of all available calc types.
    # =========================================================================
    _elec_start = 160

    # Map calc_type → (display name, description, retrieve files, extra form fields)
    _ELEC_MENU: Dict[str, tuple] = {
        "static": (
            "5.0 Static SCF (prerequisite)",
            "SCF with LWAVE=True + LCHARG=True. Writes WAVECAR+CHGCAR for all follow-up "
            "electronic calculations (DOS, band, ELF, work function, COHP).",
            ["WAVECAR", "CHGCAR", "OUTCAR"],
            [_field("kpoints","K-points","text","8x8x1"),
             _field("encut","ENCUT (eV)","number",400, step=50)],
        ),
        "dos": (
            "5.1 DOS — density of states",
            "Non-SCF (ICHARG=11) with dense k-mesh (12×12×1). LORBIT=11 for lm-decomposed "
            "PDOS. ISMEAR=-5 (tetrahedron). Extracts d-band center. Requires SCF CHGCAR.",
            ["DOSCAR", "OUTCAR", "CONTCAR"],
            [_field("kpoints","K-points","text","12x12x1"),
             _field("nedos","NEDOS","number",2000, step=200),
             _field("sigma","SIGMA","number",0.05, step=0.01)],
        ),
        "pdos": (
            "5.1b PDOS — orbital-projected DOS",
            "Same as DOS with emphasis on site-projected d/p/s decomposition. "
            "Useful for d-band center analysis and adsorbate orbital hybridization.",
            ["DOSCAR", "OUTCAR"],
            [_field("kpoints","K-points","text","12x12x1"),
             _field("nedos","NEDOS","number",2000, step=200)],
        ),
        "band": (
            "5.2 Band structure",
            "Non-SCF (ICHARG=11) along high-symmetry k-path (ASE bandpath, ISMEAR=0). "
            "Identifies band gap and d-band dispersion. Requires SCF CHGCAR.",
            ["EIGENVAL", "KPOINTS", "OUTCAR"],
            [_field("nkpoints","K-points per path segment","number",60, step=10)],
        ),
        "elf": (
            "5.3 ELF — electron localization function",
            "LELF=True, NCORE=1 (VASP hard requirement). Visualize in VESTA at isosurface "
            "ELF=0.75 to see lone pairs and covalent bonds. Reads WAVECAR from SCF.",
            ["ELFCAR", "OUTCAR"],
            [_field("kpoints","K-points","text","8x8x1")],
        ),
        "bader": (
            "5.4 Bader charge analysis",
            "LAECHG=True writes AECCAR0+AECCAR2. Post-process: chgsum.pl + bader binary "
            "(Henkelman group). Gives charge transfer per atom. ENCUT=520, LREAL=False.",
            ["CHGCAR", "AECCAR0", "AECCAR2", "ACF.dat"],
            [_field("encut","ENCUT (eV)","number",520, step=50),
             _field("kpoints","K-points","text","8x8x1")],
        ),
        "cdd": (
            "5.5 Charge density difference",
            "Three static calcs: AB (combined), A (slab), B (adsorbate). "
            "Δρ = ρ_AB − ρ_A − ρ_B. Yellow/blue isosurfaces show charge accumulation/depletion.",
            ["CHGCAR", "delta_rho.npy"],
            [_field("kpoints","K-points","text","8x8x1"),
             _field("n_slab","Number of slab atoms","number",0, step=1,
                    help="Set to split AB system into slab (A) and adsorbate (B)")],
        ),
        "work_function": (
            "5.6 Work function",
            "LVHAR=True writes LOCPOT. LDIPOL=True/IDIPOL=3 for dipole correction on "
            "asymmetric slabs. φ = E_vacuum − E_Fermi (planar average of LOCPOT).",
            ["LOCPOT", "OUTCAR", "locpot_avg.dat"],
            [_field("kpoints","K-points","text","8x8x1")],
        ),
        "cohp": (
            "5.7 COHP bonding analysis (LOBSTER)",
            "ISYM=-1 (mandatory for LOBSTER), LWAVE=True, LORBIT=11. Then run LOBSTER. "
            "−ICOHP > 0 = bonding; < 0 = antibonding. Identify adsorbate–surface bonds.",
            ["WAVECAR", "DOSCAR", "COHPCAR.lobster", "lobsterin"],
            [_field("kpoints","K-points","text","8x8x1"),
             _field("atom_pairs","Atom pairs for COHP","text","",
                    help="e.g. '1 5' for bond between atom 1 and atom 5")],
        ),
    }

    # Determine which electronic calcs to generate tasks for
    elec_requested: List[str] = intent.get("electronic_calcs") or []
    # If nothing detected, show all available as optional tasks
    if not elec_requested:
        elec_requested = list(_ELEC_MENU.keys())

    for ei, calc_id in enumerate(elec_requested):
        if calc_id not in _ELEC_MENU:
            continue
        ename, edesc, efiles, eform = _ELEC_MENU[calc_id]
        eg = _elec_start + ei
        _task(
            "5. Electronic Structure",
            ename,
            f"electronic.{calc_id}",
            edesc + " Select CONTCAR from a completed surface or adsorption task.",
            form=[
                _field("source_task","Source calc (task name)","text","",
                       help="Paste the task name whose CONTCAR to use"),
            ] + eform,
            payload={"calc_type": calc_id},
            group=eg,
            endpoint="/agent/generate_script",
            depends_on_groups=[],   # user picks source interactively
        )
        tasks[-1]["calc_types"] = [calc_id]
        tasks[-1]["retrieve_files"] = efiles
        tasks[-1]["electronic_calc_id"] = calc_id

    # =========================================================================
    # POST-ANALYSIS
    # =========================================================================
    _post_deps = list(ads_opt_group_ids.values())[:4]
    _task(
        "Post-analysis", "Assemble ΔG diagram + microkinetics", "post.thermo",
        f"Apply ZPE+entropy corrections, build ΔG free-energy diagram at T={t_default:.0f} K. "
        f"Run mean-field microkinetics: TOF, selectivity, coverages.",
        form=[
            _field("temperature","Temperature (K)","number",t_default, step=10),
            _field("potential","Potential (V vs RHE)","number",0.0, step=0.05),
            _field("reference","Reference","select","RHE", options=["RHE","SHE","Ag/AgCl"]),
            _field("microkinetics","Run microkinetics","checkbox",True),
            _field("output_xlsx","Save to Excel","text",
                   f"{catalyst}_{facet}_{reaction_label.replace('→','_')}_results.xlsx"),
        ],
        payload={"reaction": reaction_label},
        group=199, endpoint="/chat/qa/free_energy",
        depends_on_groups=_post_deps if _post_deps else []
    )

    # QA checks (run independently)
    _task(
        "QA", f"Surface stability — {catalyst}({facet})", "qa.surface",
        f"Check for known surface reconstructions, oxide formation, facet stability "
        f"(e.g. Ag surface oxide under O-containing adsorbates).",
        form=[_field("material","Material","text",catalyst),
              _field("facet","Facet","text",facet)],
        payload={"material": catalyst, "facet": facet},
        group=98, endpoint="/chat/qa/surface",
        depends_on_groups=[]
    )
    _task(
        "QA", "Functional recommendation", "qa.functional",
        f"Check if PBE-GGA is appropriate for {catalyst}({facet}). "
        f"Flag vdW/DFT+U needs (e.g. vdW for large molecule physisorption, +U for d-states).",
        form=[_field("system","System","text", f"{catalyst}({facet}) {reaction_label}")],
        payload={"system": f"{catalyst}({facet}) {reaction_label}"},
        group=98, endpoint="/chat/qa/functional",
        depends_on_groups=[]
    )

    return tasks

# ========================= /chat/plan =========================
@router.post("/chat/plan")
async def api_plan(request: Request):
    body = await request.json()
    USE_SEED_POLICY = body.get("use_seed_policy", DEFAULTS["USE_SEED_POLICY"])
    CONF_THRESHOLD  = float(body.get("conf_threshold", DEFAULTS["CONF_THRESHOLD"]))
    LIMITS          = body.get("limits", DEFAULTS["LIMITS"])
    STRICT          = bool(body.get("strict", DEFAULTS["STRICT"]))

    session_id = body.get("session_id")
    intent_in  = _extract_intent(body)

    knowledge  = body.get("knowledge") or {}
    history    = body.get("history") or []
    query      = body.get("query") or body.get("text") or ""

    # hypothesis 可能是 str(md) 或 dict(JSON)
    raw_hyp = body.get("hypothesis")
    if isinstance(raw_hyp, dict):
        hyp_dict = raw_hyp
    elif isinstance(raw_hyp, str):
        try:
            hyp_dict = json.loads(raw_hyp)
            if not isinstance(hyp_dict, dict):
                hyp_dict = {}
        except Exception:
            hyp_dict = {}
    else:
        hyp_dict = {}

    # 1) RAG 上下文
    rag_ctx = await rag_context(query, session_id=session_id, top_k=8)

    # 2) external graph 优先：显式传入 > hypothesis 中的结构化 > 无（后面 LLM 生成）
    external_graph: Dict[str, Any] = body.get("graph") or _graph_from_hyp(hyp_dict) or {}

    # 3) Dynamic mechanism seed from builder
    mech_seed = await _get_mech_seed_builder(intent_in, query, session_id)
    mech_name = mech_seed.pop("_name", "") if mech_seed else ""

    # 4) 传统兜底 seed
    seed_steps, seed_inter, seed_coads = _seed_for(intent_in.get("problem_type",""))

    # 5) 拿到候选图（优先 external，若没有则让 LLM 生成；把 RAG/knowledge/history + mech_seed 合并进 hint）
    if external_graph:
        steps_raw = [s if isinstance(s, str) else "" for s in external_graph.get("reaction_network", [])]
        inter_raw = [s for s in external_graph.get("intermediates", [])]
        coads_raw = external_graph.get("coads_pairs", [])
        ts_raw    = [s for s in external_graph.get("ts_edges", [])]
        # 补强：如果 external 很空而 mech_seed 存在，用 seed 合并增强
        if mech_seed and (len(steps_raw) < 2 or len(inter_raw) < 3):
            steps_raw = list(dict.fromkeys((mech_seed.get("steps") or []) + steps_raw))
            inter_raw = list(dict.fromkeys((mech_seed.get("intermediates") or []) + inter_raw))
            coads_raw = list(dict.fromkeys((mech_seed.get("coads") or []) + _to_list(coads_raw)))
    else:
        hint_parts = []
        if knowledge: hint_parts.append(f"[knowledge]{json.dumps(knowledge, ensure_ascii=False)[:1200]}")
        if history:   hint_parts.append(f"[history]{json.dumps(history, ensure_ascii=False)[:1200]}")
        if rag_ctx:   hint_parts.append(f"[rag]{rag_ctx[:1200]}")
        if mech_seed: hint_parts.append(f"[mechanism]{json.dumps(mech_seed, ensure_ascii=False)[:1200]}")
        hint = "\n".join(hint_parts)

        try:
            # NEW: 将 mech_seed 作为 seed_hint 传入 LLM
            llm_out = await _llm_generate(intent_in, hint, {"seed": mech_seed or {"steps": seed_steps, "intermediates": seed_inter}})
        except Exception:
            base_seed = mech_seed if mech_seed else {"steps": seed_steps, "intermediates": seed_inter, "coads": seed_coads, "ts": []}
            llm_out = {"steps": base_seed.get("steps", []), "intermediates": base_seed.get("intermediates", []),
                       "coads": base_seed.get("coads", []), "ts": []}

        steps_raw = (llm_out.get("steps") or []) + (llm_out.get("ts") or [])
        inter_raw = (llm_out.get("intermediates") or [])
        coads_raw = llm_out.get("coads") or []
        ts_raw    = llm_out.get("ts") or []

        # 若 LLM 结果很弱，再并入传统 seed
        if (len(steps_raw) < 2 or len(inter_raw) < 3):
            steps_raw = list(dict.fromkeys((mech_seed.get("steps") if mech_seed else seed_steps) + steps_raw))
            inter_raw = list(dict.fromkeys((mech_seed.get("intermediates") if mech_seed else seed_inter) + inter_raw))
            if not coads_raw:
                coads_raw = (mech_seed.get("coads") if mech_seed else seed_coads)

    # 6) 清洗 + 限流
    steps, inter = _clean_all(steps_raw, inter_raw, LIMITS, STRICT)

    # 7) 置信度（外部图不打分；非 always 模式才打）
    conf = 0.0
    if USE_SEED_POLICY != "always" and not external_graph:
        conf = await _llm_confidence(intent_in, steps, inter)

    # 8) 低置信度/强制：轻度并入传统 seed（外部图已定时跳过，避免 HER seed 污染）
    if not external_graph and (USE_SEED_POLICY == "always" or (USE_SEED_POLICY == "auto" and conf < CONF_THRESHOLD)):
        steps = _uniq_limit(seed_steps + [s for s in steps if s not in seed_steps], LIMITS["ts"])
        inter = _uniq_limit(seed_inter + [i for i in inter if i not in seed_inter], LIMITS["inter"])

    # 9) 共吸附
    ads = [s for s in inter if s.endswith("*")]
    if not coads_raw:
        # 如果 mechanism seed 有建议，用它；否则默认与 H* 共吸附
        seed_pairs = mech_seed.get("coads") if mech_seed else []
        if seed_pairs:
            norm_pairs = []
            for pr in seed_pairs:
                a, b = (pr if isinstance(pr, (list,tuple)) else (None,None))
                if isinstance(a,str) and isinstance(b,str) and a.endswith("*") and b.endswith("*"):
                    norm_pairs.append(tuple(sorted(( _normalize_species(a), _normalize_species(b) ))))
            coads_pairs = _uniq_limit(norm_pairs, LIMITS.get("coads", 80))
        else:
            coads_pairs = sorted({tuple(sorted((a, "H*"))) for a in ads if a != "H*"})
    else:
        norm_pairs = []
        for pr in coads_raw:
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                a, b = _normalize_species(pr[0]), _normalize_species(pr[1])
                if a.endswith("*") and b.endswith("*") and a != b:
                    norm_pairs.append(tuple(sorted((a,b))))
        coads_pairs = _uniq_limit(norm_pairs, LIMITS.get("coads", 80))

    # 10) TS 边
    ts_edges = [s for s in steps if "->" in s][:LIMITS["ts"]]
    if external_graph.get("ts_edges"):
        ts_edges = _uniq_limit([_normalize_species(s) for s in external_graph["ts_edges"] if "->" in s], LIMITS["ts"])

    # 11) 构建 tasks（带上 session_id 以便 HPC 页面提交）
    tasks = _build_tasks(intent_in, steps, inter, coads_pairs, ts_edges, session_id=session_id)

    # 12) 组装 workflow（便于前端 HPC 页面）
    workflow = {
        "id": f"wf-{session_id}-{int(datetime.utcnow().timestamp())}" if session_id else f"wf-{int(datetime.utcnow().timestamp())}",
        "title": f"DFT workflow — {(intent_in.get('problem_type') or 'Task')}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "project": f"session-{session_id}" if session_id else None,
        "run_id": session_id,
        "mechanisms": [mech_name] if mech_name else [],
        "summary": {
            "n_steps": len(steps),
            "n_intermediates": len(inter),
            "n_coads": len(coads_pairs),
            "n_ts": len(ts_edges)
        },
        # 前端可直接遍历 tasks，读取每个 task.meta.action_endpoint 并提交
        "tasks": tasks
    }

    result = {
        "ok": True,
        "steps": steps,
        "intermediates": inter,
        "coads": coads_pairs,
        "ts": ts_edges,
        "tasks": tasks,
        "workflow": workflow,  # NEW: 提供给 HPC 页面直接使用
        "confidence": conf,
        "limits": LIMITS,
        "use_seed_policy": USE_SEED_POLICY,
        "used_external_graph": bool(external_graph),
        "rag_context": rag_ctx,
    }

    # 13) 持久化（容错）
    try:
        await _save_artifact(session_id, "plan", result)
        await _save_artifact(session_id, "rxn_network", {
            "elementary_steps": steps,
            "intermediates": inter,
            "coads_pairs": coads_pairs,
            "ts_candidates": ts_edges,
            "mechanisms": [mech_name] if mech_name else [],
        })
    except Exception:
        pass

    return result

# ========================= Execute（示例执行器，安全容错） =========================
try:
    from ..execution.structure_agent       import StructureAgent
    from ..execution.parameters_agent      import ParametersAgent
    from ..execution.hpc_agent             import HPCAgent
    from ..execution.post_analysis_agent   import PostAnalysisAgent
except Exception:
    StructureAgent = object  # type: ignore
    ParametersAgent = object  # type: ignore
    HPCAgent = object  # type: ignore
    PostAnalysisAgent = object  # type: ignore

class PlanManager:
    def __init__(self, cluster: str = "hoffman2", dry_run: bool = False, sync_back: bool = True):
        self.struct_agent = StructureAgent() if StructureAgent != object else None
        self.param_agent  = ParametersAgent() if ParametersAgent != object else None
        self.hpc_agent    = HPCAgent(cluster=cluster, dry_run=dry_run, sync_back=sync_back) if HPCAgent != object else None
        self.post_agent   = PostAnalysisAgent() if PostAnalysisAgent != object else None
        self.dry          = dry_run

    def _exec_task(self, task: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
        # 1) 先拿到 agent；避免 NameError
        agent = (task.get("agent") or "").lower()

        # 2) 安全的子目录名
        safe = f"{int(task.get('id', 0)):02d}_{_slug(task.get('name', 'Task'))}"
        job_dir = workdir / safe
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ----- 元任务 -----
            if agent in {"meta.clarify", "meta.scope"}:
                (job_dir / "meta.json").write_text(
                    json.dumps(task.get("params", {}) or {}, ensure_ascii=False, indent=2)
                )
                return {"id": task.get("id"), "step": task.get("name"), "status": "done(meta)"}

            # ----- 结构/参数/HPC 提交类 -----
            if agent in {
                "structure.relax_slab", "structure.intermediates", "structure.relax_adsorbate",
                "adsorption.scan", "adsorption.co", "neb.run", "electronic.dos", "run_dft", "post.energy"
            }:
                if self.struct_agent:
                    self.struct_agent.build(task, job_dir)
                if self.param_agent:
                    self.param_agent.generate(task, job_dir)

                if self.hpc_agent:
                    payload = (task.get("params") or {}).get("payload") or {}
                    step_info = {
                        "name": task.get("name", "chatdft"),
                        "engine": (payload.get("engine") or "vasp").lower(),
                        "ntasks": payload.get("ntasks"),
                        "walltime": payload.get("walltime"),
                        "template_vars": payload.get("template_vars") or {},
                    }
                    try:
                        if hasattr(self.hpc_agent, "set_runtime_context"):
                            self.hpc_agent.set_runtime_context(
                                project=(task.get("meta") or {}).get("project"),
                                session_id=(task.get("meta") or {}).get("run_id"),
                            )
                    except Exception:
                        pass

                    self.hpc_agent.prepare_script(step_info, job_dir)
                    jid = self.hpc_agent.submit(job_dir)

                    if not self.dry:
                        self.hpc_agent.wait(jid, poll=60)
                        self.hpc_agent.fetch_outputs(
                            job_dir, filters=["OUTCAR", "vasprun.xml", "OSZICAR", "stdout", "stderr"]
                        )

                return {"id": task.get("id"), "step": task.get("name"), "status": "done(hpc)"}

            # ----- post-only -----
            if agent == "post.analysis":
                if self.post_agent:
                    self.post_agent.analyze(workdir)
                return {"id": task.get("id"), "step": task.get("name"), "status": "done(post)"}

            # 未知 agent
            return {"id": task.get("id"), "step": task.get("name"), "status": f"skipped (unknown agent: {agent})"}

        except Exception as e:
            return {"id": task.get("id"), "step": task.get("name"), "status": f"error: {e}"}

    def execute_selected(self, all_tasks: List[Dict[str, Any]], selected_ids: List[int]) -> Dict[str, Any]:
        work_root = Path(tempfile.mkdtemp(prefix="chatdft_"))
        results = []
        for t in all_tasks:
            if t.get("id") in selected_ids:
                results.append(self._exec_task(t, work_root))
        try:
            if self.post_agent:
                self.post_agent.analyze(work_root)
        except Exception:
            results.append({"step":"post.analysis","status":"error: post_agent.analyze failed"})
        def _sum(rows):
            done = sum(1 for r in rows if str(r.get("status","")).startswith("done"))
            err  = sum(1 for r in rows if str(r.get("status","")).startswith("error"))
            skip = sum(1 for r in rows if str(r.get("status","")).startswith("skipped"))
            return {"done": done, "error": err, "skipped": skip, "total": len(rows)}
        return {"workdir": str(work_root), "results": results, "summary": _sum(results)}

@router.post("/chat/execute")
async def api_execute(request: Request):
    data = await request.json()
    try:
        mgr = PlanManager(
            cluster   = data.get("cluster", "hoffman2"),
            dry_run   = bool(data.get("dry_run", False)),
            sync_back = bool(data.get("sync_back", True)),
        )
        all_tasks = data.get("all_tasks") or []
        selected  = data.get("selected_ids") or []
        if not isinstance(all_tasks, list) or not isinstance(selected, list):
            return {"ok": False, "detail": "all_tasks or selected_ids malformed"}

        res = mgr.execute_selected(all_tasks, selected)
        return {"ok": True, **res}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"ok": False, "detail": str(e), "traceback": tb}

# ---- uvicorn entry ----
app = FastAPI()
app.include_router(router)