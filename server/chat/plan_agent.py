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

import logging
log = logging.getLogger(__name__)

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

# ---------- Mechanism Registry (可选) ----------
# NEW: 引入机制注册表（若不存在则为空不影响运行）
try:
    from server.mechanisms.registry import REGISTRY
except Exception:
    REGISTRY = {}

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
    # deliverables 可能为 dict 或 list
    deliverables = intent.get("deliverables")
    if isinstance(deliverables, dict):
        target_products = _to_list(deliverables.get("target_products"))
    elif isinstance(deliverables, list):
        target_products = deliverables
    else:
        target_products = []
    text = " ".join([
        query or "",
        str(intent.get("task") or ""),
        str(intent.get("reaction") or intent.get("problem_type") or ""),
        " ".join(_to_list(target_products))
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
async def _build_tasks(intent: Dict[str, Any],
                       steps: List[str], inter: List[str],
                       coads_pairs: List[Tuple[str,str]], ts_edges: List[str],
                       session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    catalyst = (intent.get("system") or {}).get("catalyst") or (intent.get("system") or {}).get("material") or "Pt"
    facet    = (intent.get("system") or {}).get("facet") or "111"
    tasks: List[Dict[str, Any]] = []
    tid = 1

    def _field(key, label, ftype="text", value="", **kw):
        d = {"key": key, "label": label, "type": ftype, "value": value}
        d.update({k:v for k,v in kw.items() if v is not None})
        return d

    async def _llm_task_desc(agent, payload):
        # LLM辅助生成任务描述和参数建议
        try:
            prompt = {"agent": agent, "payload": payload, "intent": intent}
            sys = "You are a senior researcher. Given the agent type and payload, generate a concise task description and suggest key parameters (form fields) for a DFT workflow."
            raw = await chatgpt_call([
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ], model="gpt-4o-mini", temperature=0.2, max_tokens=600)
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0) if m else raw)
            desc = data.get("desc") or data.get("description") or ""
            form = data.get("form") or []
            return desc, form
        except Exception:
            return "", []

    async def _task(section, name, agent, desc, form=None, payload=None, group=0, endpoint=None):
        nonlocal tid
        t = {
            "id": tid, "section": section, "name": name, "agent": agent, "description": desc,
            "params": {"form": form or [], "payload": payload or {}},
            "meta": {"parallel_group": group, "action_endpoint": endpoint,
                     "project": f"session-{session_id}" if session_id else None,
                     "run_id": session_id}
        }
        tid += 1
        tasks.append(t)

    # 智能 slab 任务（可扩展 bulk/slab/modify）
    slab_payload = {"facet": facet, "element": catalyst}
    desc, form = await _llm_task_desc("structure.relax_slab", slab_payload)
    await _task(
        "Model", f"Build slab — {catalyst}({facet})", "structure.relax_slab",
        desc or f"Build/relax slab for {catalyst}({facet}).",
        form or [
            _field("engine","Engine","select","vasp", options=["vasp","qe"]),
            _field("element","Element","text",catalyst),
            _field("facet","Facet","text",facet),
        ],
        payload=slab_payload,
        group=1, endpoint="/agent/structure.relax_slab"
    )

    # 智能吸附任务
    adsorbates = [s for s in inter if s.endswith("*")]
    for sp in adsorbates:
        payload = {"adsorbate": sp, "facet": facet, "element": catalyst}
        desc, form = await _llm_task_desc("adsorption.scan", payload)
        await _task(
            "Adsorption", f"Relax on sites — {sp}", "adsorption.scan",
            desc or f"Enumerate adsorption sites for {sp} and relax.",
            form or [
                _field("adsorbate","Adsorbate","text",sp),
            ],
            payload=payload,
            group=2, endpoint="/agent/adsorption.scan"
        )

    # 智能共吸附任务
    for a, b in coads_pairs[:30]:
        payload = {"pair": [a, b], "facet": facet, "element": catalyst}
        desc, form = await _llm_task_desc("adsorption.co", payload)
        await _task(
            "Co-adsorption", f"Co-ads — {a}+{b}", "adsorption.co",
            desc or f"Create and relax co-adsorption configs for {a} + {b}.",
            form or [
                _field("pair","Pair","text",f"{a},{b}"),
            ],
            payload=payload,
            group=3, endpoint="/agent/adsorption.co"
        )

    # 智能 TS 任务
    for s in ts_edges[:20]:
        payload = {"step": s, "facet": facet, "element": catalyst}
        desc, form = await _llm_task_desc("neb.run", payload)
        await _task(
            "Transition States", f"NEB — {s}", "neb.run",
            desc or f"CI-NEB for elementary step: {s}",
            form or [
                _field("step","Elementary step","text",s),
            ],
            payload=payload,
            group=4, endpoint="/agent/neb.run"
        )

    # 智能电子/后处理任务
    if adsorbates:
        payload = {"species": adsorbates[:6], "facet": facet, "element": catalyst}
        desc, form = await _llm_task_desc("electronic.dos", payload)
        await _task(
            "Electronic", "DOS/PDOS/Bader", "electronic.dos",
            desc or "Compute DOS/PDOS/Bader for key species.",
            form or [
                _field("dos","DOS","checkbox",True),
            ],
            payload=payload,
            group=5, endpoint="/agent/electronic.dos"
        )
    desc, form = await _llm_task_desc("post.analysis", {})
    await _task(
        "Post-analysis", "Assemble ΔG / barriers", "post.analysis",
        desc or "Assemble ΔG profile and barrier diagram from results.",
        form or [
            _field("temperature","Temperature (K)","number",298.15),
        ],
        payload={},
        group=5, endpoint="/agent/post.analysis"
    )
    return tasks

# ========================= /chat/plan =========================
@router.post("/chat/plan")
async def api_plan(request: Request):

    import logging
    log = logging.getLogger(__name__)
    log.info("Planning running here")

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

    # 3) Mechanism seed（优先于旧 seed；若命中）
    mech_keys = _mech_guess(intent_in, query) if REGISTRY else []
    mech_seed = _mech_seed(intent_in, mech_keys) if mech_keys else {}

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

    # 8) 低置信度/强制：轻度并入传统 seed（保持你原策略）
    if USE_SEED_POLICY == "always" or (USE_SEED_POLICY == "auto" and conf < CONF_THRESHOLD):
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
    tasks = await _build_tasks(intent_in, steps, inter, coads_pairs, ts_edges, session_id=session_id)

    # 12) 组装 workflow（便于前端 HPC 页面）
    workflow = {
        "id": f"wf-{session_id}-{int(datetime.utcnow().timestamp())}" if session_id else f"wf-{int(datetime.utcnow().timestamp())}",
        "title": f"DFT workflow — {(intent_in.get('problem_type') or 'Task')}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "project": f"session-{session_id}" if session_id else None,
        "run_id": session_id,
        "mechanisms": mech_keys,
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
            "mechanisms": mech_keys
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

    async def _exec_task(self, task: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
        # 1) 先拿到 agent；避免 NameError
        agent = (task.get("agent") or "").lower()

        # 2) 安全的子目录名
        # Fake IO things
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
                    print(job_dir)
                    # Determine structure_type
                    structure_type = "slab"
                    if agent.startswith("structure.bulk"):
                        structure_type = "bulk"
                    elif agent.startswith("structure.relax_slab"):
                        structure_type = "slab"
                    elif agent.startswith("structure.intermediates") or agent.startswith("structure.relax_adsorbate"):
                        structure_type = "adsorption"
                    elif agent.startswith("adsorption.co"):
                        structure_type = "co-adsorption"
                    elif agent.startswith("neb.run"):
                        structure_type = "ts"
                    elif agent.startswith("structure.modify"):
                        structure_type = "modify"
                    # Set structure_type in payload
                    if "params" in task and "payload" in task["params"]:
                        task["params"]["payload"]["structure_type"] = structure_type
                    # Async call to StructureAgent
                    await self.struct_agent.build(task, job_dir)
                
                print("Start para running")

                if self.param_agent:
                    self.param_agent.generate(task, job_dir)

                print("Start hpc running")

                if self.hpc_agent:
                    print("Here we go for HPC!!!!")
                    payload = (task.get("params") or {}).get("payload") or {}
                    step_info = {
                        "name": task.get("name", "chatdft"),
                        "engine": (payload.get("engine") or "vasp").lower(),
                        "ntasks": payload.get("ntasks"),
                        "walltime": payload.get("walltime"),
                        "template_vars": payload.get("template_vars") or {},
                    }
                    print('Stage 1')
                    try:
                        if hasattr(self.hpc_agent, "set_runtime_context"):
                            self.hpc_agent.set_runtime_context( # Set the task info
                                project=(task.get("meta") or {}).get("project"),
                                session_id=(task.get("meta") or {}).get("run_id"),
                            )
                    except Exception as e:
                        print(f'Here we print the error: {e}')
                        pass
                    print('Stage 2')

                    self.hpc_agent.prepare_script(step_info, job_dir)
                    jid = self.hpc_agent.submit(job_dir) # Submit right here

                    print("stage 3")

                    # Try to read remote metadata (cluster/remote_dir)
                    remote_dir = None
                    try:
                        import json as _json
                        meta_path = job_dir / "_remote.json"
                        if meta_path.exists():
                            m = _json.loads(meta_path.read_text() or "{}")
                            remote_dir = m.get("remote_dir")
                    except Exception:
                        pass

                return {
                    "id": task.get("id"),
                    "step": task.get("name"),
                    "status": "Submitted",
                    "job_id": (jid if 'jid' in locals() else None),
                    "job_dir": str(job_dir),
                    "remote_dir": remote_dir,
                    "cluster": getattr(self.hpc_agent, 'cluster', None) if self.hpc_agent else None,
                }

            # ----- post-only -----
            if agent == "post.analysis":
                if self.post_agent:
                    self.post_agent.analyze(workdir)
                return {"id": task.get("id"), "step": task.get("name"), "status": "done(post)"}

            # 未知 agent
            return {"id": task.get("id"), "step": task.get("name"), "status": f"skipped (unknown agent: {agent})"}

        except Exception as e:
            return {"id": task.get("id"), "step": task.get("name"), "status": f"error: {e}"}

    async def execute_selected(self, all_tasks: List[Dict[str, Any]], selected_ids: List[int]) -> Dict[str, Any]:
        work_root = Path(tempfile.mkdtemp(prefix="chatdft_"))
        results = []
        log.info('Here we start to run the sel task')
        log.info(f'Check input task ids {selected_ids}')
        for t in all_tasks:
            if t.get("id") in selected_ids:
                log.info(f'True task check {t}')
                results.append(await self._exec_task(t, work_root))
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

    log.info("Executing running here")

    data = await request.json()

    try:
        mgr = PlanManager(
            cluster   = data.get("cluster", "hoffman2"),
            dry_run   = bool(data.get("dry_run", False)),
            sync_back = bool(data.get("sync_back", True)),
        )
        all_tasks = data.get("all_tasks") or []
        print(all_tasks)
        # Do the replacement or not?
        print("The idx keywork check")
        print(data.get("selected_task_ids"))
        selected  = data.get("selected_task_ids") or []
        if not isinstance(all_tasks, list) or not isinstance(selected, list):
            return {"ok": False, "detail": "all_tasks or selected_ids malformed"}
        
        # Wheter selection are OK or not
        # import logging
        # log = logging.getLogger(__name__)
        # log.info(f'The all task check: {all_tasks}')
        log.info(f'The sel task check: {selected}')

        res = await mgr.execute_selected(all_tasks, selected)

        print("Here we run to the end")

        return {"ok": True, **res}
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {"ok": False, "detail": str(e), "traceback": tb}

# ---- uvicorn entry ----
app = FastAPI()
app.include_router(router)
