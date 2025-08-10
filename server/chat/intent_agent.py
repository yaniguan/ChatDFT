# server/chat/intent_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
log = logging.getLogger(__name__)

# =========================
# DB save helper (async)
# =========================
async def _save_artifact(session_id: Optional[int], msg_type: str, content: Any):
    """Persist an artifact into ChatMessage; safe to call (no-op if no session)."""
    if not session_id:
        return
    try:
        from server.db_last import AsyncSessionLocal, ChatMessage  # 如路径不同，请改这里
    except Exception:
        log.warning("intent_agent: DB models not available, skip saving")
        return

    try:
        txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        async with AsyncSessionLocal() as s:
            m = ChatMessage(session_id=session_id, role="assistant", msg_type=msg_type, content=txt)
            s.add(m)
            await s.commit()
    except Exception:
        log.exception("intent_agent: save_artifact failed")

# =========================
# small utils
# =========================
def _as_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]

def _clean_step(s: Any) -> Dict[str, Any]:
    """Normalize a 'step' entry to dict."""
    if isinstance(s, dict):
        r = s.get("reactants") or s.get("lhs") or s.get("from") or s.get("src")
        p = s.get("products")  or s.get("rhs") or s.get("to")   or s.get("dst")
        out = dict(s)
        if r is not None: out["reactants"] = r
        if p is not None: out["products"]  = p
        return out
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return {"reactants": s[0], "products": s[1]}
    return {"step": s}

def _norm_pairs(pairs: List[Any]) -> List[Tuple[Any, Any]]:
    out = []
    for x in _as_list(pairs):
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            out.append((x[0], x[1]))
        elif isinstance(x, dict) and "a" in x and "b" in x:
            out.append((x["a"], x["b"]))
        else:
            out.append((x, None))
    return out

def _domain_guess(q: str, base: str | None) -> str:
    t = (q or "").lower()
    if any(k in t for k in ["electro", "rhe", "she", "ph="]):
        return "catalysis"
    if any(k in t for k in ["battery", "sei", "cathode", "anode", "electrolyte"]):
        return "batteries"
    if any(k in t for k in ["polymer", "monomer"]):
        return "polymers"
    return base or "materials_general"

def _make_tags(intent: Dict[str, Any]) -> List[str]:
    tags = set()
    dom = intent.get("domain")
    if dom: tags.add(dom)
    sys = intent.get("system") or {}
    for k in ["material","catalyst","facet","molecule","defect"]:
        v = sys.get(k)
        if isinstance(v, str) and v.strip():
            tags.add(v.strip())
    cond = intent.get("conditions") or {}
    for k in ["pH","potential","temperature","pressure","electrolyte","solvent"]:
        v = cond.get(k)
        if v not in (None, "", [], {}):
            tags.add(f"{k}:{v}")
    rn = intent.get("reaction_network") or {}
    if rn.get("elementary_steps"): tags.add(f"steps:{len(rn['elementary_steps'])}")
    if rn.get("intermediates"):    tags.add(f"inters:{len(rn['intermediates'])}")
    return sorted(tags)

def _confidence(steps: List[Any], intermediates: List[Any], ts: List[Any]) -> float:
    s = 0.0
    s += 0.45 if steps else 0.0
    s += 0.30 if intermediates else 0.0
    s += 0.15 if ts else 0.0
    s = min(1.0, max(0.0, s + 0.10))
    return round(s, 2)

import re

_RX_FACET = re.compile(r"\b([A-Za-z]{1,2})(?:\s*|\-)?\(?\s*(\d{1,3})\s*(\d{1,3})\s*(\d{1,3})\s*\)?\b")
# 例: Cu(111), Cu111, Cu-111

import re

_BATT_CATHODES = r"(NMC|NCA|LFP|LCO|LMO|LNMO|LNO)"
_BATT_ANODES   = r"(Si|Graphite|LTO|SiOx|Hard\s*Carbon)"
_POLY_METHODS  = r"(ROP|RAFT|ATRP|FRP|ROMP|Ziegler[- ]Natta)"

def _parse_query_to_fields(q: str) -> dict:
    """
    轻量解析：催化/电催化/光(电)催化、电池、聚合物。返回 {domain, domain_subtype, system:{...}, conditions:{...}, molecule}
    """
    out = {"domain": None, "domain_subtype": None, "system": {}, "conditions": {}, "molecule": None}
    if not q: return out
    t = q.strip()

    low = t.lower()

    # ===== 催化 =====
    if re.search(r"\b(co2rr|orr|oer|her|co2)\b", low) or "catalys" in low:
        out["domain"] = "catalysis"
        if "electro" in low or re.search(r"\b(rhe|she|vs)\b", low):
            out["domain_subtype"] = "electrocatalysis"
        elif "photoelectro" in low:
            out["domain_subtype"] = "photoelectrocatalysis"
        elif "photo" in low:
            out["domain_subtype"] = "photocatalysis"
        else:
            out["domain_subtype"] = "thermocatalysis"

        # 元素 & 晶面
        m_cat = re.search(r"\b(Cu|Ag|Au|Ni|Co|Fe|Pt|Pd|Sn|Bi|Ru|Rh|Ir)\b", t, re.I)
        if m_cat:
            cat = m_cat.group(1).capitalize()
            out["system"]["catalyst"] = cat
            m_fac = re.search(rf"{cat}[\s\-]*\(?\s*(111|100|110)\s*\)?", t, re.I)
            if m_fac:
                out["system"]["facet"] = f"{cat}({m_fac.group(1)})"

        if "co2" in low: out["molecule"] = "CO2"

        # 条件
        m_ph = re.search(r"\bpH\s*=\s*([0-9]+(?:\.[0-9]+)?)", t, re.I)
        if m_ph: 
            try: out["conditions"]["pH"] = float(m_ph.group(1))
            except: pass
        m_v = re.search(r"([\-+]?\d+(?:\.\d+)?)\s*V\s*(?:vs\.?\s*(RHE|SHE))?", t, re.I)
        if m_v:
            out["conditions"]["potential"] = f"{m_v.group(1)} V" + (f" vs {m_v.group(2).upper()}" if m_v.group(2) else "")

    # ===== 电池 =====
    if "battery" in low or re.search(r"\b(Li-ion|Li metal|sodium-ion|solid-state)\b", t, re.I):
        out["domain"] = "batteries"
        # 正极/负极关键词
        m_c = re.search(_BATT_CATHODES, t, re.I)
        if m_c: out["system"]["cathode"] = m_c.group(1).upper()
        m_a = re.search(_BATT_ANODES, t, re.I)
        if m_a: out["system"]["anode"] = m_a.group(1)
        # 电解液/盐
        m_salt = re.search(r"\b(LiPF6|LiFSI|LiTFSI|NaPF6)\b", t, re.I)
        if m_salt: out["system"]["salt"] = m_salt.group(1)
        if "electrolyte" in low:
            out["system"]["electrolyte"] = "liquid"
        if "solid" in low and "electrolyte" in low:
            out["system"]["electrolyte"] = "solid"

        # 条件
        m_cr = re.search(r"(\d+(?:\.\d+)?)\s*C\s*[-]?\s*rate", t, re.I)
        if m_cr: out["conditions"]["C_rate"] = m_cr.group(1)
        m_t  = re.search(r"(\-?\d+)\s*[°º]C", t)
        if m_t: out["conditions"]["temperature"] = f"{m_t.group(1)} °C"

    # ===== 聚合物 =====
    if "polymer" in low or re.search(_POLY_METHODS, t, re.I):
        out["domain"] = "polymers"
        m_method = re.search(_POLY_METHODS, t, re.I)
        if m_method: out["system"]["method"] = m_method.group(1).upper()
        # 单体（简单抓几个典型）
        mons = re.findall(r"\b(ethylene|propylene|styrene|methyl\s*methacrylate|lactide|caprolactone)\b", t, re.I)
        if mons:
            out["system"]["monomers"] = [m.strip() for m in mons]
        m_T = re.search(r"(\-?\d+)\s*[°º]C", t)
        if m_T: out["conditions"]["temperature"] = f"{m_T.group(1)} °C"

    return out
# =========================
# /chat/intent  (唯一导出的路由)
# =========================
@router.post("/chat/intent")
async def api_intent(request: Request):
    """
    Normalize INTENT from heterogeneous inputs.
    Accepts (all optional):
      - query: str
      - intent: dict (overrides)
      - hypothesis: dict OR markdown string
      - graph/reaction_network: {elementary_steps, intermediates, coads_pairs, ts_candidates}
      - conditions/system: dicts
    Returns: { ok, intent, fields, confidence, summary }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON body"}, status_code=400)

    session_id = body.get("session_id")
    query      = (body.get("query") or "").strip()
    parsed = _parse_query_to_fields(query)
    base_intent = body.get("intent") if isinstance(body.get("intent"), dict) else {}

    # hypothesis: allow dict OR markdown string
    raw_hyp = body.get("hypothesis")
    if isinstance(raw_hyp, dict):
        hyp = raw_hyp
    elif isinstance(raw_hyp, str):
        try:
            tmp = json.loads(raw_hyp)
            hyp = tmp if isinstance(tmp, dict) else {}
        except Exception:
            hyp = {}
    else:
        hyp = {}

    # graph payload
    graph = body.get("graph") or body.get("reaction_network") or {}
    if not isinstance(graph, dict):
        graph = {}

    # optional conditions/system
    conditions = body.get("conditions") if isinstance(body.get("conditions"), dict) else {}
    system     = body.get("system")     if isinstance(body.get("system"), dict)     else {}

    # collect reaction pieces (never .get() on non-dict)
    raw_steps = body.get("steps") or graph.get("elementary_steps") or (hyp.get("steps") if isinstance(hyp, dict) else []) or []
    raw_inter = body.get("intermediates") or graph.get("intermediates") or (hyp.get("intermediates") if isinstance(hyp, dict) else []) or []
    raw_coads = body.get("coads_pairs") or body.get("coads") or graph.get("coads_pairs") or (hyp.get("coads") if isinstance(hyp, dict) else []) or []
    raw_ts    = body.get("ts") or body.get("ts_candidates") or graph.get("ts_candidates") or (hyp.get("ts") if isinstance(hyp, dict) else []) or []

    steps        = [_clean_step(s) for s in _as_list(raw_steps)]
    intermediates= _as_list(raw_inter)
    coads_pairs  = _norm_pairs(raw_coads)
    ts_out       = _as_list(raw_ts)

    reaction_network = {
        "elementary_steps": steps,
        "intermediates": intermediates,
        "coads_pairs": coads_pairs,
        "ts_candidates": ts_out,
    }

    # build intent
    domain = _domain_guess(query, base_intent.get("domain") or parsed.get("domain"))
    subtype = base_intent.get("domain_subtype") or parsed.get("domain_subtype")

    sys_parsed  = parsed.get("system") or {}
    cond_parsed = parsed.get("conditions") or {}

    intent = {
        "intent_version": "1.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "domain_subtype": subtype,   # <—— 新增：子类型（electrocatalysis / thermocatalysis / batteries / polymers ...）
        "problem_type": base_intent.get("problem_type") or "reaction study",
        "system": {
            "material": (base_intent.get("system") or {}).get("material"),
            "catalyst": (base_intent.get("system") or {}).get("catalyst") or sys_parsed.get("catalyst"),
            "facet":    (base_intent.get("system") or {}).get("facet")    or sys_parsed.get("facet"),
            "defect":   (base_intent.get("system") or {}).get("defect"),
            "molecule": (base_intent.get("system") or {}).get("molecule") or parsed.get("molecule"),
            # 电池/聚合物拓展字段可直接放这里：
            "cathode":  (base_intent.get("system") or {}).get("cathode")  or sys_parsed.get("cathode"),
            "anode":    (base_intent.get("system") or {}).get("anode")    or sys_parsed.get("anode"),
            "salt":     (base_intent.get("system") or {}).get("salt")     or sys_parsed.get("salt"),
            "electrolyte": (base_intent.get("system") or {}).get("electrolyte") or sys_parsed.get("electrolyte"),
            "monomers": (base_intent.get("system") or {}).get("monomers") or sys_parsed.get("monomers"),
            "method":   (base_intent.get("system") or {}).get("method")   or sys_parsed.get("method"),
        },
        "conditions": {
            "pH":          (base_intent.get("conditions") or {}).get("pH",          cond_parsed.get("pH")),
            "potential":   (base_intent.get("conditions") or {}).get("potential",   cond_parsed.get("potential")),
            "temperature": (base_intent.get("conditions") or {}).get("temperature", cond_parsed.get("temperature")),
            "pressure":    (base_intent.get("conditions") or {}).get("pressure",    cond_parsed.get("pressure")),
            "electrolyte": (base_intent.get("conditions") or {}).get("electrolyte", cond_parsed.get("electrolyte")),
            "solvent":     (base_intent.get("conditions") or {}).get("solvent",     cond_parsed.get("solvent")),
            "C_rate":      (base_intent.get("conditions") or {}).get("C_rate",      cond_parsed.get("C_rate")),
            "v_min":       (base_intent.get("conditions") or {}).get("v_min",       cond_parsed.get("v_min")),
            "v_max":       (base_intent.get("conditions") or {}).get("v_max",       cond_parsed.get("v_max")),
            "cycles":      (base_intent.get("conditions") or {}).get("cycles",      cond_parsed.get("cycles")),
            "illumination":(base_intent.get("conditions") or {}).get("illumination",cond_parsed.get("illumination")),
            "wavelength":  (base_intent.get("conditions") or {}).get("wavelength",  cond_parsed.get("wavelength")),
        },
        "target_properties": base_intent.get("target_properties") or [],
        "metrics":           base_intent.get("metrics") or [],
        "datasets":          base_intent.get("datasets") or [],
        "normalized_query":  query,
        "reaction_network":  reaction_network,
    }
    # base_intent 覆盖
    for k, v in base_intent.items():
        if v in (None, "", [], {}):
            continue
        if k == "system" and isinstance(v, dict):
            intent["system"].update(v)
        elif k == "conditions" and isinstance(v, dict):
            intent["conditions"].update(v)
        else:
            intent[k] = v

    # new: tags
    intent["tags"] = _make_tags(intent)

    # confidence & summary
    conf = _confidence(steps, intermediates, ts_out)
    def _kv(d): return ", ".join(f"{k}={v}" for k, v in d.items() if v not in (None, "", [], {})) or "-"
    summary = (
        f"**Intent Summary**\n"
        f"- Domain: {intent.get('domain','-')}\n"
        f"- Problem: {intent.get('problem_type','-')}\n"
        f"- System: material={intent['system'].get('material')}, catalyst={intent['system'].get('catalyst')}, "
        f"facet={intent['system'].get('facet')}, defect={intent['system'].get('defect')}, molecule={intent['system'].get('molecule')}\n"
        f"- Conditions: {_kv(intent['conditions'])}\n"
        f"- RN: steps={len(steps)}, intermediates={len(intermediates)}, ts={len(ts_out)}, coads={len(coads_pairs)}\n"
        f"- Tags: {', '.join(intent.get('tags') or []) or '-'}\n"
    )

    # persist
    try:
        if session_id:
            await _save_artifact(session_id, "intent", intent)
            await _save_artifact(session_id, "rxn_network", reaction_network)  # 便于 /state 聚合
    except Exception:
        log.exception("api_intent: persist failed")

    return {"ok": True, "intent": intent, "fields": intent, "confidence": conf, "summary": summary}