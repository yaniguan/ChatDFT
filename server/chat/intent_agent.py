# server/chat/intent_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, ast
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request

router = APIRouter()

# ---------------- LLM wrapper ----------------
# 你提供的文件：server/utils/llm_wrapper.py 里定义了 async chatgpt_call(...)
_HAS_LLM = False
_chat = None

async def _call_llm(messages: List[Dict[str, str]], temperature=0.2, max_tokens=1000) -> Optional[str]:
    """懒加载 OpenAI 异步客户端；失败返回 None。"""
    global _HAS_LLM, _chat
    if not _HAS_LLM:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from server.utils.openai_wrapper import chatgpt_call
            _chat = chatgpt_call
            _HAS_LLM = True
        except Exception:
            return None
    try:
        return await _chat(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception:
        return None

# ---------------- utils ----------------
def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = re.sub(r"^json", "", s.strip(), flags=re.I)
    return s.strip()

def _safe_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s: return None
    s = _strip_fences(s)
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = re.sub(r",\s*([}\]])", r"\1", s.replace("'", '"'))
            return json.loads(s2)
        except Exception:
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

# ---------------- rule baseline ----------------
_RX_PH   = re.compile(r"(?i)\bpH\s*=?\s*([0-9]+(?:\.[0-9]+)?)")
_RX_POT  = re.compile(r"(?i)([-+]?\d+(?:\.\d+)?)\s*V\s*(?:vs\.?\s*)?(RHE|SHE|Ag/AgCl)")
_RX_EL   = re.compile(r"\b([A-Z][a-z]?)\b")
_RX_FAC  = re.compile(r"\b([A-Z][a-z]?)\s*\((\d{3})\)")

_ELTS = {"H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
         "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
         "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
         "Cs","Ba","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
         "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi"}

def _guess_domain(q: str) -> str:
    t = q.lower()
    if any(k in t for k in ["battery","cathode","anode","electrolyte","sei","lithium","solid-state"]):
        return "batteries"
    if any(k in t for k in ["polymer","monomer","smiles","inchi","thermoplastic","thermoset"]):
        return "polymers"
    if any(k in t for k in ["phonon","elastic","band","dos","neb","adsorption","surface","catalyst","co2rr","her","oer","orr","nrr"]):
        return "catalysis"
    if any(k in t for k in ["machine learning","surrogate","gnn","active learning","dataset","benchmark"]):
        return "materials_ml"
    if any(k in t for k in ["shoe","sock","dog","movie","recipe"]):
        return "out_of_domain"
    return "materials_general"

def _extract_conditions(q: str) -> Dict[str, Any]:
    cond: Dict[str, Any] = {}
    m = _RX_PH.search(q);  cond["pH"] = m.group(1) if m else None
    m = _RX_POT.search(q); cond["potential"] = f"{m.group(1)} V vs {m.group(2)}" if m else None
    return {k:v for k,v in cond.items() if v}

def _rule_intent(query: str) -> Dict[str, Any]:
    dom = _guess_domain(query)
    # 粗提催化剂/晶面
    catalyst, facet = None, None
    m = _RX_FAC.search(query)
    if m: catalyst, facet = m.group(1), f"{m.group(1)}({m.group(2)})"
    else:
        for el in set(_RX_EL.findall(query)):
            if el in _ELTS: catalyst = el; break
        if catalyst: facet = f"{catalyst}(111)"

    return {
        "domain": dom,
        "problem_type": "-",            # 让 LLM 规范化
        "system": {"material": None, "catalyst": catalyst, "facet": facet, "defect": None, "molecule": None},
        "conditions": _extract_conditions(query),
        "target_properties": [],
        "metrics": [],
        "datasets": [],
        "normalized_query": query.strip(),
        "dft_tasks": [],
        "non_dft_paths": [],
        "red_flags": [],
    }

# ---------------- prompting ----------------
SCHEMA = (
    '{'
    '"domain":"catalysis|batteries|polymers|materials_ml|materials_general|out_of_domain",'
    '"problem_type": "short noun phrase",'
    '"system":{"material":str|null,"catalyst":str|null,"facet":str|null,"defect":str|null,"molecule":str|null},'
    '"conditions":{"pH":str|null,"potential":str|null,"temperature":str|null,"pressure":str|null,"electrolyte":str|null,"solvent":str|null},'
    '"target_properties":[str],'
    '"metrics":[str],'
    '"datasets":[str],'
    '"normalized_query":str,'
    '"dft_tasks":[{"name":str,"why":str,"inputs":[str]}],'
    '"non_dft_paths":[{"method":str,"why":str,"next_step":str}],'
    '"red_flags":[str]'
    '}'
)

def _compose_messages(query: str) -> List[Dict[str,str]]:
    sys = (
        "You are an expert research planner for computational materials. "
        "Normalize the inquiry and propose DFT-suitable tasks AND complementary non-DFT paths. "
        "Return STRICT JSON ONLY. Schema:\n" + SCHEMA +
        "\nRules:\n"
        "- If out-of-domain, set domain='out_of_domain' and suggest reinterpretation in non_dft_paths.\n"
        "- Keep keys exact; unknown -> null/[]; no markdown."
    )
    return [{"role":"system","content":sys},{"role":"user","content":query}]

def _merge(rule_f: Dict[str,Any], llm_f: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(rule_f)
    if not isinstance(llm_f, dict): return out
    for k in ["domain","problem_type","system","conditions","target_properties","metrics","datasets",
              "normalized_query","dft_tasks","non_dft_paths","red_flags"]:
        v = llm_f.get(k)
        if v not in (None, "", []): out[k] = v
    return out

def _confidence(f: Dict[str,Any]) -> float:
    dm = 1.0 if f.get("domain") in {"catalysis","batteries","polymers","materials_ml","materials_general"} else 0.2
    sys = f.get("system") or {}
    spec = 0.0 + (0.6 if (sys.get("material") or sys.get("catalyst")) else 0.0) + (0.4 if isinstance(sys.get("facet"), str) else 0.0)
    cond = f.get("conditions") or {}
    comp = sum(1 for k in ["pH","potential","temperature","pressure","electrolyte","solvent"] if cond.get(k)) / 6.0
    dfts = f.get("dft_tasks") or []
    ready = 1.0 if all(isinstance(t,dict) and t.get("name") and t.get("inputs") for t in dfts) else 0.5
    score = 0.25*dm + 0.25*spec + 0.25*comp + 0.25*ready
    return round(max(0.0, min(1.0, score)), 2)

def _summary(f: Dict[str,Any]) -> str:
    sys = f.get("system") or {}
    cond = f.get("conditions") or {}
    def _kv(d): return ", ".join([f"{k}={v}" for k,v in d.items() if v]) or "-"
    dfts = "\n".join([f"- **{t.get('name','')}** — {t.get('why','')}" for t in (f.get("dft_tasks") or [])]) or "-"
    ndft = "\n".join([f"- **{p.get('method','')}** — {p.get('why','')}" for p in (f.get("non_dft_paths") or [])]) or "-"
    return (
        f"**Intent Summary**\n"
        f"- Domain: {f.get('domain','-')}\n"
        f"- Problem type: {f.get('problem_type','-')}\n"
        f"- System: material={sys.get('material')}, catalyst={sys.get('catalyst')}, facet={sys.get('facet')}, defect={sys.get('defect')}, molecule={sys.get('molecule')}\n"
        f"- Conditions: {_kv(cond)}\n"
        f"- Target properties: {', '.join(f.get('target_properties',[])) or '-'}\n"
        f"- DFT tasks:\n{dfts}\n"
        f"- Non-DFT suggestions:\n{ndft}\n"
    )

# ---------------- FastAPI route ----------------
@router.post("/chat/intent")
async def intent_route(request: Request):
    body = await request.json()
    query = _norm(body.get("query"))
    if not query:
        # 永远返回非空 intent
        fallback = _rule_intent("")
        return {"ok": False, "intent": fallback, "summary": "Empty query."}

    rule_fields = _rule_intent(query)

    # LLM 解析（可用则 2 轮）
    merged = rule_fields
    if os.getenv("OPENAI_API_KEY"):
        rounds: List[Dict[str,Any]] = []
        for _ in range(2):
            txt = await _call_llm(_compose_messages(query), temperature=0.2, max_tokens=1100)
            j = _safe_json(txt)
            if j: rounds.append(j)
        if rounds:
            # 用最后一轮（通常最完整），也可投票
            merged = _merge(rule_fields, rounds[-1])

    conf = _confidence(merged)
    summary = _summary(merged)

    # 统一、稳健的返回
    return {
        "ok": True,
        "intent": merged,        # 前端直接读这个键即可
        "fields": merged,        # 兼容你之前用 fields 的地方
        "confidence": conf,
        "summary": summary,
    }