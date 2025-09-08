# server/chat/intent_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, logging, math, re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from server.db import SessionLocal, ChatMessage, Hypothesis, WorkflowTask, IntentPhrase
from server.utils.rag_utils import rag_context
from server.utils.openai_wrapper import chatgpt_call  # async

# NEW: 引入机制注册表
try:
    from server.mechanisms.registry import REGISTRY
except Exception:
    REGISTRY = {}

router = APIRouter()
import logging
# Here the logger has been get then we have a test over this OK!!!
log = logging.getLogger(__name__) 

# -----------------------------
# 小工具：保存到 DB（同步会话）
# -----------------------------
from sqlalchemy import select
def _clip(s: str, max_chars: int) -> str:
    s = str(s or "")
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n…[truncated {len(s)-max_chars} chars]"

# ---- 轻量裁剪工具 ----
def _clip(s: Any, max_chars: int) -> str:
    """
    Safe string clipper with truncation marker.
    """
    s = "" if s is None else (s if isinstance(s, str) else json.dumps(s, ensure_ascii=False))
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n…[truncated {len(s)-max_chars} chars]"

def _limit_list(xs: List[Any] | None, max_items: int) -> List[Any]:
    xs = xs or []
    return xs[:max_items]

def _limit_fewshots(fewshots: List[Dict[str, Any]] | None, max_items: int = 6, max_chars: int = 1200) -> List[Dict[str, Any]]:
    """
    Keep only several few-shot exemplars and clip their long fields.
    """
    out = []
    for e in _limit_list(fewshots, max_items):
        if isinstance(e, dict):
            ee = {}
            for k, v in e.items():
                ee[k] = _clip(v, max_chars) if isinstance(v, str) else v
            out.append(ee)
        else:
            out.append({"example": _clip(e, max_chars)})
    return out

def _messages_len_chars(msgs: List[Dict[str, str]]) -> int:
    try:
        return sum(len(m.get("content") or "") for m in msgs)
    except Exception:
        return 0

def _hard_trim_messages(msgs: List[Dict[str, str]], budget: int = 16000) -> List[Dict[str, str]]:
    """
    If total chars exceed budget, keep system + last user only, and clip user.
    """
    total = _messages_len_chars(msgs)
    if total <= budget:
        return msgs
    # 尽量保留 system（0）和最后一条 user
    sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
    user_msg = msgs[-1] if msgs else None
    if not user_msg:
        return msgs
    keep_sys = [sys_msg] if sys_msg else []
    clipped_user = {"role": "user", "content": _clip(user_msg.get("content", ""), budget - (_messages_len_chars(keep_sys) + 200))}
    return keep_sys + [clipped_user]

def _json_from_llm_raw(raw: str) -> Dict[str, Any]:
    try:
        # 尽量取外层最大的 JSON 块
        start = raw.find("{"); end = raw.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(raw[start:end+1])
        return json.loads(raw)
    except Exception:
        return {}
    
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1:]
        if s:
            s = "".join(s)
    return s.strip()

def _first_json_block(s: str) -> str | None:
    if not s:
        return None
    s = s.strip()
    if s.lstrip().startswith("{") and s.rstrip().endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None

def _json_from_llm_raw(raw) -> dict | None:
    import json as _json
    if raw is None: return None
    if isinstance(raw, dict):
        if "content" in raw and isinstance(raw["content"], str):
            s = _strip_code_fences(raw["content"])
            block = _first_json_block(s)
            if block:
                try: return _json.loads(block)
                except Exception: pass
        if "choices" in raw:
            try:
                s = raw["choices"][0]["message"]["content"]
                s = _strip_code_fences(s)
                block = _first_json_block(s)
                if block:
                    return _json.loads(block)
            except Exception:
                pass
        return raw if raw else None
    if isinstance(raw, str):
        s = _strip_code_fences(raw)
        block = _first_json_block(s)
        if not block: return None
        try: return _json.loads(block)
        except Exception: return None
    try:
        return _json.loads(str(raw))
    except Exception:
        return None

async def _adb():
    return SessionLocal()

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _safe(obj, default=None):
    try: return obj if obj is not None else default
    except Exception: return default

def _normalize_rag(rag) -> tuple[str, list[dict]]:
    if rag is None: return "", []
    if isinstance(rag, str): return rag, []
    texts, refs = [], []
    seq = rag if isinstance(rag, list) else [rag]
    for it in seq:
        if isinstance(it, str):
            texts.append(it)
        elif isinstance(it, dict):
            texts.append(str(it.get("text", "")).strip())
            refs.append({
                "title":  it.get("title") or it.get("id") or "",
                "source": it.get("source"),
                "url":    it.get("url"),
            })
        else:
            texts.append(str(it))
    texts = [t for t in texts if t]
    return "\n\n".join(texts), refs

# -----------------------------
# NEW: 轻量规则抽取（从 user_text 兜底字段）
# -----------------------------
def _quick_parse_user_text(t: str) -> dict:
    t0 = t or ""
    tl = t0.lower()

    facet = None
    m = re.search(r"cu\s*\(?\s*(\d{3})\s*\)?", tl)
    if m: facet = f"Cu({m.group(1)})"
    if "cu111" in tl or "cu(111)" in tl: facet = "Cu(111)"
    substrate = facet or ("Cu" if "cu" in tl else None)

    ph = None
    m = re.search(r"\bph\s*=?\s*([0-9]+(?:\.[0-9]+)?)\b", tl)
    if m:
        try: ph = float(m.group(1))
        except: ph = None

    potential = None
    m = re.search(r"(-?\s*[0-9]+(?:\.[0-9]+)?)\s*v\s*(?:vs\s*)?(?:rhe|she)", tl)
    if m:
        try: potential = float(m.group(1).replace(" ", ""))
        except: potential = None

    solvent = "water" if ("water" in tl or "h2o" in tl) else None

    targets = []
    if re.search(r"\bc1\b", tl): targets.append("C1")
    if re.search(r"\bc2\b", tl): targets.append("C2")

    is_co2rr = ("co2rr" in tl) or ("co2" in tl and ("reduction" in tl or " rr" in tl))
    adsorbates = ["CO2(g)","CO2*"] if is_co2rr else []

    return {
        "stage": "catalysis",
        "area": "electro" if is_co2rr else None,
        "task": t0[:140],
        "substrate": substrate,
        "facet": facet,
        "adsorbates": adsorbates,
        "conditions": {
            "pH": ph,
            "potential_V_vs_RHE": potential,
            "solvent": solvent,
            "temperature": None,
            "electrolyte": None,
        },
        "deliverables": {
            "target_products": targets,
            "figures": ["free_energy_diagram","microkinetic_curves"],
        },
        "metrics": [
            {"name":"limiting_potential","unit":"V vs RHE"},
            {"name":"selectivity","note":"Faradaic efficiency to targets"},
        ],
        "reaction_network": {
            "intermediates": (adsorbates + ["CO*","HCOO*","CHO*"]) if is_co2rr else [],
            "steps": (
                [
                    {"reactants":["CO2*","H+","e-"], "products":["COOH*"], "kind":"PCET"},
                    {"reactants":["COOH*","H+","e-"], "products":["CO*","H2O(g)"], "kind":"PCET"},
                ] if is_co2rr else []
            ),
            "coads_pairs": []
        },
        "summary": "Parsed minimal electrochemical CO2RR context from query." if is_co2rr else t0[:140]
    }

def _merge_missing(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if k not in dst or dst[k] in (None, "", [], {}):
            dst[k] = v
        else:
            if isinstance(dst[k], dict) and isinstance(v, dict):
                _merge_missing(dst[k], v)
            elif isinstance(dst[k], list) and not dst[k] and isinstance(v, list):
                dst[k] = v

# -----------------------------
# few-shot 规则检索（IntentPhrase）
# -----------------------------
async def _fetch_fewshots(session, stage: str, area: str, task_hint: str, k: int = 6):
    stmt = select(IntentPhrase)
    if stage: stmt = stmt.where(IntentPhrase.intent_stage == stage)
    if area:  stmt = stmt.where(IntentPhrase.intent_area == area)
    if task_hint:
        stmt = stmt.where(IntentPhrase.specific_task.ilike(f"%{task_hint[:100]}%"))
    stmt = stmt.order_by(IntentPhrase.confidence.desc()).limit(k)
    res = await session.execute(stmt)
    rows = res.scalars().all()
    out = []
    for r in rows:
        out.append({
            "stage": r.intent_stage,
            "area": r.intent_area,
            "task": r.specific_task,
            "phrase": r.phrase,
            "confidence": r.confidence
        })
    return out

# -----------------------------
# 置信度计算
# -----------------------------
def _compute_confidence(intent: Dict[str, Any], fewshots: List[Dict[str, Any]], rag_refs: List[Dict[str, Any]]) -> float:
    fields = ["stage","area","task","substrate","adsorbates","conditions","metrics","reaction_network","deliverables"]
    cover = sum(1 for f in fields if intent.get(f)) / len(fields)
    fs_score  = min(len(fewshots), 6) / 6.0
    rag_score = min(len(rag_refs), 8) / 8.0

    rn = intent.get("reaction_network") or {}
    steps = (rn.get("steps") or [])
    inters = (rn.get("intermediates") or [])
    rn_score = 0.0
    if steps or inters:
        rn_score = 0.3 + 0.2 * min(len(steps), 8) / 8.0 + 0.2 * min(len(inters), 10) / 10.0
        rn_score = min(rn_score, 0.7)

    w = [0.35, 0.2, 0.2, 0.25]
    conf = w[0]*cover + w[1]*fs_score + w[2]*rag_score + w[3]*rn_score
    return round(max(0.05, min(conf, 0.98)), 3)

# -----------------------------
# 将 intent 落库
# -----------------------------
async def _persist_intent(session, session_id: int, intent: dict, rag_refs: list):
    msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        msg_type="intent",
        content=intent.get("summary") or intent.get("task") or "intent",
        intent_stage=intent.get("stage"),
        intent_area=intent.get("area"),
        specific_intent=intent.get("task"),
        confidence=float(intent.get("confidence") or 0.0),
        llm_model=intent.get("model") or "gpt-4o",
        references=rag_refs,
        attachments={
            "intent": intent,
            "reaction_network": intent.get("reaction_network"),
            "metrics": intent.get("metrics"),
            "constraints": intent.get("constraints"),
        },
        created_at=datetime.utcnow(),
    )
    session.add(msg)
    await session.flush()
    msg_id = msg.id

    hyp_id = None
    if intent.get("hypothesis"):
        hyp = Hypothesis(
            session_id=session_id,
            message_id=msg_id,
            intent_stage=intent.get("stage"),
            intent_area=intent.get("area"),
            hypothesis=intent["hypothesis"],
            confidence=float(intent.get("confidence") or 0.0),
            agent="intent_agent",
            tags=intent.get("tags"),
            created_at=datetime.utcnow(),
        )
        session.add(hyp)
        await session.flush()
        hyp_id = hyp.id

    await session.commit()
    return msg_id, hyp_id

# =========================
# 解析 helpers
# =========================
def _label_of(x) -> str:
    if isinstance(x, dict):
        return str(x.get("label") or x.get("name") or x.get("id") or "")
    if isinstance(x, (list, tuple)):
        return " + ".join(_label_of(t) for t in x)
    return "" if x is None else str(x)

def _to_list(x: Any) -> List[Any]:
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def _split_species(s: str) -> List[str]:
    toks = re.split(r"[+,/]|(?<!\() ", s)
    return [t.strip() for t in toks if t and t.strip()]

def _is_species(tok: str) -> bool:
    if not tok or " " in tok: return False
    return tok.endswith("*") or tok.endswith("(g)") or tok.endswith("(aq)")

def _extract_star(side: Any) -> List[str]:
    if isinstance(side, dict):
        raw = side.get("species") or side.get("reactants") or side.get("lhs") or side.get("side")
        return _extract_star(raw)
    seen, out = set(), []
    for item in _to_list(side):
        if isinstance(item, dict):
            tok = _label_of(item)
            if _is_species(tok) and tok not in seen:
                seen.add(tok); out.append(tok)
        elif isinstance(item, str):
            for tok in _split_species(item):
                if _is_species(tok) and tok not in seen:
                    seen.add(tok); out.append(tok)
        else:
            tok = _label_of(item)
            if _is_species(tok) and tok not in seen:
                seen.add(tok); out.append(tok)
    return out

from typing import Any, List, Tuple
def _extract_lhs_rhs(step: Any) -> Tuple[List[str], List[str]]:
    if isinstance(step, dict):
        lhs = _extract_star(step.get("reactants") or step.get("lhs") or step.get("from") or [])
        rhs = _extract_star(step.get("products")  or step.get("rhs") or step.get("to")   or [])
        return lhs, rhs
    if isinstance(step, str) and "->" in step:
        L, R = step.split("->", 1)
        return _extract_star(L), _extract_star(R)
    side = _extract_star(step)
    return side, []

# -----------------------------
# NEW: 机制识别与展开
# -----------------------------
_MECH_ALIASES = [
    # electrocatalysis
    (r"\bco2rr\b|\bco2\b.*\breduc", ["CO2RR_CO_path","CO2RR_HCOO_path","CO2RR_to_ethanol_CO_coupling"]),
    (r"\bnrr\b|\bn2\b.*\breduc",    ["NRR_distal","NRR_alternating","NRR_dissociative"]),
    (r"\borr\b|\boxygen\s+reduction", ["ORR_4e"]),
    (r"\boer\b|\boxygen\s+evolution", ["OER_lattice_oxo_skeleton"]),
    (r"\bher\b|\bhydrogen\s+evolution", ["HER_VHT"]),
    (r"\bno3rr\b|\bnitrate\b.*\breduc", ["NO3RR_to_NH3_skeleton"]),
    # thermocatalysis
    (r"\bmsr\b|\bmethane\s+steam\s+reform", ["MSR_basic"]),
    (r"\bhaber\b|\bnh3\b.*\bsynth", ["Haber_Bosch_Fe"]),
    (r"\bco\s+oxidation\b", ["CO_oxidation_LH","CO_oxidation_MvK"]),
    (r"\bisomeriz", ["Hydroisomerization_zeolite"]),
    (r"\balkylation\b", ["Alkylation_acid"]),
    (r"\bdehydration\b|\bto\s+olefin\b", ["Alcohol_dehydration"]),
    # homogeneous
    (r"\bwilkinson\b|\brhcl\(pph3\)3\b|\balkene\s+hydrogenation", ["Wilkinson_hydrogenation"]),
    (r"\bhydroformylation\b|\brh\(pph3\)3cl\b|\bhco\(co\)4\b", ["Hydroformylation_Rh"]),
    (r"\bheck\b", ["Heck_Pd"]),
    (r"\bsuzuki\b", ["Suzuki_Pd"]),
    (r"\bsonogashira\b", ["Sonogashira_Pd_Cu"]),
    (r"\bepoxidation\b|\bsharpless\b|\bjacobsen\b", ["Epoxidation_Sharpless"]),
    (r"\bnoyori\b|\bknowles\b|\basymmetric\b.*\bhydrogenation", ["Asymmetric_Hydrogenation_Noyori"]),
    # photo / photothermal
    (r"\bphotocatalysis\b|\bphoto\s+water\s+split", ["Photocatalytic_water_splitting"]),
    (r"\bphotothermal\b.*co2", ["Photothermal_CO2RR_skeleton"]),
    (r"\bphotothermal\b.*methane|\bphotothermal\b.*ch4", ["Photothermal_methane_conversion"]),
]

def _guess_mechanisms(text: str, guided: Dict[str, Any]) -> List[str]:
    tl = (text or "").lower()
    mechs: List[str] = []
    # 指定 product 线索
    if re.search(r"\bch3oh\b|\bmethanol\b", tl):
        mechs += ["CO2RR_CO_path","CO2RR_HCOO_path"]
    if re.search(r"\bethanol\b|\bch3ch2oh\b", tl):
        mechs += ["CO2RR_to_ethanol_CO_coupling"]
    # 用户可能在 guided.tags 中直呼机制键
    for k in _to_list(guided.get("tags")):
        if isinstance(k, str) and k in REGISTRY: mechs.append(k)
    # 关键词
    for pat, keys in _MECH_ALIASES:
        if re.search(pat, tl): mechs += keys
    # 去重且在 REGISTRY 中
    mechs = [k for i,k in enumerate(mechs) if k in REGISTRY and k not in mechs[:i]]
    return mechs[:4]  # 限制最多 4 条以控制任务爆量

def _expand_mechanism_network(keys: List[str], substrate: Optional[str], facet: Optional[str]) -> dict:
    """合并多个机制键的 intermediates/steps/coads，并按 variants 覆写/追加。"""
    inters, steps, coads = [], [], []
    def _apply_variant(base: dict) -> dict:
        # 选择优先：facet > substrate > 无
        var = {}
        vs = base.get("variants") or {}
        cand = None
        if facet and facet in vs: cand = vs[facet]
        elif substrate and substrate in vs: cand = vs[substrate]
        return cand or {}
    for k in keys:
        base = REGISTRY.get(k) or {}
        v = _apply_variant(base)
        inters += _to_list(base.get("intermediates")) + _to_list(v.get("intermediates"))
        steps  += _to_list(base.get("steps"))         + _to_list(v.get("steps"))
        coads  += _to_list(base.get("coads"))         + _to_list(v.get("coads"))
    # 去重（保序）
    def _uniq(seq):
        seen=set(); out=[]
        for x in seq:
            j = json.dumps(x, sort_keys=True) if isinstance(x, (dict,list)) else str(x)
            if j not in seen:
                seen.add(j); out.append(x)
        return out
    inters=_uniq(inters); steps=_uniq(steps); coads=_uniq(coads)

    # 生成 coads_pairs（字符串 "A*+B*"）
    pairs=[]
    for pair in coads:
        if isinstance(pair,(list,tuple)) and len(pair)>=2 and isinstance(pair[0],str) and isinstance(pair[1],str):
            pairs.append(f"{pair[0]}+{pair[1]}")
    # 从 steps 中额外推断
    for st in steps:
        lhs, rhs = _extract_lhs_rhs(st)
        for side in (lhs, rhs):
            ads = [x for x in side if isinstance(x,str) and x.endswith("*")]
            if len(ads) >= 2:
                pairs.append(f"{ads[0]}+{ads[1]}")
    pairs = list(dict.fromkeys(pairs))  # 去重保序

    return {"intermediates": inters, "steps": steps, "coads_pairs": pairs}

def _family_domain_from_keys(keys: List[str]) -> tuple[Optional[str], Optional[str]]:
    fam, dom = None, None
    for k in keys:
        ent = REGISTRY.get(k) or {}
        if not fam: fam = ent.get("family")
        if not dom: dom = ent.get("domain")
    return fam, dom

# -----------------------------
# 根据 intent 生成默认任务
# -----------------------------
async def _expand_workflow_tasks(session, session_id: int, message_id: int, intent: dict):
    rn = intent.get("reaction_network") or {}
    steps = rn.get("steps") or []
    inters = rn.get("intermediates") or []
    tasks_created = []
    order = 1

    async def _mk(name, agent, input_data):
        nonlocal order
        t = WorkflowTask(
            session_id=session_id,
            message_id=message_id,
            step_order=order,
            name=name,
            description=intent.get("task"),
            agent=agent,
            engine="VASP",
            status="idle",
            input_data=input_data,
            created_at=datetime.utcnow(),
        )
        session.add(t)
        await session.flush()
        tasks_created.append(t.id)
        order += 1

    # Adsorption
    for sp in _to_list(inters):
        label = _label_of(sp)
        if isinstance(label, str) and label.endswith("*"):
            await _mk(f"Adsorption {label}", "adsorption", {"species": label, "substrate": intent.get("substrate")})

    # Coadsorption
    for st in _to_list(steps):
        lhs, rhs = _extract_lhs_rhs(st)
        for pair in (lhs, rhs):
            ads = [x for x in pair if isinstance(x, str) and x.endswith("*")]
            if len(ads) >= 2:
                a, b = ads[0], ads[1]
                await _mk(f"Coadsorption {a}+{b}", "coadsorption", {"species": [a, b], "substrate": intent.get("substrate")})

    # TS / NEB
    for i, st in enumerate(_to_list(steps), 1):
        await _mk(f"TS step#{i}", "neb", {"step": st, "substrate": intent.get("substrate"), "neb_images": 5})

    # Microkinetic
    await _mk("Microkinetic Model", "microkinetic", {
        "paths": rn.get("key_paths") or [],
        "metrics": intent.get("metrics") or [],
        "conditions": intent.get("conditions") or {}
    })

    await session.commit()
    return tasks_created

# -----------------------------
# LLM 提示词（只输出 JSON）
# -----------------------------
def _intent_system_prompt():
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
        "All lists must be present (use [] if unknown). Fill best-effort defaults from context."
    )

def _make_user_prompt(user_text: str, guided: Dict[str, Any], fewshots: List[Dict[str, Any]], rag_text: str) -> List[Dict[str, str]]:
    ex = ""
    if fewshots:
        ex_lines = [f"- {fs['phrase']}" for fs in fewshots[:6]]
        ex = "Few-shot phrases (use them as hints):\n" + "\n".join(ex_lines)
    guided_txt = json.dumps(guided, ensure_ascii=False)
    return [
        {"role":"system","content":_intent_system_prompt()},
        {"role":"user","content":(
            f"USER_INQUIRY:\n{user_text}\n\n"
            f"GUIDED_FIELDS(JSON):\n{guided_txt}\n\n"
            f"{ex}\n\n"
            f"RAG_CONTEXT:\n{rag_text}\n\n"
            "Return STRICT JSON only."
        )}
    ]

# -----------------------------
# API: /chat/intent
# -----------------------------
# ==== 替换你的 api_intent ====
@router.post("/chat/intent")
async def api_intent(request: Request):

    print("This is a test")
    log.info("This is a test for chat agent")


    body = await request.json()
    session_id: int = body.get("session_id")
    user_text: str  = body.get("text") or ""
    guided: Dict[str, Any] = body.get("guided") or {}

    if not session_id or not user_text.strip():
        return JSONResponse({"ok": False, "error": "session_id and text are required"}, status_code=400)

    # 1) few-shot（已限条/限长）
    async with SessionLocal() as s:
        fewshots_raw = await _fetch_fewshots(s, guided.get("stage") or "", guided.get("area") or "", guided.get("task") or "")
    fewshots = _limit_fewshots(fewshots_raw, max_items=6, max_chars=1000)

    # 2) RAG（限长）
    try:
        rag = await rag_context(query=user_text, session_id=session_id, top_k=6)
    except Exception:
        rag = ""
    rag_text, rag_refs = _normalize_rag(_clip(rag, 4000))

    # 3) 构造紧凑消息体（不用 _make_user_prompt，自己控制体量）
    sys_prompt = (
        "You are an intent parser for computational catalysis. "
        "Return STRICT JSON only. Keys: stage, area, task, system(catalyst, facet, material, molecule), "
        "conditions(pH,potential,electrolyte,solvent,temperature,pressure), "
        "reaction_network{steps[], intermediates[], ts[], coads[]}, tags, deliverables."
    )
    # guided/knowledge/history 略去或限长；这里仅用 guided 的短字段
    guided_small = {k: _clip(v, 300) if isinstance(v, str) else v for k, v in guided.items()}

    user_payload = {
        "query": _clip(user_text, 1000),
        "guided": guided_small,
        "fewshots_hint": fewshots,      # 已限长
        "rag_hint": _clip(rag_text, 3500),
    }
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]
    messages = _hard_trim_messages(messages, budget=16000)

    # 3') 调 LLM（小模型、更小 max_tokens；失败后降载重试）
    kwargs = {"model": "gpt-4o-mini", "temperature": 0.1, "max_tokens": 900}
    try:
        raw = await chatgpt_call(messages, **kwargs)
    except Exception as e:
        # 极端情况下再降一档
        messages = _hard_trim_messages(messages, budget=8000)
        raw = await chatgpt_call(messages, **kwargs)

    intent = _json_from_llm_raw(raw)
    if not isinstance(intent, dict) or not intent:
        # 明确提醒只返回 JSON，再试一次
        messages.append({"role": "user", "content": "Return JSON only. No prose, no code fences."})
        messages = _hard_trim_messages(messages, budget=12000)
        raw = await chatgpt_call(messages, **kwargs)
        intent = _json_from_llm_raw(raw)

    if not isinstance(intent, dict) or not intent:
        preview = str(raw)
        if isinstance(preview, str) and len(preview) > 800:
            preview = preview[:800] + " ...[truncated]"
        return JSONResponse({"ok": False, "error": "LLM returned non-JSON", "llm_preview": preview, "status": 502}, status_code=502)

    # 3.5) 规则底座合并（保留你原来的逻辑）
    auto = _quick_parse_user_text(user_text)
    _merge_missing(intent, auto)

    # 3.6) 机制推断与补全（保持你原有实现）
    try:
        rn = intent.get("reaction_network") or {}
        has_rn = bool((rn.get("intermediates") or rn.get("steps")))
        mech_keys = _guess_mechanisms(user_text, guided)
        for t in _to_list(intent.get("tags")):
            if isinstance(t, str) and t in REGISTRY and t not in mech_keys:
                mech_keys.append(t)
        mech_keys = [k for i, k in enumerate(mech_keys) if k not in mech_keys[:i]]

        if mech_keys and (not has_rn or len(rn.get("steps") or []) < 2):
            expanded = _expand_mechanism_network(mech_keys, intent.get("substrate"), intent.get("facet"))
            intent.setdefault("reaction_network", {}).update(expanded)

            fam, dom = _family_domain_from_keys(mech_keys)
            if not intent.get("stage"): intent["stage"] = "catalysis"
            if not intent.get("area") and dom:
                intent["area"] = "electro" if dom == "electrocatalysis" else dom

            intent.setdefault("deliverables", {}).setdefault("figures", [])
            for fig in ["free_energy_diagram", "BEPR_scan", "microkinetic_curves", "coverage_vs_potential"]:
                if fig not in intent["deliverables"]["figures"]:
                    intent["deliverables"]["figures"].append(fig)

            tags = set(_to_list(intent.get("tags")))
            for k in mech_keys: tags.add(k)
            intent["tags"] = list(tags)

            tp = set(_to_list(intent.get("deliverables", {}).get("target_products")))
            tl = user_text.lower()
            if re.search(r"\bmethanol\b|\bch3oh\b", tl): tp.add("CH3OH")
            if re.search(r"\bethanol\b|\bch3ch2oh\b", tl): tp.add("CH3CH2OH")
            if tp:
                intent.setdefault("deliverables", {})["target_products"] = list(tp)
    except Exception as e:
        log.exception("mechanism expansion failed: %s", e)

    # 4) 标准化
    intent["stage"] = intent.get("stage") or guided.get("stage") or "catalysis"
    intent["area"]  = intent.get("area")  or guided.get("area")  or "electro"
    intent["task"]  = intent.get("task")  or guided.get("task")  or user_text[:140]
    intent["model"] = "gpt-4o-mini"

    # 5) 置信度
    conf = _compute_confidence(intent, fewshots, rag_refs)
    intent["confidence"] = conf

    # 6) 落 DB + 生成默认任务（保持你原来的逻辑）
    async with SessionLocal() as s:
        msg_id, hyp_id = await _persist_intent(s, session_id, intent, rag_refs)
        task_ids = await _expand_workflow_tasks(s, session_id, msg_id, intent)

    # 7) few-shot 沉淀（保持不变）
    try:
        phrases = intent.get("objectives") or intent.get("tags") or []
        if phrases:
            async with SessionLocal() as s:
                for p in phrases:
                    rec = IntentPhrase(
                        intent_stage=intent["stage"],
                        intent_area=intent["area"],
                        specific_task=intent["task"][:200],
                        phrase=str(p)[:500],
                        confidence=0.9,
                        source="intent_agent",
                        lang="en",
                        created_at=datetime.utcnow(),
                    )
                    s.add(rec)
                await s.commit()
    except Exception:
        pass

    summary = (
        f"[Intent] {intent['task']} | stage={intent['stage']}, area={intent['area']}, "
        f"substrate={intent.get('substrate')}, "
        f"ads={len((intent.get('reaction_network') or {}).get('intermediates',[]))} interm., "
        f"steps={len((intent.get('reaction_network') or {}).get('steps',[]))} | conf={conf}"
    )

    return JSONResponse({
        "ok": True,
        "intent": intent,
        "confidence": conf,
        "message_id": msg_id,
        "hypothesis_id": hyp_id,
        "workflow_task_ids": task_ids,
        "fewshots_used": fewshots,
        "rag_refs": rag_refs,
        "summary": summary
    }, status_code=200)