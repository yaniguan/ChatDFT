# server/chat/plan_agent.py
# -*- coding: utf-8 -*-
"""
Plan Agent — LLM + RAG + hypothesis 优先 + 轻量种子兜底

Endpoints
---------
POST /chat/plan
  - 输入: {session_id?, intent, hypothesis(str|dict)?, knowledge?, history?, query?}
  - 流程:
      1) RAG 历史/知识上下文
      2) 优先采用 hypothesis 的结构化反应网络
      3) 若缺失则 LLM 生成（注入 RAG/knowledge/history 作为 hint）
      4) 质量清洗 + 限流 + 低置信度时与种子轻度合并
      5) 产出 tasks + rxn_network（steps/inter/coads/ts）
  - 返回: {ok, steps, intermediates, coads, ts, tasks, confidence, ...}

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

# ========================= Seeds（简单反应集） =========================
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
def _build_tasks(intent: Dict[str, Any],
                 steps: List[str], inter: List[str],
                 coads_pairs: List[Tuple[str,str]], ts_edges: List[str]) -> List[Dict[str, Any]]:
    catalyst = (intent.get("system") or {}).get("catalyst") or (intent.get("system") or {}).get("material") or "Pt"
    facet    = (intent.get("system") or {}).get("facet") or "111"
    tasks: List[Dict[str, Any]] = []
    tid = 1

    def _field(key, label, ftype="text", value="", **kw):
        d = {"key": key, "label": label, "type": ftype, "value": value}
        d.update({k:v for k,v in kw.items() if v is not None})
        return d

    def _task(section, name, agent, desc, form=None, payload=None, group=0, endpoint=None):
        nonlocal tid
        t = {
            "id": tid, "section": section, "name": name, "agent": agent, "description": desc,
            "params": {"form": form or [], "payload": payload or {}},
            "meta": {"parallel_group": group, "action_endpoint": endpoint}
        }
        tid += 1
        tasks.append(t)

    # G1 slab
    _task(
        "Model", f"Build slab — {catalyst}({facet})", "structure.relax_slab",
        f"Build/relax slab for {catalyst}({facet}).",
        form=[
            _field("engine","Engine","select","vasp", options=["vasp","qe"]),
            _field("element","Element","text",catalyst),
            _field("facet","Facet","text",facet),
            _field("miller_index","Miller index","text","1 1 1" if facet=="111" else "1 0 0"),
            _field("layers","Layers","number",4, step=1, min_value=2, max_value=10),
            _field("vacuum_thickness","Vacuum (Å)","number",15.0, step=0.5, min_value=10, max_value=40),
            _field("supercell","Supercell","text","4x4x1"),
        ],
        payload={"facet": facet},
        group=1, endpoint="/agent/structure.relax_slab"
    )

    # G2 单吸附
    adsorbates = [s for s in inter if s.endswith("*")]
    for sp in adsorbates:
        _task(
            "Adsorption", f"Relax on sites — {sp}", "adsorption.scan",
            f"Enumerate adsorption sites for {sp} and relax.",
            form=[
                _field("adsorbate","Adsorbate","text",sp),
                _field("sites_csv","Sites","text","top,bridge,fcc,hcp"),
                _field("force_thr","Force (eV/Å)","number",0.02, step=0.01, min_value=0.01,max_value=0.1),
            ],
            payload={"adsorbate": sp},
            group=2, endpoint="/agent/adsorption.scan"
        )

    # G3 共吸附
    for a, b in coads_pairs[:30]:
        _task(
            "Co-adsorption", f"Co-ads — {a}+{b}", "adsorption.co",
            f"Create and relax co-adsorption configs for {a} + {b}.",
            form=[_field("pair","Pair","text",f"{a},{b}"), _field("n_configs","Configs","number",4, step=1, min_value=1, max_value=12)],
            payload={"pair":[a,b]},
            group=3, endpoint="/agent/adsorption.co"
        )

    # G4 TS
    for s in ts_edges[:20]:
        _task(
            "Transition States", f"NEB — {s}", "neb.run",
            f"CI-NEB for elementary step: {s}",
            form=[
                _field("step","Elementary step","text",s),
                _field("n_images","NEB images","number",7, step=1, min_value=3, max_value=15),
                _field("climbing","Climbing image","checkbox",True),
            ],
            payload={"step": s},
            group=4, endpoint="/agent/neb.run"
        )

    # G5 电子/后处理
    _task(
        "Electronic", "DOS/PDOS/Bader", "electronic.dos",
        "Compute DOS/PDOS/Bader for key species.",
        form=[
            _field("dos","DOS","checkbox",True),
            _field("pdos","PDOS","checkbox",True),
            _field("bader","Bader","checkbox",True),
            _field("pdos_species","PDOS species","text",", ".join(adsorbates[:6] or ["H*"])),
        ],
        payload={"species": adsorbates[:6]},
        group=5, endpoint="/agent/electronic.dos"
    )
    _task(
        "Post-analysis", "Assemble ΔG / barriers", "post.analysis",
        "Assemble ΔG profile and barrier diagram from results.",
        form=[
            _field("temperature","Temperature (K)","number",298.15, step=1, min_value=200, max_value=1000),
            _field("reference","Reference","select","RHE", options=["RHE","SHE","Ag/AgCl"]),
        ],
        payload={},
        group=5, endpoint="/agent/post.analysis"
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

    # 3) 种子（兜底）
    seed_steps, seed_inter, seed_coads = _seed_for(intent_in.get("problem_type",""))

    # 4) 拿到候选图（优先 external，若没有则让 LLM 生成；把 knowledge/history/RAG 合并进 hint）
    if external_graph:
        steps_raw = [s if isinstance(s, str) else "" for s in external_graph.get("reaction_network", [])]
        inter_raw = [s for s in external_graph.get("intermediates", [])]
        coads_raw = external_graph.get("coads_pairs", [])
        ts_raw    = [s for s in external_graph.get("ts_edges", [])]
    else:
        hint_parts = []
        if knowledge: hint_parts.append(f"[knowledge]{json.dumps(knowledge, ensure_ascii=False)[:1200]}")
        if history:   hint_parts.append(f"[history]{json.dumps(history, ensure_ascii=False)[:1200]}")
        if rag_ctx:   hint_parts.append(f"[rag]{rag_ctx[:1200]}")
        hint = "\n".join(hint_parts)

        try:
            llm_out = await _llm_generate(intent_in, hint, {"seed": {"steps": seed_steps, "intermediates": seed_inter}})
        except Exception:
            llm_out = {"steps": seed_steps, "intermediates": seed_inter, "coads": [], "ts": []}

        steps_raw = (llm_out.get("steps") or []) + (llm_out.get("ts") or [])
        inter_raw = (llm_out.get("intermediates") or [])
        coads_raw = llm_out.get("coads") or []
        ts_raw    = llm_out.get("ts") or []

    # 5) 清洗 + 限流
    steps, inter = _clean_all(steps_raw, inter_raw, LIMITS, STRICT)

    # 6) 置信度（外部图不打分；非 always 模式才打）
    conf = 0.0
    if USE_SEED_POLICY != "always" and not external_graph:
        conf = await _llm_confidence(intent_in, steps, inter)

    # 7) 低置信度/强制：轻度并入 seed
    if USE_SEED_POLICY == "always" or (USE_SEED_POLICY == "auto" and conf < CONF_THRESHOLD):
        steps = _uniq_limit(seed_steps + [s for s in steps if s not in seed_steps], LIMITS["ts"])
        inter = _uniq_limit(seed_inter + [i for i in inter if i not in seed_inter], LIMITS["inter"])

    # 8) 共吸附
    ads = [s for s in inter if s.endswith("*")]
    if not coads_raw:
        coads_pairs = sorted({tuple(sorted((a, "H*"))) for a in ads if a != "H*"})
    else:
        norm_pairs = []
        for pr in coads_raw:
            if isinstance(pr, (list, tuple)) and len(pr) == 2:
                a, b = _normalize_species(pr[0]), _normalize_species(pr[1])
                if a.endswith("*") and b.endswith("*") and a != b:
                    norm_pairs.append(tuple(sorted((a,b))))
        coads_pairs = _uniq_limit(norm_pairs, LIMITS.get("coads", 80))

    # 9) TS 边
    ts_edges = [s for s in steps if "->" in s][:LIMITS["ts"]]
    if external_graph.get("ts_edges"):
        ts_edges = _uniq_limit([_normalize_species(s) for s in external_graph["ts_edges"] if "->" in s], LIMITS["ts"])

    # 10) 构建 tasks（返回给前端）
    tasks = _build_tasks(intent_in, steps, inter, coads_pairs, ts_edges)

    result = {
        "ok": True,
        "steps": steps,
        "intermediates": inter,
        "coads": coads_pairs,
        "ts": ts_edges,
        "tasks": tasks,
        "confidence": conf,
        "limits": LIMITS,
        "use_seed_policy": USE_SEED_POLICY,
        "used_external_graph": bool(external_graph),
        "rag_context": rag_ctx,
    }

    # 11) 持久化（容错）
    try:
        await _save_artifact(session_id, "plan", result)
        await _save_artifact(session_id, "rxn_network", {
            "elementary_steps": steps,
            "intermediates": inter,
            "coads_pairs": coads_pairs,
            "ts_candidates": ts_edges,
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