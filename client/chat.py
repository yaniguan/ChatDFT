# client/home.py
# -*- coding: utf-8 -*-
"""
ChatDFT 前端（Streamlit）
Navigator:
  • Overview  —— 简介论文 & ChatDFT 功能（可编辑）
  • ChatDFT   —— 原有功能区（Tabs: Chat & Plan / Workflow / Papers/RAG / Tools / Records）
  • Projects  —— 多会话管理（选择之前的对话、创建/重命名/删除、快照保存/加载）
  • Settings  —— 集群与路径（cluster、workdir、vasp_cmd、scratch、python_env、dry_run、sync_back）
"""
from __future__ import annotations
import json, re
import pandas as pd
import streamlit as st
from utils.api import post

st.set_page_config(page_title="🔬 ChatDFT", layout="wide")

# =========================
# Session State (globals)
# =========================
DEFAULTS = {
    # active project/session
    "active_session_id": None,
    "active_session_name": "",
    # UI state payloads (当前会话的数据)
    "intent": {},
    "intent_raw": {},
    "hypothesis": "",
    "plan_tasks": [],
    "plan_raw": {},
    "rxn_net": [],
    "intermediates": [],
    "ts_candidates": [],
    "coads_pairs": [],
    "workflow_results": [],
    "selected_task_ids": [],
    # 多会话 UI 快照缓存：{session_id: {intent, hypothesis, ...}}
    "_SESSION_CACHE": {},
    # Settings
    "settings": {
        "cluster": "hoffman2",
        "workdir": "~/projects/chatdft_runs",
        "vasp_cmd": "vasp_std",
        "scratch": "/scratch/$USER",
        "python_env": "~/.conda/envs/vasp/bin/python",
        "dry_run": False,
        "sync_back": True,
    },
    # Overview markdown（可编辑）
    "overview_md": (
        "## Paper Overview\n"
        "- 在此粘贴/编辑你要介绍的论文要点（动机、方法、数据、结论、对你工作的启示）。\n\n"
        "## What is ChatDFT?\n"
        "- Intent → Hypothesis → Plan：从自然语言到可执行工作流\n"
        "- Reaction Network 抽取：自动汇总基元反应、吸附体、过渡态候选\n"
        "- Tools：并行分组执行、支持表单化参数\n"
        "- Records：执行历史与后处理摘要\n"
    ),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================
# Server API wrappers (sessions & core)
# =====================================
# ===== add near your other API wrappers =====
def fetch_session_state_from_backend(session_id: int) -> dict:
    """
    Try to fetch a structured snapshot for a session.
    1) preferred: /chat/session/state  -> returns {intent, hypothesis, plan_raw, plan_tasks, rxn_net, intermediates, ts_candidates, coads_pairs, workflow_results}
    2) fallback:  /chat/session/messages -> reconstruct from ChatMessage.msg_type
    """
    # try structured state first
    try:
        res = post("/chat/session/state", {"id": session_id}) or {}
        # if this endpoint exists and returns keys, accept it
        if any(k in res for k in ("intent", "plan_tasks", "hypothesis", "plan_raw")):
            return res
    except Exception:
        pass

    # fallback: pull recent messages and reconstruct
    try:
        msgs = post("/chat/session/messages", {"id": session_id, "limit": 500}) or {}
        items = msgs.get("messages") or []
    except Exception:
        items = []

    snap = {
        "intent": {},
        "intent_raw": {},
        "hypothesis": "",
        "plan_tasks": [],
        "plan_raw": {},
        "rxn_net": [],
        "intermediates": [],
        "ts_candidates": [],
        "coads_pairs": [],
        "workflow_results": [],
        "selected_task_ids": [],
    }

    # newest-first to take latest values
    for m in reversed(items):
        mtype = (m.get("msg_type") or "").lower()
        content = m.get("content") or ""
        # content may be JSON; be tolerant
        parsed = None
        if isinstance(content, dict):
            parsed = content
        else:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None

        if mtype in {"intent"} and parsed:
            snap["intent"] = parsed.get("intent") or parsed.get("fields") or parsed or {}
        elif mtype in {"hypothesis"}:
            snap["hypothesis"] = parsed.get("hypothesis") if parsed else content
        elif mtype in {"plan"}:
            if parsed:
                snap["plan_raw"] = parsed
                snap["plan_tasks"] = parsed.get("tasks") or snap["plan_tasks"]
        elif mtype in {"workflow_summary", "records"} and parsed:
            # optional
            snap["workflow_results"] = parsed.get("runs") or snap["workflow_results"]
        elif mtype in {"rxn_network"} and parsed:
            snap["rxn_net"]       = parsed.get("elementary_steps") or snap["rxn_net"]
            snap["intermediates"] = parsed.get("intermediates") or snap["intermediates"]
            snap["ts_candidates"] = parsed.get("ts_candidates") or snap["ts_candidates"]
            snap["coads_pairs"]   = parsed.get("coads_pairs") or snap["coads_pairs"]

    # If rxn_net/intermediates missing but we have plan_raw/tasks, derive them
    if (not snap["rxn_net"]) and (snap["plan_raw"] or snap["plan_tasks"]):
        elem, inter, ts, coads = _extract_network_from_everywhere(
            snap.get("plan_raw") or {}, snap.get("plan_tasks") or [], snap.get("hypothesis") or ""
        )
        snap["rxn_net"], snap["intermediates"], snap["ts_candidates"], snap["coads_pairs"] = elem, inter, ts, coads

    return snap

import json as _json

def _fmt_step_entry(s):
    """Normalize a step entry to a readable markdown bullet."""
    if isinstance(s, dict):
        name = s.get("name") or s.get("label")
        if name:
            why = s.get("why")
            rp  = None
            if "reactants" in s or "products" in s:
                rp = f"{s.get('reactants','?')} → {s.get('products','?')}"
            extra = f" — {why}" if why else ""
            core = f"{name}{extra}"
            return f"- {core}" + (f"\n  - {rp}" if rp else "")
        if "reactants" in s or "products" in s:
            return f"- {s.get('reactants','?')} → {s.get('products','?')}"
        return f"- {_json.dumps(s, ensure_ascii=False)}"
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return f"- {s[0]} → {s[1]}"
    return f"- {s}"

def _fmt_list_md(seq):
    seq = seq or []
    return "\n".join(_fmt_step_entry(x) for x in seq)

def _fmt_pairs_md(pairs):
    pairs = pairs or []
    lines = []
    for x in pairs:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            lines.append(f"- {x[0]} + {x[1]} (co-ads)")
        elif isinstance(x, dict) and "a" in x and "b" in x:
            lines.append(f"- {x['a']} + {x['b']} (co-ads)")
        else:
            lines.append(f"- {x}")
    return "\n".join(lines)

# ---- ChatDFT: pretty format helpers ----
import json as _json

def _fmt_step_entry(s):
    """Normalize a step entry to a readable markdown bullet."""
    # dict with explicit fields
    if isinstance(s, dict):
        name = s.get("name") or s.get("label")
        if name:
            why = s.get("why")
            rp  = None
            if "reactants" in s or "products" in s:
                rp = f"{s.get('reactants','?')} → {s.get('products','?')}"
            extra = f" — {why}" if why else ""
            core = f"{name}{extra}"
            return f"- {core}" + (f"\n  - {rp}" if rp else "")
        # dict with reactants/products
        if "reactants" in s or "products" in s:
            return f"- {s.get('reactants','?')} → {s.get('products','?')}"
        # fallback: json dump
        return f"- {_json.dumps(s, ensure_ascii=False)}"

    # list/tuple as (reactants, products)
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return f"- {s[0]} → {s[1]}"

    # plain string/other
    return f"- {s}"

def _fmt_list_md(seq):
    """Turn a heterogeneous list into markdown lines."""
    seq = seq or []
    return "\n".join(_fmt_step_entry(x) for x in seq)

def _fmt_pairs_md(pairs):
    """For co-adsorption pairs."""
    pairs = pairs or []
    out = []
    for x in pairs:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            out.append(f"- {x[0]} + {x[1]} (co-ads)")
        elif isinstance(x, dict) and "a" in x and "b" in x:
            out.append(f"- {x['a']} + {x['b']} (co-ads)")
        else:
            out.append(f"- {x}")
    return "\n".join(out)

def _fmt_step_compact(s):
    """将 step 转成紧凑的一行文字：优先显示 反应式 / 名称 / why。"""
    if isinstance(s, dict):
        r = s.get("reactants"); p = s.get("products")
        name = s.get("name") or s.get("label")
        why  = s.get("why")
        rxn  = f"{r} → {p}" if (r or p) else None
        parts = []
        if name: parts.append(str(name))
        if rxn:  parts.append(rxn)
        if why:  parts.append(why)
        return " — ".join([x for x in parts if x])
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return f"{s[0]} → {s[1]}"
    return str(s)

def _step_to_row(s):
    """用于表格：拆成多列"""
    if isinstance(s, dict):
        return {
            "Name": s.get("name") or s.get("label") or "",
            "Reaction": (f"{s.get('reactants')} → {s.get('products')}"
                         if (s.get('reactants') or s.get('products')) else ""),
            "Why": s.get("why") or "",
        }
    if isinstance(s, (list, tuple)) and len(s) >= 2:
        return {"Name": "", "Reaction": f"{s[0]} → {s[1]}", "Why": ""}
    return {"Name": "", "Reaction": str(s), "Why": ""}

def api_intent(query: str) -> dict:
    payload = {"query": query}
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id
    res = post("/chat/intent", payload) or {}
    st.session_state.intent_raw = res
    intent = res.get("intent") or res.get("fields") or {}
    if not isinstance(intent, dict):
        intent = {}
    st.session_state.intent = intent
    return intent

def api_hypothesis() -> str:
    """
    Robust hypothesis fetcher.
    Accepts server responses like:
      - "plain string"
      - {"result_md": "..."} / {"hypothesis": "..."} / {"md": "..."}
      - { ..., "steps": [...], "intermediates": [...], "ts": [...], "coads": [...] }
    Writes both the markdown text and extracted network pieces into session_state.
    """
    payload = {
        "intent": st.session_state.intent,
        "knowledge": {},
        "history": []
    }
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id

    res = post("/chat/hypothesis", payload) or {}

    # ---- normalize md text ----
    md_obj = (
        res.get("result_md")
        or res.get("hypothesis")
        or res.get("md")
        or res.get("result")
        or res
    )
    if isinstance(md_obj, dict):
        # 优先取嵌套的 "md"，否则就把 dict 转成可读 JSON 作为展示
        md_text = md_obj.get("md") if "md" in md_obj else json.dumps(md_obj, ensure_ascii=False, indent=2)
    elif isinstance(md_obj, str):
        md_text = md_obj
    else:
        md_text = str(md_obj or "")

    st.session_state.hypothesis = md_text

    # ---- pull structured fields if provided ----
    steps = res.get("steps") or res.get("elementary_steps") or []
    inter  = res.get("intermediates") or []
    ts     = res.get("ts") or res.get("ts_candidates") or []
    coads  = res.get("coads") or res.get("coads_pairs") or []

    # ---- fallback extraction (from md_text + current plan/tasks) ----
    # 用现有的抽取器把缺失项补齐
    if not (steps and inter):
        elem2, inter2, ts2, coads2 = _extract_network_from_everywhere(
            plan_res=st.session_state.get("plan_raw") or {},
            tasks=st.session_state.get("plan_tasks") or [],
            hypothesis=md_text
        )
        steps = steps or elem2
        inter = inter or inter2
        ts    = ts or ts2
        coads = coads or coads2

    # ---- commit to session_state ----
    st.session_state.rxn_net       = steps or []
    st.session_state.intermediates = inter or []
    st.session_state.ts_candidates = ts or []
    st.session_state.coads_pairs   = coads or []

    return md_text
def api_plan() -> list:
    payload = {
        "intent": st.session_state.intent,
        "hypothesis": st.session_state.hypothesis,
        "history": []
    }
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id
    res = post("/chat/plan", payload) or {}
    st.session_state["plan_raw"] = res
    tasks = res.get("tasks") or []
    st.session_state.plan_tasks = tasks
    elem, inter, ts, coads = _extract_network_from_everywhere(
        plan_res=res, tasks=tasks, hypothesis=st.session_state.hypothesis or ""
    )
    st.session_state.rxn_net        = elem
    st.session_state.intermediates  = inter
    st.session_state.ts_candidates  = ts
    st.session_state.coads_pairs    = coads
    return tasks

def api_execute(selected_ids: list):
    settings = st.session_state.settings or {}
    payload = {
        "session_id": st.session_state.active_session_id,
        "all_tasks": st.session_state.plan_tasks,
        "selected_ids": selected_ids,
        "cluster": settings.get("cluster", "hoffman2"),
        "dry_run": bool(settings.get("dry_run", False)),
        "sync_back": bool(settings.get("sync_back", True)),
        # 也可把路径类设置传给后端（若后端支持）
        "paths": {
            "workdir": settings.get("workdir"),
            "scratch": settings.get("scratch"),
            "python_env": settings.get("python_env"),
            "vasp_cmd": settings.get("vasp_cmd"),
        },
    }
    res = post("/chat/execute", payload) or {}
    if res.get("ok"):
        st.session_state.workflow_results.append(res)
    return res

def api_knowledge(q: str, use_intent: bool, limit: int, fast: bool):
    body = {"query": q, "limit": int(limit), "fast": bool(fast)}
    if use_intent:
        body["intent"] = st.session_state.intent
    if st.session_state.active_session_id:
        body["session_id"] = st.session_state.active_session_id
    return post("/chat/knowledge", body) or {}

# ---- Session management APIs（后端若无该接口，也不会崩） ----
def get_sessions() -> list[dict]:
    res = post("/chat/session/list", {}) or {}
    return res.get("sessions") or []

def create_session(name: str, project: str = "", tags: str = "", description: str = "") -> int | None:
    res = post("/chat/session/create", {
        "name": name, "project": project, "tags": tags, "description": description
    }) or {}
    return res.get("session_id")

def update_session(jaw=None, **fields) -> bool:
    if not jaw: return False
    res = post("/chat/session/update", {"id": jaw, **fields}) or {}
    return bool(res.get("ok"))

def delete_session(jaw=None) -> bool:
    if not jaw: return False
    res = post("/chat/session/delete", {"id": jaw}) or {}
    return bool(res.get("ok"))

# =========================
# Small UI helpers
# =========================

# —— 放在 Small UI helpers 下面任意位置 —— #
def _json_safe(obj):
    """将任何对象转换成可被 st.json 接收的结构。
       - dict/list 递归清理
       - Streamlit 的 DeltaGenerator 等不可序列化对象 -> repr()
    """
    try:
        json.dumps(obj)
        return obj
    except Exception:
        pass

    # DeltaGenerator 或任意不可序列化对象
    # 尽量保留关键信息，避免把 help 文档喷出来
    import streamlit.delta_generator as sdg
    if isinstance(obj, sdg.DeltaGenerator):
        return f"<UI:{obj.__class__.__name__}>"

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set,)):
        return [_json_safe(v) for v in obj]
    # 最后兜底
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"

def _render_records_block(wf: dict):
    """干净渲染单个 workflow 的记录，避免把 DeltaGenerator 当数据显示"""
    st.markdown(f"### Run — Workdir: `{wf.get('workdir','')}`")
    rows = wf.get("results") or []
    if rows:
        for r in rows:
            step = str(r.get("step", "(unnamed)"))
            status = str(r.get("status", ""))
            st.markdown(f"- **{step}** → {status}")

    # 只要 summary 可用，就做 json-safe 处理后展示
    if wf.get("summary") is not None:
        with st.expander("Post-analysis summary", expanded=False):
            st.json(_json_safe(wf["summary"]))

def _slug(s) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unnamed"

def _badge(s: str):
    st.markdown(
        f"<span style='display:inline-block; padding:6px 12px; margin:6px 8px 0 0; "
        f"background:#eef3ff; border:1px solid #dbe5ff; border-radius:999px; "
        f"font-size:13px; white-space:nowrap;'>{s}</span>",
        unsafe_allow_html=True,
    )

def _intent_summary_card(I: dict):
    """精简版：只渲染表格+原始 JSON 折叠，不再显示徽章和 RN 计数条。"""
    st.markdown("#### Intent")
    _intent_table(I or {})

def _intent_table(I: dict):
    """精简渲染 Intent：自动过滤空值；按 domain 选择字段；附带槽位缺失与澄清问题。"""
    import pandas as _pd

    def _nz(x):
        """non-empty: 去掉 None/""/[]/{}/仅空白字符串"""
        if x is None: return False
        if isinstance(x, str): return x.strip() != ""
        if isinstance(x, (list, tuple, set, dict)): return len(x) > 0
        return True

    I = I or {}
    dom  = (I.get("domain") or "").lower()
    sub  = (I.get("domain_subtype") or I.get("mode") or "").lower()
    sys  = I.get("system") or {}
    cond = I.get("conditions") or {}
    rn   = I.get("reaction_network") or {}
    miss = I.get("missing_slots") or []
    qas  = I.get("clarifying_questions") or []

    rows = []  # (Field, Value)
    def add(k, v):
        if _nz(v):
            rows.append((k, v if not isinstance(v, (list,dict)) else json.dumps(v, ensure_ascii=False)))

    # 通用最小字段
    add("domain", I.get("domain"))
    add("type", sub or None)
    add("query", I.get("normalized_query"))

    if dom in ("catalysis", "materials_general"):
        add("catalyst", sys.get("catalyst"))
        add("facet", sys.get("facet"))
        add("material", sys.get("material"))
        add("molecule/reactant", sys.get("molecule"))
        # 条件
        if sub == "electrocatalysis":
            add("pH", cond.get("pH"))
            add("potential", cond.get("potential"))
            add("electrolyte", cond.get("electrolyte"))
            add("temperature", cond.get("temperature"))
        elif sub in ("photocatalysis", "photoelectrocatalysis"):
            add("illumination", cond.get("illumination") or cond.get("light"))
            add("wavelength", cond.get("wavelength"))
            add("pH", cond.get("pH"))
            add("temperature", cond.get("temperature"))
        else:  # thermal/unspecified
            add("temperature", cond.get("temperature"))
            add("pressure", cond.get("pressure"))
            add("solvent", cond.get("solvent"))
        # RN 概览（只显示非零）
        n_steps = len(rn.get("elementary_steps") or [])
        n_inters= len(rn.get("intermediates") or [])
        n_ts    = len(rn.get("ts_candidates") or [])
        n_coads = len(rn.get("coads_pairs") or [])
        if n_steps: add("RN: steps", n_steps)
        if n_inters: add("RN: intermediates", n_inters)
        if n_ts: add("RN: ts", n_ts)
        if n_coads: add("RN: coads", n_coads)

    elif dom == "batteries":
        add("chemistry", sys.get("chemistry") or sys.get("system"))
        add("cathode",  sys.get("cathode"))
        add("anode",    sys.get("anode"))
        add("electrolyte", sys.get("electrolyte"))
        add("salt", sys.get("salt"))
        add("separator", sys.get("separator"))
        add("binder", sys.get("binder"))
        if _nz(sys.get("additives")):
            add("additives", ", ".join(sys["additives"]) if isinstance(sys["additives"], list) else sys["additives"])
        add("C-rate", cond.get("crate") or cond.get("C_rate"))
        add("temperature", cond.get("temperature"))
        add("cutoff (charge)", cond.get("v_max") or cond.get("charge_cutoff"))
        add("cutoff (discharge)", cond.get("v_min") or cond.get("discharge_cutoff"))
        add("cycles", cond.get("cycles"))

    elif dom == "polymers":
        if _nz(sys.get("monomers")):
            add("monomer(s)", ", ".join(sys["monomers"]) if isinstance(sys["monomers"], list) else sys.get("monomer"))
        add("initiator/catalyst", sys.get("initiator") or sys.get("catalyst"))
        add("method", sys.get("method") or sys.get("process"))
        add("solvent", sys.get("solvent"))
        add("temperature", cond.get("temperature"))
        add("target Mw", sys.get("target_Mw") or sys.get("Mw"))
        add("Ð (PDI)", sys.get("Đ") or sys.get("PDI"))

    # 渲染
    if rows:
        df = _pd.DataFrame(rows, columns=["Field", "Value"])
        st.table(df)
    else:
        st.info("No non-empty fields.")

    # 缺失与澄清
    if _nz(miss) or _nz(qas):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Missing slots")
            for m in (miss or []): _badge(str(m))
        with c2:
            st.caption("Clarifying questions")
            for q in (qas or []):  st.markdown(f"- {q}")

    with st.expander("Raw intent (JSON)", expanded=False):
        st.json(I)

def _badges_grid(items: list[str], cols: int = 6, empty_text: str = "N/A"):
    if not items:
        st.caption(empty_text); return
    columns = st.columns(cols)
    for i, it in enumerate(items):
        with columns[i % cols]:
            _badge(it)

def _download_bytes(label: str, data: bytes, file_name: str, help: str = ""):
    st.download_button(label, data=data, file_name=file_name, type="secondary", help=help)

def _copy_text_area(label: str, content: str):
    with st.expander(label, expanded=False):
        st.code(content or "(empty)")

def _split_parallel_groups(tasks: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for t in tasks:
        g = (t.get("params",{}).get("payload") or {}).get("parallel_group") or "Other"
        groups.setdefault(str(g), []).append(t)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x.get("id", 0))
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))

def _uniq(seq):
    out, seen = [], set()
    for s in seq:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _normalize_arrow(s: str) -> str:
    s = str(s).strip()
    s = s.replace("⇒", "->").replace("→", "->").replace("⇌", "->").replace("⟶", "->")
    s = s.replace("—>", "->").replace(" –> ", "->")
    while "  " in s:
        s = s.replace("  ", " ")
    parts = [p.strip() for p in s.split("->") if p.strip()]
    if len(parts) == 2:
        return f"{parts[0]} → {parts[1]}"
    return s

_RX_ADS = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\*$")
_RX_GAS = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\((?:g|aq)\)$")

def _looks_like_species(tok: str) -> bool:
    tok = tok.strip()
    if not tok or " " in tok or len(tok) > 12:
        return False
    if tok.count("*") > 1 or tok.count("(") > 1 or tok.count(")") > 1:
        return False
    return bool(_RX_ADS.match(tok) or _RX_GAS.match(tok))

def _only_species(tokens: list[str]) -> list[str]:
    return [t for t in tokens if _looks_like_species(t)]

import re
from typing import Any, Dict, List, Tuple

# ⚠️ 替换原函数定义整段
from typing import Any, Dict, List, Tuple
import re

def _extract_network_from_everywhere(
    plan_res: Any,
    tasks: List[dict],
    hypothesis: Any
) -> Tuple[List[Any], List[str], List[str], List[str]]:
    """从 plan_res / tasks / hypothesis 文本里尽可能抽取
       elem(steps), inter(species), ts, coads；对所有输入做类型容错。
       注意：elem 保留 dict 结构（含 name/reactants/products/why），便于前端表格渲染。
    """
    def _to_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        try:
            return str(x)
        except Exception:
            return ""

    def _normalize_arrow(s: str) -> str:
        s = _to_str(s).strip()
        if not s:
            return ""
        s = (s.replace("⇒", "->").replace("→", "->").replace("⇌", "->").replace("⟶", "->")
               .replace("—>", "->").replace(" –> ", "->"))
        while "  " in s:
            s = s.replace("  ", " ")
        parts = [p.strip() for p in s.split("->") if p.strip()]
        if len(parts) == 2:
            return f"{parts[0]} → {parts[1]}"
        return s

    _RX_ADS = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\*$")
    _RX_GAS = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\((?:g|aq)\)$")

    def _looks_like_species(tok: Any) -> bool:
        tok = _to_str(tok).strip()
        if not tok or " " in tok or len(tok) > 20:
            return False
        if tok.count("*") > 1 or tok.count("(") > 1 or tok.count(")") > 1:
            return False
        return bool(_RX_ADS.match(tok) or _RX_GAS.match(tok))

    def _only_species(tokens: List[Any]) -> List[str]:
        out = []
        for t in tokens:
            if _looks_like_species(t):
                out.append(_to_str(t))
        return out

    def _uniq_mixed(seq: List[Any]) -> List[Any]:
        """支持 dict 的去重；dict 用排序后的 JSON 作 key，其它用字符串。"""
        seen = set()
        out: List[Any] = []
        for s in seq:
            if isinstance(s, dict):
                try:
                    key = json.dumps(s, ensure_ascii=False, sort_keys=True)
                except Exception:
                    key = str(s)
            else:
                key = _to_str(s)
            key = key.strip()
            if not key:
                continue
            if key not in seen:
                seen.add(key); out.append(s)
        return out

    # -------- 开始抽取 --------
    elem: List[Any] = []
    inter: List[str] = []
    ts: List[str] = []
    coads: List[str] = []

    # 1) 从 tasks 的 payload 中拿
    for t in (tasks or []):
        payload = (t.get("params") or {}).get("payload") or {}
        elem  += payload.get("elementary_steps") or []
        inter += payload.get("intermediates")   or []
        ts    += payload.get("ts_candidates")   or []
        coads += payload.get("coads_pairs")     or []

    # 2) 从 plan 顶层拿（兼容各种键名）
    if isinstance(plan_res, dict):
        elem  += (plan_res.get("elementary_steps")
                  or plan_res.get("reaction_network")
                  or plan_res.get("steps")
                  or [])
        inter += plan_res.get("intermediates")   or []
        ts    += (plan_res.get("ts_candidates")
                  or plan_res.get("ts_edges")
                  or plan_res.get("ts")
                  or [])
        coads += plan_res.get("coads_pairs")     or plan_res.get("coads") or []

    # 3) 从 task 名称粗略挖（例如 "NEB — A -> B"）
    for t in (tasks or []):
        name = _to_str(t.get("name"))
        if ("NEB" in name or "TS" in name) and any(x in name for x in ["->", "→", "⇒", "⟶"]):
            frag = name.split("—", 1)[-1].strip() if "—" in name else name
            ts.append(_normalize_arrow(frag))
        for tok in re.split(r"[,\s/+-]+", name):
            if _looks_like_species(tok):
                inter.append(_to_str(tok))

    # 4) 从 hypothesis 文本兜底挖（保证是字符串）
    hypo_text = _to_str(hypothesis)
    if hypo_text:
        rough = re.findall(r"\b([A-Za-z0-9()*]+)\b", hypo_text)
        inter += _only_species(rough)

    # -------- 规范化/去重 --------
    # 仅对“非 dict”的 elem 做规范化；dict 原样保留
    elem_norm: List[Any] = []
    for x in elem:
        if isinstance(x, dict):
            elem_norm.append(x)
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            elem_norm.append({"reactants": x[0], "products": x[1]})
        else:
            elem_norm.append(_normalize_arrow(x))
    elem = _uniq_mixed(elem_norm)

    # TS 统一规范为字符串，便于右侧 badges 展示
    ts_norm: List[str] = []
    for x in ts:
        if isinstance(x, dict):
            r = x.get("reactants") or x.get("from") or x.get("src")
            p = x.get("products")  or x.get("to")   or x.get("dst")
            if r or p:
                ts_norm.append(_normalize_arrow(f"{r} -> {p}"))
            elif x.get("name"):
                ts_norm.append(str(x["name"]))
            else:
                ts_norm.append(_normalize_arrow(str(x)))
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            ts_norm.append(_normalize_arrow(f"{x[0]} -> {x[1]}"))
        else:
            ts_norm.append(_normalize_arrow(x))
    ts = _uniq_mixed(ts_norm)  # 这里 ts 是 List[str]

    inter = _uniq_mixed(_only_species(inter))   # List[str]
    coads = _uniq_mixed([_to_str(c) for c in coads])  # List[str]

    # 如果没有 elem 但有 ts，用 ts 兜底为字符串步骤
    if not elem and ts:
        elem = ts[:]  # 前端也能渲染（当作 Reaction 列文本）

    # 中间体排序：先吸附物，再气相
    ads = [s for s in inter if s.endswith("*")]
    gas = [s for s in inter if s.endswith("(g)") or s.endswith("(aq)")]
    inter_sorted = ads + gas + [s for s in inter if s not in ads and s not in gas]

    return elem, inter_sorted, ts, coads
# =========================
# Per-session snapshot I/O
# =========================
_SNAPSHOT_KEYS = [
    "intent", "intent_raw", "hypothesis", "plan_tasks", "plan_raw",
    "rxn_net", "intermediates", "ts_candidates", "coads_pairs",
    "workflow_results", "selected_task_ids",
]
def _save_snapshot(session_id: int | None):
    if not session_id: return
    cache = st.session_state._SESSION_CACHE
    cache[int(session_id)] = {k: st.session_state.get(k) for k in _SNAPSHOT_KEYS}
    st.session_state._SESSION_CACHE = cache

def _load_snapshot(session_id: int | None):
    if not session_id: return False
    snap = st.session_state._SESSION_CACHE.get(int(session_id))
    if not snap:  # new/空会话：清空关键区
        for k in _SNAPSHOT_KEYS:
            st.session_state[k] = DEFAULTS.get(k, [] if k.endswith("s") else {})
        st.session_state["hypothesis"] = ""
        st.session_state["plan_tasks"] = []
        st.session_state["plan_raw"] = {}
        st.session_state["rxn_net"] = []
        st.session_state["intermediates"] = []
        st.session_state["ts_candidates"] = []
        st.session_state["coads_pairs"] = []
        st.session_state["workflow_results"] = []
        st.session_state["selected_task_ids"] = []
        return False
    for k, v in snap.items():
        st.session_state[k] = v
    return True

# =========================
# Reusable sections
# =========================
def section_overview():
    st.title("📘 Overview")
    st.markdown("在这里简介**你要讨论的论文**与 **ChatDFT** 的核心功能。下方文本框可直接编辑（保存在会话状态，不落盘）。")
    st.text_area("Overview content (Markdown)", key="overview_md", height=260)
    st.markdown("---")
    st.markdown("**Current Active Session**")
    sid = st.session_state.active_session_id
    if sid:
        st.success(f"Active session: [{sid}] {st.session_state.active_session_name or '(unnamed)'}")
    else:
        st.info("No active session. 请到 **Projects** 选择或创建一个。")

def section_chatdft_with_tabs():
    st.title("🧠 ChatDFT")
    # 顶部显示当前会话
    sid = st.session_state.active_session_id
    if sid:
        st.caption(f"Active session: [{sid}] {st.session_state.active_session_name or '(unnamed)'}")
    else:
        st.warning("No active session. 建议在 **Projects** 选择/创建会话（不影响使用，仅影响记录与回溯）。")

    tab_chat, tab_workflow, tab_papers= st.tabs(
        ["💬 Chat & Plan", "🧪 Workflow", "📑 Papers / RAG"]
    )

    # --- Chat & Plan ---
    with tab_chat:
        st.subheader("User Inquiry → Intent → Hypothesis → Plan")
        if st.button("↻ Re-extract from current plan"):
            res  = st.session_state.get("plan_raw") or {}
            tasks= st.session_state.get("plan_tasks") or []
            elem, inter, ts, coads = _extract_network_from_everywhere(
                plan_res=res, tasks=tasks, hypothesis=st.session_state.hypothesis or ""
            )
            st.session_state.rxn_net        = elem
            st.session_state.intermediates  = inter
            st.session_state.ts_candidates  = ts
            st.session_state.coads_pairs    = coads
            st.success("Re-extracted.")

        query = st.text_area("Your question / task:", placeholder="e.g., CO2RR on Cu(111), pH=12, -0.5 V vs RHE …")
        c1, c2, c3 = st.columns(3)
        if c1.button("Generate Intent", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Parsing intent…"): api_intent(query)
            else:
                st.warning("Please enter a question first.")
        if c2.button("Generate Hypothesis", disabled=not st.session_state.intent, use_container_width=True):
            with st.spinner("Generating hypothesis…"): api_hypothesis()
        if c3.button("Generate Plan", disabled=(not st.session_state.hypothesis), use_container_width=True):
            with st.spinner("Building workflow plan…"):
                tasks = api_plan()
                if tasks: st.success(f"Plan generated with {len(tasks)} tasks.")
                else: st.info("No tasks produced.")

        st.markdown("---")
        _intent_summary_card(st.session_state.intent or {})
        st.markdown("#### Hypothesis")
        st.markdown(st.session_state.hypothesis or "_(empty)_")

        if st.session_state.plan_tasks:
            st.markdown("---")
            st.markdown("#### Planned Tasks (flat view)")
            for t in st.session_state.plan_tasks:
                st.markdown(f"- **[{t.get('id','?')}] {t.get('name','Task')}** · {t.get('description','')}")

    # --- Workflow ---
    def _workflow_right_panel():
        inter = st.session_state.intermediates or []
        ts    = st.session_state.ts_candidates or []
        coads = st.session_state.coads_pairs or []
        st.markdown("**Intermediates**")
        ads = [s for s in inter if s.endswith("*")]
        gas = [s for s in inter if s.endswith("(g)") or s.endswith("(aq)")]
        st.caption("Adsorbates (* on surface)"); _badges_grid(ads, cols=5, empty_text="None")
        st.caption("Gas / solution references"); _badges_grid(gas, cols=5, empty_text="None")
        if inter:
            csv = "species\n" + "\n".join(inter)
            _download_bytes("⬇️ Download intermediates.csv", csv.encode("utf-8"), "intermediates.csv")
            _copy_text_area("Copy intermediates (CSV)", csv)
        st.markdown("---"); st.markdown("**TS candidates**")
        _badges_grid(ts, cols=3, empty_text="No TS candidates.")
        if ts:
            ts_txt = "\n".join(ts)
            _download_bytes("⬇️ Download TS list", ts_txt.encode("utf-8"), "ts_candidates.txt")
            _copy_text_area("Copy TS candidates", ts_txt)
        st.markdown("---"); st.markdown("**Co-ads pairs**")
        _badges_grid(coads, cols=3, empty_text="No Co-ads pairs.")
        if coads:
            coads_txt = _fmt_pairs_md(coads)
            _download_bytes("⬇️ Download coads.txt", coads_txt.encode("utf-8"), "coads_pairs.txt")
            _copy_text_area("Copy co-ads pairs", coads_txt)

    with tab_workflow:
        st.subheader("Reaction Network & Intermediates")
        left, right = st.columns([1.3, 1])
        with left:
            steps = st.session_state.rxn_net or []
            st.markdown("**Elementary steps**")

            if steps:
                rows = [_step_to_row(s) for s in steps]
                df_steps = pd.DataFrame(rows)
                df_steps.insert(0, "#", list(range(1, len(rows)+1)))
            else:
                df_steps = pd.DataFrame(columns=["#","Name","Reaction","Why"])

            st.dataframe(df_steps, use_container_width=True, hide_index=True)

            # 下载/复制：用紧凑格式，一行一个
            lines = [_fmt_step_compact(s) for s in (steps or [])]
            txt = "\n".join(lines)
            _download_bytes("⬇️ Download steps.txt", txt.encode("utf-8"), "elementary_steps.txt")
            _copy_text_area("Copy steps (plain text)", txt)
        with right: _workflow_right_panel()

    # --- Papers / RAG ---
    with tab_papers:
        st.subheader("arXiv search (fast) + optional intent context")
        q = st.text_input("Keywords", key="paper_kw", placeholder="e.g., CO2 reduction Cu(111) alkaline kinetics")
        colA, colB, colC, colD = st.columns([1,1,1,3])
        use_intent = colA.checkbox("Use intent context", value=True)
        fast_mode  = colB.checkbox("Fast", value=True, help="仅 arXiv + 轻量 enrich")
        limit      = colC.number_input("Max papers", value=10, step=1, min_value=1, max_value=50)
        if colD.button("🔎 Search", type="primary"):
            if not q.strip() and not use_intent:
                st.warning("Enter keywords or enable intent context.")
            else:
                with st.spinner("Searching arXiv…"):
                    res = api_knowledge(q, use_intent, int(limit), fast_mode)
            papers = res.get("records") or []
            if not papers:
                st.info("No papers found.")
            else:
                show = []
                for p in papers:
                    show.append({
                        "Title": p.get("title"),
                        "Venue": p.get("venue"),
                        "Year":  p.get("year"),
                        "URL":   p.get("url"),
                        "DOI":   p.get("doi"),
                        "Relevance": f"{p.get('relevance',0):.2f}",
                    })
                st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True)
                lines = []
                for p in papers:
                    parts = [str(p.get("title") or "").strip()]
                    if p.get("venue"): parts.append(p["venue"])
                    if p.get("year"):  parts.append(str(p["year"]))
                    if p.get("doi"):   parts.append(f"doi:{p['doi']}")
                    if p.get("url"):   parts.append(p["url"])
                    lines.append(" — ".join([x for x in parts if x]))
                _copy_text_area("Copy top results (plain)", "\n".join(lines))
                st.caption(res.get("result") or "")


def section_projects():
    st.title("📂 Projects (Sessions)")
    # 上一次激活的会话先做一次快照
    _save_snapshot(st.session_state.active_session_id)

    # 列出现有会话
    sessions = get_sessions()
    if not sessions:
        st.info("No sessions yet. Create one below.")
    else:
        # 列表视图
        data = []
        for s in sessions:
            data.append({
                "ID": s.get("id"),
                "Name": s.get("name"),
                "Project": s.get("project"),
                "Tags": s.get("tags"),
                "Status": s.get("status"),
                "Pinned": s.get("pinned"),
                "Updated": s.get("updated_at"),
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("---")
    colL, colR = st.columns([2,2])
    # 选择已有会话并激活（自动加载快照）
    with colL:
        sid_options = [(f"[{s.get('id')}] {s.get('name')}", s.get("id")) for s in sessions] if sessions else []
        chosen = st.selectbox("Select a session to activate", sid_options, index=0 if sid_options else None,
                              format_func=lambda t: t[0] if isinstance(t, tuple) else str(t))
        if sid_options and st.button("✅ Set Active"):
            _, sid = chosen
            # save current
            _save_snapshot(st.session_state.active_session_id)

            # switch active
            st.session_state.active_session_id = sid
            st.session_state.active_session_name = next((s.get("name") for s in sessions if s.get("id")==sid), "")

            # try load from cache; if not present, hydrate from backend
            loaded = _load_snapshot(sid)
            if not loaded:
                snap = fetch_session_state_from_backend(sid)
                # install into session_state and cache
                for k, v in snap.items():
                    st.session_state[k] = v
                _save_snapshot(sid)

            st.success(f"Activated session [{sid}] and loaded state.")
            # go straight to ChatDFT
            st.session_state["nav"] = "ChatDFT"
            st.rerun()

    with colR:
        st.markdown("**Quick actions**")
        if sessions:
            # default to active session if available
            default_sid = st.session_state.active_session_id or (sessions[0].get("id"))
            sid2 = st.selectbox(
                "Pick a session to modify",
                [s.get("id") for s in sessions],
                index=[s.get("id") for s in sessions].index(default_sid) if default_sid in [s.get("id") for s in sessions] else 0,
                key="sid_to_mod"
            )
            new_name = st.text_input("Rename to", value="", placeholder="leave blank to keep")
            pin_val  = st.checkbox("Pinned", value=next((bool(s.get("pinned")) for s in sessions if s.get("id")==sid2), False))

            cA, cB = st.columns(2)
            if cA.button("💾 Apply (rename/pin)"):
                fields = {"pinned": pin_val}
                if new_name.strip():
                    fields["name"] = new_name.strip()
                ok = update_session(jaw=sid2, **fields)
                st.success("Updated.") if ok else st.error("Update failed.")
                st.rerun()

            if cB.button("🗑 Delete this session"):
                ok = delete_session(jaw=sid2)
                # if you deleted the active session, clear active
                if ok and st.session_state.active_session_id == sid2:
                    st.session_state.active_session_id = None
                    st.session_state.active_session_name = ""
                st.success("Deleted.") if ok else st.error("Delete failed.")
                st.rerun()

def section_settings():
    st.title("⚙️ Settings (Cluster & Paths)")
    s = st.session_state.settings
    col1, col2 = st.columns([1,1])
    with col1:
        s["cluster"] = st.text_input("Cluster name", value=s.get("cluster","hoffman2"))
        s["workdir"] = st.text_input("Workdir (on cluster)", value=s.get("workdir","~/projects/chatdft_runs"))
        s["scratch"] = st.text_input("Scratch path", value=s.get("scratch","/scratch/$USER"))
    with col2:
        s["vasp_cmd"]   = st.text_input("VASP command", value=s.get("vasp_cmd","vasp_std"))
        s["python_env"] = st.text_input("Python env (full path)", value=s.get("python_env","~/.conda/envs/vasp/bin/python"))
        s["dry_run"]    = st.checkbox("Dry run (no submit)", value=bool(s.get("dry_run", False)))
        s["sync_back"]  = st.checkbox("Sync back results", value=bool(s.get("sync_back", True)))
    if st.button("Save settings", type="primary"):
        st.session_state.settings = s
        st.success("Settings saved (kept in session_state).")

# =========================
# Sidebar Navigator
# =========================
# =========================
# Sidebar (new, like your mock)
# =========================
with st.sidebar:
    st.markdown("### Section")
    section = st.selectbox(
        "Section",
        ["Overview", "ChatDFT", "Settings"],
        index=["Overview","ChatDFT","Settings"].index(st.session_state.get("nav","ChatDFT")),
        label_visibility="collapsed"
    )
    st.session_state["nav"] = section

    st.markdown("### Chat Sessions")

    # --- New Chat ---
    new_box = st.container()
    with new_box:
        c1, c2 = st.columns([3,1])
        new_name = c1.text_input("＋ New Chat", placeholder="Name (e.g., CO2RR Cu(111))", label_visibility="visible")
        if c2.button("Create", use_container_width=True):
            if not new_name.strip():
                st.warning("Please input a name.")
            else:
                sid = create_session(name=new_name.strip())
                if sid:
                    # 给新会话建空快照并设为激活
                    _save_snapshot(st.session_state.active_session_id)
                    st.session_state.active_session_id = sid
                    st.session_state.active_session_name = new_name.strip()
                    _load_snapshot(sid)  # 空会话清空UI
                    _save_snapshot(sid)
                    st.success(f"Created & activated [{sid}] {new_name.strip()}")
                    st.session_state["nav"] = "ChatDFT"
                    st.rerun()
                else:
                    st.error("Create failed.")

    # --- Open existing ---
    sessions = get_sessions()
    sid_options = [(f"{s.get('name') or '(unnamed)'}  ·  #{s.get('id')}", s.get("id")) for s in sessions]
    # 默认选中当前激活的会话
    default_idx = 0
    if st.session_state.active_session_id and sid_options:
        ids = [sid for _, sid in sid_options]
        if st.session_state.active_session_id in ids:
            default_idx = ids.index(st.session_state.active_session_id)

    open_label = "Open chat"
    open_sel = st.selectbox(open_label, sid_options, index=default_idx if sid_options else None,
                            format_func=lambda t: t[0] if isinstance(t, tuple) else str(t),
                            label_visibility="visible", key="open_chat_sel")

    # 选择变化即激活 + hydrate
    if "___last_open_sid" not in st.session_state:
        st.session_state.___last_open_sid = None
    current_sid = open_sel[1] if open_sel else None
    if current_sid and current_sid != st.session_state.___last_open_sid:
        # 切换会话：先保存旧，再加载新
        _save_snapshot(st.session_state.active_session_id)
        st.session_state.active_session_id = current_sid
        st.session_state.active_session_name = next((s.get("name") for s in sessions if s.get("id")==current_sid), "")

        loaded = _load_snapshot(current_sid)
        if not loaded:
            snap = fetch_session_state_from_backend(current_sid)
            for k, v in snap.items():
                st.session_state[k] = v
            _save_snapshot(current_sid)

        st.session_state.___last_open_sid = current_sid
        # 跳到 ChatDFT，直接看到已回填的 intent/hypothesis/plan
        st.session_state["nav"] = "ChatDFT"
        st.rerun()

    st.markdown("---")
    st.markdown("### ✏️ Edit Chat")

    # 当前激活会话信息
    active_sid = st.session_state.active_session_id
    st.caption(f"Active: [{active_sid}] {st.session_state.active_session_name or '(none)'}" if active_sid else "Active: (none)")

    # 保存编辑：仅改名/置顶（可按需扩展）
    ec1, ec2 = st.columns([3,1])
    new_title = ec1.text_input("Rename", value="", placeholder="leave blank to keep")
    pin_state = ec2.checkbox("Pinned", value=bool(next((s.get("pinned") for s in sessions if s.get('id')==active_sid), False)) if active_sid else False)
    if st.button("💾 Save Edit", use_container_width=True, disabled=not active_sid):
        fields = {"pinned": pin_state}
        if new_title.strip(): fields["name"] = new_title.strip()
        ok = update_session(jaw=active_sid, **fields)
        st.success("Saved.") if ok else st.error("Save failed.")
        st.rerun()

    if st.button("❌ Delete Chat", use_container_width=True, disabled=not active_sid, type="secondary"):
        ok = delete_session(jaw=active_sid)
        if ok:
            # 清空激活态
            st.session_state.active_session_id = None
            st.session_state.active_session_name = ""
            st.session_state.___last_open_sid = None
            st.success("Deleted.")
            st.rerun()
        else:
            st.error("Delete failed.")


# =========================
# Router
# =========================
if st.session_state["nav"] == "Overview":
    section_overview()
elif st.session_state["nav"] == "ChatDFT":
    section_chatdft_with_tabs()
elif st.session_state["nav"] == "Projects":
    section_projects()
else:
    section_settings()
