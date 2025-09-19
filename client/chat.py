# client/home.py
# -*- coding: utf-8 -*-
"""
ChatDFT 前端（Streamlit）
Navigator:
  • Overview  —— 简介论文 & ChatDFT 功能（可编辑）
  • ChatDFT   —— 原有功能区（Tabs: Chat & Plan / Workflow / Papers/RAG / Tools / Records）
  • Settings  —— 集群与路径（cluster、workdir、vasp_cmd、scratch、python_env、dry_run、sync_back）
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# 你自己的 API 包装
from utils.api import post, get  # 如需 GET： from utils.api import post, get

st.set_page_config(page_title="🔬 ChatDFT", layout="wide")

# =========================
# Session State (globals)
# =========================
DEFAULTS = {
    "active_session_id": None,
    "active_session_name": "",
    "intent": {},
    "intent_raw": {},
    "hypothesis": "",
    "plan_tasks": [],
    "plan_raw": {},
    "rxn_net": [],
    "intermediates": [],
    "ts_candidates": [],
    "coads_pairs": [],
    "workflow_results": [],        # 关键：初始化为列表，避免 None.append
    "selected_task_ids": [],
    "_SESSION_CACHE": {},
    "settings": {
        "cluster": "hoffman2",
        "workdir": "~/projects/chatdft_runs",
        "vasp_cmd": "vasp_std",
        "scratch": "/scratch/$USER",
        "python_env": "~/.conda/envs/vasp/bin/python",
        "dry_run": False,
        "sync_back": True,
    },
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
# Server API helpers
# =====================================

def _force_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x) or {}
        except Exception:
            return {}
    return {}

def get_sessions() -> list[dict]:
    res = post("/chat/session/list", {}) or {}
    return res.get("sessions") or []

def create_session(name: str, project: str = "", tags: str = "", description: str = "") -> int | None:
    res = post("/chat/session/create", {"name": name, "project": project, "tags": tags, "description": description}) or {}
    return res.get("session_id")

def update_session(jaw=None, **fields) -> bool:
    if not jaw: return False
    res = post("/chat/session/update", {"id": jaw, **fields}) or {}
    return bool(res.get("ok"))

def delete_session(jaw=None) -> bool:
    if not jaw: return False
    res = post("/chat/session/delete", {"id": jaw}) or {}
    return bool(res.get("ok"))

def fetch_session_state_from_backend(session_id: int) -> dict:
    """优先拉结构化快照；失败则回退从消息重建。"""
    try:
        res = post("/chat/session/state", {"id": session_id}) or {}
        if any(k in res for k in ("intent", "plan_tasks", "hypothesis", "plan_raw")):
            return res
    except Exception:
        pass

    try:
        msgs = post("/chat/session/messages", {"id": session_id, "limit": 500}) or {}
        items = msgs.get("messages") or []
    except Exception:
        items = []

    snap = {
        "intent": {}, "intent_raw": {}, "hypothesis": "",
        "plan_tasks": [], "plan_raw": {},
        "rxn_net": [], "intermediates": [], "ts_candidates": [], "coads_pairs": [],
        "workflow_results": [], "selected_task_ids": [],
    }

    for m in reversed(items):
        mtype = (m.get("msg_type") or "").lower()
        content = m.get("content") or ""
        parsed = None
        if isinstance(content, dict):
            parsed = content
        else:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None

        if mtype == "intent" and parsed:
            snap["intent"] = parsed.get("intent") or parsed.get("fields") or parsed or {}
        elif mtype == "hypothesis":
            snap["hypothesis"] = parsed.get("hypothesis") if parsed else content
        elif mtype == "plan" and parsed:
            snap["plan_raw"] = parsed
            snap["plan_tasks"] = parsed.get("tasks") or snap["plan_tasks"]
        elif mtype in {"workflow_summary", "records"} and parsed:
            snap["workflow_results"] = parsed.get("runs") or snap["workflow_results"]
        elif mtype == "rxn_network" and parsed:
            snap["rxn_net"]       = parsed.get("elementary_steps") or snap["rxn_net"]
            snap["intermediates"] = parsed.get("intermediates") or snap["intermediates"]
            snap["ts_candidates"] = parsed.get("ts_candidates") or snap["ts_candidates"]
            snap["coads_pairs"]   = parsed.get("coads_pairs") or snap["coads_pairs"]

    if (not snap["rxn_net"]) and (snap["plan_raw"] or snap["plan_tasks"]):
        elem, inter, ts, coads = _extract_network_from_everywhere(
            snap.get("plan_raw") or {}, snap.get("plan_tasks") or [], snap.get("hypothesis") or ""
        )
        snap["rxn_net"], snap["intermediates"], snap["ts_candidates"], snap["coads_pairs"] = elem, inter, ts, coads
    return snap

# =========================
# Pretty helpers
# =========================
def _badge(s: str):
    st.markdown(
        f"<span style='display:inline-block; padding:6px 12px; margin:6px 8px 0 0; "
        f"background:#eef3ff; border:1px solid #dbe5ff; border-radius:999px; "
        f"font-size:13px; white-space:nowrap;'>{s}</span>",
        unsafe_allow_html=True,
    )

def _nonempty(x) -> bool:
    if x is None: return False
    if isinstance(x, str): return x.strip() != ""
    if isinstance(x, (list,tuple,dict,set)): return len(x) > 0
    return True

def _kv_badge(label: str, value) -> None:
    if value is None or value == "" or value == []:
        return
    _badge(f"{label}: {value}")

def _badges_grid(items: list[str], cols: int = 6, empty_text: str = "N/A"):
    if not items:
        st.caption(empty_text); return
    columns = st.columns(cols)
    for i, it in enumerate(items):
        with columns[i % cols]:
            _badge(it)

def _json_safe(obj):
    """把不可序列化对象转换为字符串，避免 st.json 报错/展示奇怪对象。"""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        pass
    import streamlit.delta_generator as sdg
    if isinstance(obj, sdg.DeltaGenerator):
        return f"<UI:{obj.__class__.__name__}>"
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"

# === New helpers for Intent UI enrichment ===
def _confidence_bar(conf: float | None):
    try:
        c = float(conf or 0.0)
    except Exception:
        c = 0.0
    c = max(0.0, min(c, 1.0))
    # Color hint via markdown bar
    color = "#35c759" if c >= 0.8 else ("#ff9f0a" if c >= 0.6 else "#ff3b30")
    pct = int(round(c * 100))
    st.markdown(f"Confidence: {c:.2f}")
    st.markdown(
        f"<div style='height:8px;background:#eee;border-radius:6px;'>"
        f"<div style='height:8px;width:{pct}%;background:{color};border-radius:6px;'></div>"
        f"</div>", unsafe_allow_html=True
    )

def _prov_chip(field: str, value: str | None, *, user: str | None, rag_refs: list[dict] | None):
    if not value:
        return
    src = "LLM"
    try:
        if isinstance(user, str) and value and value.lower() in user.lower():
            src = "USER"
        elif rag_refs:
            # if value appears in any ref title, mark as RAG
            for r in rag_refs[:5]:
                t = str((r or {}).get("title") or "")
                if t and value.lower() in t.lower():
                    src = "RAG"; break
    except Exception:
        pass
    _kv_badge(f"{field}·{src}", value)

def _intent_warnings(I: dict) -> list[str]:
    w = []
    if not I.get("substrate") and not (I.get("system") or {}).get("material"):
        w.append("Missing substrate/material")
    rn = I.get("reaction_network") or {}
    if (rn.get("steps") or []) and not (rn.get("intermediates") or []):
        w.append("Reaction steps present but no intermediates extracted")
    cond = (I.get("conditions") or {})
    # Potential units
    if "potential" in cond and "potential_V_vs_RHE" not in cond:
        w.append("Potential unit ambiguous (expect V vs RHE)")
    return w

def _download_bytes(label: str, data: bytes, file_name: str, help: str = ""):
    st.download_button(label, data=data, file_name=file_name, type="secondary", help=help)

def _copy_text_area(label: str, content: str):
    with st.expander(label, expanded=False):
        st.code(content or "(empty)")

# =========================
# Intent / Hypothesis / Plan
# =========================
def _extract_conditions_from_intent(I: dict) -> dict:
    I = I or {}
    cond = I.get("conditions") or {}
    ph   = cond.get("pH") or cond.get("ph")
    U    = cond.get("potential_V_vs_RHE") or cond.get("potential") or cond.get("U")
    T    = cond.get("temperature")
    elec = cond.get("electrolyte")
    solv = cond.get("solvent")
    press= cond.get("pressure")
    light= cond.get("illumination") or cond.get("light")
    return {"pH": ph, "U": U, "T": T, "electrolyte": elec, "solvent": solv, "pressure": press, "illumination": light}

def _get_mechanisms_from_state() -> list[str]:
    pr = st.session_state.get("plan_raw") or {}
    wf = pr.get("workflow") or {}
    mechs = wf.get("mechanisms") or []
    if not mechs:
        intent = st.session_state.get("intent")
        if isinstance(intent, dict):
            mechs = intent.get("tags") or []
        else:
            mechs = []
    out, seen = [], set()
    for m in mechs:
        s = str(m).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _intent_table(intent: dict) -> str:
    if not intent:
        return "_(No intent data)_"
    lines = ["| Key | Value |", "| --- | --- |"]
    for k, v in intent.items():
        if isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

def _intent_summary_card(I: dict):
    I = _force_dict(I or {})
    st.markdown("#### Intent")

    # Prefer new v2 schema if present
    raw = st.session_state.get("intent_raw") or {}
    v2 = raw.get("intent_v2") if isinstance(raw.get("intent_v2"), dict) else {}

    # Extract headline components
    sys  = I.get("system") or {}
    fam  = (I.get("area") or I.get("problem_type") or I.get("stage") or v2.get("domain") or "catalysis").upper()
    surface_v2 = (v2.get("target_system") or {}).get("surface") if isinstance(v2.get("target_system"), dict) else None
    cat  = surface_v2 or sys.get("catalyst") or sys.get("material") or I.get("substrate") or "-"
    facet= sys.get("facet") or I.get("facet") or "-"

    # Derive adsorbate: intent_v2 > intent.adsorbates > RN intermediates
    ads_v2 = v2.get("adsorbate") if isinstance(v2, dict) else None
    ads_list = I.get("adsorbates") or []
    rn = I.get("reaction_network") or {}
    def _first_adsorbate():
        if isinstance(ads_v2, str) and ads_v2.strip():
            return ads_v2
        for x in ads_list:
            s = str(x).strip()
            if s: return s[:-1] if s.endswith("*") else s
        for x in rn.get("intermediates") or []:
            s = str(x).strip()
            if s: return s[:-1] if s.endswith("*") else s
        return None
    ads = _first_adsorbate()

    # Task + confidence
    task = I.get("task") or I.get("normalized_query") or v2.get("intent_type") or "(no task text)"
    conf = raw.get("confidence")

    # RN counts
    n_steps = len(rn.get("steps") or rn.get("elementary_steps") or [])
    n_inters= len(rn.get("intermediates") or [])
    n_ts    = len(rn.get("ts") or rn.get("ts_candidates") or [])
    n_coads = len(rn.get("coads_pairs") or rn.get("coads") or [])

    # Compose richer header: ELECTRO · CO on Pt(111)
    header = f"**{fam}** · **{cat}({facet})**"
    if cat and ads:
        # If cat already includes (111), avoid double facet by using cat as-is
        header = f"**{fam}** · **{ads} on {cat}**"

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown(header)
        st.caption(task)

        # Target system badges (from v2 if available)
        ts = v2.get("target_system") if isinstance(v2.get("target_system"), dict) else {}
        cond = _extract_conditions_from_intent(I)
        with st.container():
            if ts:
                _kv_badge("surface", ts.get("surface"))
                _kv_badge("supercell", ts.get("supercell"))
                _kv_badge("vacuum", ts.get("vacuum"))
            _kv_badge("pH", cond.get("pH"))
            _kv_badge("U (vs RHE)", cond.get("U"))
            _kv_badge("T (K)", cond.get("T"))
            _kv_badge("electrolyte", cond.get("electrolyte"))
            _kv_badge("solvent", cond.get("solvent"))
            _kv_badge("illum.", cond.get("illumination"))
            _kv_badge("P", cond.get("pressure"))

        # Parameters (v2)
        params = v2.get("parameters") if isinstance(v2.get("parameters"), dict) else {}
        if params:
            st.caption("Parameters")
            with st.container():
                for k in ("exchange_correlation","spin_polarization"):
                    if k in params:
                        _kv_badge(k, params.get(k))

        # Workflow (v2)
        steps = v2.get("workflow") if isinstance(v2.get("workflow"), list) else []
        if steps:
            st.caption("Workflow")
            st.markdown("\n".join([f"{i+1}. {s}" for i, s in enumerate(steps[:8])]))

        # Deliverables (v2 or legacy)
        dels_v2 = v2.get("deliverables") if isinstance(v2.get("deliverables"), list) else []
        dels_legacy = I.get("deliverables")
        if isinstance(dels_legacy, list):
            tps = dels_legacy
        elif isinstance(dels_legacy, dict):
            tps = dels_legacy.get("target_products") or []
        else:
            tps = []
        items = dels_v2 or tps
        if _nonempty(items):
            st.caption("Deliverables")
            _badges_grid([str(x) for x in items], cols=4)

        # Next steps (v2)
        nxt = v2.get("next_step") if isinstance(v2.get("next_step"), list) else []
        if _nonempty(nxt):
            st.caption("Next Steps")
            _badges_grid([str(x) for x in nxt], cols=4)

    with c2:
        # Confidence bar
        _confidence_bar(conf)

        mechs = _get_mechanisms_from_state()
        st.caption("Mechanisms (matched)")
        if mechs:
            _badges_grid(mechs, cols=3)
        else:
            st.write("_None_")
        st.caption("Mechanisms (expanded)")
        st.write("_None_")
        st.caption("RN summary")
        st.markdown(
            f"- steps: **{n_steps}**  \n"
            f"- intermediates: **{n_inters}**  \n"
            f"- TS candidates: **{n_ts}**  \n"
            f"- co-ads: **{n_coads}**"
        )
        # RAG refs (short list)
        rag_refs = (st.session_state.get("intent_raw") or {}).get("rag_refs") or []
        if rag_refs:
            st.caption("RAG refs")
            for r in rag_refs[:3]:
                t = str((r or {}).get("title") or "")
                u = str((r or {}).get("url") or "")
                if u:
                    st.markdown(f"- [{t}]({u})")
                else:
                    st.markdown(f"- {t}")

    # Unique intent: research question, context, handoff notes
    uq = (st.session_state.get("intent_raw") or {}).get("intent_unique") or {}
    if isinstance(uq, dict) and any(uq.get(k) for k in ("research_question","scientific_context","handoff_notes")):
        st.markdown("---")
        st.subheader("Scientific Intent")
        if uq.get("research_question"):
            st.markdown(f"**Research question:** {uq['research_question']}")
        if uq.get("scientific_context"):
            st.caption(f"Context: {uq['scientific_context']}")
        hn = uq.get("handoff_notes") if isinstance(uq.get("handoff_notes"), dict) else {}
        if hn:
            st.caption("Handoff Notes")
            with st.container():
                _kv_badge("→ Hypothesis", hn.get("to_hypothesis_agent"))
                _kv_badge("→ Plan", hn.get("to_plan_agent"))

    # Show raw table as a collapsible expander for diagnostics
    with st.expander("Raw intent (debug)"):
        st.markdown(_intent_table(I))
        st.caption("Provenance (heuristic)")
        user_q = st.session_state.get("last_query") or ""
        with st.container():
            _prov_chip("substrate", cat if isinstance(cat, str) else str(cat), user=user_q, rag_refs=rag_refs)
            _prov_chip("facet", facet, user=user_q, rag_refs=rag_refs)
            if ads:
                _prov_chip("adsorbate", ads, user=user_q, rag_refs=rag_refs)
        warns = _intent_warnings(I)
        if warns:
            st.caption("Warnings")
            for w in warns:
                st.markdown(f"- {w}")

def _parse_hypothesis_md(md_text: str) -> dict:
    if not isinstance(md_text, str) or not md_text.strip():
        return {"conditions": "", "hypothesis": "", "why": [], "next": [], "exp": []}
    txt = md_text.strip().replace("\r\n","\n").replace("\r","\n")

    def _find_block(label: str) -> Tuple[int,int]:
        pat = rf"(?mi)^\s*(\*\*)?{re.escape(label)}(\*\*)?\s*:\s*"
        m = re.search(pat, txt)
        if not m: return (-1,-1)
        start = m.end()
        ANY = r"(Conditions|Hypothesis|Why it may be true|What to compute next|Optional experimental validation)"
        m2 = re.search(rf"(?mi)^\s*(\*\*)?{ANY}(\*\*)?\s*:\s*", txt[start:])
        end = start + m2.start() if m2 else len(txt)
        return (start, end)

    def _first_line(s: str) -> str:
        if not s: return ""
        line = s.split("\n",1)[0]
        line = re.sub(r"^\s*[-*]\s*", "", line).strip()
        line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line)
        return line

    def _bullets(s: str) -> list:
        if not s: return []
        items = []
        for line in s.split("\n"):
            m = re.match(r"^\s*[-*]\s+(.*)$", line)
            if m: items.append(m.group(1).strip())
        return items

    blk = {k: txt[s:e].strip() if s>=0 else "" for k,(s,e) in {
        "Conditions": _find_block("Conditions"),
        "Hypothesis": _find_block("Hypothesis"),
        "Why": _find_block("Why it may be true"),
        "Next": _find_block("What to compute next"),
        "Exp": _find_block("Optional experimental validation"),
    }.items()}

    return {
        "conditions": _first_line(blk["Conditions"]),
        "hypothesis": _first_line(blk["Hypothesis"]),
        "why":  _bullets(blk["Why"]),
        "next": _bullets(blk["Next"]),
        "exp":  _bullets(blk["Exp"]),
    }

def _add_meta_from_hypothesis(next_bullets: list[str]):
    if not next_bullets:
        st.warning("No items found in 'What to compute next'."); return
    tasks = st.session_state.get("plan_tasks") or []
    meta_task = {
        "id": 0,
        "section": "Meta",
        "name": "Plan notes — from Hypothesis",
        "agent": "meta.clarify",
        "description": "Imported from Hypothesis · What to compute next",
        "params": {
            "form": [],
            "payload": {"notes": next_bullets, "source": "hypothesis", "parallel_group": 0}
        },
        "meta": {"parallel_group": 0, "action_endpoint": None}
    }
    replaced = False
    for i, t in enumerate(tasks):
        if (t.get("agent") == "meta.clarify") and "notes" in ((t.get("params") or {}).get("payload") or {}):
            tasks[i] = meta_task; replaced = True; break
    if not replaced:
        tasks = [meta_task] + tasks
    st.session_state.plan_tasks = tasks
    plan_raw = st.session_state.get("plan_raw") or {}
    if isinstance(plan_raw, dict):
        plan_raw["tasks"] = tasks
        st.session_state.plan_raw = plan_raw
    st.success(f"Added {len(next_bullets)} note(s) to plan as meta.clarify.")

# =========================
# Plan → RN 抽取（严格 1:1）
# =========================
def _extract_network_from_everywhere(plan_res: Any, tasks: List[dict], hypothesis: Any
) -> Tuple[List[Any], List[str], List[str], List[str]]:

    def _to_str(x: Any) -> str:
        if x is None: return ""
        if isinstance(x, str): return x
        try: return str(x)
        except Exception: return ""

    def _normalize_arrow(s: str) -> str:
        s = _to_str(s).strip()
        if not s: return ""
        s = (s.replace("⇒","->").replace("→","->").replace("⇌","->").replace("⟶","->")
               .replace("—>","->").replace(" –> ","->"))
        while "  " in s: s = s.replace("  "," ")
        parts = [p.strip() for p in s.split("->") if p.strip()]
        return f"{parts[0]} → {parts[1]}" if len(parts)==2 else s

    _RX_ADS = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\*$")
    _RX_GAS = re.compile(r"^(?:[A-Z][a-z]?\d*)+(?:[A-Z][a-z]?\d*)*\((?:g|aq)\)$")

    def _looks_like_species(tok: Any) -> bool:
        tok = _to_str(tok).strip()
        if not tok or " " in tok or len(tok) > 20: return False
        if tok.count("*") > 1 or tok.count("(") > 1 or tok.count(")") > 1: return False
        return bool(_RX_ADS.match(tok) or _RX_GAS.match(tok))

    def _only_species(tokens: List[Any]) -> List[str]:
        return [_to_str(t) for t in tokens if _looks_like_species(t)]

    plan = plan_res if isinstance(plan_res, dict) else {}
    plan_elem  = (plan.get("elementary_steps")
                  or plan.get("reaction_network")
                  or plan.get("steps") or [])
    plan_inter = plan.get("intermediates") or []
    plan_ts    = (plan.get("ts_candidates") or plan.get("ts_edges") or plan.get("ts") or [])
    plan_coads = plan.get("coads_pairs") or plan.get("coads") or []

    elem_rows: List[Dict[str, Any]] = []
    ts_list: List[str] = []
    inter_list: List[str] = []
    coads_list: List[str] = []

    ARROWS = ("->","→","⇒","⟶")

    def _parse_reaction_from_task(t: dict) -> str:
        payload = (t.get("params") or {}).get("payload") or {}
        step_txt = _to_str(payload.get("step"))
        if any(a in step_txt for a in ARROWS):
            return _normalize_arrow(step_txt)
        for f in (t.get("params") or {}).get("form") or []:
            if str(f.get("key")).lower() == "step":
                v = _to_str(f.get("value"))
                if any(a in v for a in ARROWS):
                    return _normalize_arrow(v)
        name = _to_str(t.get("name"))
        s = name
        if "—" in s: s = s.split("—", 1)[1]
        elif "–" in s: s = s.split("–", 1)[1]
        elif " - " in s: s = s.split(" - ", 1)[1]
        for tail in ("·", "•", " CI-NEB", " NEB", "(CI-NEB)"):
            if tail in s:
                s = s.split(tail, 1)[0]
        s = s.strip()
        if any(a in s for a in ARROWS):
            return _normalize_arrow(s)
        return ""

    for t in (tasks or []):
        row: Dict[str, Any] = {"Name": _to_str(t.get("name") or "Task")}
        rxn = _parse_reaction_from_task(t)
        if rxn:
            row.update({
                "reaction": rxn,
                "reactants": rxn.split(" → ")[0] if " → " in rxn else "",
                "products":  rxn.split(" → ")[1] if " → " in rxn else "",
            })
            ts_list.append(rxn)
            toks = re.split(r"[+\s→]+", rxn.replace("(", " ").replace(")", " "))
            inter_list += _only_species(toks)
        name_tokens = re.split(r"[,\s/+\-·•]+", _to_str(t.get("name")))
        inter_list += _only_species(name_tokens)
        payload = (t.get("params") or {}).get("payload") or {}
        inter_list += payload.get("intermediates") or []
        ts_list    += payload.get("ts_candidates") or []
        coads_list += payload.get("coads_pairs") or []
        elem_rows.append({
            "name": row.get("Name"),
            "reaction": row.get("reaction",""),
            "reactants": row.get("reactants",""),
            "products": row.get("products",""),
        })

    extras: List[Dict[str, Any]] = []
    for x in plan_elem:
        if isinstance(x, dict):
            extras.append({
                "name": x.get("name") or x.get("label") or "",
                "reaction": _normalize_arrow(
                    f"{x.get('reactants')} -> {x.get('products')}"
                    if (x.get('reactants') or x.get('products')) else _to_str(x)
                ),
                "reactants": _to_str(x.get("reactants") or ""),
                "products":  _to_str(x.get("products") or ""),
            })
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            rxn = _normalize_arrow(f"{x[0]} -> {x[1]}")
            extras.append({"name":"", "reaction":rxn,
                           "reactants":rxn.split(" → ")[0], "products":rxn.split(" → ")[1]})
        else:
            rxn = _normalize_arrow(_to_str(x))
            extras.append({"name":"", "reaction":rxn,
                           "reactants":rxn.split(" → ")[0] if " → " in rxn else "",
                           "products":rxn.split(" → ")[1] if " → " in rxn else ""})

    elem = elem_rows + extras
    inter_list += plan_inter
    ts_list    += plan_ts
    coads_list += plan_coads

    hypo_text = _to_str(hypothesis)
    if hypo_text:
        rough = re.findall(r"\b([A-Za-z0-9()*]+)\b", hypo_text)
        inter_list += [s for s in rough if _looks_like_species(s)]

    def _uniq_order(seq):
        seen, out = set(), []
        for s in seq:
            k = _to_str(s)
            if k and k not in seen:
                seen.add(k); out.append(s)
        return out

    intermediates = _uniq_order(inter_list)
    ts_clean = _uniq_order([_normalize_arrow(x) for x in ts_list if _to_str(x)])
    coads_clean = _uniq_order([_to_str(c) for c in coads_list if _to_str(c)])

    ads = [s for s in intermediates if s.endswith("*")]
    gas = [s for s in intermediates if s.endswith("(g)") or s.endswith("(aq)")]
    rest = [s for s in intermediates if s not in ads and s not in gas]
    intermediates_sorted = ads + gas + rest

    return elem, intermediates_sorted, ts_clean, coads_clean

# =========================
# Backend calls (intent/hypo/plan/execute)
# =========================
def api_intent(query: str) -> dict:
    sid = st.session_state.get("active_session_id")
    if not sid:
        new_name = (st.session_state.get("active_session_name") or "").strip() or "Untitled"
        sid = create_session(name=new_name)
        if not sid:
            st.error("Failed to create session for intent."); return {}
        st.session_state.active_session_id = sid
        st.session_state.active_session_name = new_name

    guided = {"stage": "catalysis", "area": "electro", "task": query.strip()[:140]}
    payload = {"session_id": sid, "text": (query or "").strip(), "guided": guided}
    res = post("/chat/intent", payload) or {}
    st.session_state["last_query"] = (query or "").strip()

    st.session_state.intent_raw = res
    intent = res.get("intent") or res.get("fields") or {}
    st.session_state.intent = _force_dict(intent)

    if not res.get("ok"):
        st.error(res.get("error") or "Intent API error")
    else:
        st.success(f"Intent generated. confidence={res.get('confidence', 0):.2f}")
    return intent

def api_hypothesis() -> str:
    payload = {"intent": st.session_state.intent, "knowledge": {}, "history": []}
    # Prefer richer schemas if available from last Intent call
    raw_intent = st.session_state.get("intent_raw") or {}
    if isinstance(raw_intent, dict):
        if isinstance(raw_intent.get("intent_v2"), dict):
            payload["intent_v2"] = raw_intent.get("intent_v2")
        if isinstance(raw_intent.get("intent_unique"), dict):
            payload["intent_unique"] = raw_intent.get("intent_unique")
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id
    res = post("/chat/hypothesis", payload) or {}

    md_text = ""
    if isinstance(res, str):
        md_text = res
    elif isinstance(res, dict):
        cand = (
            res.get("result_md") or res.get("md")
            or (res.get("hypothesis") or {}).get("result_md")
            or (res.get("hypothesis") or {}).get("md")
            or res.get("hypothesis")
        )
        if isinstance(cand, str): md_text = cand
        elif isinstance(cand, dict) and isinstance(cand.get("md"), str): md_text = cand["md"]
        elif isinstance(cand, dict) and isinstance(cand.get("result_md"), str): md_text = cand["result_md"]
        else:
            try: md_text = json.dumps(res, ensure_ascii=False, indent=2)
            except Exception: md_text = str(res)
    else:
        md_text = str(res or "")

    st.session_state.hypothesis = md_text or ""

    steps = (
        res.get("steps") or res.get("elementary_steps") or
        (res.get("hypothesis") or {}).get("steps") or []
    )
    inter = res.get("intermediates") or (res.get("hypothesis") or {}).get("intermediates") or []
    ts = (
        res.get("ts") or res.get("ts_candidates") or
        (res.get("hypothesis") or {}).get("ts") or (res.get("hypothesis") or {}).get("ts_candidates") or []
    )
    coads = (
        res.get("coads") or res.get("coads_pairs") or
        (res.get("hypothesis") or {}).get("coads") or (res.get("hypothesis") or {}).get("coads_pairs") or []
    )

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

    st.session_state.rxn_net       = steps or []
    st.session_state.intermediates = inter or []
    st.session_state.ts_candidates = ts or []
    st.session_state.coads_pairs   = coads or []

    # Store structured hypotheses if provided
    hs = res.get("hypotheses_structured") if isinstance(res, dict) else []
    if isinstance(hs, list):
        st.session_state["hypotheses_structured"] = hs
    else:
        st.session_state["hypotheses_structured"] = []
    # Aggregate recommendations (top-level)
    recs = res.get("recommendations_structured") if isinstance(res, dict) else []
    if isinstance(recs, list):
        st.session_state["hypotheses_recommendations"] = recs
    else:
        st.session_state["hypotheses_recommendations"] = []
    return md_text

def api_plan() -> list:
    payload = {"intent": st.session_state.intent, "hypothesis": st.session_state.hypothesis, "history": []}
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id

    res = post("/chat/plan", payload) or {}
    st.session_state["plan_raw"] = res

    tasks = res.get("tasks") or []
    st.session_state.plan_tasks = tasks

    steps = res.get("steps") or res.get("elementary_steps") or []
    inter = res.get("intermediates") or []
    ts    = res.get("ts") or res.get("ts_candidates") or []
    coads = res.get("coads") or res.get("coads_pairs") or []

    if not steps and not inter and not ts and not coads:
        elem2, inter2, ts2, coads2 = _extract_network_from_everywhere(
            plan_res=res, tasks=tasks, hypothesis=st.session_state.hypothesis or ""
        )
        steps, inter, ts, coads = elem2, inter2, ts2, coads2

    st.session_state.rxn_net        = steps or []
    st.session_state.intermediates  = inter or []
    st.session_state.ts_candidates  = ts or []
    st.session_state.coads_pairs    = coads or []
    return tasks

def _get_hpc_session_uid() -> str:
    """
    懒创建一个 HPC session（供 /agent/run、/job/list 使用）。
    - server 侧已有 /session/create 返回 {session_uid: "..."}
    - 只创建一次，存到 st.session_state.hpc_session_uid
    """
    if st.session_state.get("hpc_session_uid"):
        return st.session_state["hpc_session_uid"]

    # 用当前 Chat 会话名拼一个更友好的 HPC 会话名
    name = (st.session_state.get("active_session_name") or "chatdft").strip() or "chatdft"
    try:
        r = post("/session/create", {"name": name}) or {}
        uid = r.get("session_uid") or r.get("uid")
        if not uid:
            # 后端没回 uid，就再兜底列一下
            ls = post("/session/list", {}) or []
            if ls and isinstance(ls, list) and isinstance(ls[0], dict):
                uid = ls[-1].get("uid")
        if not uid:
            raise RuntimeError("No session_uid returned from /session/create")
        st.session_state["hpc_session_uid"] = uid
        return uid
    except Exception as e:
        st.error(f"Create HPC session failed: {e}")
        # 返回空字符串，调用方会据此不携带 uid（但就看不到 Job 归属）
        return ""
    
def api_execute(selected_ids: list): # Here is where we start to sumbit the tasks
    """
    提交当前 plan 中被勾选的任务到 /agent/run。
    - 自动兜底 workflow_results 为 list
    - 逐任务提交，聚合返回
    - 使用 Settings 中的 cluster/dry_run/sync_back
    - ★ 新增：携带 HPC session_uid，确保 HPC 监控页能看到这些 Job
    """
    if not isinstance(st.session_state.get("workflow_results"), list):
        st.session_state.workflow_results = []

    settings = st.session_state.get("settings") or {}
    all_tasks = st.session_state.get("plan_tasks") or []
    pick = set(selected_ids or [])
    tasks = [t for t in all_tasks if t and t.get("id") in pick]

    if not tasks:
        return {"ok": False, "detail": "No tasks selected."}

    # 取得/创建 HPC session_uid（client/app.py 的监控按它查询）
    hpc_sess_uid = _get_hpc_session_uid()

    out = {"ok": True, "submitted": [], "errors": [], "session_uid": hpc_sess_uid}

    for t in tasks:
        agent = (t.get("agent") or "").strip().lower()
        payload = (t.get("params") or {}).get("payload") or {}

        body = {
            "agent": agent, # No agent selection needed OK!
            "task": t,  # 兼容后端流水线
            "engine": (payload.get("engine") or "vasp"),
            "cluster": settings.get("cluster", "hoffman2"),
            # dry_run=True => 不提交；submit 取反
            "submit": not bool(settings.get("dry_run", False)),
            "wait": False,
            "fetch": bool(settings.get("sync_back", True)),
            "job_name": t.get("name") or f"task_{t.get('id')}",
        }
        # ★ 关键：把 HPC session_uid 传给后端，这样 Job 会挂到对应会话
        if hpc_sess_uid:
            body["session_uid"] = hpc_sess_uid

        try:
            r = post("/agent/execute", body) or {} # Not exist
            if r.get("ok", True):  # 有些实现不回 ok 字段，视为成功
                out["submitted"].append({"task_id": t.get("id"), "name": t.get("name"), "resp": r})
            else:
                out["ok"] = False
                out["errors"].append({"task_id": t.get("id"), "name": t.get("name"), "error": r.get("detail") or "unknown"})
        except Exception as e:
            out["ok"] = False
            out["errors"].append({"task_id": t.get("id"), "name": t.get("name"), "error": str(e)})

    st.session_state.workflow_results.append(out)
    return out

# =========================
# Execution 持久化（双路由回退）
# =========================
def _task_type_from_agent(agent: str, fallback: str = "model") -> str:
    a = (agent or "").lower()
    if a.startswith("structure."): return "structure"
    if a.startswith("adsorption.co"): return "coadsorption"
    if a.startswith("adsorption."): return "adsorption"
    if a.startswith("neb."): return "neb"
    if a.startswith("electronic.") or "dos" in a or "bader" in a: return "dos"
    if a.startswith("post."): return "post"
    if a in {"run_dft"}: return "model"
    return fallback

def _build_exec_tasks_payload(session_id: int, tasks: list[dict], only_ids: set[int] | None = None) -> dict:
    items = []
    order = 1
    for t in (tasks or []):
        tid = t.get("id")
        if only_ids is not None and tid not in only_ids:
            continue
        title = t.get("name") or f"Task {order}"
        agent = (t.get("agent") or t.get("section") or "")
        task_type = _task_type_from_agent(agent)
        items.append({
            "order_idx": order,
            "title": title,
            "task_type": task_type,
            "payload": (t.get("params") or {}).get("payload") or {}
        })
        order += 1
    return {"session_id": int(session_id), "tasks": items}

def _get_plan_tasks_strict() -> list[dict]:
    tasks = st.session_state.get("plan_tasks")
    if isinstance(tasks, list):
        return [t for t in tasks if isinstance(t, dict)]
    plan_raw = st.session_state.get("plan_raw")
    if isinstance(plan_raw, dict):
        t = plan_raw.get("tasks")
        if isinstance(t, list):
            return [x for x in t if isinstance(x, dict)]
    return []

def api_persist_execution(selected_only: bool = False) -> dict:
    """保存任务到 Execution；优先 /exec/tasks/commit；只有在抛异常时才回退到 /task/commit。"""
    sid = st.session_state.get("active_session_id")
    if not sid:
        return {"ok": False, "detail": "No active session_id."}

    tasks = _get_plan_tasks_strict()
    if not tasks:
        return {"ok": False, "detail": "No tasks in plan."}

    if selected_only:
        sel = set(st.session_state.get("selected_task_ids") or [])
        tasks = [t for t in tasks if t.get("id") in sel]
    if not tasks:
        return {"ok": False, "detail": "No selected tasks to save."}

    # 统一构造 payload：新路由使用规范化后的 payload
    exec_payload = _build_exec_tasks_payload(sid, tasks)

    # route 1: /exec/tasks/commit（只要 2xx 就认为成功）
    try:
        res = post("/exec/tasks/commit", exec_payload) or {}
        return {"ok": True, "route": "exec", "resp": res}
    except Exception as e1:
        last_err = str(e1)

    # route 2: 仅当上面抛异常时，回退旧路由
    try:
        legacy_payload = {"session_id": sid, "tasks": tasks}
        res2 = post("/task/commit", legacy_payload) or {}
        return {"ok": True, "route": "legacy", "resp": res2}
    except Exception as e2:
        return {"ok": False, "detail": f"exec route failed: {last_err}; legacy route failed: {e2}"}

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
    if not snap:
        # 清空为默认
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
        st.info("No active session. 请到 **Settings** 或侧边栏创建。")

def _render_hypothesis_block(md_text: str):
    st.markdown("#### Hypothesis")
    if not md_text:
        st.info("Empty hypothesis."); return
    parsed = _parse_hypothesis_md(md_text)

    a, b = st.columns([1,1])
    with a:
        st.markdown("**Conditions**")
        st.write(parsed.get("conditions") or "_N/A_")
    with b:
        st.markdown("**Hypothesis**")
        st.write(parsed.get("hypothesis") or "_N/A_")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("**Why it may be true**")
        items = parsed.get("why") or []
        if items:
            for it in items:
                st.markdown(f"- {it}")
        else: st.write("_N/A_")
    with c2:
        st.markdown("**What to compute next**")
        nxt = parsed.get("next") or []
        if nxt:
            for it in nxt:
                st.markdown(f"- {it}")
        else: st.write("_N/A_")
        if nxt and st.button("➕ Add 'What to compute next' to plan (meta.clarify)", use_container_width=True):
            _add_meta_from_hypothesis(nxt)
    with c3:
        st.markdown("**Optional experimental validation**")
        items = parsed.get("exp") or []
        if items:
            for it in items:
                st.markdown(f"- {it}")
        else: st.write("_N/A_")

def _render_hypotheses_structured(items: list[dict] | None):
    st.markdown("#### Hypotheses (Structured)")
    hs = items or []
    if not hs:
        st.info("No structured hypotheses.")
        return
    for i, h in enumerate(hs, 1):
        title = h.get("title") or f"Hypothesis #{i}"
        st.markdown(f"**{title}**")
        cols = st.columns(2)
        with cols[0]:
            r = h.get("rationale") or []
            if r:
                st.caption("Rationale")
                for x in r[:6]:
                    st.markdown(f"- {x}")
            p = h.get("predictions") or h.get("predicted_outcomes") or []
            if isinstance(p, str): p = [p]
            if p:
                st.caption("Predicted outcomes")
                for x in p[:6]:
                    st.markdown(f"- {x}")
        with cols[1]:
            u = h.get("uncertainties") or []
            if isinstance(u, str): u = [u]
            if u:
                st.caption("Uncertainties / Assumptions")
                for x in u[:6]:
                    st.markdown(f"- {x}")
            rec = h.get("recommendations") or []
            if isinstance(rec, str): rec = [rec]
            if rec:
                st.caption("Recommendations for Plan Agent")
                # per-item selection checkboxes
                selected = []
                for j, x in enumerate(rec[:12]):
                    if st.checkbox(f"{x}", key=f"hrec_{i}_{j}"):
                        selected.append(str(x))
                csel1, csel2 = st.columns([1,1])
                with csel1:
                    if st.button(f"➕ Add selected (H{i})") and selected:
                        _add_meta_from_hypothesis(selected)
                with csel2:
                    if st.button(f"➕ Add all (H{i})"):
                        _add_meta_from_hypothesis([str(x) for x in rec])

def _render_hypotheses_recs(items: list[str] | None):
    recs = items or []
    if not recs: return
    st.markdown("#### Recommendations (Aggregated)")
    # simple chips with selection
    chosen = []
    for idx, x in enumerate(recs[:20]):
        if st.checkbox(f"{x}", key=f"hrec_all_{idx}"):
            chosen.append(str(x))
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("➕ Add selected to plan") and chosen:
            _add_meta_from_hypothesis(chosen)
    with c2:
        if st.button("➕ Add all to plan"):
            _add_meta_from_hypothesis([str(x) for x in recs])

def _step_to_row(s):
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

def _fmt_step_compact(s):
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

def _render_tasks_selector():
    """在 Workflow 页签渲染可勾选的计划任务 + 执行按钮。"""
    tasks = st.session_state.get("plan_tasks") or []

    ids_now = [t.get("id") for t in tasks if isinstance(t, dict)]
    ids_prev = st.session_state.get("selected_task_ids") or []
    if sorted(ids_now) != sorted([i for i in ids_prev if i in ids_now]):
        st.session_state.selected_task_ids = []

    st.markdown("### Planned Tasks")
    if not tasks:
        st.info("No tasks yet. Click **Generate Plan** in the Chat & Plan tab.")
        return

    c1, c2, c3, _ = st.columns([1,1,1,3])
    with c1:
        if st.button("Select all", use_container_width=True):
            st.session_state.selected_task_ids = ids_now[:]
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state.selected_task_ids = []
    with c3:
        if st.button("Invert", use_container_width=True):
            picked = set(st.session_state.selected_task_ids or [])
            st.session_state.selected_task_ids = [i for i in ids_now if i not in picked]

    sel = set(st.session_state.selected_task_ids or [])
    for t in tasks:
        tid = t.get("id")
        name = t.get("name", "Task")
        desc = t.get("description", "")
        section = t.get("section", "")
        colA, colB, colC = st.columns([0.12, 0.48, 0.40], gap="small")
        with colA: # Here we select the task
            on = st.checkbox(str(tid), value=(tid in sel), key=f"task_pick_{tid}")
            if on: sel.add(tid)
            else:  sel.discard(tid)
        with colB:
            st.markdown(f"**[{tid}] {name}**  \n<small>{section}</small>", unsafe_allow_html=True)
        with colC:
            st.caption(desc)
    st.session_state.selected_task_ids = list(sel)

    st.markdown("---")
    cL, cR = st.columns([2,2])
    with cL:
        st.caption(f"Selected: **{len(sel)}** / {len(tasks)}")
    with cR:
        disabled = (len(sel) == 0)
        if st.button("▶️ Execute selected on HPC", type="primary", use_container_width=True, disabled=disabled):
            try:
                with st.spinner("Submitting selected tasks to HPC…"):

                    # print("Running here 1")
                    # print(st.session_state)
                    # print(type(st.session_state))
                    # print('The selected ids are:')
                    # print(st.session_state.get('selected_task_ids'))
                    # st.write(st.session_state.keys())
                    # print('start to post')
                    
                    body = {
                        "all_tasks": st.session_state.get('plan_tasks'), 
                        "selected_task_ids": st.session_state.get('selected_task_ids'),
                    }

                    # print(type(st.session_state.get('plan_tasks')))
                    # print('The selected task are')
                    # print(st.session_state.get('plan_tasks')[1])

                    res = post("/chat/execute", body) or {} # Here we sent to the backend
                    # print("Running here 2")

                    # print(res)

                if res.get("ok"):
                    st.success("Submitted. See results below.")
                    # Store plan-executed jobs for HPC monitor (job_id + cluster)
                    try:
                        jobs = []
                        for row in (res.get("results") or []):
                            jid = row.get("job_id")
                            if not jid:
                                continue
                            jobs.append({
                                "task_id": row.get("id"),
                                "name": row.get("step"),
                                "job_id": jid,
                                "cluster": row.get("cluster") or (st.session_state.get("settings") or {}).get("cluster", "hoffman2"),
                                "status": row.get("status") or "Submitted",
                                "job_dir": row.get("job_dir"),
                                "remote_dir": row.get("remote_dir"),
                            })
                        # Merge with existing, de-dup by job_id
                        existing = {j.get("job_id"): j for j in (st.session_state.get("plan_jobs") or [])}
                        for j in jobs:
                            existing[j["job_id"]] = {**existing.get(j["job_id"], {}), **j}
                        st.session_state["plan_jobs"] = list(existing.values())
                        try:
                            sid = st.session_state.get("active_session_id")
                            if sid:
                                jobs = st.session_state.get("plan_jobs") or []
                                _ = post("/chat/session/hpc_jobs/save", {"id": int(sid), "jobs": jobs})
                                print(f'The id is {sid}, jobs: {jobs}')
                                print("running job saved")
                            else:
                                st.warning(f'no save jobs')
                        except Exception as e:
                            st.warning(f"Save job error: {e}")
                    except Exception as _:
                        pass
                else:
                    # print(res.get("detail") or (res.get("errors") and res["errors"][0].get("error")) or "Execute failed.")
                    st.error(res.get("detail") or (res.get("errors") and res["errors"][0].get("error")) or "Execute failed.")
            except Exception as e:
                st.error(f"Execute failed: {e}")

def section_chatdft_with_tabs():
    st.title("🧠 ChatDFT")
    sid = st.session_state.active_session_id
    if sid:
        st.caption(f"Active session: [{sid}] {st.session_state.active_session_name or '(unnamed)'}")
    else:
        st.warning("No active session. 建议在侧边栏创建/选择会话（不影响功能，仅影响记录）")

    tab_chat, tab_workflow, tab_papers, tab_hpc_monitor = st.tabs(["💬 Chat & Plan", "🧪 Workflow", "📑 Papers / RAG", "🖥️ HPC Monitor"])

    # Chat & Plan
    with tab_chat:
        st.subheader("User Inquiry → Intent → Hypothesis → Plan")
        if st.button("↻ Sync workflow view with latest plan"):
            res   = st.session_state.get("plan_raw") or {}
            tasks = st.session_state.get("plan_tasks") or []
            steps = res.get("steps") or res.get("elementary_steps") or []
            inter = res.get("intermediates") or []
            ts    = res.get("ts") or res.get("ts_candidates") or []
            coads = res.get("coads") or res.get("coads_pairs") or []
            if not steps and not inter and not ts and not coads:
                elem2, inter2, ts2, coads2 = _extract_network_from_everywhere(
                    plan_res=res, tasks=tasks, hypothesis=st.session_state.hypothesis or ""
                )
                steps, inter, ts, coads = elem2, inter2, ts2, coads2
            st.session_state.rxn_net        = steps
            st.session_state.intermediates  = inter
            st.session_state.ts_candidates  = ts
            st.session_state.coads_pairs    = coads
            st.success("Workflow synced with backend plan (strict).")

        query = st.text_area("Your question / task:", placeholder="e.g., CO2RR on Cu(111), pH=12, -0.5 V vs RHE …")
        c1, c2, c3 = st.columns(3)
        if c1.button("Generate Intent", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Parsing intent…"): api_intent(query)
            else:
                st.warning("Please enter a question first.")
        if c2.button("Generate Hypothesis", disabled=not st.session_state.intent, use_container_width=True):
            with st.spinner("Generating hypothesis…"): api_hypothesis()
        # Enable plan generation if we have markdown OR structured hypotheses
        can_plan = bool(st.session_state.get("hypotheses_structured") or st.session_state.hypothesis)
        if c3.button("Generate Plan", disabled=not can_plan, use_container_width=True):
            with st.spinner("Building workflow plan…"):
                tasks = api_plan()
                if tasks: st.success(f"Plan generated with {len(tasks)} tasks.")
                else: st.info("No tasks produced.")

        st.markdown("---")
        # Intent + Hypothesis 展示（仅 Structured）
        _intent_summary_card(_force_dict(st.session_state.intent) or {})
        hs_items = st.session_state.get("hypotheses_structured") or []
        _render_hypotheses_structured(hs_items)

    # Workflow
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
            coads_txt = "\n".join([str(x) for x in coads])
            _download_bytes("⬇️ Download coads.txt", coads_txt.encode("utf-8"), "coads_pairs.txt")
            _copy_text_area("Copy co-ads pairs", coads_txt)
    

    import requests

    def fetch_server_jobs_by_session():
        sess_uid = st.session_state.get("hpc_session_uid") or ""
        if not sess_uid:
            return []
        # 假设服务器地址如下，请根据实际情况修改
        url = f"http://localhost:8000/job/list?session_uid={sess_uid}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json().get("jobs", [])
        except Exception as e:
            st.error(f"Error fetching jobs: {e}")
            return []
        
    def refresh_plan_jobs():
        jobs = st.session_state.get("plan_jobs") or []
        default_cluster = (st.session_state.get("settings") or {}).get("cluster", "hoffman2")
        updated = []
        for j in jobs:
            jid = str(j.get("job_id") or "").strip()
            cluster = j.get("cluster") or default_cluster
            if not jid:
                updated.append(j)
                continue
            if j["status"] == 'COMPLETED':
                updated.append(j)
                continue
            try:
                res = post("/agent/job/status", {"cluster": cluster, "job_id": jid}) or {}
                # print('Here we do the job checking fiashrfahsfhaf')
                # print(res)
                status = (res.get("status") or "").upper()
                if not status:
                    status = "COMPLETED"  # 空输出按已完成对待（SGE qstat -j 不存在时）
                j2 = dict(j)
                j2["status"] = status
                updated.append(j2)
            except Exception:
                updated.append(j)
        st.session_state["plan_jobs"] = updated
        sid = st.session_state.get("active_session_id")
        if sid:
            _ = post("/chat/session/hpc_jobs/save", {"id": int(sid), "jobs": updated})


    with tab_hpc_monitor:

        st.subheader("HPC Task Monitoring")
        st.caption("Locally tracked plan jobs")

        plan_jobs = st.session_state.get("plan_jobs") or []
        sid = st.session_state.get("active_session_id")
        
        try:
            snap = post("/chat/session/state", {"id": int(sid)}) or {}
            rows = snap.get("hpc_jobs") or []
            st.session_state["plan_jobs"] = rows  # 不丢字段，原样存回
            plan_jobs = st.session_state["plan_jobs"]

        except Exception as e:
            st.error(f"Error: {e}")
            pass

        # Controls
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("↻ Refresh statuses"):
                with st.spinner("Refreshing job statuses..."):
                    refresh_plan_jobs()
                    # st.experimental_rerun()
                    pass
                st.success("Statuses updated.")
        with c2:
            auto = st.toggle("Auto refresh (15s)", value=False)
            if auto:
                st.experimental_rerun() if st.experimental_get_query_params() else None
                st_autorefresh = getattr(st, "autorefresh", None)
                if st_autorefresh:
                    st_autorefresh(interval=15_000, key="hpc_auto_refresh")

        # Table for locally tracked jobs
        if plan_jobs:
            import pandas as _pd
            df = _pd.DataFrame([{
                "Task": j.get("name"),
                "Job ID": j.get("job_id"),
                "Cluster": j.get("cluster"),
                "Status": j.get("status"),
                "Job Dir": j.get("job_dir"),
                "Remote Dir": j.get("remote_dir"),
            } for j in (st.session_state.get("plan_jobs") or [])])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No locally tracked jobs yet. Submit tasks in the Workflow tab.")

        st.markdown("---")

        # Server-side job list via session_uid (if you use /session/create, /job/list)
        st.caption("Server-side jobs (by session uid)")
        sess_uid = st.session_state.get("hpc_session_uid") or ""
        st.text(f"session_uid: {sess_uid or '(none)'}")
        if sess_uid:
            try:
                jobs = fetch_server_jobs_by_session()  # returns list of dicts
            except Exception as e:
                jobs = []
                st.error(f"Failed to fetch server jobs: {e}")
            if jobs:
                import pandas as _pd
                df2 = _pd.DataFrame([{
                    "Title": r.get("title"),
                    "Status": r.get("status"),
                    "PBS/SLURM ID": r.get("pbs_id"),
                    "Job UID": r.get("job_uid"),
                    "Local Dir": r.get("local_dir"),
                    "Remote Dir": r.get("remote_dir"),
                } for r in jobs])
                st.dataframe(df2, use_container_width=True, hide_index=True)
            else:
                st.write("_No server-tracked jobs yet._")
        else:
            st.caption("Server-tracked jobs unavailable (no session_uid).")

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

            lines = [_fmt_step_compact(s) for s in (steps or [])]
            txt = "\n".join(lines)
            _download_bytes("⬇️ Download steps.txt", txt.encode("utf-8"), "elementary_steps.txt")
            _copy_text_area("Copy steps (plain text)", txt)
        with right:
            _workflow_right_panel()

        _render_tasks_selector()

        # === NEW: 保存到 Execution & 一键提交 ===
        st.markdown("---")
        st.markdown("### Persist plan → Execution")

        csa, csb, csc = st.columns([1, 1, 1])

        with csa:
            if st.button("💾 Save all to Execution", use_container_width=True):
                sid = st.session_state.get("active_session_id")
                if not sid:
                    st.warning("No active session.")
                else:
                    payload = _build_exec_tasks_payload(
                        session_id=sid,
                        tasks=st.session_state.get("plan_tasks") or []
                    )
                    try:
                        _ = post("/exec/tasks/commit", payload)
                        st.success(f"Saved {len(payload['tasks'])} tasks to Execution ✔")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        with csb:
            if st.button("💾 Save selected to Execution", use_container_width=True,
                        disabled=not (st.session_state.get("selected_task_ids"))):
                sid = st.session_state.get("active_session_id")
                if not sid:
                    st.warning("No active session.")
                else:
                    selected = set(st.session_state.get("selected_task_ids") or [])
                    payload = _build_exec_tasks_payload(
                        session_id=sid,
                        tasks=st.session_state.get("plan_tasks") or [],
                        only_ids=selected
                    )
                    try:
                        _ = post("/exec/tasks/commit", payload)
                        st.success(f"Saved {len(payload['tasks'])} selected tasks to Execution ✔")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        with csc:
            if st.button("🚀 Dispatch saved Execution to HPC", use_container_width=True):
                sid = st.session_state.get("active_session_id")
                if not sid:
                    st.warning("No active session.")
                else:
                    try:
                        res = post("/exec/tasks/dispatch", {"session_id": sid, "submit": not st.session_state.settings.get("dry_run", False),
                                                            "fetch": bool(st.session_state.settings.get("sync_back", True))})
                        if res.get("ok"):
                            st.success(f"Dispatched {len(res.get('submitted', []))} task(s).")
                        else:
                            st.error(f"Dispatch partial/failed: {res}")
                    except Exception as e:
                        st.error(f"Dispatch failed: {e}")

        # 简单查看 ExecutionTask 列表（可选）
        try:
            sid = st.session_state.get("active_session_id")
            if sid:
                lst = post("/exec/tasks/list", {"session_id": sid}) or {}
                tasks_show = lst.get("tasks") or []
                if tasks_show:
                    st.caption("Saved Execution tasks")
                    df = pd.DataFrame([{
                        "#": t.get("order_idx"),
                        "Title": t.get("title"),
                        "Type": t.get("task_type"),
                        "Status": t.get("status"),
                    } for t in tasks_show])
                    st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            pass

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

    # New chat
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
                    _save_snapshot(st.session_state.active_session_id)
                    st.session_state.active_session_id = sid
                    st.session_state.active_session_name = new_name.strip()
                    _load_snapshot(sid)
                    _save_snapshot(sid)
                    st.success(f"Created & activated [{sid}] {new_name.strip()}")
                    st.session_state["nav"] = "ChatDFT"
                    st.rerun()
                else:
                    st.error("Create failed.")

    # Open existing
    sessions = get_sessions()
    sid_options = [(f"{s.get('name') or '(unnamed)'}  ·  #{s.get('id')}", s.get("id")) for s in sessions]
    default_idx = 0
    if st.session_state.active_session_id and sid_options:
        ids = [sid for _, sid in sid_options]
        if st.session_state.active_session_id in ids:
            default_idx = ids.index(st.session_state.active_session_id)

    open_sel = st.selectbox("Open chat", sid_options, index=default_idx if sid_options else None,
                            format_func=lambda t: t[0] if isinstance(t, tuple) else str(t),
                            label_visibility="visible", key="open_chat_sel")

    if "___last_open_sid" not in st.session_state:
        st.session_state.___last_open_sid = None
    current_sid = open_sel[1] if open_sel else None
    if current_sid and current_sid != st.session_state.___last_open_sid:
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
        st.session_state["nav"] = "ChatDFT"
        st.rerun()

    st.markdown("---")
    st.markdown("### ✏️ Edit Chat")

    active_sid = st.session_state.active_session_id
    st.caption(f"Active: [{active_sid}] {st.session_state.active_session_name or '(none)'}" if active_sid else "Active: (none)")

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
else:
    section_settings()
