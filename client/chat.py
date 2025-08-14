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

def fetch_session_state_from_backend(session_id: int) -> dict:
    """
    Try to fetch a structured snapshot for a session.
    1) preferred: /chat/session/state  -> returns {intent, hypothesis, plan_raw, plan_tasks, rxn_net, intermediates, ts_candidates, coads_pairs, workflow_results}
    2) fallback:  /chat/session/messages -> reconstruct from ChatMessage.msg_type
    """
    # try structured state first
    try:
        res = post("/chat/session/state", {"id": session_id}) or {}
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

def _force_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x) or {}
        except Exception:
            return {}
    return {}

# =========================
# Pretty helpers & parsing
# =========================
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

def _get_mechanisms_from_state() -> list[str]:
    """从 plan_raw.workflow 或 rxn_network 取机制标签；回退 intent.tags。"""
    pr = st.session_state.get("plan_raw") or {}
    wf = pr.get("workflow") or {}
    mechs = wf.get("mechanisms") or []
    if not mechs:
        mechs = (st.session_state.get("intent") or {}).get("tags") or []
    # 去重清洗
    out, seen = [], set()
    for m in mechs:
        s = str(m).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def _extract_conditions_from_intent(I: dict) -> dict:
    """归一化条件：pH / potential / temperature / electrolyte / solvent 等。"""
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

def _kv_badge(label: str, value) -> None:
    if value is None or value == "" or value == []:
        return
    _badge(f"{label}: {value}")

def _nonempty(x) -> bool:
    if x is None: return False
    if isinstance(x, str): return x.strip() != ""
    if isinstance(x, (list,tuple,dict,set)): return len(x) > 0
    return True
def _intent_table(intent: dict) -> str:
    """把 intent dict 转成 Markdown 表格"""
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
    """丰富版 Intent 摘要：系统/条件徽章/机制/置信度/RN 统计 + 详细表格。"""
    I = _force_dict(I or {})
    st.markdown("#### Intent")

    sys  = I.get("system") or {}
    cat  = sys.get("catalyst") or sys.get("material") or I.get("substrate") or "-"
    facet= sys.get("facet") or I.get("facet") or "-"
    fam  = (I.get("area") or I.get("problem_type") or I.get("stage") or "catalysis").upper()
    task = I.get("task") or I.get("normalized_query") or "(no task text)"
    conf = (st.session_state.get("intent_raw") or {}).get("confidence")

    rn   = I.get("reaction_network") or {}
    n_steps = len(rn.get("steps") or rn.get("elementary_steps") or [])
    n_inters= len(rn.get("intermediates") or [])
    n_ts    = len(rn.get("ts") or rn.get("ts_candidates") or [])
    n_coads = len(rn.get("coads_pairs") or rn.get("coads") or [])

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(f"**{fam}** · **{cat}({facet})**")
        st.caption(task)
        cond = _extract_conditions_from_intent(I)
        with st.container():
            _kv_badge("pH", cond.get("pH"))
            _kv_badge("U (vs RHE)", cond.get("U"))
            _kv_badge("T (K)", cond.get("T"))
            _kv_badge("electrolyte", cond.get("electrolyte"))
            _kv_badge("solvent", cond.get("solvent"))
            _kv_badge("illum.", cond.get("illumination"))
            _kv_badge("P", cond.get("pressure"))
        dels = (I.get("deliverables") or {})
        tps  = dels.get("target_products") or []
        if _nonempty(tps):
            st.caption("Targets")
            _badges_grid([str(x) for x in tps], cols=6)
    with c2:
        mechs = _get_mechanisms_from_state()
        st.caption("Mechanisms (matched)")
        if mechs:
            _badges_grid(mechs, cols=3)
        else:
            st.write("_None_")
        st.caption("RN summary")
        st.markdown(
            f"- steps: **{n_steps}**  \n"
            f"- intermediates: **{n_inters}**  \n"
            f"- TS candidates: **{n_ts}**  \n"
            f"- co-ads: **{n_coads}**"
        )
        if conf is not None:
            st.caption(f"Confidence: **{float(conf):.2f}**")

    _intent_table(I)

def _parse_hypothesis_md(md_text: str) -> dict:
    """
    解析 hypothesis_agent 的 Markdown 模板，鲁棒匹配：
    支持: "Conditions:" / "**Conditions:**" / "**Conditions**:" 等。
    子弹点自动抓取 0~N 条。
    """
    if not isinstance(md_text, str) or not md_text.strip():
        return {"conditions": "", "hypothesis": "", "why": [], "next": [], "exp": []}

    txt = md_text.strip()

    # 统一换行
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")

    # 段落切片：找到各段标题位置
    def _find_block(label: str) -> Tuple[int, int]:
        # 允许 **Label:** / Label: / **Label**:
        pat = rf"(?mi)^\s*(\*\*)?{re.escape(label)}(\*\*)?\s*:\s*"
        m = re.search(pat, txt)
        if not m:
            return (-1, -1)
        start = m.end()
        # 下一个标题（任何四个标题之一）开始处就是本段结束
        ANY = r"(Conditions|Hypothesis|Why it may be true|What to compute next|Optional experimental validation)"
        pat_next = rf"(?mi)^\s*(\*\*)?{ANY}(\*\*)?\s*:\s*"
        m2 = re.search(pat_next, txt[start:])
        if m2:
            end = start + m2.start()
        else:
            end = len(txt)
        return (start, end)

    def _block_text(label: str) -> str:
        s, e = _find_block(label)
        if s < 0:
            return ""
        return txt[s:e].strip()

    def _first_line(s: str) -> str:
        if not s:
            return ""
        # 取本段落第一行的纯文本（去掉前导 -/* 与粗体）
        line = s.split("\n", 1)[0]
        line = re.sub(r"^\s*[-*]\s*", "", line).strip()
        line = re.sub(r"^\*\*(.*?)\*\*$", r"\1", line)
        return line

    def _bullets(s: str) -> list:
        if not s:
            return []
        items = []
        for line in s.split("\n"):
            m = re.match(r"^\s*[-*]\s+(.*)$", line)
            if m:
                items.append(m.group(1).strip())
        return items

    blk_cond = _block_text("Conditions")
    blk_hypo = _block_text("Hypothesis")
    blk_why  = _block_text("Why it may be true")
    blk_next = _block_text("What to compute next")
    blk_exp  = _block_text("Optional experimental validation")

    return {
        "conditions": _first_line(blk_cond),
        "hypothesis": _first_line(blk_hypo),
        "why":  _bullets(blk_why),
        "next": _bullets(blk_next),
        "exp":  _bullets(blk_exp),
    }

def _render_tasks_selector():
    """在 Workflow 页签渲染可勾选的计划任务 + 执行按钮。"""
    tasks = st.session_state.get("plan_tasks") or []

    # 如果任务集合变化，重置选中
    ids_now = [t.get("id") for t in tasks if isinstance(t, dict)]
    ids_prev = st.session_state.get("selected_task_ids") or []
    if sorted(ids_now) != sorted([i for i in ids_prev if i in ids_now]):
        st.session_state.selected_task_ids = []

    st.markdown("### Planned Tasks")
    if not tasks:
        st.info("No tasks yet. Click **Generate Plan** in the Chat & Plan tab.")
        return

    c1, c2, c3, c4 = st.columns([1,1,1,3])
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
        with colA:
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
        if st.button("▶️ Execute selected on HPC", type="primary", use_container_width=True, disabled=(len(sel)==0)):
            with st.spinner("Submitting selected tasks to HPC…"):
                res = api_execute(st.session_state.selected_task_ids)
            st.success("Submitted. See results below.") if res.get("ok") else st.error(res.get("detail") or "Execute failed.")

def _add_meta_from_hypothesis(next_bullets: list[str]):
    """把 Next bullets 插入为首条 meta.clarify 元任务（仅前端）。"""
    if not next_bullets:
        st.warning("No items found in 'What to compute next'.")
        return
    tasks = st.session_state.get("plan_tasks") or []
    # 生成一个元任务（放在最前）
    meta_task = {
        "id": 0,  # 放最前；仅前端展示，不提交 HPC
        "section": "Meta",
        "name": "Plan notes — from Hypothesis",
        "agent": "meta.clarify",
        "description": "Imported from Hypothesis · What to compute next",
        "params": {
            "form": [],
            "payload": {
                "notes": next_bullets,
                "source": "hypothesis",
                "parallel_group": 0
            }
        },
        "meta": {"parallel_group": 0, "action_endpoint": None}
    }
    # 如果已有相同源的 meta，替换；否则插入到首位
    replaced = False
    for i, t in enumerate(tasks):
        if (t.get("agent") == "meta.clarify") and "notes" in ((t.get("params") or {}).get("payload") or {}):
            tasks[i] = meta_task
            replaced = True
            break
    if not replaced:
        tasks = [meta_task] + tasks
    st.session_state.plan_tasks = tasks
    # 同步到 plan_raw（可选）
    plan_raw = st.session_state.get("plan_raw") or {}
    plan_raw.setdefault("tasks", st.session_state.plan_tasks)
    st.session_state.plan_raw = plan_raw
    st.success(f"Added {len(next_bullets)} note(s) to plan as meta.clarify.")

def _render_hypothesis_block(md_text: str):
    st.markdown("#### Hypothesis")
    if not md_text:
        st.info("Empty hypothesis.")
        return
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
            for it in items: st.markdown(f"- {it}")
        else:
            st.write("_N/A_")
    with c2:
        st.markdown("**What to compute next**")
        items_next = parsed.get("next") or []
        if items_next:
            for it in items_next: st.markdown(f"- {it}")
        else:
            st.write("_N/A_")
        # 👉 新按钮：把 next 导入为计划注释/元任务
        if items_next and st.button("➕ Add 'What to compute next' to plan (meta.clarify)", use_container_width=True):
            _add_meta_from_hypothesis(items_next)
    with c3:
        st.markdown("**Optional experimental validation**")
        items = parsed.get("exp") or []
        if items:
            for it in items: st.markdown(f"- {it}")
        else:
            st.write("_N/A_")

    with st.expander("Raw hypothesis (Markdown)", expanded=False):
        st.markdown(md_text)

# =========================
# Formatting helpers for RN/steps
# =========================
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

# =========================
# APIs to backend
# =========================
def api_intent(query: str) -> dict:
    # 1) 确保有 session_id；没有就创建一个，避免 400
    sid = st.session_state.get("active_session_id")
    if not sid:
        new_name = (st.session_state.get("active_session_name") or "").strip() or "Untitled"
        sid = create_session(name=new_name)
        if not sid:
            st.error("Failed to create session for intent.")
            return {}
        st.session_state.active_session_id = sid
        st.session_state.active_session_name = new_name

    # 2) 组织 payload
    guided = {"stage": "catalysis", "area": "electro", "task": query.strip()[:140]}
    payload = {"session_id": sid, "text": (query or "").strip(), "guided": guided}

    # 3) 请求
    res = post("/chat/intent", payload) or {}

    # 4) 写入前端状态
    st.session_state.intent_raw = res
    intent = res.get("intent") or res.get("fields") or {}
    st.session_state.intent = _force_dict(intent)

    # 5) 反馈
    if not res.get("ok"):
        err = res.get("error") or "Intent API error"
        st.error(f"{err}. Received: {res.get('received')}")
    else:
        st.success(f"Intent generated. confidence={res.get('confidence',0):.2f}")
    return intent

def api_hypothesis() -> str:
    """
    更鲁棒的 hypothesis 拉取：
    - 兼容 {"result_md": "..."} / {"md": "..."} / {"hypothesis":{"md":"..."}}
    - 如果没有 md，兜底把整个对象 json 展示，避免空
    - 同时回填 steps/inter/ts/coads（按优先级取）
    """
    payload = {
        "intent": st.session_state.intent,
        "knowledge": {},
        "history": []
    }
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id

    res = post("/chat/hypothesis", payload) or {}

    # 1) 多套路拿 md 文本
    md_text = ""
    if isinstance(res, str):
        md_text = res
    elif isinstance(res, dict):
        cand = (
            res.get("result_md")
            or res.get("md")
            or (res.get("hypothesis") or {}).get("result_md")
            or (res.get("hypothesis") or {}).get("md")
            or res.get("hypothesis")  # 有时就是个纯字符串
        )
        if isinstance(cand, str):
            md_text = cand
        elif isinstance(cand, dict) and "md" in cand and isinstance(cand["md"], str):
            md_text = cand["md"]
        elif isinstance(cand, dict) and "result_md" in cand and isinstance(cand["result_md"], str):
            md_text = cand["result_md"]
        else:
            # 兜底：把 res 打印为 JSON，避免 UI 是空
            try:
                md_text = json.dumps(res, ensure_ascii=False, indent=2)
            except Exception:
                md_text = str(res)
    else:
        md_text = str(res or "")

    st.session_state.hypothesis = md_text or ""

    # 2) 同时抽结构（后端可能已经给了）
    steps = (
        res.get("steps")
        or res.get("elementary_steps")
        or (res.get("hypothesis") or {}).get("steps")
        or []
    )
    inter = (
        res.get("intermediates")
        or (res.get("hypothesis") or {}).get("intermediates")
        or []
    )
    ts = (
        res.get("ts")
        or res.get("ts_candidates")
        or (res.get("hypothesis") or {}).get("ts")
        or (res.get("hypothesis") or {}).get("ts_candidates")
        or []
    )
    coads = (
        res.get("coads")
        or res.get("coads_pairs")
        or (res.get("hypothesis") or {}).get("coads")
        or (res.get("hypothesis") or {}).get("coads_pairs")
        or []
    )

    # 3) 如果还缺，就用已有 extractor 兜底
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
    return md_text

def api_plan() -> list:
    """Call /chat/plan and strictly use backend-provided fields.
       This keeps Workflow tab 1:1 consistent with the plan (e.g., 45 steps)."""
    payload = {
        "intent": st.session_state.intent,
        "hypothesis": st.session_state.hypothesis,
        "history": []
    }
    if st.session_state.active_session_id:
        payload["session_id"] = st.session_state.active_session_id

    res = post("/chat/plan", payload) or {}

    # save the raw plan
    st.session_state["plan_raw"] = res

    # tasks for flat view
    tasks = res.get("tasks") or []
    st.session_state.plan_tasks = tasks

    # STRICT SYNC (no post-merging / re-extraction)
    steps = res.get("steps") or res.get("elementary_steps") or []
    inter = res.get("intermediates") or []
    ts    = res.get("ts") or res.get("ts_candidates") or []
    coads = res.get("coads") or res.get("coads_pairs") or []

    # fallback only if backend truly didn't give anything
    if not steps and not inter and not ts and not coads:
        # last resort: keep old behavior (rarely used)
        elem2, inter2, ts2, coads2 = _extract_network_from_everywhere(
            plan_res=res, tasks=tasks, hypothesis=st.session_state.hypothesis or ""
        )
        steps, inter, ts, coads = elem2, inter2, ts2, coads2

    st.session_state.rxn_net        = steps or []
    st.session_state.intermediates  = inter or []
    st.session_state.ts_candidates  = ts or []
    st.session_state.coads_pairs    = coads or []

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

# ---- Session management APIs ----
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

# =========================
# Small UI helpers
# =========================
def _json_safe(obj):
    """将任何对象转换成可被 st.json 接收的结构。"""
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
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set,)):
        return [_json_safe(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"

def _render_records_block(wf: dict):
    st.markdown(f"### Run — Workdir: `{wf.get('workdir','')}`")
    rows = wf.get("results") or []
    if rows:
        for r in rows:
            step = str(r.get("step", "(unnamed)"))
            status = str(r.get("status", ""))
            st.markdown(f"- **{step}** → {status}")
    if wf.get("summary") is not None:
        with st.expander("Post-analysis summary", expanded=False):
            st.json(_json_safe(wf["summary"]))

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

# =========================
# Extract RN from sources
# =========================
from typing import Any, Dict, List, Tuple
# --- replace the whole function in client/home.py ---
from typing import Any, Dict, List, Tuple
import re, json

def _extract_network_from_everywhere(
    plan_res: Any,
    tasks: List[dict],
    hypothesis: Any
) -> Tuple[List[Any], List[str], List[str], List[str]]:
    """
    Make Workflow view 1:1 with the generated plan list.
    - Each task becomes one row in 'elem' (Elementary steps).
    - For NEB/TS tasks: parse reaction A -> B into Reaction/Reactants/Products.
    - For non-reaction tasks (DOS/PDOS, Assemble ΔG...), keep Name-only rows.
    Also aggregates intermediates, ts, coads from plan/tasks/hypothesis.
    """

    # ---------- small utils ----------
    
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
        out = []
        for t in tokens:
            if _looks_like_species(t): out.append(_to_str(t))
        return out

    def _uniq(seq: List[Any]) -> List[Any]:
        seen, out = set(), []
        for x in seq:
            k = json.dumps(x, ensure_ascii=False, sort_keys=True) if isinstance(x, dict) else _to_str(x)
            if k not in seen:
                seen.add(k); out.append(x)
        return out

    # ---------- collect from plan (as hints) ----------
    plan = plan_res if isinstance(plan_res, dict) else {}
    plan_elem  = (plan.get("elementary_steps")
                  or plan.get("reaction_network")
                  or plan.get("steps")
                  or [])
    plan_inter = plan.get("intermediates") or []
    plan_ts    = (plan.get("ts_candidates") or plan.get("ts_edges") or plan.get("ts") or [])
    plan_coads = plan.get("coads_pairs") or plan.get("coads") or []

    # ---------- build elem strictly from tasks (1:1 rows) ----------
    elem_rows: List[Dict[str, Any]] = []
    ts_list: List[str] = []
    inter_list: List[str] = []
    coads_list: List[str] = []

    ARROWS = ("->","→","⇒","⟶")

    def _parse_reaction_from_task(t: dict) -> str:
        """Prefer explicit payload/form 'step'; fallback to parsing from 'name'."""
        # 1) payload.step
        payload = (t.get("params") or {}).get("payload") or {}
        step_txt = _to_str(payload.get("step"))
        if any(a in step_txt for a in ARROWS):
            return _normalize_arrow(step_txt)

        # 2) form field named 'step'
        for f in (t.get("params") or {}).get("form") or []:
            if str(f.get("key")).lower() == "step":
                v = _to_str(f.get("value"))
                if any(a in v for a in ARROWS):
                    return _normalize_arrow(v)

        # 3) parse from task name like: "NEB — A -> B  · CI-NEB ..."
        name = _to_str(t.get("name"))
        s = name
        if "—" in s: s = s.split("—", 1)[1]
        elif "–" in s: s = s.split("–", 1)[1]
        elif " - " in s: s = s.split(" - ", 1)[1]
        # cut common tails
        for tail in ("·", "•", " CI-NEB", " NEB", "(CI-NEB)"):
            if tail in s:
                s = s.split(tail, 1)[0]
        s = s.strip()
        if any(a in s for a in ARROWS):
            return _normalize_arrow(s)
        return ""

    # map every task to a row; keep order & count identical to tasks
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

            # mine species from reaction
            toks = re.split(r"[+\s→]+", rxn.replace("(", " ").replace(")", " "))
            inter_list += _only_species(toks)

        # also pick species from task name (e.g., H*, CO2(g))
        name_tokens = re.split(r"[,\s/+\-·•]+", _to_str(t.get("name")))
        inter_list += _only_species(name_tokens)

        # collect any explicit payload lists
        payload = (t.get("params") or {}).get("payload") or {}
        inter_list += payload.get("intermediates") or []
        ts_list    += payload.get("ts_candidates") or []
        coads_list += payload.get("coads_pairs") or []

        # final row: convert to elem-row shape the table expects
        elem_rows.append({
            "name": row.get("Name"),
            "reaction": row.get("reaction",""),
            "reactants": row.get("reactants",""),
            "products": row.get("products",""),
        })

    # ---------- merge with plan/hypothesis extras (non-dominant) ----------
    # extras go after task-rows; we don't dedup elem_rows to keep count identical to tasks
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

    # final elem = tasks-first + plan extras (do not dedup tasks part)
    elem = elem_rows + extras

    # ---------- intermediates / ts / coads tidy ----------
    # add plan-level and hypothesis-derived species
    inter_list += plan_inter
    ts_list    += plan_ts
    coads_list += plan_coads

    # from hypothesis free text (very light)
    hypo_text = _to_str(hypothesis)
    if hypo_text:
        rough = re.findall(r"\b([A-Za-z0-9()*]+)\b", hypo_text)
        inter_list += _only_species(rough)

    # uniq but keep order
    intermediates = _uniq(inter_list)
    ts_clean = _uniq([_normalize_arrow(x) for x in ts_list if _to_str(x)])
    coads_clean = _uniq([_to_str(c) for c in coads_list if _to_str(c)])

    # sort intermediates: adsorbates first, then gas/aq, then others
    ads = [s for s in intermediates if s.endswith("*")]
    gas = [s for s in intermediates if s.endswith("(g)") or s.endswith("(aq)")]
    rest = [s for s in intermediates if s not in ads and s not in gas]
    intermediates_sorted = ads + gas + rest

    # NOTE: do not collapse elem (to preserve exact count == tasks count)
    return elem, intermediates_sorted, ts_clean, coads_clean
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
    sid = st.session_state.active_session_id
    if sid:
        st.caption(f"Active session: [{sid}] {st.session_state.active_session_name or '(unnamed)'}")
    else:
        st.warning("No active session. 建议在 **Projects** 选择/创建会话（不影响使用，仅影响记录与回溯）。")

    tab_chat, tab_workflow, tab_papers= st.tabs(["💬 Chat & Plan", "🧪 Workflow", "📑 Papers / RAG"])

    # --- Chat & Plan ---
    with tab_chat:
        st.subheader("User Inquiry → Intent → Hypothesis → Plan")
        # 用当前 plan_raw 的字段，强一致同步到 Workflow 视图
        if st.button("↻ Sync workflow view with latest plan"):
            res   = st.session_state.get("plan_raw") or {}
            tasks = st.session_state.get("plan_tasks") or []

            # 直接取后端产物，保持 1:1 一致
            steps = res.get("steps") or res.get("elementary_steps") or []
            inter = res.get("intermediates") or []
            ts    = res.get("ts") or res.get("ts_candidates") or []
            coads = res.get("coads") or res.get("coads_pairs") or []

            # 如果后端真的没给（极少数情况），再兜底抽取
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
        if c3.button("Generate Plan", disabled=(not st.session_state.hypothesis), use_container_width=True):
            with st.spinner("Building workflow plan…"):
                tasks = api_plan()
                if tasks: st.success(f"Plan generated with {len(tasks)} tasks.")
                else: st.info("No tasks produced.")

        st.markdown("---")
        _intent_summary_card(_force_dict(st.session_state.intent) or {})
        _render_hypothesis_block(st.session_state.hypothesis or "")

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

            # 下载/复制
            lines = [_fmt_step_compact(s) for s in (steps or [])]
            txt = "\n".join(lines)
            _download_bytes("⬇️ Download steps.txt", txt.encode("utf-8"), "elementary_steps.txt")
            _copy_text_area("Copy steps (plain text)", txt)
        with right:
            _workflow_right_panel()
        _render_tasks_selector()

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
    _save_snapshot(st.session_state.active_session_id)

    sessions = get_sessions()
    if not sessions:
        st.info("No sessions yet. Create one below.")
    else:
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
    with colL:
        sid_options = [(f"[{s.get('id')}] {s.get('name')}", s.get("id")) for s in sessions] if sessions else []
        chosen = st.selectbox("Select a session to activate", sid_options, index=0 if sid_options else None,
                              format_func=lambda t: t[0] if isinstance(t, tuple) else str(t))
        if sid_options and st.button("✅ Set Active"):
            _, sid = chosen
            _save_snapshot(st.session_state.active_session_id)
            st.session_state.active_session_id = sid
            st.session_state.active_session_name = next((s.get("name") for s in sessions if s.get("id")==sid), "")
            loaded = _load_snapshot(sid)
            if not loaded:
                snap = fetch_session_state_from_backend(sid)
                for k, v in snap.items():
                    st.session_state[k] = v
                _save_snapshot(sid)
            st.success(f"Activated session [{sid}] and loaded state.")
            st.session_state["nav"] = "ChatDFT"
            st.rerun()

    with colR:
        st.markdown("**Quick actions**")
        if sessions:
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

    # --- Open existing ---
    sessions = get_sessions()
    sid_options = [(f"{s.get('name') or '(unnamed)'}  ·  #{s.get('id')}", s.get("id")) for s in sessions]
    default_idx = 0
    if st.session_state.active_session_id and sid_options:
        ids = [sid for _, sid in sid_options]
        if st.session_state.active_session_id in ids:
            default_idx = ids.index(st.session_state.active_session_id)

    open_label = "Open chat"
    open_sel = st.selectbox(open_label, sid_options, index=default_idx if sid_options else None,
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
elif st.session_state["nav"] == "Projects":
    section_projects()
else:
    section_settings()