# -*- coding: utf-8 -*-
"""
ChatDFT 前端（Streamlit）
- 工作流计划 & 表单值 per-session 缓存，刷新/编辑不会丢失
- Expander 打开/关闭状态持久化在 session_state
- “🔄 Regenerate Plan” 按钮：显式清除 plan 缓存并重新请求
- 全功能 sidebar: 会话创建、选择、编辑、删除
- 支持 ChatDFT 主 chat、各 tools 页面、Overview 介绍
"""

from __future__ import annotations
import json, hashlib, time
import streamlit as st
from utils.api import post

# ------------------------------------------------------------------
# 🔧  helpers --------------------------------------------------------
# ------------------------------------------------------------------

def _sig(obj) -> str:
    """Stable md5 signature of a (JSON-serialisable) object."""
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def _get_val(key: str, fallback):
    """Return a value from session_state if it exists (used to persist widgets)."""
    return st.session_state.get(key, fallback)

# ------------------------------------------------------------------
# ♻️  Per-session plan cache ----------------------------------------
# ------------------------------------------------------------------

_PLAN_CACHE_KEY = "cached_plan"

def _get_cached_plan(session_id: int, intent: dict, force: bool = False):
    cache: dict = st.session_state.setdefault(_PLAN_CACHE_KEY, {})
    intent_sig = _sig(intent)
    ent = cache.get(session_id)
    if ent and not force and ent["sig"] == intent_sig:
        return ent["tasks"], ent.get("suggestions", {})
    res = post("/chat/plan", {"session_id": session_id, "intent": intent})
    tasks        = res.get("tasks", [])
    suggestions  = res.get("suggestions", {})
    cache[session_id] = {"sig": intent_sig, "tasks": tasks, "suggestions": suggestions}
    return tasks, suggestions

# ------------------------------------------------------------------
# 🖼️  UI helpers ----------------------------------------------------
# ------------------------------------------------------------------

def _exp(title: str, state_key: str):
    """Persistent expander – remembers open/close across reruns."""
    expanded = _get_val(state_key, True)
    expander = st.expander(title, expanded=expanded)
    st.session_state[state_key] = expander._is_open  # type: ignore[attr-defined]
    return expander

# --------- 自动渲染 dict 字段的表单工具 ---------
def render_dict_form(obj_dict, exclude=("id", "created_at", "updated_at"), prefix=""):
    new_dict = obj_dict.copy()
    for k, v in obj_dict.items():
        if k in exclude:
            st.text(f"{prefix}{k}: {v}")
        elif isinstance(v, bool):
            new_dict[k] = st.checkbox(f"{prefix}{k}", value=v)
        elif isinstance(v, int) or isinstance(v, float):
            new_dict[k] = st.number_input(f"{prefix}{k}", value=v)
        elif isinstance(v, str) or v is None:
            new_dict[k] = st.text_input(f"{prefix}{k}", value=v or "")
        elif isinstance(v, dict) or isinstance(v, list):
            new_dict[k] = st.text_area(f"{prefix}{k}", value=str(v) if v is not None else "")
        else:
            new_dict[k] = st.text_input(f"{prefix}{k}", value=str(v))
    return new_dict

# ------------------------------------------------------------------
# 🌱  Initial session_state slots -----------------------------------
# ------------------------------------------------------------------

_DEFAULTS = {
    "session_id":       None,
    "current_chat":     None,
    "workflow_steps":   [],
    "workflow_results": {},
    "workflow_last_query": "",
    "last_intent":      {},
    "guided_send":      None,
    "force_send":       False,
    _PLAN_CACHE_KEY:     {},
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------
# 🔗  Thin API wrappers ---------------------------------------------
# ------------------------------------------------------------------

def get_sessions():             return post("/chat/session/list",   {}).get("sessions", [])
def create_session(**kw):       return post("/chat/session/create", kw).get("session_id")
def update_session(**kwargs):   return post("/chat/session/update", kwargs).get("ok", False)
def delete_session(sid):        return post("/chat/session/delete", {"id": sid}).get("ok")
def get_history(sid):           return post("/chat/history", {"session_id": sid, "limit": 1000}).get("messages", [])
def append_msg(sid, role, txt): post("/chat/message/create", {"session_id": sid, "role": role, "content": txt})
def get_messages(session_id):   return post("/chat/message/list", {"session_id": session_id}).get("messages", [])
def create_message(**kwargs):   return post("/chat/message/create", kwargs).get("message_id", None)
def update_message(**kwargs):   return post("/chat/message/update", kwargs).get("ok", False)
def delete_message(message_id): return post("/chat/message/delete", {"id": message_id}).get("ok", False)
def append_message(session_id, role, content):
    post("/chat/message/create", {"session_id": session_id, "role": role, "content": content})

# ------------------------------------------------------------------
# 📋  Task & workflow rendering -------------------------------------
# ------------------------------------------------------------------

def _render_task(session_id: int, t: dict):
    st.markdown(
        f"""<div style='background:#f7fbff;border-radius:8px;padding:12px 14px;margin:10px 0;'>
        <b>{t['id']}. {t.get('name','Task')}</b> <span style='color:#888'>(agent {t.get('agent','-')})</span><br>
        <span style='color:#222'>{t.get('description','')}</span></div>""",
        unsafe_allow_html=True)

    # ---- form controls -------------------------------------------
    form_vals = {}
    for f in t.get("params", {}).get("form", []):
        ctrl_key = f"{t['id']}:{f.get('key','k')}"
        ftype    = f.get("type", "text")
        label    = f.get("label", ctrl_key)
        help_t   = f.get("help", "")
        default  = _get_val(ctrl_key, f.get("value", ""))

        if ftype == "number":
            form_vals[f["key"]] = st.number_input(label, float(default or 0), step=float(f.get("step",1)),
                                                   key=ctrl_key, help=help_t)
        elif ftype == "select":
            opts = f.get("options", [])
            form_vals[f["key"]] = st.selectbox(label, opts, index=opts.index(default) if default in opts else 0,
                                                key=ctrl_key, help=help_t)
        elif ftype == "checkbox":
            form_vals[f["key"]] = st.checkbox(label, bool(default), key=ctrl_key, help=help_t)
        elif ftype == "textarea":
            form_vals[f["key"]] = st.text_area(label, str(default), key=ctrl_key, help=help_t)
        else:  # text
            form_vals[f["key"]] = st.text_input(label, str(default), key=ctrl_key, help=help_t)

    # ---- Run button ----------------------------------------------
    if st.button("Run", key=f"run-{t['id']}"):
        payload = {"session_id": session_id, **t.get("params", {}).get("payload", {}), **form_vals}
        with st.spinner("Running agent …"):
            res = post(f"/agent/{t.get('agent')}", payload)
        st.success(res.get("result", "Completed.")) if res.get("ok") else st.error(res.get("detail", "Failed."))

def _render_workflow(session_id: int, intent: dict, *, force: bool = False):
    tasks, suggestions = _get_cached_plan(session_id, intent, force)
    if not tasks:
        st.info("Workflow not ready. Try again later.")
        return

    # 🔄 regenerate button
    if st.button("🔄 Regenerate Plan", key="regen-plan"):
        _get_cached_plan(session_id, intent, force=True)  # refresh cache
        st.rerun()

    # -------- sections -------------------------------------------
    sections = {
        "ideas":         ("🧭 Ideas & Literature",     "Ideas & Literature"),
        "calc_flow":     ("🧪 Calculation Flow",        "Calculation Flow"),
        "post":          ("📊 Post-analysis",           "Post-analysis"),
        "report":        ("📝 Report",                  "Report"),
    }
    for skey, (title, sec_name) in sections.items():
        with _exp(title, f"exp_{skey}"):
            for t in filter(lambda x: x["section"] == sec_name, tasks):
                _render_task(session_id, t)
    # suggestions --------------------------------------------------
    if suggestions:
        st.markdown("---")
        for k, vals in suggestions.items():
            st.markdown(f"**Suggestions – {k}:** " + " ".join(f"`{v}`" for v in vals))

# ------------------------------------------------------------------
# 🗣️  Chat interaction ---------------------------------------------
# ------------------------------------------------------------------

def _intent_and_hypothesis(query: str):
    intent = post("/chat/intent", {"query": query})
    hypo   = post("/chat/hypothesis", {"fields": intent.get("fields", {})})
    return intent, hypo

def _pill(intent: dict) -> str:
    return (
        f"**🎯 Intent:** {intent.get('intent','-')}  "
        f"**Stage:** {intent.get('stage','-')}  "
        f"**Domain:** {intent.get('domain','-')}  "
        f"**Confidence:** {intent.get('confidence',0):.2f}")

def _handle_query(session_id: int, query: str):
    if not query.strip():
        return
    append_msg(session_id, "user", query)
    intent, hypo = _intent_and_hypothesis(query)
    st.session_state["last_intent"] = intent
    with st.chat_message("assistant"): st.markdown(_pill(intent))
    with st.chat_message("assistant"): st.markdown(hypo.get("result_md", "**Hypothesis:** N/A"))
    _render_workflow(session_id, intent)

# ------------------------------------------------------------------
# 🖼️  Chat session view --------------------------------------------
# ------------------------------------------------------------------

def _chat_session(session_id: int):
    st.markdown(f"<h2 style='margin-bottom:4px'>🔬 {_get_val('current_chat','')}</h2>", unsafe_allow_html=True)
    # ---- history -------------------------------------------------
    for m in get_history(session_id):
        with st.chat_message("assistant" if m["role"].startswith("assistant") else "user"):
            st.markdown(m["content"], unsafe_allow_html=True)
    # ---- input ---------------------------------------------------
    if q := st.chat_input("Type your DFT question …"):
        with st.chat_message("user"): st.markdown(q)
        _handle_query(session_id, q)

# --------- Overview Pages -----------
def render_introduction():
    st.header("Introduction")
    st.markdown("""
Welcome to **ChatDFT**, your AI copilot for density functional theory (DFT) modeling.
Key features:
- Intent recognition
- Hypothesis generation
- Workflow planning
- Scientific knowledge retrieval
- Full chat session history persistence
""")
    st.image("utils/figures/ChatDFT_pipeline.png", caption="ChatDFT Pipeline Overview", use_container_width=True)

def render_paper():
    st.header("Paper")
    st.info("This section will host the core paper details and annotations—coming soon.")

# --------- Tool Pages ----------
def render_materials_obtain():
    st.header("Materials Obtain 🔍")
    st.write("（此处可实现你的材料获取模块）")

def render_poscar_builder():
    st.header("POSCAR Builder 💧")
    st.write("（此处可实现POSCAR构建模块）")

def render_incar_copilot():
    st.header("INCAR Copilot 🧪")
    st.write("（此处可实现INCAR参数模块）")

def render_job_submission():
    st.header("Job Submission 🚀")
    st.write("（此处可实现作业提交模块）")

def render_error_handling():
    st.header("Error Handling 🐞")
    st.write("（此处可实现错误分析模块）")

def render_post_analysis():
    st.header("Post Analysis 📊")
    st.write("（此处可实现后处理分析模块）")

# ------------------------------------------------------------------
# 🧭  Sidebar (sessions) -------------------------------------------
# ------------------------------------------------------------------

st.set_page_config(page_title="ChatDFT", layout="wide")
st.title("🔬 ChatDFT")

SECTION_NAMES = ["Overview", "ChatDFT", "Tools"]
section = st.sidebar.selectbox("Section", SECTION_NAMES)

if section == "Overview":
    page = st.sidebar.selectbox("Page", ["Introduction", "Paper"])
elif section == "ChatDFT":
    st.sidebar.header("Chat Sessions")
    sessions = get_sessions()
    chat_names = [s["name"] for s in sessions]
    session_ids = [s["id"] for s in sessions]
    # --- 新建 Chat ---
    with st.sidebar.expander("➕ New Chat", True):
        form_new = {k:"" for k in ("name", "user_id", "project", "tags", "description", "status")}
        form_new["pinned"] = False
        form_new = render_dict_form(form_new, exclude=("id", "created_at", "updated_at"))
        if st.button("🆕 Create Chat", key="create_chat_btn"):
            if form_new["name"].strip():
                new_session_id = create_session(**form_new)
                st.session_state.session_id = new_session_id
                st.session_state.current_chat = form_new["name"]
                st.rerun()
            else:
                st.sidebar.warning("Please enter a non-empty name.")
    # --- 选择并编辑 Chat ---
    if chat_names:
        choice = st.sidebar.selectbox("Open chat", chat_names, key="open_chat")
        if choice:
            idx = chat_names.index(choice)
            st.session_state.session_id = session_ids[idx]
            st.session_state.current_chat = choice
            # 编辑当前 session
            session = [s for s in sessions if s["id"] == st.session_state.session_id][0]
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### ✏️ Edit Chat")
            edited = render_dict_form(session)
            if st.sidebar.button("💾 Save Edit", key=f"save_{session['id']}"):
                update_session(**edited)
                st.sidebar.success("Updated!")
                st.rerun()
            if st.sidebar.button("❌ Delete Chat", key=f"del_{session['id']}"):
                delete_session(session["id"])
                st.sidebar.success("Deleted.")
                st.session_state.session_id = None
                st.session_state.current_chat = None
                st.rerun()
    page = None
else:
    page = st.sidebar.selectbox("Page", [
        "Materials Obtain 🔍",
        "POSCAR Builder 💧",
        "INCAR Copilot 🧪",
        "Job Submission 🚀",
        "Error Handling 🐞",
        "Post Analysis 📊",
        "Extended Modules",
    ])

# --------- Main Dispatch ----------
if section == "Overview":
    if page == "Introduction":
        render_introduction()
    else:
        render_paper()
elif section == "ChatDFT":
    if st.session_state.get("session_id"):
        _chat_session(st.session_state["session_id"])
    else:
        st.info("Create or select a chat session from the sidebar.")
else:
    if page == "Materials Obtain 🔍":
        render_materials_obtain()
    elif page == "POSCAR Builder 💧":
        render_poscar_builder()
    elif page == "INCAR Copilot 🧪":
        render_incar_copilot()
    elif page == "Job Submission 🚀":
        render_job_submission()
    elif page == "Error Handling 🐞":
        render_error_handling()
    elif page == "Post Analysis 📊":
        render_post_analysis()
    else:
        st.error("Unknown tool page.")