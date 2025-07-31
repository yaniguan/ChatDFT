import streamlit as st
from utils.api import post
import time

st.set_page_config(page_title="ChatDFT", layout="wide")
st.title("🔬 ChatDFT")

# --------- Session State Init -----------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "workflow_steps" not in st.session_state:
    st.session_state.workflow_steps = []
if "workflow_results" not in st.session_state:
    st.session_state.workflow_results = {}
if "workflow_last_query" not in st.session_state:
    st.session_state.workflow_last_query = ""

# --------- API Helper -----------
def get_sessions():
    res = post("/chat/session/list", {})
    return res.get("sessions", [])

def create_session(name):
    res = post("/chat/session/create", {"name": name})
    return res.get("session_id", None)

def get_history(session_id):
    res = post("/chat/history", {"session_id": session_id, "limit": 1000})
    return res.get("messages", [])

def append_message(session_id, role, content):
    post("/chat/message/create", {"session_id": session_id, "role": role, "content": content})

# --------- Sidebar Chat Sessions -----------
SECTION_NAMES = ["Overview", "ChatDFT", "Tools"]
section = st.sidebar.selectbox("Section", SECTION_NAMES)

if section == "Overview":
    page = st.sidebar.selectbox("Page", ["Introduction", "Paper"])
elif section == "ChatDFT":
    st.sidebar.header("Chat Sessions")
    sessions = get_sessions()
    chat_names = [s["name"] for s in sessions]
    session_ids = [s["id"] for s in sessions]

    new_name = st.sidebar.text_input("New chat name", key="new_chat_name")
    if st.sidebar.button("🆕 Create Chat"):
        if new_name.strip():
            new_session_id = create_session(new_name)
            st.session_state.session_id = new_session_id
            st.session_state.current_chat = new_name
            # 重新拉一次 sessions，刷新 sidebar
            st.rerun()  # Streamlit 1.24+ 用 st.rerun()
        else:
            st.sidebar.warning("Please enter a non-empty name.")

    if chat_names:
        choice = st.sidebar.selectbox("Open chat", chat_names, key="open_chat")
        if choice:
            idx = chat_names.index(choice)
            st.session_state.session_id = session_ids[idx]
            st.session_state.current_chat = choice
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

def render_paper():
    st.header("Paper")
    st.info("This section will host the core paper details and annotations—coming soon.")

# --------- Chat Renderer (with DB sync) -----------
def render_chat_session(session_id):
    # 拉历史
    history = get_history(session_id)
    # st.header(f"🔬 <span style='font-weight:600'>ChatDFT — {st.session_state.current_chat}</span>", unsafe_allow_html=True)
# st.header(f"🔬 <span style='font-weight:600'>ChatDFT — {st.session_state.current_chat}</span>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='margin-bottom:6px;'>🔬 <span style='font-weight:600'>ChatDFT — {st.session_state.current_chat}</span></h2>",
        unsafe_allow_html=True
    )
    if not history:
        welcome = "👋 Hi! Welcome to ChatDFT, what can I help?"
        with st.chat_message("assistant"):
            st.markdown(welcome)
    for msg in history:
        st.markdown(msg["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Type your DFT question…")
    if not user_input:
        show_workflow_steps()
        return

    # 1. 保存用户消息
    append_message(session_id, "user", user_input)

    # 2. intent agent
    try:
        intent_res = post("/chat/intent", {"query": user_input})
        intent_msg = f"<b>🎯 Intent:</b> <span style='color:#146c43'>{intent_res.get('intent', 'N/A')}</span> &nbsp; <b>Confidence:</b> <span style='color:#146c43'>{intent_res.get('confidence', 0):.2f}</span>"
        st.markdown(
            f"""<div style='background:#e3f6e8;border-radius:10px;padding:10px 16px;margin:8px 0 4px 0;'>{intent_msg}</div>""",
            unsafe_allow_html=True)
        append_message(session_id, "assistant_intent", intent_msg)
    except Exception as e:
        st.warning(f"Intent agent error: {e}")
        intent_res = {}

    # 3. hypothesis agent
    try:
        hypo_res = post("/chat/hypothesis", {
            "query": user_input,
            "intent": intent_res.get("intent", None)
        })
        hypo_msg = f"<b>💡 Hypothesis:</b> {hypo_res.get('hypothesis', 'N/A')}"
        st.markdown(
            f"""<div style='background:#f8f5e6;border-radius:10px;padding:10px 16px;margin:8px 0 4px 0;'>{hypo_msg}</div>""",
            unsafe_allow_html=True)
        append_message(session_id, "assistant_hypothesis", hypo_msg)
    except Exception as e:
        st.warning(f"Hypothesis agent error: {e}")
        hypo_res = {}

    # 4. plan agent
    try:
        plan_res = post("/chat/plan", {
            "query": user_input,
            "intent": intent_res.get("intent"),
            "hypothesis": hypo_res.get("hypothesis"),
            "session_id": session_id,
        })
        if plan_res.get("ok") and plan_res.get("tasks"):
            plan_md = "<div style='margin:10px 0 4px 0;'><b>📋 Workflow Steps</b></div>"
            for t in plan_res["tasks"]:
                plan_md += f"""<div style='background:#f2f8fc;border-radius:8px;padding:10px 14px;margin-bottom:6px;'>
                <b>{t['id']}. {t['name']}</b> <span style='color:#888'>(agent: {t['agent']})</span><br>
                <span style='color:#222'>{t['description']}</span></div>"""
            st.markdown(plan_md, unsafe_allow_html=True)
            append_message(session_id, "assistant_plan", plan_md)
        else:
            st.warning("No workflow generated.")
    except Exception as e:
        st.warning(f"Plan agent error: {e}")

    # 5. knowledge agent
    try:
        knowledge_res = post("/chat/knowledge", {
            "query": user_input,
            "intent": intent_res.get("intent")
        })
        if knowledge_res and knowledge_res.get("result"):
            know_md = f"<b>📚 Knowledge:</b> {knowledge_res['result']}"
            st.markdown(
                f"""<div style='background:#f2fcf6;border-radius:10px;padding:10px 16px;margin:8px 0 4px 0;'>{know_md}</div>""",
                unsafe_allow_html=True)
            # 源文献也存一份
            srcs = ""
            for s in knowledge_res.get("sources", []):
                s_link = f"[{s['title']}]({s.get('url', '#')})" if s.get("url") else s["title"]
                srcs += f"• {s_link}\n"
                st.markdown(f"• {s_link}")
            append_message(session_id, "assistant_knowledge", know_md + "\n" + srcs)
    except Exception as e:
        st.info(f"Knowledge agent error: {e}")

    # 6. history agent
    try:
        hist = post("/chat/history", {"session_id": session_id, "limit": 5})
        if hist and hist.get("messages"):
            with st.expander("Recent History"):
                for m in hist["messages"]:
                    st.markdown(f"`{m['role']}`: {m['content']}")
    except Exception as e:
        st.info(f"History agent error: {e}")

def show_workflow_steps():
    steps = st.session_state.get("workflow_steps", [])
    results = st.session_state.get("workflow_results", {})
    user_input = st.session_state.get("workflow_last_query", "")

    if steps:
        st.markdown("<div style='margin-top:18px;margin-bottom:6px;font-weight:600'>📋 Workflow Steps</div>", unsafe_allow_html=True)
        for t in steps:
            key = f"workflow-task-{t['id']}"
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"""<div style='background:#f6f9fa;border-radius:8px;padding:10px 14px;margin-bottom:6px;'>
                    <b>{t['id']}. {t['name']}</b> <span style='color:#888'>(agent: {t['agent']})</span><br>
                    <span style='color:#222'>{t['description']}</span></div>""",
                    unsafe_allow_html=True)
            with col2:
                clicked = st.button("Run", key=key)
            if key in results or clicked:
                if clicked and key not in results:
                    agent_name = t.get("agent")
                    if agent_name:
                        agent_payload = {"query": user_input, "step_id": t["id"]}
                        agent_res = post(f"/agent/{agent_name}", agent_payload)
                        results[key] = agent_res
                        st.session_state.workflow_results = results
                    else:
                        results[key] = {"detail": "No agent implemented. Please check or ask LLM for manual instruction."}
                        st.session_state.workflow_results = results
                out = results[key]
                if out.get("ok"):
                    st.success(out.get("result", "Completed."))
                else:
                    st.info(out.get("detail", "No agent response."))

# --------- Tool Pages（你必须有这些定义）----------
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

# --------- Main Dispatch ----------
if section == "Overview":
    if page == "Introduction":
        render_introduction()
    else:
        render_paper()
elif section == "ChatDFT":
    if st.session_state.get("session_id"):
        render_chat_session(st.session_state.session_id)
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