import streamlit as st
from utils.api import post

# 设置页面属性（建议放在最前）
st.set_page_config(page_title="ChatDFT", layout="wide")

# 页面标题
st.title("🔬 ChatDFT")

# --------- Session State Init -----------
# ✅ 初始化前端状态
defaults = {
    # 会话管理
    "session_id": None,                # 当前 session 唯一标识
    "session_name": None,             # 当前 session 名称（可选）
    "chat_history": [],               # 当前 session 的完整对话消息（List of {role, content}）

    # 用户输入
    "prefill_chat_box": "",           # 输入框预填内容
    "guided_query_to_send": None,     # 由按钮或UI生成的自动query
    "force_send_query": False,        # 强制重新生成workflow等

    # 模块状态（用于展示和调用后端）
    "last_intent": None,              # {"domain": ..., "reaction": ..., ...}
    "last_hypothesis": None,          # {"hypothesis": ..., "confidence": ...}
    "last_knowledge": None,           # {"summary": ..., "sources": [...]}
    "last_plan": None,                # {"steps": [...], "plan_summary": ...}
    "workflow_results": {},           # 每一步执行后的输出

    # UI 状态
    "expander_state": {},             # 保存每个模块expander是否展开
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------- 页面布局 ---------
