# -*- coding: utf-8 -*-
import streamlit as st
from utils.api import post

st.set_page_config(page_title="ChatDFT", layout="wide")
st.title("🔬 ChatDFT")

# --------- Session State Init -----------
defaults = {
    "session_id": None,
    "current_chat": None,
    "workflow_results": {},
    "last_intent": {},
    "guided_query_to_send": None,
    "prefill_chat_box": "",
    "force_send_query": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------- API Helper -----------
def get_sessions():
    res = post("/chat/session/list", {})
    return res.get("sessions", [])

def create_session(**kwargs):
    res = post("/chat/session/create", kwargs)
    return res.get("session_id")

def update_session(**kwargs):
    res = post("/chat/session/update", kwargs)
    return res.get("ok", False)

def delete_session(session_id):
    res = post("/chat/session/delete", {"id": session_id})
    return res.get("ok", False)

def get_history(session_id):
    res = post("/chat/history", {"session_id": session_id, "limit": 1000})
    return res.get("messages", [])

def append_message(session_id, role, content):
    try:
        post("/chat/message/create", {"session_id": session_id, "role": role, "content": content})
    except Exception:
        pass

# --------- 小工具 ----------
def _sec_norm(s: str) -> str:
    """把 section 统一成无空格/无符号的小写，便于匹配"""
    return (s or "").lower().replace("&", "and").replace("-", "").replace(" ", "")

def _pick_section(tasks, key):
    """宽松匹配 plan_agent 的区段名字"""
    targets = {
        "ideas": {"ideasandliterature", "ideas", "literature"},
        "calc": {"calculationflow", "calcflow", "calculation"},
        "post": {"postanalysis", "post", "analysis"},
        "report": {"report"},
    }[key]
    return [t for t in tasks if _sec_norm(t.get("section")) in targets]

# --------- Workflow 渲染 ---------
def render_task(session_id: int, t: dict):
    # 任务标题与说明
    st.markdown(
        f"""<div style='background:#f7fbff;border-radius:8px;padding:12px 14px;margin:10px 0;'>
        <b>{t['id']}. {t.get('name','Task')}</b> <span style='color:#888'>(agent: {t.get('agent','-')})</span><br>
        <span style='color:#222'>{t.get('description','')}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # 表单区域
    form_vals = {}
    for f in t.get("params", {}).get("form", []):
        fkey = f"{t['id']}:{f.get('key','k')}"
        ftype = f.get("type", "text")
        label = f.get("label", fkey)
        help_ = f.get("help", "")
        default = f.get("value", "")

        if ftype == "number":
            form_vals[f["key"]] = st.number_input(
                label,
                value=float(default) if str(default) != "" else 0.0,
                step=float(f.get("step", 1.0)),
                min_value=float(f.get("min_value", -1e9)),
                max_value=float(f.get("max_value", 1e9)),
                key=fkey,
                help=help_,
            )
        elif ftype == "select":
            options = f.get("options", [])
            if default not in options and options:
                default = options[0]
            form_vals[f["key"]] = st.selectbox(
                label,
                options,
                index=options.index(default) if default in options else 0,
                key=fkey,
                help=help_,
            )
        elif ftype == "checkbox":
            form_vals[f["key"]] = st.checkbox(label, value=bool(default), key=fkey, help=help_)
        elif ftype == "textarea":
            form_vals[f["key"]] = st.text_area(label, value=str(default), key=fkey, help=help_)
        else:
            form_vals[f["key"]] = st.text_input(label, value=str(default), key=fkey, help=help_)

    # 运行按钮 + 输出容器
    col1, col2 = st.columns([1, 6])
    with col1:
        run = st.button("Run", key=f"run-{t['id']}")
    out_box = st.empty()

    if run:
        payload = {"session_id": session_id}
        payload.update(t.get("params", {}).get("payload", {}))
        payload.update(form_vals)
        try:
            with st.spinner("Running agent…"):
                res = post(f"/agent/{t.get('agent')}", payload)
        except Exception as e:
            res = {"ok": False, "detail": str(e)}

        if res.get("ok"):
            out_box.success(res.get("result", "Completed."))
        else:
            out_box.info(res.get("detail", "No agent response."))

def render_workflow(session_id: int, intent_obj: dict):
    # /chat/plan 支持传 {"intent": {...}} 或 {"fields": {...}}
    try:
        plan_res = post("/chat/plan", {"session_id": session_id, "intent": intent_obj})
    except Exception as e:
        st.warning(f"Plan error: {e}")
        return

    tasks = plan_res.get("tasks") or []
    # 从 plan 或 intent 里找 reaction_network
    rn = plan_res.get("reaction_network") or (intent_obj.get("fields") or {}).get("reaction_network") or []
    sugg = plan_res.get("suggestions") or {}

    def _sec(title):
        st.markdown(f"<h3 style='margin-top:20px'>{title}</h3>", unsafe_allow_html=True)

    _sec("🧭 Ideas & Literature")
    for t in _pick_section(tasks, "ideas"):
        render_task(session_id, t)

    _sec("🧪 Calculation Flow")
    if rn:
        with st.expander("Reaction Network (editable)", True):
            default_txt = "\n".join(rn)
            st.text_area("Elementary steps", value=default_txt, key="rxn_net_txt", height=120)
    for t in _pick_section(tasks, "calc"):
        render_task(session_id, t)

    _sec("📊 Post-analysis")
    for t in _pick_section(tasks, "post"):
        render_task(session_id, t)

    _sec("📝 Report")
    for t in _pick_section(tasks, "report"):
        render_task(session_id, t)

    # clickable suggestions (e.g., catalysts)
    for k, vals in sugg.items():
        st.markdown(f"**Suggestions – {k}:** " + " ".join([f'`{v}`' for v in vals]))
        cols = st.columns(len(vals))
        for i, v in enumerate(vals):
            if cols[i].button(f"Use {v}", key=f"use-{k}-{v}"):
                key = "catalyst" if k == "catalysts" else "material"
                (intent_obj.get("fields", intent_obj) or {})[key] = v
                st.session_state["last_intent"] = intent_obj
                st.rerun()

# --------- Guided Inquiry（引导式） ---------
def compose_query_from_spec(spec: dict) -> str:
    """把结构化 spec 拼成自然语言 query"""
    if not spec: return ""
    domain = spec.get("domain","")
    goal = spec.get("goal","").strip()
    addons = []

    if domain == "catalysis":
        rtype = spec.get("reaction_type","")
        cond = spec.get("conditions",{})
        catalyst = spec.get("catalyst")
        facet = spec.get("facet")
        rxn = spec.get("reaction")
        if rtype: addons.append(f"{rtype}")
        if rxn: addons.append(rxn)
        if catalyst: addons.append(f"catalyst {catalyst}")
        if facet: addons.append(f"facet {facet}")
        if cond:
            cs = ", ".join([f"{k}={v}" for k,v in cond.items() if v])
            addons.append(cs)
    elif domain == "batteries":
        btype = spec.get("battery_type")
        etype = spec.get("electrode_type")
        material = spec.get("material")
        mtype = spec.get("material_type")
        target = spec.get("target_property")
        extras = [x for x in [btype, etype, material, mtype, target] if x]
        if extras: addons.append(", ".join(extras))
    elif domain == "polymers":
        ptype = spec.get("polymer_type")
        rep = spec.get("repeat_unit")
        target = spec.get("target_property")
        extras = [x for x in [ptype, rep, target] if x]
        if extras: addons.append(", ".join(extras))

    head = f"[{domain}]".upper() if domain else ""
    mid = " · ".join(addons)
    tail = goal
    if head and mid and tail:
        return f"{head} {mid}. {tail}"
    return " ".join([x for x in [head, mid, tail] if x])

def guided_inquiry_panel():
    """返回 (query_text, spec)；UI 内部负责 Send/Copy 按钮"""
    with st.expander("🔎 Guided Inquiry (beta)", expanded=True):
        domain = st.selectbox("Domain", ["Catalysis","Batteries","Polymers"], key="g_domain")
        spec = {"domain": domain.lower()}

        if domain == "Catalysis":
            rxn_type = st.selectbox("Reaction type", ["Electrocatalysis","Thermocatalysis","Photocatalysis","Photoelectrocatalysis"], key="g_cat_type")
            spec["reaction_type"] = rxn_type
            spec["reaction"] = st.text_input("Reaction (e.g., HER / CO2RR / NRR)", key="g_rxn")

            cond = {}
            if rxn_type == "Electrocatalysis":
                cond["pH"] = st.text_input("pH (opt.)", key="g_ph")
                cond["Potential"] = st.text_input("Potential (e.g., 0 V vs RHE)", key="g_pot")
                cond["Electrolyte"] = st.text_input("Electrolyte (opt.)", key="g_ele")
                cond["Solvent"] = st.text_input("Solvent (opt.)", key="g_sol")
            elif rxn_type == "Thermocatalysis":
                cond["Temperature"] = st.text_input("Temperature (e.g., 700 K)", key="g_temp")
                cond["Pressure"] = st.text_input("Pressure (e.g., 1 atm)", key="g_pres")
                cond["Gas"] = st.text_input("Gas partial pressures (opt.)", key="g_gas")
            elif rxn_type == "Photocatalysis":
                cond["Wavelength"] = st.text_input("Light wavelength (nm)", key="g_wl")
                cond["Intensity"] = st.text_input("Light intensity (mW/cm²)", key="g_int")
                cond["Temperature"] = st.text_input("Temperature (opt.)", key="g_temp2")
                cond["Solvent"] = st.text_input("Solvent / pH (opt.)", key="g_sol2")
            else:  # Photoelectrocatalysis
                cond["Wavelength"] = st.text_input("Light wavelength (nm)", key="g_wl2")
                cond["Intensity"] = st.text_input("Light intensity (mW/cm²)", key="g_int2")
                cond["pH"] = st.text_input("pH (opt.)", key="g_ph2")
                cond["Potential"] = st.text_input("Potential (e.g., vs RHE)", key="g_pot2")

            spec["conditions"] = {k:v for k,v in cond.items() if v}
            spec["catalyst"] = st.text_input("Catalyst (opt., e.g., Pt, Ni, Mo2C)", key="g_cat")
            spec["facet"] = st.text_input("Facet (opt., e.g., Pt(111))", key="g_facet")
            spec["goal"] = st.text_area("Goal / target", key="g_goal", height=80)

        elif domain == "Batteries":
            spec["battery_type"] = st.selectbox("Battery type", ["Li-ion","Na-ion","K-ion","Mg-ion","Solid-state","Metal-air"], key="g_batt")
            spec["electrode_type"] = st.selectbox("Electrode", ["Cathode","Anode","Both"], key="g_elec")
            spec["material"] = st.text_input("Material (formula / file name)", key="g_mat")
            spec["material_type"] = st.text_input("Material type (oxide, sulfide, alloy...)", key="g_mtype")
            spec["target_property"] = st.text_input("Target property (voltage, capacity, diffusion barrier…)", key="g_btarget")
            spec["goal"] = st.text_area("Goal / target", key="g_goal_b", height=80)

        else:  # Polymers
            spec["polymer_type"] = st.selectbox("Polymer type", ["Thermoplastic","Thermoset","Biopolymer","Conductive polymer"], key="g_ptype")
            spec["repeat_unit"] = st.text_input("Repeat unit (SMILES/InChI/POSCAR)", key="g_rep")
            spec["target_property"] = st.text_input("Target property (Tg, strength, dielectric…)", key="g_pty")
            spec["goal"] = st.text_area("Goal / target", key="g_goal_p", height=80)

        q = compose_query_from_spec(spec)
        st.caption("Preview")
        st.code(q or "(empty)", language=None)

        c1, c2 = st.columns(2)
        if c1.button("➡️ Send", key="g_send"):
            st.session_state["guided_query_to_send"] = q
            st.session_state["force_send_query"] = True
            st.rerun()
        if c2.button("📋 Copy to chat", key="g_copy"):
            st.session_state["prefill_chat_box"] = q
            st.rerun()

# --------- Chat 主体 -----------
def handle_query(session_id: int, user_input: str):
    if not user_input.strip():
        return

    # 显示用户气泡（修复 Guided Inquiry 无显示的问题）
    with st.chat_message("user"):
        st.markdown(user_input)

    # 持久化用户消息
    append_message(session_id, "user", user_input)

    # 1) Intent
    try:
        intent_res = post("/chat/intent", {"query": user_input}) or {}
    except Exception as e:
        intent_res = {}
        st.warning(f"Intent error: {e}")

    # 顶部信息条
    pill_md = (
        f"**🎯Intent:** {intent_res.get('intent','-')}  "
        f"**Stage:** {intent_res.get('stage','-')}  "
        f"**Domain:** {intent_res.get('domain','-')}  "
        f"**Confidence:** {intent_res.get('confidence',0):.2f}"
    )
    with st.chat_message("assistant"):
        st.markdown(pill_md)

    # 详细 Intent Summary（兼容 summary / summary_md）
    summary = intent_res.get("summary") or intent_res.get("summary_md")
    if summary:
        with st.chat_message("assistant"):
            st.markdown(summary)
        append_message(session_id, "assistant_intent", summary)
    else:
        append_message(session_id, "assistant_intent", pill_md)

    st.session_state["last_intent"] = intent_res or {}

    # 2) Hypothesis
    try:
        hypo_res = post("/chat/hypothesis", {"query": user_input, "intent": intent_res}) or {}
    except Exception as e:
        hypo_res = {"hypothesis":"Please provide more context."}
        st.info(f"Hypothesis error: {e}")

    with st.chat_message("assistant"):
        st.markdown(f"**💡Hypothesis:** {hypo_res.get('hypothesis','N/A')}")
    append_message(session_id, "assistant_hypothesis", f"**Hypothesis:** {hypo_res.get('hypothesis','N/A')}")

    # 3) Workflow
    render_workflow(session_id, intent_res.get("fields") or intent_res or {})

def render_chat_session(session_id):
    st.markdown(
        f"<h2 style='margin-bottom:6px;'>🔬 <span style='font-weight:600'>ChatDFT — {st.session_state.current_chat or ''}</span></h2>",
        unsafe_allow_html=True
    )

    # —— 顶部：Guided Inquiry 面板
    guided_inquiry_panel()

    # —— 历史
    history = get_history(session_id)
    if not history:
        with st.chat_message("assistant"):
            st.markdown("👋 Hi! Welcome to ChatDFT, what can I help?")
    else:
        for m in history:
            role = m.get("role","assistant")
            content = m.get("content","")
            if role.startswith("assistant"):
                with st.chat_message("assistant"):
                    st.markdown(content, unsafe_allow_html=True)
            else:
                with st.chat_message("user"):
                    st.markdown(content)

    # —— 如果 Guided Inquiry 触发了 send，则直接处理
    if st.session_state.get("force_send_query") and st.session_state.get("guided_query_to_send"):
        q = st.session_state["guided_query_to_send"]
        st.session_state["guided_query_to_send"] = None
        st.session_state["force_send_query"] = False
        handle_query(session_id, q)
        return  # 本轮已输出

    # —— 自由输入聊天（chat_input 不支持 value 参数）
    # 若有引导表单生成的“预填内容”，直接当做一次输入处理
    prefill = st.session_state.pop("prefill_chat_box", "")
    if prefill:
        handle_query(session_id, prefill)
        return

    user_input = st.chat_input("Type your DFT question…", key="chat_box")
    if user_input:
        handle_query(session_id, user_input)

# --------- Papers Tab ----------
def render_papers_tab():
    st.subheader("📚 Papers")
    q = st.text_input("Keywords", key="paper_kw", placeholder="e.g., HER Pt(111) alkaline kinetics")
    col1, col2 = st.columns([1,5])
    do = col1.button("Search")
    if do and q.strip():
        with st.spinner("Searching…"):
            try:
                # 你的 knowledge_agent 若是 /chat/knowledge 就能直接用；否则会优雅降级
                res = post("/chat/knowledge", {"query": q}) or {}
            except Exception as e:
                res = {"ok": False, "error": str(e)}
        if not res.get("ok"):
            st.info("No knowledge endpoint available or empty result.")
            return
        papers = res.get("papers") or res.get("results") or []
        if not papers:
            st.write("No papers found.")
        else:
            for i, p in enumerate(papers, 1):
                title = p.get("title") or f"Paper {i}"
                meta = " · ".join([x for x in [p.get("venue"), p.get("year") and str(p.get("year"))] if x])
                st.markdown(f"**{i}. {title}**  \n{meta}")
                if p.get("summary"):
                    with st.expander("Summary"):
                        st.write(p["summary"])

# --------- Tools Tab ----------
def render_tools_tab():
    st.subheader("🛠️ Tools")

    st.markdown("**Potential scale conversion**")
    with st.form("pot_form"):
        val = st.number_input("E vs RHE (V)", value=0.0)
        ph  = st.number_input("pH", value=0.0)
        submitted = st.form_submit_button("Convert to SHE")
        if submitted:
            eshe = val - 0.0591 * ph
            st.success(f"E vs SHE ≈ **{eshe:.3f} V**")

    st.markdown("---")
    st.markdown("**Energy unit converter**")
    with st.form("unit_form"):
        ev = st.number_input("Energy (eV)", value=1.0)
        submitted2 = st.form_submit_button("Convert")
        if submitted2:
            kjmol = ev * 96.485
            kcalmol = kjmol / 4.184
            st.success(f"1 eV ≈ **{kjmol:.3f} kJ/mol** ≈ **{kcalmol:.3f} kcal/mol**")

# --------- Sidebar Sessions -----------
st.sidebar.header("Sessions")

with st.sidebar.expander("➕ New Chat", True):
    name = st.text_input("Name*", key="new_name")
    uid = st.number_input("User ID*", value=1, step=1, key="new_uid")
    proj = st.text_input("Project", key="new_proj")
    tags = st.text_input("Tags (JSON 或 逗号分隔)", key="new_tags")
    desc = st.text_area("Description", key="new_desc")
    status = st.selectbox("Status", ["active", "archived"], index=0, key="new_status")
    pinned = st.checkbox("Pinned", key="new_pinned")
    if st.button("🆕 Create", key="btn_create"):
        if not name.strip():
            st.warning("Name is required.")
        else:
            sid = create_session(name=name, user_id=int(uid), project=proj,
                                 tags=tags, description=desc, status=status, pinned=pinned)
            if sid:
                st.session_state.session_id = sid
                st.session_state.current_chat = name
                st.rerun()

with st.sidebar.expander("📂 Open chat", True):
    sessions = get_sessions()
    if not sessions:
        st.info("No sessions yet. Create one above.")
    else:
        labels = [f"{s['name']}  ·  #{s['id']}  ·  {s.get('project','')}" for s in sessions]
        idx = st.selectbox("Select", list(range(len(labels))), format_func=lambda i: labels[i], key="open_idx")
        cur = sessions[idx]
        c1, c2, c3 = st.columns(3)
        if c1.button("Open", key="btn_open"):
            st.session_state.session_id = cur["id"]
            st.session_state.current_chat = cur["name"]
            st.rerun()
        if c2.button("Refresh", key="btn_refresh"):
            st.rerun()
        if c3.button("Delete", key="btn_del"):
            delete_session(cur["id"])
            st.rerun()

# --------- Main with Tabs ---------
tab_home, tab_papers, tab_tools = st.tabs(["🏠 Home", "📚 Papers", "🛠️ Tools"])

with tab_home:
    if st.session_state.get("session_id"):
        render_chat_session(st.session_state.session_id)
    else:
        st.info("Create or select a chat session from the sidebar.")

with tab_papers:
    render_papers_tab()

with tab_tools:
    render_tools_tab()