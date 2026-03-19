# client/app.py
"""
ChatDFT — main Streamlit application.

Run:
    conda activate llm-agent
    streamlit run client/app.py
"""

import json
import time
import streamlit as st
from typing import Any, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client.utils import api

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatDFT",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* pill badges */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 3px;
}
.pill-blue   { background:#dbeafe; color:#1d4ed8; }
.pill-green  { background:#dcfce7; color:#15803d; }
.pill-amber  { background:#fef3c7; color:#b45309; }
.pill-purple { background:#ede9fe; color:#6d28d9; }
.pill-red    { background:#fee2e2; color:#b91c1c; }

/* agent step cards */
.step-card {
    border-left: 3px solid #6366f1;
    padding: 10px 14px;
    margin: 6px 0;
    background: #f8f9ff;
    border-radius: 0 8px 8px 0;
}
.step-header {
    font-weight: 700;
    font-size: 0.85rem;
    color: #4338ca;
    margin-bottom: 4px;
}

/* task cards */
.task-card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    background: #fff;
}
.task-name { font-weight: 600; font-size: 0.9rem; }
.task-type { font-size: 0.75rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "logged_in": False,
    "username": "",
    "session_id": None,
    "session_name": "",
    "chat_messages": [],      # {role, content, kind, data}
    "sidebar_panel": "💬 Sessions",
    "last_intent": None,
    "last_hypothesis": None,
    "last_plan": None,
    "cluster_config": {
        "host": "hoffman2.idre.ucla.edu",
        "user": "yaniguan",
        "scheduler": "sge",
        "account": "",
        "walltime": "24:00:00",
        "ntasks": 32,
        "remote_base": "/u/scratch/y/yaniguan/chatdft",
        "env_setup": (
            "source /u/local/Modules/default/init/modules.sh\n"
            "module add intel/17.0.7\n"
            "export PYTHONPATH=/u/home/y/yaniguan/miniconda3/envs/ase:$PYTHONPATH\n"
            "export PATH=/u/home/y/yaniguan/miniconda3/envs/ase/bin:$PATH\n"
            "export VASP_PP_PATH=$HOME/vasp/mypps\n"
            "export OMP_NUM_THREAD=1\n"
            "export I_MPI_COMPATIBILITY=4\n"
            "export VASP_COMMAND='mpirun -np ${NSLOTS} ~/vasp_std_vtst_sol'"
        ),
    },
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pill(label: str, color: str = "blue") -> str:
    return f'<span class="pill pill-{color}">{label}</span>'


def _conf_color(c: float) -> str:
    if c >= 0.75: return "green"
    if c >= 0.50: return "amber"
    return "red"


def _add_message(role: str, content: str, kind: str = "text", data: Any = None):
    st.session_state.chat_messages.append(
        {"role": role, "content": content, "kind": kind, "data": data}
    )


def _load_session_history(session_id: int) -> List[dict]:
    """Load and reconstruct chat history for a session from the DB."""
    import json as _json_mod

    state = api.get_session_state(session_id)
    msgs_r = api.get_session_messages(session_id, limit=200)
    messages = []

    # Reconstruct user messages from DB (msg_type="user")
    raw_msgs = msgs_r.get("messages", []) if msgs_r.get("ok") else []
    # Reverse so oldest first
    raw_msgs = list(reversed(raw_msgs))

    # Group by "user→pipeline" pairing via creation order
    # Collect user texts and pipeline artifacts separately
    user_texts: List[str] = []
    for m in raw_msgs:
        if m.get("msg_type") == "user":
            try:
                user_texts.append(_json_mod.loads(m["content"]) if m["content"].startswith("{") else m["content"])
            except Exception:
                user_texts.append(m.get("content", ""))

    # Reconstruct one pipeline card per set of user messages (simplified: show latest state)
    # If we have intent/plan in state, show a pipeline card
    intent = state.get("intent") if state.get("ok") else None
    # intent may be a JSON string from DB — ensure it's a dict
    if isinstance(intent, str):
        try:
            intent = _json_mod.loads(intent)
        except Exception:
            intent = {}
    if not isinstance(intent, dict):
        intent = {}
    plan_raw = state.get("plan_raw") if state.get("ok") else {}
    hyp_raw = state.get("hypothesis") if state.get("ok") else None

    # Parse hypothesis (may be a JSON string or dict)
    if isinstance(hyp_raw, str):
        try:
            hyp_raw = _json_mod.loads(hyp_raw)
        except Exception:
            hyp_raw = {"md": hyp_raw, "steps": [], "graph": {}}
    elif not isinstance(hyp_raw, dict):
        hyp_raw = {}

    if intent or plan_raw:
        # Re-add user messages
        for txt in user_texts:
            messages.append({"role": "user", "content": str(txt), "kind": "text", "data": None})

        hypothesis = {
            "md": (hyp_raw.get("md") or "") if isinstance(hyp_raw, dict) else "",
            "steps": (hyp_raw.get("steps") or []) if isinstance(hyp_raw, dict) else [],
            "graph": {},
            "confidence": (hyp_raw.get("confidence") or 0) if isinstance(hyp_raw, dict) else 0,
        }
        # Populate graph from state rxn fields
        if state.get("ok"):
            hypothesis["graph"] = {
                "intermediates": state.get("intermediates") or [],
                "ts_edges": state.get("ts_candidates") or [],
                "coads_pairs": state.get("coads_pairs") or [],
            }

        raw_plan = plan_raw if isinstance(plan_raw, dict) else {"tasks": state.get("plan_tasks", [])}
        # Filter out old-style sub-tasks (1.2 Relax / 1.3 Retrieve) that pre-date the compound task redesign
        _OLD_TASK_PATTERNS = ("relax", "retrieve & save", "retrieve and save")
        if "tasks" in raw_plan:
            raw_plan["tasks"] = [
                t for t in raw_plan["tasks"]
                if not any(p in (t.get("name") or "").lower() for p in _OLD_TASK_PATTERNS)
            ]
        pipeline_data = {
            "intent": intent or {},
            "hypothesis": hypothesis,
            "plan": raw_plan,
        }

        # Update session state caches
        st.session_state.last_intent    = intent
        st.session_state.last_hypothesis = hypothesis
        st.session_state.last_plan       = pipeline_data["plan"]

        # Restore persisted task step states (POSCAR, scripts, job_id, results)
        _restore_task_states(session_id)

        summary = _pipeline_summary(pipeline_data)
        messages.append({
            "role": "assistant",
            "content": summary,
            "kind": "pipeline",
            "data": pipeline_data,
        })

    return messages


def _save_task_state(session_id: int, task_id: int, **kwargs):
    """Persist per-task step state to DB (fire-and-forget; errors are non-fatal)."""
    try:
        api.save_task_state(session_id, task_id, **kwargs)
    except Exception:
        pass  # persistence is best-effort


def _clear_task_states():
    """Remove all task_{tid}_* keys from session_state to prevent stale state across sessions."""
    stale = [k for k in list(st.session_state.keys()) if k.startswith("task_")]
    for k in stale:
        del st.session_state[k]


def _restore_task_states(session_id: int):
    """Load all task states from DB and populate st.session_state."""
    try:
        resp = api.list_task_states(session_id)
        if not resp.get("ok"):
            return
        for ts in resp.get("states", []):
            tid = ts["task_plan_id"]
            key_poscar  = f"task_{tid}_poscar"
            key_plot    = f"task_{tid}_plot"
            key_scripts = f"task_{tid}_scripts"
            key_job_id  = f"task_{tid}_job_id"
            key_remote  = f"task_{tid}_remote"
            key_results = f"task_{tid}_results"

            if ts.get("poscar"):
                st.session_state[key_poscar] = ts["poscar"]
            if ts.get("plot_png_b64"):
                st.session_state[key_plot] = ts["plot_png_b64"]
            if ts.get("all_configs"):
                st.session_state[f"{key_poscar}_all_configs"] = ts["all_configs"]
                st.session_state[f"{key_poscar}_selected"] = ts.get("selected_config", 0)
            if ts.get("scripts"):
                st.session_state[key_scripts] = ts["scripts"]
            if ts.get("job_id"):
                st.session_state[key_job_id] = ts["job_id"]
            if ts.get("remote_path"):
                st.session_state[key_remote] = ts["remote_path"]
            if ts.get("results"):
                st.session_state[key_results] = ts["results"]
    except Exception:
        pass  # non-fatal


def _backend_ok() -> bool:
    r = api.get("/")
    return r.get("ok", False) or "detail" not in r


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center">
            <div style="font-size:4rem">⚛️</div>
            <h1 style="font-size:3rem;font-weight:800;letter-spacing:-1px;margin:0">ChatDFT</h1>
            <p style="font-size:1.1rem;color:#64748b;margin-top:4px">
                First-principles calculations, driven by natural language.<br>
                Describe your reaction — we handle the rest.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### Get started")
            name = st.text_input("Your name", placeholder="e.g. Alex", key="login_name")
            st.caption("No password needed — this is your personal research workspace.")
            if st.button("Enter ChatDFT →", type="primary", use_container_width=True):
                if name.strip():
                    st.session_state.logged_in = True
                    st.session_state.username = name.strip()
                    st.rerun()
                else:
                    st.warning("Please enter your name.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;color:#94a3b8;font-size:0.8rem">
            Intent → Mechanism → Plan → Execute → Analyze
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Header
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            <span style="font-size:1.6rem">⚛️</span>
            <span style="font-size:1.2rem;font-weight:700">ChatDFT</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"👤 {st.session_state.username}")
        st.divider()

        # Panel selector
        panel = st.radio(
            "Panel",
            ["💬 Sessions", "📚 Literature", "📊 Analysis",
             "🔬 QA & Debug", "🧱 Structure", "🗄 HTP Dataset", "⚙️ Settings"],
            label_visibility="collapsed",
            key="sidebar_panel_radio",
        )
        st.session_state.sidebar_panel = panel
        st.divider()

        # ── Sessions panel ────────────────────────────────────────────────
        if panel == "💬 Sessions":
            _render_session_panel()

        # ── Literature panel ──────────────────────────────────────────────
        elif panel == "📚 Literature":
            _render_literature_panel()

        # ── Analysis panel ────────────────────────────────────────────────
        elif panel == "📊 Analysis":
            _render_analysis_panel()

        # ── QA & Debug panel ─────────────────────────────────────────────
        elif panel == "🔬 QA & Debug":
            _render_qa_panel()

        # ── Structure Workspace panel ─────────────────────────────────────
        elif panel == "🧱 Structure":
            _render_structure_panel()

        # ── HTP Dataset panel ─────────────────────────────────────────────
        elif panel == "🗄 HTP Dataset":
            _render_htp_panel()

        # ── Settings panel ────────────────────────────────────────────────
        elif panel == "⚙️ Settings":
            _render_settings_panel()

        # Bottom: backend status
        st.divider()
        _render_backend_status()


def _render_session_panel():
    st.markdown("**Sessions**")

    # New session
    with st.expander("＋ New session", expanded=not bool(st.session_state.session_id)):
        new_name = st.text_input("Name", placeholder="e.g. Pt111 butane dehydrogenation", key="new_sess_name")
        new_proj = st.text_input("Project (optional)", key="new_sess_proj")
        if st.button("Create", type="primary", use_container_width=True, key="create_sess_btn"):
            if new_name.strip():
                r = api.create_session(new_name.strip(), project=new_proj.strip())
                if r.get("ok"):
                    sid = r.get("id") or r.get("session_id")
                    _clear_task_states()
                    st.session_state.session_id = sid
                    st.session_state.session_name = new_name.strip()
                    st.session_state.chat_messages = []
                    st.session_state.last_intent = None
                    st.session_state.last_hypothesis = None
                    st.session_state.last_plan = None
                    st.success(f"Session #{sid} created")
                    st.rerun()
                else:
                    st.error(r.get("detail", "Failed"))
            else:
                st.warning("Enter a name.")

    # Existing sessions
    r = api.list_sessions()
    sessions: List[dict] = r.get("sessions", [])
    if not sessions:
        st.caption("No sessions yet.")
        return

    st.markdown(f"**{len(sessions)} session(s)**")
    for s in reversed(sessions[-20:]):
        sid = s.get("id")
        sname = s.get("name", f"Session {sid}")
        is_active = sid == st.session_state.session_id
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{'▶ ' if is_active else ''}{sname}"
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                _clear_task_states()
                st.session_state.session_id = sid
                st.session_state.session_name = sname
                st.session_state.last_intent = None
                st.session_state.last_hypothesis = None
                st.session_state.last_plan = None
                st.session_state.chat_messages = _load_session_history(sid)
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{sid}", help="Delete"):
                api.delete_session(sid)
                if sid == st.session_state.session_id:
                    st.session_state.session_id = None
                    st.session_state.session_name = ""
                    st.session_state.chat_messages = []
                st.rerun()


def _render_literature_panel():
    st.markdown("**Literature Search**")

    tab_search, tab_ctx = st.tabs(["🔍 Search", "🧠 Agent Context"])

    with tab_search:
        q = st.text_input("Search query", placeholder="C4H10 dehydrogenation Pt",
                          key="lit_query")
        col_k, col_re = st.columns(2)
        with col_k:
            limit = st.number_input("Results", min_value=3, max_value=20, value=8,
                                    step=1, key="lit_limit")
        with col_re:
            use_rerank = st.checkbox("Cross-encoder rerank", value=True, key="lit_rerank",
                                     help="Rerank with cross-encoder/ms-marco-MiniLM-L-6-v2")

        if st.button("Search", use_container_width=True, key="lit_search"):
            if q.strip():
                with st.spinner("Searching..."):
                    r = api.search_knowledge(
                        st.session_state.session_id or 0,
                        q.strip(),
                        limit=int(limit),
                    )
                papers = r.get("records", r.get("results", []))
                if papers:
                    for p in papers:
                        title   = p.get("title", "?")
                        year    = p.get("year") or p.get("source_id", "")
                        url     = p.get("url", "")
                        section = p.get("section", "")
                        score   = (p.get("rerank_score") or p.get("rrf_score")
                                   or p.get("relevance") or p.get("score") or 0)
                        with st.container(border=True):
                            col_t, col_s = st.columns([4, 1])
                            with col_t:
                                st.markdown(f"**{title}** ({year})")
                                if url:
                                    st.markdown(f"[Link]({url})")
                            with col_s:
                                if score:
                                    st.metric("Score", f"{score:.2f}")
                                if section:
                                    st.caption(f"§ {section}")
                        text_snippet = p.get("text", "")[:300]
                        if text_snippet:
                            with st.expander("Snippet", expanded=False):
                                st.caption(text_snippet)
                else:
                    st.info("No results found.")
            else:
                st.warning("Enter a query.")

    with tab_ctx:
        st.caption(
            "Load all context an agent would see: literature + prior structures "
            "+ converged DFT results from this session."
        )
        sid = st.session_state.get("session_id")
        if not sid:
            st.info("Open a session first.")
        else:
            ctx_q = st.text_input("Context query", key="ctx_query",
                                  placeholder="CO adsorption on Cu(111) hollow site")
            ctx_tags = st.text_input("Tag filter (comma-sep)", key="ctx_tags",
                                     placeholder="CO2RR, Cu",
                                     help="Optional tags to filter literature")
            if st.button("Load Agent Context", type="primary",
                         use_container_width=True, key="ctx_load"):
                if ctx_q.strip():
                    tags = [t.strip() for t in ctx_tags.split(",") if t.strip()]
                    with st.spinner("Loading context from all sources..."):
                        r = api.post("/chat/knowledge", {
                            "session_id": sid,
                            "query": ctx_q.strip(),
                            "limit": 6,
                            "tags_filter": tags or None,
                            "include_structures": True,
                            "include_dft_results": True,
                        })
                    # ── Literature ──────────────────────────────────────
                    lit = r.get("records", r.get("results", []))
                    if lit:
                        st.markdown("##### 📚 Literature")
                        for p in lit[:6]:
                            section = p.get("section", "")
                            score   = (p.get("rerank_score") or p.get("rrf_score")
                                       or p.get("score") or 0)
                            with st.container(border=True):
                                badge = f" `§{section}`" if section else ""
                                st.markdown(f"**{p.get('title','?')}** ({p.get('year','?')})"
                                            f"{badge}  score={score:.3f}")
                                if p.get("text"):
                                    st.caption(p["text"][:250])

                    # ── Prior structures ────────────────────────────────
                    structs = r.get("structures", [])
                    if structs:
                        st.markdown("##### 🧱 Prior Structures (this session)")
                        for s in structs:
                            lbl = (s.get("natural_language") or
                                   f"{s.get('material','')}({s.get('facet','')}) "
                                   f"+ {s.get('adsorbates',[])} — {s.get('formula','')}")
                            with st.container(border=True):
                                col_l, col_c = st.columns([3, 1])
                                with col_l:
                                    st.caption(lbl)
                                    if s.get("is_optimized"):
                                        st.caption(f"✅ Optimized  E={s.get('energy_eV'):.4f} eV"
                                                   if s.get("energy_eV") else "✅ Optimized")
                                with col_c:
                                    if s.get("poscar_content"):
                                        st.download_button(
                                            "POSCAR",
                                            data=s["poscar_content"],
                                            file_name=f"prior_{s.get('id','')}.POSCAR",
                                            key=f"ctx_poscar_{s.get('id','')}",
                                        )

                    # ── DFT results ─────────────────────────────────────
                    dft = r.get("dft_results", [])
                    if dft:
                        st.markdown("##### ⚡ Converged DFT Results (this session)")
                        rows = []
                        for d in dft:
                            rows.append({
                                "Type": d.get("result_type",""),
                                "Species": d.get("species",""),
                                "Surface": d.get("surface",""),
                                "Value (eV)": f"{d.get('value',0):.4f}",
                                "Site": d.get("site",""),
                            })
                        import pandas as pd
                        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                     hide_index=True)
                    if not lit and not structs and not dft:
                        st.info("No context found. Try a different query.")
                else:
                    st.warning("Enter a context query.")


def _render_analysis_panel():
    st.markdown("**Results Analysis**")
    sid = st.session_state.session_id
    if not sid:
        st.caption("Open a session first.")
        return

    # Quick summary
    r = api.get(f"/chat/analyze/summary/{sid}")
    if r.get("ok"):
        stats = r.get("dft_stats", {})
        st.metric("Converged calculations", stats.get("converged", 0))
        st.metric("Failed", stats.get("failed", 0))

        if r.get("has_analysis"):
            a = r.get("analysis", {})
            status = a.get("publication_checklist", {}).get("status", "?")
            color_map = {"incomplete": "🔴", "near_complete": "🟡", "publishable": "🟢"}
            st.markdown(f"Publication status: {color_map.get(status, '⚪')} `{status}`")

    focus = st.selectbox(
        "Analysis focus",
        ["overall progress and next steps",
         "mechanism gaps",
         "publication readiness",
         "kinetics and barriers",
         "electrochemical potential effects"],
        key="analysis_focus",
    )
    if st.button("Run Analysis", type="primary", use_container_width=True, key="run_analysis"):
        with st.spinner("Analysing results..."):
            r = api.run_analyze(sid, focus=focus)
        if r.get("ok"):
            _add_message(
                "assistant",
                r.get("summary_md", ""),
                kind="analysis",
                data=r,
            )
            st.rerun()
        else:
            st.error(r.get("detail", "Analysis failed"))

    st.divider()
    st.markdown("**ΔG Free Energy Diagram**")
    reaction_opts = ["CO2RR", "HER", "OER", "NRR", "NO3RR"]
    rxn  = st.selectbox("Reaction", reaction_opts, key="fe_reaction")
    T_fe = st.number_input("Temperature (K)", value=298.15, step=10.0, key="fe_T")
    U_fe = st.number_input("Potential (V vs RHE)", value=0.0, step=0.1, key="fe_U")
    if st.button("Generate ΔG Diagram", use_container_width=True, key="fe_btn"):
        with st.spinner("Building free energy diagram..."):
            r = api.post("/chat/qa/free_energy", {
                "session_id": sid, "reaction": rxn,
                "temperature": T_fe, "potential_V": U_fe,
                "use_known_pathway": True,
            })
        if r.get("ok"):
            _add_message("assistant", r.get("interpretation_md", ""),
                         kind="free_energy", data=r)
            st.rerun()
        else:
            st.error(r.get("detail", "Failed"))

    st.divider()
    st.markdown("**Microkinetic Model**")
    T_mk = st.number_input("Temperature (K)", value=500.0, step=50.0, key="mk_T")
    U_mk = st.number_input("Potential (V vs RHE)", value=-0.8, step=0.1, key="mk_U")
    if st.button("Run Microkinetics", use_container_width=True, key="mk_btn"):
        with st.spinner("Running microkinetic model..."):
            r = api.post("/chat/qa/microkinetics", {
                "session_id": sid, "reaction": rxn,
                "temperature": T_mk, "potential_V": U_mk,
                "use_known_pathway": True,
            })
        if r.get("ok"):
            _add_message("assistant", r.get("interpretation_md", ""),
                         kind="microkinetics", data=r)
            st.rerun()
        else:
            st.error(r.get("detail", "Failed"))


def _render_qa_panel():
    """QA & Benchmarking Hub — functional advisor, surface check, OUTCAR debug."""
    st.markdown("**QA & Benchmarking**")

    tab1, tab2, tab3 = st.tabs(["🔧 Functional", "🗺 Surface", "🐛 Debug OUTCAR"])

    with tab1:
        st.caption("Get DFT functional / INCAR recommendations for your system.")
        sys_desc = st.text_area(
            "System description",
            placeholder="e.g. CO adsorption on Pt(111) at 300K",
            key="qa_sys_desc", height=80,
        )
        if st.button("Get Recommendation", use_container_width=True, key="qa_func_btn"):
            if sys_desc.strip():
                with st.spinner("Consulting functional advisor..."):
                    r = api.post("/chat/qa/functional", {"system": sys_desc.strip()})
                if r.get("ok"):
                    _add_message("assistant", r.get("advisory_md", ""),
                                 kind="qa_functional", data=r)
                    st.rerun()
                else:
                    st.error(r.get("detail", "Failed"))
            else:
                st.warning("Describe the system first.")

    with tab2:
        st.caption("Check if your surface model is stable under reaction conditions.")
        col1, col2 = st.columns(2)
        with col1:
            mat = st.text_input("Material", value="Pt", key="qa_mat")
        with col2:
            facet = st.text_input("Facet", value="111", key="qa_facet")
        cond_T = st.number_input("Temperature (K)", value=300, key="qa_surf_T")
        if st.button("Check Surface", use_container_width=True, key="qa_surf_btn"):
            with st.spinner("Checking surface stability..."):
                r = api.post("/chat/qa/surface", {
                    "material": mat, "facet": facet,
                    "conditions": {"temperature": cond_T},
                })
            if r.get("ok"):
                warn = r.get("stability_warning")
                note = r.get("llm_note", "")
                if warn:
                    st.warning(f"⚠️ {warn.get('warning','')}")
                    st.info(f"💡 {warn.get('mitigation','')}")
                else:
                    st.success(f"✅ {mat}({facet}) appears stable — no known reconstructions flagged.")
                if note:
                    st.markdown(note)
            else:
                st.error(r.get("detail", "Failed"))

    with tab3:
        st.caption("Paste a VASP job directory path to get diagnosis + INCAR fixes.")
        job_dir = st.text_input(
            "Job directory",
            placeholder="/path/to/vasp/job",
            key="qa_job_dir",
        )
        if st.button("Analyse OUTCAR", use_container_width=True, key="qa_debug_btn"):
            if job_dir.strip():
                with st.spinner("Analysing OUTCAR..."):
                    r = api.post("/chat/qa/debug", {"job_dir": job_dir.strip()})
                if r.get("ok"):
                    _add_message("assistant", r.get("llm_advice", r.get("summary", "")),
                                 kind="qa_debug", data=r)
                    st.rerun()
                else:
                    st.error(r.get("detail", "Job directory not found or no OUTCAR"))
            else:
                st.warning("Enter a job directory.")


def _render_structure_panel():
    """Interactive structure builder: surface, molecule, site finding, adsorbate placement."""
    st.markdown("**Structure Workspace**")
    tab_surf, tab_mol, tab_deprot, tab_sites, tab_ads, tab_cpx = st.tabs([
        "🏗 Build Surface", "🧬 Build Molecule", "⚗️ Deprotonate",
        "🔍 Find Sites", "📌 Place Adsorbate", "🏢 Coordination Compound"
    ])

    with tab_surf:
        st.markdown("##### Build a clean metal surface")
        st.caption("Uses ASE `fcc111`/`bcc110`/`hcp0001` with bottom-layer constraints fixed.")
        col1, col2 = st.columns(2)
        with col1:
            element = st.text_input("Element", value="Ag", key="sb_elem",
                                    help="e.g. Ag, Cu, Pt, Ni, Fe, Ru")
            surf_type = st.selectbox("Surface", ["111","100","110","211","0001","10m10","443"],
                                     key="sb_surf", help="Miller index / surface type")
        with col2:
            col_nx, col_ny, col_nl = st.columns(3)
            with col_nx: nx = st.number_input("Nx", min_value=1, max_value=8, value=4, key="sb_nx")
            with col_ny: ny = st.number_input("Ny", min_value=1, max_value=8, value=4, key="sb_ny")
            with col_nl: nl = st.number_input("Layers", min_value=2, max_value=10, value=3, key="sb_nl",
                                               help="e.g. 3 for (4,4,3)")
            vacuum = st.slider("Vacuum (Å)", 5.0, 25.0, 10.0, 0.5, key="sb_vac")
            fix_bot = st.checkbox("Fix bottom layer", value=True, key="sb_fix")

        st.caption(f"**Preview:** `fcc111('{element}', size=({nx},{ny},{nl}), vacuum={vacuum})`")

        if st.button("Build Surface", type="primary", key="sb_surf_btn"):
            with st.spinner(f"Building {element}({surf_type}) {nx}×{ny}×{nl}..."):
                r = api.structure_build_surface(
                    element=element, surface_type=surf_type,
                    nx=int(nx), ny=int(ny), nlayers=int(nl),
                    vacuum=float(vacuum), fix_bottom=fix_bot,
                )
            if r.get("ok"):
                st.success(f"✅ {r['label']}  —  {r['n_atoms']} atoms")
                st.session_state["structure_panel_poscar"] = r.get("poscar", "")
                st.session_state["structure_panel_viz"] = r.get("viz", {})
                _render_structure_viz(r.get("viz", {}), r.get("plot_png_b64", ""))
                with st.expander("POSCAR (click to expand)", expanded=False):
                    st.code(r.get("poscar", ""), language="text")
            else:
                st.error(r.get("detail") or r.get("error", "Build failed"))

    with tab_mol:
        st.markdown("##### Build gas-phase molecule from PubChem")
        st.caption("Enter a SMILES string or common name. "
                   "Uses `pubchem_atoms_search(smiles='...')` to get 3D geometry.")

        _smiles_presets = {
            "C4H10 (n-butane)":  "CCCC",
            "C4H8 (1-butene)":   "C=CCC",
            "C4H8 (2-butene)":   "CC=CC",
            "C3H8 (propane)":    "CCC",
            "C3H6 (propene)":    "C=CC",
            "CO2":               "O=C=O",
            "H2O":               "O",
            "NH3":               "N",
            "H2":                "[HH]",
            "Acetone":           "CC(=O)C",
            "Ethanol":           "CCO",
            "Custom…":           "",
        }
        preset = st.selectbox("Quick pick", list(_smiles_presets.keys()), key="mol_preset")
        default_smiles = _smiles_presets[preset]
        smiles_in = st.text_input("SMILES string", value=default_smiles, key="mol_smiles",
                                   help="e.g. CCCC for n-butane, C=CCC for 1-butene")
        mol_label = st.text_input("Label (optional)", placeholder="e.g. C4H10", key="mol_label")
        cell_sz   = st.slider("Cell size (Å)", 15.0, 30.0, 20.0, 1.0, key="mol_cell")

        if st.button("Fetch from PubChem", type="primary", key="mol_fetch_btn"):
            if smiles_in.strip():
                with st.spinner(f"Searching PubChem for {smiles_in}..."):
                    r = api.structure_build_molecule(
                        smiles=smiles_in.strip(),
                        label=mol_label.strip(),
                        cell_size=float(cell_sz),
                    )
                if r.get("ok"):
                    st.success(f"✅ {r['formula']}  —  {r['n_atoms']} atoms  |  SMILES: {r['smiles']}")
                    st.session_state["mol_panel_poscar"] = r.get("poscar", "")
                    _render_structure_viz(r.get("viz", {}), r.get("plot_png_b64", ""))
                    with st.expander("POSCAR", expanded=False):
                        st.code(r.get("poscar", ""), language="text")
                else:
                    st.error(r.get("error") or r.get("detail", "PubChem search failed"))
            else:
                st.warning("Enter a SMILES string.")

    with tab_deprot:
        st.markdown("##### Deprotonate molecule (H-removal)")
        st.caption(
            "Generate CO₂RR intermediates by removing H atoms from a parent molecule. "
            "Example: CH₃OH → CH₂OH (−1H) → CHOH (−2H) → CHO (−3H). "
            "Paste the parent molecule POSCAR below."
        )
        deprot_poscar = st.text_area(
            "Parent molecule POSCAR",
            value=st.session_state.get("mol_panel_poscar", ""),
            height=120, key="deprot_poscar_in",
            help="Built from 🧬 Build Molecule tab, or paste manually."
        )
        col_n, col_site = st.columns(2)
        with col_n:
            deprot_n = st.number_input("H atoms to remove", min_value=1, max_value=6,
                                       value=1, step=1, key="deprot_n_remove")
        with col_site:
            deprot_site = st.selectbox(
                "H selection strategy",
                ["terminal", "surface", "random"],
                index=0, key="deprot_site",
                help=(
                    "terminal: O-H / N-H (furthest from backbone) — best for alcohols\n"
                    "surface: closest to H-centroid — best for C-H activation\n"
                    "random: first H in list"
                )
            )
        if st.button("Deprotonate", type="primary", key="deprot_btn"):
            if deprot_poscar.strip():
                with st.spinner(f"Removing {deprot_n} H ({deprot_site})..."):
                    r = api.structure_deprotonate(
                        poscar_content=deprot_poscar.strip(),
                        n_remove=int(deprot_n),
                        site=deprot_site,
                    )
                if r.get("ok"):
                    st.success(
                        f"✅ {r['formula_original']} → **{r['formula_deprotonated']}** "
                        f"(removed {r['n_removed']} H, {r['n_atoms_new']} atoms total)"
                    )
                    st.session_state["deprot_panel_poscar"] = r["poscar"]
                    with st.expander("Deprotonated POSCAR", expanded=True):
                        st.code(r["poscar"], language="text")
                    st.download_button(
                        f"⬇ POSCAR ({r['formula_deprotonated']})",
                        r["poscar"],
                        file_name=f"{r['formula_deprotonated']}.POSCAR",
                        mime="text/plain",
                        key="deprot_download",
                    )
                    # Copy to mol panel so further deprotonate steps chain
                    st.session_state["mol_panel_poscar"] = r["poscar"]
                    if st.button("Use as input for next deprotonate step →",
                                 key="deprot_chain"):
                        st.session_state["deprot_poscar_in"] = r["poscar"]
                        st.rerun()
                else:
                    st.error(r.get("detail", "Deprotonate failed"))
            else:
                st.warning("Paste a POSCAR first.")

    with tab_sites:
        st.markdown("##### Find adsorption sites")
        poscar_in = st.text_area("POSCAR content (paste here or use built slab above)",
                                  value=st.session_state.get("structure_panel_poscar", ""),
                                  height=120, key="site_poscar")
        height_val = st.slider("Site height above surface (Å)", 1.0, 4.0, 2.0, 0.1, key="site_h")
        if st.button("Find Sites", key="site_btn") and poscar_in.strip():
            with st.spinner("Finding adsorption sites..."):
                r = api.structure_find_sites(poscar_in, height=height_val)
            if r.get("ok"):
                st.success(f"Found {r['n_sites']} sites: {', '.join(r['site_types'])}")
                st.session_state["structure_panel_sites"] = r.get("sites", [])
                st.json(r.get("sites", [])[:8])
            else:
                st.error(r.get("error", "Site finding failed"))

    with tab_cpx:
        st.markdown("##### Build coordination compound")
        st.caption("Constructs a metal-center + monodentate ligand complex in a cubic cell.")

        _METALS = ["Cu", "Fe", "Pt", "Ni", "Co", "Rh", "Ru", "Pd", "Au", "Ag",
                   "Ir", "Os", "Re", "Mo", "W", "Cr", "Mn", "V", "Ti", "Zn"]
        _LIGANDS = ["NH3", "H2O", "CO", "Cl", "F", "CN", "NO", "OH", "PH3", "SCN"]
        _GEOMS = ["linear", "trigonal_planar", "tetrahedral", "square_planar",
                  "trigonal_bipyramidal", "octahedral"]

        col_m, col_l = st.columns(2)
        with col_m:
            metal = st.selectbox("Metal", _METALS, key="cpx_metal")
            n_coord = st.selectbox("Coordination number", [2, 3, 4, 5, 6],
                                   index=2, key="cpx_ncoord")
        with col_l:
            ligand = st.selectbox("Ligand", _LIGANDS, key="cpx_ligand")
            geometry = st.selectbox("Geometry", _GEOMS,
                                    index=_GEOMS.index("square_planar"), key="cpx_geom")

        col_b, col_c = st.columns(2)
        with col_b:
            bond_len = st.slider("Bond length (Å)", 1.6, 2.8, 2.0, 0.05, key="cpx_bond")
        with col_c:
            cell_sz = st.slider("Cell size (Å)", 10.0, 25.0, 15.0, 1.0, key="cpx_cell")

        st.caption(f"Preview: `{metal}({ligand})_{n_coord}` — {geometry}")

        if st.button("Build Complex", type="primary", key="cpx_btn"):
            with st.spinner(f"Building {metal}({ligand})_{n_coord} ({geometry})..."):
                r = api.structure_build_complex(
                    metal=metal, ligand=ligand, n_coord=int(n_coord),
                    geometry=geometry, bond_length=float(bond_len),
                    cell_size=float(cell_sz),
                    session_id=st.session_state.session_id,
                )
            if r.get("ok"):
                st.success(
                    f"✅ {r.get('formula','?')}  —  {r.get('n_atoms','?')} atoms  |  "
                    f"{r.get('geometry','')}  bond={r.get('bond_length',bond_len):.2f} Å"
                )
                poscar = r.get("poscar", "")
                st.session_state["cpx_panel_poscar"] = poscar
                # Optionally copy to structure panel so Find Sites can use it
                if poscar:
                    st.session_state["structure_panel_poscar"] = poscar
                _render_structure_viz(r.get("viz", {}), r.get("plot_png_b64", ""))
                with st.expander("POSCAR", expanded=False):
                    st.code(poscar, language="text")
                if poscar:
                    st.download_button(
                        "📥 Download POSCAR",
                        data=poscar,
                        file_name=f"{metal}_{ligand}{n_coord}_{geometry}.POSCAR",
                        key="cpx_dl",
                    )
            else:
                st.error(r.get("error") or r.get("detail", "Build failed"))

    with tab_ads:
        st.markdown("##### Place adsorbate")
        poscar_ads = st.text_area("POSCAR (slab)", key="ads_poscar",
                                   value=st.session_state.get("structure_panel_poscar", ""),
                                   height=120)
        col_a, col_b = st.columns(2)
        with col_a:
            molecule = st.selectbox("Adsorbate", [
                "CO", "H", "O", "OH", "CO2", "CHO", "COOH", "NH3", "N2", "H2O", "OOH",
                "NO", "CH", "CH2", "CH3", "NH", "NH2", "S", "N", "C",
            ], key="ads_mol")
            n_configs = st.slider("Configurations", 1, 6, 4, key="ads_n")
        with col_b:
            site_idx = st.number_input("Site index (single placement)", min_value=0, value=0, key="ads_site")
            ads_height = st.slider("Height (Å)", 1.5, 3.5, 2.0, 0.1, key="ads_h")
        col_p, col_g = st.columns(2)
        with col_p:
            if st.button("Place at site", key="ads_place_btn") and poscar_ads.strip():
                with st.spinner("Placing adsorbate..."):
                    r = api.structure_place_adsorbate(
                        poscar_ads, molecule, site_index=int(site_idx), height=float(ads_height)
                    )
                if r.get("ok"):
                    st.success(f"{molecule} placed at {r['site_type']} site — {r['n_atoms']} atoms")
                    st.session_state["structure_panel_poscar"] = r.get("poscar", "")
                    if r.get("viz"):
                        _render_structure_viz(r["viz"])
                else:
                    st.error(r.get("error", "Placement failed"))
        with col_g:
            if st.button("Generate all configs", key="ads_gen_btn") and poscar_ads.strip():
                with st.spinner("Generating configurations..."):
                    r = api.structure_generate_configs(poscar_ads, molecule, max_configs=int(n_configs))
                if r.get("ok"):
                    st.success(f"Generated {r['n_configs']} configurations")
                    for cfg in r.get("configs", []):
                        with st.expander(f"Config {cfg['config_id']}: {cfg['site_type']} site"):
                            st.text(f"Formula: {cfg['formula']}  |  Atoms: {cfg['n_atoms']}")
                            if cfg.get("viz"):
                                _render_structure_viz(cfg["viz"])
                else:
                    st.error(r.get("error", "Config generation failed"))


def _render_htp_panel():
    """HTP NNP Dataset Generation — generate, monitor, and export training data."""
    st.markdown("**HTP NNP Dataset**")
    st.caption(
        "Generate diverse structures for neural-network-potential (NNP) training. "
        "Structures are stored in ASE database and mirrored to PostgreSQL."
    )

    tab_gen, tab_stats, tab_export = st.tabs(["⚙️ Generate", "📊 Stats", "📤 Export"])

    with tab_gen:
        st.markdown("##### Strategy")
        _STRATEGIES = {
            "rattle":             "Random atomic displacements (Gaussian noise, stdev Å)",
            "strain":             "Uniform volumetric strain sweep",
            "rattle_strain":      "Combined rattle + strain (most diverse)",
            "surface_rattle":     "Rattle only surface layers; bulk fixed",
            "temperature_rattle": "Boltzmann-weighted displacements via Einstein model",
            "alloy":              "Random binary-alloy occupancy configurations",
            "vacancy":            "Random vacancy structures",
        }
        strategy = st.selectbox(
            "Strategy",
            list(_STRATEGIES.keys()),
            key="htp_strategy",
            format_func=lambda k: f"{k}  —  {_STRATEGIES[k][:55]}…",
        )

        col_n, col_db = st.columns(2)
        with col_n:
            n_total = st.number_input("Total structures", min_value=10, max_value=50000,
                                      value=200, step=50, key="htp_n")
        with col_db:
            db_path = st.text_input("DB path", value="htp_dataset.db", key="htp_db")

        # Strategy-specific kwargs
        with st.expander("Strategy options", expanded=False):
            if strategy in ("rattle", "surface_rattle"):
                st.slider("Stdev (Å)", 0.01, 0.3, 0.1, 0.01, key="htp_stdev")
            if strategy == "temperature_rattle":
                st.number_input("Temperature (K)", 100, 3000, 500, 100, key="htp_T")
            if strategy in ("alloy",):
                st.text_input("Host element", value="Cu", key="htp_host")
                st.text_input("Dopant element", value="Ag", key="htp_dopant")
            if strategy == "vacancy":
                st.number_input("N vacancies", 1, 5, 1, key="htp_nvac")

        st.markdown("##### Base structures")
        st.caption("Provide one or more POSCAR files as base structures.")

        n_bases = st.number_input("Number of base structures", 1, 5, 1, key="htp_nbases")
        base_structures = []
        for i in range(int(n_bases)):
            with st.expander(f"Base structure {i+1}", expanded=(i == 0)):
                # Try to pre-fill from session structure panel
                default_poscar = st.session_state.get("structure_panel_poscar", "")
                poscar_txt = st.text_area(
                    "POSCAR",
                    value=default_poscar if i == 0 else "",
                    height=100,
                    key=f"htp_base_{i}",
                    help="Paste POSCAR content, or build a surface first in Structure panel",
                )
                lbl = st.text_input("Label", value=f"base_{i+1}", key=f"htp_lbl_{i}")
                if poscar_txt.strip():
                    base_structures.append({"poscar": poscar_txt.strip(), "label": lbl})

        # HPC INCAR settings for VASP single-points
        st.markdown("##### VASP settings (NNP single-points)")
        col_e, col_k = st.columns(2)
        with col_e:
            encut = st.number_input("ENCUT (eV)", 300, 600, 450, 50, key="htp_encut",
                                    help="Keep consistent across ALL structures in the dataset")
        with col_k:
            kpoints = st.text_input("KPOINTS", value="4 4 1", key="htp_kpoints")

        if st.button("Generate Dataset", type="primary", use_container_width=True,
                     key="htp_gen_btn"):
            if not base_structures:
                st.warning("Add at least one base structure.")
            else:
                # Build kwargs from strategy options
                kwargs: dict = {"encut": encut, "kpoints": kpoints}
                if strategy in ("rattle", "surface_rattle"):
                    kwargs["stdev"] = st.session_state.get("htp_stdev", 0.1)
                if strategy == "temperature_rattle":
                    kwargs["T"] = st.session_state.get("htp_T", 500)
                if strategy == "alloy":
                    kwargs["host"]   = st.session_state.get("htp_host", "Cu")
                    kwargs["dopant"] = st.session_state.get("htp_dopant", "Ag")
                if strategy == "vacancy":
                    kwargs["n_vacancies"] = int(st.session_state.get("htp_nvac", 1))

                with st.spinner(f"Generating {n_total} structures via {strategy}..."):
                    r = api.htp_generate(
                        base_structures=base_structures,
                        strategy=strategy,
                        n_total=int(n_total),
                        db_path=db_path,
                        **kwargs,
                    )
                if r.get("ok"):
                    stats = r.get("stats", {})
                    pg_id = r.get("pg_run_id")
                    st.success(
                        f"✅ Generated **{r['n_generated']}** structures  |  "
                        f"DB: `{r['db_path']}`"
                        + (f"  |  PostgreSQL run_id={pg_id}" if pg_id else "")
                    )
                    st.session_state["htp_db_path"] = db_path
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", stats.get("total", r["n_generated"]))
                    col2.metric("Pending", stats.get("pending", r["n_generated"]))
                    col3.metric("Done", stats.get("done", 0))

                    # Show batch script
                    with st.spinner("Generating batch job script..."):
                        sr = api.htp_script(
                            db_path=db_path, encut=encut,
                            kpoints=kpoints, batch_size=50,
                            scheduler=st.session_state.get("cluster_config", {}).get("scheduler", "sge"),
                        )
                    if sr.get("ok"):
                        with st.expander("📝 Batch job script (download & submit)", expanded=True):
                            st.code(sr.get("script", ""), language="bash")
                            st.download_button(
                                "📥 Download job script",
                                data=sr.get("script", ""),
                                file_name="htp_batch.sh",
                                mime="text/plain",
                                key="htp_dl_script",
                            )
                else:
                    st.error(r.get("error") or r.get("detail", "Generation failed"))

    with tab_stats:
        st.markdown("##### Dataset statistics")
        db_path_s = st.text_input("DB path", key="htp_stats_db",
                                   value=st.session_state.get("htp_db_path", "htp_dataset.db"))
        if st.button("Refresh Stats", use_container_width=True, key="htp_stats_btn"):
            with st.spinner("Loading stats..."):
                r = api.htp_stats(db_path=db_path_s)
            if r.get("ok") or "total" in r:
                stats = r if "total" in r else r.get("stats", r)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total",   stats.get("total", 0))
                col2.metric("Pending", stats.get("pending", 0))
                col3.metric("Done",    stats.get("done", 0))
                col4.metric("Failed",  stats.get("failed", 0))
                n_done = stats.get("done", 0)
                n_total_s = max(stats.get("total", 1), 1)
                st.progress(n_done / n_total_s, text=f"{n_done}/{n_total_s} completed")

                if stats.get("strategies"):
                    st.markdown("**By strategy:**")
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(
                            [{"strategy": k, "count": v}
                             for k, v in stats["strategies"].items()]
                        ),
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.error(r.get("detail") or r.get("error", "Stats unavailable"))

    with tab_export:
        st.markdown("##### Export training dataset")
        st.caption(
            "Exports completed structures (energy + forces + stress) as extXYZ "
            "ready for MACE / NequIP / DeePMD-kit training."
        )
        db_path_e  = st.text_input("DB path", key="htp_exp_db",
                                    value=st.session_state.get("htp_db_path", "htp_dataset.db"))
        out_path   = st.text_input("Output .xyz file", value="training.xyz", key="htp_out_xyz")
        only_done  = st.checkbox("Only completed structures", value=True, key="htp_only_done")

        if st.button("Export extXYZ", type="primary", use_container_width=True,
                     key="htp_export_btn"):
            with st.spinner("Exporting..."):
                r = api.htp_export(db_path=db_path_e, output_path=out_path,
                                   only_done=only_done)
            if r.get("ok"):
                st.success(
                    f"✅ Exported **{r.get('n_exported', '?')}** structures "
                    f"→ `{r.get('output_path', out_path)}`"
                )
                xyz_content = r.get("xyz_content", "")
                if xyz_content:
                    st.download_button(
                        "📥 Download training.xyz",
                        data=xyz_content,
                        file_name=out_path,
                        mime="text/plain",
                        key="htp_dl_xyz",
                    )
                else:
                    st.info(f"File saved to server at `{r.get('output_path', out_path)}`")
            else:
                st.error(r.get("error") or r.get("detail", "Export failed"))


def _render_structure_viz(viz: dict, plot_png_b64: str = ""):
    """Render structure: matplotlib PNG if available, else atom summary table."""
    if not viz and not plot_png_b64:
        return

    # Show matplotlib PNG (preferred — actual 3D-like view)
    if plot_png_b64:
        import base64
        st.image(f"data:image/png;base64,{plot_png_b64}",
                 caption=viz.get("formula", "") if viz else "",
                 use_container_width=True)
        return

    # Fallback: text summary
    if not viz:
        return
    n = viz.get("n_atoms", 0)
    formula = viz.get("formula", "")
    cell = viz.get("cell", [])
    atoms = viz.get("atoms", [])
    from collections import Counter
    counts = Counter(a["symbol"] for a in atoms)
    st.caption(f"**{formula}** — {n} atoms: "
               f"{', '.join(f'{s}:{c}' for s, c in sorted(counts.items()))}")
    if cell:
        a_len = (sum(x**2 for x in cell[0]))**0.5
        b_len = (sum(x**2 for x in cell[1]))**0.5
        c_len = (sum(x**2 for x in cell[2]))**0.5
        st.caption(f"Cell: a={a_len:.2f} Å, b={b_len:.2f} Å, c={c_len:.2f} Å")


def _render_settings_panel():
    st.markdown("**Settings**")

    tab_gen, tab_prof, tab_hpc_cfg = st.tabs(["⚙️ General", "📋 Calc Profiles", "🖥 HPC"])

    with tab_gen:
        st.text_input("Backend URL", value=api.BASE, key="cfg_backend",
                      help="Restart app to apply changes")
        st.number_input("Request timeout (s)", value=api.TIMEOUT, key="cfg_timeout", min_value=30)
        st.selectbox("Default LLM", ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6"], key="cfg_model")
        st.toggle("Dry-run mode (no HPC submission)", key="cfg_dry_run")

    with tab_prof:
        st.markdown("##### VASP Calc Profiles")
        st.caption(
            "Named VASP parameter sets from `calc_profiles.yaml`. "
            "Reference these in the chat or select to preview INCAR values."
        )
        with st.spinner("Loading profiles..."):
            pr = api.list_calc_profiles()
        profile_names = pr.get("profiles", [])
        if not profile_names:
            st.warning("Could not load profiles — is the backend running?")
        else:
            selected_profile = st.selectbox(
                "Select profile", profile_names, key="cfg_profile_sel"
            )
            if st.button("View Profile", use_container_width=True, key="cfg_view_profile"):
                with st.spinner(f"Loading {selected_profile}..."):
                    pr_detail = api.get_calc_profile(selected_profile)
                params = pr_detail.get("params", {})
                if params:
                    # Show as table
                    import pandas as pd
                    rows = [{"Parameter": k, "Value": str(v)} for k, v in sorted(params.items())]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    # Render as INCAR text
                    incar_lines = ["# INCAR — profile: " + selected_profile, ""]
                    for k, v in sorted(params.items()):
                        if isinstance(v, bool):
                            val_str = ".TRUE." if v else ".FALSE."
                        elif isinstance(v, float) and abs(v) < 1e-3:
                            val_str = f"{v:.1e}"
                        else:
                            val_str = str(v)
                        incar_lines.append(f"  {k:<12} = {val_str}")
                    incar_text = "\n".join(incar_lines)
                    with st.expander("INCAR text", expanded=False):
                        st.code(incar_text, language="text")
                    st.download_button(
                        "📥 Download INCAR",
                        data=incar_text,
                        file_name=f"INCAR_{selected_profile}",
                        mime="text/plain",
                        key="dl_incar_profile",
                    )
                else:
                    st.info("Profile returned no parameters.")

            # Quick reference table for all profiles
            with st.expander("All profiles (quick reference)", expanded=False):
                _PROFILE_DESCRIPTIONS = {
                    "fast_screening":   "Coarse single-point ~40% faster. Use for large screens.",
                    "standard":         "Default production single-point.",
                    "high_accuracy":    "Tight convergence for reference energies.",
                    "relax_bulk":       "Ionic + cell relaxation (ISIF=3).",
                    "relax_slab":       "Ionic relaxation, cell fixed (ISIF=2).",
                    "neb":              "Nudged elastic band (no climbing).",
                    "cineb":            "Climbing-image NEB for accurate TS energy.",
                    "dimer":            "Dimer method for TS search.",
                    "gcdft":            "Grand-canonical DFT with VASPsol.",
                    "nnp_singlepoint":  "NNP training: ISIF=2 stress, ENCUT=450, PREC=Accurate.",
                    "electronic_scf":   "SCF prerequisite — writes WAVECAR+CHGCAR.",
                    "dos":              "DOS (ICHARG=11, ISMEAR=-5, LORBIT=11).",
                    "pdos":             "Projected DOS — same as dos.",
                    "band":             "Band structure (line-mode KPOINTS).",
                    "elf":              "ELF — NCORE=1 mandatory!",
                    "bader":            "Bader charge — LAECHG=True, LREAL=False.",
                    "cdd":              "Charge density difference (run 3× for AB/A/B).",
                    "work_function":    "Work function — LVHAR=True, LDIPOL=True, IDIPOL=3.",
                    "cohp":             "COHP/LOBSTER — ISYM=-1 mandatory!",
                }
                import pandas as pd
                rows = [{"Profile": n, "Description": _PROFILE_DESCRIPTIONS.get(n, "")}
                        for n in profile_names]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_hpc_cfg:
        st.markdown("**HPC Cluster**")
        cc = st.session_state.get("cluster_config", {})

        host     = st.text_input("Hostname", value=cc.get("host", ""), key="cc_host")
        user     = st.text_input("Username", value=cc.get("user", ""), key="cc_user")
        sched    = st.selectbox("Scheduler", ["sge", "slurm", "pbs"],
                                index=["sge","slurm","pbs"].index(cc.get("scheduler","sge")),
                                key="cc_sched")
        account  = st.text_input("Account / Queue / Partition", value=cc.get("account", ""), key="cc_account")
        walltime = st.text_input("Walltime", value=cc.get("walltime", "24:00:00"), key="cc_walltime")
        ntasks   = st.number_input("CPU cores", min_value=1, max_value=256,
                                   value=int(cc.get("ntasks", 32)), key="cc_ntasks")
        rbase    = st.text_input("Remote base dir", value=cc.get("remote_base", ""), key="cc_rbase")

        _env_placeholder = (
            "# Paste your cluster environment setup here, e.g.:\n"
            "source /etc/profile.d/modules.sh\n"
            "module load intel/2021 vasp/5.4.4\n"
            "export VASP_PP_PATH=$HOME/vasp/pps\n"
            "export VASP_COMMAND='mpirun -np ${NSLOTS} vasp_std'\n"
            "# For SLURM: use $SLURM_NTASKS instead of $NSLOTS\n"
            "# export VASP_COMMAND='srun vasp_std'"
        )
        env_setup = st.text_area(
            "Environment setup (module loads, conda, VASP cmd)",
            value=cc.get("env_setup", ""),
            height=160,
            key="cc_env_setup",
            placeholder=_env_placeholder,
            help="These lines go verbatim into job.sh between the scheduler headers and `python script.py`",
        )

        if st.button("Save Cluster Config", use_container_width=True, key="save_cluster"):
            st.session_state["cluster_config"] = {
                "host": host.strip(),
                "user": user.strip(),
                "scheduler": sched,
                "account": account.strip(),
                "walltime": walltime.strip(),
                "ntasks": int(ntasks),
                "remote_base": rbase.strip(),
                "env_setup": env_setup,
            }
            st.success(f"Cluster config saved: {user}@{host}")

        st.divider()
        st.markdown("**File Naming Convention**")
        st.markdown("""
```
<remote_base>/
├── 01_surface/    Ag_fcc111_4x4x3/   POSCAR, ase-geo.py, job.sh
├── 02_molecules/  C4H10_gas/         POSCAR, ase-geo.py, ase-freq.py
├── 03_adsorption/ Ag111_C4H10_top/   POSCAR, ase-geo.py, ase-freq.py
├── 04_neb/        Ag111_dehydrog/    POSCAR (IS), ase-neb.py
└── 05_gcdft/      Ag111_Um0.8V/      POSCAR, ase-gcdft.py
```
Bundle download: POSCAR + scripts + job.sh per task.
""")


def _render_backend_status():
    try:
        import requests
        r = requests.get(f"{api.BASE}/", timeout=2)
        ok = r.status_code < 400
    except Exception:
        ok = False
    icon = "🟢" if ok else "🔴"
    st.caption(f"{icon} Backend: {api.BASE}")


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_message(msg: dict):
    role = msg["role"]
    kind = msg.get("kind", "text")
    data = msg.get("data")

    with st.chat_message(role, avatar="⚛️" if role == "assistant" else "👤"):
        if kind == "text":
            st.markdown(msg["content"])

        elif kind == "pipeline":
            _render_pipeline_result(data)

        elif kind == "analysis":
            _render_analysis_result(data)

        elif kind == "free_energy":
            _render_free_energy_result(data)

        elif kind == "microkinetics":
            _render_microkinetics_result(data)

        elif kind == "qa_functional":
            st.markdown(msg["content"])
            if data:
                recs = data.get("functional_recommendations", [])
                for r in recs:
                    with st.expander(f"🔬 {r.get('topic','').replace('_',' ').title()}"):
                        st.caption(r.get("issue",""))
                        for rec in r.get("recommendations", []):
                            st.markdown(f"- {rec}")

        elif kind == "qa_debug":
            st.markdown(msg["content"])
            if data:
                patch = data.get("incar_patch", {})
                if patch:
                    with st.expander("📋 Suggested INCAR patch"):
                        lines = [f"{k} = {v}" for k, v in patch.items()]
                        st.code("\n".join(lines), language="fortran")
                issues = data.get("issues", [])
                if issues:
                    with st.expander(f"⚠️ {len(issues)} issue(s) detected"):
                        for iss in issues:
                            icon = "🔴" if iss["severity"] == "critical" else "🟡"
                            st.markdown(f"{icon} **{iss['code']}**: {iss['description']}")

        elif kind == "error":
            st.error(msg["content"])


def _render_pipeline_result(data: dict):
    """Render the intent → hypothesis → plan pipeline result."""
    if not data:
        st.warning("No pipeline data.")
        return

    intent    = data.get("intent", {})
    hypothesis = data.get("hypothesis", {})
    plan       = data.get("plan", {})

    # ── Intent ────────────────────────────────────────────────────────────
    with st.expander("🎯 Intent", expanded=True):
        if not intent:
            st.caption("Intent not available.")
        else:
            # Structured fields as pills
            pills_html = ""
            domain = intent.get("domain") or intent.get("area") or intent.get("intent_area", "")
            stage  = intent.get("stage") or intent.get("intent_stage", "")
            conf   = intent.get("confidence", 0)
            task   = intent.get("task") or intent.get("problem_type", "")
            subst  = intent.get("substrate") or (intent.get("system") or {}).get("material", "")

            if domain: pills_html += _pill(domain, "blue")
            if stage:  pills_html += _pill(stage, "purple")
            if subst:  pills_html += _pill(subst, "green")
            if conf:   pills_html += _pill(f"conf: {conf:.0%}", _conf_color(conf))

            st.markdown(pills_html, unsafe_allow_html=True)
            if task:
                st.markdown(f"**Task:** {task}")

            # Reaction network summary
            rxn = intent.get("reaction_network", {})
            intermediates = rxn.get("intermediates", []) if isinstance(rxn, dict) else []
            if intermediates:
                st.caption("Intermediates: " + " → ".join(intermediates[:8]))

    # ── Hypothesis ────────────────────────────────────────────────────────
    with st.expander("🧪 Mechanism Hypothesis", expanded=True):
        if not hypothesis:
            st.caption("Hypothesis not available.")
        else:
            md = hypothesis.get("md") or hypothesis.get("hypothesis_md") or ""
            if isinstance(md, str) and md.strip():
                st.markdown(md)
            else:
                st.caption("_(Hypothesis markdown empty — check hypothesis agent)_")

            steps = hypothesis.get("steps", [])
            if steps:
                st.markdown("**Elementary steps:**")
                for step in steps[:12]:
                    st.code(step, language=None)

            # Show reaction graph if available
            graph = hypothesis.get("graph", {})
            ts_edges = graph.get("ts_edges", [])
            inter = graph.get("intermediates", [])
            if ts_edges:
                st.markdown("**TS edges:**")
                for ts in ts_edges[:6]:
                    st.code(ts, language=None)
            if inter:
                st.caption("**Intermediates:** " + "  ·  ".join(inter[:12]))

    # ── Plan ──────────────────────────────────────────────────────────────
    with st.expander("📋 Calculation Plan", expanded=True):
        tasks = plan.get("tasks", [])
        if not tasks:
            st.caption("No tasks generated.")
        else:
            conf = plan.get("confidence", 0)
            if conf:
                st.caption(f"Plan confidence: {conf:.0%}")

            # Group by section
            sections: Dict[str, List] = {}
            for t in tasks:
                sec = t.get("section", "General")
                sections.setdefault(sec, []).append(t)

            for sec_name, sec_tasks in sections.items():
                st.markdown(f"**{sec_name}**")
                for t in sec_tasks:
                    _render_task_card(t)

    # ── Next suggestions from plan ────────────────────────────────────────
    suggestions = plan.get("suggestions", [])
    if suggestions and isinstance(suggestions, list) and isinstance(suggestions[0], dict):
        with st.expander("💡 Suggestions", expanded=False):
            for s in suggestions[:5]:
                desc = s.get("description") or s.get("action", "")
                pri  = s.get("priority", "")
                color_map = {"critical": "red", "high": "amber", "medium": "blue", "low": "purple"}
                pill = _pill(pri, color_map.get(pri, "blue")) if pri else ""
                st.markdown(f"{pill} {desc}", unsafe_allow_html=True)


def _infer_calc_type(task: dict) -> str:
    """Determine ASE calc script type from task metadata."""
    section = task.get("section", "").lower()
    name    = task.get("name", "").lower()
    agent   = task.get("agent", "").lower()
    if "neb" in section or "neb" in agent or "neb" in name:
        return "neb"
    if "gc-dft" in section or "gcdft" in section or "gc_dft" in agent:
        return "gcdft"
    if "dos" in agent or "dos" in name:
        return "dos"
    if "band" in name:
        return "band"
    if "elf" in name:
        return "elf"
    if "freq" in name or "freq" in agent:
        return "freq"
    if "molecule" in agent or "molecule" in name:
        return "molecule"
    return "geo"


def _infer_system(task: dict) -> dict:
    """Extract system parameters from task payload."""
    payload = (task.get("params") or {}).get("payload", {})
    return {
        "element":      payload.get("element", ""),
        "surface_type": payload.get("surface_type") or payload.get("facet", "111"),
        "nx": payload.get("nx", 4),
        "ny": payload.get("ny", 4),
        "nlayers": payload.get("nlayers", payload.get("layers", 3)),
        "vacuum": payload.get("vacuum", 10.0),
        "fix_bottom": payload.get("fix_bottom", True),
        "formula": payload.get("formula", ""),
        "smiles": payload.get("smiles", ""),
        "label": payload.get("label", ""),
    }


def _make_job_name(task: dict) -> str:
    """
    Generate a well-organized directory name for a task following the convention:
      Surface  : Ag_fcc111_4x4x3
      Molecule : C4H10_gas
      Ads      : Ag111_C4H10_top
      Co-ads   : Ag111_C4H9+H
      NEB      : Ag111_C4H10s-C4H9s+Hs
      GC-DFT   : Ag111_U-0.8V
    """
    import re as _re
    payload  = (task.get("params") or {}).get("payload", {})
    section  = task.get("section", "").lower()
    name     = task.get("name", "")

    elem    = payload.get("element", "")
    facet   = payload.get("surface_type") or payload.get("facet", "111")
    nx, ny, nl = payload.get("nx", 4), payload.get("ny", 4), payload.get("nlayers", 3)
    formula = payload.get("formula") or payload.get("label") or payload.get("molecule", "")
    ads     = payload.get("adsorbate", "")
    site    = payload.get("site_type", "")
    U       = payload.get("potential_V", "")

    if "1. surface" in section or "build" in name.lower():
        cs = ""  # crystal system prefix omitted for brevity; Ag111 is already unambiguous
        job = f"{elem}{facet}_{nx}x{ny}x{nl}" if elem else name
    elif "2. molecule" in section or "molecule" in name.lower() or "gas" in name.lower():
        job = f"{formula}_gas" if formula else name
    elif "5. gc-dft" in section or "gcdft" in section or "gc-dft" in name.lower():
        u_str = f"U{float(U):+.1f}V".replace("+", "p").replace("-", "m") if U != "" else "Useep"
        job = f"{elem}{facet}_{u_str}" if elem else name
    elif "4. neb" in section or "neb" in name.lower():
        # IS→FS from task name or formula
        label = _re.sub(r"[^\w\+\-\>]", "", name.replace(" ", "_").replace("→", "-"))
        job = f"{elem}{facet}_{label}" if elem else label
    elif "3. adsorption" in section or "adsorb" in name.lower():
        site_tag = f"_{site}" if site else ""
        job = f"{elem}{facet}_{formula or ads}{site_tag}" if elem else name
    else:
        # Fallback: sanitise task name
        job = _re.sub(r"[^\w\-]", "_", name).strip("_") or "task"

    return job


_SECTION_PREFIX = {
    "1. surface":    "01_surface",
    "2. molecules":  "02_molecules",
    "3. adsorption": "03_adsorption",
    "4. neb":        "04_neb",
    "5. gc-dft":     "05_gcdft",
}


def _section_dir(task: dict) -> str:
    sec = task.get("section", "").lower()
    for key, prefix in _SECTION_PREFIX.items():
        if key in sec:
            return prefix
    return "00_misc"


def _make_zip_bundle(poscar: str, script: str, script_fn: str,
                     job_sh: str = "", outcar: str = "",
                     contcar: str = "", stdout: str = "") -> bytes:
    """Pack provided files into an in-memory ZIP archive."""
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if poscar:    zf.writestr("POSCAR",    poscar)
        if script:    zf.writestr(script_fn,   script)
        if job_sh:    zf.writestr("job.sh",    job_sh)
        if outcar:    zf.writestr("OUTCAR",    outcar)
        if contcar:   zf.writestr("CONTCAR",   contcar)
        if stdout:    zf.writestr("stdout",    stdout)
    buf.seek(0)
    return buf.read()


def _render_job_sh(job_name: str, script_fn: str, ntasks: int = 32,
                   walltime: str = "24:00:00", scheduler: str = "sge",
                   account: str = "", env_setup: str = "") -> str:
    """Generate job.sh using the user-supplied env_setup block (module loads, conda, VASP cmd)."""
    env_block = (env_setup.strip() + "\n") if env_setup.strip() else ""
    if scheduler == "sge":
        account_line = f"#$ -A {account}\n" if account else ""
        return (
            f"#!/bin/bash -f\n"
            f"#$ -cwd\n"
            f"#$ -o $JOB_ID.log\n"
            f"#$ -e $JOB_ID.err\n"
            f"#$ -pe dc* {ntasks}\n"
            f"#$ -l h_data=4G,h_vmem=16G,h_rt={walltime}\n"
            f"{account_line}"
            f"\n"
            f"{env_block}"
            f"\n"
            f"python {script_fn}\n"
            f'echo "run complete on `hostname`: `date` `pwd`" >> ~/job.log\n'
        )
    elif scheduler == "pbs":
        account_line = f"#PBS -A {account}\n" if account else ""
        return (
            f"#!/bin/bash\n"
            f"#PBS -N {job_name}\n"
            f"#PBS -l nodes=1:ppn={ntasks}\n"
            f"#PBS -l walltime={walltime}\n"
            f"#PBS -j oe\n"
            f"{account_line}"
            f"\n"
            f"cd $PBS_O_WORKDIR\n"
            f"{env_block}"
            f"\n"
            f"python {script_fn}\n"
            f'echo "run complete on `hostname`: `date` `pwd`" >> ~/job.log\n'
        )
    else:  # slurm
        account_line = f"#SBATCH --account={account}\n" if account else ""
        return (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --ntasks={ntasks}\n"
            f"#SBATCH --time={walltime}\n"
            f"#SBATCH --output=%j.log\n"
            f"#SBATCH --error=%j.err\n"
            f"{account_line}"
            f"\n"
            f"{env_block}"
            f"\n"
            f"python {script_fn}\n"
            f'echo "run complete on $(hostname): $(date) $(pwd)" >> ~/job.log\n'
        )


def _has_build_endpoint(task: dict) -> bool:
    """True if the task has an interactive structure-build endpoint."""
    endpoint = (task.get("params") or {}).get("endpoint", "")
    agent    = task.get("agent", "")
    build_agents = {"structure.build_surface", "structure.build_molecule"}
    build_endpoints = {"/agent/structure/build_surface", "/agent/structure/build_molecule",
                       "/agent/structure/generate_configs", "/agent/structure/build_slab"}
    return endpoint in build_endpoints or agent in build_agents


def _render_task_card(task: dict):
    tid      = task.get("id", 0)
    name     = task.get("name", f"Task {tid}")
    agent    = task.get("agent", "")
    desc     = task.get("description", "")
    sid      = st.session_state.session_id
    job_name = _make_job_name(task)
    sec_dir  = _section_dir(task)

    # Task metadata set by plan_agent
    calc_types    = task.get("calc_types") or [_infer_calc_type(task)]
    retrieve_files = task.get("retrieve_files") or ["CONTCAR", "OUTCAR", "stdout", "OSZICAR"]

    # Per-task session-state keys
    key_poscar    = f"task_{tid}_poscar"
    key_viz       = f"task_{tid}_viz"
    key_plot      = f"task_{tid}_plot"
    key_scripts   = f"task_{tid}_scripts"   # dict: {filename: content}
    key_job_id    = f"task_{tid}_job_id"
    key_remote    = f"task_{tid}_remote"
    key_results   = f"task_{tid}_results"   # dict: {filename: content}

    system = _infer_system(task)
    cc     = st.session_state.get("cluster_config", {})
    ntasks    = int(cc.get("ntasks", 32))
    walltime  = cc.get("walltime", "24:00:00")
    sched     = cc.get("scheduler", "sge")
    account   = cc.get("account", "")
    env_setup = cc.get("env_setup", "")
    remote_base = cc.get("remote_base", "")
    remote_path = f"{remote_base}/{sec_dir}/{job_name}" if remote_base else ""
    host_str = f"{cc.get('user','')}@{cc.get('host','')}" if cc.get("user") else cc.get("host","?")

    # Completion flags
    has_poscar  = bool(st.session_state.get(key_poscar))
    has_scripts = bool(st.session_state.get(key_scripts))
    has_job_id  = bool(st.session_state.get(key_job_id))
    has_results = bool(st.session_state.get(key_results))

    with st.container(border=True):
        # Header row with progress indicators
        col_h, col_badges = st.columns([6, 4])
        with col_h:
            st.markdown(f"**{name}**")
            if desc:
                st.caption(desc)
        with col_badges:
            badges = (
                ("🏗", "green" if has_poscar  else "grey") + ("  ",) +
                ("📝", "green" if has_scripts else "grey") + ("  ",) +
                ("🚀", "green" if has_job_id  else "grey") + ("  ",) +
                ("📥", "green" if has_results else "grey")
            )
            st.caption(
                f"{'✅' if has_poscar  else '⬜'} Build  "
                f"{'✅' if has_scripts else '⬜'} Params  "
                f"{'✅' if has_job_id  else '⬜'} Submit  "
                f"{'✅' if has_results else '⬜'} Retrieve"
            )
            st.caption(f"📁 `{sec_dir}/{job_name}/`")

        # ─── Step 1: Generate Structure ───────────────────────────────────
        with st.expander("🏗 Step 1 — Generate Structure", expanded=not has_poscar):
            endpoint_str = (task.get("params") or {}).get("endpoint", "")
            is_ads_task = endpoint_str == "/agent/structure/generate_configs"
            species = task.get("species", "")

            if is_ads_task:
                # ── Adsorption task: show surface + molecule sources ──
                surf_poscar  = st.session_state.get("task_1_poscar", "")
                # Find molecule POSCAR from plan tasks
                mol_poscar_val = ""
                plan_tasks = (st.session_state.get("last_plan") or {}).get("tasks", [])
                mol_task_name = ""
                for pt in plan_tasks:
                    pt_payload = (pt.get("params") or {}).get("payload", {})
                    if pt_payload.get("label","").lower() == species.lower():
                        mol_poscar_val = st.session_state.get(f"task_{pt['id']}_poscar", "")
                        mol_task_name = pt.get("name","")
                        break

                col_s, col_m = st.columns(2)
                with col_s:
                    surf_ok = bool(surf_poscar)
                    st.markdown(f"{'✅' if surf_ok else '⬜'} **Surface** (task 1.1)")
                    if surf_ok:
                        st.caption(f"{len(surf_poscar.splitlines())} lines POSCAR ready")
                    else:
                        st.caption("Complete surface task Step 1 first")
                with col_m:
                    mol_ok = bool(mol_poscar_val)
                    st.markdown(f"{'✅' if mol_ok else '⬜'} **Molecule** — {species}")
                    if mol_ok:
                        st.caption(f"From: {mol_task_name}")
                    else:
                        st.caption(f"Run gas-phase task for {species} first (Section 2)")
                        # Allow manual paste fallback
                        pasted_mol = st.text_area(
                            f"Paste {species} POSCAR manually",
                            height=60, key=f"mol_paste_{tid}"
                        )
                        if pasted_mol.strip():
                            mol_poscar_val = pasted_mol.strip()

                st.divider()
                can_generate = bool(surf_poscar)
                if not can_generate:
                    st.info("Complete the surface build (Section 1) first.")
                else:
                    n_configs = st.slider("Configurations to generate", 1, 6, 4, key=f"ncfg_{tid}")
                    height_val = st.number_input("Adsorbate height (Å)", 1.5, 4.0, 2.0, 0.1, key=f"ht_{tid}")

                    if st.button("▶ Generate Adsorption Configs", key=f"s1_{tid}", type="primary"):
                        if not sid:
                            st.warning("Open a session first.")
                        else:
                            # Inject mol_poscar into task payload for _execute_task
                            task_mod = dict(task)
                            task_mod["params"] = dict(task.get("params") or {})
                            task_mod["params"]["payload"] = dict(
                                task_mod["params"].get("payload", {}),
                                max_configs=n_configs,
                                height=height_val,
                            )
                            if mol_poscar_val:
                                task_mod["params"]["payload"]["mol_poscar"] = mol_poscar_val
                            with st.spinner(f"Generating {n_configs} configs for {species}..."):
                                r = _execute_task(sid, task_mod)
                            if r.get("ok"):
                                configs = r.get("configs", [])
                                # Store all configs in session state
                                st.session_state[f"{key_poscar}_all_configs"] = configs
                                # Default: select config 0
                                if configs:
                                    st.session_state[key_poscar] = configs[0]["poscar"]
                                    st.session_state[key_plot]   = configs[0].get("plot_png_b64","")
                                st.success(f"✅ Generated {len(configs)} configurations")
                                # Persist to DB
                                _save_task_state(sid, tid,
                                    task_name=name,
                                    poscar=st.session_state.get(key_poscar,""),
                                    plot_png_b64=st.session_state.get(key_plot,""),
                                    all_configs=configs,
                                    selected_config=0,
                                )
                            else:
                                st.error(r.get("detail") or r.get("error","Failed"))

                # Show config picker grid
                all_configs = st.session_state.get(f"{key_poscar}_all_configs", [])
                if all_configs:
                    st.markdown("**Select configuration:**")
                    selected_idx = st.session_state.get(f"{key_poscar}_selected", 0)
                    cols = st.columns(min(len(all_configs), 3))
                    for ci, cfg in enumerate(all_configs):
                        with cols[ci % 3]:
                            is_selected = (ci == selected_idx)
                            b64 = cfg.get("plot_png_b64","")
                            if b64:
                                import base64
                                st.image(base64.b64decode(b64), use_container_width=True)
                            site_lbl = cfg.get("label", cfg.get("site_type",""))
                            n_at = cfg.get("n_atoms","")
                            btn_label = f"{'✅ ' if is_selected else ''}conf{ci+1}: {site_lbl} ({n_at} atoms)"
                            if st.button(btn_label, key=f"sel_cfg_{tid}_{ci}"):
                                st.session_state[f"{key_poscar}_selected"] = ci
                                st.session_state[key_poscar] = cfg["poscar"]
                                st.session_state[key_plot]   = cfg.get("plot_png_b64","")
                                # Persist selected config
                                _save_task_state(sid, tid,
                                    task_name=name,
                                    poscar=cfg["poscar"],
                                    plot_png_b64=cfg.get("plot_png_b64",""),
                                    selected_config=ci,
                                )
                                st.rerun()

            else:
                # ── Regular build task (surface, molecule) ──
                if _has_build_endpoint(task):
                    payload_disp = (task.get("params") or {}).get("payload", {})
                    st.caption(f"`{agent}`  ←  `{payload_disp}`")
                    if st.button("▶ Generate Structure", key=f"s1_{tid}", type="primary"):
                        if not sid:
                            st.warning("Open a session first.")
                        else:
                            with st.spinner(f"Building {name}..."):
                                r = _execute_task(sid, task)
                            if r.get("ok"):
                                st.session_state[key_poscar] = r.get("poscar", "")
                                st.session_state[key_viz]    = r.get("viz", {})
                                st.session_state[key_plot]   = r.get("plot_png_b64", "")
                                n = r.get("n_atoms", "")
                                st.success(f"✅ {r.get('label', r.get('formula','Structure'))}  —  {n} atoms")
                                # Persist to DB
                                _save_task_state(sid, tid,
                                    task_name=name,
                                    poscar=r.get("poscar",""),
                                    plot_png_b64=r.get("plot_png_b64",""),
                                )
                            else:
                                st.error(r.get("detail") or r.get("error", "Build failed"))
                else:
                    st.caption("Paste POSCAR, or it will be populated automatically from a prior step.")
                    pasted = st.text_area("POSCAR", value=st.session_state.get(key_poscar, ""),
                                          height=80, key=f"poscar_paste_{tid}")
                    if pasted.strip():
                        st.session_state[key_poscar] = pasted.strip()

                if st.session_state.get(key_plot) or st.session_state.get(key_viz):
                    _render_structure_viz(st.session_state.get(key_viz, {}),
                                          st.session_state.get(key_plot, ""))

            if st.session_state.get(key_poscar):
                with st.expander("POSCAR", expanded=False):
                    st.code(st.session_state[key_poscar], language="text")

        # ─── Step 2: Generate Calculation Parameters ──────────────────────
        with st.expander("📝 Step 2 — Generate Parameters", expanded=has_poscar and not has_scripts):
            scripts_label = " + ".join(f"ase-{ct}.py" for ct in calc_types)
            st.caption(f"Scripts to generate: `{scripts_label}` + `job.sh`")
            if not has_poscar:
                st.info("Complete Step 1 first.")
            else:
                if st.button("⚙ Generate Parameters", key=f"s2_{tid}", type="primary"):
                    scripts = {}
                    all_ok = True
                    for ct in calc_types:
                        with st.spinner(f"Generating ase-{ct}.py..."):
                            r = api.generate_script(ct, system)
                        if r.get("ok"):
                            scripts[r.get("filename", f"ase-{ct}.py")] = r["script"]
                        else:
                            st.error(f"ase-{ct}.py failed: {r.get('detail','')}")
                            all_ok = False
                    if all_ok and scripts:
                        st.session_state[key_scripts] = scripts
                        st.success(f"✅ Generated: {', '.join(scripts.keys())}")
                        # Persist to DB
                        _save_task_state(sid, tid, task_name=name, scripts=scripts)

                if st.session_state.get(key_scripts):
                    for fn, src in st.session_state[key_scripts].items():
                        with st.expander(f"📄 {fn}", expanded=False):
                            st.code(src, language="python")
                            st.download_button(f"⬇ {fn}", src,
                                               file_name=fn, mime="text/x-python",
                                               key=f"dl_{tid}_{fn}")

                    # Show job.sh preview
                    # Use first script filename for the job.sh command
                    first_fn = next(iter(st.session_state[key_scripts]))
                    job_sh_preview = _render_job_sh(job_name, first_fn, ntasks, walltime, sched, account, env_setup)
                    with st.expander("📄 job.sh", expanded=False):
                        st.code(job_sh_preview, language="bash")

        # ─── Step 3: Submit to HPC ────────────────────────────────────────
        with st.expander("🚀 Step 3 — Submit to HPC", expanded=has_scripts and not has_job_id):
            st.caption(f"Target: `{host_str}`  |  cores: `{ntasks}`  |  walltime: `{walltime}`")
            st.caption(f"Remote: `{remote_path}/`")
            if not has_poscar:
                st.info("Complete Step 1 first.")
            elif not has_scripts:
                st.info("Complete Step 2 first.")
            else:
                scripts_dict = st.session_state[key_scripts]
                first_fn     = next(iter(scripts_dict))
                job_sh       = _render_job_sh(job_name, first_fn, ntasks, walltime, sched, account, env_setup)

                # Bundle includes all scripts
                all_scripts_zip = _make_zip_bundle(
                    poscar    = st.session_state[key_poscar],
                    script    = list(scripts_dict.values())[0],
                    script_fn = first_fn,
                    job_sh    = job_sh,
                )
                # If multiple scripts, add extras to zip
                if len(scripts_dict) > 1:
                    import io, zipfile
                    buf = io.BytesIO(all_scripts_zip)
                    with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as zf:
                        for fn, src in list(scripts_dict.items())[1:]:
                            zf.writestr(fn, src)
                    buf.seek(0)
                    all_scripts_zip = buf.read()

                col_dl, col_sub = st.columns(2)
                with col_dl:
                    st.download_button(
                        "📥 Download bundle",
                        data=all_scripts_zip,
                        file_name=f"{job_name}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key=f"dl_bundle_{tid}",
                    )
                with col_sub:
                    if sid and st.button("🚀 Submit Job", key=f"s3_{tid}",
                                         type="primary", use_container_width=True):
                        submit_payload = {
                            "poscar":      st.session_state[key_poscar],
                            "script":      list(scripts_dict.values())[0],
                            "filename":    first_fn,
                            "job_sh":      job_sh,
                            "remote_path": remote_path,
                            "cluster":     cc.get("host", "hoffman2"),
                            "user":        cc.get("user", ""),
                            "ntasks":      ntasks,
                            "walltime":    walltime,
                            "job_name":    job_name,
                            "session_id":  sid,
                        }
                        with st.spinner(f"Submitting to {host_str}..."):
                            r = api.post("/agent/hpc/submit", submit_payload)
                        if r.get("ok"):
                            jid = r.get("job_id", "?")
                            st.session_state[key_job_id] = jid
                            st.session_state[key_remote] = remote_path
                            st.success(f"✅ Job submitted — ID: **{jid}**")
                            st.caption(f"Remote: `{r.get('remote_path','')}`")
                            # Persist to DB
                            _save_task_state(sid, tid, task_name=name,
                                job_id=jid, remote_path=remote_path)
                        else:
                            st.error(r.get("detail") or r.get("error", "Submission failed"))

                if has_job_id:
                    jid_display = st.session_state[key_job_id]
                    st.success(f"Job ID: **{jid_display}**  |  "
                               f"`{st.session_state.get(key_remote,'')}`")

                    # ── Auto-feedback watcher ─────────────────────────────
                    key_watching = f"task_{tid}_watching"
                    if not st.session_state.get(key_watching):
                        species_hint = task.get("species", "")
                        surface_hint = task.get("surface", task.get("name", ""))
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            watch_species = st.text_input(
                                "Species (for feedback)", value=species_hint,
                                key=f"watch_species_{tid}",
                                placeholder="e.g. CO, H",
                            )
                        with col_w2:
                            watch_surface = st.text_input(
                                "Surface (for feedback)", value=surface_hint,
                                key=f"watch_surface_{tid}",
                                placeholder="e.g. Pt(111)",
                            )
                        poll_s = st.number_input(
                            "Poll interval (s)", 30, 600, 60, 30,
                            key=f"watch_poll_{tid}",
                            help="How often to check job status on cluster",
                        )
                        if st.button(
                            "👁 Watch Job (Auto-Feedback)",
                            key=f"watch_{tid}",
                            help="Starts background watcher: job done → parse OUTCAR → store DFT result → auto-feedback",
                        ):
                            db_task_id = task.get("db_id") or task.get("task_id") or 0
                            wr = api.hpc_watch(
                                task_id=int(db_task_id),
                                job_id=jid_display,
                                job_dir=st.session_state.get(key_remote, ""),
                                session_id=sid or 0,
                                cluster=cc.get("host", "hoffman2"),
                                species=watch_species,
                                surface=watch_surface,
                                poll_interval=int(poll_s),
                            )
                            if wr.get("ok") or wr.get("watching"):
                                st.session_state[key_watching] = True
                                st.success(
                                    f"👁 Watcher started (poll={poll_s}s). "
                                    "Results will auto-appear in Analysis panel when job finishes."
                                )
                                st.rerun()
                            else:
                                st.warning(
                                    wr.get("detail") or wr.get("error",
                                    "Watcher start failed — check backend logs.")
                                )
                    else:
                        st.info("👁 Watcher active — monitoring job on cluster.")

        # ─── Step 4: Retrieve Results ─────────────────────────────────────
        with st.expander("📥 Step 4 — Retrieve Results", expanded=has_job_id and not has_results):
            if not has_job_id:
                st.info("Submit job first (Step 3).")
            else:
                rpath = st.session_state.get(key_remote) or remote_path
                st.caption(f"Will fetch: `{', '.join(retrieve_files)}` from `{rpath}/`")
                if st.button("📥 Fetch Results", key=f"s4_{tid}", type="primary"):
                    with st.spinner(f"Fetching results from cluster..."):
                        r = api.hpc_fetch(
                            remote_path=rpath,
                            cluster=cc.get("host", "hoffman2"),
                            user=cc.get("user", ""),
                            files=retrieve_files,
                            session_id=sid,
                        )
                    if r.get("ok"):
                        st.session_state[key_results] = r.get("files", {})
                        energy = r.get("energy_eV")
                        st.success(f"✅ Retrieved {r['n_files']} files"
                                   + (f"  |  E = **{energy:.4f} eV**" if energy is not None else ""))
                        # Persist to DB
                        _save_task_state(sid, tid, task_name=name,
                            results=r.get("files",{}),
                            energy_eV=energy,
                        )
                        if r.get("errors"):
                            st.caption(f"Missing: {', '.join(r['errors'].keys())}")
                    else:
                        st.error(r.get("detail", "Fetch failed"))

                if st.session_state.get(key_results):
                    results = st.session_state[key_results]
                    energy_eV = None
                    if "OUTCAR" in results:
                        import re as _re
                        m = _re.findall(r"TOTEN\s*=\s*([-\d.]+)\s*eV", results["OUTCAR"])
                        if m: energy_eV = float(m[-1])
                    if energy_eV is not None:
                        st.metric("Total energy", f"{energy_eV:.4f} eV")

                    # Download zip of results
                    import io, zipfile
                    rbuf = io.BytesIO()
                    with zipfile.ZipFile(rbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fn, content in results.items():
                            zf.writestr(fn, content)
                        # Also include the ASE scripts for reference
                        if st.session_state.get(key_scripts):
                            for fn, src in st.session_state[key_scripts].items():
                                zf.writestr(fn, src)
                    rbuf.seek(0)
                    st.download_button(
                        f"⬇ Download results ({job_name})",
                        data=rbuf.read(),
                        file_name=f"{job_name}_results.zip",
                        mime="application/zip",
                        key=f"dl_results_{tid}",
                    )

                    for fn in ["CONTCAR", "stdout"]:
                        if fn in results and results[fn].strip():
                            with st.expander(f"📄 {fn}", expanded=False):
                                st.code(results[fn][:3000], language="text")


def _execute_task(session_id: int, task: dict) -> dict:
    """Execute a task via its endpoint (from params) or agent name. Returns response dict."""
    params   = task.get("params") or {}
    payload  = dict(params.get("payload", {}))
    endpoint = params.get("endpoint", "") or (task.get("meta") or {}).get("action_endpoint", "")
    agent    = task.get("agent", "")

    # For adsorption config tasks, inject surface POSCAR + mol POSCAR automatically
    if endpoint == "/agent/structure/generate_configs":
        # --- Surface POSCAR ---
        if not payload.get("poscar"):
            dep_ids = task.get("depends_on") or []
            poscar_injected = None
            for dep_id in dep_ids:
                poscar_injected = st.session_state.get(f"task_{dep_id}_poscar")
                if poscar_injected:
                    break
            if not poscar_injected:
                poscar_injected = st.session_state.get("task_1_poscar")
            if poscar_injected:
                payload["poscar"] = poscar_injected
            else:
                return {"ok": False, "detail": "Surface POSCAR not found — complete Step 1 of the surface task first."}

        # --- Molecule POSCAR (auto-lookup by species name) ---
        if not payload.get("mol_poscar"):
            species = task.get("species", "")
            # Search plan tasks for molecule task matching this species
            plan_tasks = (st.session_state.get("last_plan") or {}).get("tasks", [])
            mol_poscar_found = None
            for pt in plan_tasks:
                pt_payload = (pt.get("params") or {}).get("payload", {})
                pt_label = pt_payload.get("label", "")
                if pt_label and pt_label.lower() == species.lower():
                    mol_poscar_found = st.session_state.get(f"task_{pt['id']}_poscar")
                    if mol_poscar_found:
                        break
            if mol_poscar_found:
                payload["mol_poscar"] = mol_poscar_found
            # If not found, fall back to simple adsorbate placement (no mol_poscar key)

    if endpoint and endpoint.startswith("/"):
        r = api.post(endpoint, {"session_id": session_id, **payload})
    else:
        r = api.run_agent(agent, session_id, payload)
    return r


def _render_analysis_result(data: dict):
    summary = data.get("summary_md", "")
    if summary:
        st.markdown(summary)

    conclusions = data.get("conclusions", [])
    if conclusions:
        with st.expander(f"📌 Conclusions ({len(conclusions)})", expanded=True):
            for c in conclusions:
                conf = c.get("confidence", 0)
                st.markdown(
                    f"- {c.get('finding','')}"
                    f" {_pill(f'{conf:.0%}', _conf_color(conf))}",
                    unsafe_allow_html=True,
                )
                if c.get("evidence"):
                    st.caption(f"  Evidence: {c['evidence']}")

    gaps = data.get("gaps", [])
    if gaps:
        with st.expander(f"⚠️ Gaps ({len(gaps)})", expanded=False):
            for g in gaps:
                st.markdown(f"- {g}")

    suggestions = data.get("suggestions", [])
    if suggestions:
        with st.expander(f"💡 Next calculations ({len(suggestions)})", expanded=True):
            for s in suggestions:
                pri  = s.get("priority", "")
                desc = s.get("description", s.get("action", ""))
                rat  = s.get("rationale", "")
                color_map = {"critical": "red", "high": "amber", "medium": "blue", "low": "purple"}
                badge = _pill(pri, color_map.get(pri, "blue")) if pri else ""
                st.markdown(f"{badge} **{desc}**", unsafe_allow_html=True)
                if rat:
                    st.caption(rat)

    checklist = data.get("publication_checklist", {})
    if checklist:
        with st.expander("📄 Publication checklist", expanded=False):
            status = checklist.get("status", "?")
            icons = {"incomplete": "🔴", "near_complete": "🟡", "publishable": "🟢"}
            st.markdown(f"{icons.get(status,'⚪')} **Status:** `{status}`")
            for item in checklist.get("present", []):
                st.markdown(f"  ✅ {item}")
            for item in checklist.get("missing", []):
                st.markdown(f"  ❌ {item}")
            for item in checklist.get("nice_to_have", []):
                st.markdown(f"  💭 {item}")


def _render_free_energy_result(data: dict):
    """Render a ΔG free energy diagram result."""
    if not data:
        return
    interpretation = data.get("interpretation_md", "")
    if interpretation:
        st.markdown(interpretation)

    diagrams = data.get("diagrams", [])
    if diagrams:
        with st.expander(f"📈 ΔG Diagrams ({len(diagrams)} pathway(s))", expanded=True):
            for d in diagrams:
                st.markdown(f"**{d.get('pathway','')}**  "
                            f"| U_limiting = `{d.get('U_limiting_V','?')} V`  "
                            f"| η = `{d.get('overpotential_V','?')} V`")
                steps = d.get("steps", [])
                if steps:
                    import io
                    rows = [f"| {s['index']} | {s['label']} | {s['G']:.3f} | {s['delta_G']:+.3f} |"
                            for s in steps]
                    header = "| # | Species | G (eV) | ΔG (eV) |\n|---|---------|--------|---------|"
                    st.markdown(header + "\n" + "\n".join(rows))

    # Render base64 PNG if available
    plot_b64 = data.get("plot_png_b64")
    if plot_b64:
        st.image(f"data:image/png;base64,{plot_b64}", use_column_width=True)


def _render_microkinetics_result(data: dict):
    """Render microkinetic model results."""
    if not data:
        return
    interpretation = data.get("interpretation_md", "")
    if interpretation:
        st.markdown(interpretation)

    mk = data.get("microkinetics", {})
    if mk and mk.get("ok"):
        with st.expander("⚙️ Microkinetics Details", expanded=False):
            tof = mk.get("TOF_per_s", 0)
            rcs = mk.get("rate_controlling_step", "?")
            st.metric("TOF", f"{tof:.2e} s⁻¹")
            st.caption(f"Rate-controlling step: **{rcs}**")
            st.caption(f"Temperature: {mk.get('T_K', '?')} K")
            coverages = mk.get("coverages", {})
            if coverages:
                st.markdown("**Surface coverages (θ):**")
                for k, v in list(coverages.items())[:8]:
                    st.text(f"  {k}: {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(session_id: int, query: str) -> dict:
    """
    Run intent → hypothesis → plan sequentially.
    Returns a result dict with all three stages.
    Shows a live progress indicator while running.
    """
    result = {"intent": None, "hypothesis": None, "plan": None}

    progress = st.empty()

    # 1. Intent
    progress.status("🎯 Parsing intent...", state="running")
    ir = api.run_intent(session_id, query)
    if not ir.get("ok"):
        progress.error(f"Intent failed: {ir.get('detail','unknown error')}")
        return result
    intent = ir.get("intent") or ir
    result["intent"] = intent

    # 2. Hypothesis
    progress.status("🧪 Generating mechanism hypothesis...", state="running")
    hr = api.run_hypothesis(session_id, intent)
    if hr.get("ok"):
        # hr["hypothesis"] is a HypothesisBundle dict, not a raw string
        hyp_bundle = hr.get("hypothesis") if isinstance(hr.get("hypothesis"), dict) else {}
        hypothesis = {
            "md":           hyp_bundle.get("md") or hr.get("result_md", ""),
            "steps":        hyp_bundle.get("steps") or hr.get("steps", []),
            "intermediates": hyp_bundle.get("intermediates") or hr.get("intermediates", []),
            "ts":           hyp_bundle.get("ts") or hr.get("ts", []),
            "graph":        hr.get("graph", {}),
            "confidence":   hyp_bundle.get("confidence") or hr.get("confidence", 0),
        }
    else:
        hypothesis = {"md": f"_Hypothesis unavailable: {hr.get('detail','')}_ ", "steps": [], "graph": {}}
    result["hypothesis"] = hypothesis

    # 3. Plan  (pass full hypothesis + graph so plan_agent uses correct intermediates)
    progress.status("📋 Building calculation plan...", state="running")
    pr = api.run_plan(session_id, intent, hypothesis, graph=hypothesis.get("graph", {}))
    if pr.get("ok"):
        result["plan"] = pr
    else:
        result["plan"] = {"tasks": [], "error": pr.get("detail", "")}

    progress.status("✅ Done", state="complete")
    time.sleep(0.4)
    progress.empty()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:2px">
        <h2 style="margin:0;font-weight:800">⚛️ ChatDFT</h2>
        <span style="color:#64748b;font-size:0.95rem">
            Describe your reaction system — intent, mechanism, and calculations happen automatically.
        </span>
    </div>
    """, unsafe_allow_html=True)

    sid = st.session_state.session_id
    if sid:
        sname = st.session_state.session_name or f"Session {sid}"
        st.caption(f"Active session: **{sname}** (#{sid})")
    else:
        st.info("👈 Create or open a session from the sidebar to get started.")

    st.divider()

    # Chat history
    for msg in st.session_state.chat_messages:
        render_message(msg)

    # ── Quick Demo buttons ─────────────────────────────────────────────────────
    if sid and not st.session_state.chat_messages:
        st.markdown("**Quick demos** — click to run a pre-configured workflow:")
        _DEMOS = [
            ("🔥 Butane dehydrogenation",
             "Study the dehydrogenation of C4H10 to C4H8 on Pt(111) surface via thermal catalysis. "
             "I want the full reaction mechanism including C-H bond activation steps, "
             "transition states (NEB), and a free energy diagram at 500 K."),
            ("⚡ CO₂RR on Cu(111)",
             "Study CO2 reduction reaction (CO2RR) on Cu(111) surface to produce methanol (CH3OH). "
             "Electrochemical conditions: pH=7, U=-0.8 V vs RHE, KHCO3 electrolyte. "
             "Include full CO2→COOH*→CO*→CHO*→CH2O*→CH3O*→CH3OH pathway with PCET steps."),
        ]
        _demo_cols = st.columns(len(_DEMOS))
        _demo_query = None
        for _col, (_label, _q) in zip(_demo_cols, _DEMOS):
            with _col:
                if st.button(_label, use_container_width=True, key=f"demo_{_label[:8]}"):
                    _demo_query = _q
        if _demo_query:
            _add_message("user", _demo_query)
            with st.chat_message("user", avatar="👤"):
                st.markdown(_demo_query)
            with st.chat_message("assistant", avatar="⚛️"):
                pipeline_data = run_pipeline(sid, _demo_query)
            summary = _pipeline_summary(pipeline_data)
            _add_message("assistant", summary, kind="pipeline", data=pipeline_data)
            st.session_state.last_intent    = pipeline_data["intent"]
            st.session_state.last_hypothesis = pipeline_data["hypothesis"]
            st.session_state.last_plan       = pipeline_data["plan"]
            st.rerun()

    # Input
    placeholder = (
        "e.g. I want to study the electrochemical dehydrogenation of C4H10 to C4H8 on Pt(111)"
        if sid else
        "Create a session first →"
    )
    query = st.chat_input(placeholder, disabled=not bool(sid))

    if query and sid:
        # Show user message
        _add_message("user", query)
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Run pipeline
        with st.chat_message("assistant", avatar="⚛️"):
            pipeline_data = run_pipeline(sid, query)

        # Store result in chat history
        summary = _pipeline_summary(pipeline_data)
        _add_message("assistant", summary, kind="pipeline", data=pipeline_data)

        # Store for sidebar reuse
        st.session_state.last_intent    = pipeline_data["intent"]
        st.session_state.last_hypothesis = pipeline_data["hypothesis"]
        st.session_state.last_plan       = pipeline_data["plan"]

        st.rerun()


def _pipeline_summary(data: dict) -> str:
    """One-line text summary of a pipeline run (used as content for plain-text fallback)."""
    intent = data.get("intent") or {}
    if not isinstance(intent, dict):
        intent = {}
    task   = intent.get("task") or intent.get("problem_type", "")
    subst  = intent.get("substrate") or (intent.get("system") or {}).get("material", "")
    n_tasks = len((data.get("plan") or {}).get("tasks", []))
    return f"Pipeline complete. Task: {task}. Surface: {subst}. Plan: {n_tasks} calculations."


# ─────────────────────────────────────────────────────────────────────────────
# APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        render_login()
        return

    render_sidebar()
    render_main()


main()
