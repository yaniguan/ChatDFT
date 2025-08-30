# client/app.py
# -*- coding: utf-8 -*-
import os, time, json, pathlib, io, textwrap
import requests
import pandas as pd
import streamlit as st

try:
    import py3Dmol
    from ase.io import read as ase_read, write as ase_write
    _HAS_3D = True
except Exception:
    _HAS_3D = False

# -------------------- Config --------------------
API = os.environ.get("CHATDFT_API", "http://localhost:8080")
TEMPLATES_FILE = pathlib.Path(".job_templates.json")

st.set_page_config(page_title="ChatDFT — HPC", layout="wide")

# === Template preload hook: must run BEFORE widgets are created ===
if "__tpl_to_load__" in st.session_state:
    t = st.session_state.pop("__tpl_to_load__")
    for k in ("pbs_ppn","pbs_wall","slurm_partition","slurm_ntasks","slurm_wall"):
        if k in t and t[k] is not None:
            st.session_state[k] = t[k]
    st.session_state["ase_opt_prefill"] = t.get("ase_opt","")
    st.session_state["ase_sh_prefill"]  = t.get("ase_sh","")
    st.session_state["scheduler"] = t.get("scheduler", st.session_state.get("scheduler","PBS (qsub)"))

# ------------- Polished CSS ----------
st.markdown("""
<style>
:root { --card-bg:#fff; --muted:#6b7280; --ring:#eef2ff; --ink:#0f172a; }
.block-container{padding-top:1.0rem;padding-bottom:1.0rem}
h1,h2,h3 { color: var(--ink); }
.hero { display:flex; align-items:center; justify-content:space-between; padding:10px 16px; border-radius:14px;
  background: linear-gradient(180deg,#f6f8ff, #ffffff); border:1px solid #eef2ff; margin-bottom:14px; }
.hero .brand { font-weight:800; font-size:24px; letter-spacing:.2px; }
.hero .hint  { color:#64748b; font-size:13px; }

.card { background: var(--card-bg); border-radius: 14px; padding: 18px 18px;
  box-shadow: 0 6px 20px rgba(2,6,23,.05); border:1px solid #edf2f7; }
.card h3 { margin: 0 0 6px; font-size:18px; }
.smallmuted {color:var(--muted); font-size:12px}
.blocktitle {font-weight:800; font-size:20px; margin:6px 0 12px;}
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700;
  border:1px solid rgba(0,0,0,.06); background:#f8fafc; color:#334155; }
.badge-running {background:#E6F4EA; color:#137333; border-color:#c9f0d2;}
.badge-queued  {background:#FFF4E5; color:#8F5A00; border-color:#ffe3bf;}
.badge-done    {background:#E6F0FE; color:#1A73E8; border-color:#dbe5ff;}
.badge-failed  {background:#FDE7E9; color:#C5221F; border-color:#ffd7dc;}
.table-wrap { border:1px solid #eef2ff; border-radius:12px; overflow:hidden }
.table-wrap table { width:100%; font-size:13.5px }
.table-wrap thead tr { background:#f8fafc; }
.table-wrap td, .table-wrap th { padding:10px 12px; }
hr.sep { border:none; height:1px; background:linear-gradient(90deg,#fff, #eaeefc, #fff); margin:14px 0; }
.kv {font-size:13px;color:#334155}
.kv b{display:inline-block;width:110px;color:#0f172a}
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def api_get(path, params=None, timeout=30):
    r = requests.get(f"{API}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path, payload, timeout=120):
    r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5.0)
def list_sessions():
    try:
        return api_get("/session/list")
    except Exception:
        return []

def default_poscar():
    return (
        "Si\n 5.43\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n"
        " Si\n 2\n Direct\n 0.00 0.00 0.00\n 0.25 0.25 0.25\n"
    )

def tmpl_ase_opt_min():
    return """# minimal placeholder if ASE/Calculator not available
open('ase_results.csv','w').write('step,energy\\n0,-1.234\\n1,-1.245\\n')
open('CONTCAR','w').write('dummy contcar\\n'); print('done')
"""

def tmpl_pbs(cores=4, wall="01:00:00"):
    return f"""#!/bin/bash
#PBS -N ase_opt
#PBS -l nodes=1:ppn={cores}
#PBS -l walltime={wall}
#PBS -j oe
#PBS -V
cd "$PBS_O_WORKDIR"
module load python/3.10 || true
python ase-opt.py > ase.out 2>&1
"""

def tmpl_slurm(partition="standard", ntasks=4, wall="01:00:00"):
    return f"""#!/bin/bash
#SBATCH --job-name=ase_opt
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time={wall}
#SBATCH --output=ase.out
#SBATCH --error=ase.err
cd $SLURM_SUBMIT_DIR
module load python/3.10 || true
python ase-opt.py > ase.run.log 2>&1
"""

def status_badge(s: str) -> str:
    s = (s or "").lower()
    if s in ("running",): klass = "badge-running"
    elif s in ("queued","submitted"): klass = "badge-queued"
    elif s in ("done","synced"): klass = "badge-done"
    elif s in ("failed",): klass = "badge-failed"
    else: klass = "badge"
    return f'<span class="badge {klass}">{s}</span>'

def load_templates():
    if TEMPLATES_FILE.exists():
        try: return json.loads(TEMPLATES_FILE.read_text())
        except Exception: return {}
    return {}

def save_templates(d: dict):
    TEMPLATES_FILE.write_text(json.dumps(d, indent=2))

def auto_refresh_every(seconds: int = 5):
    now = int(time.time())
    last = st.session_state.get("_last_tick", 0)
    if now - last >= seconds:
        st.session_state["_last_tick"] = now
        try:
            st.query_params["ts"] = str(now)
        except Exception:
            st.experimental_set_query_params(ts=now)

def _normalize_sessions(sessions_raw):
    out = []
    for s in sessions_raw or []:
        if not isinstance(s, dict): continue
        uid = (s.get("uid") or "").strip()
        if not uid: continue
        out.append({
            "uid": uid,
            "name": s.get("name") or "(no name)",
            "project": s.get("project") or "",
        })
    return out

def _session_labels(sessions):
    return { s["uid"]: f'{s["name"]} · {s["uid"][:8]}' for s in sessions }

def _poscar_pretty_preview(preview: dict | None) -> str:
    if not preview or not preview.get("ok"):
        return "No POSCAR preview."
    lat = preview.get("lattice") or []
    elems = preview.get("elements") or []
    counts = preview.get("counts") or []
    nat = preview.get("n_atoms")
    lines = [
        f"<div class='kv'><b>Title</b>{preview.get('title','')}</div>",
        f"<div class='kv'><b>Scale</b>{preview.get('scale')}</div>",
        f"<div class='kv'><b>Lattice</b>{lat}</div>",
        f"<div class='kv'><b>Elements</b>{elems}</div>",
        f"<div class='kv'><b>Counts</b>{counts}  (N={nat})</div>",
    ]
    return "\n".join(lines)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### Connection")
    api_in = st.text_input("API base", API, help="FastAPI base URL")
    if api_in: API = api_in

    st.markdown("### Mode")
    scheduler = st.radio("Scheduler", ["PBS (qsub)", "Slurm (sbatch)"], index=0, key="scheduler")
    live = st.toggle("🔄 Live monitor (auto refresh 5s)", value=False)
    if live: auto_refresh_every(5)

# --------------- Hero header + status lamp -------------
st.markdown(
    '<div class="hero"><div class="brand">⚙️ ChatDFT — HPC</div>'
    '<div class="hint">Sidebar: API / Scheduler / Live monitor</div></div>',
    unsafe_allow_html=True
)
status_slot = st.empty()

def render_indicator(session_uid: str | None):
    if not session_uid:
        status_slot.markdown('<div style="text-align:right" class="smallmuted">No session</div>', unsafe_allow_html=True)
        return
    try:
        jobs = api_get("/job/list", params={"session_uid": session_uid})
        running = sum(1 for j in jobs if (j.get("status") or "").lower() == "running")
        queued  = sum(1 for j in jobs if (j.get("status") or "").lower() in ("queued","submitted"))
        html = f'''
        <div style="text-align:right">
          <span class="badge badge-running">RUN {running}</span>
          <span style="margin-left:8px" class="badge badge-queued">PEND {queued}</span>
        </div>'''
        status_slot.markdown(html, unsafe_allow_html=True)
    except Exception:
        status_slot.markdown('<div style="text-align:right" class="smallmuted">Monitor offline</div>', unsafe_allow_html=True)

# ================= Layout =================

# ---- Sessions ----
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3>1) Sessions</h3>', unsafe_allow_html=True)
sessions_raw = list_sessions()
sessions = _normalize_sessions(sessions_raw)
labels = _session_labels(sessions)
options = [s["uid"] for s in sessions]

sid = None
if not options:
    st.info("No sessions yet.")
else:
    sid = st.selectbox(
        "Open session",
        options=options,
        format_func=lambda u: labels.get(u, str(u)[:8]),
    )
newname = st.text_input("New session name", "test-session")
if st.button("➕ Create session", use_container_width=True):
    try:
        r = api_post("/session/create", {"name": newname})
        st.success(f"Created session: {r['session_uid']}")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Create failed: {e}")
st.markdown('</div>', unsafe_allow_html=True)

render_indicator(sid)

# ---- Tabs: Legacy | Agent mode | Batch | Monitor | Results ----
tab_legacy, tab_agent, tab_batch, tab_monitor, tab_results = st.tabs(
    ["🧩 Legacy submit", "🤖 Agent mode", "📦 Batch CSV", "📡 Monitor", "📈 Results & Files"]
)

# ================= LEGACY SUBMIT =================
with tab_legacy:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>2) Prepare & Submit a Job (legacy)</h3>', unsafe_allow_html=True)

    job_title = st.text_input("Job title", "opt-test", key="legacy_title")

    with st.expander("Resources (Slurm/PBS)"):
        if st.session_state.get("scheduler","PBS (qsub)").startswith("PBS"):
            pbs_ppn  = st.number_input("ppn (PBS)", 1, 64, 4, 1, key="pbs_ppn")
            pbs_wall = st.text_input("walltime (PBS)", "01:00:00", key="pbs_wall")
        else:
            slurm_partition = st.text_input("partition", "standard", key="slurm_partition")
            slurm_ntasks    = st.number_input("tasks-per-node", 1, 64, 4, 1, key="slurm_ntasks")
            slurm_wall      = st.text_input("time", "01:00:00", key="slurm_wall")

    poscar  = st.text_area("POSCAR", height=160, value=default_poscar(), key="legacy_poscar")
    ase_opt = st.text_area("ase-opt.py", height=160, value=st.session_state.get("ase_opt_prefill", tmpl_ase_opt_min()), key="legacy_ase_opt")

    if st.session_state.get("scheduler","PBS (qsub)").startswith("PBS"):
        ase_sh = st.text_area("ase.sh", height=120, value=st.session_state.get("ase_sh_prefill", tmpl_pbs(
            cores=st.session_state.get("pbs_ppn",4), wall=st.session_state.get("pbs_wall","01:00:00")
        )), key="legacy_ase_sh")
    else:
        ase_sh = st.text_area("ase.sh", height=120, value=st.session_state.get("ase_sh_prefill", tmpl_slurm(
            partition=st.session_state.get("slurm_partition","standard"),
            ntasks=st.session_state.get("slurm_ntasks",4),
            wall=st.session_state.get("slurm_wall","01:00:00")
        )), key="legacy_ase_sh")

    st.markdown("**Templates**")
    templates = load_templates()
    tcol1, tcol2, tcol3 = st.columns([1,1,2])
    with tcol1:
        tname = st.text_input("Save as", "default", key="tpl_name")
        if st.button("💾 Save template"):
            templates[tname] = {
                "scheduler": st.session_state.get("scheduler","PBS (qsub)"),
                "pbs_ppn": st.session_state.get("pbs_ppn",4),
                "pbs_wall": st.session_state.get("pbs_wall","01:00:00"),
                "slurm_partition": st.session_state.get("slurm_partition","standard"),
                "slurm_ntasks": st.session_state.get("slurm_ntasks",4),
                "slurm_wall": st.session_state.get("slurm_wall","01:00:00"),
                "ase_opt": ase_opt, "ase_sh": ase_sh,
            }
            save_templates(templates)
            st.success(f"Saved template: {tname}")
    with tcol2:
        pick = st.selectbox("Load", ["—"] + list(templates.keys()), key="tpl_pick")
        if st.button("📥 Load"):
            if pick != "—":
                st.session_state["__tpl_to_load__"] = templates[pick]
                st.rerun()
    with tcol3:
        st.caption("保存/加载常用资源与脚本。")

    ready = all([sid, job_title.strip(), poscar.strip(), ase_opt.strip(), ase_sh.strip()])
    if st.button("🚀 Submit (legacy)", use_container_width=True, disabled=not ready):
        payload = {
            "session_uid": sid,
            "title": job_title.strip(),
            "poscar": poscar,
            "ase_opt_py": ase_opt,
            "ase_sh": ase_sh,
            "scheduler": "slurm" if st.session_state.get("scheduler","PBS (qsub)").startswith("Slurm") else "pbs",
            "resources": {
                "ppn": st.session_state.get("pbs_ppn"),
                "walltime": st.session_state.get("pbs_wall"),
                "partition": st.session_state.get("slurm_partition"),
                "ntasks_per_node": st.session_state.get("slurm_ntasks"),
                "time": st.session_state.get("slurm_wall"),
            }
        }
        try:
            r = api_post("/job/create", payload)
            st.success(f"Submitted: job_uid={r.get('job_uid')} id={r.get('pbs_id') or r.get('slurm_id')}")
        except Exception as e:
            st.error(f"Submit failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= AGENT MODE =================
with tab_agent:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>2) Agent pipeline (prepare-only / submit)</h3>', unsafe_allow_html=True)

    colA, colB, colC = st.columns([1.0, 1.0, 1.0])

    with colA:
        agent = st.selectbox(
            "Agent",
            [
                "structure.relax_slab",
                "structure.relax_adsorbate",
                "adsorption.scan",
                "adsorption.co",
                "neb.run",
                "electronic.dos",
                "post.analysis",
                "knowledge.search",
            ],
            index=0
        )
        engine = st.selectbox("Engine", ["vasp","qe"], index=0)
        cluster = st.text_input("Cluster", "hoffman2")
    with colB:
        job_name = st.text_input("Job name (optional)", "agent-job")
        submit = st.checkbox("Submit to HPC", value=False, help="未勾选时为 prepare-only，直接预览结构与输入文件")
        wait = st.checkbox("Wait (block until done)", value=False, disabled=not submit)
        fetch = st.checkbox("Fetch outputs (after submit)", value=True, disabled=not submit)
    with colC:
        do_post = st.checkbox("Run post-analysis (after fetch/wait)", value=True)
        poll = st.number_input("Poll seconds", 5, 600, 60, 5)
        fetch_filters_default = "OUTCAR,vasprun.xml,OSZICAR,stdout*,stderr*,CONTCAR,DOSCAR,EIGENVAL,PROCAR,CHGCAR,AECCAR*,ELFCAR,ACF.dat,PARCHG*"
        fetch_filters = st.text_input("Fetch filters (comma-separated)", fetch_filters_default)

    st.caption("Task Params.payload（可直接编辑 JSON；已给出最常用样例骨架）")
    default_payload = {
        "engine": engine,
        "template_vars": {},
        "ntasks": 24,
        "walltime": "02:00:00",
        # 结构&吸附（示例；按 agent 取用）
        "structure": {
            "source": "mp-30",  # or "POSCAR" text …
            "slab": {"facet": "111", "layers": 4, "vacuum": 15.0}
        },
        "adsorbates": ["CO*"],
        "sites": "auto",
        # 若是 NEB，可提供反应式
        "step": "CO2(g) + H* -> COOH*",
    }
    payload_text = st.text_area(
        "task.params.payload",
        value=json.dumps(default_payload, indent=2),
        height=220
    )

    # ---- Action buttons ----
    c1, c2 = st.columns([1,1])
    if c1.button("🧪 Prepare only (no submit)", use_container_width=True):
        try:
            P = json.loads(payload_text) if payload_text.strip() else {}
        except Exception as e:
            st.error(f"Payload JSON 无法解析: {e}")
            st.stop()

        body = {
            "agent": agent,
            "task": {"id": 1, "name": job_name or agent, "params": {"payload": P}},
            "engine": P.get("engine", engine),
            "cluster": cluster,
            "submit": False,
            "fetch": False,
            "wait": False,
            "do_post": False,
            "run_id": 0
        }
        try:
            r = api_post("/agent/run", body)
            if not r.get("ok"):
                st.error(r.get("detail") or "prepare failed.")
            else:
                res = r.get("result") or {}
                st.success("Prepared.")
                st.markdown("**Generated files**")
                st.json(res.get("files") or [])
                st.markdown("**POSCAR preview**")
                prev = res.get("preview") or {}
                st.markdown(_poscar_pretty_preview(prev), unsafe_allow_html=True)
                if prev.get("error"):
                    st.caption(prev.get("error"))
                st.markdown("**Job dir**")
                st.code(res.get("job_dir") or "(unknown)")
        except Exception as e:
            st.error(f"Prepare-only failed: {e}")

    if c2.button("🚀 Submit via Agent", use_container_width=True):
        try:
            P = json.loads(payload_text) if payload_text.strip() else {}
        except Exception as e:
            st.error(f"Payload JSON 无法解析: {e}")
            st.stop()

        body = {
            "agent": agent,
            "task": {"id": 1, "name": job_name or agent, "params": {"payload": P}},
            "engine": P.get("engine", engine),
            "cluster": cluster,
            "submit": True,
            "wait": bool(wait),
            "fetch": bool(fetch),
            "poll": int(poll),
            "fetch_filters": [x.strip() for x in (fetch_filters or "").split(",") if x.strip()],
            "do_post": bool(do_post),
            "job_name": job_name,
            "run_id": 0
        }
        try:
            r = api_post("/agent/run", body)
            if not r.get("ok"):
                st.error(r.get("detail") or "submit failed.")
            else:
                res = r.get("result") or {}
                st.success("Submitted.")
                st.markdown("**HPC**")
                h = res.get("hpc") or {}
                st.markdown(f"- job_id: `{h.get('job_id','')}`  \n- script: `{h.get('script','')}`")
                if res.get("post"):
                    st.markdown("**Post-analysis**")
                    st.json(res["post"])
                st.markdown("**Job dir**")
                st.code(res.get("job_dir") or "(unknown)")
        except Exception as e:
            st.error(f"Agent submit failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= BATCH CSV =================
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>📦 Batch Builder (CSV)</h3>', unsafe_allow_html=True)

    csv_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    batch_server = st.selectbox("Target server (default)", ["hoffman2", "exp-slurm"])

    if st.session_state.get("scheduler","PBS (qsub)").startswith("PBS"):
        defaults = {"ppn": st.number_input("Default ppn", 1, 64, 4),
                    "walltime": st.text_input("Default walltime", "02:00:00")}
    else:
        defaults = {"partition": st.text_input("Default partition", "standard"),
                    "ntasks": st.number_input("Default ntasks", 1, 64, 4),
                    "time": st.text_input("Default time", "02:00:00")}

    def _require_cols(df, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"CSV 缺少必要列: {missing}")
            return False
        return True

    def _row_params(row):
        params = {}
        for k in ["ppn","walltime","partition","ntasks","time"]:
            if k in row and pd.notna(row[k]):
                params[k] = row[k]
        return params

    def _make_items(df):
        items = []
        for _, row in df.iterrows():
            if pd.isna(row.get("title")) or pd.isna(row.get("poscar")) or pd.isna(row.get("ase_opt_py")):
                continue
            items.append({
                "title": str(row["title"]),
                "poscar": str(row["poscar"]),
                "ase_opt_py": str(row["ase_opt_py"]),
                "params": _row_params(row),
                "server_name": (str(row["server_name"]).strip() if "server_name" in df.columns and pd.notna(row["server_name"]) else None)
            })
        return items

    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.dataframe(df, use_container_width=True, height=220)

        if _require_cols(df, ["title","poscar","ase_opt_py"]) and st.button("🚀 Submit batch", use_container_width=True):
            if not sid:
                st.warning("先选择/创建一个 session 再提交批量任务。")
            else:
                items = _make_items(df)
                if not items:
                    st.warning("CSV 没有有效行。")
                else:
                    groups = {}
                    for it in items:
                        svr = it.get("server_name") or batch_server
                        groups.setdefault(svr, []).append({k: it[k] for k in ["title","poscar","ase_opt_py","params"]})
                    total = 0
                    try:
                        for svr, its in groups.items():
                            payload = {"session_uid": sid, "server_name": svr, "defaults": defaults, "items": its}
                            r = api_post("/batch/create", payload)
                            st.success(f"[{svr}] Batch submitted: {r['batch_uid']} · {len(r['items'])} jobs")
                            total += len(r["items"])
                        st.info(f"合计提交 {total} 个作业。")
                    except Exception as e:
                        st.error(f"Batch submit failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= MONITOR =================
with tab_monitor:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>📡 Cluster Monitor</h3>', unsafe_allow_html=True)
    if not sid:
        st.info("Select or create a session first.")
    else:
        try:
            jobs = api_get("/job/list", params={"session_uid": sid})
        except Exception as e:
            jobs = []; st.error(f"List jobs failed: {e}")

        if jobs:
            df = pd.DataFrame(jobs)
            cols = ["pbs_id","title","status","local_dir","remote_dir"]
            df = df.reindex(columns=[c for c in cols if c in df.columns])
            if "status" in df.columns:
                df2 = df.copy()
                df2["status"] = df2["status"].map(lambda s: status_badge(s), na_action="ignore")
                st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
                st.write(df2.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.dataframe(df, use_container_width=True, height=260)

            col_sync, _ = st.columns([1,1])
            with col_sync:
                jid = st.selectbox("Select job to sync", [j["job_uid"] for j in jobs], key="sync_pick")
                if st.button("🔄 Sync now", use_container_width=True):
                    try:
                        api_post("/job/sync", {"job_uid": jid})
                        st.success("Sync triggered.")
                    except Exception as e:
                        st.error(f"Sync failed: {e}")
        else:
            st.info("No jobs yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= RESULTS & FILES =================
with tab_results:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markmarkdown = st.markdown
    st.markmarkdown('<h3>📈 Results & Files</h3>', unsafe_allow_html=True)

    if not sid:
        st.info("Select a session first.")
    else:
        try:
            jobs_for_result = api_get("/job/list", params={"session_uid": sid})
        except Exception:
            jobs_for_result = []
        if jobs_for_result:
            sel_job = st.selectbox("Select job", [j["job_uid"] for j in jobs_for_result])

            tab_csv, tab_files = st.tabs(["📈 Results", "📁 Files & 3D"])
            with tab_csv:
                if st.button("📥 Load results", key="load_csv"):
                    try:
                        rows = api_get("/job/results", params={"job_uid": sel_job})
                        rdf = pd.DataFrame(rows)
                        if rdf.empty:
                            st.warning("No result rows yet. Try Sync.")
                        else:
                            st.dataframe(rdf, use_container_width=True, height=260)
                            st.download_button("Download CSV",
                                data=rdf.to_csv(index=False).encode("utf-8"),
                                file_name=f"{sel_job}.csv")
                            try:
                                import matplotlib.pyplot as plt
                                fig = plt.figure()
                                x = pd.to_numeric(rdf["step"], errors="coerce")
                                y = pd.to_numeric(rdf["energy"], errors="coerce")
                                plt.plot(x, y, marker="o")
                                plt.xlabel("step"); plt.ylabel("energy")
                                st.pyplot(fig)
                            except Exception as e:
                                st.info(f"Plot skipped: {e}")
                    except Exception as e:
                        st.error(f"Load results failed: {e}")

            with tab_files:
                try:
                    files = api_get("/job/files", params={"job_uid": sel_job})
                    if not files:
                        st.info("No files yet.")
                    else:
                        fdf = pd.DataFrame(files)
                        st.dataframe(fdf, use_container_width=True, height=220)
                        pick = st.selectbox("Pick a file to download", [f["path"] for f in files])
                        if st.button("⬇️ Download file"):
                            resp = requests.get(f"{API}/job/file", params={"job_uid": sel_job, "path": pick}, timeout=30)
                            if resp.ok:
                                st.download_button("Save", data=resp.content, file_name=os.path.basename(pick))
                            else:
                                st.error(f"Download failed: {resp.status_code}")
                except Exception:
                    st.caption("后端未提供 /job/files 与 /job/file，文件面板暂不可用。")

                st.markdown("**3D Viewer (CONTCAR)**")
                style_choice = st.selectbox("Style", ["stick", "ball&stick", "sphere", "line"], index=1, key="style_3d")
                if _HAS_3D and st.button("🔍 Preview CONTCAR", key="btn_preview_concar"):
                    try:
                        resp = requests.get(f"{API}/job/file", params={"job_uid": sel_job, "path": "CONTCAR"}, timeout=30)
                        resp.raise_for_status()
                        text = resp.content.decode("utf-8", errors="ignore")
                        atoms = ase_read(io.StringIO(text), format="vasp")
                        buf_out = io.StringIO()
                        ase_write(buf_out, atoms, format="xyz")
                        xyz_text = buf_out.getvalue()
                        view = py3Dmol.view(width=560, height=400)
                        view.addModel(xyz_text, "xyz")
                        styles = {
                            "stick": {"stick": {}},
                            "ball&stick": {"stick": {}, "sphere": {"scale": 0.25}},
                            "sphere": {"sphere": {}},
                            "line": {"line": {}},
                        }
                        view.setStyle(styles.get(style_choice, {"stick": {}}))
                        view.zoomTo()
                        st.components.v1.html(view._make_html(), height=420, scrolling=False)
                        st.download_button("Download XYZ", data=xyz_text.encode("utf-8"),
                                           file_name=f"{sel_job}.xyz", mime="chemical/x-xyz")
                    except Exception as e:
                        st.error(f"3D 渲染失败：{e}")
                elif not _HAS_3D:
                    st.info("缺少 py3Dmol / ASE，无法渲染 3D。")
        else:
            st.info("No jobs yet.")
    st.markdown('</div>', unsafe_allow_html=True)