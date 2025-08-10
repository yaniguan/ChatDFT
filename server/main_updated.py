# server/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.chat.session_agent import router as session_router
from server.chat.intent_agent import router as intent_router
from server.chat.hypothesis_agent import router as hypothesis_router
from server.chat.plan_agent import router as plan_router
from server.chat.history_agent import router as history_router
from server.chat.knowledge_agent import router as knowledge_router
from server.execution.agent_routes import router as agent_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(session_router, tags=["chat"])
app.include_router(intent_router, tags=["chat"])
app.include_router(hypothesis_router, tags=["chat"])
app.include_router(plan_router, tags=["chat"])        # <--- 必须有这行
app.include_router(history_router, tags=["chat"])
app.include_router(knowledge_router, tags=["chat"])
app.include_router(agent_router)  # ← 新增

from server.chat.records_agent import router as records_router
app.include_router(records_router)


# ====== Added endpoints and helpers (appended by assistant) ======
# Imports for new functionality
from fastapi import HTTPException, Response, Body
from pydantic import BaseModel
from typing import List, Dict, Optional
import os, json, time, threading, re as _re, uuid
from datetime import datetime
from jinja2 import Template
import pathlib

# Project imports
try:
    from server.utils.openai_wrapper import LLMAgent  # if present
except Exception:
    LLMAgent = None

from server.db import SessionLocal, engine
try:
    # If you use a separate models module
    from server.db import Base, ChatSession, Job, FileAsset, ResultRow
except Exception:
    # Fallback to definitions in server.db if that's where models live
    from server.db import Base, ChatSession, Job, FileAsset, ResultRow  # type: ignore

from server.schema import SessionCreate, JobCreate, JobId
from server.settings import settings, get_server
from server import sshio, parser

# CORS (idempotent if added multiple times)
try:
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
except NameError:
    # If `app` doesn't exist yet (shouldn't happen in this file), create it
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# DB init (safe print & create_all wrapped)
try:
    print("[DB] using:", engine.url)
    Base.metadata.create_all(bind=engine)
except Exception as _e:
    print("[DB init skipped]", _e)

# --- utils ---
_slug_re = _re.compile(r"[^a-zA-Z0-9._-]+")
def slugify(name: str, default: str = "job") -> str:
    """Make a filesystem-safe slug."""
    name = (name or "").strip() or default
    name = name.replace(" ", "-")
    name = _slug_re.sub("-", name)
    name = _re.sub("-{2,}", "-", name).strip("-")
    return name or default

def _sync_job(job_uid: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_uid == job_uid).first()
        if not job:
            return

        # 1) Poll status if we have a scheduler id
        if job.slurm_or_pbs_id:
            state = sshio.poll_status(job.slurm_or_pbs_id)
            if state in ("queued", "running"):
                if job.status != state:
                    job.status = state
                    db.commit()
            else:
                if job.status != "done":
                    job.status = "done"
                    db.commit()

        # 2) Pull results and parse when done
        if job.status == "done":
            if not job.local_dir:
                job.local_dir = os.path.join(settings.LOCAL_RUNS, job.job_uid)
                os.makedirs(job.local_dir, exist_ok=True)

            sshio.pull_results(job.job_uid, job.local_dir)
            rows = parser.parse_job_to_rows(job.local_dir) or []
            for r in rows:
                db.add(ResultRow(
                    job_uid=job.job_uid,
                    step=str(r.get("step", "")),
                    energy=str(r.get("energy", "")),
                    info=r.get("info", {}),
                ))
            job.status = "synced"
            db.commit()
    except Exception as e:
        print(f"_sync_job error for {job_uid}: {e}")
        db.rollback()
    finally:
        db.close()

# --- session APIs ---
# Avoid name collision if a function with same name already exists
if 'session_create' in globals():
    _session_create_name = 'session_create_v2'
else:
    _session_create_name = 'session_create'

def _make_session_create():
    @app.post('/session/create')
    def session_create(req: SessionCreate):
        db = SessionLocal()
        try:
            s = ChatSession(name=req.name, project=req.project)
            db.add(s); db.commit(); db.refresh(s)
            return {'session_uid': getattr(s, 'uid', None), 'name': s.name}
        finally:
            db.close()
    return session_create

globals()[_session_create_name] = _make_session_create()

@app.get('/session/list')
def session_list():
    db = SessionLocal()
    try:
        rec = db.query(ChatSession).order_by(ChatSession.id.desc()).all()
        return [{'uid': r.uid, 'name': r.name, 'project': r.project} for r in rec]
    finally:
        db.close()

# --- job APIs ---
@app.post('/job/create')
def job_create(req: JobCreate):
    db = SessionLocal()
    try:
        # Create Job
        job = Job(session_uid=req.session_uid, title=req.title, status='created')
        db.add(job); db.commit(); db.refresh(job)

        # Human-readable local directory
        title_slug = slugify(req.title or "job")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        shortuid = str(uuid.uuid4())[:8]
        human_dirname = f"{title_slug}_{ts}_{shortuid}"

        local_dir = os.path.join(settings.LOCAL_RUNS, human_dirname)
        os.makedirs(local_dir, exist_ok=True)

        # Write files
        with open(os.path.join(local_dir, 'POSCAR'), 'w') as f:
            f.write(req.poscar)
        with open(os.path.join(local_dir, 'ase-opt.py'), 'w') as f:
            f.write(req.ase_opt_py)
        with open(os.path.join(local_dir, 'ase.sh'), 'w') as f:
            f.write(req.ase_sh)

        # Remote ops using job_uid
        sshio.ensure_remote_dir(job.job_uid)
        sshio.push_files(job.job_uid, local_dir)
        qid = sshio.submit_qsub(job.job_uid)

        # DB backfill
        job.local_dir = local_dir
        job.remote_dir = f"{settings.REMOTE_BASE}/{job.job_uid}"
        job.slurm_or_pbs_id = qid
        job.status = 'submitted'
        db.commit()

        return {'job_uid': job.job_uid, 'pbs_id': qid}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get('/job/list')
def job_list(session_uid: str):
    db = SessionLocal()
    try:
        rec = db.query(Job).filter(Job.session_uid == session_uid).order_by(Job.id.desc()).all()
        out = []
        for r in rec:
            out.append({
                "job_uid": getattr(r, "job_uid", "") or "",
                "title": getattr(r, "title", "") or "",
                "status": getattr(r, "status", "") or "",
                "pbs_id": getattr(r, "slurm_or_pbs_id", "") or "",
                "local_dir": getattr(r, "local_dir", None),
                "remote_dir": getattr(r, "remote_dir", None),
            })
        return out
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post('/job/sync')
def job_sync(req: JobId):
    _sync_job(req.job_uid)
    return {'ok': True}

@app.get('/job/results')
def job_results(job_uid: str):
    db = SessionLocal()
    try:
        rec = db.query(ResultRow).filter(ResultRow.job_uid == job_uid).all()
        return [{'step': r.step, 'energy': r.energy, 'info': r.info} for r in rec]
    finally:
        db.close()

@app.get("/job/files")
def job_files(job_uid: str):
    db = SessionLocal()
    try:
        j = db.query(Job).filter(Job.job_uid == job_uid).first()
        if not j or not j.local_dir:
            return []
        root = pathlib.Path(j.local_dir)
        candidates = ["ase_results.csv","OUTCAR","CONTCAR","vasprun.xml","ase.out","ase.run.log","opt.log"]
        out = []
        for name in candidates:
            p = root / name
            if p.exists() and p.is_file():
                out.append({"path": name, "bytes": p.stat().st_size})
        return out
    finally:
        db.close()

@app.get("/job/file")
def job_file(job_uid: str, path: str):
    db = SessionLocal()
    try:
        j = db.query(Job).filter(Job.job_uid == job_uid).first()
        if not j or not j.local_dir:
            return Response(status_code=404)
        fp = os.path.join(j.local_dir, path)
        if not os.path.isfile(fp):
            return Response(status_code=404)
        with open(fp, "rb") as f:
            data = f.read()
        return Response(content=data, media_type="application/octet-stream")
    finally:
        db.close()

# --- batch create ---
@app.post("/batch/create")
def batch_create(payload=Body(...)):
    """Create a batch of jobs on a specific server."""
    db = SessionLocal()
    batch_uid = str(uuid.uuid4())

    svr = get_server(payload["server_name"])
    sh_tmpl = Template(svr["sh_template"])
    out = []

    try:
        for item in payload["items"]:
            title = item["title"]
            params = {**(payload.get("defaults") or {}), **(item.get("params") or {})}

            job = Job(
                session_uid=payload["session_uid"],
                title=title,
                server_name=payload["server_name"],
                scheduler=svr["scheduler"],
                batch_uid=batch_uid,
                params=params,
                status='created'
            )
            db.add(job); db.commit(); db.refresh(job)

            def _slug(s):
                return _re.sub(r"[^a-zA-Z0-9._-]+","-", (s or "job")).strip("-")
            local_dir = os.path.join(
                settings.LOCAL_RUNS,
                "{}_{}_{}".format(_slug(title), datetime.now().strftime('%Y%m%d-%H%M%S'), job.job_uid[:8])
            )
            os.makedirs(local_dir, exist_ok=True)

            open(os.path.join(local_dir,"POSCAR"),"w").write(item["poscar"])
            open(os.path.join(local_dir,"ase-opt.py"),"w").write(item["ase_opt_py"])
            ase_sh = item.get("ase_sh") or sh_tmpl.render(title=title, **params)
            open(os.path.join(local_dir,"ase.sh"),"w").write(ase_sh)

            sshio.ensure_remote_dir(job.job_uid, svr)
            sshio.push_files(job.job_uid, local_dir, svr)
            qid = sshio.submit(job.job_uid, svr)

            job.local_dir = local_dir
            job.remote_dir = f'{svr["remote_base"]}/{job.job_uid}'
            job.slurm_or_pbs_id = qid
            job.status = 'submitted'
            db.commit()

            out.append({"job_uid": job.job_uid, "pbs_or_slurm_id": qid})

        return {"batch_uid": batch_uid, "items": out}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# --- LLM Chat proxy ---
class ChatTurnReq(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: Optional[float] = 0.6
    system_prompt: Optional[str] = None
    extra_instructions: Optional[str] = None
    strict_json: Optional[bool] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

class ChatTurnResp(BaseModel):
    assistant_text: str

import traceback as _tb

@app.post("/chat/turn", response_model=ChatTurnResp)
def chat_turn(req: ChatTurnReq):
    try:
        if LLMAgent is None:
            raise RuntimeError("LLMAgent not available")
        text = LLMAgent.chat(
            messages=req.messages,
            model=req.model or "gpt-4o",
            temperature=req.temperature or 0.3,
            system_prompt=req.system_prompt,
            extra_instructions=req.extra_instructions,
            strict_json=bool(req.strict_json) if req.strict_json is not None else False,
            max_tokens=req.max_tokens,
            stop=req.stop,
        )
        return ChatTurnResp(assistant_text=text)
    except Exception as e:
        print(_tb.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

