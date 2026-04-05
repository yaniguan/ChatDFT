# server/main.py
import os
from dotenv import load_dotenv
load_dotenv()   # loads .env before any other import reads os.environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.chat.session_agent import router as session_router
from server.chat.intent_agent import router as intent_router
from server.chat.hypothesis_agent import router as hypothesis_router
from server.chat.plan_agent import router as plan_router
from server.chat.history_agent import router as history_router
from server.chat.knowledge_agent import router as knowledge_router
from server.execution.agent_routes import router as agent_router
from server.execution.task_routes import router as exec_tasks_router

# --- 放在 imports 后、routers include 之前 ---

import logging
from contextlib import asynccontextmanager

_log = logging.getLogger("chatdft")

@asynccontextmanager
async def lifespan(app):
    # 首次启动时创建缺失的表（开发/测试环境用）
    try:
        from server.db import Base, engine  # 已有
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        import logging
        logging.getLogger("chatdft").info("DB create_all done.")
    except ImportError as e:
        import logging
        logging.getLogger("chatdft").warning("DB create_all skipped: %s", e)

    # Daily literature update scheduler (runs once at startup + every 24 h)
    import asyncio as _asyncio

    async def _daily_literature_loop():
        # Wait a few seconds for the server to fully start before first run
        await _asyncio.sleep(15)
        while True:
            try:
                from server.chat.knowledge_agent import run_daily_update
                _log.info("Daily arXiv update starting...")
                await run_daily_update(trigger="scheduler")
                _log.info("Daily arXiv update complete.")
            except ImportError as _e:
                _log.warning("Daily update error: %s", _e)
            await _asyncio.sleep(24 * 3600)  # 24 hours

    _task = _asyncio.create_task(_daily_literature_loop())
    try:
        yield
    finally:
        _task.cancel()
        try:
            await _task
        except _asyncio.CancelledError:
            pass

# 用带 lifespan 的 FastAPI 覆盖原来的 app = FastAPI()
app = FastAPI(
    lifespan=lifespan,
    title="ChatDFT",
    version="0.3.0",
    description="Autonomous reaction pathway discovery via LLM-guided DFT",
)
_ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API v1 routers (all under /v1/ prefix for versioning) ---
from fastapi import APIRouter
v1 = APIRouter(prefix="/v1")
v1.include_router(session_router, tags=["chat"])
v1.include_router(intent_router, tags=["chat"])
v1.include_router(hypothesis_router, tags=["chat"])
v1.include_router(plan_router, tags=["chat"])
v1.include_router(history_router, tags=["chat"])
v1.include_router(knowledge_router, tags=["chat"])
v1.include_router(agent_router, tags=["agents"])
v1.include_router(exec_tasks_router, tags=["exec"])
app.include_router(v1)

# Backward-compatible: also mount at root for existing clients
app.include_router(session_router, tags=["chat"])
app.include_router(intent_router, tags=["chat"])
app.include_router(hypothesis_router, tags=["chat"])
app.include_router(plan_router, tags=["chat"])
app.include_router(history_router, tags=["chat"])
app.include_router(knowledge_router, tags=["chat"])
app.include_router(agent_router)
app.include_router(exec_tasks_router, tags=["exec"])

@app.get("/")
async def root():
    return {"ok": True, "service": "ChatDFT", "status": "running"}


@app.get("/health")
async def health():
    """Detailed health check with dependency status."""
    checks = {"service": "running"}
    # DB check
    try:
        from server.db import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as s:
            await s.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except ImportError as e:
        checks["database"] = f"error: {e}"
    # ML monitoring
    try:
        from server.mlops.monitoring import production_monitor
        checks["ml_health"] = production_monitor.health_status()
    except ImportError:
        checks["ml_health"] = "unavailable"
    # Model registry
    try:
        from server.mlops.model_registry import model_registry
        checks["models_registered"] = len(model_registry.list_all())
    except ImportError:
        checks["models_registered"] = 0
    # Feature store
    try:
        from server.feature_store.store import feature_store
        checks["features_registered"] = len(feature_store.list_features())
        checks["features_cached"] = len(feature_store._cache)
    except ImportError:
        checks["features_registered"] = 0

    all_ok = checks.get("database") == "ok"
    return {"ok": all_ok, "checks": checks}

from server.chat.records_agent import router as records_router
app.include_router(records_router)

from server.chat.analyze_agent import router as analyze_router
app.include_router(analyze_router, tags=["chat"])

from server.chat.qa_agent import router as qa_router
app.include_router(qa_router, tags=["qa"])

from server.chat.taskstate_routes import router as taskstate_router
app.include_router(taskstate_router)

from server.chat.structure_library_routes import router as strulib_router
app.include_router(strulib_router)

from server.science_routes import router as science_router
app.include_router(science_router)

# --- Scientist-facing API layer (zero-friction endpoints) ---
from server.api.model_api import router as model_api_router
app.include_router(model_api_router)

from server.api.preprocessor import router as preprocessor_router
app.include_router(preprocessor_router)

from server.api.one_click import router as one_click_router
app.include_router(one_click_router)



# ====== Added endpoints and helpers (async-ready, drop-in) ======
from fastapi import HTTPException, Response, Body, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import os, json, re as _re, uuid, pathlib
from datetime import datetime
from jinja2 import Template

# 项目内导入
try:
    from server.utils.openai_wrapper import LLMAgent  # 如果不存在，也能容错
except ImportError:
    LLMAgent = None

from server.db import engine
try:
    # 你的模型（按你项目结构，若在 server.models 则改这里）
    from server.db import Base, ChatSession, Job, FileAsset, ResultRow
except ImportError:
    from server.db import Base, ChatSession, Job, FileAsset, ResultRow  # type: ignore

from server.schema import SessionCreate, JobCreate, JobId
from server.settings import settings, get_server
from server import sshio, parser

# --- SQLAlchemy async 适配 ---
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

IS_ASYNC = isinstance(engine, AsyncEngine)
if not IS_ASYNC:
    # 你的 engine 明显是异步的（根据之前报错），若不是异步，这里给出提示防误用
    raise RuntimeError("Detected sync engine. Either switch engine to AsyncEngine or revert endpoints to sync Session.")

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
from typing import AsyncGenerator  # 或 AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as db:
        yield db

# DB 信息打印（异步引擎需通过 sync_engine 取 URL）
try:
    _log.info("DB engine: %s", engine.sync_engine.url)
except (ValueError, KeyError, TypeError) as _e:
    _log.debug("DB url print skipped: %s", _e)
# 建表建议交给 Alembic；如必须代码建表，可用：
# async def _init_models():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
# import asyncio; asyncio.run(_init_models())

# --- utils ---
_slug_re = _re.compile(r"[^a-zA-Z0-9._-]+")
def slugify(name: str, default: str = "job") -> str:
    name = (name or "").strip() or default
    name = name.replace(" ", "-")
    name = _slug_re.sub("-", name)
    name = _re.sub("-{2,}", "-", name).strip("-")
    return name or default

# --- session APIs ---
@app.post('/session/create')
async def session_create(req: SessionCreate, db: AsyncSession = Depends(get_db)):
    s = ChatSession(name=req.name, project=req.project)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return {'session_uid': getattr(s, 'uid', None), 'name': s.name}

@app.get('/session/list')
async def session_list(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(ChatSession).order_by(ChatSession.id.desc())
    )).scalars().all()
    return [
        {"uid": r.uid or "", "name": r.name or "", "project": r.project or ""}
        for r in rows
    ]

# --- job APIs ---
@app.post('/job/create')
async def job_create(req: JobCreate, db: AsyncSession = Depends(get_db)):
    try:
        # 1) 新建 Job（先拿到 job_uid）
        job = Job(session_uid=req.session_uid, title=req.title, status='created')
        db.add(job)
        await db.commit()
        await db.refresh(job)

        # 2) 本地人类可读目录
        title_slug = slugify(req.title or "job")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        shortuid = str(uuid.uuid4())[:8]
        human_dirname = f"{title_slug}_{ts}_{shortuid}"
        local_dir = os.path.join(settings.LOCAL_RUNS, human_dirname)
        os.makedirs(local_dir, exist_ok=True)

        # 3) 写文件
        open(os.path.join(local_dir, 'POSCAR'), 'w').write(req.poscar)
        open(os.path.join(local_dir, 'ase-opt.py'), 'w').write(req.ase_opt_py)
        open(os.path.join(local_dir, 'ase.sh'), 'w').write(req.ase_sh)

        # 4) 远端操作（仍用 job_uid）
        sshio.ensure_remote_dir(job.job_uid)
        sshio.push_files(job.job_uid, local_dir)
        qid = sshio.submit_qsub(job.job_uid)

        # 5) 回填数据库
        job.local_dir = local_dir
        job.remote_dir = f"{settings.REMOTE_BASE}/{job.job_uid}"
        job.slurm_or_pbs_id = qid
        job.status = 'submitted'
        await db.commit()

        return {'job_uid': job.job_uid, 'pbs_id': qid}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/job/list')
async def job_list(session_uid: str, db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(Job).where(Job.session_uid == session_uid).order_by(Job.id.desc())
    )).scalars().all()
    out = []
    for r in rows:
        out.append({
            "job_uid": getattr(r, "job_uid", "") or "",
            "title": getattr(r, "title", "") or "",
            "status": getattr(r, "status", "") or "",
            "pbs_id": getattr(r, "slurm_or_pbs_id", "") or "",
            "local_dir": getattr(r, "local_dir", None),
            "remote_dir": getattr(r, "remote_dir", None),
        })
    return out

@app.post('/job/sync')
async def job_sync(req: JobId, db: AsyncSession = Depends(get_db)):
    # 简化版：只触发拉取与解析（根据你之前的逻辑）
    # 若要完全异步化可把 ssh 调用放线程或队列里，这里保留同步调用以最小改动可用
    job = (await db.execute(
        select(Job).where(Job.job_uid == req.job_uid)
    )).scalars().first()
    if not job:
        return {'ok': True}

    # 轮询状态
    if job.slurm_or_pbs_id:
        state = sshio.poll_status(job.slurm_or_pbs_id)
        if state in ("queued", "running"):
            if job.status != state:
                job.status = state
                await db.commit()
        else:
            if job.status != "done":
                job.status = "done"
                await db.commit()

    # done 后拉取&解析
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
        await db.commit()

    return {'ok': True}

@app.get('/job/results')
async def job_results(job_uid: str, db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(ResultRow).where(ResultRow.job_uid == job_uid)
    )).scalars().all()
    return [{'step': r.step, 'energy': r.energy, 'info': r.info} for r in rows]

@app.get("/job/files")
async def job_files(job_uid: str, db: AsyncSession = Depends(get_db)):
    j = (await db.execute(
        select(Job).where(Job.job_uid == job_uid)
    )).scalars().first()
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

@app.get("/job/file")
async def job_file(job_uid: str, path: str, db: AsyncSession = Depends(get_db)):
    j = (await db.execute(
        select(Job).where(Job.job_uid == job_uid)
    )).scalars().first()
    if not j or not j.local_dir:
        return Response(status_code=404)
    fp = os.path.join(j.local_dir, path)
    if not os.path.isfile(fp):
        return Response(status_code=404)
    with open(fp, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="application/octet-stream")

# --- 批量创建 ---
@app.post("/batch/create")
async def batch_create(payload=Body(...), db: AsyncSession = Depends(get_db)):
    """
    payload = {
      "session_uid": "...",
      "server_name": "hoffman2",
      "defaults": {"ppn": 8, "walltime":"04:00:00"},
      "items": [
        {"title":"job1","poscar":"...", "ase_opt_py":"...", "params":{"ppn":4}},
        {"title":"job2","poscar":"...", "ase_opt_py":"...", "params":{"ppn":8}}
      ]
    }
    """
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
            db.add(job)
            await db.commit()
            await db.refresh(job)

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
            await db.commit()

            out.append({"job_uid": job.job_uid, "pbs_or_slurm_id": qid})

        return {"batch_uid": batch_uid, "items": out}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- LLM Chat 代理 ---
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

@app.post("/chat/turn", response_model=ChatTurnResp)
async def chat_turn(req: ChatTurnReq):
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
        _log.exception("Unexpected error in endpoint")
        raise HTTPException(status_code=500, detail=str(e))
# ====== End patch ======