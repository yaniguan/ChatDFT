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
except Exception:
    LLMAgent = None

from server.db import engine
try:
    # 你的模型（按你项目结构，若在 server.models 则改这里）
    from server.db import Base, ChatSession, Job, FileAsset, ResultRow
except Exception:
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
    print("[DB] using:", engine.sync_engine.url)
except Exception as _e:
    print("[DB] url print skipped:", _e)
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
        import traceback; print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
# ====== End patch ======