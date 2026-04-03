# server/execution/agent_routes.py
# -*- coding: utf-8 -*-
"""
微路由：供前端“单步 Run / Prepare-only”使用。
路径前缀：/agent/...

功能概览
- StructureAgent：bulk/slab/adsorb/modify
- ParametersAgent：VASP/QE 输入生成（INCAR/KPOINTS 或 *.in）
- HPCAgent：渲染脚本 → rsync → 提交（可选等待 & 拉回）
- PostAnalysisAgent：本地结果聚合（可选）
- Prepare-only：submit=False 时仅生成本地输入与结构，返回预览，不提交 HPC

请求体可选字段：
- task: Dict（等价 plan 里单个 task，至少带 params.payload）
- agent: str（如 structure.relax_slab / adsorption.scan / electronic.dos / neb.run / run_dft / meta.clarify / knowledge.search）
- engine: "vasp"|"qe"（若没给，从 task.params.payload.engine 取，默认 vasp）
- cluster: "hoffman2"|...（在你的 cluster 配置中定义）
- submit: bool = True      # 提交到 HPC；False=仅本地生成
- wait: bool = False       # 是否阻塞等待队列/运行完成
- fetch: bool = True       # 是否从 HPC 拉回输出（submit=True 时有效）
- poll: int = 60           # wait 的轮询秒
- fetch_filters: [ ... ]   # fetch 时的文件过滤
- do_post: bool = True     # 拉回后是否本地 PostAnalysis
- job_name: str            # HPC 脚本 job 名称
- run_id: int              # 事件记录用
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path
import tempfile
import json
import os
import re

from fastapi import APIRouter, Request

# 执行层（与你的现有文件匹配）
from .structure_agent import StructureAgent
from .parameters_agent import ParametersAgent
from .hpc_agent import HPCAgent
from .post_analysis_agent import PostAnalysisAgent

router = APIRouter(prefix="/agent", tags=["agent"])

# ---------------------------------------------------------------------------
# Structure-library auto-save helper
# ---------------------------------------------------------------------------
async def _save_to_library(result: dict, session_id: Optional[int] = None) -> None:
    """Fire-and-forget: persist a built structure to StructureLibrary."""
    try:
        from server.db import StructureLibrary, get_session as _get_session
        from sqlalchemy.ext.asyncio import AsyncSession
        from server.db import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            row = StructureLibrary(
                session_id     = session_id,
                structure_type = result.get("structure_type", "surface"),
                label          = result.get("label", ""),
                formula        = result.get("formula"),
                smiles         = result.get("smiles"),
                description    = result.get("description"),
                ase_code       = result.get("ase_code"),
                poscar         = result.get("poscar"),
                plot_png_b64   = result.get("plot_png_b64"),
                meta           = {k: result.get(k) for k in
                                  ("element","crystal_system","surface_type","size",
                                   "vacuum","cell_size","site_type","rotation","height","n_atoms")
                                  if result.get(k) is not None},
            )
            db.add(row)
            await db.commit()
    except (ValueError, KeyError, TypeError):
        pass  # non-fatal

# ============================ 事件（可选） ============================
try:
    from server.execution.utils.events import post_event  # 如果没有就用兜底
except ImportError:
    async def post_event(_payload):  # 不阻塞主流程
        return

async def _emit(run_id: Optional[int], step_id: Optional[int], phase: str, payload: Optional[dict] = None):
    """phase: queued | pre_submit | running | done | error | skipped"""
    try:
        await post_event({
            "run_id": run_id,
            "step_id": step_id,
            "phase": phase,
            "payload": payload or {}
        })
    except Exception:
        pass

# ============================ 工具函数 ============================

async def _json(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _ok(msg: str, extra: Dict[str, Any] | None = None):
    payload = {"ok": True, "message": msg}
    if extra: payload.update(extra)
    return payload

def _fail(detail: str, status: int = 400):
    return {"ok": False, "detail": detail, "status_code": status}

def _make_job_dir(task: Dict[str, Any]) -> Path:
    # 更安全/可读的 job 目录名： 00_Name（去除特殊字符）
    tid = int(task.get("id", 0))
    name = (task.get("name") or "Task")
    name = re.sub(r"[^\w\-\.]+", "_", name).strip("_") or "Task"
    base = tempfile.mkdtemp(prefix="chatdft_")
    job_dir = Path(base) / f"{tid:02d}_{name}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir

def _norm_agent(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "clarify": "meta.clarify",
        "scope_proposal": "meta.scope",
        "lit_search": "knowledge.search",
        "builder": "structure.relax_slab",
        "intermediate_builder": "structure.intermediates",
        "adsorption_builder": "adsorption.scan",
        "postprocess_energy": "post.energy",
        "postprocess_coads": "adsorption.co",
        "electronic_props": "electronic.dos",
    }
    return aliases.get(n, n or "unknown")

def _extract_engine(task: Dict[str, Any], fallback: str = "vasp") -> str:
    p = ((task.get("params") or {}).get("payload") or {})
    return (p.get("engine") or fallback).lower()

def _ensure_json(path: Path, obj: Any):
    try:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    except (ValueError, KeyError, TypeError):
        pass

# POSCAR 轻量预览（不依赖 pymatgen/ase）
def _poscar_preview(p: Path) -> dict:
    try:
        t = p.read_text().splitlines()
        if len(t) < 8:
            return {"ok": False, "error": "too short POSCAR"}
        title = t[0].strip()
        scale = float(t[1].strip())
        # 3×3 晶格
        lat = []
        for i in range(2, 5):
            parts = [float(x) for x in t[i].split()]
            lat.append(parts[:3])
        elems = [x for x in t[5].split() if x]
        nums  = [int(x) for x in t[6].split() if x]
        natom = sum(nums) if nums else None
        return {"ok": True, "title": title, "scale": scale, "lattice": lat, "elements": elems, "counts": nums, "n_atoms": natom}
    except (ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": str(e)}

def _make_sge_job_sh(script_filename: str, ntasks: int = 32,
                     walltime: str = "24:00:00") -> str:
    """
    Generate a Hoffman2 SGE job.sh that runs an ASE Python script.
    walltime format: HH:MM:SS
    """
    return f"""\
#!/bin/bash -f
#$ -cwd
#$ -o $JOB_ID.log
#$ -e $JOB_ID.err
#$ -pe dc* {ntasks}
#$ -l h_data=4G,h_vmem=16G,h_rt={walltime}

source /u/local/Modules/default/init/modules.sh
module add intel/17.0.7
export PYTHONPATH=/u/home/y/yaniguan/miniconda3/envs/ase/:$PYTHONPATH
export PATH=/u/home/y/yaniguan/miniconda3/envs/ase/bin:$PATH

export VASP_PP_PATH=$HOME/vasp/mypps

export OMP_NUM_THREAD=1
export I_MPI_COMPATIBILITY=4

export VASP_COMMAND='mpirun -np ${{NSLOTS}} ~/vasp_std_vtst_sol'
python {script_filename}
echo "run complete on `hostname`: `date` `pwd`" >> ~/job.log
"""


def _first_file(d: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = d / n
        if p.exists(): return p
    return None

# ============================ 执行适配 ============================

def _run_structure(task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    sagent = StructureAgent()
    return sagent.build(task, job_dir)

def _run_parameters(task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    pagent = ParametersAgent()
    return pagent.generate(task, job_dir)

def _run_hpc(task: Dict[str, Any], job_dir: Path, *, cluster: str, engine: str,
             submit: bool, wait: bool, fetch: bool, poll: int, fetch_filters: list | None,
             job_name: str | None) -> Dict[str, Any]:
    step_ctx = {
        "name": job_name or (task.get("name") or f"{engine}_job"),
        "engine": engine,
        "template_vars": ((task.get("params") or {}).get("payload") or {}).get("template_vars") or {},
        "ntasks": ((task.get("params") or {}).get("payload") or {}).get("ntasks"),
        "walltime": ((task.get("params") or {}).get("payload") or {}).get("walltime"),
    }
    hpc = HPCAgent(cluster=cluster, dry_run=not submit and False, sync_back=fetch)
    script = hpc.prepare_script(step_ctx, job_dir)

    job_id = "NOT_SUBMITTED"
    if submit:
        job_id = hpc.submit(job_dir)
        if wait:
            hpc.wait(job_id, poll=poll)
        if fetch:
            hpc.fetch_outputs(job_dir, filters=fetch_filters)

    return {"script": str(script), "job_id": job_id}

def _run_post(job_dir: Path) -> Dict[str, Any]:
    post = PostAnalysisAgent()
    try:
        meta = post.analyze(job_dir)
        return {"ok": True, "post": meta}
    except Exception as e:
        return {"ok": False, "error": f"post-analysis failed: {e}"}

async def _run_knowledge_local(task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    import httpx
    base = os.getenv("CHATDFT_BACKEND", "http://127.0.0.1:8000").rstrip("/")
    payload = {
        "query":  (((task.get("params") or {}).get("payload") or {}).get("query")) or "",
        "intent": (((task.get("params") or {}).get("payload") or {}).get("intent")) or {},
        "limit":  int((((task.get("params") or {}).get("payload") or {}).get("limit")) or 10),
        "fast":   True,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{base}/chat/knowledge", json=payload)
            r.raise_for_status()
            kb = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        _ensure_json(job_dir / "knowledge_error.json", {"error": str(e), "payload": payload})
        return {"ok": False, "error": f"knowledge.search failed: {e}"}

    _ensure_json(job_dir / "papers.json", kb.get("records") or [])
    (job_dir / "background.txt").write_text(kb.get("background") or "")
    (job_dir / "summary.txt").write_text(kb.get("result") or "ok")
    return {"ok": True, "n": len(kb.get("records") or []), "files": ["papers.json","background.txt","summary.txt"]}

# ============================ 统一流水线 ============================

async def _pipeline(agent_name: str, task: Dict[str, Any], opts: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    - knowledge.search: 仅拉文献落盘
    - meta.*: 仅落 meta.json
    - structure.* / adsorption.* / electronic.* / neb.* / run_dft:
        prepare-only（submit=False）→ 早返回预览
        submit=True → Structure → Parameters → HPC（submit/wait/fetch）
    - post.analysis: 本地聚合
    """
    agent = _norm_agent(agent_name)
    job_dir = _make_job_dir(task)

    # 事件元数据
    run_id  = opts.get("run_id") or ((task.get("meta") or {}).get("run_id"))
    step_id = task.get("id")

    # 入队事件
    await _emit(run_id, step_id, "queued", {"agent": agent, "task": task, "job_dir": str(job_dir)})

    try:
        # 选项
        engine  = (opts.get("engine") or _extract_engine(task, "vasp")).lower()
        cluster = (opts.get("cluster") or "hoffman2")
        submit  = bool(opts.get("submit", True))
        wait    = bool(opts.get("wait", False))
        fetch   = bool(opts.get("fetch", True))
        poll    = int(opts.get("poll", 60))
        fetch_filters = opts.get("fetch_filters") or None
        do_post = bool(opts.get("do_post", True))
        job_name = opts.get("job_name")

        out: Dict[str, Any] = {"agent": agent, "job_dir": str(job_dir)}

        # 默认 fetch 过滤（前端传了就覆盖）
        DEFAULT_FETCH_FILTERS = [
            "OUTCAR", "vasprun.xml", "OSZICAR", "stdout*", "stderr*",
            "CONTCAR", "DOSCAR", "EIGENVAL", "PROCAR",
            "CHGCAR", "AECCAR*", "ELFCAR", "ACF.dat", "PARCHG*"
        ]
        fetch_filters = fetch_filters or DEFAULT_FETCH_FILTERS

        # 0) meta / knowledge
        if agent in {"meta.clarify", "meta.scope"}:
            (job_dir / "meta.json").write_text(json.dumps(task.get("params") or {}, indent=2))
            out.update({"status": "done(meta)"})
            await _emit(run_id, step_id, "done", {"status": out["status"], "job_dir": str(job_dir)})
            return True, out

        if agent == "knowledge.search":
            k = await _run_knowledge_local(task, job_dir)
            status = "done(knowledge)" if k.get("ok") else "error(knowledge)"
            out.update({"status": status, **k})
            phase = "done" if k.get("ok") else "error"
            await _emit(run_id, step_id, phase, {"status": status, "job_dir": str(job_dir), **k})
            return k.get("ok", False), out

        # 1) structure / adsorption / electronic / neb / run_dft
        if agent.startswith("structure.") or agent.startswith("adsorption.") or agent in {
            "electronic.dos", "neb.run", "run_dft", "post.energy"
        }:
            sres = _run_structure(task, job_dir)
            out["structure"] = sres

            pres = _run_parameters(task, job_dir)
            out["parameters"] = pres

            # ====== 早返回（Prepare-only）======
            if not submit:
                poscar = _first_file(job_dir, ["slab.POSCAR", "ads.POSCAR", "bulk.POSCAR", "structure.POSCAR", "POSCAR"])
                preview = _poscar_preview(poscar) if poscar else {"ok": False, "error": "no POSCAR found"}
                files = sorted([p.name for p in job_dir.iterdir() if p.is_file()])
                out.update({
                    "status": "done(prepared)",
                    "prepared": True,
                    "files": files,
                    "poscar_file": poscar.name if poscar else None,
                    "preview": preview,
                })
                await _emit(run_id, step_id, "done", {"status": "done(prepared)", "job_dir": str(job_dir), "preview": preview})
                return True, out
            # ====== 早返回到此为止 ======

            # 提交 HPC
            await _emit(run_id, step_id, "pre_submit", {
                "engine": engine, "cluster": cluster, "job_name": job_name, "job_dir": str(job_dir)
            })

            hres = _run_hpc(task, job_dir, cluster=cluster, engine=engine,
                            submit=submit, wait=wait, fetch=fetch, poll=poll,
                            fetch_filters=fetch_filters, job_name=job_name)
            out["hpc"] = hres

            job_id = hres.get("job_id", "NOT_SUBMITTED")
            if submit:
                await _emit(run_id, step_id, "running", {"job_id": job_id, "job_dir": str(job_dir)})

            if do_post and (wait or fetch):
                post_res = _run_post(job_dir)
                out["post"] = post_res

            status = "done(hpc)" if submit else "done(prepared)"
            out["status"] = status
            await _emit(run_id, step_id, "done", {
                "status": status, "job_id": job_id, "job_dir": str(job_dir), "has_post": bool(out.get("post"))
            })
            return True, out

        # 2) post-only
        if agent == "post.analysis":
            post_res = _run_post(job_dir)
            ok = bool(post_res.get("ok"))
            status = "done(post)" if ok else "error(post)"
            out.update({"status": status, "post": post_res})
            await _emit(run_id, step_id, "done" if ok else "error", {
                "status": status, "job_dir": str(job_dir), **post_res
            })
            return ok, out

        # 未知 agent
        out["status"] = f"skipped (unknown agent: {agent_name})"
        await _emit(run_id, step_id, "skipped", {"status": out["status"], "job_dir": str(job_dir)})
        return True, out

    except Exception as e:
        await _emit(run_id, step_id, "error", {"error": str(e), "agent": agent, "job_dir": str(job_dir)})
        return False, {"agent": agent, "job_dir": str(job_dir), "status": f"error: {e}"}

# ============================ 路由 ============================

@router.post("/run")
async def agent_run(request: Request):
    body = await _json(request)
    # 合理默认：打开 post + fetch，并给出常用过滤
    body.setdefault("do_post", True)
    body.setdefault("fetch", True)
    body.setdefault("fetch_filters", [
        "OUTCAR","vasprun.xml","OSZICAR","stdout*","stderr*",
        "CONTCAR","DOSCAR","EIGENVAL","PROCAR","CHGCAR","AECCAR*","ELFCAR","ACF.dat","PARCHG*"
    ])
    agent = body.get("agent") or ""
    task  = body.get("task")  or {}
    ok, res = await _pipeline(agent, task, body)
    # 语义上执行失败也返回 200 给前端（便于界面展示错误）
    return _ok("Pipeline finished.", res) if ok else _fail(res.get("status","unknown error"), 200)

@router.post("/clarify")
async def agent_clarify(request: Request):
    body = await _json(request)
    return _ok("Scope clarified.", {"echo": body})

@router.post("/scope_proposal")
async def agent_scope_proposal(request: Request):
    body = await _json(request)
    n = int(body.get("n_options", 3) or 3)
    options = [f"Option {i+1}: focus on {body.get('focus','kinetics')}" for i in range(n)]
    return _ok("Scope proposed.", {"options": options, "echo": body})

@router.post("/knowledge.search")
async def agent_knowledge_search(request: Request):
    body = await _json(request)
    task = {"id": body.get("id", 0), "name": "Literature scan", "params": {"payload": {
        "query": body.get("query") or (body.get("payload") or {}).get("query") or "",
        "intent": body.get("intent") or (body.get("payload") or {}).get("intent") or {},
        "limit": int(body.get("limit") or 10),
    }}}
    ok, res = await _pipeline("knowledge.search", task, {"submit": False})
    return _ok("Literature fetched.", res) if ok else _fail(res.get("status","knowledge failed"), 200)

@router.post("/structure.relax_slab")
async def agent_structure_relax_slab(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("structure.relax_slab", task, body)
    return _ok("Slab/bulk/chain/crystal prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/structure.relax_adsorbate")
async def agent_structure_relax_adsorbate(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("structure.relax_adsorbate", task, body)
    return _ok("Adsorbate prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/adsorption.scan")
async def agent_adsorption_scan(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("adsorption.scan", task, body)
    return _ok("Adsorption site scan prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/adsorption.co")
async def agent_adsorption_co(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("adsorption.co", task, body)
    return _ok("Co-adsorption prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/neb.run")
async def agent_neb_run(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("neb.run", task, body)
    return _ok("NEB prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/electronic.dos")
async def agent_electronic_dos(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("electronic.dos", task, body)
    return _ok("DOS/PDOS/Bader prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/post.analysis")
async def agent_post_analysis(request: Request):
    body = await _json(request)
    if body.get("job_dir"):
        try:
            post = PostAnalysisAgent()
            meta = post.analyze(Path(body["job_dir"]))
            return _ok("Post analysis finished.", {"post": meta})
        except (ValueError, KeyError, TypeError) as e:
            return _fail(f"post-analysis failed: {e}", 200)
    task = body.get("task") or {}
    ok, res = await _pipeline("post.analysis", task, body)
    return _ok("Post analysis finished.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/report_plot")
async def agent_report_plot(request: Request):
    body = await _json(request)
    return _ok("Figures generated.", {"figs": ["plot1.svg", "plot2.svg"], "echo": body})


# ============================ Structure interactive API ============================

@router.post("/structure/find_sites")
async def agent_structure_find_sites(request: Request):
    """Find adsorption sites for a given POSCAR string."""
    body = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content") or ""
    if not poscar:
        return _fail("poscar field is required", 400)
    height = float(body.get("height", 2.0))
    sagent = StructureAgent()
    result = sagent.find_sites(poscar, height=height)
    return _ok("Sites found.", result) if result.get("ok") else _fail(result.get("error", "failed"), 200)


@router.post("/structure/place_adsorbate")
async def agent_structure_place_adsorbate(request: Request):
    """Place one adsorbate at a specific site index."""
    body = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content") or ""
    adsorbate = body.get("adsorbate") or body.get("molecule") or "CO"
    site_index = int(body.get("site_index", 0))
    height = float(body.get("height", 2.0))
    if not poscar:
        return _fail("poscar field is required", 400)
    sagent = StructureAgent()
    result = sagent.place_ads(poscar, adsorbate, site_index=site_index, height=height)
    return _ok("Adsorbate placed.", result) if result.get("ok") else _fail(result.get("error", "failed"), 200)


@router.post("/structure/generate_configs")
async def agent_structure_generate_configs(request: Request):
    """Generate multiple adsorption configurations.
    Accepts either:
      - surface poscar + mol_poscar  → use generate_ads_from_poscars (molecule from POSCAR)
      - surface poscar + adsorbate   → use original generate_ads_configs (simple atom/fragment)
    """
    body = await _json(request)
    poscar   = body.get("poscar") or body.get("poscar_content") or ""
    mol_poscar = body.get("mol_poscar") or ""
    adsorbate  = body.get("adsorbate") or body.get("molecule") or "CO"
    max_configs = int(body.get("max_configs", 4))
    height = float(body.get("height", 2.0))

    if not poscar:
        return _fail("poscar field is required", 400)

    sagent = StructureAgent()

    if mol_poscar:
        # Full molecule from POSCAR — use advanced placement
        from .structure_agent import generate_ads_from_poscars
        result = generate_ads_from_poscars(
            surface_poscar=poscar,
            mol_poscar=mol_poscar,
            max_configs=max_configs,
            height=height,
        )
    else:
        # Simple adsorbate fragment
        result = sagent.generate_ads_configs(poscar, adsorbate, max_configs)

    if result.get("ok"):
        import asyncio
        session_id = body.get("session_id")
        for cfg in result.get("configs", []):
            asyncio.ensure_future(_save_to_library(cfg, session_id=session_id))
        return _ok("Configurations generated.", result)
    else:
        return _fail(result.get("error", "failed"), 200)


@router.post("/structure/build_slab")
async def agent_structure_build_slab(request: Request):
    """Build a slab directly from parameters (no HPC submission)."""
    body = await _json(request)
    task = {
        "id": 0,
        "name": body.get("element", "Cu"),
        "params": {"payload": body},
    }
    sagent = StructureAgent()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = sagent.build(task, Path(tmp))
    return _ok("Slab built.", result) if result.get("ok") else _fail(result.get("error", "failed"), 200)


@router.post("/structure/build_molecule")
async def agent_structure_build_molecule(request: Request):
    """
    Build a gas-phase molecule from PubChem using SMILES.
    Body: {"smiles": "CCCC", "label": "C4H10", "cell_size": 20.0}
    Returns: poscar, viz, formula, n_atoms
    """
    body = await _json(request)
    smiles     = body.get("smiles") or body.get("formula") or "CCCC"
    label      = body.get("label") or body.get("name") or ""
    cell_size  = float(body.get("cell_size", 20.0))
    sagent = StructureAgent()
    result = sagent.build_molecule(smiles=smiles, label=label, cell_size=cell_size)
    if result.get("ok"):
        import asyncio
        asyncio.ensure_future(_save_to_library(result, session_id=body.get("session_id")))
    return _ok("Molecule built.", result) if result.get("ok") else _fail(result.get("error", "failed"), 200)


@router.post("/structure/build_surface")
async def agent_structure_build_surface(request: Request):
    """
    Build a clean metal surface using ASE fcc111/bcc110/hcp0001 pattern.
    Body: {"element":"Ag","surface_type":"111","nx":4,"ny":4,"nlayers":4,"vacuum":10.0}
    Returns: poscar, viz, formula, n_atoms, label
    """
    body = await _json(request)
    element      = body.get("element", "Cu")
    surface_type = body.get("surface_type") or body.get("facet", "111")
    nx           = int(body.get("nx", body.get("supercell_x", 4)))
    ny           = int(body.get("ny", body.get("supercell_y", 4)))
    nlayers      = int(body.get("nlayers", body.get("layers", 3)))
    vacuum       = float(body.get("vacuum", 10.0))
    fix_bottom   = bool(body.get("fix_bottom", True))
    sagent = StructureAgent()
    result = sagent.build_surface(
        element=element, surface_type=surface_type,
        nx=nx, ny=ny, nlayers=nlayers, vacuum=vacuum, fix_bottom=fix_bottom,
    )
    if result.get("ok"):
        import asyncio
        asyncio.ensure_future(_save_to_library(result, session_id=body.get("session_id")))
    return _ok("Surface built.", result) if result.get("ok") else _fail(result.get("error", "failed"), 200)


@router.post("/structure/add_layer")
async def agent_structure_add_layer(request: Request):
    """Add one atomic layer on top. Body: {"poscar": "..."}"""
    body  = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content", "")
    if not poscar:
        return _fail("poscar required", 400)
    from .structure_agent import slab_add_layer
    result = slab_add_layer(poscar)
    return _ok("Layer added.", result) if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/delete_layer")
async def agent_structure_delete_layer(request: Request):
    """Delete topmost atomic layer. Body: {"poscar": "..."}"""
    body  = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content", "")
    if not poscar:
        return _fail("poscar required", 400)
    from .structure_agent import slab_delete_layer
    result = slab_delete_layer(poscar)
    return _ok("Layer deleted.", result) if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/set_vacuum")
async def agent_structure_set_vacuum(request: Request):
    """Set vacuum thickness. Body: {"poscar": "...", "vacuum": 20.0}"""
    body   = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content", "")
    vacuum = float(body.get("vacuum", 15.0))
    if not poscar:
        return _fail("poscar required", 400)
    from .structure_agent import slab_set_vacuum
    result = slab_set_vacuum(poscar, vacuum=vacuum)
    return _ok("Vacuum set.", result) if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/dope")
async def agent_structure_dope(request: Request):
    """
    Dope surface by substituting host atoms with dopant.
    Body: {"poscar":"...", "host_element":"Pt", "dopant_element":"Co",
           "n_dopants":1, "site":"surface"}
    """
    body   = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content", "")
    if not poscar:
        return _fail("poscar required", 400)
    from .structure_agent import slab_dope
    result = slab_dope(
        poscar,
        host_element    = body.get("host_element", "Pt"),
        dopant_element  = body.get("dopant_element", "Co"),
        n_dopants       = int(body.get("n_dopants", 1)),
        site            = body.get("site", "surface"),
    )
    return _ok("Doping applied.", result) if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/make_symmetric")
async def agent_structure_make_symmetric(request: Request):
    """
    Build an inversion-symmetric slab for GC-DFT calculations.
    Body: {"poscar": "...", "vacuum": 20.0}
    Returns a slab with identical top/bottom terminations.
    Base layer is the mirror plane (3-D inversion through base layer center).
    """
    body   = await _json(request)
    poscar = body.get("poscar") or body.get("poscar_content", "")
    vacuum = float(body.get("vacuum", 20.0))
    if not poscar:
        return _fail("poscar required", 400)
    from .structure_agent import slab_make_symmetric
    result = slab_make_symmetric(poscar, vacuum=vacuum)
    return _ok("Symmetric slab built.", result) if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/build_interface")
async def agent_structure_build_interface(request: Request):
    """
    Build a heterogeneous interface (e.g. Pt thin film on Cu substrate).
    Body: {"poscar_a": "...(substrate)...", "poscar_b": "...(film)...",
           "vacuum": 15.0, "interface_gap": 2.2, "strain_a": false}
    """
    body       = await _json(request)
    poscar_a   = body.get("poscar_a") or body.get("poscar", "")
    poscar_b   = body.get("poscar_b") or body.get("poscar_film", "")
    if not poscar_a or not poscar_b:
        return _fail("poscar_a (substrate) and poscar_b (film) are both required", 400)
    from .structure_agent import build_interface
    result = build_interface(
        poscar_a,
        poscar_b,
        vacuum        = float(body.get("vacuum", 15.0)),
        interface_gap = float(body.get("interface_gap", 2.2)),
        strain_a      = bool(body.get("strain_a", False)),
    )
    return _ok("Interface built.", result) if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/generate_neb_images")
async def agent_structure_generate_neb_images(request: Request):
    """
    Generate NEB interpolated images between IS and FS.
    Body: {"is_poscar": "...", "fs_poscar": "...", "n_images": 6, "method": "linear"}
    Returns list of {index, label, poscar, plot_png_b64} for all images including IS and FS.
    """
    body      = await _json(request)
    is_poscar = body.get("is_poscar") or body.get("poscar_is", "")
    fs_poscar = body.get("fs_poscar") or body.get("poscar_fs", "")
    if not is_poscar or not fs_poscar:
        return _fail("is_poscar and fs_poscar are both required", 400)
    from .structure_agent import generate_neb_images
    result = generate_neb_images(
        is_poscar,
        fs_poscar,
        n_images = int(body.get("n_images", 6)),
        method   = body.get("method", "linear"),
    )
    return _ok(f"Generated {result.get('n_images','?')} NEB images.", result) \
        if result.get("ok") else _fail(result.get("error","failed"), 200)


@router.post("/structure/build_complex")
async def agent_build_complex(request: Request):
    """
    Build a coordination compound: metal center + monodentate ligands.
    Body: {"metal": "Cu", "ligand": "NH3", "n_coord": 4, "geometry": "square_planar",
           "bond_length": 2.0, "cell_size": 15.0, "session_id": null}
    """
    body       = await _json(request)
    session_id = body.get("session_id")
    from .structure_agent import build_complex as _build_complex
    result = _build_complex(
        metal       = body.get("metal", "Cu"),
        ligand      = body.get("ligand", "NH3"),
        n_coord     = int(body.get("n_coord", 4)),
        geometry    = body.get("geometry", "square_planar"),
        bond_length = float(body.get("bond_length", 2.0)),
        cell_size   = float(body.get("cell_size", 15.0)),
    )
    if not result.get("ok"):
        return _fail(result.get("error", "build_complex failed"), 200)
    import asyncio
    asyncio.ensure_future(_save_to_library(result, session_id))
    return _ok(result.get("description", "Coordination complex built."), result)


@router.post("/structure/deprotonate")
async def agent_structure_deprotonate(request: Request):
    """
    Remove n_remove hydrogen atoms from a molecule POSCAR.
    Used to generate CO2RR / dehydrogenation intermediates from a parent molecule.

    Body:
      poscar    : str  — POSCAR content of the parent molecule
      n_remove  : int  — number of H atoms to remove (default 1)
      site      : str  — 'surface' | 'terminal' | 'random' (default 'terminal')

    Returns: poscar of the deprotonated molecule + formula metadata.
    """
    body     = await _json(request)
    poscar   = body.get("poscar", "")
    n_remove = int(body.get("n_remove", 1))
    site     = body.get("site", "terminal")
    if not poscar:
        return _fail("poscar is required", 400)
    sagent = StructureAgent()
    result = sagent.deprotonate_molecule(poscar=poscar, n_remove=n_remove, site=site)
    if not result.get("ok"):
        return _fail(result.get("detail", "deprotonate failed"), 200)
    return _ok(
        f"Removed {n_remove} H: {result['formula_original']} → {result['formula_deprotonated']}",
        result,
    )


@router.post("/generate_script")
async def agent_generate_script(request: Request):
    """
    Generate an ASE calculation script.
    Body: {"calc_type": "geo"|"freq"|"neb"|"gcdft"|"dos"|"band"|"elf"|
                        "static"|"bader"|"cdd"|"work_function"|"cohp"|"surface"|"molecule",
           "system": {...}, "params": {...}}
    Returns: {"script": "...", "filename": "ase-geo.py"}
    """
    body = await _json(request)
    calc_type = body.get("calc_type", "geo")
    system    = body.get("system") or {}
    params    = body.get("params") or {}
    try:
        from .ase_scripts import generate_script, script_filename
        script   = generate_script(calc_type, system, params)
        filename = script_filename(calc_type)
        return _ok("Script generated.", {"script": script, "filename": filename, "calc_type": calc_type})
    except ImportError as e:
        return _fail(f"Script generation failed: {e}", 200)


@router.post("/incar_preset")
async def agent_incar_preset(request: Request):
    """
    Return the VASP INCAR parameter preset for a given calc_type.
    Body: {"calc_type": "static"|"dos"|"band"|"elf"|"bader"|"cdd"|"work_function"|"cohp"}
    Returns: {"incar": {...}, "incar_string": "...", "comment": "...", "suggested_kpoints": "..."}
    """
    body = await _json(request)
    calc_type = body.get("calc_type", "static")
    try:
        from .vasp_incar import get_incar, get_comment, incar_to_string, suggested_kpoints
        incar   = get_incar(calc_type)
        comment = get_comment(calc_type)
        kpts    = suggested_kpoints(calc_type)
        return _ok("INCAR preset returned.", {
            "calc_type":        calc_type,
            "incar":            incar,
            "incar_string":     incar_to_string(incar),
            "comment":          comment,
            "suggested_kpoints": kpts,
        })
    except Exception as e:
        return _fail(f"INCAR preset failed: {e}", 200)


@router.post("/htp/generate")
async def agent_htp_generate(request: Request):
    """
    Generate diverse structures for NNP training dataset.
    Body: {"base_structures": [{"poscar": str, "label": str}], "strategy": "rattle",
           "n_total": 1000, "db_path": "htp_dataset.db", ...strategy_kwargs}
    Returns: {ok, n_generated, db_path, stats}
    """
    body = await _json(request)
    base_structures = body.get("base_structures", [])
    if not base_structures:
        return _fail("base_structures is required (list of {poscar, label})", 400)
    try:
        from .htp_agent import generate_htp_dataset
        result = generate_htp_dataset(
            base_structures=base_structures,
            strategy=body.get("strategy", "rattle"),
            n_total=int(body.get("n_total", 1000)),
            db_path=body.get("db_path", "htp_dataset.db"),
            **{k: v for k, v in body.items()
               if k not in ("base_structures", "strategy", "n_total", "db_path")},
        )
        if not result.get("ok"):
            return _fail(result.get("error", "HTP generation failed"), 200)
        return _ok(
            f"Generated {result['n_generated']} structures (strategy={body.get('strategy','rattle')}).",
            result,
        )
    except (ValueError, KeyError, TypeError) as e:
        return _fail(f"HTP generate error: {e}", 200)


@router.post("/htp/stats")
async def agent_htp_stats(request: Request):
    """
    Get dataset statistics.
    Body: {"db_path": "htp_dataset.db"}
    """
    body = await _json(request)
    db_path = body.get("db_path", "htp_dataset.db")
    try:
        from .htp_agent import HTPDataset
        dataset = HTPDataset(db_path=db_path)
        stats = dataset.stats()
        return _ok("HTP dataset stats.", {"stats": stats, "db_path": db_path})
    except ImportError as e:
        return _fail(f"HTP stats error: {e}", 200)


@router.post("/htp/export")
async def agent_htp_export(request: Request):
    """
    Export extXYZ training set from HTP database.
    Body: {"db_path": "htp_dataset.db", "output_path": "training.xyz", "only_done": true}
    """
    body = await _json(request)
    db_path = body.get("db_path", "htp_dataset.db")
    output_path = body.get("output_path", "training.xyz")
    only_done = bool(body.get("only_done", True))
    try:
        from .htp_agent import HTPDataset
        dataset = HTPDataset(db_path=db_path)
        n = dataset.export_extxyz(output_path, only_done=only_done)
        return _ok(f"Exported {n} structures to {output_path}.",
                   {"n_exported": n, "output_path": output_path})
    except ImportError as e:
        return _fail(f"HTP export error: {e}", 200)


@router.post("/htp/script")
async def agent_htp_script(request: Request):
    """
    Generate HTP batch job script.
    Body: {"db_path": "htp_dataset.db", "encut": 450, "kpoints": "4 4 1",
           "batch_size": 50, "scheduler": "sge"}
    """
    body = await _json(request)
    try:
        from .ase_scripts import script_htp_batch, script_filename
        script = script_htp_batch(
            db_path=body.get("db_path", "htp_dataset.db"),
            encut=int(body.get("encut", 450)),
            kpoints=body.get("kpoints", "4 4 1"),
            batch_size=int(body.get("batch_size", 50)),
            scheduler=body.get("scheduler", "sge"),
        )
        return _ok("HTP batch script generated.", {
            "script": script, "filename": "htp_batch.py"
        })
    except (ValueError, KeyError, TypeError) as e:
        return _fail(f"HTP script error: {e}", 200)


@router.post("/hpc/fetch")
async def agent_hpc_fetch(request: Request):
    """
    Fetch calculation results from the cluster via rsync.
    Body: {"remote_path": str, "cluster": str, "user": str,
           "files": ["CONTCAR","OUTCAR","stdout","OSZICAR"]}
    Returns: {"ok": True, "files": {"CONTCAR": "...", "OUTCAR": "...", ...}}
    """
    import shlex, tempfile
    body = await _json(request)
    remote_path = body.get("remote_path", "")
    cluster_key = body.get("cluster", "hoffman2")
    user        = body.get("user", "")
    want_files  = body.get("files") or ["CONTCAR", "OUTCAR", "stdout", "OSZICAR"]

    if not remote_path:
        return _fail("remote_path is required", 400)

    try:
        from .hpc_agent import CONF as _HPC_CONF
        resolved_key = cluster_key if cluster_key in _HPC_CONF else next(
            (k for k, v in _HPC_CONF.items() if cluster_key in v.get("host", "")), "hoffman2"
        )
        cluster_conf = dict(_HPC_CONF.get(resolved_key, {}))
    except ImportError as e:
        return _fail(f"Cannot load cluster config: {e}", 200)

    if user and cluster_conf.get("host"):
        import re as _re
        host_only = _re.sub(r"^[^@]+@", "", cluster_conf["host"])
        cluster_conf["host"] = f"{user}@{host_only}"

    host         = cluster_conf.get("host", "")
    ssh_opts_str = cluster_conf.get("ssh_options",
                                    "-o BatchMode=yes -o StrictHostKeyChecking=accept-new")
    # Sanitise remote_path: parentheses break shell glob expansion
    import re as _re
    remote_path_safe = _re.sub(r"[()\\]", "", remote_path)

    fetched = {}
    errors  = {}
    with tempfile.TemporaryDirectory(prefix="chatdft_fetch_") as tmpdir:
        import subprocess as _sp
        for fname in want_files:
            src = f"{host}:{remote_path_safe}/{fname}"
            dst = f"{tmpdir}/{fname}"
            try:
                proc = _sp.run(
                    ["rsync", "-az",
                     "-e", f"ssh {ssh_opts_str}",
                     src, dst],
                    shell=False, capture_output=True, text=True, timeout=60
                )
                if proc.returncode == 0:
                    import os
                    if os.path.exists(dst):
                        content = open(dst, errors="replace").read()
                        fetched[fname] = content
                    else:
                        errors[fname] = "not found on remote"
                else:
                    errors[fname] = proc.stderr.strip() or "rsync failed"
            except (ValueError, KeyError, TypeError) as e:
                errors[fname] = str(e)

    if not fetched:
        return _fail(f"No files retrieved. Errors: {errors}", 200)

    # Quick energy parse from OUTCAR
    energy = None
    if "OUTCAR" in fetched:
        import re as _re
        m = _re.findall(r"TOTEN\s*=\s*([-\d.]+)\s*eV", fetched["OUTCAR"])
        if m:
            energy = float(m[-1])

    return _ok("Results fetched.", {
        "files": fetched,
        "errors": errors,
        "remote_path": remote_path,
        "energy_eV": energy,
        "n_files": len(fetched),
    })


@router.post("/hpc/submit")
async def agent_hpc_submit(request: Request):
    """
    Write POSCAR + ase-xxx.py + job.sh to a temp dir, rsync to cluster, submit via qsub/sbatch.

    Body: {
        "poscar":      str,           # POSCAR content
        "script":      str,           # ase-xxx.py content
        "filename":    str,           # e.g. "ase-geo.py"
        "job_sh":      str,           # job.sh content (optional — uses default SGE template if omitted)
        "remote_path": str,           # full remote path e.g. /u/scratch/.../01_surface/Ag111_4x4x3
        "cluster":     str,           # hostname or config key (e.g. "hoffman2" or "hoffman2.idre.ucla.edu")
        "user":        str,           # ssh user (optional, overrides config)
        "job_name":    str,
        "ntasks":      int,
        "walltime":    str,
        "session_id":  int,
    }
    Returns: {"ok": True, "job_id": "...", "remote_path": "..."}
    """
    import shlex, tempfile
    body = await _json(request)

    poscar    = body.get("poscar", "")
    script    = body.get("script", "")
    filename  = body.get("filename", "ase-geo.py")
    job_sh    = body.get("job_sh", "")
    job_name  = body.get("job_name", "chatdft_job")
    ntasks    = int(body.get("ntasks", 32))
    walltime  = body.get("walltime", "24:00:00")
    cluster_key = body.get("cluster", "hoffman2")
    user      = body.get("user", "")
    remote_path = body.get("remote_path", "")
    session_id  = body.get("session_id")

    if not poscar:
        return _fail("poscar is required", 400)
    if not script:
        return _fail("script is required", 400)

    # Resolve cluster config key: accept full hostname or short key
    cluster_conf = {}
    resolved_key = "hoffman2"
    try:
        from .hpc_agent import CONF as _HPC_CONF
        # Try exact match first, then hostname match
        if cluster_key in _HPC_CONF:
            resolved_key = cluster_key
        else:
            for k, v in _HPC_CONF.items():
                if cluster_key in v.get("host", ""):
                    resolved_key = k
                    break
        cluster_conf = dict(_HPC_CONF.get(resolved_key, {}))
    except (ValueError, KeyError, TypeError) as e:
        return _fail(f"Cannot load cluster config: {e}", 200)

    # Override user in host string if provided
    if user and cluster_conf.get("host"):
        host = cluster_conf["host"]
        # Replace existing user@ prefix or prepend
        import re as _re
        host = _re.sub(r"^[^@]+@", "", host)
        cluster_conf["host"] = f"{user}@{host}"

    # Build default job.sh from template if not provided by caller
    scheduler = cluster_conf.get("scheduler", "sge")
    if not job_sh:
        if scheduler == "sge":
            job_sh = _make_sge_job_sh(filename, ntasks=ntasks, walltime=walltime)
        else:
            job_sh = (
                f"#!/bin/bash\n"
                f"#SBATCH --job-name={job_name}\n"
                f"#SBATCH --ntasks={ntasks}\n"
                f"#SBATCH --time={walltime}\n"
                f"#SBATCH --output=$SLURM_JOB_ID.log\n"
                f"#SBATCH --error=$SLURM_JOB_ID.err\n\n"
                f"python {filename}\n"
                f"echo \"run complete on $(hostname): $(date) $(pwd)\" >> ~/job.log\n"
            )

    # Determine remote directory
    if not remote_path:
        remote_base = cluster_conf.get("remote_base", "~/chatdft_jobs")
        import re as _re
        safe_name = _re.sub(r"[^\w\-_]", "_", job_name).strip("_") or "job"
        if session_id:
            remote_path = f"{remote_base}/s{session_id}/{safe_name}"
        else:
            remote_path = f"{remote_base}/{safe_name}"

    # Write files to temp dir
    with tempfile.TemporaryDirectory(prefix="chatdft_hpc_") as tmpdir:
        td = Path(tmpdir)
        (td / "POSCAR").write_text(poscar)
        (td / filename).write_text(script)
        (td / "job.sh").write_text(job_sh)
        (td / "job.sh").chmod(0o755)

        # rsync up
        host = cluster_conf.get("host", "")
        ssh_opts_str = cluster_conf.get("ssh_options",
                                        "-o BatchMode=yes -o StrictHostKeyChecking=accept-new")
        ssh_opts_list = ssh_opts_str.split()

        # Sanitise remote_path: parentheses break shell glob expansion
        import re as _re
        remote_path_safe = _re.sub(r"[()\\]", "", remote_path)

        try:
            import subprocess as _sp

            # mkdir on remote — use list form (shell=False) to avoid shell quoting issues
            _sp.run(
                ["ssh"] + ssh_opts_list + [host, f"mkdir -p '{remote_path_safe}'"],
                shell=False, check=True, timeout=30
            )

            # rsync — path after colon doesn't need shell quoting with shell=False
            _sp.run(
                ["rsync", "-az",
                 "-e", f"ssh {ssh_opts_str}",
                 f"{tmpdir}/",
                 f"{host}:{remote_path_safe}/"],
                shell=False, check=True, timeout=120
            )

            # submit job
            submit_cmd = cluster_conf.get("submit_cmd") or (
                "qsub job.sh" if scheduler == "sge" else "sbatch run.slurm"
            )
            remote_shell = f"set -e; cd '{remote_path_safe}'; {submit_cmd}"
            proc = _sp.run(
                ["ssh"] + ssh_opts_list + [host, remote_shell],
                shell=False, capture_output=True, text=True, timeout=60
            )
            if proc.returncode != 0:
                return _fail(
                    f"qsub failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}",
                    200
                )

            # parse job id
            import re as _re
            jid_regex = cluster_conf.get("job_id_regex", r"(\d+)")
            m = _re.search(jid_regex, (proc.stdout or "") + "\n" + (proc.stderr or ""))
            job_id = m.group(1) if m else (proc.stdout.strip() or "SUBMITTED")

            return _ok("Job submitted.", {
                "job_id": job_id,
                "remote_path": remote_path_safe,
                "host": host,
                "scheduler": scheduler,
                "submit_stdout": proc.stdout.strip(),
            })

        except _sp.TimeoutExpired:
            return _fail("SSH/rsync timed out. Check cluster connectivity.", 200)
        except _sp.CalledProcessError as e:
            return _fail(f"rsync/ssh failed: {e.stderr or str(e)}", 200)
        except OSError as e:
            return _fail(f"HPC submit error: {e}", 200)


@router.post("/parameters/analyze_benchmarks")
async def agent_parameters_analyze_benchmarks(request: Request):
    """Parse OSZICAR benchmark sub-directories and return convergence + plots."""
    body = await _json(request)
    bench_dir = body.get("bench_dir") or body.get("job_dir")
    if not bench_dir:
        return _fail("bench_dir is required", 400)
    from .parameters_agent import analyze_benchmarks
    result = analyze_benchmarks(Path(bench_dir))
    return _ok("Benchmarks analyzed.", result)


# ── Calc profiles ──────────────────────────────────────────────────────────

@router.post("/parameters/profile")
async def agent_calc_profile(request: Request):
    """
    Return a VASP parameter profile from calc_profiles.yaml.

    Body: { "profile": "relax_slab" | "high_accuracy" | "nnp_singlepoint" | ... }
    """
    body = await _json(request)
    profile_name = body.get("profile") or "standard"
    from .parameters_agent import _get_vasp_profile
    params = _get_vasp_profile(profile_name)
    return _ok(f"Profile '{profile_name}' loaded.", {
        "profile": profile_name,
        "params": params,
    })


@router.get("/parameters/profiles")
async def list_calc_profiles():
    """List all available calc profile names from calc_profiles.yaml."""
    import yaml, pathlib
    cfg_path = pathlib.Path(__file__).parent / "calc_profiles.yaml"
    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        names = [k for k in raw if k != "base"]
    except OSError:
        names = []
    return _ok("Profiles listed.", {"profiles": names})


# ── Job watcher (background task) ─────────────────────────────────────────

@router.post("/hpc/watch")
async def agent_hpc_watch(request: Request):
    """
    Start a background JobWatcher for an already-submitted HPC job.

    Body:
    {
      "task_id"       : int,          # WorkflowTask.id to update
      "job_id"        : str,          # cluster job id (from sbatch / qsub)
      "job_dir"       : str,          # local directory path
      "cluster"       : str,          # cluster name (cluster_config.yaml key)
      "session_id"    : int,
      "species"       : str,          # e.g. "H"  (for hypothesis feedback)
      "surface"       : str,          # e.g. "Pt(111)"
      "poll_interval" : int           # seconds between polls (default 60)
    }

    Returns immediately with {"ok": true, "watching": true}.
    The watcher runs in the background and updates the DB when the job ends.
    """
    body = await _json(request)
    task_id    = int(body.get("task_id", 0) or 0)
    job_id     = str(body.get("job_id") or "")
    job_dir    = str(body.get("job_dir") or "")
    cluster    = str(body.get("cluster") or "hoffman2")
    session_id = int(body.get("session_id", 0) or 0)
    species    = str(body.get("species") or "")
    surface    = str(body.get("surface") or "")
    poll       = int(body.get("poll_interval") or 60)

    if not job_id or not job_dir:
        return _fail("job_id and job_dir are required", 400)
    if not task_id or not session_id:
        return _fail("task_id and session_id are required", 400)

    try:
        hpc = HPCAgent(cluster=cluster)
    except Exception as exc:
        return _fail(f"HPCAgent init failed: {exc}", 200)

    from .job_watcher import watch_job
    import asyncio
    asyncio.get_event_loop().create_task(
        watch_job(
            task_id=task_id,
            job_id=job_id,
            job_dir=Path(job_dir),
            hpc=hpc,
            session_id=session_id,
            species=species,
            surface=surface,
            poll_interval=poll,
        )
    )
    return _ok("Job watcher started in background.", {
        "watching": True,
        "task_id": task_id,
        "job_id": job_id,
        "poll_interval": poll,
    })