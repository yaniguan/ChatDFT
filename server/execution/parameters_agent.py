# server/execution/parameters_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json, math, textwrap, datetime, asyncio

from pymatgen.core import Structure

# ========= 事件上报（可选，失败不阻塞） =========
try:
    from server.execution.utils.events import post_event
except Exception:
    async def post_event(_payload):  # 兜底
        return

try:
    from server.chat.contracts import RunEvent  # 如果没有就用 dict
    def _pack_event(run_id, step_id, phase, payload=None):
        return RunEvent(run_id=run_id, step_id=step_id, phase=phase, payload=payload or {}).model_dump()
except Exception:
    def _pack_event(run_id, step_id, phase, payload=None):
        return {"run_id": run_id, "step_id": step_id, "phase": phase, "payload": payload or {}}

def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def _append_jsonl(job_dir: Path, name: str, row: Dict[str, Any]):
    try:
        p = Path(job_dir) / name
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now_iso(), **row}, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _emit_sync(run_id, step_id, phase: str, payload: Dict[str, Any] | None = None):
    try:
        coro = post_event(_pack_event(run_id, step_id, phase, payload))
        try:
            # 在已有事件循环里 fire-and-forget
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # 没有事件循环就直接跑一小下
            asyncio.run(coro)
    except Exception:
        pass

# ========= 原有逻辑 =========

_VASP_KEYS = {
    "SYSTEM","ISTART","ICHARG","ENCUT","PREC","LREAL","ISMEAR","SIGMA","ISPIN","MAGMOM",
    "EDIFF","EDIFFG","ALGO","LWAVE","LCHARG","ISYM","AMIX","BMIX","AMIX_MAG","BMIX_MAG",
    "IBRION","NSW","POTIM","ISIF","NELM","NELMIN","GGA","LASPH","LMAXMIX","ADDGRID",
    "LDAU","LDAUTYPE","LDAUL","LDAUU","LDAUJ","LDIPOL","IDIPOL","LDOS","NEDOS","LORBIT",
    "LELF","LFORK","LAECHG","NELECT","LSOL","EB_K", "IOAP"  # 兼容之前的 NEB/vaspsol 字段
}


# ------------------------------- 小工具 -------------------------------

def _load_structure_from_job(job_dir: Path) -> Optional[Structure]:
    """从典型命名的 POSCAR 里拿结构：优先 slab/ads，然后 bulk；兜底扫描 *.POSCAR。"""
    prefer = [
        "slab.POSCAR",
        "ads.POSCAR", "ads_H.POSCAR", "ads_CO.POSCAR",
        "bulk.POSCAR",
    ]
    for name in prefer:
        p = job_dir / name
        if p.exists():
            try:
                return Structure.from_file(str(p))
            except Exception:
                pass
    for p in job_dir.glob("*.POSCAR"):
        try:
            return Structure.from_file(str(p))
        except Exception:
            continue
    return None

def _monkhorst_from_kppra(struct: Structure, kppra: int) -> Tuple[int,int,int]:
    """按 KPPRA 估算 Monkhorst-Pack 网格。"""
    natom = max(1, len(struct))
    target = max(kppra // natom, 1)
    rl = struct.lattice.reciprocal_lattice_crystallographic
    G = (rl.a, rl.b, rl.c)
    geo = (G[0]*G[1]*G[2]) ** (1/3)
    k = [max(1, int(round(g/geo * math.pow(target, 1/3)))) for g in G]
    return (k[0], k[1], k[2])

def _monkhorst_from_kdensity(struct: Structure, kdensity: float) -> Tuple[int,int,int]:
    """kdensity： reciprocal 方向的点密度 (1/Å)。"""
    rl = struct.lattice.reciprocal_lattice_crystallographic
    ka = max(1, int(round(rl.a * kdensity)))
    kb = max(1, int(round(rl.b * kdensity)))
    kc = max(1, int(round(rl.c * kdensity)))
    return (ka, kb, kc)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_text(path: Path, content: str):
    path.write_text((content or "").strip() + "\n")

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    out.update(b or {})
    return out

def _read_form_overrides(task: Dict[str, Any]) -> Dict[str, Any]:
    """把 plan 的 form 值转成 overrides（key→value）。"""
    out: Dict[str, Any] = {}
    fields = (task.get("params") or {}).get("form") or []
    for f in fields:
        key = f.get("key")
        if key is None:
            continue
        out[key] = f.get("value")
    return out

# ------------------------ VASP 模板与生成 ------------------------

_VASP_BASE = dict(
    PREC="Accurate",
    ENCUT=520,
    EDIFF=1e-5,
    IBRION=2,        # 缺省几何优化（CG）
    ISMEAR=0, SIGMA=0.05,
    LWAVE=False, LCHARG=False,
    ISYM=0,
)

_VASP_RELAX_BULK = dict(ISIF=3, NSW=120)
_VASP_RELAX_SLAB = dict(ISIF=2, NSW=120)  # slab 常用：不改变体积；面内/原子放松
_VASP_SCF   = dict(IBRION=-1, NSW=0, ISMEAR=1, SIGMA=0.1)
_VASP_DOS   = dict(IBRION=-1, NSW=0, ISMEAR=0, SIGMA=0.05, NEDOS=3001, EDIFF=1e-6, LORBIT=11)
_VASP_BANDS = dict(IBRION=-1, NSW=0, ISMEAR=0, SIGMA=0.05, LWAVE=True, LORBIT=11)
_VASP_ELF   = dict(LFORK=True, LELF=True, IBRION=-1, NSW=0)
_VASP_BADER = dict(LCHARG=True, LAECHG=True, IBRION=-1, NSW=0)
_VASP_VASPSOL = dict(LSOL=True, EB_K=78.4)  # 水的介电
_VASP_GGAU  = dict(LDAU=True, LDAUTYPE=2, LDAUL=-1, LDAUU=0.0, LDAUJ=0.0)
_VASP_NELECT= dict(NELECT=None)  # 仅声明，由 overrides 赋值

def _vasp_incar_lines(params: Dict[str, Any]) -> str:
    lines: List[str] = []
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = ".TRUE." if v else ".FALSE."
        lines.append(f"{k} = {v}")
    return "\n".join(lines)

def _vasp_kpoints_grid(kmesh: Dict[str, Any], struct: Optional[Structure]) -> str:
    mode = (kmesh or {}).get("mode") or "kppra"
    val  = (kmesh or {}).get("value") or (1600 if mode=="kppra" else 0.15)
    if struct is None:
        grid = (3, 3, 1)
    else:
        grid = _monkhorst_from_kdensity(struct, float(val)) if mode=="kdensity" else _monkhorst_from_kppra(struct, int(val))
    return textwrap.dedent(f"""\
    Automatic mesh
      0
    Monkhorst-Pack
      {grid[0]} {grid[1]} {grid[2]}
      0  0  0
    """)

def _vasp_write(job_dir: Path, params: Dict[str, Any], kmesh: Dict[str, Any], struct: Optional[Structure]):
    _write_text(job_dir / "INCAR",   _vasp_incar_lines(params))
    _write_text(job_dir / "KPOINTS", _vasp_kpoints_grid(kmesh, struct))
    # 赝势映射清单（不生成真 POTCAR；供 HPC 端脚本使用）
    pp = params.get("_POTCAR_SPEC") or {}
    if pp:
        (job_dir / "POTCAR.spec").write_text(json.dumps(pp, indent=2))

# ------------------------ QE 模板与生成 ------------------------

def _qe_sections(calc_type: str, params: Dict[str, Any], struct: Optional[Structure]) -> str:
    control = dict(calculation=calc_type, prefix="qe", verbosity="low", tstress=True, tprnfor=True)
    system  = dict(ecutwfc=params.get("ecutwfc", 60), ecutrho=params.get("ecutrho", 480))
    electrons = dict(conv_thr=1e-8, mixing_beta=0.7)
    ions, cell = {}, {}

    if calc_type in ("relax", "vc-relax"):
        ions["ion_dynamics"] = "bfgs"
    if calc_type == "vc-relax":
        cell["cell_dynamics"] = "bfgs"

    # k 点
    kmesh = params.get("kmesh") or {"mode":"kppra", "value":1200}
    if struct is None:
        kline = "K_POINTS automatic\n 4 4 1  0 0 0\n"
    else:
        g = _monkhorst_from_kdensity(struct, float(kmesh["value"])) if kmesh.get("mode")=="kdensity" \
            else _monkhorst_from_kppra(struct, int(kmesh["value"]))
        kline = f"K_POINTS automatic\n {g[0]} {g[1]} {g[2]}  0 0 0\n"

    def block(title: str, d: Dict[str, Any]) -> str:
        lines = [f"&{title}"]
        for k, v in d.items():
            if isinstance(v, bool): v = ".true." if v else ".false."
            if isinstance(v, str):  v = f"'{v}'"
            lines.append(f"  {k} = {v}")
        lines.append("/")
        return "\n".join(lines)

    atomic = ""
    if struct is not None:
        lat = struct.lattice.matrix
        cell_str = "CELL_PARAMETERS angstrom\n" + "\n".join(["  %.10f %.10f %.10f" % tuple(v) for v in lat]) + "\n"
        apos = "ATOMIC_POSITIONS crystal\n" + "\n".join(
            f"  {sp.symbol}  {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for sp, c in zip(struct.species, struct.frac_coords)
        ) + "\n"
        sp_set = sorted({s.symbol for s in struct.species})
        amass = "ATOMIC_SPECIES\n" + "\n".join([f"  {el}  1.0  {el}.UPF" for el in sp_set]) + "\n"
        atomic = amass + apos + cell_str

    s = "\n".join([
        block("CONTROL", control),
        block("SYSTEM", system),
        block("ELECTRONS", electrons),
        block("IONS", ions),
        block("CELL", cell),
        atomic,
        kline,
    ])
    return s.strip() + "\n"

def _qe_write(job_dir: Path, calc_type: str, params: Dict[str, Any], struct: Optional[Structure]):
    fname = {"scf":"scf.in", "relax":"relax.in", "vc-relax":"relax.in", "bands":"bands.in", "dos":"dos.in"}.get(calc_type, "calc.in")
    _write_text(job_dir / fname, _qe_sections(calc_type, params, struct))
    pp = params.get("_PP_SPEC") or {}
    if pp:
        (job_dir / "pp.spec").write_text(json.dumps(pp, indent=2))

# ------------------------ 解释文档 ------------------------

def _write_explained(job_dir: Path, engine: str, calc_types: str, params: Dict[str, Any]):
    kmesh = params.get("kmesh") or {}
    md = f"""\
# Parameters explained

**Engine**: `{engine}`  
**Calc type(s)**: `{calc_types}`

## K-point strategy
- Mode: `{kmesh.get('mode','kppra')}`
- Value: `{kmesh.get('value')}`
- Rationale: KPPRA (k-points per reciprocal atom) 或 reciprocal density (1/Å)。由晶格 reciprocal 尺寸估算 Monkhorst-Pack。

## Key parameters (VASP)
- `PREC` 计算精度；`Accurate` 通常安全  
- `ENCUT` 截断能（eV），建议 400–600 做收敛  
- `ISMEAR/SIGMA`：金属常用 `ISMEAR=1,SIGMA=0.1`；分子/绝缘 `ISMEAR=0,SIGMA=0.05`  
- `ISIF`：bulk 缺省 `3`；slab 缺省 `2`  
- `LSOL/EB_K`：溶剂模型（VASPsol），水 `EB_K≈78.4`  
- `LDAU`：Dudarev 形式 GGA+U (`LDAUTYPE=2`)  
- `NELECT`：带电或电子数控制（按需使用）

## Key parameters (QE)
- `ecutwfc/ecutrho`：平面波/电荷密度截断；经验 `ecutrho≈8×ecutwfc`  
- `calculation`：`scf`/`relax`/`vc-relax`/`bands`/`nscf` 等  
- `K_POINTS automatic`：与上面策略一致

> 收敛建议：  
> - ENCUT: [400, 450, 500, 550, 600]  
> - kppra: [800, 1600, 2400] 或 kdensity: [0.12, 0.18, 0.24] (1/Å)

所有参数均可通过 `payload.overrides` 或前端表单覆盖。
"""
    _write_text(job_dir / "parameters_explained.md", md)

# ------------------------ Benchmarks 生成 ------------------------

def _write_benchmarks(job_dir: Path, engine: str, struct: Optional[Structure], encuts: List[int], kppra_list: List[int]):
    bdir = job_dir / "benchmarks"
    _ensure_dir(bdir)
    (bdir / "benchmarks.json").write_text(json.dumps({"encut": encuts, "kppra": kppra_list}, indent=2))

    if engine.lower() == "vasp":
        for e in encuts:
            sub = bdir / f"encut_{e}"
            _ensure_dir(sub)
            _write_text(sub / "INCAR", _vasp_incar_lines(_merge(_VASP_BASE, {"ENCUT": e, "IBRION": -1, "NSW": 0})))
            _write_text(sub / "KPOINTS", _vasp_kpoints_grid({"mode":"kppra","value":1600}, struct))
        for k in kppra_list:
            sub = bdir / f"kppra_{k}"
            _ensure_dir(sub)
            _write_text(sub / "INCAR", _vasp_incar_lines(_merge(_VASP_BASE, {"IBRION": -1, "NSW": 0})))
            _write_text(sub / "KPOINTS", _vasp_kpoints_grid({"mode":"kppra","value":k}, struct))
    else:
        for e in encuts:
            sub = bdir / f"ecut_{e}"
            _ensure_dir(sub)
            params = {"ecutwfc": e, "ecutrho": 8*e, "kmesh":{"mode":"kppra","value":1600}}
            _write_text(sub / "scf.in", _qe_sections("scf", params, struct))
        for k in kppra_list:
            sub = bdir / f"kppra_{k}"
            _ensure_dir(sub)
            params = {"ecutwfc": 60, "ecutrho": 480, "kmesh":{"mode":"kppra","value":k}}
            _write_text(sub / "scf.in", _qe_sections("scf", params, struct))

# ------------------------ 主 Agent ------------------------

class ParametersAgent:
    """
    generate(task, job_dir) 根据 payload 生成输入文件：
      payload = {
        "engine": "vasp" | "qe",
        "calc_type": "relax|scf|dos|bands|elf|bader|neb|vaspsol|gga_u|nelect",
        "modes": ["dos","bands"],            # 可并列；若未给，使用 calc_type；再兜底 "relax"
        "system_kind": "bulk|slab|adsorption",
        "kmesh": {"mode":"kppra"|"kdensity","value":1600},
        "pp_map": {...},                     # POTCAR/UPF 映射清单（不直接生成）
        "overrides": {...},                  # 任意键值覆盖模板
        "benchmark": {"encut":[...], "kppra":[...]},
      }
    """
    def generate(self, task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
        # ---- 上下文/事件元数据 ----
        run_id  = ((task.get("meta") or {}).get("run_id"))
        step_id = task.get("id")
        payload   = ((task.get("params") or {}).get("payload") or {})
        engine    = (payload.get("engine") or "vasp").lower()
        system    = (payload.get("system_kind") or "slab").lower()
        kmesh     = payload.get("kmesh") or {"mode":"kppra","value":1600}
        modes     = payload.get("modes") or ([payload.get("calc_type")] if payload.get("calc_type") else ["relax"])
        overrides = _merge(payload.get("overrides") or {}, _read_form_overrides(task))

        # 事件：准备
        _emit_sync(run_id, step_id, "params.prepare", {
            "engine": engine, "system_kind": system, "modes": modes, "kmesh": kmesh, "overrides": overrides
        })
        _append_jsonl(job_dir, "_param_log.jsonl", {
            "phase": "prepare", "engine": engine, "system_kind": system, "modes": modes, "kmesh": kmesh, "overrides": overrides
        })

        struct = _load_structure_from_job(job_dir)
        results: Dict[str, Any] = {"engine": engine, "generated": []}

        try:
            if engine == "vasp":
                for m in modes:
                    params = self._build_vasp_params(m, system, overrides, payload)
                    _vasp_write(job_dir, params, kmesh, struct)
                    results["generated"].append({"calc": m, "files": ["INCAR","KPOINTS","POTCAR.spec"]})
            elif engine == "qe":
                for m in modes:
                    qetype = {"relax":"relax", "vc-relax":"vc-relax"}.get(m, m if m in ("scf","dos","bands") else "scf")
                    params = self._build_qe_params(m, system, overrides, payload, kmesh)
                    _qe_write(job_dir, qetype, params, struct)
                    results["generated"].append({"calc": m, "files": ["*.in","pp.spec"]})
            else:
                raise RuntimeError(f"Unknown engine: {engine}")

            # 解释文档 & 基准
            _write_explained(job_dir, engine, ",".join(modes), {"kmesh": kmesh})
            bench = payload.get("benchmark") or {}
            encuts = list(map(int, bench.get("encut") or []))
            kppra_list = list(map(int, bench.get("kppra") or []))
            if encuts or kppra_list:
                _write_benchmarks(job_dir, engine, struct, encuts, kppra_list)

            meta = dict(engine=engine, modes=modes, system_kind=system, kmesh=kmesh, overrides=overrides)
            (job_dir / "param.json").write_text(json.dumps(meta, indent=2))

            out = {"ok": True, "engine": engine, "modes": modes, "job_dir": str(job_dir)}
            # 事件：完成
            _emit_sync(run_id, step_id, "params.done", {"result": out, "files": results.get("generated")})
            _append_jsonl(job_dir, "_param_log.jsonl", {"phase": "done", "result": out, "files": results.get("generated")})
            return out

        except Exception as e:
            err = {"ok": False, "error": str(e), "engine": engine}
            _emit_sync(run_id, step_id, "params.error", err)
            _append_jsonl(job_dir, "_param_log.jsonl", {"phase": "error", **err})
            raise

    # ---------------- VASP 参数拼装 ----------------
    def _build_vasp_params(self, mode: str, system: str, overrides: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(_VASP_BASE)
        if system == "bulk":
            base = _merge(base, _VASP_RELAX_BULK)
            base["ISMEAR"], base["SIGMA"] = 1, 0.1
        else:
            base = _merge(base, _VASP_RELAX_SLAB)
            base["ISMEAR"], base["SIGMA"] = 0, 0.05

        mode = (mode or "relax").lower()
        if mode == "scf":
            base = _merge(base, _VASP_SCF)
        elif mode == "dos":
            base = _merge(_merge(base, _VASP_SCF), _VASP_DOS)
        elif mode == "bands":
            base = _merge(_merge(base, _VASP_SCF), _VASP_BANDS)
        elif mode == "elf":
            base = _merge(base, _VASP_ELF)
        elif mode == "bader":
            base = _merge(base, _VASP_BADER)
        elif mode == "neb":
            base.update(dict(IBRION=3, POTIM=0, NSW=200, IOAP=0))
        elif mode == "vaspsol":
            base = _merge(base, _VASP_VASPSOL)
        elif mode in ("gga_u","ldau","u"):
            base = _merge(base, _VASP_GGAU)
        elif mode == "nelect":
            base = _merge(base, _VASP_NELECT)
        # 赝势映射
        if payload.get("pp_map"):
            base["_POTCAR_SPEC"] = payload["pp_map"]
        # 最终覆盖
        return _merge(base, overrides)

    # ---------------- QE 参数拼装 ----------------
    def _build_qe_params(self, mode: str, system: str, overrides: Dict[str, Any], payload: Dict[str, Any], kmesh: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(ecutwfc=overrides.get("ecutwfc", 60),
                 ecutrho=overrides.get("ecutrho", 480),
                 kmesh=kmesh)
        if payload.get("pp_map"):
            p["_PP_SPEC"] = payload["pp_map"]
        if mode in ("vc-relax","cellopt"):
            p["calculation"] = "vc-relax"
        elif mode == "scf":
            p["calculation"] = "scf"
        elif mode in ("dos","bands"):
            p["calculation"] = "scf"  # 需配合后处理
        # 覆盖
        return _merge(p, overrides)