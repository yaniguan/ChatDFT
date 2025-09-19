# server/execution/post_analysis_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv, json, re, math

# --- plotting (best-effort，缺库不报错) ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# --- 可选：用 pymatgen 解析 VASP ---
try:
    from pymatgen.io.vasp import Vasprun, Outcar
    from pymatgen.core import Structure
    _HAS_PMG = True
except Exception:
    _HAS_PMG = False

# --- 可选：LLM 总结 ---
_HAS_LLM = False
try:
    import os
    if os.getenv("OPENAI_API_KEY"):
        from server.utils.openai_wrapper import chatgpt_call
        _HAS_LLM = True
except Exception:
    _HAS_LLM = False


# =========================================
# 数据结构
# =========================================
@dataclass
class JobRecord:
    job_dir: str
    label: str
    engine: str
    calc: str
    E_eV: Optional[float] = None
    fermi_eV: Optional[float] = None
    bandgap_eV: Optional[float] = None
    spin_mag: Optional[float] = None
    E_ads_eV: Optional[float] = None
    barrier_eV: Optional[float] = None
    notes: str = ""
    figs: List[str] = None  # 相对 job_dir 的图路径
    remote: Dict[str, Any] = None  # 透传 _remote.json（如果有）


# =========================================
# 引擎适配层
# =========================================
class EngineAdapter:
    name = "unknown"

    def detect(self, d: Path) -> bool:
        raise NotImplementedError

    def calc_type(self, d: Path) -> str:
        raise NotImplementedError

    def parse_job(self, d: Path) -> JobRecord:
        raise NotImplementedError

    # 可选：绘图接口，返回相对路径列表
    def render_figs(self, d: Path) -> List[str]:
        return []


class VaspAdapter(EngineAdapter):
    name = "vasp"

    def detect(self, d: Path) -> bool:
        return any((d / x).exists() for x in ["INCAR", "vasprun.xml", "OUTCAR", "DOSCAR"])

    def calc_type(self, d: Path) -> str:
        inc = (d / "INCAR")
        name = d.name.lower()
        if "neb" in name or (d / "neb.dat").exists():
            return "neb"
        if "dos" in name or (d / "DOSCAR").exists():
            return "dos"
        if "band" in name:
            return "bands"
        if "elf" in name:
            return "elf"
        if "bader" in name or (d / "ACF.dat").exists():
            return "bader"
        if inc.exists():
            try:
                txt = inc.read_text()
                m = re.search(r"NSW\s*=\s*(\d+)", txt)
                if m and int(m.group(1)) > 0:
                    return "relax"
            except Exception:
                pass
            return "scf"
        return "scf"

    def _pmg_parse(self, d: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
        notes = []
        E = Ef = Eg = M = None
        if not _HAS_PMG:
            return None, None, None, None, "pymatgen not available"

        vrun = d / "vasprun.xml"
        out = d / "OUTCAR"

        if vrun.exists():
            try:
                vr = Vasprun(str(vrun), parse_dos=False, parse_potcar_file=False)
                E = float(getattr(vr, "final_energy", None)) if hasattr(vr, "final_energy") else None
                try:
                    Ef = float(vr.efermi)
                except Exception:
                    pass
                try:
                    Eg = float(vr.eigenvalue_band_properties[0])
                except Exception:
                    pass
            except Exception as e:
                notes.append(f"vasprun: {e}")

        if out.exists():
            try:
                oc = Outcar(str(out))
                if M is None:
                    try:
                        M = float(oc.total_mag)
                    except Exception:
                        pass
                if E is None:
                    try:
                        text = out.read_text(errors="ignore")[-150000:]
                        m = re.findall(r"free  energy   TOTEN\s*=\s*([-\d\.Ee+]+)", text)
                        if m:
                            E = float(m[-1])
                    except Exception:
                        pass
            except Exception as e:
                notes.append(f"outcar: {e}")

        return E, Ef, Eg, M, "; ".join(notes)

    def parse_job(self, d: Path) -> JobRecord:
        calc = self.calc_type(d)
        label = d.name
        E, Ef, Eg, M, n = self._pmg_parse(d)
        notes = n or ""

        if E is None:
            try:
                osz = d / "OSZICAR"
                if osz.exists():
                    text = osz.read_text(errors="ignore")[-100000:]
                    m = re.findall(r"E0=\s*([-\d\.Ee+]+)", text)
                    if m:
                        E = float(m[-1])
            except Exception:
                pass

        remote = {}
        rj = d / "_remote.json"
        if rj.exists():
            try:
                remote = json.loads(rj.read_text())
            except Exception:
                pass

        return JobRecord(
            job_dir=str(d),
            label=label,
            engine=self.name,
            calc=calc,
            E_eV=E,
            fermi_eV=Ef,
            bandgap_eV=Eg,
            spin_mag=M,
            notes=notes,
            figs=[],
            remote=remote or None,
        )

    # ----------- 绘图 -----------
    def render_figs(self, d: Path) -> List[str]:
        figs: List[str] = []
        if not _HAS_MPL:
            return figs
        fdir = d / "figs"
        fdir.mkdir(exist_ok=True)

        # DOS
        try:
            dos_ok = False
            if _HAS_PMG and (d / "vasprun.xml").exists():
                vr = Vasprun(str(d / "vasprun.xml"), parse_dos=True, parse_potcar_file=False)
                td = vr.complete_dos
                if td is not None:
                    e = td.energies - td.efermi
                    dos = td.densities.get("total")
                    if dos is not None:
                        plt.figure()
                        plt.plot(dos, e)
                        plt.xlabel("DOS")
                        plt.ylabel("E - Ef (eV)")
                        plt.title("Total DOS")
                        plt.tight_layout()
                        outp = fdir / "DOS.png"
                        plt.savefig(outp)
                        plt.close()
                        figs.append(str(outp.relative_to(d)))
                        dos_ok = True
            # 如果没有 pmg 的总 DOS，略过
        except Exception:
            pass

        # NEB：neb.dat 或 neb_energies.json
        try:
            neb = d / "neb.dat"
            nebj = d / "neb_energies.json"
            ys = []
            if neb.exists():
                for line in neb.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split()
                    if len(parts) >= 2:
                        ys.append(float(parts[1]))
            elif nebj.exists():
                arr = json.loads(nebj.read_text()).get("energies") or []
                ys = [float(v) for v in arr]
            if ys:
                plt.figure()
                plt.plot(range(len(ys)), ys, marker="o")
                plt.xlabel("Image")
                plt.ylabel("Relative energy (eV)")
                plt.title("NEB profile")
                plt.tight_layout()
                outp = fdir / "NEB.png"
                plt.savefig(outp)
                plt.close()
                figs.append(str(outp.relative_to(d)))
        except Exception:
            pass

        return figs


class QeAdapter(EngineAdapter):
    name = "qe"

    def detect(self, d: Path) -> bool:
        return bool(list(d.glob("*.in"))) or bool(list(d.glob("*.out")))

    def calc_type(self, d: Path) -> str:
        nm = d.name.lower()
        if "vc-relax" in nm:
            return "vc-relax"
        if "relax" in nm:
            return "relax"
        if "dos" in nm:
            return "dos"
        if "band" in nm or "bands" in nm:
            return "bands"
        return "scf"

    def parse_job(self, d: Path) -> JobRecord:
        calc = self.calc_type(d)
        label = d.name
        E = Ef = Eg = M = None
        notes = []

        outs = sorted(d.glob("*.out"))
        if outs:
            try:
                text = outs[-1].read_text(errors="ignore")
                m = re.findall(r"!\s*total energy\s*=\s*([-\d\.Ee+]+)\s*Ry", text)
                if m:
                    E = float(m[-1]) * 13.605693009
                m = re.findall(r"the Fermi energy is\s*([-\d\.Ee+]+)\s*ev", text, flags=re.I)
                if m:
                    Ef = float(m[-1])
                m = re.findall(r"band gap\s*=\s*([-\d\.Ee+]+)\s*eV", text, flags=re.I)
                if m:
                    Eg = float(m[-1])
            except Exception as e:
                notes.append(f"qe.out parse: {e}")

        remote = {}
        rj = d / "_remote.json"
        if rj.exists():
            try:
                remote = json.loads(rj.read_text())
            except Exception:
                pass

        return JobRecord(
            job_dir=str(d),
            label=label,
            engine=self.name,
            calc=calc,
            E_eV=E,
            fermi_eV=Ef,
            bandgap_eV=Eg,
            spin_mag=M,
            notes="; ".join([x for x in notes if x]),
            figs=[],
            remote=remote or None,
        )

    def render_figs(self, d: Path) -> List[str]:
        figs: List[str] = []
        if not _HAS_MPL:
            return figs
        fdir = d / "figs"
        fdir.mkdir(exist_ok=True)

        # 简单 DOS：找 *.pdos, *.dos（projwfc.x / dos.x 输出常见）
        try:
            pdos_files = list(d.glob("*.pdos")) + list(d.glob("*.dos"))
            if pdos_files:
                xs, ys = [], []
                with pdos_files[0].open() as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"): continue
                        parts = re.split(r"\s+", line)
                        if len(parts) >= 2:
                            xs.append(float(parts[0]))
                            ys.append(float(parts[1]))
                if xs and ys:
                    plt.figure()
                    plt.plot(ys, xs)
                    plt.xlabel("DOS")
                    plt.ylabel("Energy (eV)")
                    plt.title("DOS (QE)")
                    plt.tight_layout()
                    outp = fdir / "DOS.png"
                    plt.savefig(outp)
                    plt.close()
                    figs.append(str(outp.relative_to(d)))
        except Exception:
            pass

        # 简单 bands：gnuplot dat
        try:
            gnu = d / "bands.dat.gnu"
            if gnu.exists():
                xs, ys = [], []
                for line in gnu.read_text().splitlines():
                    if line.strip() == "":
                        continue
                    parts = re.split(r"\s+", line.strip())
                    if len(parts) >= 2:
                        xs.append(float(parts[0])); ys.append(float(parts[1]))
                if xs and ys:
                    plt.figure()
                    plt.plot(xs, ys, linewidth=0.8)
                    plt.xlabel("k-path")
                    plt.ylabel("Energy (eV)")
                    plt.title("Band structure (QE)")
                    plt.tight_layout()
                    outp = fdir / "BANDS.png"
                    plt.savefig(outp)
                    plt.close()
                    figs.append(str(outp.relative_to(d)))
        except Exception:
            pass

        return figs


ADAPTERS: List[EngineAdapter] = [VaspAdapter(), QeAdapter()]


# =========================================
# 助手：识别/派生量
# =========================================
def _guess_role_by_name(name: str) -> str:
    n = name.lower()
    if n.startswith("ads_") or "adsorb" in n:
        return "ads"
    if "coads" in n or "co-ads" in n:
        return "coads"
    if n.startswith("slab") or "slab" in n:
        return "slab"
    if n.startswith("bulk") or "bulk" in n:
        return "bulk"
    if "neb" in n or "ts" in n:
        return "neb"
    if "dos" in n or "band" in n or "bader" in n or "elf" in n:
        return "electronic"
    return "other"


def _find_energy_by_role(records: List[JobRecord], role: str) -> List[JobRecord]:
    return [r for r in records if _guess_role_by_name(Path(r.job_dir).name) == role and r.E_eV is not None]


def _calc_adsorption_energies(records: List[JobRecord]) -> None:
    slabs = _find_energy_by_role(records, "slab")
    if not slabs:
        return
    for r in records:
        role = _guess_role_by_name(Path(r.job_dir).name)
        if role in ("ads", "coads") and r.E_eV is not None:
            slab_ref = None
            parent = Path(r.job_dir).parent
            cand = [x for x in slabs if Path(x.job_dir).parent == parent]
            slab_ref = cand[0] if cand else None
            if slab_ref is None and slabs:
                slab_ref = slabs[0]
            if slab_ref and slab_ref.E_eV is not None:
                r.E_ads_eV = r.E_eV - slab_ref.E_eV


def _calc_neb_barriers(records: List[JobRecord]) -> None:
    for r in records:
        d = Path(r.job_dir)
        if r.calc == "neb" or "neb" in d.name.lower() or "ts" in d.name.lower():
            barrier = None
            jf = d / "neb_energies.json"
            if jf.exists():
                try:
                    arr = json.loads(jf.read_text()).get("energies") or []
                    if arr:
                        barrier = float(max(arr) - min(arr))
                except Exception:
                    pass
            if barrier is None:
                dat = d / "neb.dat"
                if dat.exists():
                    try:
                        vals = []
                        for line in dat.read_text(errors="ignore").splitlines():
                            line = line.strip()
                            if not line or line.startswith("#"): continue
                            parts = line.split()
                            if len(parts) >= 2:
                                vals.append(float(parts[1]))
                        if vals:
                            barrier = float(max(vals))
                    except Exception:
                        pass
            r.barrier_eV = barrier


# =========================================
# LLM 总结（可选）
# =========================================
async def _llm_job_insight(job: JobRecord) -> str:
    if not _HAS_LLM:
        return ""
    prompt = {
        "job_dir": job.job_dir,
        "engine": job.engine,
        "calc": job.calc,
        "E_eV": job.E_eV, "E_ads_eV": job.E_ads_eV, "barrier_eV": job.barrier_eV,
        "fermi_eV": job.fermi_eV, "bandgap_eV": job.bandgap_eV, "spin_mag": job.spin_mag,
        "notes": job.notes,
        "ask": "Give a terse 3-5 bullet insight focusing on HER/CO2RR relevance and obvious follow-ups."
    }
    try:
        txt = await chatgpt_call(
            [{"role":"system","content":"You are a concise computational catalysis analyst."},
             {"role":"user","content":json.dumps(prompt, ensure_ascii=False)}],
            model="gpt-4o-mini", temperature=0.2, max_tokens=250
        )
        return txt.strip()
    except Exception:
        return ""

async def _llm_overall_summary(records: List[JobRecord]) -> str:
    if not _HAS_LLM:
        return ""
    table = [
        {k: getattr(r, k) for k in
         ("label","engine","calc","E_eV","E_ads_eV","barrier_eV","fermi_eV","bandgap_eV","spin_mag")}
        for r in records
    ]
    prompt = {
        "table": table[:200],
        "ask": "Summarize: best adsorption (most negative), lowest barrier, any electronic trends. "
               "Then propose next computations (convergence, co-ads pairs, NEB steps). Markdown, <= 180 words."
    }
    try:
        txt = await chatgpt_call(
            [{"role":"system","content":"You are a PI reviewing a DFT campaign; be decisive."},
             {"role":"user","content":json.dumps(prompt, ensure_ascii=False)}],
            model="gpt-4o-mini", temperature=0.2, max_tokens=350
        )
        return txt.strip()
    except Exception:
        return ""


# =========================================
# 主 Agent
# =========================================
class PostAnalysisAgent:
    """
    使用：
        PostAnalysisAgent().analyze(work_root)

    输出：
        work_root/results.csv
        work_root/results.json
        work_root/summary.json
        work_root/figs/index.json      # 全部图片清单
        每个 job_dir/figs/*.png        # DOS/NEB 等图（尽力）
        每个 job_dir/insight.md        # LLM 逐作业简评（可选）
        work_root/analysis.md          # LLM 全局总结（可选）
    """

    def _scan_jobs(self, root: Path) -> List[Path]:
        subs = [p for p in root.iterdir() if p.is_dir()]
        return [p for p in subs if not p.name.startswith(".")]

    def _pick_adapter(self, d: Path) -> Optional[EngineAdapter]:
        for ad in ADAPTERS:
            try:
                if ad.detect(d):
                    return ad
            except Exception:
                continue
        return None

    def _parse_all(self, root: Path) -> List[JobRecord]:
        recs: List[JobRecord] = []
        for d in self._scan_jobs(root):
            ad = self._pick_adapter(d)
            if ad is None:
                recs.append(JobRecord(job_dir=str(d), label=d.name, engine="unknown", calc="unknown",
                                      notes="no engine detected", figs=[]))
                continue
            try:
                rec = ad.parse_job(d)
                # 绘图
                try:
                    figs = ad.render_figs(d)
                    rec.figs = figs or []
                except Exception:
                    rec.figs = []
            except Exception as e:
                rec = JobRecord(job_dir=str(d), label=d.name, engine=ad.name, calc=ad.calc_type(d),
                                notes=f"parse error: {e}", figs=[])
            recs.append(rec)
        return recs

    def _write_csv_json(self, root: Path, recs: List[JobRecord]) -> None:
        csv_path = root / "results.csv"
        cols = ["job_dir","label","engine","calc","E_eV","E_ads_eV","barrier_eV",
                "fermi_eV","bandgap_eV","spin_mag","notes"]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in recs:
                w.writerow([r.job_dir, r.label, r.engine, r.calc, r.E_eV, r.E_ads_eV,
                            r.barrier_eV, r.fermi_eV, r.bandgap_eV, r.spin_mag, r.notes])

        # JSON（包含 figs 与 remote）
        (root / "results.json").write_text(json.dumps([asdict(r) for r in recs], indent=2))

        # 图片索引
        all_figs = []
        for r in recs:
            for relp in (r.figs or []):
                all_figs.append({
                    "job_dir": r.job_dir,
                    "label": r.label,
                    "calc": r.calc,
                    "path": str(Path(r.job_dir) / relp)
                })
        (root / "figs").mkdir(exist_ok=True)
        (root / "figs" / "index.json").write_text(json.dumps(all_figs, indent=2))

    def _write_summary(self, root: Path, recs: List[JobRecord]) -> None:
        best_ads = None
        ads_list = [r for r in recs if _guess_role_by_name(Path(r.job_dir).name) in ("ads", "coads")
                    and r.E_ads_eV is not None]
        if ads_list:
            best_ads = sorted(ads_list, key=lambda x: x.E_ads_eV)[0]

        best_neb = None
        neb_list = [r for r in recs if (r.calc == "neb" or "neb" in Path(r.job_dir).name.lower()
                                        or "ts" in Path(r.job_dir).name.lower()) and r.barrier_eV is not None]
        if neb_list:
            best_neb = sorted(neb_list, key=lambda x: x.barrier_eV)[0]

        elec = [r for r in recs if r.bandgap_eV is not None or r.fermi_eV is not None]
        top_elec = elec[0] if elec else None

        summary = {
            "n_jobs": len(recs),
            "best_adsorption": asdict(best_ads) if best_ads else None,
            "best_neb": asdict(best_neb) if best_neb else None,
            "electronic_example": asdict(top_elec) if top_elec else None,
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2))

    async def _maybe_llm_writeups(self, root: Path, recs: List[JobRecord]) -> None:
        if not _HAS_LLM:
            return
        # 逐作业 insight
        for r in recs:
            try:
                txt = await _llm_job_insight(r)
                if txt:
                    (Path(r.job_dir) / "insight.md").write_text(txt)
            except Exception:
                continue
        # 全局总结
        try:
            overview = await _llm_overall_summary(recs)
            if overview:
                (Path(root) / "analysis.md").write_text(overview)
        except Exception:
            pass

    # 公开 API
    def analyze(self, work_root: Path | str) -> Dict[str, Any]:
        root = Path(work_root)
        root.mkdir(parents=True, exist_ok=True)

        recs = self._parse_all(root)
        _calc_adsorption_energies(recs)
        _calc_neb_barriers(recs)

        self._write_csv_json(root, recs)
        self._write_summary(root, recs)

        # LLM 写稿（同步调用外壳，内部是异步）
        if _HAS_LLM:
            import asyncio
            try:
                asyncio.run(self._maybe_llm_writeups(root, recs))
            except RuntimeError:
                # 在已有事件循环内（如 FastAPI）时，退化为创建任务并忽略
                pass

        return {
            "ok": True,
            "results_csv": str(root / "results.csv"),
            "results_json": str(root / "results.json"),
            "summary_json": str(root / "summary.json"),
            "figs_index": str(root / "figs" / "index.json"),
            "n_records": len(recs),
        }