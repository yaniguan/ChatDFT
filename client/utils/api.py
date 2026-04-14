# client/utils/api.py
import os
import requests
from typing import Any, Dict, List, Optional

BASE = os.getenv("CHATDFT_BACKEND", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("CHATDFT_TIMEOUT", "120"))


def _url(ep: str) -> str:
    return f"{BASE}/{ep.lstrip('/')}"


def post(ep: str, body: dict, timeout: int = TIMEOUT) -> dict:
    try:
        r = requests.post(_url(ep), json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        data.setdefault("ok", True)
        return data
    except requests.exceptions.ConnectionError:
        return {"ok": False, "detail": f"Cannot connect to backend at {BASE}. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"ok": False, "detail": f"Request timed out after {timeout}s."}
    except requests.HTTPError as e:
        try:
            err = r.json()
        except Exception:
            err = {"detail": r.text}
        err.update({"ok": False, "status_code": r.status_code})
        return err
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def get(ep: str, params: Optional[dict] = None, timeout: int = TIMEOUT) -> dict:
    try:
        r = requests.get(_url(ep), params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"ok": False, "detail": str(e)}


# ── Sessions ──────────────────────────────────────────────────────────────────
def list_sessions() -> dict:
    return post("/chat/session/list", {})

def create_session(name: str, project: str = "", tags: str = "") -> dict:
    return post("/chat/session/create", {"name": name, "project": project, "tags": tags})

def delete_session(session_id: int) -> dict:
    return post("/chat/session/delete", {"id": session_id})

def get_session_state(session_id: int) -> dict:
    return post("/chat/session/state", {"id": session_id})

def get_session_messages(session_id: int, limit: int = 200) -> dict:
    return post("/chat/session/messages", {"id": session_id, "limit": limit})

# ── Agent pipeline ────────────────────────────────────────────────────────────
def run_intent(session_id: int, text: str) -> dict:
    return post("/chat/intent", {"session_id": session_id, "text": text})

def run_hypothesis(session_id: int, intent: dict) -> dict:
    return post("/chat/hypothesis", {"session_id": session_id, "intent": intent})

def run_plan(session_id: int, intent: dict, hypothesis: Any = None, graph: dict = None) -> dict:
    body: dict = {"session_id": session_id, "intent": intent, "hypothesis": hypothesis}
    if graph:
        body["graph"] = graph
    return post("/chat/plan", body)

def run_analyze(session_id: int, focus: str = "overall progress and next steps") -> dict:
    return post("/chat/analyze", {"session_id": session_id, "focus": focus})

# ── Knowledge / Literature ────────────────────────────────────────────────────
def search_knowledge(session_id: int, query: str, limit: int = 10) -> dict:
    return post("/chat/knowledge", {"session_id": session_id, "query": query, "limit": limit})

# ── Agent execution ───────────────────────────────────────────────────────────
def run_agent(agent_name: str, session_id: int, payload: dict) -> dict:
    return post(f"/agent/{agent_name}", {"session_id": session_id, **payload})

# ── QA & Benchmarking ─────────────────────────────────────────────────────────
def qa_functional(system_desc: str, session_id: int = None) -> dict:
    return post("/chat/qa/functional", {"system": system_desc, "session_id": session_id})

def qa_surface(material: str, facet: str, conditions: dict = None) -> dict:
    return post("/chat/qa/surface", {"material": material, "facet": facet,
                                      "conditions": conditions or {}})

def qa_debug_outcar(job_dir: str, session_id: int = None) -> dict:
    return post("/chat/qa/debug", {"job_dir": job_dir, "session_id": session_id})

def qa_free_energy(session_id: int, reaction: str, temperature: float = 298.15,
                   potential_V: float = 0.0, use_known: bool = True,
                   pathways: list = None) -> dict:
    return post("/chat/qa/free_energy", {
        "session_id": session_id, "reaction": reaction,
        "temperature": temperature, "potential_V": potential_V,
        "use_known_pathway": use_known,
        **({"pathways": pathways} if pathways else {}),
    })

def qa_microkinetics(session_id: int, reaction: str, temperature: float = 500.0,
                     potential_V: float = -0.8) -> dict:
    return post("/chat/qa/microkinetics", {
        "session_id": session_id, "reaction": reaction,
        "temperature": temperature, "potential_V": potential_V,
        "use_known_pathway": True,
    })

def qa_neb_prep(session_id: int, reaction: str, surface: str) -> dict:
    return post("/chat/qa/neb_prep", {
        "session_id": session_id, "reaction": reaction, "surface": surface,
    })

# ── Structure agent ─────────────────────────────────────────────────────────
def structure_build_slab(element: str, miller: str, layers: int = 4,
                          vacuum: float = 15.0, supercell: str = "4x4x1",
                          crystal_system: str = "") -> dict:
    return post("/agent/structure/build_slab", {
        "element": element, "facet": miller, "layers": layers,
        "vacuum": vacuum, "supercell": supercell, "crystal_system": crystal_system,
    })

def structure_find_sites(poscar_content: str, height: float = 2.0) -> dict:
    return post("/agent/structure/find_sites", {
        "poscar": poscar_content, "height": height,
    })

def structure_place_adsorbate(poscar_content: str, adsorbate: str,
                               site_index: int = 0, height: float = 2.0) -> dict:
    return post("/agent/structure/place_adsorbate", {
        "poscar": poscar_content, "adsorbate": adsorbate,
        "site_index": site_index, "height": height,
    })

def structure_generate_configs(poscar_content: str, adsorbate: str,
                                max_configs: int = 4) -> dict:
    return post("/agent/structure/generate_configs", {
        "poscar": poscar_content, "adsorbate": adsorbate, "max_configs": max_configs,
    })

def structure_build_molecule(smiles: str, label: str = "", cell_size: float = 20.0) -> dict:
    return post("/agent/structure/build_molecule", {
        "smiles": smiles, "label": label, "cell_size": cell_size,
    })

def structure_build_complex(
    metal: str,
    ligand: str,
    n_coord: int = 4,
    geometry: str = "square_planar",
    bond_length: float = 2.0,
    cell_size: float = 15.0,
    session_id: Optional[int] = None,
) -> dict:
    """Build a coordination compound (metal center + monodentate ligands).

    Supported ligands: H2O, NH3, CO, Cl, F, CN, NO, OH, PH3, SCN
    Supported geometries: linear, trigonal_planar, tetrahedral, square_planar,
                          trigonal_bipyramidal, octahedral
    """
    return post("/agent/structure/build_complex", {
        "metal": metal, "ligand": ligand, "n_coord": n_coord,
        "geometry": geometry, "bond_length": bond_length,
        "cell_size": cell_size, "session_id": session_id,
    })

def structure_deprotonate(poscar_content: str, n_remove: int = 1,
                          site: str = "terminal") -> dict:
    """Remove n_remove H atoms from a molecule POSCAR.
    site: 'terminal' (O-H / N-H, furthest from backbone),
          'surface' (closest to H-centroid),
          'random' (first n H atoms in list).
    """
    return post("/agent/structure/deprotonate", {
        "poscar": poscar_content, "n_remove": n_remove, "site": site,
    })


def structure_build_surface(element: str, surface_type: str = "111",
                             nx: int = 4, ny: int = 4, nlayers: int = 3,
                             vacuum: float = 10.0, fix_bottom: bool = True) -> dict:
    return post("/agent/structure/build_surface", {
        "element": element, "surface_type": surface_type,
        "nx": nx, "ny": ny, "nlayers": nlayers,
        "vacuum": vacuum, "fix_bottom": fix_bottom,
    })

# ── Parameters agent ─────────────────────────────────────────────────────────
def hpc_fetch(remote_path: str, cluster: str = "hoffman2", user: str = "",
              files: list = None, session_id: int = None) -> dict:
    return post("/agent/hpc/fetch", {
        "remote_path": remote_path, "cluster": cluster, "user": user,
        "files": files or ["CONTCAR", "OUTCAR", "stdout", "OSZICAR"],
        "session_id": session_id,
    })

def generate_script(calc_type: str, system: dict, params: dict = None) -> dict:
    return post("/agent/generate_script", {
        "calc_type": calc_type, "system": system, "params": params or {}
    })

# ── Electronic structure scripts ──────────────────────────────────────────────
def generate_electronic_script(calc_type: str, system: dict = None,
                                kpoints: str = "", encut: int = 400,
                                **kwargs) -> dict:
    """
    Generate an ASE/VASP script for an electronic structure calculation.
    calc_type: 'static' | 'dos' | 'pdos' | 'band' | 'elf' |
               'bader' | 'cdd' | 'work_function' | 'cohp'
    """
    params = {"kpoints": kpoints, "encut": encut, **kwargs}
    return post("/agent/generate_script", {
        "calc_type": calc_type, "system": system or {}, "params": params
    })

def get_incar_preset(calc_type: str) -> dict:
    """Fetch the VASP INCAR parameter preset for a given calc_type."""
    return post("/agent/incar_preset", {"calc_type": calc_type})

def parameters_analyze_benchmarks(bench_dir: str) -> dict:
    return post("/agent/parameters/analyze_benchmarks", {"bench_dir": bench_dir})

# ── Slab manipulation ─────────────────────────────────────────────────────────
def slab_add_layer(poscar: str) -> dict:
    return post("/agent/structure/add_layer", {"poscar": poscar})

def slab_delete_layer(poscar: str) -> dict:
    return post("/agent/structure/delete_layer", {"poscar": poscar})

def slab_set_vacuum(poscar: str, vacuum: float = 15.0) -> dict:
    return post("/agent/structure/set_vacuum", {"poscar": poscar, "vacuum": vacuum})

def slab_dope(poscar: str, host: str, dopant: str,
              n_dopants: int = 1, site: str = "surface") -> dict:
    return post("/agent/structure/dope", {
        "poscar": poscar, "host_element": host,
        "dopant_element": dopant, "n_dopants": n_dopants, "site": site,
    })

def slab_make_symmetric(poscar: str, vacuum: float = 20.0) -> dict:
    return post("/agent/structure/make_symmetric", {"poscar": poscar, "vacuum": vacuum})

def build_interface(poscar_a: str, poscar_b: str,
                    vacuum: float = 15.0, interface_gap: float = 2.2,
                    strain_a: bool = False) -> dict:
    return post("/agent/structure/build_interface", {
        "poscar_a": poscar_a, "poscar_b": poscar_b,
        "vacuum": vacuum, "interface_gap": interface_gap, "strain_a": strain_a,
    })

def generate_neb_images(is_poscar: str, fs_poscar: str,
                        n_images: int = 6, method: str = "linear") -> dict:
    return post("/agent/structure/generate_neb_images", {
        "is_poscar": is_poscar, "fs_poscar": fs_poscar,
        "n_images": n_images, "method": method,
    })

# ── HTP dataset generation ────────────────────────────────────────────────────
def htp_generate(
    base_structures: List[Dict],
    strategy: str = "rattle",
    n_total: int = 1000,
    db_path: str = "htp_dataset.db",
    **kwargs,
) -> dict:
    """Generate diverse structures for NNP training dataset.

    Parameters
    ----------
    base_structures : list of {"poscar": str, "label": str}
    strategy        : rattle | strain | rattle_strain | surface_rattle |
                      temperature_rattle | alloy | vacancy
    n_total         : total number of structures to generate
    db_path         : path to ASE database file
    **kwargs        : strategy-specific kwargs (stdev, strains, host, dopant, etc.)
    """
    return post("/agent/htp/generate", {
        "base_structures": base_structures,
        "strategy": strategy,
        "n_total": n_total,
        "db_path": db_path,
        **kwargs,
    })


def htp_stats(db_path: str = "htp_dataset.db") -> dict:
    """Get dataset statistics from HTP database."""
    return post("/agent/htp/stats", {"db_path": db_path})


def htp_export(
    db_path: str = "htp_dataset.db",
    output_path: str = "training.xyz",
    only_done: bool = True,
) -> dict:
    """Export extXYZ training set from HTP database."""
    return post("/agent/htp/export", {
        "db_path": db_path,
        "output_path": output_path,
        "only_done": only_done,
    })


def htp_script(
    db_path: str = "htp_dataset.db",
    encut: int = 450,
    kpoints: str = "4 4 1",
    batch_size: int = 50,
    scheduler: str = "sge",
) -> dict:
    """Generate HTP batch job script for cluster submission."""
    return post("/agent/htp/script", {
        "db_path": db_path,
        "encut": encut,
        "kpoints": kpoints,
        "batch_size": batch_size,
        "scheduler": scheduler,
    })


# ── Calc profiles ─────────────────────────────────────────────────────────────
def get_calc_profile(profile: str) -> dict:
    """Fetch a named VASP parameter profile from calc_profiles.yaml via the server."""
    return post("/agent/parameters/profile", {"profile": profile})

def list_calc_profiles() -> dict:
    """List all available calc profile names."""
    return get("/agent/parameters/profiles")

# ── HPC job watcher ───────────────────────────────────────────────────────────
def hpc_watch(
    task_id: int,
    job_id: str,
    job_dir: str,
    session_id: int,
    cluster: str = "hoffman2",
    species: str = "",
    surface: str = "",
    poll_interval: int = 60,
) -> dict:
    """Start a background watcher that monitors an HPC job and triggers feedback."""
    return post("/agent/hpc/watch", {
        "task_id": task_id, "job_id": job_id, "job_dir": job_dir,
        "session_id": session_id, "cluster": cluster,
        "species": species, "surface": surface,
        "poll_interval": poll_interval,
    })

# ── Persistent task state ─────────────────────────────────────────────────────
def save_task_state(session_id: int, task_plan_id: int, **kwargs) -> dict:
    body = {"session_id": session_id, "task_plan_id": task_plan_id, **kwargs}
    return post("/chat/task_state/save", body)

def list_task_states(session_id: int) -> dict:
    return post("/chat/task_state/list", {"session_id": session_id})

# ── Hypothesis feedback ───────────────────────────────────────────────────────
def hypothesis_feedback(session_id: int, result_type: str, species: str,
                        surface: str, value: float, converged: bool = True,
                        extra: dict = None) -> dict:
    return post("/chat/hypothesis/feedback", {
        "session_id": session_id, "result_type": result_type,
        "species": species, "surface": surface, "value": value,
        "converged": converged, "extra": extra or {},
    })

# ── Monitoring dashboard ────────────────────────────────────────────────────
def dashboard_overview(window_minutes: int = 60, n_buckets: int = 12) -> dict:
    return get("/dashboard/overview", {
        "window_minutes": window_minutes, "n_buckets": n_buckets,
    })

def dashboard_help() -> dict:
    return get("/dashboard/help", {})

def dashboard_alerts(window_minutes: int = 60) -> dict:
    return get("/dashboard/alerts", {"window_minutes": window_minutes})

# ── Closed-loop orchestrator ────────────────────────────────────────────────
def orchestrator_start(
    session_id: int,
    *,
    max_iterations: int = 10,
    confidence_threshold: float = 0.85,
    no_new_actions_threshold: int = 2,
    auto_submit: bool = False,
    cluster: str = "hoffman2",
    engine: str = "vasp",
) -> dict:
    return post("/api/orchestrator/start", {
        "session_id": session_id,
        "max_iterations": max_iterations,
        "confidence_threshold": confidence_threshold,
        "no_new_actions_threshold": no_new_actions_threshold,
        "auto_submit": auto_submit,
        "cluster": cluster,
        "engine": engine,
    })

def orchestrator_status(run_id: int) -> dict:
    return post("/api/orchestrator/status", {"run_id": run_id})

def orchestrator_stop(run_id: int) -> dict:
    return post("/api/orchestrator/stop", {"run_id": run_id})

def orchestrator_runs(session_id: int, limit: int = 20) -> dict:
    return post("/api/orchestrator/runs",
                {"session_id": session_id, "limit": limit})
