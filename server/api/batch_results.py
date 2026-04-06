# server/api/batch_results.py
"""
Batch results: poll job statuses, pull results, export to Excel.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server import sshio, parser
from server.settings import get_server

_log = logging.getLogger("chatdft.batch_results")

router = APIRouter(prefix="/api", tags=["batch-results"])


class JobStatus(BaseModel):
    job_uid: str
    title: str
    metal: str
    site: str
    adsorbate: str
    is_reference: bool
    pbs_id: str
    status: str
    energy: Optional[float] = None
    max_force: Optional[float] = None
    converged: Optional[bool] = None
    local_dir: str = ""


class BatchStatusResponse(BaseModel):
    ok: bool
    batch_uid: str
    n_total: int = 0
    n_done: int = 0
    n_running: int = 0
    n_queued: int = 0
    all_done: bool = False
    jobs: List[JobStatus] = []
    error: str = ""


# In-memory batch store (for MVP; production would use DB)
_batches: Dict[str, List[Dict[str, Any]]] = {}


def register_batch(batch_uid: str, jobs: List[Dict[str, Any]]):
    """Called by batch_adsorption to register jobs for tracking."""
    _batches[batch_uid] = jobs


@router.get("/batch_status", response_model=BatchStatusResponse)
async def batch_status(batch_uid: str = Query(...)):
    """Poll status of all jobs in a batch."""
    jobs = _batches.get(batch_uid)
    if jobs is None:
        return BatchStatusResponse(ok=False, batch_uid=batch_uid, error="Batch not found")

    svr = None
    try:
        svr = get_server("hoffman2")
    except Exception:
        pass

    statuses = []
    for job in jobs:
        pbs_id = job.get("pbs_id", "")
        current_status = job.get("status", "unknown")

        # Poll if not yet done
        if current_status not in ("done", "synced", "failed"):
            try:
                new_status = sshio.poll_status(pbs_id, svr)
                job["status"] = new_status
                current_status = new_status
            except Exception as e:
                _log.warning("Poll failed for %s: %s", pbs_id, e)

        energy = job.get("energy")
        max_force = job.get("max_force")
        converged = job.get("converged")

        # Pull results if done
        if current_status == "done" and job.get("local_dir"):
            try:
                sshio.pull_results(job["job_uid"], job["local_dir"], svr)
                rows = parser.parse_job_to_rows(job["local_dir"])
                if rows:
                    energy = rows[-1].get("energy")
                    info = rows[-1].get("info", {})
                    max_force = info.get("max_force")
                    converged = info.get("converged")
                    job["energy"] = energy
                    job["max_force"] = max_force
                    job["converged"] = converged
                job["status"] = "synced"
                current_status = "synced"
            except Exception as e:
                _log.warning("Pull failed for %s: %s", job["title"], e)

        statuses.append(JobStatus(
            job_uid=job["job_uid"],
            title=job["title"],
            metal=job.get("metal", ""),
            site=job.get("site", ""),
            adsorbate=job.get("adsorbate", ""),
            is_reference=job.get("is_reference", False),
            pbs_id=pbs_id,
            status=current_status,
            energy=energy,
            max_force=max_force,
            converged=converged,
            local_dir=job.get("local_dir", ""),
        ))

    done_statuses = {"done", "synced", "failed"}
    n_done = sum(1 for s in statuses if s.status in done_statuses)
    n_running = sum(1 for s in statuses if s.status == "running")
    n_queued = sum(1 for s in statuses if s.status in ("queued", "submitted"))

    return BatchStatusResponse(
        ok=True,
        batch_uid=batch_uid,
        n_total=len(statuses),
        n_done=n_done,
        n_running=n_running,
        n_queued=n_queued,
        all_done=(n_done == len(statuses)),
        jobs=statuses,
    )


@router.get("/batch_results_excel")
async def batch_results_excel(batch_uid: str = Query(...)):
    """Export batch results as Excel file with adsorption energies."""
    import pandas as pd

    jobs = _batches.get(batch_uid)
    if not jobs:
        return {"ok": False, "error": "Batch not found"}

    # Collect energies
    ref_energies = {}  # metal -> slab energy
    gas_energy = {}    # molecule -> energy
    ads_data = []

    for job in jobs:
        e = job.get("energy")
        if e is None:
            continue

        if job.get("is_reference"):
            if job["site"] == "gas":
                gas_energy[job["adsorbate"]] = e
            else:
                ref_energies[job["metal"]] = e
        else:
            ads_data.append({
                "Metal": job["metal"],
                "Facet": job.get("facet", "111"),
                "Site": job["site"],
                "Adsorbate": job["adsorbate"],
                "E_total (eV)": e,
                "Max Force (eV/A)": job.get("max_force"),
                "Converged": job.get("converged"),
            })

    df = pd.DataFrame(ads_data)

    # Compute adsorption energy: E_ads = E(slab+H) - E(slab) - 0.5*E(H2)
    e_h2 = gas_energy.get("H2")
    if e_h2 is not None and not df.empty:
        def calc_eads(row):
            e_slab = ref_energies.get(row["Metal"])
            if e_slab is not None:
                return row["E_total (eV)"] - e_slab - 0.5 * e_h2
            return None
        df["E_ads (eV)"] = df.apply(calc_eads, axis=1)

    # Add reference data as separate sheet
    ref_rows = []
    for metal, e in ref_energies.items():
        ref_rows.append({"System": f"{metal}(111) clean slab", "Energy (eV)": e})
    for mol, e in gas_energy.items():
        ref_rows.append({"System": f"{mol} gas-phase", "Energy (eV)": e})
    df_ref = pd.DataFrame(ref_rows)

    # Write to Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Adsorption", index=False)
        df_ref.to_excel(writer, sheet_name="References", index=False)
    buf.seek(0)

    # Also save locally
    local_path = os.path.join("runs", f"batch_{batch_uid[:8]}_results.xlsx")
    os.makedirs("runs", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(buf.getvalue())
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=H_adsorption_results.xlsx"},
    )
