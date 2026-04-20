# server/skills/execution/submit_and_wait.py
"""Skill: submit a prepared VASP job to HPC and wait for completion."""

from __future__ import annotations

from server.skills.base import Domain, FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill


@register_skill
class SubmitAndWait(Skill):
    spec = SkillSpec(
        name="execution.submit_and_wait",
        description="Submit a prepared VASP job directory to HPC, wait for completion, and fetch outputs.",
        domain=Domain.EXECUTION,
        inputs={
            "cluster": FieldSpec(type="str", description="HPC cluster name", default="hoffman2", required=False),
            "ntasks": FieldSpec(type="int", description="Number of CPU tasks", default=32, required=False),
            "walltime": FieldSpec(type="str", description="Wall time limit", default="24:00:00", required=False),
            "poll_interval": FieldSpec(
                type="int", description="Polling interval in seconds", default=60, required=False
            ),
            "dry_run": FieldSpec(
                type="bool",
                description="If True, only generate scripts without submitting",
                default=False,
                required=False,
            ),
        },
        outputs={
            "job_id": FieldSpec(type="str", description="HPC job ID"),
            "status": FieldSpec(type="str", description="Final job status"),
            "output_files": FieldSpec(type="list", description="List of fetched output file names"),
        },
        cost=1,
        side_effects=["submits_hpc_job", "fetches_remote_files"],
        requires=["parameters.vasp_relaxation"],
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        from pathlib import Path

        from server.execution.hpc_agent import HPCAgent

        job_dir = ctx.job_dir
        if job_dir is None:
            return SkillResult(ok=False, error="job_dir is required for execution skill.")

        inp = ctx.inputs
        cluster = inp.get("cluster", "hoffman2")
        ntasks = int(inp.get("ntasks", 32))
        walltime = inp.get("walltime", "24:00:00")
        poll = int(inp.get("poll_interval", 60))
        dry_run = bool(inp.get("dry_run", False))

        agent = HPCAgent(
            cluster=cluster,
            ntasks=ntasks,
            time_limit=walltime,
            dry_run=dry_run,
        )

        if dry_run:
            return SkillResult(
                ok=True,
                outputs={"job_id": "DRY_RUN", "status": "prepared", "output_files": []},
            )

        try:
            job_id = agent.submit(job_dir)
        except Exception as exc:
            return SkillResult(ok=False, error=f"submit failed: {exc}")

        try:
            agent.wait(job_id, poll=poll)
        except Exception as exc:
            return SkillResult(
                ok=False,
                outputs={"job_id": job_id, "status": "timeout_or_error", "output_files": []},
                error=f"wait failed: {exc}",
            )

        try:
            agent.fetch_outputs(job_dir)
        except Exception as exc:
            return SkillResult(
                ok=True,
                outputs={"job_id": job_id, "status": "done_fetch_failed", "output_files": []},
                error=f"fetch warning: {exc}",
            )

        output_files = [
            f.name
            for f in Path(job_dir).iterdir()
            if f.is_file()
            and f.name
            in (
                "OUTCAR",
                "vasprun.xml",
                "OSZICAR",
                "CONTCAR",
                "DOSCAR",
                "EIGENVAL",
                "CHGCAR",
                "PROCAR",
            )
        ]

        return SkillResult(
            ok=True,
            outputs={"job_id": job_id, "status": "done", "output_files": output_files},
        )
