from pathlib import Path
from server.execution.hpc_agent import HPCAgent

job_dir = Path("/tmp/demo_pbs"); job_dir.mkdir(exist_ok=True)
step = {"name": "hello", "engine": "vasp", "ntasks": 1}
hpc  = HPCAgent(cluster="hoffman2")   # 自动识别 PBS
hpc.prepare_script(step, job_dir)
jid = hpc.submit(job_dir)
hpc.wait(jid)
hpc.fetch_outputs(job_dir)
print("✔ finished  →", job_dir)