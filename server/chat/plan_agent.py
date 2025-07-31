from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional
from server.db import AsyncSessionLocal, WorkflowTask

router = APIRouter()

class Step(BaseModel):
    id: int
    name: str
    description: str
    agent: str
    intent: Optional[str] = None

class PlanResult(BaseModel):
    ok: bool
    tasks: List[Step]

# ---- 1. 内置 workflow map（你可以持续扩展） ----

ELECTRONIC_WORKFLOW = [
    {"id": 1, "name": "Get Structure", "description": "Obtain target material structure.", "agent": "material", "intent": "material_search"},
    {"id": 2, "name": "Geometry Optimization", "description": "Relax structure to minimum energy.", "agent": "job", "intent": "structure_building"},
    {"id": 3, "name": "SCF Calculation", "description": "Run high-precision SCF calculation.", "agent": "job", "intent": "param_benchmark"},
    {"id": 4, "name": "Electronic Analysis", "description": "Calculate DOS/band/charge as needed.", "agent": "job", "intent": "param_generation"},
    {"id": 5, "name": "Post-processing", "description": "Analyze outputs (DOS, band, Bader etc).", "agent": "post", "intent": "postprocess_dos"},
    {"id": 6, "name": "Generate Report", "description": "Summarize and visualize results.", "agent": "report", "intent": "other"}
]

GCE_WORKFLOW = [
    {"id": 1, "name": "Get Catalyst Structure", "description": "Download/build catalyst slab.", "agent": "material", "intent": "material_search"},
    {"id": 2, "name": "Surface Opt", "description": "Relax slab/adsorbate structure.", "agent": "job", "intent": "structure_building"},
    {"id": 3, "name": "Symmetry Analysis", "description": "Check/reduce symmetry for surface.", "agent": "job", "intent": "param_suggestion"},
    {"id": 4, "name": "Implicit Solvation", "description": "Apply continuum solvent & optimize.", "agent": "job", "intent": "param_generation"},
    {"id": 5, "name": "Surface Charging (GCE)", "description": "Tune charge/run GCE-DFT.", "agent": "job", "intent": "job_submission"},
    {"id": 6, "name": "Post-processing", "description": "Analyze charge, energy, workfunction.", "agent": "post", "intent": "postprocess_charge_density"},
    {"id": 7, "name": "Generate Report", "description": "Summarize results and plots.", "agent": "report", "intent": "other"}
]

NEB_WORKFLOW = [
    {"id": 1, "name": "Get Initial Structure", "description": "Obtain initial/final structures.", "agent": "material", "intent": "material_search"},
    {"id": 2, "name": "Transition State Path", "description": "Set up NEB or CI-NEB path.", "agent": "job", "intent": "param_generation"},
    {"id": 3, "name": "NEB Calculation", "description": "Run NEB to search TS.", "agent": "job", "intent": "job_submission"},
    {"id": 4, "name": "Post-process NEB", "description": "Analyze TS, barriers, visualize path.", "agent": "post", "intent": "postprocess_band"},
    {"id": 5, "name": "Generate Report", "description": "Summarize reaction path.", "agent": "report", "intent": "other"}
]

WORKFLOW_MAP = {
    "postprocess_dos": ELECTRONIC_WORKFLOW,
    "electronic_structure": ELECTRONIC_WORKFLOW,
    "gce_dft": GCE_WORKFLOW,
    "grand_canonical": GCE_WORKFLOW,
    "neb": NEB_WORKFLOW,
    "transition_state": NEB_WORKFLOW,
    # ...你可以继续扩展
}

def match_workflow(intent, query=""):
    intent = (intent or '').strip().lower()
    q = (query or "").lower()
    print("[DEBUG] intent:", intent)
    for k, workflow in WORKFLOW_MAP.items():
        print("[DEBUG] compare to:", k)
        if (intent and intent == k) or (k in q):
            print("[DEBUG] Matched:", k)
            return workflow
    print("[DEBUG] No workflow matched.")
    return None

# ---- 2. LLM planner fallback ----
from server.utils.openai_wrapper import chatgpt_call

PLANNER_SYSTEM_PROMPT = """
You are a scientific workflow planner for DFT research. 
Given user's query, intent, hypothesis, output a workflow as structured JSON steps.
Each step: id, name, description, agent, (optional) intent.
Output strict JSON:
{
  "ok": true,
  "tasks": [
    {"id": 1, "name": "...", "description": "...", "agent": "...", "intent": "..."},
    ...
  ]
}
No explanation.
"""

async def call_gpt4o_planner(query, intent, hypothesis):
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\nIntent: {intent}\nHypothesis: {hypothesis}"}
    ]
    import json
    text = await chatgpt_call(messages)
    try:
        res = json.loads(text)
        return res
    except Exception:
        return {"ok": False, "tasks": []}

# ---- 3. API endpoint ----

@router.post("/chat/plan", response_model=PlanResult)
async def chat_plan(request: Request):
    data = await request.json()
    query = data.get("query", "")
    intent = data.get("intent", "")
    hypothesis = data.get("hypothesis", "")
    session_id = data.get("session_id")
    # ① 规则优先
    tasks = match_workflow(intent, query)
    result = None
    if tasks:
        result = {"ok": True, "tasks": tasks}
    else:
        # ② LLM fallback
        result = await call_gpt4o_planner(query, intent, hypothesis)
    # ③ 插入到 DB
    async with AsyncSessionLocal() as session:
        if result.get("ok") and result.get("tasks"):
            for t in result["tasks"]:
                wf = WorkflowTask(
                    session_id=session_id,
                    step_id=t["id"],
                    name=t["name"],
                    description=t["description"],
                    agent=t["agent"],
                    intent=t.get("intent"),
                    input_data={},  # 可根据实际补充
                    status="planned"
                )
                session.add(wf)
            await session.commit()
    return result