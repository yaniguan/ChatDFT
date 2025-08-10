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