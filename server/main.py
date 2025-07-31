from fastapi import FastAPI

from server.chat.intent_agent import router as intent_router
from server.chat.hypothesis_agent import router as hypothesis_router
from server.chat.plan_agent import router as plan_router
from server.chat.knowledge_agent import router as knowledge_router
from server.chat.history_agent import router as history_router
from server.chat.session_agent import router as session_router

app = FastAPI()
app.include_router(intent_router)
app.include_router(hypothesis_router)
app.include_router(plan_router)
app.include_router(knowledge_router)
app.include_router(history_router)
app.include_router(session_router)
