# intent agent 要实现的功能：
# 1. 读取历史intent（意图识别的上下文/历史）
# 2. 新的 intent 或同类 intent 的新表达法经过确认后自动存入数据库，形成“意图短语库”或“高频表述归档”；
# 3. 以后可以用于检索/推荐/自我强化（甚至 finetune/active learning）

# input：
# - 用户查询（自然语言）
# - 历史对话（可选）
# - 上下文（可选）
# - 其他元数据（可选）

# output：
# - intent: 意图分类（如 "material_search", "structure_building"）
# - confidence: 置信度（0.0-1.0）
# - matched_phrase: 匹配到的短语（可选，便于归因/优化）
# {
#   "intent": "structure_building",
#   "confidence": 0.93,
#   "object": "Cs2CO3",
#   "task_type": "structure_relaxation",
#   "matched_phrase": "relax Cs2CO3 structure"
# }

from fastapi import APIRouter, Request
from pydantic import BaseModel
from server.utils.openai_wrapper import chatgpt_call  # 你的 wrapper
from server.db import AsyncSessionLocal, IntentPhrase
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
import re, json, asyncio

router = APIRouter()

# 1. 定义主 intent schema（类别，优先以代码 schema 为主，便于维护和统一）
CATEGORIES = [
    ("material_search", "find/query a material or download structure"),
    ("structure_building", "build/edit structure, slabs, adsorbates, doping, alloy, TS, etc."),
    ("param_explanation", "explain parameter meaning"),
    ("param_generation", "generate all input parameters for a run"),
    ("param_suggestion", "recommend parameters for a given task"),
    ("param_benchmark", "compare settings or suggest best-practice"),
    ("job_submission", "submit jobs (general)"),
    ("job_cloud", "submit jobs on cloud (AWS, Aliyun, etc.)"),
    ("job_hpc", "submit jobs to HPC/cluster"),
    ("error_analysis", "analyze error message/log"),
    ("postprocess_bader", "post-process: bader analysis"),
    ("postprocess_dos", "post-process: density of states"),
    ("postprocess_elf", "post-process: electron localization function"),
    ("postprocess_charge_density", "post-process: charge density"),
    ("postprocess_charge_difference", "post-process: charge difference"),
    ("postprocess_band", "post-process: band structure"),
    ("other", "other or unclear"),
]
intent_keys = [k for k, _ in CATEGORIES]
intent_desc = "\n".join([f'- "{k}": {desc}' for k, desc in CATEGORIES])

class IntentResult(BaseModel):
    intent: str
    confidence: float

# 2. few-shot prompt 动态拼接，提升鲁棒性
async def get_fewshot_examples(limit=10):
    # 动态抓一些历史高置信度表达做 prompt
    async with AsyncSessionLocal() as session:
        q = await session.execute(
            select(IntentPhrase).where(IntentPhrase.confidence >= 0.7).order_by(IntentPhrase.created_at.desc()).limit(limit)
        )
        return [
            (row.intent, row.phrase) for row in q.scalars().all()
        ]

# 3. intent LLM 推理
async def call_gpt4o_intent(query: str) -> dict:
    # 拼 few-shot
    fewshots = await get_fewshot_examples(10)
    fewshot_str = ""
    if fewshots:
        fewshot_str = "\n".join([f'Q: {ex[1]}\nIntent: {ex[0]}' for ex in fewshots])
    PROMPT = f"""
You are an intent classification agent for a DFT copilot. The possible intent categories are:
{intent_desc}

Your job: For each user query, assign one and only one intent from the above.
If you see similar questions as below, follow their intent:

{fewshot_str}

For this query, output strict JSON only:
{{"intent": "...", "confidence": 0.0}}
Do not explain. Output JSON only.
Q: {query}
"""
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": query}
    ]
    text = await chatgpt_call(messages)
    text = text.strip()
    # 处理代码块
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        text = match.group()
    try:
        obj = json.loads(text)
        # 强制类型合法性
        if obj.get("intent") not in intent_keys:
            obj["intent"] = "other"
        if "confidence" not in obj:
            obj["confidence"] = 0.3
        return obj
    except Exception:
        print("Intent parse failed:", text)
        return {"intent": "other", "confidence": 0.3}

# 4. 入库
async def save_intent_phrase(intent, phrase, confidence, source="user", author=None):
    async with AsyncSessionLocal() as session:
        q = await session.execute(
            select(IntentPhrase).where(
                IntentPhrase.intent == intent,
                IntentPhrase.phrase == phrase,
            )
        )
        exist = q.scalar()
        if not exist:
            try:
                entry = IntentPhrase(
                    intent=intent,
                    phrase=phrase,
                    confidence=confidence,
                    source=source,
                    author=author,
                )
                session.add(entry)
                await session.commit()
            except IntegrityError:
                await session.rollback()

@router.post("/chat/intent", response_model=IntentResult)
async def chat_intent(request: Request):
    data = await request.json()
    user_query = data["query"]
    intent_obj = await call_gpt4o_intent(user_query)
    # 写入 intent_phrase 表
    await save_intent_phrase(
        intent=intent_obj["intent"],
        phrase=user_query,
        confidence=intent_obj.get("confidence", 1.0)
    )
    # 你可以在这里 return intent_obj，也可以同时写入 chat_message.intent 字段
    return intent_obj