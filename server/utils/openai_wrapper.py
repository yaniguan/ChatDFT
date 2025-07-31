import os
from openai import AsyncOpenAI, OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# 异步版本
async def chatgpt_call(messages, model="gpt-4o", temperature=0.2, max_tokens=1024):
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()