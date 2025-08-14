# server/utils/openai_wrapper.py
from openai import OpenAI
client = OpenAI()

async def chatgpt_call(messages, model="gpt-4o", temperature=0.1, max_tokens=1800, **kwargs):
    # 如果可以，开启 JSON 模式（新 SDK）
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},  # 关键
    )
    # 返回原始 dict，供 _json_from_llm_raw 解析
    return response.to_dict() if hasattr(response, "to_dict") else response