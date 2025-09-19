# server/utils/openai_wrapper.py
from openai import OpenAI
client = OpenAI()

# If this could be repeat for the whole things 

# Here I need to replace everything with deepseek API OK!!!

async def chatgpt_call(messages, model="gpt-5", temperature=0.0, max_tokens=100000, **kwargs):
    
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},  # 关键
    )
    # 返回原始 dict，供 _json_from_llm_raw 解析
    return response.to_dict() if hasattr(response, "to_dict") else response