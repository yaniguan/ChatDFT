from openai import OpenAI

client = OpenAI()  # 使用环境变量中的 OPENAI_API_KEY

def generate_incar(task_type, material):
    prompt = f"""
You are an expert in VASP input file generation.

Generate a reasonable INCAR file for a VASP calculation for the following task:
Task type: {task_type}
Material: {material}

Only return the INCAR content, no explanation.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()