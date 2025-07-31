from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # 显式读取密钥更安全

def generate_poscar_from_prompt(prompt: str, system_message: str = None) -> str:
    if system_message is None:
        system_message = (
            "You are a VASP expert. Given a user's instruction, generate a VASP POSCAR file "
            "for a slab structure. Use Cartesian coordinates and include Selective Dynamics if requested. "
            "Return only the POSCAR content in correct format."
        )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    user_prompt = (
        "请基于 FCC 铜晶格（晶格常数 3.615 Å），使用 Miller 指数 (1 1 1)，生成一个 3 层 ABC 堆叠的 Cu slab 表面，使用 3×3 表面超胞，并添加 15 Å 真空层。使用 ASE 从 bulk 构建并输出标准 VASP POSCAR 文件，采用 Cartesian 坐标和 Selective Dynamics，最底层固定（F F F），其余原子松弛（T T T）。"
    )
    poscar = generate_poscar_from_prompt(user_prompt)
    
    with open("POSCAR", "w") as f:
        f.write(poscar + "\n")
    
    print("✅ POSCAR saved to file.")