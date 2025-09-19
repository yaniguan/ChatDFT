import csv
import os
import time
import json
# 导入新的 OpenAI 客户端和错误类型
from openai import OpenAI, OpenAIError

# --- Real API Configuration ---
# 请确保您已经将 OPENAI_API_KEY 设置为环境变量
# 这样可以避免将您的密钥直接写入代码中
try:
    # 初始化 OpenAI 客户端
    # 客户端会自动从环境变量 OPENAI_API_KEY 读取密钥
    client = OpenAI()
except KeyError:
    print("错误：未设置 OPENAI_API_KEY 环境变量。")
    print("请在运行脚本前设置该变量。")
    exit()

def get_structured_output_from_api(inquiry):
    """
    调用 OpenAI API 以获取查询的结构化输出 (意图、假设、方案)。

    此函数会:
    1.  构建一个明确的、要求 JSON 输出的提示 (Prompt)。
    2.  向 OpenAI API 发送 HTTP POST 请求。
    3.  处理 JSON 响应和潜在的错误。
    """
    print(f"正在处理查询: '{inquiry[:50]}...'")

    # --- 提示工程 (Prompt Engineering) 示例 ---
    # 这里是您精心设计，用于指导模型生成结构化 JSON 的提示
    system_prompt = f"""
    You are a world-class expert in computational chemistry and materials science, specifically using VASP. Your role is to act as an intelligent agent that analyzes a user's inquiry and breaks it down into a structured computational workflow.

    Given the user's inquiry, you MUST generate a valid JSON object with the following four keys:
    1. "intent_classification": Classify the primary goal into ONE of the following categories: [Structure Optimization, Adsorption Energy Calculation, Reaction Pathway Analysis (NEB), Electronic Structure Analysis, Surface Modeling, Thermodynamic Stability Analysis, Molecular Dynamics (MD), Defect Calculation, Electrochemical Analysis, Property Prediction].
    2. "refined_intent": A single, clear sentence describing the specific scientific objective. This connects the inquiry to the hypothesis.
    3. "hypothesis": A plausible scientific hypothesis that the calculation is designed to test. It should state an expected outcome or relationship.
    4. "computational_plan": A concise, step-by-step list of the key VASP calculations required to test the hypothesis.

    Example Inquiry: "Where are the most stable adsorption sites for CO on Cu(100)?"
    Example JSON Output:
    {{
      "intent_classification": "Adsorption Energy Calculation",
      "refined_intent": "To determine the most energetically favorable adsorption site for a CO molecule on a Cu(100) surface.",
      "hypothesis": "The CO molecule is expected to bind most strongly at the hollow site due to maximal coordination with the surface copper atoms.",
      "computational_plan": [
        "Optimize the bulk structure of Cu.",
        "Create a Cu(100) slab model with appropriate vacuum.",
        "Place a CO molecule at high-symmetry adsorption sites (e.g., atop, bridge, hollow).",
        "Perform geometry optimization for each adsorbate configuration.",
        "Calculate and compare the adsorption energies for each site to identify the most stable configuration."
      ]
    }}

    Now, analyze the following user inquiry.
    Inquiry: "{inquiry}"
    """

    try:
        # --- 真实 API 调用 (更新后) ---
        # 使用新的 client 语法进行调用
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # 推荐使用 gpt-4-turbo 以获得更好的结构化数据生成能力
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate the JSON for this inquiry: {inquiry}"}
            ],
            temperature=0.1,  # 对于结构化输出，使用较低的 temperature
            max_tokens=1000   # 增加 token 限制以容纳更长的 JSON 输出
        )
        # 解析返回的 JSON 字符串 (更新的访问方式)
        structured_data = json.loads(response.choices[0].message.content)
        return structured_data

    except json.JSONDecodeError:
        print("错误：API 返回的不是有效的 JSON。")
        return None
    # --- 错误处理 (更新后) ---
    except OpenAIError as e:
        print(f"发生 OpenAI API 错误: {e}")
        time.sleep(1)
        return None
    except Exception as e:
        print(f"调用 API 时发生未知错误: {e}")
        return None


def process_inquiries(input_file, output_file):
    """
    从 CSV 文件中读取查询，获取它们的结构化输出，然后写入新的 CSV 文件。
    """
    try:
        with open(input_file, mode='r', encoding='utf-8') as infile, \
             open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

            reader = csv.DictReader(infile)
            # 更新输出文件的表头
            fieldnames = ['Domain', 'Inquiry', 'Intent_Classification', 'Refined_Intent', 'Hypothesis', 'Computational_Plan']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            print("开始处理查询...")
            for row in reader:
                inquiry = row.get('Inquiry', '')
                if not inquiry:
                    continue

                # 使用我们的 API 函数获取结构化数据
                output_data = get_structured_output_from_api(inquiry)

                # 准备要写入行的数据
                write_row = {
                    'Domain': row.get('Domain', 'N/A'),
                    'Inquiry': inquiry
                }

                if output_data:
                    # 将计算方案列表转换为带编号的字符串
                    plan_steps = output_data.get('computational_plan', [])
                    plan_str = "; ".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))

                    write_row['Intent_Classification'] = output_data.get('intent_classification', 'Error')
                    write_row['Refined_Intent'] = output_data.get('refined_intent', 'Error')
                    write_row['Hypothesis'] = output_data.get('hypothesis', 'Error')
                    write_row['Computational_Plan'] = plan_str
                else:
                    # 如果 API 调用失败，则填充错误信息
                    write_row['Intent_Classification'] = 'API_Error'
                    write_row['Refined_Intent'] = 'API_Error'
                    write_row['Hypothesis'] = 'API_Error'
                    write_row['Computational_Plan'] = 'API_Error'

                writer.writerow(write_row)

        print(f"\n处理完成。结果已保存至 '{output_file}'。")

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{input_file}'。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_csv = 'chatdft_inquiries.csv'
    output_csv = 'inquiry_structured_outputs.csv' # 更新输出文件名以反映新内容

    # 运行处理函数
    process_inquiries(input_csv, output_csv)

