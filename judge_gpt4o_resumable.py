import pandas as pd
import json
import time
import os
from openai import OpenAI

# ============================================================
# 配置
# ============================================================

DEEPSEEK_API_KEY = "sk-df2047469caa40a1876d2c373bef1e26"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 文件路径
INPUT_FILE = "gpt4o_results_full.json"
OUTPUT_FILE = "gpt4o_results_full_judged.csv"
PROGRESS_FILE = "judge_progress.txt"  # 记录进度的文件

# ============================================================
# 读取原始数据
# ============================================================

print("正在读取 GPT-4o 结果...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    results = json.load(f)

df = pd.DataFrame(results)
print(f"总记录数: {len(df)}")

# 过滤掉错误记录
valid_df = df[~df['model_answer'].str.contains('ERROR', na=False)].copy()
error_count = len(df) - len(valid_df)
print(f"有效记录: {len(valid_df)}")
print(f"错误记录（已过滤）: {error_count}")

# ============================================================
# 断点续传：加载已完成的进度
# ============================================================

completed_indices = set()

# 方法1：检查是否已有输出文件
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    print(f"\n找到已有判断结果文件，已判断 {len(existing_df)} 条")
    completed_indices = set(existing_df.index)
    final_df = existing_df.copy()
else:
    print("\n未找到已有判断结果，从头开始")
    final_df = pd.DataFrame()

# 方法2：检查进度文件
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        saved_indices = f.read().strip().split(',')
        if saved_indices and saved_indices[0]:
            completed_indices.update([int(i) for i in saved_indices if i])
    print(f"从进度文件加载: {len(completed_indices)} 条")

# 找出未完成的记录
pending_mask = []
for idx in valid_df.index:
    if idx not in completed_indices:
        pending_mask.append(True)
    else:
        pending_mask.append(False)

pending_df = valid_df[pending_mask].copy()
print(f"\n已完成: {len(completed_indices)}")
print(f"剩余: {len(pending_df)}")

if len(pending_df) == 0:
    print("✅ 所有记录已判断完成！")
    exit()

# ============================================================
# 判断函数
# ============================================================

def judge_with_deepseek(question, correct_answer, model_answer, max_retries=3):
    """用 DeepSeek 判断答案是否正确，带重试"""
    
    prompt = f"""判断以下答案是否正确。只输出"正确"或"错误"。

问题: {question}
标准答案: {correct_answer}
学生答案: {model_answer}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            return result == "正确"
        except Exception as e:
            print(f"    判断出错 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避：1,2,4秒
            else:
                print(f"    判断失败，标记为错误")
                return False
    return False

# ============================================================
# 运行判断（每一条立即保存）
# ============================================================

print(f"\n开始判断剩余 {len(pending_df)} 条记录...")
print("="*50)

total_pending = len(pending_df)
success_count = 0

for idx, (original_idx, row) in enumerate(pending_df.iterrows()):
    current_num = len(completed_indices) + idx + 1
    print(f"[{current_num}/{len(valid_df)}] 判断中...", end=" ")
    
    # 判断
    is_correct = judge_with_deepseek(
        row['question'],
        row['correct_answer'],
        row['model_answer']
    )
    
    # 添加到结果
    row_copy = row.copy()
    row_copy['is_correct'] = is_correct
    
    # 添加到 final_df
    final_df = pd.concat([final_df, pd.DataFrame([row_copy])], ignore_index=True)
    
    # 【关键】每判断一条立即保存
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    # 记录进度索引
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"{original_idx},")
    
    if is_correct:
        success_count += 1
        print(f"✓ 正确")
    else:
        print(f"✗ 错误")
    
    # 每 50 条显示一次统计
    if (idx + 1) % 50 == 0:
        print(f"  进度: {current_num}/{len(valid_df)} (本批正确率: {success_count/50:.1%})")
        success_count = 0
    
    # 避免 API 限流
    time.sleep(0.3)

# ============================================================
# 最终统计
# ============================================================

print("\n" + "="*50)
print("✅ 判断完成！")
print(f"总记录: {len(final_df)}/{len(valid_df)}")
print(f"结果已保存到: {OUTPUT_FILE}")

# 删除进度文件
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

# ============================================================
# 准确率统计
# ============================================================

print("\n" + "="*50)
print("GPT-4o 准确率统计")
print("="*50)

persona_names = {
    'neutral': '对照组',
    'scientist': '严谨科学家',
    'influencer': '幽默博主',
    'fake_info': '虚假信息源'
}

for persona, name in persona_names.items():
    subset = final_df[final_df['persona'] == persona]
    if len(subset) > 0:
        total = len(subset)
        correct = subset['is_correct'].sum()
        acc = correct/total if total>0 else 0
        print(f"{name}: {acc:.1%} ({int(correct)}/{total})")

# ============================================================
# 保存统计摘要
# ============================================================

summary = final_df.groupby('persona_description').agg(
    total=('is_correct', 'count'),
    correct=('is_correct', 'sum'),
    accuracy=('is_correct', 'mean')
).reset_index()

summary['accuracy'] = summary['accuracy'].apply(lambda x: f"{x:.1%}")
summary.to_csv("gpt4o_accuracy_summary.csv", index=False)

print("\n统计摘要已保存到: gpt4o_accuracy_summary.csv")