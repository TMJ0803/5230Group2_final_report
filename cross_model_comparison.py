import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# ============================================================
# 数据整理
# ============================================================

# DeepSeek-R1 结果
deepseek_data = {
    'neutral': {'correct': 518, 'total': 790, 'acc': 0.656},
    'scientist': {'correct': 583, 'total': 790, 'acc': 0.738},
    'influencer': {'correct': 354, 'total': 790, 'acc': 0.448},
    'fake_info': {'correct': 315, 'total': 790, 'acc': 0.399}
}

# GPT-4o 结果（从你的截图）
gpt4o_data = {
    'neutral': {'correct': 502, 'total': 790, 'acc': 0.635},
    'scientist': {'correct': 443, 'total': 659, 'acc': 0.672},
    'influencer': {'correct': 235, 'total': 584, 'acc': 0.402},
    'fake_info': {'correct': 125, 'total': 628, 'acc': 0.199}
}

persona_names = {
    'neutral': '对照组',
    'scientist': '严谨科学家',
    'influencer': '幽默博主',
    'fake_info': '虚假信息源'
}

print("="*70)
print("跨模型对比分析：DeepSeek-R1 vs GPT-4o")
print("="*70)

# ============================================================
# 1. 准确率对比表
# ============================================================

print("\n表 1: 跨模型准确率对比")
print("-"*70)
print(f"{'人格':<15} {'DeepSeek-R1':<15} {'GPT-4o':<15} {'差异':<10} {'优胜者':<10}")
print("-"*70)

for persona, name in persona_names.items():
    ds_acc = deepseek_data[persona]['acc']
    gpt_acc = gpt4o_data[persona]['acc']
    diff = ds_acc - gpt_acc
    winner = "DeepSeek" if diff > 0 else ("GPT-4o" if diff < 0 else "平局")
    print(f"{name:<15} {ds_acc:.1%}        {gpt_acc:.1%}        {diff:+.1%}      {winner:<10}")

# ============================================================
# 2. 准确率降幅对比（相对于对照组）
# ============================================================

print("\n" + "="*70)
print("准确率降幅对比（相对于对照组）")
print("="*70)

control_ds = deepseek_data['neutral']['acc']
control_gpt = gpt4o_data['neutral']['acc']

print(f"\n{'人格':<15} {'DeepSeek降幅':<15} {'GPT-4o降幅':<15} {'差异':<10}")
print("-"*55)

for persona, name in persona_names.items():
    if persona == 'neutral':
        continue
    
    ds_drop = deepseek_data[persona]['acc'] - control_ds
    gpt_drop = gpt4o_data[persona]['acc'] - control_gpt
    diff = ds_drop - gpt_drop
    
    print(f"{name:<15} {ds_drop:+.1%}        {gpt_drop:+.1%}        {diff:+.1%}")

# ============================================================
# 3. 核心发现
# ============================================================

print("\n" + "="*70)
print("核心发现")
print("="*70)

# 3.1 科学家正面效应
ds_boost = deepseek_data['scientist']['acc'] - deepseek_data['neutral']['acc']
gpt_boost = gpt4o_data['scientist']['acc'] - gpt4o_data['neutral']['acc']

print(f"\n1. 科学家正面效应:")
print(f"   DeepSeek-R1: +{ds_boost:.1%}")
print(f"   GPT-4o: +{gpt_boost:.1%}")
if ds_boost > gpt_boost:
    print(f"   → DeepSeek-R1 从正面人格中获益更多 (+{(ds_boost - gpt_boost):.1%})")
    print(f"   → 推理模型更能利用科学思维范式")
else:
    print(f"   → GPT-4o 从正面人格中获益更多")

# 3.2 虚假信息源损害
ds_damage = deepseek_data['neutral']['acc'] - deepseek_data['fake_info']['acc']
gpt_damage = gpt4o_data['neutral']['acc'] - gpt4o_data['fake_info']['acc']

print(f"\n2. 虚假信息源损害程度:")
print(f"   DeepSeek-R1: -{ds_damage:.1%}")
print(f"   GPT-4o: -{gpt_damage:.1%}")
if ds_damage > gpt_damage:
    print(f"   → DeepSeek-R1 更容易被虚假人格诱导 (差异 {(ds_damage - gpt_damage):.1%})")
else:
    print(f"   → GPT-4o 更容易被虚假人格诱导")

# 3.3 幽默博主损害
ds_humor_damage = deepseek_data['neutral']['acc'] - deepseek_data['influencer']['acc']
gpt_humor_damage = gpt4o_data['neutral']['acc'] - gpt4o_data['influencer']['acc']

print(f"\n3. 幽默博主损害程度:")
print(f"   DeepSeek-R1: -{ds_humor_damage:.1%}")
print(f"   GPT-4o: -{gpt_humor_damage:.1%}")

# 3.4 模型稳健性排名
print(f"\n4. 模型稳健性排名（准确率波动范围）:")
ds_range = max([d['acc'] for d in deepseek_data.values()]) - min([d['acc'] for d in deepseek_data.values()])
gpt_range = max([d['acc'] for d in gpt4o_data.values()]) - min([d['acc'] for d in gpt4o_data.values()])
print(f"   DeepSeek-R1: {ds_range:.1%} (从 {min([d['acc'] for d in deepseek_data.values()]):.1%} 到 {max([d['acc'] for d in deepseek_data.values()]):.1%})")
print(f"   GPT-4o: {gpt_range:.1%} (从 {min([d['acc'] for d in gpt4o_data.values()]):.1%} 到 {max([d['acc'] for d in gpt4o_data.values()]):.1%})")

if ds_range > gpt_range:
    print(f"   → DeepSeek-R1 受人格影响更大，GPT-4o 更稳健")
else:
    print(f"   → GPT-4o 受人格影响更大，DeepSeek-R1 更稳健")

# ============================================================
# 4. 统计检验
# ============================================================

print("\n" + "="*70)
print("统计显著性检验（卡方检验）")
print("="*70)

for persona, name in persona_names.items():
    # 构建列联表
    ds_correct = deepseek_data[persona]['correct']
    ds_total = deepseek_data[persona]['total']
    gpt_correct = gpt4o_data[persona]['correct']
    gpt_total = gpt4o_data[persona]['total']
    
    # 列联表
    table = np.array([
        [ds_correct, ds_total - ds_correct],
        [gpt_correct, gpt_total - gpt_correct]
    ])
    
    chi2, p, dof, expected = chi2_contingency(table)
    
    print(f"\n{name}:")
    print(f"   卡方值 = {chi2:.3f}")
    print(f"   p值 = {p:.4f}")
    if p < 0.05:
        print(f"   → 两个模型在该人格下差异显著 (p < 0.05)")
    else:
        print(f"   → 两个模型在该人格下差异不显著")

# ============================================================
# 5. 保存对比结果
# ============================================================

comparison_df = pd.DataFrame([
    {'人格': name, 'DeepSeek-R1': f"{deepseek_data[p]['acc']:.1%}", 
     'GPT-4o': f"{gpt4o_data[p]['acc']:.1%}", 
     '差异': f"{deepseek_data[p]['acc'] - gpt4o_data[p]['acc']:+.1%}"}
    for p, name in persona_names.items()
])

comparison_df.to_csv("cross_model_comparison.csv", index=False)
print("\n" + "="*70)
print("对比结果已保存到: cross_model_comparison.csv")

## 可视化
import matplotlib.pyplot as plt
import numpy as np

# 数据
personas = ['Control', 'Scientist', 'Influencer', 'Fake Source']
deepseek_acc = [65.6, 73.8, 44.8, 39.9]
gpt4o_acc = [63.5, 67.2, 40.2, 19.9]

x = np.arange(len(personas))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, deepseek_acc, width, label='DeepSeek-R1', color='#2E86AB')
bars2 = ax.bar(x + width/2, gpt4o_acc, width, label='GPT-4o', color='#E74C3C')

ax.set_ylabel('MC1 accuracy (%)')
ax.set_xlabel('Persona Type')
ax.set_title('Cross-Model Comparison: DeepSeek-R1 vs GPT-4o')
ax.set_xticks(x)
ax.set_xticklabels(personas)
ax.legend()
ax.axhline(y=65.6, color='#2E86AB', linestyle='--', alpha=0.5)
ax.axhline(y=63.5, color='#E74C3C', linestyle='--', alpha=0.5)

# 添加数值标签
for bar, acc in zip(bars1, deepseek_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc}%', ha='center', fontsize=10)
for bar, acc in zip(bars2, gpt4o_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('cross_model_chart.png', dpi=150)
plt.show()