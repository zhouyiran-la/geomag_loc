import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import matplotlib.patheffects as pe

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
plt_rc = matplotlib.rcParams

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "plot" / "output" / "seq_num_new.png"

# 字体
plt_rc["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
plt_rc["axes.unicode_minus"] = False

plt_rc.update({
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.4,
    "axes.edgecolor": "0.25",
    "axes.linewidth": 1.5,
})

# ===== 数据 =====
bar_width = 0.06
groups = ['信息学馆-路径1', '信息学馆-路径2', '文管学馆-路径1']
values = [
    [4.02, 3.62, 7.50],
    [1.11, 1.37, 2.93],
    [0.68, 0.78, 0.82],
    [0.70, 0.74, 0.77],
]
labels = ['序列长度=32', '序列长度=64', '序列长度=128', "序列长度=256"]

# 每组中心（按组数量自动生成，组间距略缩小）
x_group = np.arange(len(groups)) * 0.3
category_num = len(values)
offsets = np.linspace(
    - (category_num - 1) * bar_width / 2,
    + (category_num - 1) * bar_width / 2,
    category_num
)

# colors = [
#   "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
# ]

colors = [
  "#EAEAEA",  # light gray
  "#C8DFFF",  # soft sky blue
  "#CCCCCC",  # medium gray
  "#7EA4D3",  # steel blue
]

# colors = [
# #   "#EAEAEA",  # light gray
#     "#FFC300",
#   "#FDE4AD",  # soft sky blue
#   "#D1EDF3",  # medium gray
#   "#FCD7D4",  # steel blue
# ]

plt.figure(figsize=(8, 7))

for i in range(category_num):
    bar_positions = x_group + offsets[i]
    
    # 绘制柱状图
    plt.bar(
        bar_positions,
        values[i],
        width=bar_width,
        color=colors[i],
        edgecolor="black",
        linewidth=1.5,
        label=labels[i],
    )

    # ===== 添加数值标记 =====
    for j, x_pos in enumerate(bar_positions):
        plt.text(
            x_pos, 
            values[i][j] + 0.005,       # 文字放在柱子稍上方
            f"{values[i][j]:.2f}",
            ha='center',
            va='bottom',
            fontsize=11
        )

# 坐标轴
plt.xticks(x_group, groups)
plt.ylabel("平均定位误差(m)")
plt.ylim(0, 8)

plt.legend(title="", loc='best')
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=600)
plt.close()

print(f"Saved CDF plot to {OUTPUT_PATH.resolve()}")
