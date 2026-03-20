import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import matplotlib.patheffects as pe
from plot.utils.plot_style import setup_plot_equal_style, style_axis, save_figure

setup_plot_equal_style()

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT  / "figures" / "different_scale_num.png"


x = np.array([1, 2, 3, 4])

# 每一行表示一个场景在不同尺度数量下的平均定位误差
values = [
    [1.33, 0.95, 0.79, 0.76],  # 信息学馆-路径1
    [1.23, 1.16, 0.80, 0.87],  # 信息学馆-路径2
    [1.78, 1.13, 0.95, 1.03],  # 文管学馆-路径1
]

# 数值标注偏移 (dx, dy)
offsets = [
    [(0.00, 0.02), (0.00, -0.03), (0.00, -0.02), (0.00, 0.02)],  # 信息学馆-路径1
    [(0.00, 0.02), (0.00, 0.02), (0.00, 0.02), (0.00, 0.02)],    # 信息学馆-路径2
    [(0.00, 0.02), (0.00, -0.03), (0.00, 0.02), (0.00, 0.02)],   # 文管学馆-路径1
    
]

labels = [
    "信息学馆-路径1",
    "信息学馆-路径2",
    "文管学馆-路径1",
]

# 可选配色
colors = [
   "#38C1F3", "#8048AA", "#FF0000"
    # "#9ACE87", "#D2D2D2", "#EE634B"
   	# "#6D65A3", "#6F6F6F", "#F6631C"
]

# marker样式
markers = ["o", "s", "^"]

plt.figure(figsize=(8, 7))

for i, y in enumerate(values):
    plt.plot(
        x,
        y,
        label=labels[i],
        color=colors[i],
        marker=markers[i],
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.2,
    )

    # ===== 数值标注 =====
    for j, (xi, yi) in enumerate(zip(x, y)):

        dx, dy = offsets[i][j]
        va = "bottom" if dy >= 0 else "top"

        plt.text(
            xi + dx,
            yi + dy,
            f"{yi:.2f}",
            ha="center",
            va=va,
            fontsize=11,
            color="0.15",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5
        )

# 坐标轴设置
plt.tick_params(axis="both", which="major", direction="in", length=6, width=1.0)
plt.xticks(x, [f"{i}" for i in x])
plt.xlabel("尺度数量")
plt.ylabel("平均定位误差(m)")

# 自动设置y轴范围，也可以手动改

plt.ylim(0.5, 2)

plt.legend(loc="best", frameon=False)
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
# plt.grid(False)
save_figure(plt.gcf(), OUTPUT_PATH, show=False, tight=False)
# plt.tight_layout()
# plt.savefig(OUTPUT_PATH, dpi=600)
# plt.close()

# print(f"Saved line plot to {OUTPUT_PATH.resolve()}")