import csv
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# ================== matplotlib 样式（与你给的保持一致） ==================
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
plt_rc = matplotlib.rcParams
plt_rc["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
plt_rc["axes.unicode_minus"] = False
plt_rc.update(
    {
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.4,
        "axes.edgecolor": "0.25",
        "axes.linewidth": 1.5,
    }
)

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "runs" / "loc_res" / "your_result.csv"
OUTPUT_PATH = ROOT / "plot" / "output" / "localization_error_boxplot.png"

# ================== 读取误差 ==================
def read_errors(csv_path: Path) -> np.ndarray:
    errors: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "euclidean_error" not in (reader.fieldnames or []):
            raise ValueError("CSV 中未找到 euclidean_error 字段")
        for row in reader:
            try:
                errors.append(float(row["euclidean_error"]))
            except (TypeError, ValueError):
                continue

    if not errors:
        raise ValueError("未读取到有效的定位误差数据")
    return np.asarray(errors, dtype=np.float32)

# ================== 主函数 ==================
def main():
    errors = read_errors(CSV_PATH)

    plt.figure(figsize=(6.5, 6))

    plt.boxplot(
        errors,
        vert=True,
        whis=1.5,          # 1.5 IQR
        showfliers=True,  # 显示异常点
        widths=0.35,
        patch_artist=False
    )

    plt.ylabel("定位误差（m）")
    plt.xticks([1], ["定位结果"])

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.6)

    y_min = max(0.0, float(errors.min()) - 0.1)
    y_max = float(errors.max()) + 0.1
    plt.ylim(y_min, y_max)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()

    print(f"Saved boxplot to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
