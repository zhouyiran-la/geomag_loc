#!/usr/bin/env python3
"""Plot geomagnetic magnitude curves from multiple CSV files (research style, full CJK support)."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# ✅ 科研出版风格 + 中文支持设置
# ---------------------------------------------------------
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
plt_rc = matplotlib.rcParams

# ✅ 字体策略：
# - 英文：Times New Roman（论文标准）
plt_rc["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
plt_rc["axes.unicode_minus"] = False  # 解决负号显示问题

# ✅ 基础样式：科研风格
plt_rc.update({
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 1.6,
    "grid.alpha": 0.4,
    "axes.edgecolor": "0.25",
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------
# 📁 路径设置
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 📊 数据集配置
# ---------------------------------------------------------
DATASETS: Iterable[Tuple[str, Path, float]] = (
    (
        "Huawei P60",
        PROJECT_ROOT / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_wqh极慢_T.csv",
        1.5
    ),
    (
        "MEIZU 20",
        PROJECT_ROOT  / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_wqh慢速1_T.csv",
        25.0
    ),
    (
        "Redmi K70 Pro",
        PROJECT_ROOT  / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_wqh慢速2_T.csv",
        -3.5
    ),

    (
        "OPPO Find X",
        PROJECT_ROOT  / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_ghw慢速2_T.csv",
        10.0
    ),
)

USE_COLUMNS = ["timestamp", "geomagneticx", "geomagneticy", "geomagneticz"]
MAX_SAMPLES = 5000


def load_magnitude(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"无法找到数据文件: {path}")
    df = pd.read_csv(path, usecols=USE_COLUMNS, encoding="utf-8-sig")
    df.rename(columns=lambda c: c.strip(), inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[:MAX_SAMPLES].copy()

    mags = df["geomagneticx"]
    return pd.DataFrame({"magnitude": mags}, index=df.index)


def plot_magnitudes(datasets: Iterable[Tuple[str, pd.DataFrame, float]]) -> Path:
    dataset_list = list(datasets)
    fig, ax = plt.subplots(figsize=(12, 6))

    palette = ["#1C3885", "#4F8CBB", "#F4A25C", "#DD542F"]

    for idx, (label, df, offset) in enumerate(dataset_list):
        color = palette[idx % len(palette)]
        curve = df["magnitude"] + offset
        ax.plot(df.index, curve, label=label, color=color)

    # 🧩 坐标轴与标题（中文正常显示）
    ax.set_xlabel("样本序号", labelpad=6)
    ax.set_ylabel("地磁模值 (µT)", labelpad=6)
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_png = OUTPUT_DIR / "geomagnetic_magnitude_comparison_utf8.png"
    fig.savefig(output_png, dpi=600)
    plt.close(fig)
    return output_png


def main() -> None:
    loaded = []
    for label, path, offset in DATASETS:
        try:
            df = load_magnitude(path)
        except FileNotFoundError as exc:
            print(f"跳过 {label}: {exc}")
            continue
        loaded.append((label, df, offset))

    if not loaded:
        raise SystemExit("未找到任何可绘制的数据，请检查 DATASETS 配置。")

    output = plot_magnitudes(loaded)
    print(f"✅ 绘图完成：{output}")


if __name__ == "__main__":
    main()
