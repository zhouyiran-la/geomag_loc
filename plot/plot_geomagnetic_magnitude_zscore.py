#!/usr/bin/env python3
"""Plot Z-score geomagnetic magnitude curves with the same styling and data as plot_geomagnetic_magnitude.py."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 📐 科研出版风格 + 中文支持设置（与原脚本保持一致）
# ---------------------------------------------------------
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
plt_rc = matplotlib.rcParams
plt_rc["font.family"] = ["Times New Roman", "SimHei", "Microsoft YaHei"]
plt_rc["axes.unicode_minus"] = False
plt_rc.update(
    {
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.6,
        "grid.alpha": 0.4,
        "axes.edgecolor": "0.25",
        "axes.linewidth": 0.8,
    }
)

# ---------------------------------------------------------
# 📁 路径设置
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 📊 数据集配置（保持与原脚本完全一致）
# ---------------------------------------------------------
DATASETS: Iterable[Tuple[str, Path,]] = (
    (
        "Huawei P60",
        PROJECT_ROOT / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_wqh极慢_T_Z.csv",
    ),
    (
        "MEIZU 20",
        PROJECT_ROOT  / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_wqh慢速1_T_Z.csv",
    ),
    (
        "Redmi K70 Pro",
        PROJECT_ROOT  / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_wqh慢速2_T_Z.csv",
    ),

    (
        "OPPO Find X",
        PROJECT_ROOT  / "data" / "origin" / "4.26数据" / "100" / "TZ" / "data_with_label_ghw慢速2_T_Z.csv",
    ),
)

USE_COLUMNS = ["timestamp", "geomagneticx", "geomagneticy", "geomagneticz"]
MAX_SAMPLES = 5000


def zscore_std(values: np.ndarray) -> np.ndarray:
    """对一维地磁模值做 Z-score 标准化。"""
    mean = np.mean(values)
    std = np.sqrt(np.var(values))
    if std == 0:
        std = 1e-8
    return (values - mean) / std


def load_magnitude(path: Path) -> pd.DataFrame:
    """读取 CSV，计算模值并进行 Z-score 处理。"""
    if not path.exists():
        raise FileNotFoundError(f"无法找到数据文件: {path}")

    df = pd.read_csv(path, usecols=USE_COLUMNS, encoding="utf-8-sig")
    df.rename(columns=lambda c: c.strip(), inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[:MAX_SAMPLES].copy()

    mags_z = df["geomagneticx"]

    return pd.DataFrame({"magnitude": mags_z}, index=df.index)


def plot_magnitudes(datasets: Iterable[Tuple[str, pd.DataFrame]]) -> Path:
    """按照原脚本风格绘制 Z-score 模值曲线。"""
    dataset_list = list(datasets)
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#1C3885", "#4F8CBB", "#F4A25C", "#DD542F"]

    for idx, (label, df) in enumerate(dataset_list):
        color = palette[idx % len(palette)]
        curve = df["magnitude"]
        ax.plot(df.index, curve, label=label, color=color)

    ax.set_xlabel("样本序号", labelpad=6)
    ax.set_ylabel("地磁模值 (µT)", labelpad=6)
    ax.tick_params(direction="in", length=4, width=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_png = OUTPUT_DIR / "geomagnetic_magnitude_zscore_comparison_utf8.png"
    fig.savefig(output_png, dpi=600)
    plt.close(fig)
    return output_png


def main() -> None:
    loaded = []
    for label, path in DATASETS:
        try:
            df = load_magnitude(path)
        except FileNotFoundError as exc:
            print(f"跳过 {label}: {exc}")
            continue
        loaded.append((label, df))

    if not loaded:
        raise SystemExit("未找到任何可绘制的数据，请检查 DATASETS 配置。")

    output = plot_magnitudes(loaded)
    print(f"绘图完成：{output}")


if __name__ == "__main__":
    main()
