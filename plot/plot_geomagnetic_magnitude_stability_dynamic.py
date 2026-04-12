#!/usr/bin/env python3
"""Plot Z-score geomagnetic magnitude curves with the same styling and data as plot_geomagnetic_magnitude.py."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot.utils.plot_style import setup_plot_resample_style, style_axis, save_figure

setup_plot_resample_style()

# ---------------------------------------------------------
# 📁 路径设置
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 📊 数据集配置（保持与原脚本完全一致）
# ---------------------------------------------------------
DATASETS: Iterable[Tuple[str, Path,]] = (
    (
        "2025-07-12 AM",
        PROJECT_ROOT / "data" / "other_experiment" / "dynamic" / "data_with_label_dataset_2025-12-25_20-49-28-656_T_resample.csv",
    ),
    (
        "2025-07-12 PM",
        PROJECT_ROOT / "data" / "other_experiment" / "dynamic" / "data_with_label_dataset_2025-12-25_20-52-40-158_T_resample.csv",
    ),
    (
        "2025-11-08",
        PROJECT_ROOT  / "data" / "other_experiment" / "dynamic" / "data_with_label_dataset_2025-12-25_20-55-56-595_T_resample.csv",
    ),
    (
        "2026-01-28",
        PROJECT_ROOT  / "data" / "other_experiment" / "dynamic" / "data_with_label_dataset_2025-12-25_20-59-06-612_T_resample.csv",
    ),
)

USE_COLUMNS = ["geomagneticx", "geomagneticy", "geomagneticz"]
# USE_COLUMNS = ['magX', 'magY', 'magZ']
MIN_SAMPLES = 500
MAX_SAMPLES = 1000


def load_magnitude(path: Path) -> pd.DataFrame:
    """读取 CSV，计算模值并进行 Z-score 处理。"""
    if not path.exists():
        raise FileNotFoundError(f"无法找到数据文件: {path}")
    # print(str(path))
    df = pd.read_csv(path, usecols=USE_COLUMNS, encoding="utf-8-sig")
    # print(df.iloc[:5])
    df = df.iloc[:MAX_SAMPLES].copy()
    mags_z = df[USE_COLUMNS[0]]
    # print(mags_z.size)
    return pd.DataFrame({"magnitude": mags_z}, index=df.index)

def compute_stats(label: str, df: pd.DataFrame) -> dict:
    """计算地磁模值的统计指标"""
    mag = df["magnitude"]

    return {
    "label": label,
    "mean": mag.mean(),
    "variance": mag.var(),
    "max": mag.max(),
    "min": mag.min(),
    "range": mag.max() - mag.min(),
    }

def plot_magnitudes(datasets: Iterable[Tuple[str, pd.DataFrame]]) -> Path:
    """按照原脚本风格绘制 Z-score 模值曲线。"""
    dataset_list = list(datasets)
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = [
        "#1C3885", "#4F8CBB", "#F4A25C", "#DD542F",
        "#2A9D8F", "#264653", "#6A4C93", "#F2C14E", 
        "#3FA7D6", "#8D99AE"  
    ]
    for idx, (label, df) in enumerate(dataset_list):
        color = palette[idx % len(palette)]
        curve = df["magnitude"]
        ax.plot(df.index, curve, label=label, color=color)

    ax.set_xlabel("样本序号", labelpad=6)
    ax.set_ylabel("地磁模值 (µT)", labelpad=6)
    ax.tick_params(direction="in", length=4, width=1.5)
    ax.grid(False)
    ax.margins(x=0.03)
    # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_png = OUTPUT_DIR / "geomagnetic_magnitude_static_dynamic.svg"
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return output_png




def main() -> None:
    loaded = []
    stats_list = []

    for label, path in DATASETS:
        try:
            df = load_magnitude(path)
        except FileNotFoundError as exc:
            print(f"跳过 {label}: {exc}")
            continue
        loaded.append((label, df))
        stats_list.append(compute_stats(label, df))

    if not loaded:
        raise SystemExit("未找到任何可绘制的数据，请检查 DATASETS 配置。")

    # === 绘图 ===
    output = plot_magnitudes(loaded)
    print(f"绘图完成：{output}")


    # === 输出统计结果（控制台）===
    print("\n地磁模值统计结果：")
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False, float_format="%.4f"))

    # === 保存为 CSV ===
    stats_csv = PROJECT_ROOT / "data" / "other_experiment" / "dynamic"  / "geomagnetic_magnitude_statistics.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    print(f"\n统计结果已保存：{stats_csv}")


if __name__ == "__main__":
    main()
