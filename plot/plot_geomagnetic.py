#!/usr/bin/env python3
"""Plot geomagnetic XYZ components for the requested datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import re

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, relative path)
DATASETS: Iterable[Tuple[str, Path]] = (
     (
        "姿态固定",
        PROJECT_ROOT / "data" / "origin" / "4.26数据" / "50" / "com" / "data_with_label_wqh慢速.csv",
    ),
    (
        "姿态变化",
        PROJECT_ROOT
        / "data"
        / "origin"
        / "4.26数据"
        / "50"
        / "com"
        / "data_with_label_zm匀速姿态变化.csv",
    ),
)

USE_COLUMNS = ["timestamp", "geomagneticx", "geomagneticy", "geomagneticz"]
MAX_SAMPLES = 2000


def _slugify(label: str) -> str:
    """Return a filesystem-friendly version of the label."""
    safe = re.sub(r"[^\w.-]+", "_", label)
    safe = safe.strip("_") or "geomagnetic"
    return safe


def load_geomagnetic(path: Path) -> pd.DataFrame:
    """Load and clean the geomagnetic columns from the CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"无法找到数据文件: {path}")

    df = pd.read_csv(path, usecols=USE_COLUMNS, encoding="utf-8-sig")
    df.rename(columns=lambda c: c.strip(), inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES].copy()
    return df


def plot_comparison(datasets: Iterable[Tuple[str, pd.DataFrame]]) -> Path:
    """Plot XYZ geomagnetic components for both datasets in one figure."""
    axis_labels = {
        "geomagneticx": "X (µT)",
        "geomagneticy": "Y (µT)",
        "geomagneticz": "Z (µT)",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, column in zip(axes, axis_labels):
        for idx, (label, df) in enumerate(datasets):
            color = palette[idx % len(palette)]
            ax.plot(
                df.index,
                df[column],
                color=color,
                label=label if column == "geomagneticx" else None,
            )
        ax.set_ylabel(axis_labels[column])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("样本序号")
    # fig.suptitle("姿态固定 vs. 姿态变化 - 地磁XYZ分量对比")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))

    output_path = OUTPUT_DIR / "geomagnetic_pose_comparison.png"
    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    return output_path


def main() -> None:
    loaded = []
    for label, path in DATASETS:
        df = load_geomagnetic(path)
        loaded.append((label, df))

    output = plot_comparison(loaded)
    print(f"绘图完成：{output}")


if __name__ == "__main__":
    main()
