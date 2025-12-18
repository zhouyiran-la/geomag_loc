#!/usr/bin/env python3
"""Plot geomagnetic components after geo_trans_fast transformation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

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

USE_COLUMNS = [
    "timestamp",
    "geomagneticx",
    "geomagneticy",
    "geomagneticz",
    "gravityx",
    "gravityy",
    "gravityz",
]
MAX_SAMPLES = 2000


def geo_trans_fast(mag_data: np.ndarray, gra_data: np.ndarray) -> np.ndarray:
    """Vectorized geomagnetic transform as provided."""
    ms = np.linalg.norm(mag_data, axis=1)
    gra_norm = np.linalg.norm(gra_data, axis=1)
    gra_norm = np.where(gra_norm == 0, 1e-8, gra_norm)
    dot = np.einsum("ij,ij->i", mag_data, gra_data)
    mv = np.abs(dot / gra_norm)
    mh_sq = np.clip(ms**2 - mv**2, a_min=0, a_max=None)
    mh = np.sqrt(mh_sq)
    return np.column_stack((ms, mh, mv))


def load_transformed(path: Path) -> pd.DataFrame:
    """Load CSV data, apply geo_trans_fast, and return transformed components."""
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

    mag_data = df[["geomagneticx", "geomagneticy", "geomagneticz"]].to_numpy(dtype=float)
    gra_data = df[["gravityx", "gravityy", "gravityz"]].to_numpy(dtype=float)
    transformed = geo_trans_fast(mag_data, gra_data)
    df_trans = pd.DataFrame(
        transformed,
        columns=["ms", "mh", "mv"],
        index=df.index,
    )
    return df_trans


def plot_transformed(datasets: Iterable[Tuple[str, pd.DataFrame]]) -> Path:
    """Plot transformed components for both datasets."""
    axis_labels = {
        "ms": "Ms (µT)",
        "mh": "Mh (µT)",
        "mv": "Mv (µT)",
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
                label=label if column == "ms" else None,
            )
        ax.set_ylabel(axis_labels[column])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("样本序号")
    # fig.suptitle("姿态固定 vs. 姿态变化 - 转换后地磁分量对比")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))

    output_path = OUTPUT_DIR / "geomagnetic_transformed_pose_comparison.png"
    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    return output_path


def main() -> None:
    loaded = []
    for label, path in DATASETS:
        df = load_transformed(path)
        loaded.append((label, df))

    output = plot_transformed(loaded)
    print(f"绘图完成：{output}")


if __name__ == "__main__":
    main()
