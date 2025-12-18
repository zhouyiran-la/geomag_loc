from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import cast

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

# 路径列表（按需手动调整）
RAW_PATH_LIST = [
    Path("data/origin/4.26数据/50/TZ/data_with_label_ghw加速_T.csv"),
    Path("data/origin/4.26数据/50/TZ/data_with_label_wqh快速_T.csv"),
    Path("data/origin/4.26数据/50/TZ/data_with_label_ghw匀速_T.csv"),
]
RESAMPLE_PATH_LIST = [
    Path("data/origin/4.26数据/50/resample/data_with_label_ghw加速_T_resample.csv"),
    Path("data/origin/4.26数据/50/resample/data_with_label_wqh快速_T_resample.csv"),
    Path("data/origin/4.26数据/50/resample/data_with_label_ghw匀速_T_resample.csv"),
]
LABELS = ["ghw加速", "wqh快速", "ghw匀速"]
# PALETTE = ["#1C3885", "#F4A25C", "#DD542F"]
OUTPUT_DIR = Path("plot/resample")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric while keeping the Series type for static checkers."""
    return cast(pd.Series, pd.to_numeric(series, errors="coerce"))


def plot_raw_geomagneticx(files: list[Path], out_path: Path) -> None:
    """绘制原始地磁模值序列（geomagneticx 列，横轴为样本序号，单位：μT）。"""
    plt.figure(figsize=(12, 6))
    for index, csv_path in enumerate(files):
        df = pd.read_csv(csv_path)
        magx = _to_numeric_series(df["geomagneticx"]).to_numpy()
        x = np.arange(len(magx))
        plt.plot(x, magx, label=LABELS[index], linewidth=1.2)

    plt.xlabel("样本序号", labelpad=6)
    plt.ylabel("地磁模值 (μT)", labelpad=6)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_resampled_geomagneticx(files: list[Path], out_path: Path) -> None:
    """绘制空间重采样后的地磁模值序列（横轴为距离，单位：米；纵轴单位：μT）。"""
    plt.figure(figsize=(12, 6))
    for index, csv_path in enumerate(files):
        df = pd.read_csv(csv_path)
        xs = _to_numeric_series(df["pos_x"]).to_numpy()
        ys = _to_numeric_series(df["pos_y"]).to_numpy()
        magx = _to_numeric_series(df["geomagneticx"]).to_numpy()

        dx = np.diff(xs)
        dy = np.diff(ys)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0.0)

        plt.plot(s, magx, label=LABELS[index], linewidth=1.2)

    plt.xlabel("距离 (m)", labelpad=6)
    plt.ylabel("地磁模值 (μT)", labelpad=6)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    _ensure_output_dir()

    raw_files = [p for p in RAW_PATH_LIST if p.exists()]
    resample_files = [p for p in RESAMPLE_PATH_LIST if p.exists()]

    raw_out = OUTPUT_DIR / "raw_geomagneticx_3.png"
    resample_out = OUTPUT_DIR / "resample_geomagneticx_3.png"

    if raw_files:
        plot_raw_geomagneticx(raw_files, raw_out)
        print(f"原始地磁模值图已保存: {raw_out}")
    else:
        print("未找到原始文件，请在 RAW_PATH_LIST 中确认路径是否存在")

    if resample_files:
        plot_resampled_geomagneticx(resample_files, resample_out)
        print(f"空间重采样后地磁模值图已保存: {resample_out}")
    else:
        print("未找到重采样文件，请在 RESAMPLE_PATH_LIST 中确认路径是否存在")


if __name__ == "__main__":
    main()
