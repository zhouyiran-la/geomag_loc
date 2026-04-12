from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import cast

from plot.utils.plot_style import setup_plot_resample_style, style_axis, save_figure

setup_plot_resample_style()
# ROOT = Path(__file__).resolve().parents[1]

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
LABELS = ["匀速", "快速", "慢速"]
PALETTE = ["#A9CA70", "#C5D6F0", "#e6b745","#F18C54" "#d0dd97", "#dddddd", "#e6b745"]
OUTPUT_DIR = Path("figures")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric while keeping the Series type for static checkers."""
    return cast(pd.Series, pd.to_numeric(series, errors="coerce"))


def plot_raw_geomagneticx(files: list[Path], out_path: Path) -> None:
    """绘制原始地磁模值序列（geomagneticx 列，横轴为样本序号，单位：μT）。"""
    plt.figure(figsize=(8, 7))
    for index, csv_path in enumerate(files):
        df = pd.read_csv(csv_path)
        magx = _to_numeric_series(df["geomagneticx"]).to_numpy()
        x = np.arange(len(magx))
        plt.plot(x, magx, label=LABELS[index], color=PALETTE[index])

    plt.tick_params(
        axis='both',        # x 和 y 轴
        which='major',      # 主刻度
        direction='in',     # 刻度线向内
        length=5,           # 刻度线长度
        width=1
    )
    # plt.xlim(-200, 6000)
    plt.ylim(0, 82)
    plt.xlabel("样本序号")
    plt.ylabel("地磁模值 (μT)")
    
    plt.grid(False)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_resampled_geomagneticx(files: list[Path], out_path: Path) -> None:
    """绘制空间重采样后的地磁模值序列（横轴为距离，单位：米；纵轴单位：μT）。"""
    plt.figure(figsize=(8, 7))
    for index, csv_path in enumerate(files):
        df = pd.read_csv(csv_path)
        xs = _to_numeric_series(df["pos_x"]).to_numpy()
        ys = _to_numeric_series(df["pos_y"]).to_numpy()
        magx = _to_numeric_series(df["geomagneticx"]).to_numpy()

        dx = np.diff(xs)
        dy = np.diff(ys)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.insert(np.cumsum(ds), 0, 0.0)

        plt.plot(s, magx, label=LABELS[index], color=PALETTE[index])

    plt.tick_params(
        axis='both',        # x 和 y 轴
        which='major',      # 主刻度
        direction='in',     # 刻度线向内
        length=5,           # 刻度线长度
        width=1
    )
    plt.xlim(-12, 180)
    plt.ylim(0, 82)
    plt.xlabel("距离 (m)")
    plt.ylabel("地磁模值 (μT)")
    plt.grid(False)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    _ensure_output_dir()

    raw_files = [p for p in RAW_PATH_LIST if p.exists()]
    resample_files = [p for p in RESAMPLE_PATH_LIST if p.exists()]

    raw_out = OUTPUT_DIR / "raw_geomagneticx_3.svg"
    resample_out = OUTPUT_DIR / "resample_geomagneticx_3.svg"

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
