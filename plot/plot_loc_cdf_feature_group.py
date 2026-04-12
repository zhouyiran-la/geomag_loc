import csv
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib import font_manager
import matplotlib
from plot.utils.plot_style import setup_plot_cdf_style

setup_plot_cdf_style()


ROOT = Path(__file__).resolve().parents[1]
CSV_PATHS = [
    ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_wenguan" / "1405_wenguan_test3_season+trend_loc_res_meanerr_0.8808.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_wenguan" / "1759_wenguan_test1_no_mix_no_decompose_loc_res_meanerr_1.1121.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer" / "0028_wenguan_test1_trend_loc_res_meanerr_1.4801.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_wenguan" / "0028_wenguan_test2_trend_loc_res_meanerr_1.2151.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_wenguan" / "0051_wenguan_test2_no_decompose_loc_res_meanerr_1.6449.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_wenguan" / "1317_wenguan_test1_no_decompose_loc_res_meanerr_1.7089.csv",
    
    # ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_xinxi" / "2141_xinxi_test2_season+trend_loc_res_meanerr_0.6825.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_xinxi" / "0012_xinxi_test3_no_mix_loc_res_meanerr_0.8363.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_xinxi" / "2302_xinxi_test1_trend_only_loc_res_meanerr_0.8995.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_xinxi" / "2322_xinxi_test3_trend_only_loc_res_meanerr_1.2581.csv",
    # # ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_xinxi" / "2322_xinxi_test3_trend_only_loc_res_meanerr_1.2581.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_feature_group_xinxi" / "2333_xinxi_test1_no_decompose_loc_res_meanerr_1.5111.csv",
    
]

# LABELS = ["Proposed", "Wang(2024)", "HLSTM(2022)", "MAIL(2020)", "RNN"]
LABELS = [
    "Season-Trend", "Season-Trend-No-Mixed", "Trend-Only", "Season-Only" , "No-Decompose"
]
OUTPUT_PATH = ROOT / "figures" / "loc_cdf_differernt_feature_wenguan.svg"
PLOT_TITLE = "Localization Error CDF"
X_MAX = None  # Set to a float to force xmax, or None to auto-scale.


def read_errors(csv_path: Path) -> np.ndarray:
    errors: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        # 先逐行读，直到找到真正数据表头（包含 euclidean_error）
        header_line = None
        while True:
            line = f.readline()
            if not line:  # EOF
                break
            s = line.strip()
            if not s:
                continue
            # 找到第二段的表头行
            if "euclidean_error" in s.split(","):
                header_line = s
                break

        if header_line is None:
            raise ValueError(f"Data header with 'euclidean_error' not found in {csv_path}")

        headers = [h.strip() for h in header_line.split(",")]
        if "euclidean_error" not in headers:
            raise ValueError(f"'euclidean_error' column not found in {csv_path}")

        # 从当前文件指针位置继续读数据行
        reader = csv.DictReader(f, fieldnames=headers)
        for row in reader:
            try:
                v = row.get("euclidean_error", "")
                if v is None or v == "":
                    continue
                errors.append(float(v))
            except (TypeError, ValueError):
                continue

    if not errors:
        raise ValueError(f"No valid errors loaded from {csv_path}")
    return np.asarray(errors, dtype=np.float32)


def plot_cdf(errors: np.ndarray):
    sorted_errors = np.sort(errors)
    xs = np.concatenate(([0.0], sorted_errors))
    n = len(sorted_errors)
    probs = np.arange(n + 1, dtype=np.float32) / max(n, 1)
    return xs, probs


def main():
    if LABELS and len(LABELS) != len(CSV_PATHS):
        raise ValueError("Number of labels must match number of CSV files.")
    labels = LABELS or [p.stem for p in CSV_PATHS]

    pairs: list[tuple[Path, str]] = []
    missing = []
    for path, label in zip(CSV_PATHS, labels):
        if path.exists():
            pairs.append((path, label))
        else:
            missing.append(str(path))

    if not pairs:
        raise FileNotFoundError("No CSV files found. Please check CSV_PATHS.")
    if missing:
        print(f"Skipping missing files: {', '.join(missing)}")

    plt.figure(figsize=(10, 8))
    curves = []
    max_x = 0.0
    for path, label in pairs:
        errors = read_errors(path)
        xs, ys = plot_cdf(errors)
        curves.append((xs, ys, label))
        if xs.size:
            max_x = max(max_x, float(xs.max()))

    for xs, ys, label in curves:
        plt.plot(xs, ys, label=label)

    plt.xlabel("定位误差（m）")
    plt.ylabel("概率")
    # plt.title(PLOT_TITLE)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="lower right")
    plt.xticks(np.linspace(0, 25, num=6))
    plt.yticks(np.linspace(0.0, 1.0, num=6))
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.6)
    plt.xlim(-0.5, 20)
    plt.ylim(0, 1.0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"Saved CDF plot to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
