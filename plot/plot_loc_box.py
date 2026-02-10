import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# ================== matplotlib 样式 ==================
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
plt_rc = matplotlib.rcParams
plt_rc["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt_rc["axes.unicode_minus"] = False
plt_rc.update({
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.4,
    "axes.edgecolor": "0.25",
    "axes.linewidth": 1.5,
})

# ================== 路径配置 ==================
ROOT = Path(__file__).resolve().parents[1]

CSV_PATHS = [
    # ROOT / "runs" / "loc_res" / "time_mixer_different_encoder_wenguan" / "wenguan_test1_bilstm_loc_res_meanerr_1.3088.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_encoder_wenguan" / "wenguan_test1_lstm_loc_res_meanerr_1.3494.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_encoder_wenguan" / "wenguan_test1_rnn_loc_res_meanerr_2.2984.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_encoder_wenguan" / "wenguan_test1_trans_loc_res_meanerr_1.6811.csv",
    # ROOT / "runs" / "loc_res" / "time_mixer_different_encoder_wenguan" / "wenguan_test1_tcn_loc_res_meanerr_0.9090.csv",

    ROOT / "runs" / "loc_res" / "time_mixer_time_mixer_different_encoder_xinxi_new_data" / "xinxi_test3_bilstm_loc_res_meanerr_1.4181.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_time_mixer_different_encoder_xinxi_new_data" / "xinxi_test1_lstm_loc_res_meanerr_1.4615.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_time_mixer_different_encoder_xinxi_new_data" / "xinxi_test1_rnn_loc_res_meanerr_2.0144.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_time_mixer_different_encoder_xinxi_new_data" / "xinxi_test1_trans_loc_res_meanerr_1.5700.csv",
    ROOT / "runs" / "loc_res" / "time_mixer_time_mixer_different_encoder_xinxi_new_data" / "xinxi_test1_tcn_loc_res_meanerr_0.6621.csv",

]

LABELS = [
    "BiLSTM-encoder",
    "LSTM-encoder",
    "RNN-encoder",
    "Transformer-encoder",
    "TCN-encoder",
]

OUTPUT_PATH = ROOT / "plot" / "output" / "loc_error_boxplot_multi_mean_xinxi.png"
Y_MAX = None  # 可手动设上限，如 6.0

# ================== 读取误差 ==================
def read_errors(csv_path: Path) -> np.ndarray:
    errors: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "euclidean_error" not in (reader.fieldnames or []):
            raise ValueError(f"'euclidean_error' not found in {csv_path}")
        for row in reader:
            try:
                errors.append(float(row["euclidean_error"]))
            except (TypeError, ValueError):
                continue
    if not errors:
        raise ValueError(f"No valid errors in {csv_path}")
    return np.asarray(errors, dtype=np.float32)

# ================== 主函数 ==================
def main():
    if len(LABELS) != len(CSV_PATHS):
        raise ValueError("LABELS length must match CSV_PATHS length")

    all_errors: List[np.ndarray] = []
    means: List[float] = []
    used_labels: List[str] = []
    y_max_seen = 0.0

    for p, lab in zip(CSV_PATHS, LABELS):
        if not p.exists():
            print(f"Skip missing file: {p}")
            continue
        errs = read_errors(p)
        all_errors.append(errs)
        means.append(float(np.mean(errs)))
        used_labels.append(lab)
        y_max_seen = max(y_max_seen, float(np.max(errs)))

    if not all_errors:
        raise RuntimeError("No valid CSV files found.")

    positions = np.arange(1, len(all_errors) + 1)

    plt.figure(figsize=(10, 8))

    # -------- 箱线图 --------
    plt.boxplot(
        all_errors,
        positions=positions,
        widths=0.55,
        whis=1.5,
        showfliers=True,
        patch_artist=False,
    )

    # -------- 均值点 --------
    plt.scatter(
        positions,
        means,
        marker="o",
        s=40,
        zorder=3,
        label="Mean",
    )

    plt.ylabel("定位误差（m）")
    plt.xticks(positions, used_labels, rotation=15)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", direction="in", length=6, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.6)

    y_min = 0.0
    y_max = Y_MAX if Y_MAX is not None else y_max_seen * 1.10
    plt.ylim(y_min, 22.9)

    plt.legend(loc="upper right")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=600)
    plt.close()

    print(f"Saved boxplot with mean to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
