import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ========== 你已有的全局设置 ==========
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

ROOT = Path(__file__).resolve().parents[1]
CSV_PATHS = [
    ROOT / "runs" / "loc_res" / "time_mixer" / "data_with_label_ghw匀速1_loc_res_rnn_meanerr_1.4801.csv",
    ROOT / "runs" / "loc_res" / "time_mixer" / "data_with_label_ghw匀速1_loc_res_lstm_meanerr_1.2520.csv",
    ROOT / "runs" / "loc_res" / "time_mixer" / "data_with_label_ghw匀速1_loc_res_trans_meanerr_1.0572.csv",
    ROOT / "runs" / "loc_res" / "time_mixer" / "data_with_label_ghw匀速1_loc_res_bilstm_meanerr_0.8231.csv",
    ROOT / "runs" / "loc_res" / "time_mixer" / "data_with_label_ghw匀速1_loc_res_tcn_meanerr_0.5809.csv",
    # ROOT / "runs" / "loc_res" / "different_method" / "result_plus_1.15.csv",
    # ROOT / "runs" / "loc_res" / "different_method" / "result_plus_0.53.csv",
    # ROOT / "runs" / "loc_res" / "different_method" / "result_plus_0.88.csv",
    # ROOT / "runs" / "loc_res" / "different_method" / "result_plus_2.20.csv",
]

LABELS = ["TimeMixer+RNN", "TimeMixer+LSTM", "TimeMixer+Transformer", "TimeMixer+BiLSTM", "TimeMixer+TCN"]

# 建议把文件名改成轨迹图名字，例如 "loc_traj.png"
OUTPUT_PATH = ROOT / "plot" / "output" / "loc_traj.png"
PLOT_TITLE = "Localization Trajectories"


def load_multi_trajectories(csv_paths):
    """
    读取多条轨迹：
    返回：
        true_x, true_y: (N,) 真值轨迹
        traj_list: 每个元素是 dict，包含 pred_x, pred_y, error
    """
    true_x, true_y = None, None
    traj_list = []

    for path in csv_paths:
        df = pd.read_csv(path)

        # 初始化真值轨迹（用第一份）
        if true_x is None:
            true_x = df["true_x"].values
            true_y = df["true_y"].values
        else:
            # 如需严格检查真值是否相同，可以打开下面的断言
            # assert (true_x == df["true_x"].values).all()
            # assert (true_y == df["true_y"].values).all()
            pass

        traj_list.append(
            dict(
                pred_x=df["pred_x"].values,
                pred_y=df["pred_y"].values,
                error=df["euclidean_error"].values
                if "euclidean_error" in df.columns
                else None,
            )
        )

    return true_x, true_y, traj_list


# def plot_multi_trajectories(csv_paths, labels, output_path, title="Localization Trajectories"):
#     assert len(csv_paths) == len(labels), "CSV_PATHS 和 LABELS 数量必须一致"

#     true_x, true_y, traj_list = load_multi_trajectories(csv_paths)

#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(6, 6))
#     # assert true_x
#     # assert true_y
#     # 画真值轨迹（黑色实线，圆点）
#     ax.plot(true_x, true_y, linestyle="-", color="k", label="Ground Truth") # type: ignore

#     # 画多条预测轨迹
#     for traj, label in zip(traj_list, labels):
#         ax.plot(
#             traj["pred_x"],
#             traj["pred_y"],
#             linestyle="--",
#             # marker="x",
#             label=label,
#         )

#         # 如果想画真值到预测点的误差连线，可取消注释：
#         # for px, py, tx, ty in zip(traj["pred_x"], traj["pred_y"], true_x, true_y):
#         #     ax.plot([tx, px], [ty, py], linestyle=":", linewidth=0.8, alpha=0.4)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True)
#     ax.axis("equal")  # 保持坐标比例一致

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=300)
#     plt.close(fig)

def plot_multi_trajectories(
    csv_paths,
    labels,
    output_path,
    title="Localization Trajectories",
    show_start_end=True,
):
    assert len(csv_paths) == len(labels), "CSV_PATHS 和 LABELS 数量必须一致"

    true_x, true_y, traj_list = load_multi_trajectories(csv_paths)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # ===== 颜色方案：统一用 tab10 调色板 =====
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(traj_list))]

    # 起点 / 终点
    if show_start_end:
        ax.scatter(
            true_x[0], # type: ignore
            true_y[0], # type: ignore
            s=25,
            marker="o",
            edgecolor="black",
            facecolor="white",
            zorder=5,
            label="Start",
        )
        ax.scatter(
            true_x[-1], # type: ignore
            true_y[-1], # type: ignore
            s=30,
            marker="X",
            edgecolor="black",
            facecolor="white",
            zorder=6,
            label="End",
        )

    # ===== 真值轨迹：粗黑线 =====
    ax.plot(
        true_x,  # type: ignore
        true_y,  # type: ignore
        linestyle="-",
        color="black",
        linewidth=1,
        label="Ground Truth",
        zorder=2,
    ) 

    # ===== 多条预测轨迹 =====
    for idx, (traj, label) in enumerate(zip(traj_list, labels)):
        ax.plot(
            traj["pred_x"],
            traj["pred_y"],
            linestyle="-",
            linewidth=1.2,
            color=colors[idx],
            alpha=0.9,
            label=label,
            zorder=3,
        )
        # 不画误差连线

    # ===== 轴标签 & 标题 =====
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis="both", which="major", length=2, direction="in", top=True, right=True)

    # ===== 图例美化 =====
    legend = ax.legend(
        loc="best",
        frameon=False,
    )

    # # 坐标比例 & 边距
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.05)

    # 网格稍微淡一点（你全局有 grid.alpha，也可以再调）
    ax.grid(False)

    # # 上右边框稍微淡一点
    # ax.spines["top"].set_alpha(0.6)
    # ax.spines["right"].set_alpha(0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_multi_trajectories(CSV_PATHS, LABELS, OUTPUT_PATH, PLOT_TITLE)
