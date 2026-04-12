import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer
from plot.utils.plot_style import setup_plot_equal_style, style_axis, save_figure




def compute_grouped_mean_attention(
    all_attn_weights: torch.Tensor,
    group_size: int = 30,
) -> torch.Tensor:
    """
    all_attn_weights: (N, S)
    return: grouped_mean: (G, S)
    """
    if all_attn_weights.dim() != 2:
        raise ValueError(f"Expected all_attn_weights shape (N, S), got {tuple(all_attn_weights.shape)}")

    n_samples, _ = all_attn_weights.shape
    n_groups = math.ceil(n_samples / group_size)

    grouped = []
    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, n_samples)
        chunk = all_attn_weights[start:end]
        chunk_mean = chunk.mean(dim=0)
        grouped.append(chunk_mean)

    return torch.stack(grouped, dim=0)


def build_model(device: torch.device, input_key: str = "x_mag") -> MagneticLocalizationTimeMixer:
    input_dim_reflect = {
        "x_mag": 3,
        "x_mag_grad": 9,
    }

    model = MagneticLocalizationTimeMixer(
        input_dim=input_dim_reflect[input_key],
        d_model=128,
        seq_len=128,
        down_sampling_window=2,
        down_sampling_layers=2,
        num_pdm_blocks=2,
        moving_avg_kernel=11,
        nhead=8,
        num_layers=2,
        output_dim=2,
    )
    return model.to(device)


def collect_grouped_attention(
    test_dir: Path,
    ckpt_path: Path,
    device: torch.device,
    group_size: int = 30,
    batch_size: int = 16,
    input_key: str = "x_mag",
) -> torch.Tensor:
    """
    返回 grouped_attn: (G, S)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(device=device, input_key=input_key)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    feature_transform = build_transform(input_key=input_key)

    test_loader = create_magnetic_dataset_v2_dataloaders(
        str(test_dir),
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=False,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=128,
        stride=20,
    )

    all_attn_weights = []

    with torch.no_grad():
        for batch in test_loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            _, attn_weights = model(x)   # (B, S)
            all_attn_weights.append(attn_weights.detach().cpu())

    if len(all_attn_weights) == 0:
        raise RuntimeError(f"No attention weights collected from: {test_dir}")

    all_attn_weights = torch.cat(all_attn_weights, dim=0)   # (N, S)
    grouped_attn = compute_grouped_mean_attention(all_attn_weights, group_size=group_size)

    print(f"{test_dir.name}: all_attn={tuple(all_attn_weights.shape)}, grouped_attn={tuple(grouped_attn.shape)}")
    print(f"{test_dir.name}: overall mean attention = {all_attn_weights.mean(dim=0).tolist()}")

    return grouped_attn


def plot_three_grouped_attention_heatmaps(
    grouped_list: List[torch.Tensor],
    titles: List[str],
    save_path: Path,
):
    """
    grouped_list: [Tensor(G1,S), Tensor(G2,S), Tensor(G3,S)]
    titles:       ["文管学馆路径", "信息学馆路径1", "信息学馆路径2"]
    """
    if len(grouped_list) != 3 or len(titles) != 3:
        raise ValueError("grouped_list 和 titles 必须都恰好包含 3 个元素。")

    setup_plot_equal_style()

    arrays = [g.detach().cpu().numpy() for g in grouped_list]

    # 统一颜色范围，保证三张热力图可比
    global_min = min(arr.min() for arr in arrays)
    global_max = max(arr.max() for arr in arrays)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12.5, 4.2),
        squeeze=False,
    )
    axes = axes[0]   # 变成长度为 3 的一维数组

    im = None
    for i, (ax, data, title) in enumerate(zip(axes, arrays, titles)):
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
        )

        style_axis(
            ax,
            title=title,
            xlabel=None, # type: ignore
            ylabel=None # type: ignore
        )

        
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels([f"尺度 {k+1}" for k in range(data.shape[1])])
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels([f"G{k+1}" for k in range(data.shape[0])])
    

    # 先给右侧留出 colorbar 空间
    fig.subplots_adjust(
        left=0.06,
        right=0.90,
        bottom=0.14,
        top=0.88,
        wspace=0.10,
    )

    # 单独创建 colorbar 轴，避免挤进第三张图里
    cax = fig.add_axes([0.915, 0.14, 0.015, 0.74])  # type: ignore # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax) # type: ignore
    cbar.set_label("注意力权重", rotation=270, labelpad=18)

    save_figure(fig, save_path, show=False, tight=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_size = 30

    configs: List[Dict[str, object]] = [
        {
            "title": "文管学馆路径",
            "test_dir": Path("data") / "data_for_train_test_v14" / "12.25-wenguan-resample-zscore" / "test1",
            "ckpt_path": Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260112_1405_rmse_2d_1.129_wenguan.pt",
        },
        {
            "title": "信息学馆路径1",
            "test_dir": Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "test1",
            "ckpt_path": Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260208_2141_rmse_2d_1.109_tcn_xinxi.pt",
        },
        {
            "title": "信息学馆路径2",
            "test_dir": Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "test5",
            "ckpt_path": Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260208_2141_rmse_2d_1.109_tcn_xinxi.pt",
        },
    ]

    grouped_list = []
    titles = []

    for cfg in configs:
        grouped_attn = collect_grouped_attention(
            test_dir=cfg["test_dir"],      # type: ignore
            ckpt_path=cfg["ckpt_path"],    # type: ignore
            device=device,
            group_size=group_size,
            batch_size=16,
            input_key="x_mag",
        )
        grouped_list.append(grouped_attn)
        titles.append(cfg["title"])        # type: ignore

    save_path = Path("figures") / "attention" / "grouped_mean_attention_combined.svg"
    plot_three_grouped_attention_heatmaps(
        grouped_list=grouped_list,
        titles=[
            "文管学馆路径",
            "信息学馆路径1",
            "信息学馆路径2",
        ],
        save_path=save_path,
)


if __name__ == "__main__":
    main()