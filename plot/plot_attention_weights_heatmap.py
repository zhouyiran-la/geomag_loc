import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer


def compute_grouped_mean_attention(
    all_attn_weights: torch.Tensor,
    group_size: int = 50,
) -> torch.Tensor:
    """
    all_attn_weights: (N, S)
        N = 全部测试样本数
        S = 尺度数

    return:
        grouped_mean: (G, S)
        G = ceil(N / group_size)
    """
    if all_attn_weights.dim() != 2:
        raise ValueError(f"Expected all_attn_weights shape (N, S), got {tuple(all_attn_weights.shape)}")

    n_samples, n_scales = all_attn_weights.shape
    n_groups = math.ceil(n_samples / group_size)

    grouped = []
    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, n_samples)
        chunk = all_attn_weights[start:end]      # (group_len, S)
        chunk_mean = chunk.mean(dim=0)           # (S,)
        grouped.append(chunk_mean)

    grouped_mean = torch.stack(grouped, dim=0)   # (G, S)
    return grouped_mean


def plot_grouped_mean_attention_heatmap(
    grouped_attn: torch.Tensor,
    save_path: Path,
    title: str = "Grouped Mean Multi-scale Attention Heatmap",
):
    """
    grouped_attn: (G, S)
        G = 分组数
        S = 尺度数
    """
    data = grouped_attn.detach().cpu().numpy()

    plt.figure(figsize=(7, max(4, 0.45 * data.shape[0])))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Mean Attention Weight")

    plt.xlabel("Scale")
    plt.ylabel("Sample Group")
    plt.title(title)

    plt.xticks(
        np.arange(data.shape[1]),
        [f"Scale {i}" for i in range(data.shape[1])]
    )
    plt.yticks(
        np.arange(data.shape[0]),
        [f"G{i}" for i in range(data.shape[0])]
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"grouped mean attention heatmap saved to: {save_path}")
    plt.close()


def plot_overall_mean_attention_heatmap(
    all_attn_weights: torch.Tensor,
    save_path: Path,
    title: str = "Overall Mean Multi-scale Attention Heatmap",
):
    """
    把所有样本平均成 1 x S，再画 heatmap
    """
    overall_mean = all_attn_weights.mean(dim=0, keepdim=True)   # (1, S)
    data = overall_mean.detach().cpu().numpy()

    plt.figure(figsize=(7, 2.2))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Mean Attention Weight")

    plt.xlabel("Scale")
    plt.ylabel("All Samples")
    plt.title(title)

    plt.xticks(
        np.arange(data.shape[1]),
        [f"Scale {i}" for i in range(data.shape[1])]
    )
    plt.yticks([0], ["Mean"])

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"overall mean attention heatmap saved to: {save_path}")
    plt.close()


def main():
    save_dir = Path("plot") / "output" / "attention"
    # test_dir = Path("data") / "data_for_train_test_v14" / "12.25-wenguan-resample-filter-v2" / "test1"
    # ckpt_path = Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260112_1405_rmse_2d_1.129_wenguan.pt"
    
    test_dir = Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "test1"
    ckpt_path = Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260208_2141_rmse_2d_1.109_tcn_xinxi.pt"

    
    grouped_heatmap_path = save_dir / "grouped_mean_attention_heatmap_xinxi_test1.png"
    overall_heatmap_path = save_dir / "overall_mean_attention_heatmap_xinxi_test1.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    input_dim_reflect = {
        "x_mag": 3,
        "x_mag_grad": 9,
    }

    feature_transform = build_transform(input_key=input_key)

    # 这里要和你的 checkpoint 对应
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

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

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

    print("Collecting attention weights...")

    with torch.no_grad():
        for batch in test_loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            _, attn_weights = model(x)   # attn_weights: (B, S)
            all_attn_weights.append(attn_weights.detach().cpu())

    if len(all_attn_weights) == 0:
        raise RuntimeError("No attention weights collected from test_loader.")

    all_attn_weights = torch.cat(all_attn_weights, dim=0)   # (N, S)

    print("all_attn_weights.shape =", tuple(all_attn_weights.shape))
    print("overall mean attention =", all_attn_weights.mean(dim=0).tolist())

    # 你可以改这个分组大小，论文里常用 30 / 50 / 100
    group_size = 30
    grouped_attn = compute_grouped_mean_attention(
        all_attn_weights=all_attn_weights,
        group_size=group_size,
    )

    print("grouped_attn.shape =", tuple(grouped_attn.shape))

    plot_grouped_mean_attention_heatmap(
        grouped_attn,
        save_path=grouped_heatmap_path,
        title=f"Grouped Mean Multi-scale Attention Heatmap (group_size={group_size})",
    )

    plot_overall_mean_attention_heatmap(
        all_attn_weights,
        save_path=overall_heatmap_path,
        title="Overall Mean Multi-scale Attention Heatmap",
    )


if __name__ == "__main__":
    main()