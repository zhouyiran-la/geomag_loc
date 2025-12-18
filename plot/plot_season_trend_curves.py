import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets.single_npz_dataset import create_single_npz_dataloader
from datasets.utils import build_transform
from pathlib import Path
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer



def plot_scale_season_trend(x_list, season_list, trend_list, res_dir: Path):
    """
    可视化所有尺度下 season/trend 的分解效果。
    
    x_list      : list of (B, L_m, D)
    season_list : 同 shape
    trend_list  : 同 shape
    """

    res_dir.mkdir(parents=True, exist_ok=True)

    num_scales = len(x_list)
    B = x_list[0].shape[0]

    # 只画第一条样本 (B=0)，你也可以改成所有
    idx = 0

    for m in range(num_scales):
        x     = x_list[m][idx].detach().cpu()        # (L_m, D)
        s     = season_list[m][idx].detach().cpu()   # (L_m, D)
        t     = trend_list[m][idx].detach().cpu()    # (L_m, D)

        L = x.shape[0]

        plt.figure(figsize=(10, 4))
        plt.plot(x[:, 0], label="original", linewidth=1)
        plt.plot(s[:, 0], label="seasonal", linewidth=1)
        plt.plot(t[:, 0], label="trend", linewidth=1.5)
        plt.title(f"Scale {m}: Length={L}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / f"scale_{m}_season_trend.png", dpi=300)
        plt.close()

    print(f"[TimeMixer] 各尺度季节/趋势分解图已保存至：{res_dir}")



def do_plot(
    npz_path: str,
    ckpt_path: Path,
    res_dir: Path,
    input_key: str = "x_mag_grad",
    *,
    seq_len: int = 128,
    d_model: int = 128,
    down_sampling_window: int = 2,
    down_sampling_layers: int = 3,
    num_pdm_blocks: int = 1,
    moving_avg_kernel: int = 11,
    nhead: int = 8,
    num_layers: int = 4,
    batch_size: int = 64,
):
    """
    Evaluate a single npz file using MagneticLocalizationTimeMixer and dump per-sample predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    feature_transform = build_transform(input_key=input_key)

    input_dim_reflect = {"x_mag": 3, "x_mag_grad": 9}

    loader = create_single_npz_dataloader(
        npz_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform=feature_transform,
    )
    if loader is None:
        raise RuntimeError("loader为空，请检查文件路径是否存在！")
    model = MagneticLocalizationTimeMixer(
        input_dim=input_dim_reflect[input_key],
        d_model=d_model,
        seq_len=seq_len,
        down_sampling_window=down_sampling_window,
        down_sampling_layers=down_sampling_layers,
        num_pdm_blocks=num_pdm_blocks,
        moving_avg_kernel=moving_avg_kernel,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=2,
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    one_batch = next(iter(loader)).to(device)
    
