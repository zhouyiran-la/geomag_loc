# visualize_mixing.py

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer

def plot_matrix(W, title, path):
    W = W.detach().cpu().numpy()
    plt.figure(figsize=(6,5))
    plt.imshow(W, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_season_trend_weights(model, out_dir:Path):
    # === 3. 提取 PDM block ===
    pdm = model.timemixer_encoder.pdm_blocks[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    # === 4. 提取 Seasonal Mixer 第一层 (128→64) ===
    season_layer = pdm.season_mixer.layers[0][0]  # 第一层 Linear
    W_season = season_layer.weight
    plot_matrix(W_season, "Seasonal Mixing (128→64)", out_dir / "season_128_64_2.png")

    # === 5. 提取 Trend Mixer 最后一层 (64→128) ===
    trend_layer = pdm.trend_mixer.layers[-1][0]
    W_trend = trend_layer.weight
    plot_matrix(W_trend, out_dir / "Trend Mixing (64→128)", out_dir / "trend_64_128_2.png")

    print("图已生成：season_128_64_2.png, trend_64_128_2.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("plot") / "output"
    # === 1. 创建模型 ===
    model = MagneticLocalizationTimeMixer(
        input_dim=9,
        d_model=128,
        seq_len=128,
        down_sampling_window=2,
        down_sampling_layers=3,
        num_pdm_blocks=1,
        moving_avg_kernel=11, 
        nhead=8,
        num_layers=4,
        output_dim=2,
    )
    ckpt = torch.load("checkpoints/time_mixer_grad/mag_localization_time_mixer_best_20251121.pt", map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    # === 2. 读取训练好的参数 ===
    model.load_state_dict(state_dict)

    # === 3. 提取 PDM block ===
    pdm = model.timemixer_encoder.pdm_blocks[0]

    # === 4. 提取 Seasonal Mixer 第一层 (128→64) ===
    season_layer = pdm.season_mixer.layers[0][0]  # 第一层 Linear
    W_season = season_layer.weight
    plot_matrix(W_season, "Seasonal Mixing (128→64)", out_dir / "season_128_64_2.png")

    # === 5. 提取 Trend Mixer 最后一层 (64→128) ===
    trend_layer = pdm.trend_mixer.layers[-1][0]
    W_trend = trend_layer.weight
    plot_matrix(W_trend, "Trend Mixing (64→128)", out_dir / "trend_64_128_2.png")

    print("图已生成：season_128_64.png, trend_64_128.png")

if __name__ == "__main__":
    main()
