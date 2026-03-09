from pathlib import Path

import matplotlib.pyplot as plt
import torch
from typing import Optional
from datasets import create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer
from plot.utils.plot_style import setup_plot_season_trend_style, style_axis, save_figure

setup_plot_season_trend_style()

def extract_multiscale_decomposition(model, x: torch.Tensor):
    """
    直接从 model.timemixer_encoder 提取多尺度分解结果
    x: (B, T, C)

    return:
        {
            "x_scales": list[(B, L_k, C)],
            "season_scales": list[(B, L_k, C)],
            "trend_scales": list[(B, L_k, C)],
        }
    """
    encoder = model.timemixer_encoder

    if not hasattr(encoder, "_multi_scale_inputs"):
        raise AttributeError("timemixer_encoder 缺少 _multi_scale_inputs 方法")

    if not getattr(encoder, "use_decomp", False):
        raise ValueError("当前模型 use_decomp=False，无法提取 season/trend 分解结果")

    if not hasattr(encoder, "decomp") or encoder.decomp is None:
        raise ValueError("当前模型没有可用的 decomp 模块")

    with torch.no_grad():
        x_scales = encoder._multi_scale_inputs(x)

        season_scales = []
        trend_scales = []
        for xs in x_scales:
            s, t = encoder.decomp(xs)
            season_scales.append(s)
            trend_scales.append(t)

    return {
        "x_scales": x_scales,
        "season_scales": season_scales,
        "trend_scales": trend_scales,
    }


def plot_multiscale_decomposition(
    decomp_dict,
    save_path: Path,
    batch_idx: int = 0,
    channel_idx: int = 0,
    show: bool = False,
    suptitle: Optional[str] = None
):
    x_scales = decomp_dict["x_scales"]
    season_scales = decomp_dict["season_scales"]
    trend_scales = decomp_dict["trend_scales"]

    num_scales = len(x_scales)

    base = 3.3   # 单个子图边长
    fig, axes = plt.subplots(
        nrows=num_scales,
        ncols=3,
        figsize=(3 * base, num_scales * base),
        squeeze=False,
    )
    # for ax in axes.flat:
    #     ax.margins(x=0)
    for i in range(num_scales):
        x_i = x_scales[i][batch_idx, :, channel_idx].detach().cpu().numpy()
        s_i = season_scales[i][batch_idx, :, channel_idx].detach().cpu().numpy()
        t_i = trend_scales[i][batch_idx, :, channel_idx].detach().cpu().numpy()

        # 只有最后一行显示 x label
        xlabel = "序号" if i == num_scales - 1 else None

        # 只有最左列显示 y label
        ylabel0 = "值"
        ylabel1 = None
        ylabel2 = None

        axes[i, 0].plot(x_i)
        style_axis(
            axes[i, 0],
            title=f"Scale {i}: Original",
            xlabel=xlabel, # type: ignore
            ylabel=ylabel0,
        )

        axes[i, 1].plot(t_i)
        style_axis(
            axes[i, 1],
            title=f"Scale {i}: Trend",
            xlabel=xlabel, # type: ignore
            ylabel=ylabel1, # type: ignore
        )

        axes[i, 2].plot(s_i)
        style_axis(
            axes[i, 2],
            title=f"Scale {i}: Seasonal",
            xlabel=xlabel, # type: ignore
            ylabel=ylabel2, # type: ignore
        )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16, y=0.985)

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.06,
        top=0.94,
        hspace=0.25,
        wspace=0.15,
    )

    save_figure(fig, save_path, show=show, tight=False)


def plot_one_scale_all_channels(
    decomp_dict,
    save_path: Path,
    scale_idx: int = 0,
    batch_idx: int = 0,
    show: bool = False,
    suptitle: Optional[str] = None
):
    x = decomp_dict["x_scales"][scale_idx][batch_idx].detach().cpu().numpy()        # (L, C)
    s = decomp_dict["season_scales"][scale_idx][batch_idx].detach().cpu().numpy()
    t = decomp_dict["trend_scales"][scale_idx][batch_idx].detach().cpu().numpy()

    num_channels = x.shape[1]

    base = 3.3   # 单个子图边长
    fig, axes = plt.subplots(
        nrows=num_channels,
        ncols=3,
        figsize=(3 * base, num_channels * base),
        squeeze=False,
    )

    for c in range(num_channels):
        # 只有最后一行显示 x label
        xlabel = "序号" if c == num_channels - 1 else None

        # 只有最左列显示 y label
        ylabel0 = "值"
        ylabel1 = None
        ylabel2 = None

        axes[c, 0].plot(x[:, c])
        style_axis(
            axes[c, 0],
            title=f"Channel {c}: Original",
            xlabel=xlabel,  # type: ignore
            ylabel=ylabel0,
        )

        axes[c, 1].plot(t[:, c])
        style_axis(
            axes[c, 1],
            title=f"Channel {c}: Trend",
            xlabel=xlabel,  # type: ignore
            ylabel=ylabel1,  # type: ignore
        )

        axes[c, 2].plot(s[:, c])
        style_axis(
            axes[c, 2],
            title=f"Channel {c}: Seasonal",
            xlabel=xlabel,  # type: ignore
            ylabel=ylabel2,  # type: ignore
        )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16, y=0.985)

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.06,
        top=0.94,
        hspace=0.25,
        wspace=0.15,
    )

    save_figure(fig, save_path, show=show, tight=False)


def main():
    test_dir = Path("data") / "data_for_train_test_v14" / "12.25-wenguan-resample-zscore" / "test1"
    ckpt_path = Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260123_2059_rmse_2d_1.000_256_wenguan.pt"
    res_dir = Path("plot") / "output" / "decomp_plots"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    input_dim_reflect = {"x_mag": 3, "x_mag_grad": 9}
    feature_transform = build_transform(input_key=input_key)

    model = MagneticLocalizationTimeMixer(
        input_dim=input_dim_reflect[input_key],
        d_model=128,
        seq_len=256,
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
        seq_len=256,
        stride=20,
    )

    first_batch = None
    for batch in test_loader or []:
        first_batch = batch
        break

    if first_batch is None:
        raise RuntimeError("test_loader 为空，没有可视化样本")

    x = first_batch[input_key].to(device, non_blocking=True)

    with torch.no_grad():
        decomp_dict = extract_multiscale_decomposition(model, x)

    print("===== decomposition shapes =====")
    for i, xs in enumerate(decomp_dict["x_scales"]):
        s = decomp_dict["season_scales"][i]
        t = decomp_dict["trend_scales"][i]
        print(
            f"scale {i}: "
            f"x={tuple(xs.shape)}, "
            f"season={tuple(s.shape)}, "
            f"trend={tuple(t.shape)}"
        )

    # 画第一个样本、第0通道，在所有尺度上的分解
    plot_multiscale_decomposition(
        decomp_dict=decomp_dict,
        save_path=res_dir / "multiscale_decomp_channel0_wenguan_test1_256_2.png",
        batch_idx=12,
        channel_idx=0,
        show=False,
    )

    # 画第0个尺度上，所有通道的分解
    plot_one_scale_all_channels(
        decomp_dict=decomp_dict,
        save_path=res_dir / "scale0_all_channels_wenguan_test1.png",
        scale_idx=0,
        batch_idx=12,
        show=False,
    )


if __name__ == "__main__":
    main()