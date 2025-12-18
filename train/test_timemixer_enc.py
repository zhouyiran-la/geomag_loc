import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets.single_npz_dataset import create_single_npz_dataloader
from datasets.utils import build_transform
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer


def _eval_single_npz(
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
    model.eval()

    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0

    all_preds, all_labels, all_errors = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch[input_key].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()
            preds, _ = model(x)

            loss = criterion(preds, y)
            total_loss += loss.item()
            total_samples += x.size(0)

            preds_np = preds.cpu().numpy()
            labels_np = y.cpu().numpy()
            errors = np.linalg.norm(preds_np - labels_np, axis=1)

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            all_errors.extend(errors)

    mse = total_loss / max(total_samples, 1)
    mean_error = float(np.mean(all_errors)) if all_errors else float("nan")
    print(
        f"Single file eval -> {npz_path} | samples={total_samples} | "
        f"MSE={mse:.6f} | mean_error={mean_error:.6f}"
    )

    res_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{Path(npz_path).stem}_meanerr_{mean_error:.4f}.csv"
    output_csv = res_dir / file_name

    results_df = pd.DataFrame(
        {
            "pred_x": [pred[0] for pred in all_preds],
            "pred_y": [pred[1] for pred in all_preds],
            "true_x": [label[0] for label in all_labels],
            "true_y": [label[1] for label in all_labels],
            "euclidean_error": all_errors,
        }
    )
    results_df.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")

    return mse, mean_error, results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single NPZ file with TimeMixer.")
    parser.add_argument("--npz-path", required=True, help="Path to the .npz file to evaluate.")
    parser.add_argument("--best-path", required=True, help="Path to the trained checkpoint (.pt).")
    parser.add_argument("--res-dir", required=True, help="Directory to store evaluation csv files.")
    parser.add_argument(
        "--input-key",
        default="x_mag_grad",
        choices=("x_mag", "x_mag_grad"),
        help="Input feature key used by the dataset and model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    npz_path = args.npz_path
    best_path = Path(args.best_path)
    res_dir = Path(args.res_dir)
    input_key = args.input_key

    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}")

    _eval_single_npz(npz_path, best_path, res_dir, input_key=input_key)
