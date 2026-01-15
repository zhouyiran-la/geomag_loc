import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from pathlib import Path
from datasets import MagneticDataSetV2, create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform, denorm_y
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer
from network.losses import WeightedSmoothL1

def test(
    model, 
    loader, 
    criterion, 
    device, 
    ckpt_path:Path, 
    res_dir:Path, 
    input_key="x_mag", *, 
    y_norm_mode="per_file_minmax", 
    stats=None
    ):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    total_loss = 0.0
    total_samples = 0

    # real-space error
    sum_l2 = 0.0          # mean L2 error累计（单位：你的坐标单位）
    sum_l1 = 0.0          # mean |dx|+|dy|
    sum_dx2 = 0.0         # RMSE用
    sum_dy2 = 0.0
    all_preds, all_labels, all_errors = [], [], []

    with torch.no_grad():
        for batch in loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()

            preds, _ = model(x)
            loss = criterion(preds, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            # ===== 逆归一化到真实坐标 =====
            # 需要 batch["y_raw"] (B,2)，以及 per_file 模式下 batch["y_stats"] (B,4)
            preds_real = denorm_y(preds, batch, y_norm_mode=y_norm_mode, stats=stats, device=device)
            y_real = batch["y_raw"].to(device, non_blocking=True).float()

            diff = preds_real - y_real
            l2 = torch.norm(diff, dim=1)          # (B,)
            l1 = diff.abs().sum(dim=1)            # (B,)

            preds_np = preds_real.cpu().numpy()
            labels_np = y_real.cpu().numpy()
            errors = np.linalg.norm(preds_np - labels_np, axis=1)

            sum_l2 += l2.sum().item()
            sum_l1 += l1.sum().item()
            sum_dx2 += (diff[:, 0] ** 2).sum().item()
            sum_dy2 += (diff[:, 1] ** 2).sum().item()

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            all_errors.extend(errors)

    # 指标计算
    denom = max(total_samples, 1)
    val_loss = total_loss / denom
    mean_l2 = sum_l2 / denom
    mean_l1 = sum_l1 / denom
    mse_x = sum_dx2 / denom
    mse_y = sum_dy2 / denom
    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)
    print(f"val_loss={val_loss:.6f} | "
            f"mean_l1={mean_l1:.3f} mean_l2={mean_l2:.3f} rmse_x={rmse_x:.3f} rmse_y={rmse_y:.3f} rmse_2d={rmse_2d:.3f}")
    
    res_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"wenguan_test1_64_loc_res_meanerr_{mean_l2:.4f}.csv"
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


if __name__ == "__main__":

    test_dir = Path("data") / "data_for_train_test_v14" / "12.25-wenguan-resample-filter" / "test3"
    ckpt_path = Path("checkpoints") / "time_mixer" / "time_mixer_enc_loc_best_20260112_1405_rmse_2d_1.129_wenguan.pt"
    res_dir = Path("runs") / "loc_res" / "time_mixer"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    input_dim_reflect = {"x_mag":3, "x_mag_grad":9}
    feature_transform = build_transform(input_key=input_key)
    criterion = WeightedSmoothL1(beta=0.05, w_x=1.0, w_y=1.0).to(device)

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

    test_loader = create_magnetic_dataset_v2_dataloaders(
        str(test_dir),
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=False,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=128,
        stride=20
    )

    test(model, test_loader, criterion, device, ckpt_path, res_dir, input_key=input_key)
