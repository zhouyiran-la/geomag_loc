import time
import csv
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from datetime import datetime
from pathlib import Path
from typing import cast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from datasets import MagneticDataSetV2, create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform, denorm_y
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer
from network.losses import WeightedSmoothL1

from train.test_timemixer_enc_v2 import test
    
def train_one_epoch(model, loader, criterion, optimizer, device, input_key="x_mag_grad"):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader or []:
        x = batch[input_key].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        preds, _ = model(x)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / max(total_samples, 1)

def evaluate(model, loader, criterion, device, input_key="x_mag", *, y_norm_mode="per_file_minmax", stats=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # real-space error
    sum_l2 = 0.0          # mean L2 error累计（单位：你的坐标单位）
    sum_l1 = 0.0          # mean |dx|+|dy|
    sum_dx2 = 0.0         # RMSE用
    sum_dy2 = 0.0

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

            sum_l2 += l2.sum().item()
            sum_l1 += l1.sum().item()
            sum_dx2 += (diff[:, 0] ** 2).sum().item()
            sum_dy2 += (diff[:, 1] ** 2).sum().item()

    denom = max(total_samples, 1)
    val_loss = total_loss / denom
    mean_l2 = sum_l2 / denom
    mean_l1 = sum_l1 / denom
    mse_x = sum_dx2 / denom
    mse_y = sum_dy2 / denom
    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)

    return val_loss, {"mean_l2": mean_l2, "mean_l1": mean_l1, "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse_2d":rmse_2d}

# def evaluate(model, loader, criterion, device, input_key="x_mag_grad"):
#     model.eval()
#     total_loss = 0.0
#     total_samples = 0
#     with torch.no_grad():
#         for batch in loader or []:
#             x = batch[input_key].to(device, non_blocking=True)
#             y = batch["y"].to(device, non_blocking=True).float()
#             preds, _ = model(x)
#             loss = criterion(preds, y)
#             bs = x.size(0)
#             total_loss += loss.item() * bs
#             total_samples += bs
#     return total_loss / max(total_samples, 1)


def plot_and_save_losses(train_losses, val_losses, out_dir: Path, suffix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"losses_time_mixer{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for idx, train_loss in enumerate(train_losses, start=1):
            val_loss = val_losses[idx - 1] if idx - 1 < len(val_losses) else float("nan")
            writer.writerow([idx, train_loss, val_loss])

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss", color="#1f77b4")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("WeightedSmoothL1 Loss")
    plt.title("MagneticLocalizationTimeMixer Training")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = out_dir / f"loss_curve_time_mixer{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved TimeMixer loss curve to {fig_path}")
    print(f"Saved TimeMixer loss csv to {csv_path}")


def main():
    train_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "train")
    val_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "eval")
    test_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "test1")

    
    gpu_id = 0  # 用第1张卡
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    

    batch_size = 32
    lr = 5e-4
    epochs = 200
    # weight_decay改大了一些
    weight_decay = 5e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    input_dim_reflect = {"x_mag":3, "x_mag_grad":9}
    feature_transform = build_transform(input_key=input_key)

    train_loader = create_magnetic_dataset_v2_dataloaders(
        train_dir,
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=256,
        stride=20
    )

    val_loader = create_magnetic_dataset_v2_dataloaders(
        val_dir,
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=False,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=256,
        stride=20
    )

    test_loader = create_magnetic_dataset_v2_dataloaders(
        str(test_dir),
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=False,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=256,
        stride=20
    )

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
    ).to(device)

    criterion = WeightedSmoothL1(beta=0.05, w_x=1.0, w_y=1.0).to(device)
    # criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 学习率调度器
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val = float("inf")

    if input_key == "x_mag_grad":
        run_dir = Path("runs") / "loss_time_mixer_grad"
        checkpoints_dir = Path("checkpoints") / "time_mixer_grad"
    else:
        run_dir = Path("runs") / "loss_time_mixer"
        checkpoints_dir = Path("checkpoints") / "time_mixer"

    date_suffix = datetime.now().strftime("_%Y%m%d_%H%M")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / f"time_mixer_enc_loc_best{date_suffix}.pt"

    
    train_losses, val_losses = [], []
    print("----------- Training MagneticLocalizationTimeMixer -----------")
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, input_key=input_key)
        if val_loader is not None:
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device,
                input_key=input_key,
                y_norm_mode="per_file_minmax",
                stats=cast(MagneticDataSetV2, val_loader.dataset).stats, 
            )
        else:
            val_loss, val_metrics = float("nan"), {}

        elapsed = time.time() - start

        train_losses.append(train_loss)
        if val_loader is not None:
            val_losses.append(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"mean_l2={val_metrics['mean_l2']:.3f} rmse_x={val_metrics['rmse_x']:.3f} rmse_y={val_metrics['rmse_y']:.3f} rmse_2d={val_metrics['rmse_2d']:.3f} lr={cur_lr:.2e} ({elapsed:.1f}s)")
        # rmse_2d 用作保存 best checkpoint
        score = val_metrics["rmse_2d"]
        if val_loader is not None and score < best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  Saved TimeMixer checkpoint -> {best_path}")
        scheduler.step()

    plot_and_save_losses(train_losses, val_losses, run_dir, suffix=date_suffix)

    loc_res_dir = Path("runs") / "loc_res" / "time_mixer"
    test(model, test_loader, criterion, device, best_path, loc_res_dir, input_key=input_key)

if __name__ == "__main__":
    main()
