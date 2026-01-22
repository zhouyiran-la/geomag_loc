import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import MagneticDataSetV2, create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform, denorm_y
from network.losses import WeightedSmoothL1
from network.hlstm import HLSTMRegressor


# ---------------------------
# 1) 训练 
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, input_key: str = "x_mag"):
    model.train()
    epoch_loss = 0.0
    num_samples = 0
    for batch in loader or []:
        x = batch[input_key].to(device, non_blocking=True)        # (B, L, 3)
        y = batch["y"].to(device, non_blocking=True).float()      # (B, 2)

        optimizer.zero_grad(set_to_none=True)
        preds, _ = model(x)                                       # (B, 2)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        epoch_loss += loss.item() * bs
        num_samples += bs
    return epoch_loss / max(num_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, input_key: str = "x_mag", *, y_norm_mode="per_file_minmax", stats=None):
    model.eval()
    epoch_loss = 0.0
    num_samples = 0

    sum_l2 = 0.0
    sum_l1 = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0

    for batch in loader or []:
        x = batch[input_key].to(device, non_blocking=True)         # (B, L, 3)
        y = batch["y"].to(device, non_blocking=True).float()       # (B, 2)

        preds, _ = model(x)
        loss = criterion(preds, y)

        bs = x.size(0)
        epoch_loss += loss.item() * bs
        num_samples += bs

        # 反归一化到真实坐标，用于算定位误差
        preds_real = denorm_y(preds, batch, y_norm_mode=y_norm_mode, stats=stats, device=device)
        y_real = batch["y_raw"].to(device, non_blocking=True).float()
        diff = preds_real - y_real

        l2 = torch.norm(diff, dim=1)                 # 欧式距离
        l1 = diff.abs().sum(dim=1)

        sum_l2 += l2.sum().item()
        sum_l1 += l1.sum().item()
        sum_dx2 += (diff[:, 0] ** 2).sum().item()
        sum_dy2 += (diff[:, 1] ** 2).sum().item()

    denom = max(num_samples, 1)
    val_loss = epoch_loss / denom
    mean_l2 = sum_l2 / denom
    mean_l1 = sum_l1 / denom

    mse_x = sum_dx2 / denom
    mse_y = sum_dy2 / denom
    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)

    metrics = {
        "mean_l2": mean_l2,
        "mean_l1": mean_l1,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_2d": rmse_2d,
    }
    return val_loss, metrics


def plot_and_save_losses(train_losses, val_losses, out_dir: Path, suffix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"losses_hlstm{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, tl in enumerate(train_losses, start=1):
            vl = val_losses[i - 1] if i - 1 < len(val_losses) else float("nan")
            writer.writerow([i, tl, vl])

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("WeightedSmoothL1 Loss")
    plt.title("HLSTM Regression Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = out_dir / f"loss_curve_hlstm{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {fig_path}")
    print(f"Saved loss csv to {csv_path}")


# ---------------------------
# 2) main：整体训练流程
# ---------------------------

def main():
    train_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "train")
    val_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "eval")

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # 训练超参（可按需调）
    batch_size = 32
    lr = 5e-4
    epochs = 300
    weight_decay = 3e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    # Dataset/transform 设置
    input_key = "x_mag"
    feature_transform = build_transform(input_key=input_key)

    # DataLoader 里 seq_len=256, stride=20（注意：这里的 stride 是窗口步长，不是 HLSTM 的 frame_stride）
    train_loader = create_magnetic_dataset_v2_dataloaders(
        train_dir,
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=256,
        stride=20,
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
        stride=20,
    )

    # -----------------------
    # 构造 HLSTM baseline
    # 下面这组是一个合理的起点：
    #   L=256
    #   frame_len=32, frame_stride=16 -> 大约 (256-32)/16+1 = 15 帧
    # 你也可以试：frame_len=64, frame_stride=32
    # -----------------------
    model = HLSTMRegressor(
        input_dim=3,
        frame_len=32,
        frame_stride=16,
        frame_hidden=64,
        frame_layers=1,
        seq_hidden=128,
        seq_layers=1,
        bidirectional=True,
        dropout=0.2,
        pool="last",
        output_dim=2,
    ).to(device)

    # 损失/优化器/调度器（完全沿用你的设置）
    criterion = WeightedSmoothL1(beta=0.05, w_x=1.0, w_y=1.3).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 输出路径
    checkpoints_dir = Path("checkpoints") / "hlstm"
    run_dir = Path("runs") / "loss_hlstm"
    date_suffix = datetime.now().strftime("_%Y%m%d_%H%M")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / f"mag_localization_hlstm_best{date_suffix}.pt"

    best_val = float("inf")
    train_losses, val_losses = [], []
    print("----------- Training HLSTM Regression Baseline -----------")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, input_key=input_key)

        if val_loader is not None:
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                input_key=input_key,
                y_norm_mode="per_file_minmax",
                stats=cast(MagneticDataSetV2, val_loader.dataset).stats,
            )
        else:
            val_loss, val_metrics = float("nan"), {}

        dt = time.time() - t0
        train_losses.append(train_loss)
        if val_loader is not None:
            val_losses.append(val_loss)

        cur_lr = optimizer.param_groups[0]["lr"]
        if val_loader is not None:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"mean_l2={val_metrics['mean_l2']:.3f} rmse_x={val_metrics['rmse_x']:.3f} "
                f"rmse_y={val_metrics['rmse_y']:.3f} rmse_2d={val_metrics['rmse_2d']:.3f} "
                f"lr={cur_lr:.2e} ({dt:.1f}s)"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | lr={cur_lr:.2e} ({dt:.1f}s)")

        # 这里沿用你原来用 rmse_2d 作为 best 选择标准
        score = val_metrics.get("rmse_2d", float("inf"))
        if val_loader is not None and score < best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_rmse_2d": best_val,
                    "hlstm_cfg": {
                        "frame_len": model.frame_len,
                        "frame_stride": model.frame_stride,
                        "frame_hidden": model.frame_lstm.hidden_size,
                        "seq_hidden": model.seq_lstm.hidden_size,
                        "bidirectional": model.seq_lstm.bidirectional,
                        "pool": model.pool,
                    },
                },
                best_path,
            )
            print(f"  Saved best checkpoint -> {best_path}")

        scheduler.step()

    plot_and_save_losses(train_losses, val_losses, run_dir, suffix=date_suffix)


if __name__ == "__main__":
    main()