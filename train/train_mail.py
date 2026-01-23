import time
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import MagneticDataSetV2, create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform, denorm_y
from network.losses import WeightedSmoothL1
from network.mail import MAIL


def train_one_epoch(model, loader, criterion, optimizer, device, input_key: str = "x_mag"):
    model.train()
    running_loss = 0.0
    seen = 0

    for batch in loader or []:
        x = batch[input_key].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


def evaluate(model, loader, criterion, device, input_key: str = "x_mag", *, y_norm_mode="per_file_minmax", stats=None):
    model.eval()
    running_loss = 0.0
    seen = 0

    sum_l2 = 0.0
    sum_l1 = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0

    with torch.no_grad():
        for batch in loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()

            preds = model(x)
            loss = criterion(preds, y)

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

            preds_real = denorm_y(preds, batch, y_norm_mode=y_norm_mode, stats=stats, device=device)
            y_real = batch["y_raw"].to(device, non_blocking=True).float()
            diff = preds_real - y_real

            l2 = torch.norm(diff, dim=1)
            l1 = diff.abs().sum(dim=1)

            sum_l2 += l2.sum().item()
            sum_l1 += l1.sum().item()
            sum_dx2 += (diff[:, 0] ** 2).sum().item()
            sum_dy2 += (diff[:, 1] ** 2).sum().item()

    denom = max(seen, 1)
    val_loss = running_loss / denom
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
    csv_path = out_dir / f"losses_mail{suffix}.csv"
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
    plt.title("MAIL Training")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = out_dir / f"loss_curve_mail{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved MAIL loss curve to {fig_path}")
    print(f"Saved MAIL loss csv to {csv_path}")


def main():
    train_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "train")
    val_dir = str(Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "eval")

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    batch_size = 32
    lr = 5e-4
    epochs = 400
    weight_decay = 3e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    feature_transform = build_transform(input_key=input_key)

    train_loader = create_magnetic_dataset_v2_dataloaders(
        train_dir,
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
        transform=feature_transform,
        seq_len=128,
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
        seq_len=128,
        stride=20,
    )

    model = MAIL(
        input_dim=3,
        seq_len=128,
        scale_lengths=(32, 64, 128),
        gru_hidden=128,
        proj_dim=64,
        attn_hidden=128,
    ).to(device)

    # criterion = nn.MSELoss()
    criterion = WeightedSmoothL1(beta=0.05, w_x=1.0, w_y=1.3).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoints_dir = Path("checkpoints") / "mail"
    logs_dir = Path("runs") / "mail"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_suffix = datetime.now().strftime("_%Y%m%d_%H%M")
    best_ckpt = checkpoints_dir / f"mag_localization_mail_best{date_suffix}.pt"

    best_val = float("inf")
    train_losses, val_losses = [], []

    print("----------- Training MAIL -----------")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
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

        elapsed = time.time() - start_time

        train_losses.append(train_loss)
        if val_loader is not None:
            val_losses.append(val_loss)

        cur_lr = optimizer.param_groups[0]["lr"]
        if val_loader is not None:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"mean_l2={val_metrics['mean_l2']:.3f} rmse_x={val_metrics['rmse_x']:.3f} "
                f"rmse_y={val_metrics['rmse_y']:.3f} rmse_2d={val_metrics['rmse_2d']:.3f} "
                f"lr={cur_lr:.2e} ({elapsed:.1f}s)"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | lr={cur_lr:.2e} ({elapsed:.1f}s)")

        score = val_metrics.get("rmse_2d", float("inf"))
        if val_loader is not None and score < best_val:
            best_val = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt,
            )
            print(f"  Saved new best checkpoint to {best_ckpt}")

        scheduler.step()

    plot_and_save_losses(train_losses, val_losses, logs_dir, suffix=date_suffix)


if __name__ == "__main__":
    main()
