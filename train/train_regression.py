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
from network.magnetic_localization_model_regress_fast import MagneticLocalizationModelFast


def train_one_epoch(model, loader, criterion, optimizer, device, input_key: str = "x_mag"):
    model.train()
    epoch_loss = 0.0
    num_samples = 0
    for batch in loader or []:
        x = batch[input_key].to(device, non_blocking=True)  # (B, N, C)
        y = batch["y"].to(device, non_blocking=True).float()  # (B, 2)

        optimizer.zero_grad(set_to_none=True)
        preds, _ = model(x)  # (B, 2)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        epoch_loss += loss.item() * bs
        num_samples += bs
    return epoch_loss / max(num_samples, 1)


def evaluate(model, loader, criterion, device, input_key: str = "x_mag", *, y_norm_mode="per_file_minmax", stats=None):
    model.eval()
    epoch_loss = 0.0
    num_samples = 0

    sum_l2 = 0.0
    sum_l1 = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0

    with torch.no_grad():
        for batch in loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()
            preds, _ = model(x)
            loss = criterion(preds, y)
            bs = x.size(0)
            epoch_loss += loss.item() * bs
            num_samples += bs

            preds_real = denorm_y(preds, batch, y_norm_mode=y_norm_mode, stats=stats, device=device)
            y_real = batch["y_raw"].to(device, non_blocking=True).float()
            diff = preds_real - y_real

            l2 = torch.norm(diff, dim=1)
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
    csv_path = out_dir / f"losses_regression{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, tl in enumerate(train_losses, start=1):
            vl = val_losses[i - 1] if i - 1 < len(val_losses) else float("nan")
            writer.writerow([i, tl, vl])

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss", color="#1f77b4")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("WeightedSmoothL1 Loss")
    plt.title("MagneticLocalization Regression Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = out_dir / f"loss_curve_regression{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {fig_path}")
    print(f"Saved loss csv to {csv_path}")


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
    epochs = 300
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

    model = MagneticLocalizationModelFast(
        input_dim=3,
        d_model=64,
        scales=[64, 128, 256],
        nhead=8,
        num_layers=3,
        output_dim=2,
    ).to(device)

    criterion = WeightedSmoothL1(beta=0.05, w_x=1.0, w_y=1.3).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoints_dir = Path("checkpoints") / "multiscale_transformer"
    run_dir = Path("runs") / "loss_multiscale_transformer"
    date_suffix = datetime.now().strftime("_%Y%m%d_%H%M")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / f"mag_localization_regress_best{date_suffix}.pt"

    best_val = float("inf")
    train_losses, val_losses = [], []
    print("----------- Training Magnetic Localization Regression -----------")
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

        score = val_metrics.get("rmse_2d", float("inf"))
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
            print(f"  Saved best checkpoint -> {best_path}")

        scheduler.step()

    plot_and_save_losses(train_losses, val_losses, run_dir, suffix=date_suffix)


if __name__ == "__main__":
    main()
