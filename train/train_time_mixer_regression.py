import time
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from datasets import create_magnetic_dataset_v1_dataloaders
from datasets.utils import build_transform
from network.magnetic_localization_model_time_mixer_regress import MagneticLocalizationTimeMixer

    
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
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / max(total_samples, 1)


def evaluate(model, loader, criterion, device, input_key="x_mag_grad"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()
            preds, _ = model(x)
            loss = criterion(preds, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / max(total_samples, 1)


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
    plt.ylabel("MSE Loss")
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
    data_dir = str(Path("data") / "data_for_train_test_v11" / "seq_128_10" / "mag")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    batch_size = 64
    lr = 1e-3
    epochs = 200
    weight_decay = 1e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    input_dim_reflect = {"x_mag":3, "x_mag_grad":9}
    feature_transform = build_transform(input_key=input_key)

    train_loader, val_loader = create_magnetic_dataset_v1_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_imu=False,
        pattern=".npz",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
        transform=feature_transform
    )

    model = MagneticLocalizationTimeMixer(
        input_dim=input_dim_reflect[input_key],
        d_model=128,
        seq_len=128,
        down_sampling_window=2,
        down_sampling_layers=3,
        num_pdm_blocks=1,
        moving_avg_kernel=11, 
        nhead=8,
        num_layers=4,
        output_dim=2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")

    if input_key == "x_mag_grad":
        run_dir = Path("runs") / "loss_time_mixer_grad"
        checkpoints_dir = Path("checkpoints") / "time_mixer_grad"
    else:
        run_dir = Path("runs") / "loss_time_mixer"
        checkpoints_dir = Path("checkpoints") / "time_mixer"

    date_suffix = datetime.now().strftime("_%Y%m%d")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / f"mag_localization_time_mixer_best{date_suffix}.pt"

    
    train_losses, val_losses = [], []
    print("----------- Training MagneticLocalizationTimeMixer -----------")
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, input_key=input_key)
        val_loss = evaluate(model, val_loader, criterion, device, input_key=input_key) if val_loader is not None else float("nan")
        elapsed = time.time() - start

        train_losses.append(train_loss)
        if val_loader is not None:
            val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} ({elapsed:.1f}s)")

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"  Saved TimeMixer checkpoint -> {best_path}")

    plot_and_save_losses(train_losses, val_losses, run_dir, suffix=date_suffix)


if __name__ == "__main__":
    main()
