import time
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from datasets import create_magnetic_dataset_v1_dataloaders
from network.mail import MAIL


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    seen = 0

    for batch in loader or []:
        x_mag = batch["x_mag"].to(device, non_blocking=True)  # (B, T, 3)
        y = batch["y"].to(device, non_blocking=True).float()  # (B, 2)

        optimizer.zero_grad(set_to_none=True)
        preds = model(x_mag)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        batch_size = x_mag.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    if loader is None:
        return float("nan")

    model.eval()
    running_loss = 0.0
    seen = 0

    for batch in loader:
        x_mag = batch["x_mag"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True).float()

        preds = model(x_mag)
        loss = criterion(preds, y)

        batch_size = x_mag.size(0)
        running_loss += loss.item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


def log_losses_csv(train_losses, val_losses, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "losses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, train_loss in enumerate(train_losses, 1):
            val_loss = val_losses[epoch - 1] if epoch - 1 < len(val_losses) else float("nan")
            writer.writerow([epoch, train_loss, val_loss])
    print(f"Saved loss log to {csv_path}")


def plot_losses(train_losses, val_losses, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "loss_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss", color="#1f77b4")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MAIL Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {fig_path}")


def main():
    data_dir = Path("data") / "data_for_train_test_v4" / "seq_300" / "mag"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    # Hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 50
    weight_decay = 1e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader, val_loader = create_magnetic_dataset_v1_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        use_imu=False,
        pattern=".npz",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
    )
    if train_loader is None:
        raise RuntimeError(f"Training split not found under {data_dir}")

    model = MAIL(
        input_dim=3,
        seq_len=300,
        scale_lengths=(10, 30, 50, 100, 300),
        gru_hidden=64,
        proj_dim=64,
        attn_hidden=64,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoints_dir = Path("checkpoints") / "mail"
    logs_dir = Path("runs") / "mail"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_dir / "mail_best.pt"

    best_val = float("inf")
    train_losses, val_losses = [], []

    print("----- MAIL 训练开始 -----")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        train_losses.append(train_loss)
        if not math.isnan(val_loss):
            val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} ({elapsed:.1f}s)")

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
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

    log_losses_csv(train_losses, val_losses, logs_dir)
    plot_losses(train_losses, val_losses, logs_dir)


if __name__ == "__main__":
    main()
