import os
import time
import csv
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import matplotlib.pyplot as plt

from datasets import create_magnetic_dataset_v1_dataloaders
from network.magnetic_localization_model_regress import MagneticLocalizationModel
from network.magnetic_localization_model_regress_fast import MagneticLocalizationModelFast

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    num_samples = 0
    for batch in loader or []:
        x_mag = batch["x_mag"].to(device, non_blocking=True)  # (B, N, 3)
        y = batch["y"].to(device, non_blocking=True).float()  # (B, 2)

        optimizer.zero_grad(set_to_none=True)
        preds, _ = model(x_mag)  # (B, 2)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        bs = x_mag.size(0)
        epoch_loss += loss.item() * bs
        num_samples += bs
    return epoch_loss / max(num_samples, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch in loader or []:
            x_mag = batch["x_mag"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()
            preds, _ = model(x_mag)
            loss = criterion(preds, y)
            bs = x_mag.size(0)
            epoch_loss += loss.item() * bs
            num_samples += bs
    return epoch_loss / max(num_samples, 1)


def plot_and_save_losses(train_losses, val_losses, out_dir: Path, suffix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 保存 CSV
    csv_path = out_dir / f"losses{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, tl in enumerate(train_losses, start=1):
            vl = val_losses[i-1] if i-1 < len(val_losses) else float("nan")
            writer.writerow([i, tl, vl])
    # 绘图
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train_loss", color="#1f77b4")
    # if len(val_losses) > 0:
    #     plt.plot(range(1, len(val_losses)+1), val_losses, label="val_loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = out_dir / f"loss_curve{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ 已保存损失曲线: {fig_path}")
    print(f"✓ 已保存损失明细: {csv_path}")


def main():
    data_dir = str(Path("data") / "data_for_train_test_v7" / "seq_100_20" / "mag")
    # 设备与加速设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 输入尺寸固定时可加速
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    # Hyperparams
    batch_size = 64
    lr = 1e-3
    epochs = 50
    weight_decay = 1e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    
    # Dataloaders (train/eval/test)
    train_loader, val_loader = create_magnetic_dataset_v1_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_imu=False,
        pattern=".npz",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
    )

    # Model
    model = MagneticLocalizationModelFast(input_dim=3, d_model=64, scales=[5, 10, 20, 50, 100], nhead=8, num_layers=3, output_dim=2)
    model.to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Logs & checkpoints
    best_val = float("inf")
    save_dir = Path("checkpoints")
    run_dir = Path("runs") / "loss"
    date_suffix = datetime.now().strftime("_%Y%m%d")
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"mag_localization_regress_best{date_suffix}.pt"

    train_losses, val_losses = [], []
    print("----------------训练开始---------------------")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader is not None else float("nan")
        dt = time.time() - t0

        train_losses.append(train_loss)
        if not (val_loader is None):
            val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} ({dt:.1f}s)")

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print(f"  ✓ 保存最优模型 -> {best_path}")

    # 绘制并保存曲线
    plot_and_save_losses(train_losses, val_losses, run_dir, suffix=date_suffix)



if __name__ == "__main__":
    main()
