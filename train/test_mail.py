import csv
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from datasets import MagneticDataSetV2, create_magnetic_dataset_v2_dataloaders
from datasets.utils import build_transform, denorm_y
from network.losses import WeightedSmoothL1
from network.mail import MAIL


def build_mail_regressor_if_needed(model, loader, device, input_key="x_mag"):
    """
    由于 MAIL 的 regressor 是在第一次 forward 时动态创建的，
    所以在 load_state_dict 前，先用一个 batch 做一次前向传播来完成构建。
    """
    if len(model.regressor) > 0:
        return

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            _ = model(x)
            break

    if len(model.regressor) == 0:
        raise RuntimeError("MAIL regressor 构建失败，请检查 DataLoader 或模型 forward 逻辑。")


def test(
    model,
    loader,
    criterion,
    device,
    ckpt_path: Path,
    res_dir: Path,
    input_key="x_mag",
    *,
    y_norm_mode="per_file_minmax",
    stats=None,
):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 先用一个 batch 触发 MAIL 动态构建 regressor
    build_mail_regressor_if_needed(model, loader, device, input_key=input_key)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    sum_l2 = 0.0
    sum_l1 = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0

    all_preds = []
    all_labels = []
    all_errors = []

    with torch.no_grad():
        for batch in loader or []:
            x = batch[input_key].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()

            preds = model(x)
            loss = criterion(preds, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            preds_real = denorm_y(
                preds,
                batch,
                y_norm_mode=y_norm_mode,
                stats=stats,
                device=device,
            )
            y_real = batch["y_raw"].to(device, non_blocking=True).float()

            diff = preds_real - y_real
            l2 = torch.norm(diff, dim=1)
            l1 = diff.abs().sum(dim=1)

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

    denom = max(total_samples, 1)
    val_loss = total_loss / denom
    mean_l2 = sum_l2 / denom
    mean_l1 = sum_l1 / denom
    mse_x = sum_dx2 / denom
    mse_y = sum_dy2 / denom
    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)

    print(
        f"val_loss={val_loss:.6f} | "
        f"mean_l1={mean_l1:.3f} mean_l2={mean_l2:.3f} "
        f"rmse_x={rmse_x:.3f} rmse_y={rmse_y:.3f} rmse_2d={rmse_2d:.3f}"
    )

    res_dir.mkdir(parents=True, exist_ok=True)

    ckpt_stem = ckpt_path.stem
    file_name = f"MAIL_test1_loc_res_meanerr_{mean_l2:.4f}.csv"
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

    metrics = {
        "val_loss": val_loss,
        "mean_l1": mean_l1,
        "mean_l2": mean_l2,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_2d": rmse_2d,
        "num_samples": total_samples,
    }

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
        w.writerow([])
        results_df.to_csv(f, index=False)

    print(f"结果已保存到: {output_csv}")


if __name__ == "__main__":
    # test_dir = Path("data") / "data_for_train_test_v14" / "12.25-xinxi-resample-zscore" / "test1"

    test_dir = Path("data") / "data_for_train_test_v14" / "12.25-wenguan-resample-filter-v2" / "test1"
    ckpt_path = Path("checkpoints") / "mail" / "mag_localization_mail_best_20260401_1319_rmse_2d_4.473_wenguan.pt"
    res_dir = Path("runs") / "loc_res" / "different_method_wenguan"
    # res_dir = Path("runs") / "loc_res" / "different_method_xinxi"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    input_key = "x_mag"
    feature_transform = build_transform(input_key=input_key)

    criterion = WeightedSmoothL1(beta=0.05, w_x=1.0, w_y=1.3).to(device)

    model = MAIL(
        input_dim=3,
        seq_len=128,
        scale_lengths=(64, 128),
        # scale_lengths=(32, 64, 128),
        gru_hidden=128,
        proj_dim=64,
        attn_hidden=128,
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
        stride=20,
    )

    test(
        model,
        test_loader,
        criterion,
        device,
        ckpt_path,
        res_dir,
        input_key=input_key,
        y_norm_mode="per_file_minmax",
        stats=cast(MagneticDataSetV2, test_loader.dataset).stats, # type: ignore
    )