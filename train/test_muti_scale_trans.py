import os
import time
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from datasets.single_npz_dataset import create_single_npz_dataloader
from network.magnetic_localization_model_regress import MagneticLocalizationModel

def _eval_single_npz(npz_path: str, ckpt_path: Path, res_dir: Path, d_model: int = 64, scales=[10, 30, 50, 100, 300], nhead: int = 8, num_layers: int = 3,
                     batch_size: int = 64):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    loader = create_single_npz_dataloader(npz_path, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if loader is None:
        raise RuntimeError('loader为空，请检查文件路径是否存在！')

    model = MagneticLocalizationModel(input_dim=3, d_model=d_model, scales=scales,
                                      nhead=nhead, num_layers=num_layers, output_dim=2)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0
    
    # 创建列表来存储结果
    all_predictions = []
    all_labels = []
    all_errors = []
    
    with torch.no_grad():
        for batch in loader:
            x_mag = batch['x_mag'].to(device, non_blocking=True)
            print(x_mag.size())
            y = batch['y'].to(device, non_blocking=True)
            preds, _ = model(x_mag)
            loss = criterion(preds, y)
            total_loss += loss.item()
            total_samples += x_mag.size(0)
            
            # 将张量转换为numpy数组并计算误差
            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()
            
            # 计算欧几里得距离（误差）
            errors = np.sqrt(np.sum((preds_np - y_np) ** 2, axis=1))
            
            # 存储结果
            all_predictions.extend(preds_np)
            all_labels.extend(y_np)
            all_errors.extend(errors)
    
    mse = total_loss / max(total_samples, 1)
    print(f"Single file eval -> {npz_path} | samples={total_samples} | MSE={mse:.6f}")
    
    # 创建DataFrame并保存为CSV
    results_df = pd.DataFrame({
        'pred_x': [pred[0] for pred in all_predictions],
        'pred_y': [pred[1] for pred in all_predictions],
        'true_x': [label[0] for label in all_labels],
        'true_y': [label[1] for label in all_labels],
        'euclidean_error': all_errors
    })
    # 先创建输出目录
    res_dir.mkdir(parents=True, exist_ok=True)
    # 生成输出文件名
    file_name = Path(npz_path).stem + ".csv"
    output_csv = res_dir / file_name
    results_df.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")
    
    return mse, results_df


if __name__ == "__main__":
    best_path = Path("checkpoints") / "mag_localization_regress_best_scale_10_30_50_100_300.pt"
    npz_path = "./data/data_for_train_test_v1/seq_300/mag/test/dataset57.npz"
    res_dir = Path("runs") / "loc_res"
    # Load best and test
    if best_path.exists():
        _eval_single_npz(npz_path, best_path, res_dir)
