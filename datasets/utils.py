import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets.transforms import DefaultTransform, MagneticGradientTransform, ComposeTransform


def load_all_npz_files(data_dir, pattern=".npz", use_imu=True):
    """
    批量加载指定目录下的 .npz 文件，并拼接为统一数组。
    - 支持传入 '.npz'（自动补全为 '*.npz'）或完整通配（如 '*.npz' / '**/*.npz'）。
    - 若未找到文件，抛出明确异常以便排查路径/模式问题。
    """
    X_MAG_list, X_IMU_list, y_list = [], [], []
    path = Path(data_dir)

    # 规范化 pattern（兼容传入 '.npz' 场景）
    patt = pattern
    if patt.startswith('.') and not patt.startswith('*.'):
        patt = f"*{patt}"

    # 查找文件并排序（保证加载顺序稳定）
    files = sorted(path.glob(patt))

    # 如果使用递归匹配（包含 "**/"），改用 rglob
    if not files and ('**/' in patt or patt.startswith('**')):
        files = sorted(path.rglob(patt.replace('**/', '')))

    if not files:
        raise FileNotFoundError(
            f"未在目录 {path} 下按模式 '{pattern}' 匹配到任何文件。\n"
            f"请检查: 1) data_dir 是否正确 2) pattern 是否应为 '*.npz' 或 '**/*.npz' 3) 运行时工作目录。"
        )

    for file in files:
        print(f"正在加载 {str(file)} 文件")
        data = np.load(file)
        X_mag = data["X_mag"]
        print(f"{str(file.name)} - X_mag.shape = {X_mag.shape}")
        X_MAG_list.append(X_mag)

        y = data["y"]
        print(f"{str(file.name)} - y.shape = {y.shape}")
        y_list.append(y)

        if use_imu and "X_imu" in data:
            X_IMU_list.append(data["X_imu"])

    if len(X_MAG_list) == 0 or len(y_list) == 0:
        raise RuntimeError("匹配到文件但未成功读取到 X_mag 或 y 数据，请检查 .npz 内部键名是否为 'X_mag' 与 'y'")

    X_MAG = np.concatenate(X_MAG_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if use_imu and len(X_IMU_list) > 0:
        X_IMU = np.concatenate(X_IMU_list, axis=0)
    else:
        X_IMU = None

    return X_MAG, X_IMU, y


def build_transform(input_key):
    if input_key == "x_mag_grad":
        return ComposeTransform([
            MagneticGradientTransform(),
            DefaultTransform(),
        ])
    else:
        return DefaultTransform()
    

def compute_train_stats_from_csv_files(file_paths, mag_cols, pos_cols, eps=1e-6):
    """
    只用 train 文件计算统计量：x_mean/x_std/y_mean/y_std/y_min/y_max
    """
    n = 0
    sum_x = None
    sum_x2 = None
    y_list = []

    usecols = list(mag_cols) + list(pos_cols)

    for p in file_paths:
        df = pd.read_csv(p, usecols=usecols)
        x = df[mag_cols].to_numpy(dtype=np.float32)  # (T,C)
        y = df[pos_cols].to_numpy(dtype=np.float32)  # (T,2)

        if sum_x is None:
            sum_x = x.sum(axis=0)
            sum_x2 = (x * x).sum(axis=0)
        else:
            sum_x += x.sum(axis=0)
            sum_x2 += (x * x).sum(axis=0)

        n += x.shape[0]
        y_list.append(y)

    if n <= 0:
        raise RuntimeError("train files empty: cannot compute stats")

    x_mean = (sum_x / n).astype(np.float32) # type: ignore
    x_var = (sum_x2 / n - x_mean * x_mean).astype(np.float32) # type: ignore
    x_std = np.sqrt(np.maximum(x_var, 0.0)).astype(np.float32)
    x_std = np.maximum(x_std, eps)

    Y = np.concatenate(y_list, axis=0)
    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": Y.mean(axis=0).astype(np.float32),
        "y_std": np.maximum(Y.std(axis=0).astype(np.float32), eps),
        "y_min": Y.min(axis=0).astype(np.float32),
        "y_max": Y.max(axis=0).astype(np.float32),
    }
    return stats


def norm_y(y_norm_mode:str, y_true, pf=None, stats=None):
    """
    Docstring for norm_y
    Args:
        y_norm_mode: str 标准化模式 global_zscore global_minmax per_file_minmax
        y_true: nd.array (2,)
        pf: dict {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max} 
        stats: global*模式下 训练文件的统计量
    Return：
        y_train
        y_stats
    """
    if y_norm_mode == "none":
        y_train = y_true
        y_stats = None
        return y_train, y_stats
    elif y_norm_mode == "global_zscore":
        assert stats, "stats不能为None"
        y_train = (y_true - stats["y_mean"]) / (stats["y_std"] + 1e-6)
        y_stats = np.array([*stats["y_mean"], *stats["y_std"]], dtype=np.float32)  # optional
        return y_train, y_stats
    elif y_norm_mode == "global_minmax":
        assert stats, "stats不能为None"
        y_train = (y_true - stats["y_min"]) / (stats["y_max"] - stats["y_min"] + 1e-6)
        y_stats = np.array([*stats["y_min"], *stats["y_max"]], dtype=np.float32)
        return y_train, y_stats
    elif y_norm_mode == "per_file_minmax":
        assert pf, "每个文件统计量pf不能为None"
        # 用 per-file stats 做 minmax
        y_denx = (pf["x_max"] - pf["x_min"]) if (pf["x_max"] > pf["x_min"]) else 1.0 # type: ignore
        y_deny = (pf["y_max"] - pf["y_min"]) if (pf["y_max"] > pf["y_min"]) else 1.0
        y_train = np.array([
            (y_true[0] - pf["x_min"]) / y_denx,
            (y_true[1] - pf["y_min"]) / y_deny
        ], dtype=np.float32)
        y_stats = np.array([pf["x_min"], pf["x_max"], pf["y_min"], pf["y_max"]], dtype=np.float32)
        return y_train, y_stats

    raise ValueError(f"Unknown y_norm_mode={y_norm_mode}")
   

def denorm_y(preds_norm, batch, y_norm_mode, stats=None, device=None):
    """
    Args:
        preds_norm: (B,2) torch
        batch: dict from dataloader, include y_raw and y_stats
        y_norm_mode: same string as dataset
        stats: train stats dict when using global_* modes
    Return: 
        preds_real: (B,2) torch in real coordinate system
    """
    if device is None:
        device = preds_norm.device

    if y_norm_mode == "none":
        return preds_norm

    if y_norm_mode == "per_file_minmax":
        assert "y_stats" in batch and batch["y_stats"] is not None
        # batch["y_stats"]: (B,4) = [x_min, x_max, y_min, y_max]
        s = batch["y_stats"].to(device).float()
        x_min, x_max, y_min, y_max = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        pred_x = preds_norm[:, 0] * (x_max - x_min) + x_min
        pred_y = preds_norm[:, 1] * (y_max - y_min) + y_min
        return torch.stack([pred_x, pred_y], dim=1)

    if y_norm_mode == "global_zscore":
        assert stats is not None, "global_zscore 需要 train stats"
        y_mean = torch.tensor(stats["y_mean"], device=device, dtype=torch.float32).view(1, 2)
        y_std  = torch.tensor(stats["y_std"],  device=device, dtype=torch.float32).view(1, 2)
        return preds_norm * (y_std + 1e-6) + y_mean

    if y_norm_mode == "global_minmax":
        assert stats is not None, "global_minmax 需要 train stats"
        y_min = torch.tensor(stats["y_min"], device=device, dtype=torch.float32).view(1, 2)
        y_max = torch.tensor(stats["y_max"], device=device, dtype=torch.float32).view(1, 2)
        return preds_norm * (y_max - y_min + 1e-6) + y_min

    raise ValueError(f"Unknown y_norm_mode: {y_norm_mode}")
    
