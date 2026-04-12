# train/baseline_utils.py
from pathlib import Path
import csv
import math
import numpy as np
import pandas as pd


def compute_global_mag_stats_from_train_files(files, mag_cols):
    xs = []
    for p in files:
        df = pd.read_csv(p)
        xs.append(df[mag_cols].to_numpy(dtype=np.float32))
    x_all = np.concatenate(xs, axis=0)

    return {
        "x_mean": x_all.mean(axis=0).astype(np.float32),
        "x_std": (x_all.std(axis=0) + 1e-6).astype(np.float32),
    }


def build_reference_database(dataset):
    """
    从 MagneticDataSetV2 构建检索数据库
    """
    seq_list = []
    feat_list = []
    coord_list = []
    fid_list = []

    for i in range(len(dataset)):
        sample = dataset[i]
        x_mag = np.asarray(sample["x_mag"], dtype=np.float32)   # (L, 3)
        y_raw = np.asarray(sample["y_raw"], dtype=np.float32)   # (2,)
        fid = int(sample["fid"])

        seq_list.append(x_mag)
        feat_list.append(x_mag.reshape(-1))
        coord_list.append(y_raw)
        fid_list.append(fid)

    return {
        "seqs": np.stack(seq_list, axis=0),      # (N, L, 3)
        "feats": np.stack(feat_list, axis=0),    # (N, L*3)
        "coords": np.stack(coord_list, axis=0),  # (N, 2)
        "fids": np.asarray(fid_list, dtype=np.int64),
    }


def list_csv_files(data_dir, pattern="*.csv"):
    files = sorted([str(p) for p in Path(data_dir).glob(pattern)])
    if len(files) == 0:
        raise FileNotFoundError(f"{data_dir} 下未找到 {pattern}")
    return files


def evaluate_localizer(localizer, dataset, verbose=True):
    """
    对 KNN / DTW / KNN+DTW 这类检索式定位器进行评估
    返回:
        metrics, preds, gts, errors
    """
    preds = []
    gts = []

    for i in range(len(dataset)):
        sample = dataset[i]
        x = np.asarray(sample["x_mag"], dtype=np.float32)
        y = np.asarray(sample["y_raw"], dtype=np.float32)

        pred = localizer.predict_one(x)
        preds.append(pred)
        gts.append(y)

    preds = np.stack(preds, axis=0)   # (N,2)
    gts = np.stack(gts, axis=0)       # (N,2)

    metrics, errors = compute_localization_metrics(preds, gts)

    if verbose:
        print(
            f"val_loss={metrics['val_loss']:.6f} | "
            f"mean_l1={metrics['mean_l1']:.3f} "
            f"mean_l2={metrics['mean_l2']:.3f} "
            f"rmse_x={metrics['rmse_x']:.3f} "
            f"rmse_y={metrics['rmse_y']:.3f} "
            f"rmse_2d={metrics['rmse_2d']:.3f}"
        )

    return metrics, preds, gts, errors


def compute_localization_metrics(preds: np.ndarray, gts: np.ndarray):
    """
    按你的深度模型 test() 中的指标定义来计算
    """
    preds = np.asarray(preds, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.float32)

    if preds.shape != gts.shape:
        raise ValueError(f"preds.shape={preds.shape} 与 gts.shape={gts.shape} 不一致")
    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError("preds/gts 期望形状为 (N, 2)")

    diff = preds - gts
    dx = diff[:, 0]
    dy = diff[:, 1]

    l1 = np.abs(dx) + np.abs(dy)
    l2 = np.sqrt(dx ** 2 + dy ** 2)

    num_samples = len(preds)
    denom = max(num_samples, 1)

    mean_l1 = float(np.sum(l1) / denom)
    mean_l2 = float(np.sum(l2) / denom)

    mse_x = float(np.sum(dx ** 2) / denom)
    mse_y = float(np.sum(dy ** 2) / denom)

    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)

    metrics = {
        # baseline 没有监督 loss，这里统一写成 0.0，方便和你现有 CSV 格式兼容
        "val_loss": 0.0,
        "mean_l1": mean_l1,
        "mean_l2": mean_l2,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_2d": rmse_2d,
        "num_samples": int(num_samples),
    }

    return metrics, l2.astype(np.float32)


def save_localization_results_csv(
    preds: np.ndarray,
    gts: np.ndarray,
    errors: np.ndarray,
    metrics: dict,
    res_dir,
    file_name: str,
):
    """
    保存成和深度模型 test() 一样格式的 CSV
    """
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    output_csv = res_dir / file_name

    preds = np.asarray(preds, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.float32)
    errors = np.asarray(errors, dtype=np.float32)

    results_df = pd.DataFrame(
        {
            "pred_x": preds[:, 0],
            "pred_y": preds[:, 1],
            "true_x": gts[:, 0],
            "true_y": gts[:, 1],
            "euclidean_error": errors,
        }
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
        w.writerow([])
        results_df.to_csv(f, index=False)

    print(f"结果已保存到: {output_csv}")
    return output_csv