# train/test_knn_dtw.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.multi_session_dataset import MagneticDataSetV2, MAG_COLS
from network.knn_dtw import MagneticKNNDTWLocalizer
from train.baseline_utils import (
    compute_global_mag_stats_from_train_files,
    build_reference_database,
    evaluate_localizer,
    save_localization_results_csv,
    list_csv_files,
)


def main():
    # -----------------------------
    # 1. 路径配置
    # -----------------------------
    data_root = ROOT / "data" / "data_for_train_test_v14"/ "12.25-wenguan-resample-filter-v2"
    train_dir = data_root / "train"
    eval_dir = data_root / "test1"
    res_dir = ROOT / "runs" / "loc_res / different_wenguan"

    # -----------------------------
    # 2. 参数配置
    # -----------------------------
    seq_len = 32
    stride_train = 20
    stride_eval = 20
    normalize_x = True

    coarse_top_m = 50
    fine_top_k = 3
    weighted = True

    # -----------------------------
    # 3. 读取文件
    # -----------------------------
    train_files = list_csv_files(train_dir)
    eval_files = list_csv_files(eval_dir)

    # -----------------------------
    # 4. 统计训练集 x 的均值方差
    # KNN / DTW 建议只归一化 x_mag，不归一化位置标签
    # -----------------------------
    stats = None
    if normalize_x:
        stats = compute_global_mag_stats_from_train_files(train_files, MAG_COLS)

    # -----------------------------
    # 5. 构建数据集
    # y_norm_mode="none"：位置标签不归一化
    # -----------------------------
    train_set = MagneticDataSetV2(
        train_files,
        seq_len=seq_len,
        stride=stride_train,
        stats=stats,
        normalize_x=normalize_x,
        y_norm_mode="none",
        cache_in_memory=True,
    )

    eval_set = MagneticDataSetV2(
        eval_files,
        seq_len=seq_len,
        stride=stride_eval,
        stats=stats,
        normalize_x=normalize_x,
        y_norm_mode="none",
        cache_in_memory=True,
    )

    print(f"train files   : {len(train_files)}")
    print(f"eval files    : {len(eval_files)}")
    print(f"train samples : {len(train_set)}")
    print(f"eval samples  : {len(eval_set)}")

    # -----------------------------
    # 6. 构建参考库
    # -----------------------------
    ref_db = build_reference_database(train_set)
    print(f"reference db feats shape : {ref_db['feats'].shape}")
    print(f"reference db seqs shape  : {ref_db['seqs'].shape}")
    print(f"reference db coords shape: {ref_db['coords'].shape}")

    # -----------------------------
    # 7. 构建两阶段定位器
    # -----------------------------
    localizer = MagneticKNNDTWLocalizer(
        ref_feats=ref_db["feats"],
        ref_seqs=ref_db["seqs"],
        ref_coords=ref_db["coords"],
        coarse_top_m=coarse_top_m,
        fine_top_k=fine_top_k,
        weighted=weighted,
    )

    # -----------------------------
    # 8. 评估
    # -----------------------------
    print("\n[KNN + DTW Two-Stage Evaluation]")
    metrics, preds, gts, errors = evaluate_localizer(localizer, eval_set, verbose=True)

    # -----------------------------
    # 9. 保存 CSV
    # 与你当前深度模型测试结果格式一致
    # -----------------------------
    file_name = (
        f"knn_dtw_test1_32_loc_res_meanerr_{metrics['mean_l2']:.4f}.csv"
    )
    save_localization_results_csv(
        preds=preds,
        gts=gts,
        errors=errors,
        metrics=metrics,
        res_dir=res_dir,
        file_name=file_name,
    )


if __name__ == "__main__":
    main()