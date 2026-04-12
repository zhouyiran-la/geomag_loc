# train/test_knn.py
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from datasets.multi_session_dataset import MagneticDataSetV2, MAG_COLS
from network.knn import MagneticKNNLocalizer
from train.baseline_utils import (
    compute_global_mag_stats_from_train_files,
    build_reference_database,
    evaluate_localizer,
    list_csv_files,
    save_localization_results_csv
)


def main():
    data_root = ROOT / "data" / "data_for_train_test_v14"/ "12.25-wenguan-resample-filter-v2"
    train_dir = data_root / "train"
    eval_dir = data_root / "test1"
    res_dir = ROOT / "runs" / "loc_res" / "different_method_wenguan"

    seq_len = 64
    stride_train = 20
    stride_eval = 20
    normalize_x = True
    knn_k = 3

    train_files = list_csv_files(train_dir)
    eval_files = list_csv_files(eval_dir)

    stats = None
    if normalize_x:
        stats = compute_global_mag_stats_from_train_files(train_files, MAG_COLS)

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

    print(f"train samples: {len(train_set)}")
    print(f"eval samples : {len(eval_set)}")

    ref_db = build_reference_database(train_set)

    localizer = MagneticKNNLocalizer(
        ref_feats=ref_db["feats"],
        ref_coords=ref_db["coords"],
        k=knn_k,
        weighted=False,
    )

    print("\n[KNN Evaluation]")
    metrics, preds, gts, errors = evaluate_localizer(localizer, eval_set, verbose=True)

    file_name = f"knn_test1_loc_res_meanerr_48_{metrics['mean_l2']:.4f}.csv"
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