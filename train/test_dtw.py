# train/test_dtw.py
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from datasets.multi_session_dataset import MagneticDataSetV2
from datasets.multi_session_dataset import MagneticDataSetV2, MAG_COLS
from network.dtw import MagneticDTWLocalizer
from train.baseline_utils import (
    compute_global_mag_stats_from_train_files,
    build_reference_database,
    evaluate_localizer,
    list_csv_files,
)


def main():
    data_root = ROOT / "data" / "data_for_train_test_v14"
    train_dir = data_root / "train"
    eval_dir = data_root / "eval"

    seq_len = 128
    stride_train = 4   # DTW 很慢，建议训练库先稀疏一点
    stride_eval = 1
    normalize_x = True
    dtw_k = 3

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

    localizer = MagneticDTWLocalizer(
        ref_seqs=ref_db["seqs"],
        ref_coords=ref_db["coords"],
        k=dtw_k,
        weighted=True,
    )

    print("\n[DTW Evaluation]")
    evaluate_localizer(localizer, eval_set, verbose=True)


if __name__ == "__main__":
    main()