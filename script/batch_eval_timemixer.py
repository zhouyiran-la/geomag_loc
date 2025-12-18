import argparse
from pathlib import Path

from train.test_timemixer_enc import _eval_single_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate NPZ files with TimeMixer.")
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Directory containing .npz files to evaluate.",
    )
    parser.add_argument(
        "--best-path",
        required=True,
        help="Path to the trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--res-dir",
        required=True,
        help="Directory to store evaluation csv files.",
    )
    parser.add_argument(
        "--input-key",
        default="x_mag_grad",
        choices=("x_mag", "x_mag_grad"),
        help="Input feature key used by the dataset and model.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern to match files inside test-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = Path(args.test_dir)
    best_path = Path(args.best_path)
    res_dir = Path(args.res_dir)
    input_key = args.input_key

    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    npz_files = sorted(test_dir.glob(args.pattern))
    if not npz_files:
        print(f"No files matching pattern '{args.pattern}' in {test_dir}")
        return

    for npz_path in npz_files:
        print(f"开始评估文件: {npz_path}")
        _eval_single_npz(str(npz_path), best_path, res_dir, input_key=input_key)


if __name__ == "__main__":
    main()
