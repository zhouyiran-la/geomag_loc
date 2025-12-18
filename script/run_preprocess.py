import argparse
import subprocess
import sys
from pathlib import Path


DATA_JOBS = [
    ("data/origin/4.19数据/50/norm", "data/preprocessed/4.19数据/50"),
    ("data/origin/4.19数据/100/norm", "data/preprocessed/4.19数据/100"),
    ("data/origin/4.25数据/50/norm", "data/preprocessed/4.25数据/50"),
    ("data/origin/4.25数据/100/norm", "data/preprocessed/4.25数据/100"),
    ("data/origin/4.26数据/50/norm", "data/preprocessed/4.26数据/50"),
    ("data/origin/4.26数据/100/norm", "data/preprocessed/4.26数据/100"),
    ("data/origin/4.26数据/xy用/norm", "data/preprocessed/4.26数据/xy用"),
    ("data/origin/4.28数据/50/norm", "data/preprocessed/4.28数据/50"),
    ("data/origin/4.28数据/100/norm", "data/preprocessed/4.28数据/100"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run sliding-window preprocessing.")
    parser.add_argument("win", type=int, help="Window size (--win).")
    parser.add_argument("stride", type=int, help="Stride (--stride).")
    parser.add_argument("mode", type=int, help="Mode (--mode).")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    script = root / "preprocess" / "get_seq_by_slide_window.py"

    for src, dst in DATA_JOBS:
        src_path = root / src
        dst_path = root / dst
        print("-" * 50)
        print(f"Input : {src_path}")
        print(f"Output: {dst_path}")
        cmd = [
            args.python,
            str(script),
            str(src_path),
            str(dst_path),
            "--win",
            str(args.win),
            "--stride",
            str(args.stride),
            "--mode",
            str(args.mode),
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("-" * 50)
    print("All preprocessing jobs completed.")


if __name__ == "__main__":
    main()
