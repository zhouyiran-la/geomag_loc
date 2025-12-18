import argparse
import os
from pathlib import Path
import shutil
from typing import List, Tuple, Optional

SEQ_DIRS = ["seq_100", "seq_200", "seq_300"]
SUB_DIRS = ["mag", "mag_aug", "mag_imu"]
NPZ_EXT = ".npz"


def find_source_session_dirs(base_dir: Path) -> List[Path]:
    """查找 base_dir 下的会话子目录（排除以 seq_ 开头的新目标目录）。"""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    session_dirs = []
    for p in base_dir.iterdir():
        if p.is_dir() and not p.name.startswith("seq_"):
            session_dirs.append(p)
    return session_dirs


def ensure_target_dirs(target_dir: Path) -> None:
    """在 target_dir 下创建目标目录结构：seq_*/{mag,mag_aug,mag_imu}"""
    for seq in SEQ_DIRS:
        for sub in SUB_DIRS:
            target = target_dir / seq / sub
            target.mkdir(parents=True, exist_ok=True)


def iter_source_npz_files(session_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    遍历一个会话目录，找到所有 seq_*/{mag,mag_aug,mag_imu} 下的 .npz 文件。
    返回元组 (seq_dir_name, sub_dir_name, file_path)。
    """
    results: List[Tuple[Path, Path, Path]] = []
    # 兼容中间层级（如 50/100/xy用 等），使用 rglob
    for seq in SEQ_DIRS:
        for sub in SUB_DIRS:
            for f in session_dir.rglob(f"{seq}/{sub}/*{NPZ_EXT}"):
                if f.is_file():
                    results.append((Path(seq), Path(sub), f))
    return results


def move_files(base_dir: Path, target_dir: Optional[Path] = None, dry_run: bool = False) -> None:
    """
    合并所有会话目录下的 .npz 文件到目标目录，按 dataset01.npz, dataset02.npz ... 顺序命名
    注意：使用复制操作而非移动，源文件保留不变
    
    Args:
        base_dir: 源数据根目录，用于查找会话子目录
        target_dir: 目标目录，用于创建 seq_* 目录结构。如果为 None，则使用 base_dir
        dry_run: 是否仅预览，不实际复制文件
    """
    if target_dir is None:
        target_dir = base_dir
    else:
        target_dir = Path(target_dir).resolve()
    
    ensure_target_dirs(target_dir)
    print(f"[DEBUG] 源目录: {base_dir}")
    print(f"[DEBUG] 目标目录: {target_dir}")
    
    session_dirs = find_source_session_dirs(base_dir)
    print(f"[DEBUG] 找到 {len(session_dirs)} 个会话目录:")
    for sd in session_dirs:
        print(f"  - {sd}")
    
    if not session_dirs:
        print(f"未发现会话目录，已存在的仅为目标 seq_* 目录？ base_dir={base_dir}")
        return

    # 按目标目录分组收集所有文件
    target_files = {}  # {(seq_name, sub_name): [file_paths]}
    for session in session_dirs:
        print(f"[DEBUG] 搜索会话目录: {session}")
        files = iter_source_npz_files(session)
        print(f"[DEBUG]   找到 {len(files)} 个 .npz 文件")
        for seq_name, sub_name, fpath in files:
            key = (seq_name, sub_name)
            if key not in target_files:
                target_files[key] = []
            target_files[key].append(fpath)
    
    print(f"[DEBUG] 按目标目录分组后:")
    for (seq_name, sub_name), file_paths in sorted(target_files.items()):
        print(f"  {seq_name}/{sub_name}: {len(file_paths)} 个文件")

    # 对每个目标目录，按顺序重命名为 dataset01.npz, dataset02.npz ...
    total = 0
    for (seq_name, sub_name), file_paths in sorted(target_files.items()):
        dest_dir = target_dir / seq_name / sub_name
        # 计算已有文件数量，从下一个编号开始
        existing_count = len(list(dest_dir.glob("dataset*.npz")))
        counter = existing_count + 1
        
        for fpath in sorted(file_paths):  # 按文件路径排序，保证顺序一致
            dst = dest_dir / f"dataset{counter:02d}{NPZ_EXT}"
            print(f"COPY: {fpath} -> {dst}")
            if not dry_run:
                shutil.copy2(str(fpath), str(dst))  # 使用 copy2 保留文件元数据
            counter += 1
            total += 1

    print(f"完成：共处理 {total} 个文件。dry_run={dry_run}")


def main():
    parser = argparse.ArgumentParser(description="合并 preprocessed 下各会话目录中的 .npz 至根级 seq_*/ 子目录（复制操作）")
    parser.add_argument("--base_dir", type=str, default=str(Path("data") / "preprocessed"),
                        help="源数据根目录，用于查找会话子目录（默认 data/preprocessed）")
    parser.add_argument("--target_dir", type=str, default=None,
                        help="目标目录，用于创建 seq_* 目录结构。如果未指定，则使用 base_dir")
    parser.add_argument("--dry_run", action="store_true", help="仅打印计划复制，不执行")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    target_dir = Path(args.target_dir).resolve() if args.target_dir else None
    move_files(base_dir=base_dir, target_dir=target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
