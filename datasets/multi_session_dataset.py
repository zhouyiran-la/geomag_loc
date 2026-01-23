import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .transforms import DefaultTransform
from .utils import load_all_npz_files, compute_train_stats_from_csv_files, norm_y

MAG_COLS = ["geomagneticx", "geomagneticy", "geomagneticz"]
POS_COLS = ["pos_x", "pos_y"]

class MagneticDatasetV1(Dataset):
    """
    支持加载多个 .npz 文件的地磁/地磁-IMU 数据集
    """

    def __init__(self, data_dir, pattern=".npz", use_imu=True, transform=None):
        """
        Args:
            data_dir: str, 存放 .npz 文件的目录
            use_imu: bool, 是否包含 IMU 数据
            transform: callable, 对每个样本执行的预处理
        """
        self.data_dir = data_dir
        self.pattern = pattern;
        self.use_imu = use_imu
        self.transform = transform or DefaultTransform()

        # === 加载所有 npz 文件 ===
        self.X_MAG, self.X_IMU, self.y = load_all_npz_files(data_dir, pattern, use_imu=use_imu)
        
        print(f"✅ 已加载 {len(self.X_MAG)} 个样本（来自 {data_dir}）")

    def __len__(self):
        return len(self.X_MAG)

    def __getitem__(self, idx):
        x_mag = self.X_MAG[idx]
        y = self.y[idx]
            
        if self.use_imu and self.X_IMU is not None:
            x_imu = self.X_IMU[idx]
            sample = {"x_mag": x_mag, "x_imu": x_imu, "y": y}
        # 包含use_imu为true但是X_IMU为None的情况
        else:
            sample = {"x_mag": x_mag, "y": y}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class MagneticDataSetV2(Dataset):
    def __init__(
        self,
        file_paths,
        *,
        seq_len=128,
        stride=1,
        stats=None,                 # train stats: x_mean/x_std/y_mean/y_std/y_min/y_max
        normalize_x=False,          # 全局x zscore
        y_norm_mode="per_file_minmax",  # "none"|"global_zscore"|"global_minmax"|"per_file_minmax"
        transform=None,
        cache_in_memory=True,
    ):
        self.file_paths = [str(p) for p in file_paths]
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.stats = stats
        self.normalize_x = bool(normalize_x)
        self.y_norm_mode = y_norm_mode
        self.transform = transform
        self.cache_in_memory = bool(cache_in_memory)

        if self.normalize_x and self.stats is None:
            raise ValueError("normalize_x=True 需要 train stats (x_mean/x_std)")

        if self.y_norm_mode in ("global_zscore", "global_minmax") and self.stats is None:
            raise ValueError("global y normalization 需要 train stats (y_*)")

        self._cache = {}
        self.index = []
        self.lengths = []

        for fid, p in enumerate(self.file_paths):
            T = self._peek_length(p)
            self.lengths.append(T)
            if T >= self.seq_len:
                for s in range(0, T - self.seq_len + 1, self.stride):
                    self.index.append((fid, s))

    def _peek_length(self, path):
        df = pd.read_csv(path, usecols=[MAG_COLS[0]])
        return len(df)

    def _pos_minmax_stats(self, pos):
        x = pos[:, 0].astype(np.float32)
        y = pos[:, 1].astype(np.float32)
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def _pos_minmax_norm_with_stats(self, pos, st):
        x = pos[:, 0].astype(np.float32)
        y = pos[:, 1].astype(np.float32)
        x_den = (st["x_max"] - st["x_min"]) if (st["x_max"] > st["x_min"]) else 1.0
        y_den = (st["y_max"] - st["y_min"]) if (st["y_max"] > st["y_min"]) else 1.0
        x_norm = (x - st["x_min"]) / x_den
        y_norm = (y - st["y_min"]) / y_den
        return np.stack([x_norm, y_norm], axis=1).astype(np.float32)

    def _load_one_file(self, fid):
        if self.cache_in_memory and fid in self._cache:
            return self._cache[fid]

        p = self.file_paths[fid]
        df = pd.read_csv(p)
        print(f"已读取{p}，length={len(df)}")
        x = df[MAG_COLS].to_numpy(dtype=np.float32)      # (T,3)
        y_raw = df[POS_COLS].to_numpy(dtype=np.float32)  # (T,2)

        # per-file stats（即使不用也存着，方便debug/可视化）
        y_pf_stats = self._pos_minmax_stats(y_raw)

        data = {
            "x": x,
            "y_raw": y_raw,
            "y_pf_stats": y_pf_stats,
        }

        if self.cache_in_memory:
            self._cache[fid] = data
        return data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fid, s = self.index[idx]
        data = self._load_one_file(fid)

        e = s + self.seq_len
        x_win = data["x"][s:e]                # (L,3)
        y_true = data["y_raw"][e - 1]         # (2,)
        pf = data["y_pf_stats"]               # dict
        
        # x: global zscore
        if self.normalize_x:
            assert self.stats, "stats不能为None"
            mu = self.stats["x_mean"][None, :]
            sd = self.stats["x_std"][None, :]
            x_win = (x_win - mu) / (sd + 1e-6)

        y_train, y_stats = norm_y(self.y_norm_mode, y_true, pf, self.stats)

        sample = {
            "x_mag": x_win,
            "y": y_train,           # 训练loss用
            "y_raw": y_true,        # 真实坐标评估用
            "fid": fid,
            "y_stats": y_stats,     # per-file 逆归一化用（global模式可不依赖它）
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_magnetic_dataset_v1_dataloaders(
    data_dir,
    batch_size=64,
    use_imu=True,
    pattern=".npz",
    num_workers=0,
    shuffle_train=True,
    pin_memory=False,
    transform=None,
):
    """按目录创建 train/eval/test DataLoader。
    data_dir 目录结构需包含：train/、eval/、test/（若不存在则跳过）。
    返回: (train_loader, val_loader, test_loader) —— 不存在则为 None。
    """
    def build_loader(split_name: str, shuffle: bool):
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.isdir(split_dir):
            print(f"子目录不存在，跳过: {split_dir}")
            return None
        dataset = MagneticDatasetV1(split_dir, use_imu=use_imu, pattern=pattern, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    train_loader = build_loader("train", shuffle_train)
    val_loader = build_loader("eval", False)

    return train_loader, val_loader

def create_magnetic_dataset_v2_dataloaders(
    data_dir,
    batch_size=64,
    pattern=".csv",
    num_workers=0,
    shuffle_train=True,
    pin_memory=False,
    transform=None,
    stats=None,
    *,
    seq_len=128,
    stride=1,
    normalize_x=False,
    y_norm_mode="per_file_minmax",  # "none"|"global_zscore"|"global_minmax"|"per_file_minmax"
    cache_in_memory=True,
):
    """
    按目录创建 train/eval(/test) DataLoader.
    目录结构：data_dir/train, data_dir/eval, (可选)data_dir/test

    返回: (train_loader, val_loader) —— 不存在则为 None
    """

    def list_split_files(file_dir: str):
        if not os.path.isdir(file_dir):
            print(f"子目录不存在，跳过: {file_dir}")
            return None
        files = sorted([str(p) for p in Path(file_dir).glob(f"*{pattern}")])
        if len(files) == 0:
            print(f"目录下无匹配文件，跳过: {file_dir}")
            return None
        return files

    files = list_split_files(data_dir)

    if files is None:
        return None

    # 是否需要 stats：normalize_x=True 或 y_norm_mode 为 global
    need_cal_stats = bool(normalize_x) or (y_norm_mode in ("global_zscore", "global_minmax"))
    
    if need_cal_stats:
        stats = compute_train_stats_from_csv_files(files, MAG_COLS, POS_COLS)

    def build_loader(files, shuffle: bool):
        if files is None:
            return None
        dataset = MagneticDataSetV2(
            files,
            seq_len=seq_len,
            stride=stride,
            stats=stats,
            normalize_x=normalize_x,
            y_norm_mode=y_norm_mode,
            transform=transform,
            cache_in_memory=cache_in_memory,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    loader = build_loader(files, shuffle_train)

    return loader


