from torch.utils.data import Dataset, DataLoader
from .transforms import DefaultTransform
from pathlib import Path
import numpy as np
import os

class _SingleNPZDataset(Dataset):
    """
    加载单个.npz文件作为地磁数据集
    暂时先不考虑加IMU的情况
    """
    def __init__(self, npz_path: str, transform=None):
        data = np.load(npz_path)
        self.transform = transform or DefaultTransform()
        self.X_MAG = data['X_mag']
        self.y =  data['y']

    def __len__(self):
        return self.X_MAG.shape[0]
    
    def __getitem__(self, idx: int):
        x_mag = self.X_MAG[idx]
        y = self.y[idx]
        sample = {"x_mag": x_mag, "y": y}
        if self.transform:
            sample = self.transform(sample)
        return sample


def create_single_npz_dataloader(
    file_path,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    transform=None,
):
    """
    加载单个.npz文件的dataloader，如果文件路径不存在返回None
    主要用于模型测试
    """
    def build_loader(file_path: str, shuffle: bool):
        path_ = Path(file_path)
        if not path_.exists():
            print(f"文件路径不存在，请检查: {file_path}")
            return None
        dataset = _SingleNPZDataset(file_path, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
    test_loader = build_loader(file_path, False)
    return test_loader