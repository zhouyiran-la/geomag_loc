import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1️⃣ Default Transform
# ============================================================

class DefaultTransform:
    """仅进行 numpy → tensor 转换，并保证shape为[N, 3]"""
    def __call__(self, sample):
        sample["x_mag"] = torch.tensor(sample["x_mag"], dtype=torch.float32)
        sample["y"] = torch.tensor(sample["y"], dtype=torch.float32)

        if "x_imu" in sample:
            sample["x_imu"] = torch.tensor(sample["x_imu"], dtype=torch.float32)
        if "y_raw" in sample:
            sample["y_raw"] = torch.tensor(sample["y_raw"], dtype=torch.float32)
        if "fid" in sample:
            sample["fid"] = torch.tensor(sample["fid"], dtype=torch.int32)
        if "y_stats" in sample:
            sample["y_stats"] = torch.tensor(sample["y_stats"], dtype=torch.float32)
        return sample


# ============================================================
# 2️⃣ 论文特征增强（FeatureAugmentTransform）
# ============================================================

class FeatureAugmentTransform:
    """
    二次滑窗 + 邻域特征增强 Transform
    输入: x_mag (W₁, 3)
    输出: x_mag_aug (W₁−W₂+1, W₂, 3×W₂)
    """

    def __init__(self, W2: int = 5):
        self.W2 = W2

    def __call__(self, sample):
        x_mag = sample["x_mag"]
        if isinstance(x_mag, torch.Tensor):
            x_mag = x_mag.numpy()

        W1, C = x_mag.shape
        W2 = self.W2
        assert C == 3, f"地磁输入应为三维 (m_s, m_h, m_v)，但得到 {C} 维"
        assert W2 <= W1, f"W2 必须小于等于 W1 (当前: W2={W2}, W1={W1})"

        local_features = []
        for start in range(W1 - W2 + 1):
            window = x_mag[start:start+W2]
            neighborhood = []
            for j in range(W2):
                rotated = np.roll(window, -j, axis=0).reshape(-1)
                neighborhood.append(rotated)
            neighborhood = np.stack(neighborhood, axis=0)
            local_features.append(neighborhood)

        x_mag_aug = np.stack(local_features, axis=0)
        sample["x_mag_aug"] = torch.from_numpy(x_mag_aug).float()
        return sample


# ============================================================
# 3️⃣ 地磁梯度特征增强
# ============================================================

class MagneticGradientTransform:
    """
    添加地磁梯度与二阶差分特征：
    输入: x_mag (W₁, 3)
    输出: x_mag_grad (W₁−2, 9)
    """
    def __call__(self, sample):
        x_mag = sample["x_mag"]
        if isinstance(x_mag, torch.Tensor):
            x_mag = x_mag.numpy()

        delta = np.diff(x_mag, axis=0)
        delta2 = np.diff(delta, axis=0)
        x_base = x_mag[2:]
        x_aug = np.concatenate([x_base, delta[1:], delta2], axis=-1)
        sample["x_mag_grad"] = torch.tensor(x_aug, dtype=torch.float32)
        return sample


# ============================================================
# 4️⃣ 频谱特征增强
# ============================================================

class MagneticSpectralTransform:
    """
    计算地磁信号的短时傅里叶谱特征（STFT）
    输出: x_mag_spectral
    """
    def __init__(self, n_fft=64, hop_length=16):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sample):
        x_mag = sample["x_mag"]
        if isinstance(x_mag, np.ndarray):
            x_mag = torch.tensor(x_mag, dtype=torch.float32)
        x_mag = x_mag.transpose(0, 1)  # (3, W₁)

        specs = []
        for i in range(3):
            spec = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft, hop_length=self.hop_length
            )(x_mag[i])
            spec = torch.log1p(spec)
            spec_mean = torch.mean(spec, dim=0)
            spec_max = torch.max(spec, dim=0).values
            spec_vec = torch.cat([spec_mean, spec_max], dim=0)
            specs.append(spec_vec)
        sample["x_mag_spectral"] = torch.stack(specs, dim=0).T  # (time, feature)
        return sample


# ============================================================
# 5️⃣ 组合多种 Transform
# ============================================================

class ComposeTransform:
    """顺序执行多个 transform"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

