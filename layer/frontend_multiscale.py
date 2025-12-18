import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDecomposition(nn.Module):
    """
    简化版 AMD：对磁信号做多尺度分解
    - 低频 trend（大窗口平滑）
    - 中频 season（中窗口）
    - 高频 residual（原信号减去低频）
    """

    def __init__(self, input_dim=3, k_trend=25, k_season=7):
        super().__init__()
        self.trend_conv = nn.Conv1d(input_dim, input_dim, kernel_size=k_trend,
                                    padding=k_trend//2, groups=input_dim, bias=False)
        self.season_conv = nn.Conv1d(input_dim, input_dim, kernel_size=k_season,
                                     padding=k_season//2, groups=input_dim, bias=False)
        nn.init.constant_(self.trend_conv.weight, 1.0/k_trend)
        nn.init.constant_(self.season_conv.weight, 1.0/k_season)

    def forward(self, x):
        """
        x : (B, T, C)
        return: trend, season, residual
        """
        x_t = x.transpose(1, 2)  # (B, C, T)

        trend = self.trend_conv(x_t).transpose(1, 2)
        season = self.season_conv(x_t).transpose(1, 2)
        residual = x - trend

        return trend, season, residual
