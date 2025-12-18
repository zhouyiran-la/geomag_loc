import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusionV2(nn.Module):
    """
    Pathformer 风格的多尺度融合：
    - 每尺度一个 gate
    - 加权融合
    """

    def __init__(self, feature_dim, num_scales):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_scales)
        )

    def forward(self, scale_features):
        """
        scale_features: List[ (B, feature_dim) ]
        """
        feats = torch.stack(scale_features, dim=1)  # (B, S, D)
        B, S, D = feats.shape

        # 根据全局特征生成 gate
        global_feature = feats.mean(dim=1)  # (B, D)
        gates = self.gate_mlp(global_feature)  # (B, S)
        gates = F.softmax(gates, dim=-1)

        fused = torch.sum(gates.unsqueeze(-1) * feats, dim=1)  # (B, D)
        return fused, gates
