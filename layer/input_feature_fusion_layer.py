import torch
import torch.nn as nn
import numpy as np

from .positional_encoding import PositionalEncoding

class FeatureFusionLayer(nn.Module):
    """
    将多种地磁特征 (原始/梯度/邻域/频谱) 统一映射到 Transformer 输入维度。
    """

    def __init__(self, 
                 input_dims,      # dict，例如 {"x_mag_grad":9, "x_mag_aug":15, "x_mag_spectral":128}
                 d_model=64,
                 fusion_mode="concat",  # 或 "add"、"attn"
                 use_pos_encoding=True):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.projections = nn.ModuleDict({
            k: nn.Linear(v, d_model) for k, v in input_dims.items()
        })
        self.feature_keys = list(self.projections.keys())
        if fusion_mode == "concat":
            self.concat_projection = nn.Linear(d_model * len(self.feature_keys), d_model)

        if fusion_mode == "attn":
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, features):
        """
        Args:
            features: dict, 包含若干键:
                'x_mag_grad': (B, T1, F1)
                'x_mag_aug': (B, T2, F2)
                'x_mag_spectral': (B, T3, F3)
        Returns:
            x_out: (B, T, d_model)
        """
        available_feats = []
        for key in self.feature_keys:
            feat = features.get(key, None)
            if feat is None:
                continue
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).float()
            available_feats.append((key, feat))

        if not available_feats:
            raise ValueError("FeatureFusionLayer: no valid features found in input.")

        min_len = min(feat.shape[1] for _, feat in available_feats)
        proj_feats = []
        for key, feat in available_feats:
            feat = feat[:, :min_len, :]
            proj_feats.append(self.projections[key](feat))

        if len(proj_feats) == 1:
            x_out = proj_feats[0]
        elif self.fusion_mode == "concat":
            x_out = torch.cat(proj_feats, dim=-1)
            in_dim = self.concat_projection.in_features
            feat_dim = x_out.shape[-1]
            if feat_dim < in_dim:
                pad = torch.zeros(*x_out.shape[:-1], in_dim - feat_dim, device=x_out.device, dtype=x_out.dtype)
                x_out = torch.cat([x_out, pad], dim=-1)
            elif feat_dim > in_dim:
                x_out = x_out[..., :in_dim]
            x_out = self.concat_projection(x_out)
        elif self.fusion_mode == "add":
            x_out = torch.stack(proj_feats, dim=0).mean(dim=0)
        elif self.fusion_mode == "attn":
            x_stack = torch.stack(proj_feats, dim=1)  # (B, num_feats, T, d_model)
            B, N, T, D = x_stack.shape
            x_stack = x_stack.view(B*N, T, D)
            x_out, _ = self.attn(x_stack, x_stack, x_stack)
            x_out = x_out.view(B, N, T, D).mean(dim=1)
        else:
            raise ValueError(f"未知融合模式: {self.fusion_mode}")

        if self.use_pos_encoding:
            x_out = self.pos_encoder(x_out)
        return x_out
