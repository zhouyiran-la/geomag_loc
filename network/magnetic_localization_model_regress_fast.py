import torch
import torch.nn as nn

from layer.denoising_autoencoder import DenoisingAutoencoder
from layer.multi_scale_attention_fusion import MultiScaleAttentionFusion
from layer.single_scale_transformer_fast import SingleScaleTransformerFast


class MagneticLocalizationModelFast(nn.Module):
    """
    MagneticLocalizationModel 的轻量版：
    - 单尺度 Transformer 批量处理所有子序列，避免 Python 循环热点；
    - 子序列聚合改为注意力汇聚，摒弃高维线性投影。
    """

    def __init__(self, input_dim=3, d_model=64, scales=None, nhead=8, num_layers=3, output_dim=2):
        super().__init__()
        if scales is None:
            scales = [10, 20, 30]

        self.scales = scales
        self.dae = DenoisingAutoencoder(input_dim=input_dim, hidden_dim=64, output_dim=input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.scale_transformers = nn.ModuleList(
            [
                SingleScaleTransformerFast(
                    input_dim=input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    scale_size=scale,
                )
                for scale in scales
            ]
        )

        feature_dim = d_model // 4
        self.attention_fusion = MultiScaleAttentionFusion(feature_dim=feature_dim, num_scales=len(scales))

        self.localization_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        return: coords, attn_weights
        """
        denoised_x = self.dae(x)
        normalized_x = self.layer_norm(denoised_x)

        scale_features = [transformer(normalized_x) for transformer in self.scale_transformers]
        fused_features, attn_weights = self.attention_fusion(scale_features)
        coords = self.localization_head(fused_features)
        return coords, attn_weights
