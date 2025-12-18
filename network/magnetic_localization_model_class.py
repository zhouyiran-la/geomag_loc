import torch
import torch.nn as nn

from layer.denoising_autoencoder import DenoisingAutoencoder
from layer.single_scale_transformer import SingleScaleTransformer
from layer.multi_scale_attention_fusion import MultiScaleAttentionFusion

class MagneticLocalizationModel(nn.Module):
    """完整的磁序列定位模型（多尺度特征融合 + 尺度级注意力）"""
    def __init__(self, input_dim=3, d_model=64, scales=[10, 20, 30],
                 nhead=8, num_layers=3, num_positions=100):
        super(MagneticLocalizationModel, self).__init__()
        self.scales = scales
        self.input_dim = input_dim
        # ===== 1. 数据预处理模块 =====
        self.dae = DenoisingAutoencoder(input_dim=input_dim, hidden_dim=64, output_dim=input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        # ===== 2. 多尺度特征提取模块 =====
        self.scale_transformers = nn.ModuleList([
            SingleScaleTransformer(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                scale_size=scale
            ) for scale in scales
        ])
        # ===== 3. 多尺度注意力融合模块 =====
        feature_dim = d_model // 4  # 每个尺度特征的维度
        self.attention_fusion = MultiScaleAttentionFusion(
            feature_dim=feature_dim,
            num_scales=len(scales)
        )
        # ===== 4. 定位模块 =====
        self.localization_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim*2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_positions),  # 输出位置概率
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x):
        """
        x: 输入磁序列 (batch_size, seq_len, input_dim)
        return: (batch_size, num_positions)
        """
        # 1. 数据预处理
        denoised_x = self.dae(x)  # 去噪 (batch_size, seq_len, input_dim)
        normalized_x = self.layer_norm(denoised_x)
        # 2. 多尺度特征提取
        scale_features = []
        for transformer in self.scale_transformers:
            feat = transformer(normalized_x)  # (batch_size, feature_dim)
            scale_features.append(feat)
        # 3. 多尺度注意力融合
        fused_features, attn_weights = self.attention_fusion(scale_features)  # (batch_size, feature_dim)
        # 4. 定位预测
        position_logits = self.localization_head(fused_features)  # (batch_size, num_positions)
        return position_logits, attn_weights
