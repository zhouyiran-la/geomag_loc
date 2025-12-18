import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .positional_encoding import PositionalEncoding

class SingleScaleTransformer(nn.Module):
    """单尺度Transformer特征提取器（输出该尺度整体特征）"""
    def __init__(self, input_dim=3, d_model=64, nhead=8, num_layers=3, scale_size=10):
        super(SingleScaleTransformer, self).__init__()
        self.scale_size = scale_size
        self.d_model = d_model
        # 输入投影层 + 位置编码
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        # Transformer 编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 自适应映射层：将变长的拼接特征映射到固定维度 d_model
        # 使用 LazyLinear 自动适应输入维度（第一次 forward 时确定）
        self.adaptive_proj = nn.LazyLinear(d_model)
        # 特征映射层：固定维度 d_model -> feature_dim
        self.fc_scale = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Tanh(),
            nn.Linear(d_model//2, d_model//4)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        return: (batch_size, feature_dim=d_model//4)
        """
        batch_size, seq_len, _ = x.shape
        # 确保长度能被scale_size整除
        if seq_len % self.scale_size != 0:
            pad_size = self.scale_size - (seq_len % self.scale_size)
            x = F.pad(x, (0, 0, 0, pad_size))
            seq_len += pad_size
        num_subseq = seq_len // self.scale_size
        x = x.view(batch_size, num_subseq, self.scale_size, -1)
        subseq_features = []
        for i in range(num_subseq):
            subseq = x[:, i, :, :]  # (batch_size, scale_size, input_dim)
            proj = self.input_proj(subseq) * math.sqrt(self.d_model)
            encoded = self.pos_encoding(proj)
            encoded = self.transformer_encoder(encoded)  # (batch_size, scale_size, d_model)
            pooled = encoded.mean(dim=1)  # (batch_size, d_model)
            subseq_features.append(pooled)
        
        # 拼接所有子序列的特征（保留拼接方式）
        scale_concat = torch.cat(subseq_features, dim=1)  # (batch_size, num_subseq * d_model)
        
        # 自适应映射：将变长拼接特征映射到固定维度 d_model
        scale_repr = self.adaptive_proj(scale_concat)  # (batch_size, d_model)
        
        # 对整尺度特征进行映射压缩
        scale_feature = self.fc_scale(scale_repr)  # (batch_size, d_model//4)
        return scale_feature
