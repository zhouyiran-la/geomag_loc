import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class SingleScaleTransformerFast(nn.Module):
    """优化版单尺度 Transformer：批量处理子序列并使用轻量聚合."""

    def __init__(self, input_dim=3, d_model=64, nhead=8, num_layers=3, scale_size=10):
        super().__init__()
        self.scale_size = scale_size
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 注意力式聚合，避免将所有子序列特征拼成长向量
        self.subseq_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.fc_scale = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model // 4),
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        return: (batch_size, feature_dim=d_model//4)
        """
        batch_size, seq_len, _ = x.shape
        if seq_len % self.scale_size != 0:
            pad = self.scale_size - (seq_len % self.scale_size)
            x = F.pad(x, (0, 0, 0, pad))
            seq_len += pad

        num_subseq = seq_len // self.scale_size
        x = x.view(batch_size, num_subseq, self.scale_size, -1)
        x = x.reshape(batch_size * num_subseq, self.scale_size, -1)

        proj = self.input_proj(x) * math.sqrt(self.d_model)
        encoded = self.pos_encoding(proj)
        encoded = self.transformer_encoder(encoded)
        pooled = encoded.mean(dim=1)  # (B * num_subseq, d_model)

        pooled = pooled.view(batch_size, num_subseq, self.d_model)
        attn_scores = self.subseq_attn(pooled)
        attn_weights = torch.softmax(attn_scores, dim=1)
        scale_repr = (pooled * attn_weights).sum(dim=1)

        return self.fc_scale(scale_repr)
