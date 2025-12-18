import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块（batch-first）"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 变为 (1, max_len, d_model) 以适配 batch-first 输入
        pe = pe.unsqueeze(0)
        self.pe: torch.Tensor
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return x + self.pe[:, :x.size(1), :]

