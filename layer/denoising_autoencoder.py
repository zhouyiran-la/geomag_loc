import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    """去噪自编码器(DAE)用于数据降噪"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(DenoisingAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
