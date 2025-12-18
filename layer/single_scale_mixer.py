import torch
import torch.nn as nn

class PatchMixerBlock(nn.Module):
    """
    TimeMixer / MLP-Mixer 风格：Token-Mixing + Channel-Mixing
    """

    def __init__(self, d_model=64, token_mlp_dim=128, channel_mlp_dim=128):
        super().__init__()

        # Token mixing
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, d_model)
        )

        # Channel mixing
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, d_model)
        )

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class SingleScaleMixer(nn.Module):
    """
    替代原来的 SingleScaleTransformer：
    - patch 化
    - Mixer blocks
    """

    def __init__(self, input_dim, d_model=64, patch_size=10, depth=3):
        super().__init__()
        self.patch_size = patch_size

        self.patch_embed = nn.Linear(input_dim * patch_size, d_model)
        self.blocks = nn.ModuleList([PatchMixerBlock(d_model) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, T, C = x.shape
        # Patch 化
        T_p = T // self.patch_size
        x = x[:, :T_p * self.patch_size, :]
        x = x.reshape(B, T_p, self.patch_size * C)
        x = self.patch_embed(x)  # (B, T_p, d_model)

        for blk in self.blocks:
            x = blk(x)

        # 池化得到该尺度的特征 (B, d_model)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x
