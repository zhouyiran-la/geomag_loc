import torch
import torch.nn as nn

from layer.denoising_autoencoder import DenoisingAutoencoder
from layer.frontend_multiscale import MultiScaleDecomposition
from layer.single_scale_mixer import SingleScaleMixer
from layer.multi_scale_fusion_v2 import MultiScaleFusionV2


class MagneticLocalizationModelFastV2(nn.Module):
    def __init__(self,
                 input_dim=3,
                 d_model=64,
                 patch_sizes=[10, 20, 30],
                 depth=3,
                 output_dim=2):
        super().__init__()

        self.decomp = MultiScaleDecomposition(input_dim=input_dim)
        self.dae = DenoisingAutoencoder(input_dim=input_dim,
                                        hidden_dim=64,
                                        output_dim=input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        # 多尺度 Mixer
        self.scale_mixers = nn.ModuleList([
            SingleScaleMixer(input_dim=input_dim,
                             d_model=d_model,
                             patch_size=ps,
                             depth=depth)
            for ps in patch_sizes
        ])

        feature_dim = d_model
        self.fusion = MultiScaleFusionV2(feature_dim=feature_dim,
                                         num_scales=len(patch_sizes))

        # 回归头
        self.localization_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim)
        )

    def forward(self, x):
        """
        x: (B, T, 3)
        """
        trend, season, residual = self.decomp(x)

        denoised = self.dae(trend + season + residual)
        x = self.layer_norm(denoised)

        scale_features = [mix(x) for mix in self.scale_mixers]

        fused, gates = self.fusion(scale_features)
        coords = self.localization_head(fused)

        return coords, gates
