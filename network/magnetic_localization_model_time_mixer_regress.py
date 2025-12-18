import torch
import torch.nn as nn

from layer.denoising_autoencoder import DenoisingAutoencoder
from layer.multi_scale_attention_fusion import MultiScaleAttentionFusion, MultiScaleAttentionFusionV2
# from layer.timemixer_multiscale_encoder import TimeMixerMultiScaleEncoder
# from layer.timemixer_multiscale_encoder import TimeMixerMultiScaleEncoderV2
from layer.timemixer_multiscale_encoder_v2 import TimeMixerMultiScaleEncoderV3

class MagneticLocalizationTimeMixer(nn.Module):
    """
    使用 TimeMixer 风格的多尺度季节/趋势分解 + 双向混合，
    每个尺度后接 Transformer Encoder，再做多尺度注意力融合，最后回归坐标。
    """

    def __init__(
        self,
        input_dim=3,
        d_model=64,
        seq_len=256,             # 训练时地磁序列长度
        down_sampling_window=2,  # 下采样基数
        down_sampling_layers=2,  # 得到 M+1 个尺度
        num_pdm_blocks=1,        # PDM层数
        moving_avg_kernel=25,    # 分解核大小
        nhead=8,                 # 注意力头数
        num_layers=2,            # 编码器层数
        output_dim=2,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.dae = DenoisingAutoencoder(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=input_dim,
        )
        self.layer_norm = nn.LayerNorm(input_dim)

        self.timemixer_encoder = TimeMixerMultiScaleEncoderV3(
            input_dim=input_dim,
            d_model=d_model,
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            moving_avg_kernel=moving_avg_kernel,
            num_pdm_blocks=num_pdm_blocks,
            nhead=nhead,
            num_layers=num_layers,
            enc_out_dim = 64
        )

        feature_dim = 64
        self.attention_fusion = MultiScaleAttentionFusionV2(
            feature_dim=feature_dim,
            num_scales=down_sampling_layers + 1,
            d_k=64,
            residual=True
        )

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
        x: (B, T, input_dim)
        return: coords, attn_weights  # attn_weights 是多尺度权重
        """
        # 保证输入长度为固定 seq_len（你训练时可以先 pad/截断一遍）
        if x.size(1) != self.seq_len:
            if x.size(1) > self.seq_len:
                x = x[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - x.size(1)
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))

        # denoised = self.dae(x)
        # normed = self.layer_norm(x)

        scale_features = self.timemixer_encoder(x)  # list of (B, feature_dim)
        fused, attn_weights = self.attention_fusion(scale_features)

        coords = self.localization_head(fused)
        return coords, attn_weights
