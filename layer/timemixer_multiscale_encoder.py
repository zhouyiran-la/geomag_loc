import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding
from .autoformer_encdec import series_decomp
from .embed import DataEmbedding_wo_pos
from .per_scale_encoder import PerScaleTransformerEncoder, PerScaleTransformerEncoderV2, PerScaleLSTMEncoder


# ===== 多尺度季节/趋势 mixing （时间维线性映射，仿 TimeMixer 思路） =====

class MultiScaleSeasonMixer(nn.Module):
    """
    Bottom-up：细 -> 粗，逐层注入季节信息。
    参考 TimeMixer 的 MultiScaleSeasonMixing 设计思路简化实现。
    """

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int, d_model: int):
        super().__init__()
        self.L = seq_len
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.d_model = d_model

        # 为每一层准备一个线性映射： L_m -> L_{m+1}
        self.layers = nn.ModuleList()
        for i in range(self.M):
            L_in = self.L // (self.r ** i)
            L_out = self.L // (self.r ** (i + 1))
            self.layers.append(
                nn.Sequential(
                    nn.Linear(L_in, L_out),
                    nn.GELU(),
                    nn.Linear(L_out, L_out),
                )
            )

    def forward(self, season_list):
        """
        season_list: list of (B, L_m, D)，长度 M+1，L_m = L / r^m
        返回同长度 list，每个已经做完 bottom-up mixing。
        """
        # 先统一成 (B, D, L_m) 便于在时间维上做线性映射
        tmp = [s.permute(0, 2, 1) for s in season_list]  # (B, D, L_m)

        out_list = []
        high = tmp[0]
        low = tmp[1] if len(tmp) > 1 else None
        out_list.append(high)  # scale 0

        for i in range(len(tmp) - 1):
            B, D, L_in = high.shape
            # (B, D, L_in) -> (B*D, L_in)
            x = high.reshape(B * D, L_in)
            x = self.layers[i](x)  # (B*D, L_out)
            x = x.view(B, D, -1)   # (B, D, L_out)

            low = low + x
            high = low

            out_list.append(high)
            if i + 2 < len(tmp):
                low = tmp[i + 2]

        # 再转回 (B, L_m, D)
        out_list = [o.permute(0, 2, 1) for o in out_list]
        return out_list


class MultiScaleTrendMixer(nn.Module):
    """
    Top-down：粗 -> 细，逐层向下传递 trend 信息。
    """

    def __init__(self, seq_len: int, down_sampling_window: int, down_sampling_layers: int, d_model: int):
        super().__init__()
        self.L = seq_len
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.d_model = d_model

        # 为每一层准备一个线性映射： L_{m+1} -> L_m
        self.layers = nn.ModuleList()
        # 注意：这里从 coarse 到 fine，所以反向构建
        for i in reversed(range(self.M)):
            L_in = self.L // (self.r ** (i + 1))
            L_out = self.L // (self.r ** i)
            self.layers.append(
                nn.Sequential(
                    nn.Linear(L_in, L_out),
                    nn.GELU(),
                    nn.Linear(L_out, L_out),
                )
            )

    def forward(self, trend_list):
        """
        trend_list: list of (B, L_m, D)，长度 M+1
        返回同长度 list，每个已经做完 top-down mixing。
        """
        tmp = [t.permute(0, 2, 1) for t in trend_list]  # (B, D, L_m)
        tmp_rev = list(reversed(tmp))

        out_list = []
        low = tmp_rev[0]    # 最粗
        high = tmp_rev[1] if len(tmp_rev) > 1 else None
        out_list.append(low)

        for i in range(len(tmp_rev) - 1):
            B, D, L_in = low.shape
            x = low.reshape(B * D, L_in)
            x = self.layers[i](x)   # (B*D, L_out)
            x = x.view(B, D, -1)

            high = high + x
            low = high

            out_list.append(low)
            if i + 2 < len(tmp_rev):
                high = tmp_rev[i + 2]

        out_list = [o.permute(0, 2, 1) for o in out_list]
        out_list.reverse()
        return out_list


class PastDecomposableMixing(nn.Module):
    """
    PDM using official TimeMixer seasonal/trend decomposition.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,
        moving_avg_kernel: int = 25,
        channel_independence: bool = False,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.channel_independence = channel_independence

        # === 使用官方 series_decomp 实现 ===
        self.decomp = series_decomp(moving_avg_kernel)

        self.dropout = nn.Dropout(dropout)

        if not channel_independence:
            self.cross = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        self.season_mixer = MultiScaleSeasonMixer(
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            d_model=d_model,
        )

        self.trend_mixer = MultiScaleTrendMixer(
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            d_model=d_model,
        )

        self.out_cross = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]

        # === 官方 seasonal/trend 分解 ===
        season_list = []
        trend_list = []
        for x in x_list:
            s, t = self.decomp(x)
            if not self.channel_independence:
                s = self.cross(s)
                t = self.cross(t)
            season_list.append(s)
            trend_list.append(t)

        # bottom-up seasonal mixing
        out_season = self.season_mixer(season_list)

        # top-down trend mixing
        out_trend = self.trend_mixer(trend_list)

        # 合并
        out_list = []
        for ori, s, t, L in zip(x_list, out_season, out_trend, length_list):
            y = s + t
            # y = t
            if self.channel_independence:
                y = ori + self.out_cross(y)
            out_list.append(y[:, :L, :])
        return out_list

# ===== 整体多尺度 TimeMixer Encoder，用于地磁特征抽取 =====

class TimeMixerMultiScaleEncoder(nn.Module):
    """
    输入一段地磁序列，返回每个尺度的 embedding 向量（给 MultiScaleAttentionFusion 用）。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        seq_len: int,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,
        moving_avg_kernel: int = 25,
        num_pdm_blocks: int = 1,
        nhead: int = 8,
        num_layers: int = 2,
        
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.r = down_sampling_window
        self.M = down_sampling_layers
        if num_pdm_blocks < 1:
            raise ValueError("num_pdm_blocks must be >= 1")
        self.num_pdm_blocks = num_pdm_blocks

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=seq_len // (self.r ** 0),  # 顶层长度
                d_model=d_model,
                down_sampling_window=down_sampling_window,
                down_sampling_layers=down_sampling_layers,
                moving_avg_kernel=moving_avg_kernel,
            )
            for _ in range(self.num_pdm_blocks)
        ])

        # 每个尺度一个 Transformer encoder
        self.scale_encoders = nn.ModuleList([
            PerScaleTransformerEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
            for _ in range(self.M + 1)
        ])

    def _multi_scale_inputs(self, x):
        """
        TimeMixer 的多尺度输入生成方式：avg pooling 下采样。
        x: (B, T, D)
        return: list of (B, L_m, D), m=0..M
        """
        B, T, D = x.shape

        # 先确保长度能整除 r^M，Pad 一点尾部
        target_len = math.ceil(T / (self.r ** self.M)) * (self.r ** self.M)
        if target_len > T:
            pad = target_len - T
            x = F.pad(x, (0, 0, 0, pad))
            T = target_len

        x_list = []
        cur = x.permute(0, 2, 1)  # (B, D, T)

        x_list.append(cur.permute(0, 2, 1))  # scale 0

        for i in range(self.M):
            cur = F.avg_pool1d(cur, kernel_size=self.r, stride=self.r)
            x_list.append(cur.permute(0, 2, 1))  # (B, L_m, D)

        return x_list

    def forward(self, x):
        """
        x: (B, T, input_dim)
        return: list of (B, feature_dim=d_model//4)
        """
        h = self.input_proj(x)      # (B, T, D)
        x_scales = self._multi_scale_inputs(h) # list of (B, L_m, D)

        # num_pdm_blocks 控制 PastDecomposableMixing 堆叠次数
        for pdm in self.pdm_blocks:
            x_scales = pdm(x_scales)

        features = [
            enc(x_scales[m]) for m, enc in enumerate(self.scale_encoders)
        ]  # 每个 (B, d_model//4)

        return features

class TimeMixerMultiScaleEncoderV2(nn.Module):
    """
    多尺度 TimeMixer Encoder —— 完全对齐 TimeMixer 官方流程：
    1) multi-scale 下采样
    2) 对每个尺度做 series_decomp（季节/趋势分解）
    3) 对分解后的 season 做 DataEmbedding_wo_pos（Conv1d + Temporal Emb）
    4) PastDecomposableMixing（再分解 + 双向 mixing）
    5) Per-scale Transformer Encoder
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        seq_len: int,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,
        moving_avg_kernel: int = 25,
        num_pdm_blocks: int = 1,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.num_pdm_blocks = num_pdm_blocks

        # === 1) 先使用 Autoformer/TimeMixer 的 series_decomp ===
        self.decomp = series_decomp(moving_avg_kernel)

        # === 2) 官方 TimeMixer 的 embedding（TokenEmbedding + Dropout）===
        # 替换掉你原来的 Linear(input_dim -> d_model)
        self.value_embedding = DataEmbedding_wo_pos(
            c_in=input_dim,
            d_model=d_model
        )

        # === 3) 多个 PDM block ===
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=seq_len,
                d_model=d_model,
                down_sampling_window=down_sampling_window,
                down_sampling_layers=down_sampling_layers,
                moving_avg_kernel=moving_avg_kernel,
            )
            for _ in range(num_pdm_blocks)
        ])

        # === 4) 每个尺度一个 Transformer Encoder ===
        self.scale_encoders = nn.ModuleList([
            PerScaleTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers
            )
            for _ in range(self.M + 1)
        ])


    # -------- Multi-scale 输入保持不变 -------- #
    def _multi_scale_inputs(self, x):
        B, T, D = x.shape

        target_len = math.ceil(T / (self.r ** self.M)) * (self.r ** self.M)
        if target_len > T:
            pad = target_len - T
            x = F.pad(x, (0, 0, 0, pad))
            T = target_len

        x_list = []
        cur = x.permute(0, 2, 1)  # (B, D, T)

        x_list.append(cur.permute(0, 2, 1))  # scale 0

        for i in range(self.M):
            cur = F.avg_pool1d(cur, kernel_size=self.r, stride=self.r)
            x_list.append(cur.permute(0, 2, 1))

        return x_list


    # ----------------- forward 流程 ----------------- #
    def forward(self, x):
        """
        x: (B, T, input_dim)
        return: list of (B, feature_dim=d_model//4)
        """

        # === Step 1：先多尺度下采样 ===
        x_scales = self._multi_scale_inputs(x)

        # === Step 2：每个尺度先进行季节/趋势分解 ===
        season_list = []
        trend_list = []
        for xs in x_scales:
            s, t = self.decomp(xs)
            season_list.append(s)
            trend_list.append(t)

        # === Step 3：官方 TimeMixer 设计：embedding 作用在季节项 ===
        # 注意：DataEmbedding_wo_pos 内部使用 Conv1d
        emb_list = [self.value_embedding(s, None) for s in season_list]

        # === Step 4：多个 PDM block ===
        for pdm in self.pdm_blocks:
            emb_list = pdm(emb_list)

        # === Step 5：每个尺度送入 Transformer Encoder ===
        encoded_features = [
            encoder(emb_list[m]) for m, encoder in enumerate(self.scale_encoders)
        ]

        return encoded_features
    
class TimeMixerMultiScaleEncoderV3(nn.Module):
    """
    推荐版：只在 PDM 内做一次季节/趋势分解（embedding 后）。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        seq_len: int,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,
        moving_avg_kernel: int = 25,
        num_pdm_blocks: int = 1,
        nhead: int = 8,
        num_layers: int = 2,
        enc_out_dim = 64
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.num_pdm_blocks = num_pdm_blocks
        self.embedding = DataEmbedding_wo_pos(
            c_in=input_dim,
            d_model=d_model
        )
        self.out_dim = enc_out_dim

        # === 2) PDM blocks（内部做唯一一次分解）===
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=seq_len,
                d_model=d_model,
                down_sampling_window=down_sampling_window,
                down_sampling_layers=down_sampling_layers,
                moving_avg_kernel=moving_avg_kernel,
            )
            for _ in range(num_pdm_blocks)
        ])

        # === 3) 每个尺度一个 Transformer encoder ===
        self.scale_encoders = nn.ModuleList([
            PerScaleTransformerEncoderV2(d_model=d_model, nhead=nhead, num_layers=num_layers, out_dim=enc_out_dim)
            for _ in range(self.M + 1)
        ])

    def _multi_scale_inputs(self, x):
        B, T, C = x.shape

        target_len = math.ceil(T / (self.r ** self.M)) * (self.r ** self.M)
        if target_len > T:
            pad = target_len - T
            x = F.pad(x, (0, 0, 0, pad))
            T = target_len

        x_list = []

        cur = x.permute(0, 2, 1)

        # scale 0
        x_list.append(cur.permute(0, 2, 1))

        for _ in range(self.M):
            cur = F.avg_pool1d(cur, kernel_size=self.r, stride=self.r)
            # cur = F.max_pool1d(cur, kernel_size=self.r, stride=self.r)
            x_list.append(cur.permute(0, 2, 1))

        return x_list

    def forward(self, x):
        # === Step 1: multi-scale ===
        x_scales = self._multi_scale_inputs(x) #(B, T, 3)
        # === Step 2: embedding (Conv1d) ===
        emb_scales = [self.embedding(xs, None) for xs in x_scales] # list of (B, L_M, D)
        # === Step 3: PDM blocks (唯一一次分解在内部) ===
        for pdm in self.pdm_blocks:
            emb_scales = pdm(emb_scales) 

        # === Step 4: per-scale Transformer ===
        encoded = [enc(emb_scales[i]) for i, enc in enumerate(self.scale_encoders)]

        return encoded
