import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding
from .autoformer_encdec import series_decomp
from .embed import DataEmbedding_wo_pos
from .per_scale_encoder import *

# ===== 2) 多尺度季节/趋势 mixing （时间维线性映射，仿 TimeMixer 思路） =====

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


# class PastDecomposableMixing(nn.Module):
#     """
#     PDM multi-scale mixing block (season bottom-up + trend top-down).
#     """

#     def __init__(
#         self,
#         seq_len: int,
#         d_model: int,
#         down_sampling_window: int = 2,
#         down_sampling_layers: int = 2,
#         channel_independence: bool = False,
#         d_ff: int = 256,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.seq_len = seq_len
#         self.d_model = d_model
#         self.r = down_sampling_window
#         self.M = down_sampling_layers
#         self.channel_independence = channel_independence
#         self.dropout = nn.Dropout(dropout)

#         if not channel_independence:
#             self.cross = nn.Sequential(
#                 nn.Linear(d_model, d_ff),
#                 nn.GELU(),
#                 nn.Linear(d_ff, d_model),
#             )

#         self.season_mixer = MultiScaleSeasonMixer(
#             seq_len=seq_len,
#             down_sampling_window=down_sampling_window,
#             down_sampling_layers=down_sampling_layers,
#             d_model=d_model,
#         )

#         self.trend_mixer = MultiScaleTrendMixer(
#             seq_len=seq_len,
#             down_sampling_window=down_sampling_window,
#             down_sampling_layers=down_sampling_layers,
#             d_model=d_model,
#         )

#     def forward(self, x_list, season_list, trend_list):
#         length_list = [s.size(1) for s in season_list]
#         season_cross_list = []
#         trend_cross_list = []
#         for s, t in zip(season_list, trend_list):
#             if not self.channel_independence:
#                 s = self.cross(s)
#                 t = self.cross(t)
#             season_cross_list.append(s)
#             trend_cross_list.append(t)
#         # bottom-up seasonal mixing
#         out_season = self.season_mixer(season_cross_list)

#         # top-down trend mixing
#         out_trend = self.trend_mixer(trend_cross_list)

#         # 合并
#         out_list = []
#         for ori, s, t, L in zip(x_list, out_season, out_trend, length_list):
#             # y = s + t
#             y = s
#             if self.channel_independence:
#                 y = ori + self.out_cross(y)
#             out_list.append(y[:, :L, :])
#         return out_list

class PastDecomposableMixing(nn.Module):
    """
    PDM multi-scale mixing block (season bottom-up + trend top-down).
    可配置版本：
    - use_mixing=True  -> 原始 PDM（bottom-up + top-down）
    - use_mixing=False -> 仅分解，不做跨尺度 mixing
    - fuse_mode        -> 尺度内融合策略(add season_only trend_only)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,
        channel_independence: bool = False,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_mixing: bool = True,    
        fuse_mode: str = "add",    
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.channel_independence = channel_independence
        self.dropout = nn.Dropout(dropout)

        self.use_mixing = use_mixing
        self.fuse_mode = fuse_mode

        if not channel_independence:
            self.cross = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        # 只有在 use_mixing=True 时才构建 mixer
        if self.use_mixing:
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
        else:
            self.season_mixer = None
            self.trend_mixer = None

    def forward(self, x_list, season_list, trend_list):

        length_list = [s.size(1) for s in season_list]

        season_cross_list = []
        trend_cross_list = []

        # === 1) 通道映射（不涉及尺度间交互） ===
        for s, t in zip(season_list, trend_list):
            if not self.channel_independence:
                s = self.cross(s)
                t = self.cross(t)
            season_cross_list.append(s)
            trend_cross_list.append(t)

        # === 2) 是否做跨尺度 mixing ===
        if self.use_mixing:
            out_season = self.season_mixer(season_cross_list) # type: ignore
            out_trend = self.trend_mixer(trend_cross_list) # type: ignore
        else:
            out_season = season_cross_list
            out_trend = trend_cross_list

        # === 3) 尺度内融合 ===
        out_list = []

        for ori, s, t, L in zip(x_list, out_season, out_trend, length_list):

            if self.fuse_mode == "add":
                y = s + t
            elif self.fuse_mode == "season_only":
                y = s
            elif self.fuse_mode == "trend_only":
                y = t
            else:
                raise ValueError(f"Unknown fuse_mode={self.fuse_mode}")

            out_list.append(y[:, :L, :])

        return out_list


# class TimeMixerMultiScaleEncoderV3(nn.Module):
#     """
#     推荐版：多尺度下采样 -> 分解 -> 分别 embedding -> PDM 只做多尺度 mixing。
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         d_model: int,
#         seq_len: int,
#         down_sampling_window: int = 2,
#         down_sampling_layers: int = 2,
#         moving_avg_kernel: int = 25,
#         num_pdm_blocks: int = 1,
#         nhead: int = 8,
#         num_layers: int = 2,
#         enc_out_dim = 64
#     ):
#         super().__init__()

#         self.seq_len = seq_len
#         self.input_dim = input_dim
#         self.d_model = d_model
#         self.r = down_sampling_window
#         self.M = down_sampling_layers
#         self.num_pdm_blocks = num_pdm_blocks
#         self.decomp = series_decomp(moving_avg_kernel)
#         self.embedding = DataEmbedding_wo_pos(
#             c_in=input_dim,
#             d_model=d_model,
#         )

#         self.out_dim = enc_out_dim
#         self.pdm_blocks = nn.ModuleList([
#             PastDecomposableMixing(
#                 seq_len=seq_len,
#                 d_model=d_model,
#                 down_sampling_window=down_sampling_window,
#                 down_sampling_layers=down_sampling_layers,
#             )
#             for _ in range(num_pdm_blocks)
#         ])

#         # === 3) 每个尺度一个 Transformer encoder ===
#         # self.scale_encoders = nn.ModuleList([
#         #     PerScaleTransformerEncoderV2(d_model=d_model, nhead=nhead, num_layers=num_layers, out_dim=enc_out_dim)
#         #     for _ in range(self.M + 1)
#         # ])

#         # # === 3) 每个尺度一个 LSTM encoder ===
#         # self.scale_encoders = nn.ModuleList([
#         #     PerScaleLSTMEncoder(
#         #         d_model=d_model,
#         #         num_layers=num_layers,
#         #         out_dim=enc_out_dim,
#         #         hidden_dim=d_model,      # 或者你想要的其他 hidden_dim
#         #         bidirectional=False,     # 想试双向的话改成 True
#         #         dropout=0.1,
#         #         use_pos_encoding=False,  # 如果想保留 PE，可以改成 True
#         #     )
#         #     for _ in range(self.M + 1)
#         # ])
#         # === 3) 每个尺度一个 RNN encoder ===
#         # self.scale_encoders = nn.ModuleList([
#         #     PerScaleRNNEncoder(
#         #         d_model=d_model,
#         #         num_layers=num_layers,
#         #         out_dim=enc_out_dim,
#         #         hidden_dim=d_model,      # 或者你想要的其他 hidden_dim
#         #         bidirectional=False,     # 想试双向的话改成 True
#         #         dropout=0.1,
#         #         use_pos_encoding=False,  # 如果想保留 PE，可以改成 True
#         #     )
#         #     for _ in range(self.M + 1)
#         # ])
#         # === 3) 每个尺度一个 TCN encoder ===
#         self.scale_encoders = nn.ModuleList([
#             PerScaleTCNEncoder(
#                 d_model=d_model,
#                 num_layers=num_layers,
#                 out_dim=enc_out_dim,
#                 hidden_dim=d_model,     
#                 kernel_size=3,     
#                 dropout=0.1,
#                 use_pos_encoding=False,  
#             )
#             for _ in range(self.M + 1)
#         ])

#     def _multi_scale_inputs(self, x):
#         B, T, C = x.shape

#         target_len = math.ceil(T / (self.r ** self.M)) * (self.r ** self.M)
#         if target_len > T:
#             pad = target_len - T
#             x = F.pad(x, (0, 0, 0, pad))
#             T = target_len

#         x_list = []

#         cur = x.permute(0, 2, 1)

#         # scale 0
#         x_list.append(cur.permute(0, 2, 1))

#         for _ in range(self.M):
#             cur = F.avg_pool1d(cur, kernel_size=self.r, stride=self.r)
#             # cur = F.max_pool1d(cur, kernel_size=self.r, stride=self.r)
#             x_list.append(cur.permute(0, 2, 1))

#         return x_list

#     def forward(self, x):
#         # === Step 1: multi-scale ===
#         x_scales = self._multi_scale_inputs(x) #(B, T, 3)

#         # === Step 2: decomposition after sampling ===
#         season_scales = []
#         trend_scales = []
        
#         for xs in x_scales:
#             s, t = self.decomp(xs)
#             season_scales.append(s)
#             trend_scales.append(t)
        
#         # === Step 3: embedding for origin & season & trend ===
#         # origin seq embed
#         emb_scales = [self.embedding(xs, None) for xs in x_scales] # list of (B, L_M, D)
#         # season seq embed
#         season_emb = [self.embedding(s, None) for s in season_scales]
#         # trend seq embed
#         trend_emb = [self.embedding(t, None) for t in trend_scales]

#         # === Step 4: PDM mixing blocks ===
#         out_list = None
#         for pdm in self.pdm_blocks:
#             out_list = pdm(emb_scales, season_emb, trend_emb)
        
#         assert out_list
#         # === Step 5: per-scale Transformer ===
#         encoded = [enc(out_list[i]) for i, enc in enumerate(self.scale_encoders)]

#         return encoded


class TimeMixerMultiScaleEncoderV3(nn.Module):
    """
    多尺度下采样 -> (可选) 分解+PDM mixing -> 每尺度 encoder
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
        enc_out_dim=64,
        use_decomp: bool = True,   # ★ 新增
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.r = down_sampling_window
        self.M = down_sampling_layers
        self.num_pdm_blocks = num_pdm_blocks
        self.use_decomp = use_decomp

        # embedding（无论是否分解都用）
        self.embedding = DataEmbedding_wo_pos(c_in=input_dim, d_model=d_model)

        # 只有在 use_decomp=True 时才需要这些模块
        if self.use_decomp:
            self.decomp = series_decomp(moving_avg_kernel)
            self.pdm_blocks = nn.ModuleList([
                PastDecomposableMixing(
                    seq_len=seq_len,
                    d_model=d_model,
                    down_sampling_window=down_sampling_window,
                    down_sampling_layers=down_sampling_layers,
                    use_mixing=True,
                    fuse_mode='add',  # 先试只保留趋势信息，看看效果；后续可以改成 'add' 或 'season_only' 做对比
                )
                for _ in range(num_pdm_blocks)
            ])
        else:
            self.decomp = None
            self.pdm_blocks = None
        # === 3) 每个尺度一个 Transformer encoder ===
        # self.scale_encoders = nn.ModuleList([
        #     PerScaleTransformerEncoderV2(d_model=d_model, nhead=nhead, num_layers=num_layers, out_dim=enc_out_dim)
        #     for _ in range(self.M + 1)
        # ])

        # # === 3) 每个尺度一个 LSTM encoder ===
        # self.scale_encoders = nn.ModuleList([
        #     PerScaleLSTMEncoder(
        #         d_model=d_model,
        #         num_layers=num_layers,
        #         out_dim=enc_out_dim,
        #         hidden_dim=d_model,      # 或者你想要的其他 hidden_dim
        #         bidirectional=False,     # 想试双向的话改成 True
        #         dropout=0.1,
        #         use_pos_encoding=False,  # 如果想保留 PE，可以改成 True
        #     )
        #     for _ in range(self.M + 1)
        # ])
        # === 3) 每个尺度一个 RNN encoder ===
        # self.scale_encoders = nn.ModuleList([
        #     PerScaleRNNEncoder(
        #         d_model=d_model,
        #         num_layers=num_layers,
        #         out_dim=enc_out_dim,
        #         hidden_dim=d_model,      # 或者你想要的其他 hidden_dim
        #         bidirectional=False,     # 想试双向的话改成 True
        #         dropout=0.1,
        #         use_pos_encoding=False,  # 如果想保留 PE，可以改成 True
        #     )
        #     for _ in range(self.M + 1)
        # ])
        # === 3) 每个尺度一个 TCN encoder ===
        self.scale_encoders = nn.ModuleList([
            PerScaleTCNEncoder(
                d_model=d_model,
                num_layers=num_layers,
                out_dim=enc_out_dim,
                hidden_dim=d_model,
                kernel_size=3,
                dropout=0.1,
                use_pos_encoding=False,
            )
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
            x_list.append(cur.permute(0, 2, 1))

        return x_list

    def forward(self, x):
        # Step 1: multi-scale
        x_scales = self._multi_scale_inputs(x)  # list of (B, L_k, C)

        # ====== 分支 1：多尺度 + 无分解（baseline）======
        if not self.use_decomp:
            emb_scales = [self.embedding(xs, None) for xs in x_scales]   # list (B,L_k,D)
            encoded = [enc(emb_scales[i]) for i, enc in enumerate(self.scale_encoders)]
            return encoded

        # ====== 分支 2：完整模型（分解 + PDM mixing）======
        season_scales, trend_scales = [], []
        for xs in x_scales:
            s, t = self.decomp(xs) # type: ignore
            season_scales.append(s)
            trend_scales.append(t)

        emb_scales = [self.embedding(xs, None) for xs in x_scales]
        season_emb = [self.embedding(s, None) for s in season_scales]
        trend_emb  = [self.embedding(t, None) for t in trend_scales]

        out_list = None
        for pdm in self.pdm_blocks: # type: ignore
            out_list = pdm(emb_scales, season_emb, trend_emb)

        encoded = [enc(out_list[i]) for i, enc in enumerate(self.scale_encoders)] # type: ignore
        return encoded
