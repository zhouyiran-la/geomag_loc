import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# HLSTM 模型（回归版）
# ---------------------------

class HLSTMRegressor(nn.Module):
    """
    HLSTM（分层 LSTM）回归模型：
      输入：x (B, L, D)
      过程：
        1) 在模型内部滑窗切成 frames: (B, T, frame_len, D)
        2) frame-level LSTM -> 每个 frame 得到 embedding (B, T, H1)
        3) sequence-level BiLSTM -> (B, T, H2)
        4) 聚合（last/mean）-> (B, H2)
        5) head -> (B, 2)
      输出：preds (B,2), aux(dict)
    """

    def __init__(
        self,
        *,
        input_dim: int = 3,
        frame_len: int = 32,
        frame_stride: int = 16,
        frame_hidden: int = 64,
        frame_layers: int = 1,
        seq_hidden: int = 128,
        seq_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pool: str = "last",   # "last" or "mean"
        output_dim: int = 2,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.frame_len = int(frame_len)
        self.frame_stride = int(frame_stride)
        self.pool = pool

        # 子序列级 LSTM（frame-level）
        self.frame_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=int(frame_hidden),
            num_layers=int(frame_layers),
            batch_first=True,
            dropout=dropout if frame_layers > 1 else 0.0,
            bidirectional=False,
        )

        # 全序列级 LSTM（sequence-level）
        self.seq_lstm = nn.LSTM(
            input_size=int(frame_hidden),
            hidden_size=int(seq_hidden),
            num_layers=int(seq_layers),
            batch_first=True,
            dropout=dropout if seq_layers > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )

        self.seq_out_dim = int(seq_hidden) * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

        # 回归 head
        self.head = nn.Linear(self.seq_out_dim, int(output_dim))

    def _make_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        把长序列切成 frames（滑窗）
        输入 x: (B, L, D)
        输出 frames: (B, T, fl, D)
        """
        B, L, D = x.shape
        fl = self.frame_len
        fs = self.frame_stride

        if L < fl:
            raise ValueError(f"L={L} < frame_len={fl}")

        # unfold 后形状通常是 (B, T, D, fl)
        frames = x.unfold(dimension=1, size=fl, step=fs)

        # 调整为 (B, T, fl, D)
        frames = frames.permute(0, 1, 3, 2).contiguous()
        return frames


    def forward(self, x: torch.Tensor):
        """
        x: (B, L, D)
        返回 (preds, aux)
        """
        if x.dim() != 3:
            raise ValueError(f"期望输入形状 (B,L,D)，但得到 {tuple(x.shape)}")

        # 1) 切 frames
        frames = self._make_frames(x)                      # (B, T, fl, D)
        B, T, fl, D = frames.shape

        # 2) frame-level LSTM：每个 frame 编码成 embedding
        frames_ = frames.reshape(B * T, fl, D)             # (B*T, fl, D)
        _, (h_n, _) = self.frame_lstm(frames_)             # h_n: (layers, B*T, H1)
        frame_emb = h_n[-1]                                # (B*T, H1)
        frame_emb = frame_emb.reshape(B, T, -1)            # (B, T, H1)

        # 3) seq-level BiLSTM：对 embedding 序列建模
        seq_out, _ = self.seq_lstm(frame_emb)              # (B, T, H2)
        seq_out = self.dropout(seq_out)

        # 4) 聚合
        if self.pool == "last":
            pooled = seq_out[:, -1, :]                     # (B, H2)
        elif self.pool == "mean":
            pooled = seq_out.mean(dim=1)                   # (B, H2)
        else:
            raise ValueError(f"未知 pool={self.pool}，请用 'last' 或 'mean'")

        # 5) 回归输出
        preds = self.head(pooled)                          # (B,2)

        aux = {
            "num_frames": T,                               # 方便你 debug：一段序列被切成多少帧
        }
        return preds, aux

