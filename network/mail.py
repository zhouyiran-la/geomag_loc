import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utils
# ---------------------------
def gradient_sequence(x, keep_length=True):
    g = torch.diff(x, dim=1)
    if keep_length:
        pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        g = torch.cat([pad, g], dim=1)
    return g


def pad_to_multiple(x, multiple, dim):
    size = x.size(dim)
    remainder = size % multiple
    if remainder == 0:
        return x
    pad_len = multiple - remainder
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(*pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=dim)


# ---------------------------
# SFE Unit
# ---------------------------
class SFEUnit(nn.Module):
    def __init__(self, input_dim=3, subseq_len=100, gru_hidden=64, proj_dim=64):
        super().__init__()
        self.subseq_len = subseq_len
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, proj_dim)
        self.prelu = nn.PReLU()
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, grad_seq):
        B, T, C = grad_seq.shape
        x = pad_to_multiple(grad_seq, self.subseq_len, dim=1)
        T_pad = x.size(1)
        P = T_pad // self.subseq_len
        x = x.view(B * P, self.subseq_len, C)
        _, h_n = self.gru(x)
        gp = self.prelu(self.fc(h_n[-1]))  # (B*P, proj_dim)
        gp = gp.view(B, P, -1)
        gp = self.norm(gp)
        psi_m = gp.reshape(B, -1) 
        return psi_m


# ---------------------------
# Attention Generator
# ---------------------------
class AttentionGenerator(nn.Module):
    def __init__(self, input_dim=3, hidden=64, num_scales=6):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2)
        self.act = nn.PReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, num_scales)

    def forward(self, grad_seq):
        x = grad_seq.transpose(1, 2)  # (B, C, T)
        h = self.act(self.conv(x))
        h = self.pool(h).squeeze(-1)
        logits = self.fc(h) # (B, num_scales)
        return F.softmax(logits, dim=-1)


# ---------------------------
# MAIL Model
# ---------------------------
class MAIL(nn.Module):
    def __init__(self, input_dim=3, seq_len=500,
                 scale_lengths=(50, 100, 150, 200, 250, 500),
                 gru_hidden=64, proj_dim=64, attn_hidden=64):
        super().__init__()
        self.seq_len = seq_len
        self.scales = list(scale_lengths)
        self.num_scales = len(self.scales)

        self.sfe_units = nn.ModuleList([
            SFEUnit(input_dim, c, gru_hidden, proj_dim) for c in self.scales
        ])
        self.attn = AttentionGenerator(input_dim, attn_hidden, self.num_scales)

        # 初始化 regressor 为一个空 Sequential（避免 Pylance 报错）
        self.regressor: nn.Sequential = nn.Sequential()

    def build_regressor(self, psi_list, w):
        with torch.no_grad():
            fused_dim = sum([psi.size(-1) for psi in psi_list])
        regressor = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 2)
        )
        device = psi_list[0].device if psi_list else w.device
        self.regressor = regressor.to(device)

    def forward(self, x):
        # grad_seq = gradient_sequence(x, keep_length=True)
        grad_seq = x
        psi_list = [sfe(grad_seq) for sfe in self.sfe_units]
        w = self.attn(grad_seq)

        weighted = [w[:, i].unsqueeze(-1) * psi for i, psi in enumerate(psi_list)]
        f = torch.cat(weighted, dim=-1)

        if len(self.regressor) == 0:
            self.build_regressor(psi_list, w)

        out = self.regressor(f)
        return out
