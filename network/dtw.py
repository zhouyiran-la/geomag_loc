# network/dtw_localizer.py
import numpy as np


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    标准 DTW 距离
    seq_a: (L1, C)
    seq_b: (L2, C)
    """
    seq_a = np.asarray(seq_a, dtype=np.float32)
    seq_b = np.asarray(seq_b, dtype=np.float32)

    L1, C1 = seq_a.shape
    L2, C2 = seq_b.shape
    if C1 != C2:
        raise ValueError(f"通道数不一致: {C1} vs {C2}")

    dp = np.full((L1 + 1, L2 + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1], ord=2)
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    return float(dp[L1, L2])


def weighted_average_coords(neighbor_coords: np.ndarray,
                            neighbor_dists: np.ndarray,
                            eps: float = 1e-8) -> np.ndarray:
    weights = 1.0 / (neighbor_dists + eps)
    weights = weights / np.sum(weights)
    pred = np.sum(neighbor_coords * weights[:, None], axis=0)
    return pred.astype(np.float32)


class MagneticDTWLocalizer:
    """
    基于 DTW 距离的地磁序列匹配定位器
    """
    def __init__(self, ref_seqs: np.ndarray, ref_coords: np.ndarray,
                 k: int = 3, weighted: bool = True):
        """
        ref_seqs:   (N, L, 3)
        ref_coords: (N, 2)
        """
        self.ref_seqs = np.asarray(ref_seqs, dtype=np.float32)
        self.ref_coords = np.asarray(ref_coords, dtype=np.float32)
        self.k = int(k)
        self.weighted = bool(weighted)

        if len(self.ref_seqs) != len(self.ref_coords):
            raise ValueError("ref_seqs 和 ref_coords 数量不一致")

    def predict_one(self, x_query: np.ndarray) -> np.ndarray:
        x_query = np.asarray(x_query, dtype=np.float32)

        dists = np.empty(len(self.ref_seqs), dtype=np.float64)
        for i, ref_seq in enumerate(self.ref_seqs):
            dists[i] = dtw_distance(x_query, ref_seq)

        nn_idx = np.argsort(dists)[:self.k]
        nn_dists = dists[nn_idx]
        nn_coords = self.ref_coords[nn_idx]

        if self.weighted:
            return weighted_average_coords(nn_coords, nn_dists)
        return np.mean(nn_coords, axis=0).astype(np.float32)

    def predict_batch(self, x_batch: np.ndarray) -> np.ndarray:
        preds = [self.predict_one(x) for x in x_batch]
        return np.stack(preds, axis=0)