# network/knn_dtw_localizer.py
import numpy as np


def weighted_average_coords(neighbor_coords: np.ndarray,
                            neighbor_dists: np.ndarray,
                            eps: float = 1e-8) -> np.ndarray:
    """
    基于距离倒数加权平均坐标
    neighbor_coords: (k, 2)
    neighbor_dists:  (k,)
    """
    neighbor_coords = np.asarray(neighbor_coords, dtype=np.float32)
    neighbor_dists = np.asarray(neighbor_dists, dtype=np.float64)

    weights = 1.0 / (neighbor_dists + eps)
    weights = weights / np.sum(weights)
    pred = np.sum(neighbor_coords * weights[:, None], axis=0)
    return pred.astype(np.float32)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    标准多维 DTW 距离
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
                dp[i - 1, j],      # insertion
                dp[i, j - 1],      # deletion
                dp[i - 1, j - 1],  # match
            )

    return float(dp[L1, L2])


class MagneticKNNDTWLocalizer:
    """
    两阶段地磁定位器：
    1) KNN 粗召回（展平特征 + 欧氏距离）
    2) DTW 精匹配（仅在 Top-M 候选中计算 DTW）
    """
    def __init__(self,
                 ref_feats: np.ndarray,
                 ref_seqs: np.ndarray,
                 ref_coords: np.ndarray,
                 coarse_top_m: int = 50,
                 fine_top_k: int = 3,
                 weighted: bool = True):
        """
        ref_feats:  (N, D)
        ref_seqs:   (N, L, C)
        ref_coords: (N, 2)
        """
        self.ref_feats = np.asarray(ref_feats, dtype=np.float32)
        self.ref_seqs = np.asarray(ref_seqs, dtype=np.float32)
        self.ref_coords = np.asarray(ref_coords, dtype=np.float32)

        self.coarse_top_m = int(coarse_top_m)
        self.fine_top_k = int(fine_top_k)
        self.weighted = bool(weighted)

        n1 = len(self.ref_feats)
        n2 = len(self.ref_seqs)
        n3 = len(self.ref_coords)
        if not (n1 == n2 == n3):
            raise ValueError(
                f"参考库数量不一致: feats={n1}, seqs={n2}, coords={n3}"
            )

        if self.coarse_top_m <= 0:
            raise ValueError("coarse_top_m 必须 > 0")
        if self.fine_top_k <= 0:
            raise ValueError("fine_top_k 必须 > 0")

    def _coarse_retrieve(self, x_query: np.ndarray):
        """
        第一阶段：KNN 粗召回
        返回 coarse_idx, coarse_dists
        """
        q_feat = np.asarray(x_query, dtype=np.float32).reshape(-1)   # (D,)
        dists = np.linalg.norm(self.ref_feats - q_feat[None, :], axis=1)

        top_m = min(self.coarse_top_m, len(dists))
        coarse_idx = np.argsort(dists)[:top_m]
        coarse_dists = dists[coarse_idx]
        return coarse_idx, coarse_dists

    def _fine_match(self, x_query: np.ndarray, candidate_idx: np.ndarray):
        """
        第二阶段：对粗召回候选做 DTW 精匹配
        返回 fine_idx_global, fine_dists
        """
        x_query = np.asarray(x_query, dtype=np.float32)

        candidate_seqs = self.ref_seqs[candidate_idx]
        dtw_dists = np.empty(len(candidate_seqs), dtype=np.float64)

        for i, ref_seq in enumerate(candidate_seqs):
            dtw_dists[i] = dtw_distance(x_query, ref_seq)

        top_k = min(self.fine_top_k, len(dtw_dists))
        local_idx = np.argsort(dtw_dists)[:top_k]
        fine_idx_global = candidate_idx[local_idx]
        fine_dists = dtw_dists[local_idx]
        return fine_idx_global, fine_dists

    def predict_one(self, x_query: np.ndarray) -> np.ndarray:
        """
        x_query: (L, C)
        """
        coarse_idx, _ = self._coarse_retrieve(x_query)
        fine_idx, fine_dists = self._fine_match(x_query, coarse_idx)

        fine_coords = self.ref_coords[fine_idx]

        if self.weighted:
            return weighted_average_coords(fine_coords, fine_dists)
        return np.mean(fine_coords, axis=0).astype(np.float32)

    def predict_one_with_details(self, x_query: np.ndarray) -> dict:
        """
        返回中间检索细节，方便调试和可视化
        """
        coarse_idx, coarse_dists = self._coarse_retrieve(x_query)
        fine_idx, fine_dists = self._fine_match(x_query, coarse_idx)
        fine_coords = self.ref_coords[fine_idx]

        if self.weighted:
            pred = weighted_average_coords(fine_coords, fine_dists)
        else:
            pred = np.mean(fine_coords, axis=0).astype(np.float32)

        return {
            "pred": pred,
            "coarse_idx": coarse_idx,
            "coarse_dists": coarse_dists,
            "fine_idx": fine_idx,
            "fine_dists": fine_dists,
            "fine_coords": fine_coords,
        }

    def predict_batch(self, x_batch: np.ndarray) -> np.ndarray:
        preds = [self.predict_one(x) for x in x_batch]
        return np.stack(preds, axis=0)