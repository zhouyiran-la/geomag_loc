# network/knn_localizer.py
import numpy as np


def weighted_average_coords(neighbor_coords: np.ndarray,
                            neighbor_dists: np.ndarray,
                            eps: float = 1e-8) -> np.ndarray:
    weights = 1.0 / (neighbor_dists + eps)
    weights = weights / np.sum(weights)
    pred = np.sum(neighbor_coords * weights[:, None], axis=0)
    return pred.astype(np.float32)


class MagneticKNNLocalizer:
    """
    基于欧氏距离的地磁指纹 KNN 定位器
    """
    def __init__(self, ref_feats: np.ndarray, ref_coords: np.ndarray,
                 k: int = 3, weighted: bool = True):
        """
        ref_feats:  (N, D)
        ref_coords: (N, 2)
        """
        self.ref_feats = np.asarray(ref_feats, dtype=np.float32)
        self.ref_coords = np.asarray(ref_coords, dtype=np.float32)
        self.k = int(k)
        self.weighted = bool(weighted)

        if len(self.ref_feats) != len(self.ref_coords):
            raise ValueError("ref_feats 和 ref_coords 数量不一致")

    def predict_one(self, x_query: np.ndarray) -> np.ndarray:
        """
        x_query: (L, 3)
        """
        q = np.asarray(x_query, dtype=np.float32).reshape(-1)
        dists = np.linalg.norm(self.ref_feats - q[None, :], axis=1)

        nn_idx = np.argsort(dists)[:self.k]
        nn_dists = dists[nn_idx]
        nn_coords = self.ref_coords[nn_idx]

        if self.weighted:
            return weighted_average_coords(nn_coords, nn_dists)
        return np.mean(nn_coords, axis=0).astype(np.float32)

    def predict_batch(self, x_batch: np.ndarray) -> np.ndarray:
        preds = [self.predict_one(x) for x in x_batch]
        return np.stack(preds, axis=0)