import numpy as np
import pandas as pd
from scipy import interpolate

def DA_interp_Xinterp_Ylinear(
    data,
    geomagnetic_dims=[0, 1, 2],
    coord_dims=[3, 4],
    scales=[0.8, 1, 1.5, 2],
    kind="quadratic"
):
    """
    功能: 对地磁与坐标数据进行插值增强。
    - 地磁数据使用 `kind` 指定的插值方法；
    - 坐标数据使用线性插值；
    - 每个 scale 的增强结果单独生成一个矩阵并返回。

    Args:
    ----------
    data : np.ndarray
        原始三维数据 (样本数 × 维度数)
        例如每一行包含 [geo_x, geo_y, geo_z, coord_x, coord_y]

    geomagnetic_dims : list[int]
        地磁数据对应的列索引（默认 [0, 1, 2]）

    coord_dims : list[int]
        坐标数据对应的列索引（默认 [3, 4]）

    scales : list[float]
        插值比例列表，例如 [1, 1.5, 2]

    kind : str
        地磁插值方法，可选 'linear'、'quadratic'、'cubic' 等

    返回:
    ----------
    results : list[np.ndarray]
        每个 scale 对应一个插值后的二维矩阵。
        例如 results[i].shape = (new_length_i, data.shape[1])
    """
    data = np.array(data)
    results = []

    warp_size = data.shape[0]  # 原始时间步长度
    window_steps = np.arange(warp_size)

    for scale in scales:
        new_length = int(np.ceil(warp_size * scale))
        x_new = np.linspace(0, warp_size - 1, num=new_length)

        matrix = np.zeros((new_length, data.shape[1]))

        for dim in range(data.shape[1]):
            y = data[:, dim]
            if dim in geomagnetic_dims:
                f = interpolate.interp1d(window_steps, y, kind=kind)
                y_new = f(x_new)
            elif dim in coord_dims:
                f = interpolate.interp1d(window_steps, y, kind='linear')
                y_new = f(x_new)
            else:
                # 其他维度不做插值，使用最近邻填充
                f = interpolate.interp1d(window_steps, y, kind='nearest')
                y_new = f(x_new)

            matrix[:, dim] = y_new

        results.append(matrix)

    return results