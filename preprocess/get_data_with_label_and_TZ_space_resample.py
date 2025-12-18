import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

PATH_BORDER = np.array([[0, 0], [0, 45.415], [38.4, 45.415], [38.4, 0], [0, 0]])

def npyload(filename):
    """
    功能：读取npy文件
    :param filename:文件路径
    :return: npy文件存储内容
    """
    print('read file: %s' % (filename))
    return np.load(filename, allow_pickle=True).item()

def label_data(endpoints, num):
    """
    功能：打标签
    :param lst:一条路径的两个端点
    :param num: 要切分的数量
    :return: 切分后的路径标签
    """
    
    x_lst = np.linspace(endpoints[0, 0], endpoints[1, 0], num)
    y_lst = np.linspace(endpoints[0, 1], endpoints[1, 1], num)
    
    return x_lst, y_lst

def pos_normalize(pos_x, pos_y):
    '''
    功能:坐标数据归一化
    :param data: 坐标数据
    :return: 归一化后的坐标数据
    '''
    x_min = np.min(pos_x)
    x_max = np.max(pos_x)
    y_min = np.min(pos_y)
    y_max = np.max(pos_y)
    x_length = x_max-x_min
    y_length = y_max-y_min
    if x_length == 0 and x_max == 1:
        pos_y = (pos_y - y_min) / y_length

    elif x_length == 0 and x_max > 1:
        pos_x = (pos_x) / x_max
        pos_y = (pos_y - y_min) / y_length

    elif y_length == 0 and y_max == 1:
        pos_x = (pos_x - x_min) / x_length

    elif y_length == 0 and y_max > 1:
        pos_x = (pos_x - x_min) / x_length
        pos_y = (pos_y) / y_max

    else:
        pos_x = (pos_x - x_min) / x_length
        pos_y = (pos_y - y_min) / y_length
    return pos_x, pos_y

def get_data_with_pos_label(origin_data: pd.DataFrame, norm:bool=True) -> pd.DataFrame:
    """
    给原始数据添加位置坐标（线性插值）
    Args:
        origin_data: 原始单次采集数据
        normalize: 是否进行位置坐标归一化
    Returns:
        
    """
    pathid = origin_data.loc[:, ['pathid']].values.astype(int)
    max_pathid = np.max(pathid)
    x_list = []
    y_list = []
    for j in range(0, max_pathid + 1):
        # 获取pathid=j的所有行索引
        path_id_row_index = np.where(pathid == j)[0]
        length = len(path_id_row_index)
        endpoints = PATH_BORDER[j:j+2, :]
        x_arr_path_id, y_arr_path_id = label_data(endpoints, length)
        x_list.append(x_arr_path_id)
        y_list.append(y_arr_path_id)
    pos_x = np.concatenate(x_list)
    pos_y = np.concatenate(y_list)
    if(norm):
        pos_x, pos_y = pos_normalize(pos_x, pos_y)
    origin_data['pos_x'] = pos_x
    origin_data['pos_y'] = pos_y
    return origin_data

def geo_trans_fast(mag_data, gra_data):
    """
    批量生成转换后的地磁分量数据
    :param mag_data 单次采集的所有三轴地磁数据(N, 3)
    :param gra_data 单次采集的所有三轴重力加速度数据(N, 3)
    :return 转换后的三轴地磁数据(N, 3)
    """
    ms = np.linalg.norm(mag_data, axis=1)
    gra_norm = np.linalg.norm(gra_data, axis=1)
    dot = np.einsum('ij,ij->i', mag_data, gra_data)  # 高效点积
    mv = np.abs(dot / gra_norm)
    mh = np.sqrt(ms**2 - mv**2)
    return np.column_stack((ms, mh, mv))

def zscore_std(mag_data):
    """
    对地磁要素进行Z-score标准化
    :param mag_data:传进来的经过处理数据:第1，2，3列代表ms,mh,mv
    :return: z_data :Zscore后的地磁数据
    """
    mean = np.mean(mag_data, axis=0)
    standrad = np.sqrt(np.var(mag_data, axis=0))
    z_data = np.divide((mag_data - mean), standrad)
    return z_data

def resample_bins(mag_pos_data, bin_size=0.2, samples_per_bin=5):
    """
    对一条路径按空间区间进行重采样：
      - 每 bin_size 米一个空间段
      - 每段内强制保持 samples_per_bin 个点

    参数:
        mag_pos_data: nd.array，必须有 geomagneticx,y,z pos_x, pos_y, 
        bin_size: 空间区间长度（例如 0.2米）
        samples_per_bin: 每段需要的采样点数（例如 10）

    返回:
        new_df: 重采样后的 DataFrame
    """

    # 取出坐标
    xs = mag_pos_data[:, 3]
    ys = mag_pos_data[:, 4]
    # 取出三轴地磁值
    mag_x = mag_pos_data[:, 0]
    mag_y = mag_pos_data[:, 1]
    mag_z = mag_pos_data[:, 2]
    # 计算每个点的累积距离
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx**2 + dy**2)
    s_orig = np.insert(np.cumsum(ds), 0, 0.0)

    total_len = s_orig[-1]

    # 构建每个 bin 的区间边界
    bin_edges = np.arange(0, total_len + bin_size, bin_size)

    # 输出数据
    all_new_points = []

    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        # 找出落入该区间的原始点
        mask = (s_orig >= left) & (s_orig < right)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            # 该 bin 没有点 → 用左右边界插值补点
            s_bin = np.linspace(left, right, samples_per_bin)
            x_new = np.interp(s_bin, s_orig, xs)
            y_new = np.interp(s_bin, s_orig, ys)
            magx_new = np.interp(s_bin, s_orig, mag_x)
            magy_new = np.interp(s_bin, s_orig, mag_y)
            magz_new = np.interp(s_bin, s_orig, mag_z)
        else:
            # 该 bin 有点 → 取出数据
            s_seg = s_orig[idx]
            x_seg = xs[idx]
            y_seg = ys[idx]
            magx_seg = mag_x[idx]
            magy_seg = mag_y[idx]
            magz_seg = mag_z[idx]

            if len(idx) >= samples_per_bin:
                # 点很多 → 均匀下采样
                indices = np.linspace(0, len(idx)-1, samples_per_bin).astype(int)
                s_bin = s_seg[indices]
                x_new = x_seg[indices]
                y_new = y_seg[indices]
                magx_new = magx_seg[indices]
                magy_new = magy_seg[indices]
                magz_new = magz_seg[indices]
            else:
                # 点不足 → 插值补成 samples_per_bin 个点
                s_bin = np.linspace(s_seg.min(), s_seg.max(), samples_per_bin)
                x_new = np.interp(s_bin, s_seg, x_seg)
                y_new = np.interp(s_bin, s_seg, y_seg)
                magx_new = np.interp(s_bin, s_seg, magx_seg)
                magy_new = np.interp(s_bin, s_seg, magy_seg)
                magz_new = np.interp(s_bin, s_seg, magz_seg)

        # 记录到输出
        for k in range(samples_per_bin):
            all_new_points.append([
                magx_new[k], magy_new[k], magz_new[k], x_new[k], y_new[k], 
            ])
   

    return all_new_points

def visualize_resampling(df_raw, df_resampled, out_path:Path):
    """
    对比可视化：原始地磁 vs 重采样后地磁
    
    df_raw: 原始数据（不均匀采样）
    df_resampled: 重采样后的数据（空间均匀）
    """

    # --- 原始数据 ---

    # 坐标
    xs_raw = df_raw["pos_x"].to_numpy()
    ys_raw = df_raw["pos_y"].to_numpy()

    # 计算原始累计距离 s_raw
    dx_raw = np.diff(xs_raw)
    dy_raw = np.diff(ys_raw)
    ds_raw = np.sqrt(dx_raw**2 + dy_raw**2)
    s_raw = np.insert(np.cumsum(ds_raw), 0, 0.0)

    # 地磁（X 分量）
    magx_raw = df_raw["geomagneticx"].to_numpy()



    # --- 重采样后的数据 ---

    xs_new = df_resampled["pos_x"].to_numpy()
    ys_new = df_resampled["pos_y"].to_numpy()

    dx_new = np.diff(xs_new)
    dy_new = np.diff(ys_new)
    ds_new = np.sqrt(dx_new**2 + dy_new**2)
    s_new = np.insert(np.cumsum(ds_new), 0, 0.0)

    magx_new = df_resampled["geomagneticx"].to_numpy()



    # --- 绘图 ---
    plt.figure(figsize=(15, 5))

    # 原始数据（不均匀采样）
    plt.plot(s_raw, magx_raw, label="Original (raw samples)", 
             marker="o", markersize=3, alpha=0.6)

    # 重采样（空间均匀）
    plt.plot(s_new, magx_new, label="Resampled (uniform space)", 
             marker="x", markersize=4, alpha=0.8)
    plt.xlabel("Distance along path (m)", fontsize=14)
    plt.ylabel("Geomagnetic X value", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)

def get_save_data_with_label_and_resample(input_dir, output_dir, trans:bool=True, zscore:bool=True):
    """
    保存位置打标之后的数据 (CSV)
    
    参数:
    ----------
    file_dir : str
        输出目录路径
    origin_data_loaded : dict[str, pd.DataFrame]
        原始数据字典 {key: DataFrame}
    trans : bool
        是否进行地磁坐标变换
    zscore : bool
    是否进行Z-score标准化
    
    返回:
    ----------
    saved_files : list[str]
        返回保存的 CSV 文件路径列表
    """

    origin_file_path = input_dir + "data.npy"
    origin_data_loaded = npyload(origin_file_path)

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for key, value in origin_data_loaded.items():
        # 获取带位置标签的数据
        origin_data_with_label = get_data_with_pos_label(value, norm=False)
        mag_cols = ['geomagneticx', 'geomagneticy', 'geomagneticz']
        imu_cols = ['accelerometerx', 'accelerometery', 'accelerometerz',
                    'gyroscopex', 'gyroscopey', 'gyroscopez']
        gra_cols = ['gravityx', 'gravityy', 'gravityz']
        pos_cols = ['pos_x', 'pos_y']

        # === 2. 取出特征矩阵 ===
        feature_cols = mag_cols + imu_cols + gra_cols
        # 防止传入的 DataFrame 含有字符串/空值导致 np.linalg.norm 报错，统一转为浮点型
        X_all = origin_data_with_label[feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        pos_data = origin_data_with_label[pos_cols].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

        # === 3. 地磁转换与标准化 ===
        mag_data = X_all[:, :3]
        gra_data = X_all[:, -3:]
        
        ## 初始化文件名
        file_name = f"data_with_label_{key}"
        if trans:
            print("正在执行三轴地磁分量转换...")
            mag_data = geo_trans_fast(mag_data, gra_data)
            file_name += "_T"
        if zscore:
            print("正在执行Z-score标准化...")
            mag_data = zscore_std(mag_data)
            file_name += "Z"
    
        mag_pos_data = np.hstack((mag_data, pos_data))
        
        resampled_data = resample_bins(mag_pos_data)
        # 构建 DataFrame
        res_df = pd.DataFrame(
            resampled_data,
            columns=["geomagneticx", "geomagneticy", "geomagneticz", "pos_x", "pos_y"]
        )

        # raw_path = Path(input_dir) / "com" / f"data_with_label_{key}.csv"
        # raw_df = pd.read_csv(raw_path)
        # fig_out_path = Path("plot") / "resample" / f"data_with_label_{key}.png"
        # visualize_resampling(raw_df, res_df, fig_out_path)
        file_name += "_resample.csv"

        save_path = os.path.join(output_dir, file_name)
        res_df.to_csv(save_path, index=False)
        saved_files.append(save_path)

    print(f"已保存 {len(saved_files)} 个 CSV 文件至: {output_dir}")
    return saved_files
    


if __name__ == "__main__":

    input_dir = "./data/origin/4.26数据/50/"
    output_dir = "./data/origin/4.26数据/50/resample"
    get_save_data_with_label_and_resample(input_dir, output_dir, trans=True, zscore=False)
