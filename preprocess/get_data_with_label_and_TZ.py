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
            file_name += "_Z"
        file_name += ".csv"
        origin_data_with_label[mag_cols] = mag_data
        save_path = os.path.join(output_dir, file_name)
        origin_data_with_label.to_csv(save_path, index=False)
        saved_files.append(save_path)

    print(f"已保存 {len(saved_files)} 个 CSV 文件至: {output_dir}")
    return saved_files
    


if __name__ == "__main__":

    input_dir = "./data/origin/6.03数据/训练数据/"
    output_dir = "./data/origin/6.03数据/训练数据/TZ"
    get_save_data_with_label_and_resample(input_dir, output_dir, trans=True, zscore=False)
