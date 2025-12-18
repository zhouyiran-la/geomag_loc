import numpy as np
import pandas as pd
import os

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

def get_save_data_with_label(input_dir, output_dir, selected_cols=None, norm:bool=True):
    """
    保存位置打标之后的数据 (CSV)
    
    参数:
    ----------
    file_dir : str
        输出目录路径
    origin_data_loaded : dict[str, pd.DataFrame]
        原始数据字典 {key: DataFrame}
    selected_cols : list[str] | None
        若为 None → 保存全部列
        若指定 → 仅保存指定列
    
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
        origin_data_with_label = get_data_with_pos_label(value, norm=norm)
        
        # 如果用户指定了列，则仅保留这些列
        if selected_cols:
            # 检查列是否存在
            missing_cols = [col for col in selected_cols if col not in origin_data_with_label.columns]
            if missing_cols:
                raise ValueError(f"Data for '{key}' 缺少列: {missing_cols}")
            data_to_save = origin_data_with_label[selected_cols]
        else:
            # 未指定列 → 保存所有列
            data_to_save = origin_data_with_label
        # 如果位置标签归一化
        if norm:
            # 保存为 CSV 文件
            save_path = os.path.join(output_dir, f"data_with_label_{key}_normalize.csv")
        # 如果位置标签不归一化
        else:
            # 保存为 CSV 文件
            save_path = os.path.join(output_dir, f"data_with_label_{key}.csv")
        data_to_save.to_csv(save_path, index=False)
        saved_files.append(save_path)

    print(f"已保存 {len(saved_files)} 个 CSV 文件至: {output_dir}")
    return saved_files



if __name__ == "__main__":
    input_dir = "./data/origin/4.25数据/100/"
    output_dir = "./data/origin/4.25数据/100/com"
    get_save_data_with_label(input_dir, output_dir, norm=False)