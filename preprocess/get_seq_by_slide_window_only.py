import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
from pathlib import Path
from data_augmentation import DA_interp_Xinterp_Ylinear


WIN_SIZE = 300
STRIDE = 1

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

def build_sliding_window_sequences(data: np.ndarray, 
                                   pos_data: np.ndarray):
    """
    通用滑动窗口构造函数
    """
    N = len(data)
    X, y = [], []

    for start in range(0, N - WIN_SIZE + 1, STRIDE):
        end = start + WIN_SIZE
        seq = data[start:end]
        center = start + WIN_SIZE // 2
        label = pos_data[center]
        X.append(seq)
        y.append(label)

    return np.stack(X), np.stack(y)


def build_dataset(X_all, pos_data, mode=1):
    """
    根据滑动窗口构造训练数据集（统一滑窗后再拆分）

    Args:
        X_all: 需要进行滑窗的数据 nd.array
        pos_data: 位置坐标数据 nd.array
        window_size: 滑动窗口大小
        stride: 滑动步长
        mode: 0-地磁序列数据集 1-地磁-IMU序列数据集

    Returns:
        X_seq: (num_samples, window_size, 3 or 9)
        y: (num_samples, 2)
    """

    # === 4. 滑动窗口构造 ===
    X_seq, y = build_sliding_window_sequences(X_all, pos_data)
    # === 5. 拆分出不同模态 ===
    if mode == 0:
        print("正在构造地磁序列数据集")
        print(f"构造完成: {len(X_seq)} 个样本，每个样本长度 {WIN_SIZE}")
        return X_seq[:, :, :3], y
    else:
        print("正在构造地磁—IMU序列数据集")
        print(f"构造完成: {len(X_seq)} 个样本，每个样本长度 {WIN_SIZE}")
        return X_seq[:, :, 0:9], y


def save_mag_imu_seq_to_npz(input_file_path, output_file_dir, trans:bool=True, zscore:bool=True, aug:bool=False, mode=1):
    """
    保存地磁序列数据、惯导序列数据以及对应坐标到npz
    Args:
        intput_file_path: 输入文件路径
        output_file_path: 输出文件目录
        trans: 是否进行地磁坐标变换
        zscore: 是否进行Z-score标准化
        aug: 是否进行数据增强
        mode: 0-地磁序列数据集 1-地磁-IMU序列数据集

    """

    df = pd.read_csv(input_file_path)
    
    mag_cols = ['geomagneticx', 'geomagneticy', 'geomagneticz']
    imu_cols = ['accelerometerx', 'accelerometery', 'accelerometerz',
                'gyroscopex', 'gyroscopey', 'gyroscopez']
    gra_cols = ['gravityx', 'gravityy', 'gravityz']
    pos_cols = ['pos_x', 'pos_y']

    # === 2. 取出特征矩阵 ===
    feature_cols = mag_cols + imu_cols + gra_cols
    X_all = df[feature_cols].to_numpy()
    pos_data = df[pos_cols].to_numpy()

    # === 3. 地磁转换与标准化 ===
    mag_data = X_all[:, :3]
    gra_data = X_all[:, -3:]

    if trans:
        print("正在执行三轴地磁分量转换...")
        mag_data = geo_trans_fast(mag_data, gra_data)
        X_all[:, :3] = mag_data

    if zscore:
        print("正在执行Z-score标准化...")
        X_all[:, :3] = zscore_std(X_all[:, :3])

    if aug and mode == 0:
        print("正在执行数据增强...")
        mag_pos_data = np.hstack((mag_data, pos_data))
        # 数据增强后的
        aug_res = DA_interp_Xinterp_Ylinear(mag_pos_data)
        for index, item in enumerate(aug_res):
            item_X_all = item[:, :3]
            item_pos_data = item[:, 3:]
            X_mag, y = build_dataset(
                item_X_all, item_pos_data, 
            )
            output_file_path = output_file_dir + f"{mode}_" + Path(input_file_path).stem + f"_W_{WIN_SIZE}_S_{STRIDE}_AUG{index:02d}.npz"
            # 保存为npz方便后续加载
            np.savez(output_file_path, X_mag=X_mag, y=y)
            print(f"数据保存至:{output_file_path}")
    elif not aug and mode == 0:
        X_seq, y = build_dataset(
            X_all, pos_data, 
            mode=0           # 构造模式
        )
        output_file_path = output_file_dir + f"{mode}_"+ Path(input_file_path).stem + f"_W_{WIN_SIZE}_S_{STRIDE}.npz"
        # 保存为npz方便后续加载
        np.savez(output_file_path, X_mag=X_seq[:, :, :3], y=y)
        print(f"数据保存至:{output_file_path}")

    else:
        X_seq, y = build_dataset(
            X_all, pos_data, 
            mode=1            # 构造模式
        )
        output_file_path = output_file_dir + f"{mode}_"+ Path(input_file_path).stem + f"_W_{WIN_SIZE}_S_{STRIDE}.npz"
        # 保存为npz方便后续加载
        np.savez(output_file_path, X_mag=X_seq[:, :, :3], X_imu=X_seq[:, :, 3:9], y=y)
        print(f"数据保存至:{output_file_path}")


def save_each_sample_to_txt(X_mag, save_dir="./data/preprocessed/4.25数据/100/"):
    """
    将每个样本的地磁序列保存为独立的txt文件
    Args:
        X_mag: np.ndarray, shape (num_samples, window_size, 3)
        save_dir: 保存文件的目录
    """
    os.makedirs(save_dir, exist_ok=True)  # 若目录不存在则自动创建

    num_samples = X_mag.shape[0]
    for i in range(num_samples):
        file_path = os.path.join(save_dir, f"mag_sample_{i:04d}.txt")
        np.savetxt(file_path, X_mag[i], fmt="%.6f", delimiter=",", header="ms,mh,mv", comments="")
    
    print(f"✅ 已保存 {num_samples} 个样本到目录: {save_dir}")


def main():
    global WIN_SIZE
    global STRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--win', type=int, default=300, help='窗口大小')
    parser.add_argument('--stride', type=int, default=1, help='步长')
    parser.add_argument('--mode', type=int, default=0, help='处理模式')
    parser.add_argument('--aug', action='store_true', help='启用数据增强')
    
    args = parser.parse_args()
    cur_mode = args.mode
    cur_aug = args.aug
    input_file_dir = Path(args.input_dir)
    output_base_dir = args.output_dir
    ## 修改窗口大小
    WIN_SIZE = args.win
    ## 修改步长
    STRIDE = args.stride
    
    if(cur_mode == 0 and cur_aug):
        output_file_dir = output_base_dir + f"/seq_{WIN_SIZE}/mag_aug/"
    elif(cur_mode == 0 and not cur_aug):
        output_file_dir = output_base_dir + f"/seq_{WIN_SIZE}/mag/"
    else:
        output_file_dir = output_base_dir + f"/seq_{WIN_SIZE}/mag_imu/"


    # 检查输入目录
    if not input_file_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_file_dir}")
    
    input_files = list(input_file_dir.glob("*.csv"))
    Path(output_file_dir).mkdir(parents=True, exist_ok=True)
    print("=============================================")
    print(f"输入目录: {input_file_dir}")
    print(f"输出目录: {output_file_dir}")
    print(f"处理模式: {cur_mode}")
    print(f"数据增强: {cur_aug}")
    print(f"找到 {len(input_files)} 个CSV文件")
    print("=============================================")
    for input_file_path in input_files:
        save_mag_imu_seq_to_npz(input_file_path, output_file_dir, 
                               mode=cur_mode, aug=cur_aug)

if __name__ == '__main__':
    main()


# if __name__ == '__main__':
#     intput_file_dir = Path('./data/origin/4.25数据/100/norm')
#     intput_files = list(intput_file_dir.glob("*.csv"))
#     output_file_dir = './data/preprocessed/4.25数据/100/seq_300/'
#     cur_mode = 0
#     cur_aug = False
#     Path(output_file_dir).mkdir(parents=True, exist_ok=True)
#     for input_file_path in intput_files:
#         save_mag_imu_seq_to_npz(input_file_path, output_file_dir, mode=1)

#     # # 读取模式1的数据
#     # data = np.load('./data/preprocessed/4.25数据/100/seq_100/0_data_with_label_ghw匀速1_normalize_W_100_S_1_AUG00.npz')
#     # print(type(data))
#     # # 获取数据
#     # X_mag = data['X_mag']   # 地磁序列数据，形状为 [样本数, 序列长度, 3]
#     # # X_imu = data['X_imu']   # IMU序列数据，形状为 [样本数, 序列长度, 6]
#     # y = data['y']           # 坐标标签，形状为 [样本数, 2]

#     # print(f"地磁数据形状: {X_mag.shape}")
#     # # print(f"IMU数据形状: {X_imu.shape}")
#     # print(f"坐标标签形状: {y.shape}")

    