import numpy as np
import os

def calculate_rgb_mean(file_path):
    """
    计算单个 npy 文件中点云 RGB 的总和及点数
    :param file_path: npy 文件路径
    :return: (sum_r, sum_g, sum_b, point_count)
    """
    # 加载 npy 文件
    data = np.load(file_path)
    
    # 假设 RGB 信息在点云数据的 3:6 列
    rgb = data[:, 3:6]  # 提取 RGB 列 (N, 3)
    
    # 计算每个通道的总和
    sum_r = np.sum(rgb[:, 0])
    sum_g = np.sum(rgb[:, 1])
    sum_b = np.sum(rgb[:, 2])
    
    # 返回 RGB 总和及点数
    return sum_r, sum_g, sum_b, rgb.shape[0]

def process_npy_files(directory):
    """
    遍历指定目录中的所有 npy 文件，计算所有文件的 RGB 总体均值
    :param directory: 存放 npy 文件的目录路径
    :return: 总体 RGB 均值
    """
    total_r, total_g, total_b = 0, 0, 0
    total_points = 0  # 总点数

    # 遍历目录中的文件
    for file_name in os.listdir(directory):
        if file_name.endswith('_vert.npy'):  # 只处理符合条件的 .npy 文件
            file_path = os.path.join(directory, file_name)
            try:
                # 计算单个文件的 RGB 总和及点数
                sum_r, sum_g, sum_b, point_count = calculate_rgb_mean(file_path)
                
                # 累加总和和点数
                total_r += sum_r
                total_g += sum_g
                total_b += sum_b
                total_points += point_count

                print(f"Processed {file_name}: Points = {point_count}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    # 计算总体均值
    if total_points > 0:
        mean_r = total_r / total_points
        mean_g = total_g / total_points
        mean_b = total_b / total_points
        return mean_r, mean_g, mean_b
    else:
        print("No valid files were found.")
        return None

# 主程序
if __name__ == "__main__":
    # 指定包含 npy 文件的目录路径
    directory = "/mnt/d/our/data/urbanbis/urbanbis_data"  # 修改为你的实际目录路径

    # 处理文件并获取总体均值
    overall_mean_rgb = process_npy_files(directory)
    
    if overall_mean_rgb:
        print("\nOverall Mean RGB Value:")
        print(f"Mean RGB = ({overall_mean_rgb[0]:.4f}, {overall_mean_rgb[1]:.4f}, {overall_mean_rgb[2]:.4f})")



