import os
import numpy as np
import csv

# 设置包含 .npy 文件的文件夹路径
folder_path = "/mnt/d/our/data/urbanbis/urbanbis_data"  # 替换为你实际的文件夹路径

# 打开 CSV 文件进行写入
csv_filename = "/mnt/d/our/new_cls.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # 写入 CSV 表头
    header = ['Prefix', 'x', 'y', 'z', 'r', 'g', 'b', 'sem', 'ins']
    csv_writer.writerow(header)
    
    # 遍历文件夹中的所有 .npy 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('_aligned_bbox.npy'):  # 只处理以 '_vert.npy' 结尾的文件
            # 获取文件的完整路径
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 加载 .npy 文件中的数据
                data = np.load(file_path)
                
                # 提取文件名前缀（例如 'Lihu_Area9'，从文件名中去掉 '_vert.npy' 部分）
                prefix = filename.replace('_aligned_bbox.npy', '')
                
                # 将数据写入 CSV 文件
                for row in data:
                    # 将文件名前缀与每行数据拼接
                    csv_writer.writerow([prefix] + row.tolist())
                print(f"已处理文件: {filename}")
            
            except Exception as e:
                print(f"加载文件 {filename} 时出错: {e}")

print(f"数据已保存到 {csv_filename}")

