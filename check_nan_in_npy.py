import numpy as np
import os

def check_nan_in_npy(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            try:
                # 加载.npy文件
                data = np.load(filepath)
                # 检查是否存在NaN值
                if np.isnan(data).any():
                    print(f"文件 {filename} 包含NaN值1111111111111111111111111111111111111111111。")
                else:
                    print(f"文件 {filename} 不包含NaN值。")
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

# 将'your_directory_path'替换为你的文件夹路径
directory_path = 'Submit_result/test_predictions'
check_nan_in_npy(directory_path)