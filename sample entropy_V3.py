import os
import pandas as pd
import numpy as np
import nolds
from tqdm import tqdm

def calculate_sample_entropy(data: np.ndarray, m: int):
    """
    计算 Sample Entropy。
    
    参数:
    data (np.ndarray): 时间序列数据。
    m (int): 嵌入维度。
    
    返回:
    float: Sample Entropy。
    """
    return nolds.sampen(data, emb_dim=m)

def process_file(file_path: str, m: int):
    """
    处理单个文件，计算 'X' 和 'Y' 列的 Sample Entropy。
    
    参数:
    file_path (str): 文件路径。
    m (int): 嵌入维度。
    
    返回:
    dict: 包含文件名和 Sample Entropy 的字典。
    """
    try:
        df = pd.read_excel(file_path)
        if 'X' in df.columns and 'Y' in df.columns:
            x_data = df['X'].values
            y_data = df['Y'].values

            x_sampen = calculate_sample_entropy(x_data, m)
            y_sampen = calculate_sample_entropy(y_data, m)

            return {
                'Filename': os.path.basename(file_path),
                'XSample Entropy': x_sampen,
                'YSample Entropy': y_sampen
            }
        else:
            print(f"Columns 'X' or 'Y' not found in {file_path}.")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main(folder_path: str, output_file: str, m: int):
    """
    主程序，遍历文件夹中的所有 Excel 文件，计算 Sample Entropy 并保存结果。
    
    参数:
    folder_path (str): 文件夹路径。
    output_file (str): 输出文件路径。
    m (int): 嵌入维度。
    """
    results = []
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    # 使用 tqdm 显示处理进度
    for file in tqdm(files, desc="Processing files"):
        result = process_file(file, m)
        if result:
            results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)
    print(f"All files processed. Results are saved to {output_file}.")

# 设定参数
# 設置資料夾路徑
folder_path = r'C:\Users\User\Desktop\T\eye\內差法後excel'  # 替換為你的資料夾路徑
output_file = r'C:\Users\User\Desktop\T\eye\Entropy\post-entropy_results.xlsx'  # 結果輸出文件
m = 1  # 嵌入维度

# 运行主程序
main(folder_path, output_file, m)
