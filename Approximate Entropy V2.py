import os
import pandas as pd
import numpy as np
import EntropyHub as EH
from tqdm import tqdm  # 引入進度條庫

# 定義自訂的 Approximate Entropy 函數
def ApEn(Datalist, r=0.2, m=2):
    th = r * np.std(Datalist)
    return EH.ApEn(Datalist, m, r=th)[0][-1]

# 設定資料夾路徑
folder_path = r'C:\Users\User\Desktop\T\eye\內差法後excel'

# 初始化結果列表
results = []

# 讀取資料夾中的所有 Excel 檔案
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
for filename in tqdm(file_list, desc="Processing files", unit="file"):
    try:
        file_path = os.path.join(folder_path, filename)
        data = pd.read_excel(file_path)
        
        # 確保數據是一維數組
        x_data = data['X'].values.flatten()
        y_data = data['Y'].values.flatten()
        
        # 計算 X 和 Y 軸的 Approximate Entropy
        x_entropy = ApEn(x_data)
        y_entropy = ApEn(y_data)
        
        # 將結果添加到列表中
        results.append({
            'File Name': filename,
            'XApproximate Entropy': x_entropy,
            'YApproximate Entropy': y_entropy
        })
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        continue

# 將結果轉換為 DataFrame
results_df = pd.DataFrame(results)

# 將結果輸出到 Excel 文件
output_file = 'approximate_entropy_results.xlsx'
results_df.to_excel(output_file, index=False)

print(f'Results saved to {output_file}')
