import os
import numpy as np
import pandas as pd
from scipy import signal

# 定義計算 coherence 的函式
def calculate_coherence(X, Y, fs=1000, nperseg=None):
    """
    使用 SciPy 的 signal.coherence 計算 X 與 Y 的頻譜 coherence，
    並回傳其平均值作為單一指標。
    """
    # 如果 nperseg 沒有給定，則預設使用 scipy 預設值
    f, Cxy = signal.coherence(X, Y, fs=fs, nperseg=nperseg)
    avg_coherence = np.mean(Cxy)
    return avg_coherence

# 設定資料夾路徑，請根據實際情況修改
folder_path = r"C:\Users\User\Desktop\eyetracker sample data\採樣結果\採樣後post"

# 用來存放結果的清單
results = []

# 遍歷資料夾中的所有 Excel 檔案
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)
        try:
            # 讀取 Excel 檔案
            df = pd.read_excel(file_path)
            # 檢查是否含有 'X' 與 'Y' 欄位
            if 'X' in df.columns and 'Y' in df.columns:
                coh = calculate_coherence(df['X'], df['Y'])
                results.append({'檔名': file, 'coherence': coh})
            else:
                print(f"檔案 {file} 中不含有 'X' 與 'Y' 欄位")
        except Exception as e:
            print(f"處理檔案 {file} 時發生錯誤: {e}")

# 將結果轉換為 DataFrame 並輸出為 Excel 檔案
results_df = pd.DataFrame(results)
results_df.to_excel("post_coherence_output.xlsx", index=False)

print("計算完成，結果已存入 coherence_output.xlsx")
