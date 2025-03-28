 
import os
import numpy as np
import pandas as pd

# 定義計算 Pearson 相關係數的函式
def Pearson_correlation(X, Y):
    if len(X) == len(Y):
        Sum_xy = sum((X - X.mean()) * (Y - Y.mean()))
        Sum_x_squared = sum((X - X.mean())**2)
        Sum_y_squared = sum((Y - Y.mean())**2)
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
        return corr
    else:
        raise ValueError("X 與 Y 的長度不相等")

# 設定資料夾路徑，請自行更新
folder_path = r"C:\Users\User\Desktop\eyetracker sample data\採樣結果\採樣後post"

# 用來存放結果的清單
results = []

# 遍歷資料夾中的所有檔案
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)
        try:
            # 讀取 Excel 檔案
            df = pd.read_excel(file_path)
            # 檢查是否包含 'X' 與 'Y' 欄位
            if 'X' in df.columns and 'Y' in df.columns:
                correlation = Pearson_correlation(df['X'], df['Y'])
                results.append({'檔名': file, 'correlation': correlation})
            else:
                print(f"檔案 {file} 中不含有 'X' 與 'Y' 欄位")
        except Exception as e:
            print(f"處理檔案 {file} 時發生錯誤: {e}")

# 將結果轉換成 DataFrame 並輸出到 Excel 檔案
results_df = pd.DataFrame(results)
results_df.to_excel("post-correlations_output.xlsx", index=False)

print("計算完成，結果已存入 correlations_output.xlsx")
