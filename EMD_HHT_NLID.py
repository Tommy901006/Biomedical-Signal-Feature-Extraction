import os
import numpy as np
import pandas as pd
from nlid_analysis import compute_nlid_from_signals, export_nlid_results_to_excel

# 設定資料夾路徑 (請替換成你的資料夾路徑)
folder_path = r"T"

# 確認資料夾是否存在
if not os.path.exists(folder_path):
    print("資料夾不存在！")
else:
    # 取得資料夾中的所有 Excel 檔案 (僅限 .xlsx)
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    
    # 用來儲存所有檔案 NLID 結果的字典 (每個檔案一行)
    all_nlid_data = {}
    
    # 逐一處理每個 Excel 檔案
    for file_name in excel_files:
        try:
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path)
            
            # 確保檔案中有 "X" 與 "Y" 欄位
            if "X" not in df.columns or "Y" not in df.columns:
                print(f"檔案 {file_name} 內缺少 'X' 或 'Y' 欄位，跳過此檔案。")
                continue
            
            # 取得 X 與 Y 信號
            signal_x = df["X"].values
            signal_y = df["Y"].values
            
            # 呼叫函式庫計算 NLID，參數可依需求調整
            # 設定 use_hilbert=True 表示先對 IMF 進行 Hilbert 變換
            # 若檔案很多建議將 plot_imfs_flag 設為 False，避免繪圖彈出視窗
            nlid_results = compute_nlid_from_signals(signal_x, signal_y,
                                                      m=2, tau=1, threshold=0.05,
                                                      use_hilbert=False, plot_imfs_flag=False)
            
            # 建立一個字典儲存該檔案的 NLID 結果，並加入檔名
            file_nlid_dict = {"File_Name": file_name}
            file_nlid_dict.update(nlid_results)
            all_nlid_data[file_name] = file_nlid_dict
            
        except Exception as e:
            print(f"處理檔案 {file_name} 時發生錯誤，跳過此檔案。錯誤訊息：{e}")
            continue

    # 將所有 NLID 結果轉換成 DataFrame 並匯出至 Excel
    nlid_df = pd.DataFrame.from_dict(all_nlid_data, orient='index')
    nlid_df.reset_index(drop=True, inplace=True)
    
    output_file = os.path.join(folder_path, "NLID_all_results.xlsx")
    nlid_df.to_excel(output_file, index=False)
    
    print(f"所有檔案的 NLID 結果已成功儲存到 {output_file}")
