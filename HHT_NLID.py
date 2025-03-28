import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from pyhht.emd import EMD
from NLIDOOP3 import RecurrenceAnalysis  # 確保這個模組存在

# 設定資料夾路徑
folder_path = r"F:\ts\T\t"  # 請替換成你的資料夾路徑

# 確認資料夾是否存在
if not os.path.exists(folder_path):
    print("資料夾不存在！")
else:
    # 取得資料夾中的所有 Excel 檔案 (篩選 .xlsx)
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # 用來儲存最終「一檔一行」的結果：key=檔名, value=字典(包含該檔案各 IMF 分析結果)
    all_nlid_data = {}

    # 對每個 Excel 檔案進行處理
    for file_name in excel_files:
        # 以 try-except 包覆，若中途發生錯誤就跳過
        try:
            file_path = os.path.join(folder_path, file_name)
            
            # 讀取 Excel 檔案
            df = pd.read_excel(file_path)

            # 確保 X 和 Y 欄位存在
            if "X" not in df.columns or "Y" not in df.columns:
                print(f"檔案 {file_name} 內缺少 'X' 或 'Y' 欄位，跳過此檔案。")
                continue

            # 取得 X 和 Y 信號
            signal_x = df["X"].values
            signal_y = df["Y"].values
            
            # 建立用來存放該檔案所有 IMF 計算結果的字典
            file_nlid_dict = {"File_Name": file_name}

            # 定義一個執行 HHT (EMD) 的函式
            def apply_hht(sig_x, sig_y):
                decomposer_x = EMD(sig_x)
                imfs_x = decomposer_x.decompose()

                decomposer_y = EMD(sig_y)
                imfs_y = decomposer_y.decompose()

                return imfs_x, imfs_y

            # 執行 HHT 分析並獲取所有 IMF 分量
            imfs_x, imfs_y = apply_hht(signal_x, signal_y)

            # -------------------- Recurrence Analysis --------------------
            num_imfs = min(len(imfs_x), len(imfs_y))  # 可配對的 IMF 數量

            for i in range(num_imfs):
                # 使用第 i 個 IMF 進行分析
                x_imf = imfs_x[i]
                y_imf = imfs_y[i]

                # 創建 RecurrenceAnalysis 對象並重構相空間
                ra_x = RecurrenceAnalysis(x_imf, m=3, tau=1)
                phase_space_x = ra_x.reconstruct_phase_space()

                ra_y = RecurrenceAnalysis(y_imf, m=3, tau=1)
                phase_space_y = ra_y.reconstruct_phase_space()

                # 計算重建矩陣（動態閾值）
                AR_HR_BW = RecurrenceAnalysis.compute_reconstruction_matrix(
                    phase_space_x, threshold=0.1, threshold_type="dynamic"
                )
                AR_RP_BW = RecurrenceAnalysis.compute_reconstruction_matrix(
                    phase_space_y, threshold=0.1, threshold_type="dynamic"
                )

                # 計算 NLID
                NLID_XY_avg, NLID_YX_avg = RecurrenceAnalysis.calculate_nlid(AR_HR_BW, AR_RP_BW)

                # 將第 i 個 IMF 的結果存到字典裡
                file_nlid_dict[f"{i+1}_NLID(X|Y)"] = NLID_XY_avg
                file_nlid_dict[f"{i+1}_NLID(Y|X)"] = NLID_YX_avg

            # 迴圈處理完該檔案所有 IMF 後，將此字典放入 all_nlid_data
            all_nlid_data[file_name] = file_nlid_dict

        except Exception as e:
            print(f"處理檔案 {file_name} 時發生錯誤，跳過該檔案。錯誤訊息：{e}")
            continue  # 直接跳到下一個檔案

    # 將所有結果儲存到 DataFrame
    nlid_df = pd.DataFrame.from_dict(all_nlid_data, orient='index')
    nlid_df.reset_index(drop=True, inplace=True)

    # 設定輸出檔案名稱
    output_file = os.path.join(folder_path, "NLID_all_results.xlsx")

    # 儲存為 Excel 檔案
    nlid_df.to_excel(output_file, index=False)

    print(f"所有檔案的 NLID 結果已成功儲存到 {output_file}")
