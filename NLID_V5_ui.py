# -*- coding: utf-8 -*-
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from NLIDOOP3 import RecurrenceAnalysis

def process_folder(folder_path):
    results = []
    num_processed = 0

    for filename in os.listdir(folder_path):
        filename_lower = filename.lower().strip()
        if filename_lower.endswith((".xlsx", ".xls")):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_excel(file_path)

                # 標準化欄位名稱（去除空白與大小寫差異）
                df.columns = df.columns.str.strip().str.upper()

                if not {'X', 'Y'}.issubset(df.columns):
                    print(f"跳過檔案（缺少 X 或 Y 欄位）: {filename}")
                    continue

                x = df['X'].dropna().values
                y = df['Y'].dropna().values

                min_len = min(len(x), len(y))
                if min_len < 1:
                    print(f"跳過檔案（資料不足 1 筆）: {filename}")
                    continue
                x = x[:min_len]
                y = y[:min_len]

                ra_x = RecurrenceAnalysis(x, m=3, tau=1)
                ps_x = ra_x.reconstruct_phase_space()

                ra_y = RecurrenceAnalysis(y, m=3, tau=1)
                ps_y = ra_y.reconstruct_phase_space()

                AR_X = RecurrenceAnalysis.compute_reconstruction_matrix(
                    ps_x, threshold=0.1, threshold_type="dynamic"
                )
                AR_Y = RecurrenceAnalysis.compute_reconstruction_matrix(
                    ps_y, threshold=0.1, threshold_type="dynamic"
                )

                NLID_XY, NLID_YX = RecurrenceAnalysis.calculate_nlid(AR_X, AR_Y)

                results.append({
                    "檔名": filename,
                    "NLID(X|Y)": NLID_XY,
                    "NLID(Y|X)": NLID_YX
                })
                num_processed += 1
                print(f"✅ 已處理：{filename}")

            except Exception as e:
                print(f"❌ 錯誤處理檔案 {filename}: {e}")

    if results:
        result_df = pd.DataFrame(results)
        output_path = os.path.join(folder_path, "NLID_Results.xlsx")
        result_df.to_excel(output_path, index=False)
        messagebox.showinfo("完成", f"{num_processed} 筆資料處理完成，結果儲存於：\n{output_path}")
    else:
        messagebox.showwarning("未找到資料", "沒有合適的 Excel 檔案可供處理。\n請確認包含 X 與 Y 欄位。")

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path)

# 建立 GUI
root = tk.Tk()
root.title("NLID 批次分析工具（僅限 Excel）")
root.geometry("400x200")

label = tk.Label(root, text="請選擇包含 Excel 檔的資料夾", font=("Arial", 14))
label.pack(pady=20)

button = tk.Button(root, text="選擇資料夾並開始分析", command=select_folder, font=("Arial", 12), bg="#4CAF50", fg="white")
button.pack(pady=10)

root.mainloop()
