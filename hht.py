import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from pyhht.emd import EMD

# 讀取 Excel 檔案
file_path = "T\pre-p3.xlsx"  # 請替換成你的 Excel 檔案名稱
df = pd.read_excel(file_path)

# 確保 X 和 Y 欄位存在
if "X" not in df.columns or "Y" not in df.columns:
    raise ValueError("Excel 檔案內必須包含 'X' 和 'Y' 欄位！")

# 取得 X 和 Y 信號
signal_x = df["X"].values
signal_y = df["Y"].values
t = np.arange(len(signal_x))  # 假設時間軸是索引序列

def apply_hht(signal, label):
    # 進行 EMD 分解
    decomposer = EMD(signal)
    imfs = decomposer.decompose()

    # 繪製 IMF 分量
    fig, axs = plt.subplots(len(imfs) + 1, 1, figsize=(10, 8))
    axs[0].plot(t, signal, 'k', label=f"Original Signal ({label})")
    axs[0].legend()

    for i, imf in enumerate(imfs):
        axs[i + 1].plot(t, imf, label=f'IMF {i+1}')
        axs[i + 1].legend()

    plt.tight_layout()
    plt.show()

    # Hilbert 變換
    analytic_imfs = [hilbert(imf) for imf in imfs]
    instantaneous_frequencies = [np.diff(np.unwrap(np.angle(analytic_imf))) / (2.0 * np.pi * (t[1] - t[0])) for analytic_imf in analytic_imfs]

    # 繪製 Hilbert 頻譜
    plt.figure(figsize=(10, 6))
    for i, freq in enumerate(instantaneous_frequencies):
        plt.plot(t[1:], freq, label=f'IMF {i+1} Frequency')
    
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Hilbert-Huang Spectrum ({label})")
    plt.legend()
    plt.show()

# 對 X 和 Y 兩個信號分別執行 HHT
apply_hht(signal_x, "X")
apply_hht(signal_y, "Y")
