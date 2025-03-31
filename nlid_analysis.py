import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert
from pyhht.emd import EMD
from NLIDOOP3 import RecurrenceAnalysis  # 確保此模組存在

def compute_imfs(signal, use_hilbert=True):
    """
    對輸入信號進行 EMD 分解，並依據參數決定是否使用 Hilbert 變換
    
    參數:
        signal: 一維 numpy 陣列，原始信號
        use_hilbert: 布林值，是否使用 Hilbert 變換 (預設 True)
    
    回傳:
        imfs: 若 use_hilbert 為 True，回傳解析 IMF 的列表（每個元素為複數型態）
              否則直接回傳 EMD 分解得到的 IMF (實數陣列)
    """
    imfs = EMD(signal).decompose()
    if use_hilbert:
        imfs = [hilbert(imf) for imf in imfs]
    return imfs

def plot_imfs(imfs, title_prefix="IMF", use_hilbert=True):
    """
    繪製 IMF 圖形，若使用 Hilbert 變換則同時顯示實部與虛部，
    否則只繪製 IMF 曲線
    
    參數:
        imfs: IMF 列表
        title_prefix: 每個圖形標題前綴字串
        use_hilbert: 是否使用 Hilbert 變換的 IMF (預設 True)
    """
    for i, imf in enumerate(imfs):
        plt.figure(figsize=(10, 4))
        if use_hilbert:
            plt.plot(np.real(imf), label='Real Part')
            plt.plot(np.imag(imf), label='Imaginary Part', linestyle='--')
        else:
            plt.plot(imf, label='IMF')
        plt.title(f"{title_prefix} - IMF {i+1}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

def compute_nlid_from_signals(signal_x, signal_y, m=2, tau=1, threshold=0.05, use_hilbert=True, plot_imfs_flag=False):
    """
    給定 X 與 Y 信號，依據參數決定是否先使用 Hilbert 變換計算解析 IMF，
    並依據 IMF 計算 NLID。
    
    參數:
        signal_x: 一維 numpy 陣列，X 信號
        signal_y: 一維 numpy 陣列，Y 信號
        m: 相空間重構維度（預設為 2）
        tau: 重構延遲（預設為 1）
        threshold: 重構矩陣計算時使用的動態門檻值（預設 0.05）
        use_hilbert: 是否使用 Hilbert 變換 (預設 True)
        plot_imfs_flag: 是否繪製 IMF 圖形 (預設 False)
    
    回傳:
        nlid_results: 字典，包含每個 IMF 的 NLID 結果，
            格式為 { "1_NLID(X|Y)": value, "1_NLID(Y|X)": value, ... }
    """
    imfs_x = compute_imfs(signal_x, use_hilbert=use_hilbert)
    imfs_y = compute_imfs(signal_y, use_hilbert=use_hilbert)

    if plot_imfs_flag:
        plot_imfs(imfs_x, title_prefix="X IMF", use_hilbert=use_hilbert)
        plot_imfs(imfs_y, title_prefix="Y IMF", use_hilbert=use_hilbert)

    nlid_results = {}
    # 若使用 Hilbert，假設取解析信號的實部；否則直接使用 IMF
    num_imfs = min(len(imfs_x), len(imfs_y))
    for i in range(num_imfs):
        if use_hilbert:
            x_signal = np.real(imfs_x[i])
            y_signal = np.real(imfs_y[i])
        else:
            x_signal = imfs_x[i]
            y_signal = imfs_y[i]

        # 利用 RecurrenceAnalysis 進行相空間重構
        ra_x = RecurrenceAnalysis(x_signal, m=m, tau=tau)
        phase_space_x = ra_x.reconstruct_phase_space()

        ra_y = RecurrenceAnalysis(y_signal, m=m, tau=tau)
        phase_space_y = ra_y.reconstruct_phase_space()

        # 計算重構矩陣 (動態門檻)
        AR_HR_BW = RecurrenceAnalysis.compute_reconstruction_matrix(
            phase_space_x, threshold=threshold, threshold_type="dynamic"
        )
        AR_RP_BW = RecurrenceAnalysis.compute_reconstruction_matrix(
            phase_space_y, threshold=threshold, threshold_type="dynamic"
        )

        # 計算 NLID (雙向：X->Y 與 Y->X)
        NLID_XY_avg, NLID_YX_avg = RecurrenceAnalysis.calculate_nlid(AR_HR_BW, AR_RP_BW)

        nlid_results[f"{i+1}_NLID(X|Y)"] = NLID_XY_avg
        nlid_results[f"{i+1}_NLID(Y|X)"] = NLID_YX_avg

    return nlid_results

def export_nlid_results_to_excel(nlid_results, output_file):
    """
    將 NLID 結果字典匯出至 Excel 檔案

    參數:
        nlid_results: 字典，包含 NLID 分析結果
        output_file: 匯出檔案完整路徑 (例如: r"C:\path\to\output.xlsx")
    """
    nlid_df = pd.DataFrame([nlid_results])
    nlid_df.to_excel(output_file, index=False)
    print(f"NLID 結果已成功匯出至 {output_file}")


