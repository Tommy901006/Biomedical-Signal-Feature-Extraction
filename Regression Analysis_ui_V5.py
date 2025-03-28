import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import itertools

class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Model Evaluation")
        self.root.geometry("700x700")
        self.root.configure(bg="#f0f0f0")

        self.file_path = ""
        self.dataset = None
        self.input_columns = []
        self.target_columns = []
        # 用來儲存訓練後的預測資料，方便後續輸出圖片
        self.plot_data = []

        # Title
        title_label = tk.Label(root, text="Regression Model Evaluation", font=("Arial", 24, "bold"), fg="#333", bg="#f0f0f0")
        title_label.pack(pady=20)

        # File selection
        file_frame = tk.Frame(root, bg="#f0f0f0")
        file_frame.pack(pady=10, fill="x", padx=20)
        file_label = tk.Label(file_frame, text="CSV File:", font=("Arial", 12), bg="#f0f0f0")
        file_label.pack(side="left", padx=10)
        self.file_entry = tk.Entry(file_frame, width=50, font=("Arial", 12))
        self.file_entry.pack(side="left", padx=10)
        file_button = tk.Button(file_frame, text="Select", command=self.select_file, bg="#4caf50", fg="#fff", font=("Arial", 12))
        file_button.pack(side="left", padx=10)

        # Input columns list
        input_frame = tk.Frame(root, bg="#f0f0f0")
        input_frame.pack(pady=10, fill="x", padx=20)
        input_label = tk.Label(input_frame, text="Select Input Columns (Multiple):", font=("Arial", 12), bg="#f0f0f0")
        input_label.pack(side="left", padx=10)
        self.input_listbox = tk.Listbox(input_frame, selectmode="multiple", width=50, height=6, font=("Arial", 12))
        self.input_listbox.pack(side="left", padx=10)
        input_button = tk.Button(input_frame, text="Confirm", command=self.confirm_input_columns, bg="#4caf50", fg="#fff", font=("Arial", 12))
        input_button.pack(side="left", padx=10)

        # Target columns list
        target_frame = tk.Frame(root, bg="#f0f0f0")
        target_frame.pack(pady=10, fill="x", padx=20)
        target_label = tk.Label(target_frame, text="Select Target Columns:", font=("Arial", 12), bg="#f0f0f0")
        target_label.pack(side="left", padx=10)
        self.target_listbox = tk.Listbox(target_frame, selectmode="multiple", width=50, height=6, font=("Arial", 12))
        self.target_listbox.pack(side="left", padx=10)
        target_button = tk.Button(target_frame, text="Confirm", command=self.confirm_target_columns, bg="#4caf50", fg="#fff", font=("Arial", 12))
        target_button.pack(side="left", padx=10)

        # Button section
        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=20)
        train_button = tk.Button(button_frame, text="Train Model", command=self.train_models, bg="#4caf50", fg="#fff", font=("Arial", 14, "bold"))
        train_button.pack(side="left", padx=10)

        # Result display
        self.result_text = scrolledtext.ScrolledText(root, height=15, font=("Arial", 12))
        self.result_text.pack(fill="both", expand=True, padx=20, pady=20)

        # 建立 plots 資料夾
        if not os.path.exists("plots"):
            os.makedirs("plots")

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, self.file_path)
            self.load_columns()

    def load_columns(self):
        self.dataset = pd.read_csv(self.file_path)
        columns = self.dataset.columns.tolist()
        self.input_listbox.delete(0, tk.END)
        self.target_listbox.delete(0, tk.END)
        for col in columns:
            self.input_listbox.insert(tk.END, col)
            self.target_listbox.insert(tk.END, col)

    def confirm_input_columns(self):
        self.input_columns = self.get_selected_columns(self.input_listbox)
        if not self.input_columns:
            messagebox.showerror("Error", "Please select at least one input column.")
        else:
            messagebox.showinfo("Confirmed", f"Input columns selected: {', '.join(self.input_columns)}")

    def confirm_target_columns(self):
        self.target_columns = self.get_selected_columns(self.target_listbox)
        if not self.target_columns:
            messagebox.showerror("Error", "Please select at least one target column.")
        else:
            messagebox.showinfo("Confirmed", f"Target columns selected: {', '.join(self.target_columns)}")

    def get_selected_columns(self, listbox):
        selected_indices = listbox.curselection()
        return [listbox.get(i) for i in selected_indices]

    def augment_data(self, dataset, input_columns, num_augments=1, samples_per_iteration=1):
        """Augment data by creating multiple augmented versions for each sample."""
        augmented_data = dataset.copy()
        for _ in range(num_augments):
            new_data = dataset.copy()
            for col in input_columns:
                for _ in range(samples_per_iteration):
                    noise = np.random.normal(0, 0.3 * dataset[col].std(), size=len(dataset))
                    augmented_sample = new_data.copy()
                    augmented_sample[col] += noise
                    augmented_data = pd.concat([augmented_data, augmented_sample], ignore_index=True)
        return augmented_data

    def export_excel_summary(self, summary_dict):
        # 讓使用者選擇 Excel 輸出路徑
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                                                 title="Save Excel File")
        if file_path:
            with pd.ExcelWriter(file_path) as writer:
                for model_name, df in summary_dict.items():
                    sheet_name = model_name[:31]  # 若超過 31 字元則取前 31 個字
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            messagebox.showinfo("Exported", f"Excel file saved to:\n{file_path}")

    def plot_regression_segment(self, test_y, pred_y, model_name, target_column, feature_combo_str,
                                segment_label="FullRange", low=None, high=None):
        """繪製回歸圖 (全範圍或區段)"""
        if low is not None and high is not None:
            mask = (test_y >= low) & (test_y < high)
            seg_test_y = test_y[mask]
            seg_pred_y = pred_y[mask]
            if len(seg_test_y) == 0:
                return
            title_suffix = f"\nSegment: ({low}-{high})"
            safe_label = f"{low}_{high}"
        else:
            seg_test_y = test_y
            seg_pred_y = pred_y
            title_suffix = "\nFull Range"
            safe_label = "FullRange"

        plt.figure()
        plt.scatter(seg_test_y, seg_pred_y, alpha=0.6, label="Predicted vs Actual")
        min_val = min(seg_test_y.min(), seg_pred_y.min())
        max_val = max(seg_test_y.max(), seg_pred_y.max())
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} Regression\nTarget: {target_column}\nFeatures: {feature_combo_str}{title_suffix}")
        plt.legend()
        safe_features = feature_combo_str.replace(", ", "_").replace(" ", "")
        filename = f"plots/regression_plot_{model_name}_{target_column}_{safe_features}_{safe_label}.png"
        plt.savefig(filename)
        plt.close()
        self.result_text.insert(tk.END, f"Regression plot saved: {filename}\n", "content")

    def choose_and_export_plots(self):
        """
        訓練完成後，彈出一個新視窗讓使用者選擇要輸出哪一筆參數組合的圖。
        視窗中會依模型分區顯示 (例如 RandomForestRegressor、XGBRegressor、KNeighborsRegressor 等)，
        每個區塊中顯示該模型所有參數組合（依 Overall R2 由高到低排序），
        讓使用者可以依自己喜好選取要輸出的項目。
        """
        # 整理出依模型分群的唯一記錄 (以 (model, target, parameters) 為 key)，取最高 Overall R2 的紀錄
        records_by_model = {}
        for record in self.plot_data:
            key = (record["model_name"], record["target_column"], record["features"])
            if record["model_name"] not in records_by_model:
                records_by_model[record["model_name"]] = {}
            if key not in records_by_model[record["model_name"]]:
                records_by_model[record["model_name"]][key] = record
            else:
                if record["overall_r2"] > records_by_model[record["model_name"]][key]["overall_r2"]:
                    records_by_model[record["model_name"]][key] = record

        # 將每個模型的記錄轉為列表，並依 Overall R2 由高到低排序
        model_records = {}
        for model, rec_dict in records_by_model.items():
            rec_list = list(rec_dict.values())
            rec_list.sort(key=lambda x: x["overall_r2"], reverse=True)
            model_records[model] = rec_list

        # 建立一個新的 Toplevel 視窗 (內含捲動區域)
        select_win = tk.Toplevel(self.root)
        select_win.title("Select Parameter Combination to Export")
        select_win.geometry("700x500")

        canvas = tk.Canvas(select_win)
        scrollbar = tk.Scrollbar(select_win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 對每個模型建立一個 LabelFrame
        listboxes = {}  # 儲存每個模型對應的 listbox 與其記錄
        for model, rec_list in model_records.items():
            frame = tk.LabelFrame(scrollable_frame, text=model, font=("Arial", 12, "bold"), padx=10, pady=10)
            frame.pack(fill="x", padx=10, pady=5)

            lb = tk.Listbox(frame, selectmode="extended", width=100, font=("Arial", 10))
            lb.pack(side="left", fill="x", expand=True)
            listboxes[model] = (lb, rec_list)

            # 將每筆紀錄顯示
            for rec in rec_list:
                display_text = f"Target: {rec['target_column']} | Params: {rec['features']} | Overall R2: {rec['overall_r2']:.4f}"
                lb.insert(tk.END, display_text)

        def export_selected():
            segments = [(0, 100), (0, 200), (0, 300), (0, 400), (0, 500), (0, 600)]
            exported = False
            # 用來存放每筆選取記錄對應的真實值與預測值 DataFrame
            excel_sheets = {}
            for model, (lb, rec_list) in listboxes.items():
                selected_indices = lb.curselection()
                for idx in selected_indices:
                    rec = rec_list[idx]
                    test_y = rec["test_y"]
                    pred_y = rec["pred_y"]
                    model_name = rec["model_name"]
                    target_column = rec["target_column"]
                    features = rec["features"]
                    safe_features = features.replace(", ", "_").replace(" ", "")
                    # 儲存 DataFrame，欄位為 "Actual" 與 "Predicted"
                    sheet_df = pd.DataFrame({
                        "Actual": test_y,
                        "Predicted": pred_y
                    })
                    # 為避免工作表名稱超過 31 字元，將名稱做限制
                    sheet_name = f"{model_name}_{target_column}_{safe_features}"[:31]
                    excel_sheets[sheet_name] = sheet_df

                    # 輸出全範圍圖
                    self.plot_regression_segment(test_y, pred_y, model_name, target_column, features, segment_label="FullRange")
                    # 輸出各區段圖
                    for low, high in segments:
                        self.plot_regression_segment(test_y, pred_y, model_name, target_column, features,
                                                     segment_label=f"{low}-{high}", low=low, high=high)
                    exported = True
            if not exported:
                messagebox.showerror("Error", "Please select at least one parameter combination to export.")
            else:
                # 先輸出圖片，再提示使用者選擇 Excel 輸出路徑
                file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                                         filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                                                         title="Save Excel File (Actual vs Predicted)")
                if file_path:
                    with pd.ExcelWriter(file_path) as writer:
                        for sheet_name, df in excel_sheets.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    messagebox.showinfo("Export Completed", f"Selected regression plots and Excel file have been exported.\nExcel file saved to:\n{file_path}")
                else:
                    messagebox.showwarning("Export Skipped", "Excel export skipped.")
                # 匯出後不關閉選擇視窗，讓使用者可以重新選擇

        btn_export = tk.Button(select_win, text="Export Selected", command=export_selected, bg="#4caf50", fg="#fff", font=("Arial", 12))
        btn_export.pack(pady=10)

    def train_models(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return
        if not self.input_columns or not self.target_columns:
            messagebox.showerror("Error", "Please select input and target columns.")
            return

        try:
            dataset = self.dataset.dropna()
            original_data_count = len(dataset)
            augmented_dataset = self.augment_data(dataset, self.input_columns, num_augments=3)
            augmented_data_count = len(augmented_dataset)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Original data count: {original_data_count}\n", "content")
            self.result_text.insert(tk.END, f"Augmented data count: {augmented_data_count}\n", "content")
            self.result_text.insert(tk.END, "-" * 24 + "\n", "separator")

            models = [
                ('RandomForestRegressor', RandomForestRegressor()),
                ('XGBRegressor', XGBRegressor()),
                ('LGBMRegressor', lgb.LGBMRegressor()),
                ('LinearRegression', LinearRegression()),
                ('SVR', SVR()),
                ('KNeighborsRegressor', KNeighborsRegressor())
            ]
            summary_dict = {name: [] for name, _ in models}
            segments = [(0, 100), (0, 200), (0, 300), (0, 400), (0, 500), (0, 600)]

            for target_column in self.target_columns:
                y = augmented_dataset[target_column]
                for r in range(1, len(self.input_columns) + 1):
                    for feature_combo in itertools.combinations(self.input_columns, r):
                        feature_combo = list(feature_combo)
                        X_combo = augmented_dataset[feature_combo]
                        train_X, test_X, train_y, test_y = train_test_split(X_combo, y, test_size=0.025, random_state=20)
                        scaler = MinMaxScaler()
                        scaler.fit(train_X)
                        train_X = scaler.transform(train_X)
                        test_X = scaler.transform(test_X)
                        input_combo_str = ", ".join(feature_combo)
                        
                        self.result_text.insert(tk.END, f"Target: {target_column} | Features: {input_combo_str}\n", "header")
                        
                        for name, model in models:
                            model.fit(train_X, train_y)
                            pred_y = model.predict(test_X)
                            
                            overall_r2 = max(r2_score(test_y, pred_y), 0)
                            mse = mean_squared_error(test_y, pred_y)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(test_y, pred_y)
                            
                            seg_r2_results = {}
                            for low, high in segments:
                                mask_segment = (test_y >= low) & (test_y < high)
                                seg_test_y = test_y[mask_segment]
                                seg_pred_y = pred_y[mask_segment]
                                if len(seg_test_y) > 0:
                                    seg_r2 = max(r2_score(seg_test_y, seg_pred_y), 0)
                                else:
                                    seg_r2 = None
                                seg_r2_results[f"R2 ({low}-{high})"] = seg_r2

                            summary_dict[name].append({
                                "Target": target_column,
                                "Parameters": input_combo_str,
                                "Overall R2": overall_r2,
                                "MSE": mse,
                                "RMSE": rmse,
                                "MAE": mae,
                                "R2 (0-100)": seg_r2_results["R2 (0-100)"],
                                "R2 (0-200)": seg_r2_results["R2 (0-200)"],
                                "R2 (0-300)": seg_r2_results["R2 (0-300)"],
                                "R2 (0-400)": seg_r2_results["R2 (0-400)"],
                                "R2 (0-500)": seg_r2_results["R2 (0-500)"],
                                "R2 (0-600)": seg_r2_results["R2 (0-600)"]
                            })
                            
                            self.result_text.insert(tk.END, f"Model: {name} | Overall R2: {overall_r2}\n", "content")
                            self.result_text.insert(tk.END, f"MSE: {mse}\n", "content")
                            self.result_text.insert(tk.END, f"RMSE: {rmse}\n", "content")
                            self.result_text.insert(tk.END, f"MAE: {mae}\n", "content")
                            for low, high in segments:
                                key = f"R2 ({low}-{high})"
                                self.result_text.insert(tk.END, f"{key}: {seg_r2_results[key]}\n", "content")
                            self.result_text.insert(tk.END, "-" * 24 + "\n", "separator")
                            
                            self.plot_data.append({
                                "test_y": test_y,
                                "pred_y": pred_y,
                                "model_name": name,
                                "target_column": target_column,
                                "features": input_combo_str,
                                "overall_r2": overall_r2
                            })

            summary_excel = {}
            for model_name, results in summary_dict.items():
                df = pd.DataFrame(results)
                summary_excel[model_name] = df
            self.export_excel_summary(summary_excel)

            # 訓練完成並匯出 Excel 後，讓使用者選擇要輸出的參數組合圖
            if messagebox.askyesno("Export Images?", "Do you want to select parameter combination(s) to export regression plots?"):
                self.choose_and_export_plots()

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    app.result_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#333")
    app.result_text.tag_configure("content", font=("Arial", 12))
    app.result_text.tag_configure("separator", font=("Arial", 12, "bold"), foreground="#888")
    root.mainloop()
