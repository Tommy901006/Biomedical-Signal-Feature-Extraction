import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Classifier UI')
        self.root.geometry('800x600')
        self.root.configure(bg='#2c3e50')

        self.dataset = None
        self.selected_classifier = None
        self.input_vars = []
        self.target_var = ""

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("TButton", padding=6, relief="flat", background="#3498db", foreground="#ffffff", font=("Helvetica", 12))
        style.map("TButton", background=[("active", "#2980b9")])
        style.configure("TFrame", background="#34495e")
        style.configure("TLabel", background="#34495e", foreground="#ecf0f1", font=("Helvetica", 12))
        style.configure("TCombobox", padding=6, font=("Helvetica", 12))
        
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=BOTH, expand=True)

        canvas = Canvas(main_frame, bg="#34495e")
        canvas.pack(fill=BOTH, expand=True)

        frame = ttk.Frame(canvas, padding="20 20 20 20")
        frame.place(relx=0.5, rely=0.5, anchor=CENTER)

        ttk.Label(frame, text="選擇CSV檔案:").grid(row=0, column=0, pady=10, padx=10)
        ttk.Button(frame, text="選擇文件", command=self.load_file).grid(row=0, column=1, pady=10, padx=10)

        ttk.Label(frame, text="選擇輸入參數:").grid(row=1, column=0, pady=10, padx=10)
        self.input_vars_listbox = Listbox(frame, selectmode=MULTIPLE, width=50, bg="#2c3e50", fg="#ecf0f1", font=("Helvetica", 12))
        self.input_vars_listbox.grid(row=1, column=1, pady=10, padx=10)
        ttk.Button(frame, text="確認輸入參數", command=self.confirm_input_vars).grid(row=1, column=2, pady=10, padx=10)

        ttk.Label(frame, text="選擇目標參數:").grid(row=2, column=0, pady=10, padx=10)
        self.target_var_combobox = ttk.Combobox(frame)
        self.target_var_combobox.grid(row=2, column=1, pady=10, padx=10)
        ttk.Button(frame, text="確認目標參數", command=self.confirm_target_var).grid(row=2, column=2, pady=10, padx=10)

        ttk.Label(frame, text="選擇分類器:").grid(row=3, column=0, pady=10, padx=10)
        self.classifier_combobox = ttk.Combobox(frame, values=['Random Forest', 'Gradient Boosting', 'SVM', 'KNN'])
        self.classifier_combobox.grid(row=3, column=1, pady=10, padx=10)

        ttk.Button(frame, text="開始執行", command=self.run_classifier).grid(row=4, column=1, pady=20, padx=10)

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.dataset = pd.read_csv(file_path)
            self.update_listboxes()
            messagebox.showinfo("信息", "文件加載成功")

    def update_listboxes(self):
        if self.dataset is not None:
            columns = list(self.dataset.columns)
            self.input_vars_listbox.delete(0, END)
            for col in columns:
                self.input_vars_listbox.insert(END, col)
            self.target_var_combobox['values'] = columns

    def confirm_input_vars(self):
        self.input_vars = [self.input_vars_listbox.get(idx) for idx in self.input_vars_listbox.curselection()]
        if not self.input_vars:
            messagebox.showwarning("警告", "請選擇輸入參數")
        else:
            messagebox.showinfo("信息", "輸入參數確認成功")

    def confirm_target_var(self):
        self.target_var = self.target_var_combobox.get()
        if not self.target_var:
            messagebox.showwarning("警告", "請選擇目標參數")
        else:
            messagebox.showinfo("信息", "目標參數確認成功")

    def run_classifier(self):
        if self.dataset is not None:
            if not self.input_vars or not self.target_var or not self.classifier_combobox.get():
                messagebox.showwarning("警告", "請選擇輸入參數、目標參數和分類器")
                return

            x = self.dataset[self.input_vars]
            y = self.dataset[self.target_var]
            # 資料正規化
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            smote = SMOTE(random_state=100)
            X_res, y_res = smote.fit_resample(x, y)

            train_X, test_X, train_y, test_y = train_test_split(X_res, y_res, test_size=0.2, random_state=22)
            scaler = MinMaxScaler()
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)

            classifier_name = self.classifier_combobox.get()
            if classifier_name == 'Random Forest':
                clf = RandomForestClassifier(random_state=6)
            elif classifier_name == 'Gradient Boosting':
                clf = GradientBoostingClassifier(random_state=6)
            elif classifier_name == 'SVM':
                clf = SVC(probability=True, random_state=6)
            elif classifier_name == 'KNN':
                clf = KNeighborsClassifier()

            clf.fit(train_X, train_y)
            y_pred = clf.predict(test_X)
            y_score = clf.predict_proba(test_X)

            report = classification_report(test_y, y_pred)
            print("Classifier - 分類報告:")
            print(report)

            # 計算並繪製ROC曲線和AUC
            n_classes = len(np.unique(y))
            test_y_binary = label_binarize(test_y, classes=np.unique(y))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                if n_classes == 2:
                    fpr[i], tpr[i], _ = roc_curve(test_y_binary, y_score[:, 1])
                else:
                    fpr[i], tpr[i], _ = roc_curve(test_y_binary[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure()
            colors = ['aqua', 'darkorange', 'cornflowerblue']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic for multi-class')
            plt.legend(loc="lower right")
            plt.show()

            # 計算混淆矩陣
            cm = confusion_matrix(test_y, y_pred)
            print(cm)

            # 繪製混淆矩陣
            plt.figure(figsize=(10, 7))
            class_names = list(np.unique(y))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
            messagebox.showinfo("信息", "分類器執行成功")

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
