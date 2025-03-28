import os
import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_entropy(column):
    value_counts = column.value_counts()
    probabilities = value_counts / len(column)
    return entropy(probabilities, base=2)

def calculate_probabilities(column):
    value_counts = column.value_counts()
    probabilities = value_counts / len(column)
    return probabilities

def cross_entropy_loss(true_prob, pred_prob):
    true_prob = np.array(true_prob)
    pred_prob = np.array(pred_prob)
    
    epsilon = 1e-10
    pred_prob = np.clip(pred_prob, epsilon, 1. - epsilon)
    
    loss = -np.sum(true_prob * np.log2(pred_prob))
    return loss

def process_file(filepath):
    df = pd.read_excel(filepath)
    
    df['X'] = df['X'].round().astype(int)
    df['Y'] = df['Y'].round().astype(int)
    
    entropy_value_X = calculate_entropy(df['X'])
    entropy_value_Y = calculate_entropy(df['Y'])
    
    prob_X = calculate_probabilities(df['X'])
    prob_Y = calculate_probabilities(df['Y'])
    
    prob_X, prob_Y = prob_X.align(prob_Y, fill_value=0)
    
    nonzero_indices = prob_Y > 0
    prob_X_filtered = prob_X[nonzero_indices]
    prob_Y_filtered = prob_Y[nonzero_indices]
    
    prob_X_filtered = prob_X_filtered / prob_X_filtered.sum()
    prob_Y_filtered = prob_Y_filtered / prob_Y_filtered.sum()
    
    cross_entropy = cross_entropy_loss(prob_X_filtered, prob_Y_filtered)
    
    return {
        'File Name': os.path.basename(filepath),
        'Entropy_X': entropy_value_X,
        'Entropy_Y': entropy_value_Y,
        'Cross_Entropy': cross_entropy
    }

def process_all_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(directory, filename)
            result = process_file(filepath)
            results.append(result)
    return results

def save_results_to_excel(results, output_file):
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False)

# 设置你的数据文件夹路径
data_folder = r'C:\Users\User\Desktop\T\eye\內差法後excel'
output_file = r'C:\Users\User\Desktop\T\eye\Entropy\post結果.xlsx'

# 处理所有文件并保存结果
results = process_all_files(data_folder)
save_results_to_excel(results, output_file)
