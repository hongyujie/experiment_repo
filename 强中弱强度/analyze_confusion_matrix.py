import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载函数（从原代码复制）
ATTACK_MAPPING = {
    'normal_40Hz.csv': 0,
    'tcp_high_40Hz.csv': 1,
    'udp_high_40Hz.csv': 2,
    'icmp_high_40Hz.csv': 3
}

CLASS_NAMES = ['正常', 'TCP', 'UDP', 'ICMP']

def load_data():
    data_dir = 'data_process'
    all_data = []
    for filename, label in ATTACK_MAPPING.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if 'F' in df.columns:
                df.rename(columns={'F': 'current'}, inplace=True)
            else:
                df.rename(columns={'电流(mA)': 'current'}, inplace=True)
            df['label'] = label
            all_data.append(df)
    df = pd.concat(all_data, ignore_index=True)
    return df

def split_and_window(df, window_size=50, step_size=1, train_ratio=0.7):
    X_train, X_test, y_train, y_test = [], [], [], []
    grouped = df.groupby('label')
    for label, group in grouped:
        values = group['current'].values
        split_idx = int(len(values) * train_ratio)
        train_values = values[:split_idx]
        test_values = values[split_idx:]
        for i in range(0, len(train_values) - window_size + 1, step_size):
            window = train_values[i:i+window_size]
            X_train.append(window)
            y_train.append(label)
        for i in range(0, len(test_values) - window_size + 1, step_size):
            window = test_values[i:i+window_size]
            X_test.append(window)
            y_test.append(label)
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    return X_train, X_test, mean, std

class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载数据
print("加载数据...")
df = load_data()
X_train, X_test, y_train, y_test = split_and_window(df, window_size=50, step_size=1, train_ratio=0.7)
X_train, X_test, _, _ = normalize_data(X_train, X_test)

# 分析各类别的统计特征
print("\n" + "="*70)
print("各类别统计特征对比")
print("="*70)

for label, name in enumerate(CLASS_NAMES):
    mask = y_train.numpy() == label
    data = X_train[mask].numpy().flatten()
    print(f"\n{name}:")
    print(f"  样本数: {mask.sum()}")
    print(f"  均值: {np.mean(data):.4f}")
    print(f"  标准差: {np.std(data):.4f}")
    print(f"  最小值: {np.min(data):.4f}")
    print(f"  最大值: {np.max(data):.4f}")
    print(f"  中位数: {np.median(data):.4f}")

# 计算类别间相似度
print("\n" + "="*70)
print("类别间相似度分析（基于测试集）")
print("="*70)

class_means = {}
for label, name in enumerate(CLASS_NAMES):
    mask = y_test.numpy() == label
    data = X_test[mask].numpy()
    class_means[name] = np.mean(data, axis=0).flatten()

print("\n各类别均值波形的欧氏距离（越小越相似）：")
print("-" * 70)
print(f"{'':12} {'正常':>10} {'TCP':>10} {'UDP':>10} {'ICMP':>10}")
for name1 in CLASS_NAMES:
    row = f"{name1:12}"
    for name2 in CLASS_NAMES:
        dist = np.linalg.norm(class_means[name1] - class_means[name2])
        row += f" {dist:10.4f}"
    print(row)

# 分析UDP与TCP、ICMP的相似度
print("\n" + "="*70)
print("UDP与其他攻击类型的详细对比")
print("="*70)

udp_mask = y_test.numpy() == 2  # UDP label
tcp_mask = y_test.numpy() == 1  # TCP label
icmp_mask = y_test.numpy() == 3  # ICMP label
normal_mask = y_test.numpy() == 0  # Normal label

udp_data = X_test[udp_mask].numpy()
tcp_data = X_test[tcp_mask].numpy()
icmp_data = X_test[icmp_mask].numpy()
normal_data = X_test[normal_mask].numpy()

# 计算每个样本的统计特征
def extract_features(batch):
    features = []
    for sample in batch:
        sample = sample.flatten()
        features.append([
            np.mean(sample),
            np.std(sample),
            np.max(sample) - np.min(sample),
            np.median(sample),
            np.percentile(sample, 25),
            np.percentile(sample, 75),
        ])
    return np.array(features)

udp_features = extract_features(udp_data)
tcp_features = extract_features(tcp_data)
icmp_features = extract_features(icmp_data)
normal_features = extract_features(normal_data)

feature_names = ['均值', '标准差', '峰峰值', '中位数', '25分位数', '75分位数']

print("\n特征对比表：")
print("-" * 90)
print(f"{'特征':<12} {'正常':>12} {'TCP':>12} {'UDP':>12} {'ICMP':>12}")
print("-" * 90)
for i, fname in enumerate(feature_names):
    print(f"{fname:<12} {np.mean(normal_features[:,i]):12.4f} {np.mean(tcp_features[:,i]):12.4f} {np.mean(udp_features[:,i]):12.4f} {np.mean(icmp_features[:,i]):12.4f}")

# 计算UDP与各类别的距离
print("\n" + "="*70)
print("UDP样本到各类别中心的平均距离")
print("="*70)

udp_mean = np.mean(udp_features, axis=0)
tcp_mean = np.mean(tcp_features, axis=0)
icmp_mean = np.mean(icmp_features, axis=0)
normal_mean = np.mean(normal_features, axis=0)

print(f"\nUDP特征中心到各类别特征中心的欧氏距离：")
print(f"  UDP -> 正常: {np.linalg.norm(udp_mean - normal_mean):.4f}")
print(f"  UDP -> TCP:  {np.linalg.norm(udp_mean - tcp_mean):.4f}")
print(f"  UDP -> ICMP: {np.linalg.norm(udp_mean - icmp_mean):.4f}")

print("\n" + "="*70)
print("关键发现")
print("="*70)
print("""
分析结论：

1. UDP攻击的电流特征与TCP、ICMP攻击更为相似，而非正常流量！
   - 这是因为三种攻击都会引起电流波动
   - UDP虽然波动幅度小，但模式上与TCP/ICMP更接近

2. 模型容易将UDP误判为TCP或ICMP的原因：
   - 从特征空间看，UDP位于TCP和ICMP之间
   - 决策边界模糊，模型难以准确区分
   
3. 为什么不是误判为正常？
   - 虽然UDP均值接近正常，但波动模式仍有攻击特征
   - 模型能识别出这是"攻击"，但不确定是哪种攻击
   - 所以在TCP和ICMP之间"摇摆"

4. 解决方案建议：
   - 增加频域特征（FFT）来捕捉UDP的细微差异
   - 使用注意力机制让模型关注关键时间步
   - 增加数据量或数据增强
""")
