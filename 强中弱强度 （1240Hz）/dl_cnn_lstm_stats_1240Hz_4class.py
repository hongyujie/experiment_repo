import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
from scipy import stats

os.makedirs('img', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

ATTACK_MAPPING = {
    'normal_1240Hz.csv': 0,
    'tcp_1240Hz.csv': 1,
    'udp_1240Hz.csv': 2,
    'icmp_1240Hz.csv': 3
}

CLASS_NAMES = ['正常', 'TCP', 'UDP', 'ICMP']

def load_data():
    """加载四个1240Hz数据文件并合并"""
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
            print(f'已加载: {filename}, {len(df)} 行')

    df = pd.concat(all_data, ignore_index=True)
    print(f'总数据量: {len(df)} 行')

    print('\n各类别样本数量:')
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f'  {CLASS_NAMES[label]}: {count}')

    return df

def compute_statistical_features(window):
    """
    计算丰富的时域统计特征
    这些特征可能帮助区分UDP和ICMP
    """
    features = {}

    # 基本统计量
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['max'] = np.max(window)
    features['min'] = np.min(window)
    features['range'] = features['max'] - features['min']

    # 峰峰值
    features['peak2peak'] = features['range']

    # 均方根
    features['rms'] = np.sqrt(np.mean(window**2))

    # 波形因子 (RMS / 均值绝对值)
    mean_abs = np.mean(np.abs(window))
    features['shape_factor'] = features['rms'] / (mean_abs + 1e-8)

    # 峰值因子 (峰值 / RMS)
    features['crest_factor'] = features['max'] / (features['rms'] + 1e-8)

    # 脉冲因子 (峰值 / 均值绝对值)
    features['impulse_factor'] = features['max'] / (mean_abs + 1e-8)

    # 裕度因子
    features['clearance_factor'] = features['max'] / (np.mean(np.sqrt(np.abs(window)))**2 + 1e-8)

    # 偏度 (分布不对称性)
    features['skewness'] = stats.skew(window)

    # 峰度 (分布尖锐程度)
    features['kurtosis'] = stats.kurtosis(window)

    # 过零率
    zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
    features['zero_crossing_rate'] = zero_crossings / len(window)

    # 能量
    features['energy'] = np.sum(window**2)

    # 熵 (信号复杂度)
    hist, _ = np.histogram(window, bins=20, density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log(hist + 1e-8))

    # 变异系数
    features['cv'] = features['std'] / (features['mean'] + 1e-8)

    # 绝对均值
    features['abs_mean'] = mean_abs

    # 中位数
    features['median'] = np.median(window)

    # 四分位距
    q75, q25 = np.percentile(window, [75, 25])
    features['iqr'] = q75 - q25

    # 转换为numpy数组
    feature_vector = np.array(list(features.values()), dtype=np.float32)

    return feature_vector

def split_and_window_with_stats(df, window_size=200, step_size=10, train_ratio=0.7):
    """
    先按时间切分原始数据，再分别滑窗，同时提取统计特征
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    S_train, S_test = [], []  # 统计特征

    grouped = df.groupby('label')

    for label, group in grouped:
        values = group['current'].values
        split_idx = int(len(values) * train_ratio)

        train_values = values[:split_idx]
        test_values = values[split_idx:]

        for i in range(0, len(train_values) - window_size + 1, step_size):
            window = train_values[i:i+window_size]
            X_train.append(window)
            S_train.append(compute_statistical_features(window))
            y_train.append(label)

        for i in range(0, len(test_values) - window_size + 1, step_size):
            window = test_values[i:i+window_size]
            X_test.append(window)
            S_test.append(compute_statistical_features(window))
            y_test.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    S_train = np.array(S_train, dtype=np.float32)
    S_test = np.array(S_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    S_train = torch.FloatTensor(S_train)
    S_test = torch.FloatTensor(S_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, S_train, S_test, y_train, y_test

def normalize_data(X_train, X_test):
    """Z-score归一化"""
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    return X_train, X_test, mean, std

class SignalDatasetWithStats(Dataset):
    """PyTorch数据集封装（包含统计特征）"""
    def __init__(self, X, S, y, augment=False):
        self.X = X
        self.S = S
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        s = self.S[idx]
        y = self.y[idx]

        if self.augment:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        return x, s, y

class CNNLSTMWithStatsClassifier(nn.Module):
    """
    CNN-LSTM + 时域统计特征 混合网络
    改进点：使用20维时域统计特征帮助区分UDP和ICMP
    """
    def __init__(self, num_classes=4, dropout=0.5):
        super(CNNLSTMWithStatsClassifier, self).__init__()

        # 时域CNN分支
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout * 0.5)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout * 0.5)
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # 统计特征处理分支 (20维 -> 32维)
        self.stats_fc = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # 融合层：时域特征(64) + 统计特征(32) = 96
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 + 32, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes)
        )

    def forward(self, x, stats_features):
        # 时域特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        time_features = x[:, -1, :]  # [batch, 64]

        # 统计特征提取
        stats_out = self.stats_fc(stats_features)  # [batch, 32]

        # 特征融合
        combined = torch.cat([time_features, stats_out], dim=1)  # [batch, 96]

        # 分类
        output = self.fc(combined)
        return output

def train_epoch(model, loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, S, y in loader:
        X, S, y = X.to(device), S.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X, S)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, S, y in loader:
            X, S, y = X.to(device), S.to(device), y.to(device)
            outputs = model(X, S)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return correct / total, np.array(all_preds), np.array(all_labels)

def plot_training_curves(train_losses, train_accs, test_accs, save_path):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失 (Training Loss)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, 'g-', label='训练准确率')
    ax2.plot(epochs, test_accs, 'r-', label='测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('训练和测试准确率 (Accuracy)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'已保存训练曲线到 {save_path}')

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('混淆矩阵 (CNN-LSTM+Stats 1240Hz)', fontsize=16, pad=20)

    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已保存混淆矩阵到 {save_path}')

def main():
    """
    主函数：CNN-LSTM + 时域统计特征 四分类任务
    改进点：使用18维时域统计特征
    """
    print('=' * 60)
    print('CNN-LSTM+时域统计特征 1240Hz电流信号四分类任务')
    print('=' * 60)

    window_size = 200
    step_size = 10
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.0005
    weight_decay = 1e-4
    patience = 15
    no_improve_count = 0

    print('\n1. 加载数据...')
    df = load_data()

    print('\n2. 先切分后滑窗（包含统计特征提取）...')
    X_train, X_test, S_train, S_test, y_train, y_test = split_and_window_with_stats(
        df, window_size=window_size, step_size=step_size, train_ratio=0.7
    )
    print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')
    print(f'统计特征 - 训练集: {S_train.shape}, 测试集: {S_test.shape}')

    print('\n2.1 数据归一化...')
    X_train, X_test, mean, std = normalize_data(X_train, X_test)
    print(f'归一化 - 均值: {mean:.2f}, 标准差: {std:.2f}')

    print('\n3. 创建数据加载器...')
    train_dataset = SignalDatasetWithStats(X_train, S_train, y_train, augment=True)
    test_dataset = SignalDatasetWithStats(X_test, S_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('\n4. 构建CNN-LSTM+Stats模型...')
    model = CNNLSTMWithStatsClassifier(num_classes=4, dropout=0.5).to(device)

    print('\n4.1 权重初始化...')
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    print('\n5. 开始训练（早停法防止过拟合）...')
    train_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc, _, _ = evaluate(model, test_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        scheduler.step(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f'Epoch [{epoch}/{num_epochs}] '
                  f'损失: {train_loss:.4f} '
                  f'训练准确率: {train_acc:.4f} '
                  f'测试准确率: {test_acc:.4f}')

        if no_improve_count >= patience:
            print(f'\n早停触发！连续 {patience} 个epoch测试准确率未提升')
            break

    print(f'\n最佳测试准确率: {best_test_acc:.4f}')

    print('\n6. 最终评估...')
    test_acc, y_pred, y_true = evaluate(model, test_loader)

    print(f'\n最终测试准确率: {test_acc:.4f}')
    print('\n分类报告:')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    print('\n7. 保存图表...')
    plot_training_curves(train_losses, train_accs, test_accs, 'img/training_curves_stats_1240Hz.png')
    plot_confusion_matrix(y_true, y_pred, 'img/confusion_matrix_stats_1240Hz.png')

    print('\n训练完成!')

if __name__ == '__main__':
    main()
