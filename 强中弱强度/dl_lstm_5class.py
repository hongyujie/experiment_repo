import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

os.makedirs('img', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

ATTACK_MAPPING = {
    'normal_40Hz.csv': 0,
    'tcp_high_40Hz.csv': 1,
    'udp_high_40Hz.csv': 2,
    'icmp_high_40Hz.csv': 3
}

CLASS_NAMES = ['正常', 'TCP', 'UDP', 'ICMP']

def load_data():
    """加载五个数据文件并合并，统计各类别样本数"""
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

def split_and_window(df, window_size=50, step_size=1, train_ratio=0.7):
    """
    先按时间切分原始数据，再分别滑窗提取窗口
    防止数据泄露：确保训练集和测试集的原始数据完全不重叠

    处理流程：
    1. 按label分组
    2. 每个类别内部按train_ratio切分原始序列
    3. 分别对训练段和测试段滑窗提取窗口
    """
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

    X_train = torch.FloatTensor(X_train).unsqueeze(2)
    X_test = torch.FloatTensor(X_test).unsqueeze(2)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """
    对输入数据进行标准化（Z-score归一化）
    对每个通道计算均值和标准差，然后用训练集的统计量进行归一化
    这样可以加速神经网络收敛
    """
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, mean, std

class SignalDataset(Dataset):
    """PyTorch数据集封装（带数据增强）"""
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        return x, y

class LSTMClassifier(nn.Module):
    """
    长短期记忆网络用于时序信号分类

    架构：
    - 2层LSTM网络（捕获时序依赖关系）
    - 全连接层分类器
    - Dropout防止过拟合
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4, dropout=0.5):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def train_epoch(model, loader, criterion, optimizer):
    """训练一个epoch，返回平均损失和准确率"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    """在测试集上评估模型，返回准确率和预测结果"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return correct / total, np.array(all_preds), np.array(all_labels)

def plot_training_curves(train_losses, train_accs, test_accs, save_path):
    """绘制训练曲线：损失和准确率"""
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
    ax.set_title('混淆矩阵 (LSTM)', fontsize=16, pad=20)

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
    主函数：LSTM时序分类

    流程：
    1. 加载数据
    2. 先切分后滑窗（避免数据泄露）
    3. 数据归一化
    4. 构建模型
    5. 训练和评估
    6. 保存结果
    """
    print('=' * 60)
    print('LSTM 电流信号四分类任务')
    print('=' * 60)

    window_size = 50
    step_size = 1
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.0005
    weight_decay = 1e-4
    patience = 15
    no_improve_count = 0

    print('\n1. 加载数据...')
    df = load_data()

    print('\n2. 先切分后滑窗...')
    X_train, X_test, y_train, y_test = split_and_window(
        df, window_size=window_size, step_size=step_size, train_ratio=0.7
    )
    print(f'原始数据 - 训练集: {X_train.shape}, 测试集: {X_test.shape}')

    print('\n2.1 数据归一化...')
    X_train, X_test, data_mean, data_std = normalize_data(X_train, X_test)
    print(f'归一化后 - 训练集: {X_train.shape}, 均值={data_mean:.2f}, 标准差={data_std:.2f}')

    print('\n3. 创建数据加载器...')
    train_dataset = SignalDataset(X_train, y_train, augment=True)
    test_dataset = SignalDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('\n4. 构建LSTM模型...')
    model = LSTMClassifier(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_classes=4,
        dropout=0.5
    ).to(device)

    print('\n4.1 权重初始化（LSTM对初始化敏感）...')
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
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
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print('\n7. 保存图表...')
    plot_training_curves(train_losses, train_accs, test_accs, 'img/training_curves_lstm.png')
    plot_confusion_matrix(y_true, y_pred, 'img/confusion_matrix_lstm.png')

    print('\n训练完成!')

if __name__ == '__main__':
    main()
