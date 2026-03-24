import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def split_and_window(df, window_size=200, step_size=10, train_ratio=0.7):
    """
    先按时间切分原始数据，再分别滑窗
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

    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test):
    """Z-score归一化"""
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    return X_train, X_test, mean, std


class SignalDataset(Dataset):
    """
    PyTorch数据集封装 - 数据增强版本
    """
    def __init__(self, X, y, augment=False, augment_strength=1.0):
        self.X = X
        self.y = y
        self.augment = augment
        self.augment_strength = augment_strength

    def __len__(self):
        return len(self.y)

    def jitter(self, x, noise_level=0.02):
        """添加高斯噪声（抖动）"""
        noise = torch.randn_like(x) * noise_level * self.augment_strength
        return x + noise

    def scaling(self, x, scale_range=(0.9, 1.1)):
        """随机缩放幅度"""
        scale = torch.FloatTensor(1).uniform_(
            scale_range[0], scale_range[1]
        ).to(x.device)
        return x * scale

    def magnitude_warp(self, x, warp_range=0.1):
        """幅度扭曲 - 对信号幅度进行非线性变换"""
        seq_len = x.shape[-1]
        num_knots = 4
        knot_points = torch.linspace(0, seq_len - 1, num_knots)
        knot_values = torch.FloatTensor(num_knots).uniform_(
            1 - warp_range * self.augment_strength,
            1 + warp_range * self.augment_strength
        )

        # 插值得到完整的扭曲因子
        indices = torch.arange(seq_len).float()
        warp_factors = torch.nn.functional.interpolate(
            knot_values.unsqueeze(0).unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=True
        ).squeeze()

        return x * warp_factors.unsqueeze(0)

    def random_crop_resize(self, x, crop_ratio=0.9):
        """随机裁剪并调整大小"""
        seq_len = x.shape[-1]
        crop_len = int(seq_len * crop_ratio)

        if crop_len < seq_len:
            start_idx = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
            cropped = x[:, start_idx:start_idx + crop_len]
            # 调整回原始大小
            resized = torch.nn.functional.interpolate(
                cropped.unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).squeeze(0)
            return resized
        return x

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # 克隆以避免修改原始数据
        y = self.y[idx]

        if self.augment:
            # 随机选择应用哪些增强（增加多样性）
            aug_methods = []

            # 基础噪声（始终应用）
            x = self.jitter(x, noise_level=0.02)

            # 随机应用其他增强
            if torch.rand(1).item() > 0.3:  # 70%概率应用缩放
                x = self.scaling(x, scale_range=(0.95, 1.05))

            if torch.rand(1).item() > 0.5:  # 50%概率应用幅度扭曲
                x = self.magnitude_warp(x, warp_range=0.08)

            if torch.rand(1).item() > 0.7:  # 30%概率应用裁剪
                x = self.random_crop_resize(x, crop_ratio=0.95)

        return x, y


class CNNLSTMWithTCPPostBranch(nn.Module):
    """
    CNN-LSTM + LSTM后TCP专用分支
    改进点：在LSTM输出后添加TCP专用分支，先学习时序特征再增强TCP特征
    """
    def __init__(self, num_classes=4, dropout=0.5):
        super(CNNLSTMWithTCPPostBranch, self).__init__()

        # CNN特征提取
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

        # LSTM层（所有类别共享）
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # TCP专用分支（在LSTM后）
        # 输入：LSTM输出64维，输出：TCP增强特征32维
        self.tcp_branch = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3)
        )

        # TCP门控网络：预测TCP概率，用于动态加权
        self.tcp_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出TCP概率
        )

        # 特征融合层
        # LSTM特征64维 + TCP增强特征32维 = 96维
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 分类层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, labels=None):
        batch_size = x.size(0)

        # CNN特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)  # [batch, 64, seq_len]

        # 转换为时序格式
        x = x.permute(0, 2, 1)  # [batch, seq_len, 64]

        # LSTM处理（所有类别共享时序特征）
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 64]
        lstm_out = lstm_out[:, -1, :]  # [batch, 64] 取最后一个时间步

        # TCP门控：预测TCP概率
        tcp_prob = self.tcp_gate(lstm_out)  # [batch, 1]

        # TCP专用分支处理
        tcp_feat = self.tcp_branch(lstm_out)  # [batch, 32]

        # 动态加权：TCP概率高的样本给予更多TCP特征
        weighted_tcp_feat = tcp_feat * tcp_prob  # [batch, 32]

        # 融合LSTM特征和加权TCP特征
        combined = torch.cat([lstm_out, weighted_tcp_feat], dim=-1)  # [batch, 96]

        # 特征融合
        fused = self.fusion(combined)  # [batch, 64]

        # 分类
        output = self.fc(fused)
        return output


def train_epoch(model, loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X, labels=y)
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
    ax.set_title('混淆矩阵 (CNN-LSTM+TCP-Post-Branch 1240Hz)', fontsize=16, pad=20)

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
    主函数：CNN-LSTM + LSTM后TCP专用分支
    改进点：在LSTM输出后添加TCP专用分支，先学习时序特征再增强TCP特征
    """
    print('=' * 60)
    print('CNN-LSTM+LSTM后TCP分支 1240Hz电流信号四分类任务')
    print('=' * 60)

    window_size = 400
    step_size = 10
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
    print(f'训练集: {X_train.shape}, 测试集: {X_test.shape}')

    print('\n2.1 数据归一化...')
    X_train, X_test, mean, std = normalize_data(X_train, X_test)
    print(f'归一化 - 均值: {mean:.2f}, 标准差: {std:.2f}')

    print('\n3. 创建数据加载器（使用数据增强）...')
    train_dataset = SignalDataset(X_train, y_train, augment=True, augment_strength=1.0)
    test_dataset = SignalDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('\n4. 构建CNN-LSTM+LSTM后TCP分支模型...')
    model = CNNLSTMWithTCPPostBranch(num_classes=4, dropout=0.5).to(device)

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

    # 类别权重：最佳配置
    class_weights = torch.FloatTensor([1.0, 1.0, 2.0, 1.0]).to(device)
    print(f'\n类别权重: 正常={class_weights[0]:.1f}, TCP={class_weights[1]:.1f}, '
          f'UDP={class_weights[2]:.1f}, ICMP={class_weights[3]:.1f}')

    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    plot_training_curves(train_losses, train_accs, test_accs,
                         'img/training_curves_tcp_post_1240Hz.png')
    plot_confusion_matrix(y_true, y_pred,
                          'img/confusion_matrix_tcp_post_1240Hz.png')

    print('\n训练完成!')


if __name__ == '__main__':
    main()
