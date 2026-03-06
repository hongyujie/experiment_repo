import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据文件路径
data_dir = Path("data")

class CurrentDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CNNClassifier(nn.Module):
    """CNN分类器"""
    def __init__(self, input_dim=7, num_classes=7):
        super(CNNClassifier, self).__init__()
        
        # 1D卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 2, 128)  # 输入维度需要根据实际计算调整
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 调整输入形状: (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)  # 添加通道维度
        
        # 卷积层1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # 卷积层2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MLPClassifier(nn.Module):
    """多层感知机分类器"""
    def __init__(self, input_dim=7, num_classes=7):
        super(MLPClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data():
    """
    加载并预处理数据
    返回特征矩阵X和标签y
    """
    # 定义攻击类型映射
    attack_mapping = {
        "normal_40Hz.csv": 0,      # 正常
        "tcp_icmp_40Hz.csv": 1,    # TCP+ICMP攻击
        "tcp_udp_40Hz.csv": 2,     # TCP+UDP攻击
        "tcp_dns_40Hz.csv": 3,     # TCP+DNS攻击
        "icmp_udp_40Hz.csv": 4,    # ICMP+UDP攻击
        "icmp_dns_40Hz.csv": 5,    # ICMP+DNS攻击
        "udp_dns_40Hz.csv": 6      # UDP+DNS攻击
    }
    
    X_list = []  # 特征列表
    y_list = []  # 标签列表
    
    print("开始加载数据...")
    
    for filename, label in attack_mapping.items():
        file_path = data_dir / filename
        if file_path.exists():
            # 读取CSV文件
            df = pd.read_csv(file_path)
            df.columns = ['时间(s)', '电流(mA)']  # 重命名列
            
            # 提取特征：使用滑动窗口提取统计特征
            window_size = 100  # 窗口大小
            step_size = 50     # 步长
            
            for i in range(0, len(df) - window_size + 1, step_size):
                window = df['电流(mA)'].iloc[i:i+window_size]
                
                # 提取统计特征
                features = [
                    window.mean(),      # 均值
                    window.std(),       # 标准差
                    window.max(),       # 最大值
                    window.min(),       # 最小值
                    window.median(),    # 中位数
                    window.skew(),      # 偏度
                    window.kurtosis()   # 峰度
                ]
                
                X_list.append(features)
                y_list.append(label)
            
            print(f"已处理 {filename}: {len(df)} 个数据点 -> {len(X_list) - sum(y_list)} 个样本")
        else:
            print(f"警告：文件 {filename} 不存在")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n总样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    return X, y

def train_dl_model(X, y, model_type='cnn'):
    """
    训练深度学习模型
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建数据集
    train_dataset = CurrentDataset(X_train_scaled, y_train)
    test_dataset = CurrentDataset(X_test_scaled, y_test)
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 选择模型
    if model_type == 'cnn':
        model = CNNClassifier(input_dim=X.shape[1], num_classes=7)
        model_name = "CNN"
    else:
        model = MLPClassifier(input_dim=X.shape[1], num_classes=7)
        model_name = "MLP"
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练参数
    num_epochs = 50
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\n开始训练{model_name}模型...")
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # 计算训练准确率
        train_acc = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # 评估模式
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # 最终测试
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    final_accuracy = accuracy_score(all_targets, all_preds)
    
    return {
        'model': model,
        'accuracy': final_accuracy,
        'y_pred': np.array(all_preds),
        'y_test': np.array(all_targets),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'model_name': model_name
    }

def plot_dl_results(results):
    """
    绘制深度学习结果可视化
    """
    # 类别名称
    class_names = ["正常", "TCP+ICMP", "TCP+UDP", "TCP+DNS", "ICMP+UDP", "ICMP+DNS", "UDP+DNS"]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练损失曲线
    axes[0, 0].plot(results['train_losses'], label='训练损失')
    axes[0, 0].set_title(f'{results["model_name"]} - 训练损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    axes[0, 1].plot(results['train_accuracies'], label='训练准确率')
    axes[0, 1].plot(results['test_accuracies'], label='测试准确率')
    axes[0, 1].set_title(f'{results["model_name"]} - 准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 混淆矩阵
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title(f'{results["model_name"]} - 混淆矩阵')
    axes[1, 0].set_xlabel('预测标签')
    axes[1, 0].set_ylabel('真实标签')
    
    # 4. 各类别准确率
    class_accuracies = []
    for i in range(len(class_names)):
        mask = results['y_test'] == i
        if mask.sum() > 0:
            class_acc = (results['y_pred'][mask] == i).mean()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    axes[1, 1].bar(class_names, class_accuracies, color='lightcoral')
    axes[1, 1].set_title(f'{results["model_name"]} - 各类别准确率')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'dl_{results["model_name"].lower()}_7class_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== 基于深度学习的七分类攻击检测 ===")
    
    # 加载数据
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("未找到有效数据！")
        return
    
    # 训练CNN模型
    cnn_results = train_dl_model(X, y, model_type='cnn')
    
    # 训练MLP模型
    mlp_results = train_dl_model(X, y, model_type='mlp')
    
    # 显示结果
    print("\n=== 最终结果 ===")
    print(f"CNN模型准确率: {cnn_results['accuracy']:.4f}")
    print(f"MLP模型准确率: {mlp_results['accuracy']:.4f}")
    
    # 绘制结果
    plot_dl_results(cnn_results)
    plot_dl_results(mlp_results)
    
    print("\n分析完成！结果已保存到 dl_cnn_7class_results.png 和 dl_mlp_7class_results.png")

if __name__ == "__main__":
    main()