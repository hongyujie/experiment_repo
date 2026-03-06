import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据文件路径
data_dir = Path("data")

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

def train_ml_models(X, y):
    """
    训练多个机器学习模型并比较性能
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
        "支持向量机": SVC(kernel='rbf', random_state=42),
        "逻辑回归": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    print("\n开始训练模型...")
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred
        }
        
        print(f"{name} 准确率: {accuracy:.4f}")
    
    return results, X_test, y_test

def plot_results(results, X_test, y_test):
    """
    绘制结果可视化
    """
    # 类别名称
    class_names = ["正常", "TCP+ICMP", "TCP+UDP", "TCP+DNS", "ICMP+UDP", "ICMP+DNS", "UDP+DNS"]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 模型准确率比较
    accuracies = [results[name]['accuracy'] for name in results.keys()]
    axes[0, 0].bar(results.keys(), accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title('模型准确率比较')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)
    
    # 添加准确率数值标签
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 最佳模型的混淆矩阵（选择准确率最高的模型）
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model_name]
    
    cm = confusion_matrix(y_test, best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
    axes[0, 1].set_title(f'{best_model_name} - 混淆矩阵')
    axes[0, 1].set_xlabel('预测标签')
    axes[0, 1].set_ylabel('真实标签')
    
    # 3. 分类报告热力图
    report = classification_report(y_test, best_result['y_pred'], target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1]  # 排除支持度和平均值
    
    sns.heatmap(report_df, annot=True, cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_title(f'{best_model_name} - 分类报告')
    
    # 4. 各类别准确率
    class_accuracies = []
    for i in range(len(class_names)):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (best_result['y_pred'][mask] == i).mean()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    axes[1, 1].bar(class_names, class_accuracies, color='lightgreen')
    axes[1, 1].set_title(f'{best_model_name} - 各类别准确率')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('ml_7class_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== 基于机器学习的七分类攻击检测 ===")
    
    # 加载数据
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("未找到有效数据！")
        return
    
    # 训练模型
    results, X_test, y_test = train_ml_models(X, y)
    
    # 显示结果
    print("\n=== 最终结果 ===")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    # 绘制结果
    plot_results(results, X_test, y_test)
    
    print("\n分析完成！结果已保存到 ml_7class_results.png")

if __name__ == "__main__":
    main()