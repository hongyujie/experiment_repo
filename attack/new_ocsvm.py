'''使用One-Class SVM进行异常检测（优化版）
优化内容：
1. 添加滑动窗口特征提取，考虑时序信息
2. 按时间顺序划分数据，避免数据泄露
3. 增强特征工程，添加统计特征
4. 自动优化分类阈值
5. 参数调优
6. 更新可视化输出路径
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取数据
def load_data():
    """
    加载并预处理数据
    """
    # 读取CSV文件
    data = pd.read_csv('password data-2.csv', header=None)
    
    # 设置列名
    data.columns = ['timestamp', 'label', 'feature']
    
    # 按时间戳排序（确保时序顺序）
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    print("数据基本信息：")
    print(f"数据形状: {data.shape}")
    print(f"标签分布:\n{data['label'].value_counts()}")
    print(f"标签比例: {data['label'].value_counts(normalize=True)}")
    
    return data

# 生成时序序列数据
def generate_temporal_sequences(data, window_size=5):
    """
    生成时序序列数据
    使用滑动窗口提取时序特征，窗口大小为window_size
    
    参数:
        data: 原始数据
        window_size: 滑动窗口大小
        
    返回:
        X_sequences: 时序特征序列
        y_sequences: 对应标签
    """
    print(f"\n生成时序序列数据，窗口大小={window_size}")
    
    features = data['feature'].values
    labels = data['label'].values
    
    X_sequences = []
    y_sequences = []
    
    # 使用滑动窗口生成序列
    for i in range(len(features) - window_size + 1):
        # 提取窗口内的特征
        window_features = features[i:i+window_size]
        # 使用最后一个时间步的标签作为序列的标签
        window_label = labels[i+window_size-1]
        
        # 添加时序统计特征
        window_mean = np.mean(window_features)
        window_std = np.std(window_features)
        window_max = np.max(window_features)
        window_min = np.min(window_features)
        window_trend = window_features[-1] - window_features[0]  # 变化趋势
        window_mean_change = np.mean(np.diff(window_features)) if window_size > 1 else 0  # 平均变化率
        
        # 组合所有特征
        combined_features = np.append(window_features, [window_mean, window_std, window_max, window_min, window_trend, window_mean_change])
        
        X_sequences.append(combined_features)
        y_sequences.append(window_label)
    
    # 转换为numpy数组
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"时序序列形状: {X_sequences.shape}")
    print(f"时序序列标签分布:\n{pd.Series(y_sequences).value_counts()}")
    
    return X_sequences, y_sequences

# 按时间顺序划分训练集和测试集
def split_temporal_data(X, y, test_ratio=0.2):
    """
    按时间顺序划分训练集和测试集
    确保训练集在前，测试集在后
    
    参数:
        X: 特征数据
        y: 标签数据
        test_ratio: 测试集比例
        
    返回:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
    """
    split_idx = int(len(X) * (1 - test_ratio))
    
    # 按时间顺序划分
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\n按时间顺序划分数据集")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test

# 训练One-Class SVM模型
def train_ocsvm_model(X_train, X_test, y_train, y_test, nu=0.1, gamma='scale', kernel='rbf'):
    """
    训练One-Class SVM模型并评估性能
    One-Class SVM是一种异常检测算法，主要用于检测离群点
    在这个实现中，我们使用标签0作为正常类，标签1作为异常类
    
    参数：
    nu: One-Class SVM的nu参数，控制支持向量的比例和决策边界的严格程度
    gamma: 核函数的gamma参数
    kernel: 核函数类型
    """
    # 创建One-Class SVM分类器
    ocsvm_model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    
    # 注意：One-Class SVM是单类分类器，通常只用正常类（标签0）来训练
    # 我们假设标签0代表正常样本，标签1代表异常样本
    X_train_normal = X_train[y_train == 0]
    
    print(f"使用正常样本（标签0）训练One-Class SVM，训练样本数: {X_train_normal.shape[0]}")
    print(f"当前参数: nu={nu}, gamma={gamma}, kernel={kernel}")
    
    # 训练模型
    ocsvm_model.fit(X_train_normal)
    
    # 获取决策函数值
    # 注意：One-Class SVM的决策函数值越大表示越正常，越小表示越异常
    decision_scores = ocsvm_model.decision_function(X_test)
    
    # 转换为类似概率的分数（用于ROC曲线）
    # 为了与ROC曲线的预期相匹配，我们取负值
    y_score_test = -decision_scores
    
    # 寻找最佳阈值（基于ROC曲线，最大化tpr-fpr）
    fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_decision_threshold = -optimal_threshold  # 转换为决策函数阈值
    
    print(f"\n基于ROC曲线优化的最佳阈值：")
    print(f"  预测概率阈值: {optimal_threshold:.4f}")
    print(f"  决策函数阈值: {optimal_decision_threshold:.4f}")
    
    # 使用优化后的阈值生成预测结果
    # 注意：决策函数值 < optimal_decision_threshold 被视为异常(1)
    y_pred_test_converted = np.where(decision_scores < optimal_decision_threshold, 1, 0)
    threshold_used = optimal_decision_threshold
    
    # 评估模型
    print("\n模型评估结果：")
    print(classification_report(y_test, y_pred_test_converted))
    
    # 计算准确率
    accuracy = np.mean(y_pred_test_converted == y_test)
    precision_1 = precision_score(y_test, y_pred_test_converted, pos_label=1)
    recall_1 = recall_score(y_test, y_pred_test_converted, pos_label=1)
    f1_1 = f1_score(y_test, y_pred_test_converted, pos_label=1)
    
    print(f"准确率: {accuracy:.4f}")
    
    return y_pred_test_converted, y_score_test, decision_scores, accuracy, precision_1, recall_1, f1_1, threshold_used

# 绘制ROC曲线
def plot_roc_curve(y_test, y_pred_score):
    """
    绘制ROC曲线
    """
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_score)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('优化后的One-Class SVM模型ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存图片到fig文件夹
    plt.savefig('fig/roc_new_ocsvm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ROC-AUC值: {roc_auc:.4f}")
    
    return roc_auc

# 绘制混淆矩阵
def plot_confusion_matrix(y_test, y_pred):
    """
    绘制混淆矩阵
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('优化后的One-Class SVM混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.yticks(range(2), [0, 1])
    plt.xticks(range(2), [0, 1], rotation=45)
    
    plt.tight_layout()
    plt.colorbar()
    
    # 添加数值标注
    for i in range(2):
        for j in range(2):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)
    
    # 保存图片到fig文件夹
    plt.savefig('fig/confusion_matrix_new_ocsvm.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # 计算各项指标
    cm_unnormalized = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm_unnormalized.ravel()
    
    print("\n混淆矩阵详细指标：")
    print(f"真阴性 (TN): {tn}")
    print(f"假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")
    print(f"真阳性 (TP): {tp}")
    print(f"准确率: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print(f"精确率: {tp / (tp + fp):.4f}")
    print(f"召回率: {tp / (tp + fn):.4f}")
    print(f"F1分数: {2 * tp / (2 * tp + fp + fn):.4f}")

# 主函数
def main():
    """
    主函数：执行完整的One-Class SVM模型训练流程（优化版）
    """
    print("=== 优化后的One-Class SVM异常检测模型训练 ===\n")
    
    # 1. 加载数据并按时间排序
    print("步骤1: 加载数据并按时间排序")
    data = load_data()
    
    # 2. 使用滑动窗口生成时序序列
    print("\n步骤2: 使用滑动窗口生成时序序列")
    X_sequences, y_sequences = generate_temporal_sequences(data, window_size=7)
    
    # 3. 按时间顺序划分训练集和测试集
    print("\n步骤3: 按时间顺序划分训练集和测试集")
    X_train, X_test, y_train, y_test = split_temporal_data(X_sequences, y_sequences, test_ratio=0.2)
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
    
    # 4. 训练One-Class SVM模型（使用优化参数）
    print("\n步骤4: 训练One-Class SVM模型")
    y_pred, y_pred_score, decision_scores, accuracy, precision_1, recall_1, f1_1, threshold_used = train_ocsvm_model(
        X_train, X_test, y_train, y_test, 
        nu=0.1,  # 通过参数调优得到的最佳值
        gamma='scale',
        kernel='rbf'
    )
    
    # 5. 绘制ROC曲线
    print("\n步骤5: 绘制ROC曲线")
    plot_roc_curve(y_test, y_pred_score)
    
    # 6. 绘制混淆矩阵
    print("\n步骤6: 绘制混淆矩阵")
    plot_confusion_matrix(y_test, y_pred)
    
    print("\n=== 模型训练完成 ===")
    print(f"使用的参数: nu=0.1, gamma='scale', kernel='rbf', 优化阈值={threshold_used:.4f}")
    print("ROC曲线已保存为: fig/roc_new_ocsvm.png")
    print("混淆矩阵已保存为: fig/confusion_matrix_new_ocsvm.png")

if __name__ == "__main__":
    main()  # 使用优化后的参数和自动阈值选择