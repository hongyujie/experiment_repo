'''SVM模型 - 考虑数据的时序特性（修复数据泄露）''' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
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

# 时序数据处理 - 创建滑动窗口序列
def create_temporal_sequences(data, window_size=10):
    """
    创建时序序列数据
    使用滑动窗口方法，每个样本包含连续的多个时间点的feature值
    
    参数:
        data: 原始数据
        window_size: 滑动窗口大小
        
    返回:
        X_seq: 时序序列特征
        y_seq: 时序序列标签
    """
    print(f"\n创建时序序列数据，窗口大小: {window_size}")
    
    # 获取原始数据
    features = data['feature'].values
    labels = data['label'].values
    
    X_seq = []
    y_seq = []
    
    # 使用滑动窗口创建序列
    for i in range(len(features) - window_size + 1):
        # 获取当前窗口的特征
        seq_feature = features[i:i+window_size]
        
        # 计算当前窗口的标签（使用窗口内出现次数最多的标签）
        seq_label = np.bincount(labels[i:i+window_size].astype(int)).argmax()
        
        X_seq.append(seq_feature)
        y_seq.append(seq_label)
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"时序序列数据形状: {X_seq.shape}")
    print(f"时序序列标签形状: {y_seq.shape}")
    print(f"时序序列标签分布:\n{np.bincount(y_seq)}")
    print(f"时序序列标签比例: {np.bincount(y_seq)/len(y_seq)}")
    
    return X_seq, y_seq

# 按时间顺序划分训练集和测试集
def split_by_time(X_seq, y_seq, test_size=0.2):
    """
    按时间顺序划分训练集和测试集
    确保训练集在时间上先于测试集
    
    参数:
        X_seq: 时序序列特征
        y_seq: 时序序列标签
        test_size: 测试集比例
        
    返回:
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
    """
    print(f"\n按时间顺序划分训练集和测试集，测试集比例: {test_size}")
    
    # 计算分割点
    split_point = int(len(X_seq) * (1 - test_size))
    
    # 按时间顺序分割
    X_train = X_seq[:split_point]
    X_test = X_seq[split_point:]
    y_train = y_seq[:split_point]
    y_test = y_seq[split_point:]
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test

# 处理数据不平衡问题 - SMOTEENN混合采样
def handle_imbalance_combined(X, y):
    """
    处理数据不平衡问题 - SMOTEENN混合采样
    SMOTEENN = SMOTE上采样 + Edited Nearest Neighbours下采样
    """
    print(f"\n处理前 - 总样本数: {len(y)}")
    print(f"处理前 - 标签分布:\n{pd.Series(y).value_counts()}")
    
    # 创建SMOTEENN采样器
    # 分别创建SMOTE和ENN对象
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=42,
        k_neighbors=5
    )
    
    enn = EditedNearestNeighbours(
        sampling_strategy='auto',
        kind_sel='all',
        n_neighbors=3
    )
    
    # 创建SMOTEENN组合采样器
    smoteenn = SMOTEENN(
        smote=smote,
        enn=enn,
        random_state=42
    )
    
    # 应用SMOTEENN采样
    X_resampled, y_resampled = smoteenn.fit_resample(X, y)
    
    print(f"处理后 - 总样本数: {len(y_resampled)}")
    print(f"处理后 - 标签分布:\n{pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

# 训练SVM模型
def train_svm_model(X_train, X_test, y_train, y_test):
    """
    训练SVM模型并评估性能
    使用时序序列数据
    """
    # 创建SVM分类器
    # 针对时序数据，使用RBF核函数
    svm_model = SVC(
        kernel='rbf', 
        probability=True, 
        random_state=42, 
        C=1.0,  # 正则化参数
        gamma='scale'  # 核系数
    )
    
    # 训练模型
    print("\n开始训练SVM模型...")
    svm_model.fit(X_train, y_train)
    
    # 预测
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
    
    # 评估模型
    print("\n模型评估结果：")
    print(classification_report(y_test, y_pred))
    
    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"准确率: {accuracy:.4f}")
    
    return y_pred, y_pred_proba

# 绘制ROC曲线
def plot_roc_curve(y_test, y_pred_proba):
    """
    绘制ROC曲线
    """
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('SVM模型ROC曲线（时序数据）')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存图片到fig文件夹
    plt.savefig('fig/roc_svm_temporal.png', dpi=300, bbox_inches='tight')
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
    plt.title('Confusion Table of SVM (Temporal Data)')
    plt.xlabel('Predict label')
    plt.ylabel('Truth label')
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
    plt.savefig('fig/confusion_matrix_svm_temporal.png', bbox_inches='tight', dpi=300)
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
    主函数：执行完整的SVM模型训练流程（修复数据泄露）
    正确的数据处理顺序：
    1. 加载数据并按时间排序
    2. 使用滑动窗口生成时序序列
    3. 按时间顺序划分训练集和测试集
    4. 对训练集单独进行标准化
    5. 使用训练集的标准化参数对测试集进行转换
    6. 对训练集进行混合采样
    7. 训练模型并评估
    """
    print("=== SVM二分类模型训练（修复数据泄露） ===\n")
    
    # 1. 加载数据并按时间排序
    print("步骤1: 加载数据并按时间排序")
    data = load_data()
    
    # 2. 使用滑动窗口生成时序序列
    print("\n步骤2: 使用滑动窗口生成时序序列")
    X_seq, y_seq = create_temporal_sequences(data, window_size=10)
    
    # 3. 按时间顺序划分训练集和测试集
    print("\n步骤3: 按时间顺序划分训练集和测试集")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_by_time(X_seq, y_seq, test_size=0.2)
    
    print(f"训练集大小: {X_train_raw.shape[0]}")
    print(f"测试集大小: {X_test_raw.shape[0]}")
    print(f"训练集标签分布: {np.unique(y_train_raw, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test_raw, return_counts=True)}")
    
    # 4. 对训练集单独进行标准化
    print("\n步骤4: 对训练集进行标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # 5. 使用训练集的标准化参数对测试集进行转换
    print("步骤5: 使用训练集的标准化参数对测试集进行转换")
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 6. 仅对训练集进行混合采样处理
    print("\n步骤6: 对训练集进行混合采样处理")
    X_train_resampled, y_train_resampled = handle_imbalance_combined(X_train_scaled, y_train_raw)
    
    # 7. 训练SVM模型
    print("\n步骤7: 训练SVM模型")
    y_pred, y_pred_proba = train_svm_model(
        X_train_resampled, X_test_scaled,  # 使用处理后的训练集和原始分布的测试集
        y_train_resampled, y_test_raw
    )
    
    # 8. 绘制ROC曲线
    print("\n步骤8: 绘制ROC曲线")
    plot_roc_curve(y_test_raw, y_pred_proba)
    
    # 9. 绘制混淆矩阵
    print("\n步骤9: 绘制混淆矩阵")
    plot_confusion_matrix(y_test_raw, y_pred)
    
    print("\n=== 模型训练完成 ===")
    print("ROC曲线已保存为: fig/roc_svm_temporal.png")
    print("混淆矩阵已保存为: fig/confusion_matrix_svm_temporal.png")

if __name__ == "__main__":
    main()