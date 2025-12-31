'''数据处理采用混合采样（上采样少数类+下采样多数类）使用逻辑回归分类器进行分类'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    
    print("数据基本信息：")
    print(f"数据形状: {data.shape}")
    print(f"标签分布:\n{data['label'].value_counts()}")
    print(f"标签比例: {data['label'].value_counts(normalize=True)}")
    
    return data

# 数据预处理
def preprocess_data(data):
    """
    数据预处理
    """
    # 分离特征和标签
    X = data[['feature']].values
    y = data['label'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# 处理数据不平衡问题 - SMOTEENN混合采样
def handle_imbalance_combined(X, y):
    """
    处理数据不平衡问题 - SMOTEENN混合采样
    SMOTEENN = SMOTE上采样 + Edited Nearest Neighbours下采样
    """
    print(f"处理前 - 总样本数: {len(y)}")
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

# 训练逻辑回归模型
def train_lr_model(X_train, X_test, y_train, y_test):
    """
    训练逻辑回归模型并评估性能
    """
    # 创建逻辑回归分类器
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    # 训练模型
    print("开始训练逻辑回归模型...")
    lr_model.fit(X_train, y_train)
    
    # 预测
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
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
    plt.title('逻辑回归模型ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存图片到fig文件夹
    plt.savefig('fig/roc_lr_combined.png', dpi=300, bbox_inches='tight')
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
    plt.title('Confusion Table of Logistic Regression')
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
    plt.savefig('fig/confusion_matrix_lr_combined.png', bbox_inches='tight', dpi=300)
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
    主函数：执行完整的逻辑回归模型训练流程（混合采样）
    """
    print("=== 逻辑回归二分类模型训练（混合采样） ===\n")
    
    # 1. 加载数据
    print("步骤1: 加载数据")
    data = load_data()
    
    # 2. 数据预处理
    print("\n步骤2: 数据预处理")
    X, y = preprocess_data(data)
    
    # 3. 处理数据不平衡（混合采样）
    print("\n步骤3: 处理数据不平衡（混合采样）")
    X_balanced, y_balanced = handle_imbalance_combined(X, y)
    
    # 4. 划分训练集和测试集
    print("\n步骤4: 划分训练集和测试集")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_balanced
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
    
    # 5. 训练逻辑回归模型
    print("\n步骤5: 训练逻辑回归模型")
    y_pred, y_pred_proba = train_lr_model(X_train, X_test, y_train, y_test)
    
    # 6. 绘制ROC曲线
    print("\n步骤6: 绘制ROC曲线")
    plot_roc_curve(y_test, y_pred_proba)
    
    # 7. 绘制混淆矩阵
    print("\n步骤7: 绘制混淆矩阵")
    plot_confusion_matrix(y_test, y_pred)
    
    print("\n=== 模型训练完成 ===")
    print("ROC曲线已保存为: fig/roc_lr_combined.png")
    print("混淆矩阵已保存为: fig/confusion_matrix_lr_combined.png")

if __name__ == "__main__":
    main()