'''
使用One-Class SVM进行异常检测（原始数据）
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



# 训练One-Class SVM模型
def train_ocsvm_model(X_train, X_test, y_train, y_test, nu=0.1, custom_threshold=None):
    """
    训练One-Class SVM模型并评估性能
    One-Class SVM是一种异常检测算法，主要用于检测离群点
    在这个实现中，我们使用标签0作为正常类，标签1作为异常类
    
    参数：
    nu: One-Class SVM的nu参数，控制支持向量的比例和决策边界的严格程度
        默认0.05（通过网格搜索得到的最佳参数）
    custom_threshold: 自定义阈值，默认None使用模型默认阈值0
                     注意：决策函数值 < threshold 被视为异常(1)
    """
    # 创建One-Class SVM分类器
    ocsvm_model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    
    # 注意：One-Class SVM是单类分类器，通常只用正常类（标签0）来训练
    # 我们假设标签0代表正常样本，标签1代表异常样本
    X_train_normal = X_train[y_train == 0]
    
    print(f"使用正常样本（标签0）训练One-Class SVM，训练样本数: {X_train_normal.shape[0]}")
    print(f"当前nu参数: {nu}")
    
    # 训练模型
    ocsvm_model.fit(X_train_normal)
    
    # 获取决策函数值
    # 注意：One-Class SVM的决策函数值越大表示越正常，越小表示越异常
    decision_scores = ocsvm_model.decision_function(X_test)
    
    # 根据阈值生成预测结果
    if custom_threshold is None:
        # 使用默认阈值0
        y_pred_test = ocsvm_model.predict(X_test)
        y_pred_test_converted = np.where(y_pred_test == 1, 0, 1)
        threshold_used = 0
        print(f"使用默认阈值0进行预测")
    else:
        # 使用自定义阈值
        # 决策函数值 < threshold 被视为异常(1)
        y_pred_test_converted = np.where(decision_scores < custom_threshold, 1, 0)
        threshold_used = custom_threshold
        print(f"使用自定义阈值{custom_threshold}进行预测")
    
    # 将决策函数值转换为类似概率的分数（用于ROC曲线）
    # 为了与ROC曲线的预期相匹配，我们取负值
    y_score_test = -decision_scores
    
    # 评估模型
    print("\n模型评估结果：")
    print(classification_report(y_test, y_pred_test_converted))
    
    # 计算准确率
    accuracy = np.mean(y_pred_test_converted == y_test)
    precision_1 = precision_score(y_test, y_pred_test_converted, pos_label=1)
    recall_1 = recall_score(y_test, y_pred_test_converted, pos_label=1)
    f1_1 = f1_score(y_test, y_pred_test_converted, pos_label=1)
    
    print(f"准确率: {accuracy:.4f}")
    
    # 返回预测结果、决策分数和决策函数原始值
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
    plt.title('OCSVM模型ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存图片到fig文件夹
    plt.savefig('fig/roc_ocsvm_original.png', dpi=300, bbox_inches='tight')
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
    plt.title('Confusion Table of OCSVM')
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
    plt.savefig('fig/confusion_matrix_ocsvm_original.png', bbox_inches='tight', dpi=300)
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
def main(threshold=-1.0):
    """
    主函数：执行完整的One-Class SVM模型训练流程（原始数据）
    
    参数：
    threshold: 自定义阈值，默认3.5（通过阈值搜索得到的最佳阈值）
               注意：决策函数值 < threshold 被视为异常(1)
    """
    print("=== One-Class SVM异常检测模型训练（原始数据） ===\n")
    
    # 1. 加载数据
    print("步骤1: 加载数据")
    data = load_data()
    
    # 2. 数据预处理
    print("\n步骤2: 数据预处理")
    X, y = preprocess_data(data)
    
    # 3. 划分训练集和测试集
    print("\n步骤3: 划分训练集和测试集")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"训练集标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集标签分布: {np.unique(y_test, return_counts=True)}")
    
    # 4. 训练One-Class SVM模型
    print("\n步骤4: 训练One-Class SVM模型")
    y_pred, y_pred_score, decision_scores, accuracy, precision_1, recall_1, f1_1, threshold_used = train_ocsvm_model(
        X_train, X_test, y_train, y_test, 
        custom_threshold=threshold
    )
    
    # 5. 绘制ROC曲线
    print("\n步骤5: 绘制ROC曲线")
    plot_roc_curve(y_test, y_pred_score)
    
    # 6. 绘制混淆矩阵
    print("\n步骤6: 绘制混淆矩阵")
    plot_confusion_matrix(y_test, y_pred)
    
    print("\n=== 模型训练完成 ===")
    print(f"使用的参数: 阈值={threshold_used}, nu=0.05")
    print("ROC曲线已保存为: fig/roc_ocsvm_original.png")
    print("混淆矩阵已保存为: fig/confusion_matrix_ocsvm_original.png")

if __name__ == "__main__":
    main()  # 使用默认最佳参数：threshold=3.5, nu=0.05