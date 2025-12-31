import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_data():
    """加载密码强度数据"""
    # 读取数据，没有列名，手动指定列名
    df = pd.read_csv('password data-2.csv', header=None, names=['timestamp', 'label', 'value'])
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查数据基本信息
    print("\n数据基本信息:")
    print(df.info())
    
    # 检查标签分布
    print("\n标签分布:")
    print(df['label'].value_counts())
    
    # 提取特征和标签
    # 使用value列作为特征，label列作为标签
    X = df[['value']].values  # 使用value作为特征
    y = df['label'].values
    
    print(f"\n二分类标签分布:")
    print(pd.Series(y).value_counts())
    
    return X, y

# 数据标准化
def standardize_data(X, y):
    """标准化特征数据"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"标准化后特征范围: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    
    return X_scaled, y

# 计算类别权重
def calculate_class_weights(y):
    """
    计算类别权重
    权重与类别样本数成反比，样本越少的类别权重越大
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # 获取类别分布
    class_counts = pd.Series(y).value_counts()
    print(f"类别分布: 0类={class_counts[0]}个, 1类={class_counts[1]}个")
    
    # 计算类别权重
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    # 转换为字典格式
    class_weights = dict(zip(classes, weights))
    
    print(f"计算得到的类别权重: {class_weights}")
    
    # 手动计算验证
    n_samples = len(y)
    n_classes = len(classes)
    
    # 手动计算平衡权重
    manual_weights = {}
    for class_label in classes:
        n_class_samples = np.sum(y == class_label)
        weight = n_samples / (n_classes * n_class_samples)
        manual_weights[class_label] = weight
    
    print(f"手动计算的权重: {manual_weights}")
    
    return class_weights

# 使用类别权重训练SVM模型
def train_svm_with_weights(X_train, X_test, y_train, y_test):
    """使用类别权重训练SVM模型"""
    
    # 计算类别权重
    class_weights = calculate_class_weights(y_train)
    
    print("\n开始训练带类别权重的SVM模型...")
    
    # 创建带类别权重的SVM模型
    svm_model = SVC(
        kernel='rbf',
        C=1.0,  # 正则化参数
        gamma='scale',
        probability=True,  # 启用概率预测，用于计算AUC
        class_weight=class_weights,  # 使用类别权重
        random_state=42
    )
    
    # 训练模型
    svm_model.fit(X_train, y_train)
    
    # 预测
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]  # 正类的概率
    
    # 评估模型
    print("\n" + "="*50)
    print("带类别权重的SVM模型评估结果:")
    print("="*50)
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 计算各项指标
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n详细指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    
    # 计算AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC值: {roc_auc:.4f}")
    
    return svm_model, y_pred, y_pred_proba, fpr, tpr, roc_auc

# 混淆矩阵热力图函数（参考cnn.py的实现）
def heatmap(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """
    label_true->真实标签，label_pred->预测标签，label_name->标签名称，
    title->标题，pdf_save_path->保存路径，dpi->分辨率
    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        # 确保目录存在
        import os
        dir_path = os.path.dirname(pdf_save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)  # 创建目录及任何必要的中间目录
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

# 可视化结果
def visualize_results(y_test, y_pred, y_pred_proba, fpr, tpr, roc_auc):
    """可视化模型结果"""
    
    # 1. 混淆矩阵热力图 - 使用cnn.py的格式
    plt.figure(figsize=(8, 6))
    heatmap(label_true=y_test,
            label_pred=y_pred,
            label_name=[0, 1],
            title="Confusion Matrix - SVM with Class Weights",
            pdf_save_path='fig/confusion_matrix_svm_weights.png',
            dpi=300)
    plt.show()
    
    # 2. ROC曲线 - 单独一张图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线 - SVM with Class Weights')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig/roc_svm_weights.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    """主函数"""
    print("="*60)
    print("SVM模型 - 类别权重方法处理不平衡数据")
    print("="*60)
    
    # 1. 加载数据
    X, y = load_data()
    
    # 2. 数据标准化
    X_scaled, y = standardize_data(X, y)
    
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 4. 训练带类别权重的SVM模型
    model, y_pred, y_pred_proba, fpr, tpr, roc_auc = train_svm_with_weights(
        X_train, X_test, y_train, y_test
    )
    
    # 5. 可视化结果
    visualize_results(y_test, y_pred, y_pred_proba, fpr, tpr, roc_auc)
    
    print("\n" + "="*60)
    print("模型训练完成！")
    print("="*60)

if __name__ == "__main__":
    main()