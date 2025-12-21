'''SVM for Time Series Classification - Based on CNN.py Structure'''

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import os

# 超参数
model_name = "SVM_CNN"  # 基于CNN结构的SVM模型
batch_size = 128
length = 64
random_state = 42

# label_true->真实标签，label_pred->预测标签，label_name->标签名称，title->标题，pdf_save_path->保存路径，dpi->分辨率
def heatmap(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
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

    if not pdf_save_path is None:
        # 确保目录存在
        dir_path = os.path.dirname(pdf_save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)  # 创建目录及任何必要的中间目录
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.close()

# first->第一个列表，y_label->y轴标签，title->标题，num_iterations->迭代次数，x_label->x轴标签，second->第二个列表，third->第三个列表，show_min->是否显示最大值，show_max->是否显示最大值
def plots(first, y_label, title, num_iterations, x_label='iterations', second=None, third=None, show_min=True, show_max=True):
    fig = plt.figure()
    fig.gca().set_position((.15, .3, .80, .6))
    plt.ylabel(y_label + "->")
    t = np.linspace(0, num_iterations - 1, num=num_iterations)
    plt.figtext(.5, .92, title, fontsize=14, ha='center', fontweight='bold')

    first_list, first_list_name = first
    first_array = np.array(first_list)
    (first_max_x, first_max_y) = (round((np.argmax(first_array)), 3), round(np.amax(first_array), 3))

    if show_min:
        plt.scatter(first_max_x, first_max_y, c='b',
                    label='max_' + first_list_name + '(' + str(first_max_x) + ',' + str(first_max_y) + ')')

    plt.plot(t, np.squeeze(first_list), 'b-', linewidth=1.5, label=first_list_name)
    if second is not None:
        second_list, second_list_name = second
        plt.plot(t, np.squeeze(second_list), 'r-', linewidth=1.5, label=second_list_name)
    if third is not None:
        third_list, third_list_name = third
        plt.plot(t, np.squeeze(third_list), 'g-', linewidth=1.5, label=third_list_name)

    plt.legend(bbox_to_anchor=(0.4, -0.15))
    plt.savefig("./fig/p_" + model_name + ".png")
    plt.show()
    plt.close()

# ----------准确率类----------
class AccMectric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0
    # targets->真实标签，outputs->预测标签
    def update(self, targets, outputs):
        pred = outputs.argmax(axis=1) if outputs.ndim > 1 else outputs
        self._sum += (pred == targets).sum()
        self._count += targets.shape[0]

    def get(self):
        return self._sum / self._count

# ----------读取和处理数据----------
def read_data():
    # 使用相对路径读取CSV文件
    path = r"./password data-2.csv"# ./代表从当前目录开始查找
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    return data


def get_seq_data(length):
    seq_data, seq_labels = [], []
    d = read_data()
    n_samples = d.shape[0] - length + 1
    for j in range(n_samples):# 滑动窗口取样本
        data = d[j: j + length, 2]  # 第3列是功耗/电流
        seq_data.append(data)
        bincount = np.bincount(d[j: j + length, 1].astype(int))  # 第二列是标签
        # np.bincount统计标签出现次数，返回的是一个向量，例如[1, 0, 2]，表示标签0出现1次，标签1出现0次，标签2出现2次
        label = np.argmax(bincount)
        # 找到出现次数最多的标签作为该序列的标签
        seq_labels.append(label)
    return np.array(seq_data), np.array(seq_labels)
    # seq_data.shape: (n_samples, length), seq_labels.shape: (n_samples,)

# ----------模型----------
# 替换为SVM模型
def create_svm_model():
    """
    创建SVM模型
    使用RBF核函数，适合处理序列数据
    """
    svm_model = SVC(
        kernel='rbf',  # 使用RBF核处理非线性关系
        probability=True,  # 启用概率估计
        random_state=random_state,  # 固定随机种子
        C=1.0,  # 正则化参数
        gamma='scale'  # 核系数
    )
    return svm_model

# ----------训练与验证----------
def train_svm(model, train_data, train_labels):
    """
    训练SVM模型
    """
    print("开始训练SVM模型...")
    model.fit(train_data, train_labels)
    
    # 计算训练准确率
    train_pred = model.predict(train_data)
    train_acc = AccMectric()
    train_acc.update(train_labels, train_pred)
    
    return train_acc.get()


def validate_svm(model, test_data, test_labels):
    """
    验证SVM模型
    """
    print("开始验证SVM模型...")
    # 预测
    pred = model.predict(test_data)
    
    # 计算准确率
    acc = AccMectric()
    acc.update(test_labels, pred)
    
    return (acc.get(), np.asarray(pred), np.asarray(test_labels))


# ----------主程序----------
seq_data, seq_labels = get_seq_data(length)
# 划分测试数据集和训练数据集，20%作为测试集，80%作为训练集
train_data, test_data, train_labels, test_labels = train_test_split(seq_data, seq_labels, test_size=0.2, random_state=42)

# SVM不需要扩展维度，直接使用原始序列数据
# train_data = np.expand_dims(train_data, axis=1)
# test_data = np.expand_dims(test_data, axis=1)

print("训练数据形状:", train_data.shape)
print("测试数据形状:", test_data.shape)

# 创建SVM模型
model = create_svm_model()

# 训练模型
train_acc = train_svm(model, train_data, train_labels)

# 验证模型
test_acc, pred, targets = validate_svm(model, test_data, test_labels)

print('{}\n训练准确率: {:0.3f}\n测试准确率: {:0.3f}'.format(datetime.now(), train_acc, test_acc))

print(f"\n=== 测试准确率: {test_acc:.3f} ===")

# 输出最终结果
print(f"\n最终结果:")
print(f"训练集准确率: {train_acc:.3f}")
print(f"测试集准确率: {test_acc:.3f}")

# 绘制混淆矩阵
heatmap(label_true=targets,
        label_pred=pred,
        label_name=[0, 1],
        title="Confusion Table of SVM_CNN",
        pdf_save_path="./fig/h_" + model_name + ".png",
        dpi=300)

# 绘制准确率变化曲线 (SVM只需要一个点，这里简化处理)
plots(([train_acc, test_acc], 'accuracy'), 'accuracy',
      'Accuracy Comparison for ' + model_name, num_iterations=2,
      x_label='Phase', second=None, third=None, show_min=True)