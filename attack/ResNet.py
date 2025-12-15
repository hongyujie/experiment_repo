import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.data as tchdata
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

# 超参数
model_name = "ResNet18"
batch_size = 128
epoch = 10
learning_rate = 0.001
dropout = 0
length = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # plt.show()
    if not pdf_save_path is None:
        # 确保目录存在
        dir_path = os.path.dirname(pdf_save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)  # 创建目录及任何必要的中间目录
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

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


# ----------准确率类----------
class AccMectric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0
    # targets->真实标签，outputs->预测标签
    def update(self, targets, outputs):
        pred = outputs.argmax(axis=1)
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

# 残差块
class Residual(torch.nn.Module):  
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = torch.nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv1d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = torch.nn.BatchNorm1d(num_channels)
        self.bn2 = torch.nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y) 

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
# torch.nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# 调用要加上*，因为这个函数返回的是一个列表，里面包含多个Residual，使用*操作符来解包这个列表，将列表中的每个元素作为单独的参数传递给torch.nn.Sequential。
    blk = []
    for i in range(num_residuals):
        # 这里设置这样的条件是因为，首先在每一个resnet_block中，包含两个Residual，前一个Residual用于高宽减半，通道数增加，后一个Residual用于通道数不变，大小不变。由于第一个resnet_block的输入已经将高宽减半了，所以直接进行后一个Residual用于通道数不变，大小不变。
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# ----------模型----------
class Net(torch.nn.Module):# ResNet_18
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block1 = torch.nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.block2 = torch.nn.Sequential(*resnet_block(64, 128, 2))
        self.block3 = torch.nn.Sequential(*resnet_block(128, 256, 2))
        self.block4 = torch.nn.Sequential(*resnet_block(256, 512, 2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2)
        )

        self.model = torch.nn.Sequential(
            self.layer1,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.layer2
        )
        
    def forward(self, x):
        return self.model(x)

# ----------训练与验证----------
def train(model, optimizer, train_loader):
    model.train()
    acc = AccMectric()
    for data, labels in train_loader:
        x = data.to(device)# 特征
        y = labels.to(device)# 标签
        o = model(x)# 输出
        loss = torch.nn.CrossEntropyLoss()(o, y)# 计算损失函数
        # numpy()是将数据转换成numpy类型
        acc.update(labels.numpy(), o.data.cpu().numpy())# 更新准确率
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return acc.get(), loss


def validate(model, test_loader):
    model.eval()
    acc = AccMectric()
    pred, targets = [], []
    with torch.no_grad():# 不计算梯度，因为在验证阶段不需要更新模型参数
        for data, labels in test_loader:
            x = data.to(device)
            o = model(x)
            outputs = o.data.cpu().numpy()
            acc.update(labels.numpy(), outputs)
            pred.extend(outputs.argmax(axis=1))
            targets.extend(labels.numpy())
            y = labels.to(device)
            loss = torch.nn.CrossEntropyLoss()(o, y)
        return (acc.get(), np.asarray(pred), np.asarray(targets), loss)


# ----------主程序----------
seq_data, seq_labels = get_seq_data(length)
# 划分测试数据集和训练数据集，20%作为测试集，80%作为训练集
train_data, test_data, train_labels, test_labels = train_test_split(seq_data, seq_labels, test_size=0.2, random_state=42)

# 增加数据集的维度，将原本（样本数量，特征数量）增加成（样本数量，1，特征数量）
# 满足cnn的输入尺寸，例如对于图像的话，输入为（样本数量，通道数，高度，宽度）
train_data = np.expand_dims(train_data, axis=1)
test_data = np.expand_dims(test_data, axis=1)

# 将类型转换成torch的tensor类型
train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_labels))
test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data).double(), torch.from_numpy(test_labels))

# 定义数据加载器，batch_size可以自己，即一次取多少
train_loader = tchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = tchdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

model = Net().to(device).double()

# 优化器选择Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_acc, test_acc = [], []
best_test_acc = 0.0  # 记录最佳测试准确率

for i in range(epoch):
    train_acc_current, train_loss = train(model, optimizer, train_loader)
    test_acc_current, pred, targets, test_loss = validate(model, test_loader)
    train_acc.append(train_acc_current)
    test_acc.append(test_acc_current)

    # 更新最佳测试准确率
    if test_acc_current > best_test_acc:
        best_test_acc = test_acc_current

    print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}\ttrain loss: {:0.3f}\ttest loss: {:0.3f}'.format(datetime.now(), i, train_acc_current, test_acc_current, train_loss, test_loss))

print(f"\n=== 测试最优准确率: {best_test_acc:.3f} ===")

# 输出最终结果
print(f"\n最终结果:")
print(f"训练集最终准确率: {train_acc[-1]:.3f}")
print(f"测试集最终准确率: {test_acc[-1]:.3f}")
print(f"测试集最优准确率: {best_test_acc:.3f}")

# 绘制混淆矩阵
heatmap(label_true=targets,
        label_pred=pred,
        label_name=[0, 1],
        title="Confusion Table of ResNet18",
        pdf_save_path="./fig/h_" + model_name + ".png",
        dpi=300)

# 绘制准确率变化曲线
plots((train_acc, 'train accuracy'), 'accuracy',
        'Accuracy Comparison for ' + model_name, num_iterations=epoch,
        x_label='iterations', second=(test_acc, 'test accuracy'), third=None, show_min=True)