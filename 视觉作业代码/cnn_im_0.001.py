# 改进的CNN模型 - 基于cnn.py进行修改
# 修改内容：
# 1. 模型结构优化：
#    - 在每个卷积层后增加了BatchNorm层
#    - 调整网络宽度：将32-32-64结构改为32-64-128
#    - 增加了一层全连接层（256个神经元）
#    - 调整了全连接层的输入特征数量
# 2. 正则化：
#    - 在全连接层之间增加了Dropout层（p=0.5）防止过拟合
# 3. 学习率：
#    - 将学习率从0.01改为0.1
# 4. 输出：
#    - 去掉了acc和loss图像生成
#    - 添加了测试精度均值的计算和输出

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import os
# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建控制台对象，用于终端显示
console = Console()

# 下载数据集
# train=True表示为训练集，transforms.ToTensor()将图片转换成张量
# 使用绝对路径并设置download=False，避免重复下载已经存在的数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(current_dir, "dataset")

train_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, transform=transforms.ToTensor(),download=False)  # 已下载，设置为False

test_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=False, transform=transforms.ToTensor(),download=False)  # 已下载，设置为False
ic("数据集为FashionMNIST")
console.print(f"[bold yellow]训练数据集长度为：{len(train_dataset)}[/bold yellow]")
console.print(f"[bold yellow]验证数据集长度为：{len(test_dataset)}[/bold yellow]")

# 查看数据集图片大小
# 获取第一个训练样本
first_img, first_label = train_dataset[0]
console.print(f"[bold red]图片张量形状：{first_img.shape}[/bold red]")

# 加载数据集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64)

# 搭建神经网络 - 改进版
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.model = nn.Sequential(# 用该方法更方便
            # 第一层卷积
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第二层卷积
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 第三层卷积
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            # 全连接层：Dropout层防止过拟合
            nn.Linear(in_features=1152, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),  
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),  
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        return self.model(x)

# 创建网络模型
my_net = network()
my_net = my_net.to(device)# 转移到我们设置的网络上去训练

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.001  
optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练次数
epoch = 20
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录每次测试的精确度
test_accuracy_list = []
# 记录每次测试的损失
test_loss_list = []
# 记录每次训练的损失
train_loss_list = []
# 记录每次训练的精确度
train_accuracy_list = []

# 训练开始
for i in range(epoch):# 训练和测试一起进行
    my_net.train()  # 对于某些特定的层有用，如Dropout层、BatchNorm层等。
    total_train_loss = 0
    total_train_accuracy = 0
    
    for data in train_data_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = my_net(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        total_train_loss += loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_train_accuracy += accuracy.item()
        # console.print(f"[bold magenta]第{total_train_step}次训练，loss为：{loss.item()}[/bold magenta]")
    
    # 计算每轮训练的平均loss和acc
    train_loss = total_train_loss / len(train_data_loader)
    train_accuracy = total_train_accuracy / len(train_dataset)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)

    # 测试步骤开始
    my_net.eval()  # 对于某些特定的层有用，如Dropout层、BatchNorm层等。
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():  # 去除梯度即不进行权重更新
        for data in test_data_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = my_net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    test_accuracy = total_accuracy / len(test_dataset)
    test_loss = total_test_loss / len(test_data_loader)
    test_accuracy_list.append(test_accuracy)
    test_loss_list.append(test_loss)
    
    console.print(f"[bold magenta]第{i+1}轮测试数据集loss:{test_loss}[/bold magenta]")
    console.print(f"[bold green]第{i+1}轮测试数据集准确率:{test_accuracy}[/bold green]")
    total_test_step += 1

# 计算并输出测试精度的均值
avg_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)
console.print(f"[bold cyan]测试精度的均值为:{avg_test_accuracy:.4f}[/bold cyan]")

# 计算并输出最高测试准确率及其对应的轮次
max_test_accuracy = max(test_accuracy_list)
max_epoch = test_accuracy_list.index(max_test_accuracy) + 1
console.print(f"[bold yellow]最高测试准确率为:{max_test_accuracy:.4f}，出现在第{max_epoch}轮[/bold yellow]")