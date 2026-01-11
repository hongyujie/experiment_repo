# ResNet-18模型 - 基于cnn_improved.py进行修改
# 修改内容：
# 1. 模型替换：
#    - 将自定义CNN模型替换为ResNet-18
#    - 修改ResNet-18的输入通道数，从3改为1以适应灰度图像
# 2. 保持原有训练和测试流程不变

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import matplotlib.pyplot as plt
import os
# 创建ResNet-18模型并修改输入通道
# 从torchvision.models导入ResNet-18
from torchvision.models import resnet18

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建控制台对象，用于终端显示
console = Console()

# 下载数据集
# train=True表示为训练集，transforms.ToTensor()将图片转换成张量
# 使用绝对路径并设置download=False，避免重复下载已经存在的数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(current_dir, "dataset")
imgs_dir = os.path.join(current_dir, "imgs")
os.makedirs(imgs_dir, exist_ok=True)  # 确保图像目录存在

train_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, transform=transforms.ToTensor(), download=False)  # 已下载，设置为False
test_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=False, transform=transforms.ToTensor(), download=False)  # 已下载，设置为False

ic("数据集为FashionMNIST")
console.print(f"[bold yellow]训练数据集长度为：{len(train_dataset)}[/bold yellow]")
console.print(f"[bold yellow]验证数据集长度为：{len(test_dataset)}[/bold yellow]")

# 查看数据集图片大小
# 获取第一个训练样本
first_img, first_label = train_dataset[0]
console.print(f"[bold red]图片张量形状：{first_img.shape}[/bold red]")

# 加载数据集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 创建ResNet-18模型
my_net = resnet18()

# 修改输入通道数：将第一个卷积层的输入通道从3改为1
my_net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# 修改全连接层：保持输出类别数为10
num_ftrs = my_net.fc.in_features
my_net.fc = nn.Linear(num_ftrs, 10)

# 将模型转移到设备上
my_net = my_net.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-3  # ResNet通常使用较小的学习率
optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate, momentum=0.9)

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

# 训练开始
for i in range(epoch):  # 训练和测试一起进行
    my_net.train()  # 对于某些特定的层有用，如Dropout层、BatchNorm层等。
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
    
    console.print(f"[bold magenta]第{i+1}轮测试数据集loss:{test_loss:.4f}[/bold magenta]")
    console.print(f"[bold green]第{i+1}轮测试数据集准确率:{test_accuracy:.4f}[/bold green]")
    total_test_step += 1

# 绘制测试精确度随训练次数变化的图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch+1), test_accuracy_list, 'b-o', label='Test Accuracy')
plt.title('Test Accuracy', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(imgs_dir, 'test_accuracy_resnet.png'))
plt.show()

# 绘制测试损失随训练次数变化的图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch+1), test_loss_list, 'r-o', label='Test Loss')
plt.title('Test Loss', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(imgs_dir, 'test_loss_resnet.png'))
plt.show()

console.print(f"[bold blue]ResNet模型的训练图像已保存到：{imgs_dir} 文件夹中[/bold blue]")