# ResNet-18模型 - 基于resnet.py进行改进
# 修改内容：
# 1. 学习率调度策略优化：
#    - 使用CosineAnnealingLR实现余弦退火学习率调度
#    - 设置T_max=epoch，实现一个完整的余弦周期
#    - 设置eta_min=1e-6，确保学习率不会降得过低
# 2. 数据增强：
#    - 为训练集添加随机水平翻转、随机裁剪、随机旋转等增强操作
#    - 对训练集和测试集都进行归一化处理
# 3. 优化器升级：
#    - 将SGD+momentum替换为Adam优化器
#    - 设置合适的学习率和权重衰减

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import matplotlib.pyplot as plt
import os

# 从torchvision.models导入ResNet-18
from torchvision.models import resnet18

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建控制台对象，用于终端显示
console = Console()

# 使用绝对路径定位数据集和图像保存目录
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(current_dir, "dataset")
imgs_dir = os.path.join(current_dir, "imgs")
os.makedirs(imgs_dir, exist_ok=True)  # 确保图像目录存在

# 数据增强和预处理
# 训练集转换：包含数据增强操作
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomCrop(28, padding=2),     # 随机裁剪
    transforms.RandomRotation(5),             # 随机旋转±5度
    transforms.ToTensor(),                    # 转换为张量
    transforms.Normalize((0.2860,), (0.3530,))  # 归一化（FashionMNIST的均值和标准差）
])

# 测试集转换：仅包含必要的预处理，不进行数据增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # 与训练集相同的归一化参数
])

# 下载和加载数据集
train_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, transform=train_transform, download=False)  # 已下载，设置为False
test_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=False, transform=test_transform, download=False)  # 已下载，设置为False

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

# 优化器升级：使用Adam代替SGD
learning_rate = 1e-3
optimizer = torch.optim.Adam(
    my_net.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4  # 添加L2正则化
)

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

# 学习率调度策略优化：使用CosineAnnealingLR
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epoch,  # 周期长度等于训练轮数
    eta_min=1e-6  # 最小学习率
)

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

    # 更新学习率
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    console.print(f"[bold cyan]第{i+1}轮学习率: {current_lr:.6f}[/bold cyan]")

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
plt.savefig(os.path.join(imgs_dir, 'test_accuracy_resnet_2.png'))
plt.show()

# 绘制测试损失随训练次数变化的图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch+1), test_loss_list, 'r-o', label='Test Loss')
plt.title('Test Loss', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(imgs_dir, 'test_loss_resnet_2.png'))
plt.show()

console.print(f"[bold blue]ResNet-2模型的训练图像已保存到：{imgs_dir} 文件夹中[/bold blue]")