# 改进的CNN模型 - 基于cnn.py进行修改
# 修改内容：
# 1. 模型结构优化：
#    - 在每个卷积层后增加了BatchNorm层
#    - 调整网络宽度：将32-32-64结构改为32-64-128
#    - 增加了一层全连接层（256个神经元）
#    - 调整了全连接层的输入特征数量
# 2. 正则化：
#    - 在全连接层之间增加了Dropout层（p=0.5）防止过拟合
# 3. 统计功能：
#    - 增加了参数量统计
#    - 增加了模型大小统计
#    - 增加了训练时长统计

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import os
import time

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

# 统计参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(my_net)
console.print(f"[bold blue]总参数量: {total_params:,}[/bold blue]")
console.print(f"[bold blue]可训练参数量: {trainable_params:,}[/bold blue]")

# 保存模型并统计大小
torch.save(my_net.state_dict(), 'temp_model.pth')
model_size = os.path.getsize('temp_model.pth')
os.remove('temp_model.pth')  # 删除临时文件

# 转换模型大小单位
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0 Bytes"
    size_name = ("Bytes", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

import math
console.print(f"[bold blue]模型大小: {convert_size(model_size)}[/bold blue]")

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练次数
epoch = 20
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 记录训练时长
train_start_time = time.time()

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

    # 测试步骤开始
    my_net.eval()  # 对于某些特定的层有用，如Dropout层、BatchNorm层等。
    total_test_loss = 0
    total_accuracy = 0
    total_inference_time = 0  # 总推理时间（秒）
    total_inference_samples = 0  # 总推理样本数

    with torch.no_grad():  # 去除梯度即不进行权重更新
        for data in test_data_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            
            # 记录单个batch的推理时间
            start_time = time.time()
            outputs = my_net(imgs)
            end_time = time.time()
            
            # 累计推理时间和样本数
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            total_inference_samples += imgs.size(0)
            
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    test_accuracy = total_accuracy / len(test_dataset)
    test_loss = total_test_loss / len(test_data_loader)
    
    console.print(f"[bold magenta]第{i+1}轮测试数据集loss:{test_loss}，准确率:{test_accuracy*100:.2f}%[/bold magenta]")
    total_test_step += 1

# 计算训练总时长
train_end_time = time.time()
train_duration = train_end_time - train_start_time

# 转换训练时长单位
def convert_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}分{seconds:.2f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours}小时{minutes}分{seconds:.2f}秒"

console.print(f"[bold green]训练总时长: {convert_time(train_duration)}[/bold green]")
# 计算单个样本平均测试时长
if total_inference_samples > 0:
    avg_inference_time_per_sample = (total_inference_time / total_inference_samples) * 1000  # 转换为毫秒
else:
    avg_inference_time_per_sample = 0

console.print(f"[bold cyan]模型统计信息：[/bold cyan]")
console.print(f"[bold cyan]- 总参数量: {total_params:,}[/bold cyan]")
console.print(f"[bold cyan]- 可训练参数量: {trainable_params:,}[/bold cyan]")
console.print(f"[bold cyan]- 模型大小: {convert_size(model_size)}[/bold cyan]")
console.print(f"[bold cyan]- 训练总时长: {convert_time(train_duration)}[/bold cyan]")
console.print(f"[bold cyan]- 单个样本平均测试时长: {avg_inference_time_per_sample:.2f} ms[/bold cyan]")