# DenseNet模型 - 基于cnn_im_densenet.py进行修改
# 修改内容：
# 1. 保留DenseNet121模型结构，适配FashionMNIST灰度图像
# 2. 增加了参数量统计（以M为单位）
# 3. 增加了模型大小统计（以MB为单位）
# 4. 增加了训练时长统计（以min为单位）
# 5. 增加了单个样本平均测试时长统计（以ms为单位）

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import os
import time
# 从torchvision.models导入DenseNet121
from torchvision.models import densenet121

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

# 加载数据集 - 扩大批次大小
train_data_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# 创建DenseNet121模型并进行FashionMNIST适配
my_net = densenet121(weights=None)  # 使用weights参数代替pretrained，设置为None避免加载预训练权重

# 优化DenseNet结构以更好地适配FashionMNIST小尺寸图像
# 1. 修改输入通道数：将第一个卷积层的输入通道从3改为1以适应灰度图像
# 2. 使用更小的卷积核和步长，避免过早减小特征图尺寸
my_net.features.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

# 调整第一个池化层：移除或修改以保持特征图尺寸
# 对于28x28的小图像，我们可以完全移除第一个池化层或使用更小的步长
my_net.features.pool0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

# 修改分类器：保持输出类别数为10（FashionMNIST的类别数）
num_ftrs = my_net.classifier.in_features
my_net.classifier = nn.Linear(num_ftrs, 10)

# 将模型转移到设备上
my_net = my_net.to(device)

# 统计参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(my_net)
total_params_m = total_params / 1e6  # 转换为M单位
trainable_params_m = trainable_params / 1e6  # 转换为M单位
console.print(f"[bold blue]总参数量: {total_params_m:.2f} M[/bold blue]")
console.print(f"[bold blue]可训练参数量: {trainable_params_m:.2f} M[/bold blue]")

# 保存模型并统计大小
torch.save(my_net.state_dict(), 'temp_model.pth')
model_size = os.path.getsize('temp_model.pth')
os.remove('temp_model.pth')  # 删除临时文件

# 转换模型大小单位（以MB为单位）
model_size_mb = model_size / (1024 * 1024)  # 转换为MB单位
console.print(f"[bold blue]模型大小: {model_size_mb:.2f} MB[/bold blue]")

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器 - 使用更大批次时，学习率可以适当调整
learning_rate = 1e-3  # DenseNet通常使用较小的学习率
optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# 设置训练网络的一些参数
# 训练次数
epoch = 25  # 适当增加训练轮次以充分利用更大批次带来的稳定性
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录每次测试的精确度
test_accuracy_list = []

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
    test_accuracy_list.append(test_accuracy)
    
    console.print(f"[bold magenta]第{i+1}轮测试数据集loss:{test_loss}[/bold magenta]")
    console.print(f"[bold green]第{i+1}轮测试数据集准确率:{test_accuracy}[/bold green]")
    total_test_step += 1

# 计算训练总时长
train_end_time = time.time()
train_duration = train_end_time - train_start_time
train_duration_min = train_duration / 60  # 转换为分钟单位

# 计算并输出测试精度的均值
avg_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)
# 计算并输出最高测试准确率及其对应的轮次
highest_test_accuracy = max(test_accuracy_list)
highest_accuracy_epoch = test_accuracy_list.index(highest_test_accuracy) + 1

# 计算单个样本平均测试时长
if total_inference_samples > 0:
    avg_inference_time_per_sample = (total_inference_time / total_inference_samples) * 1000  # 转换为毫秒
else:
    avg_inference_time_per_sample = 0

console.print(f"\n[bold cyan]测试精度的平均值: {avg_test_accuracy:.4f}[/bold cyan]")
console.print(f"[bold cyan]最高测试准确率: {highest_test_accuracy:.4f} (第{highest_accuracy_epoch}轮)[/bold cyan]")
console.print(f"[bold blue]总训练时长: {train_duration_min:.2f} min[/bold blue]")
console.print(f"[bold blue]单个样本平均测试时长: {avg_inference_time_per_sample:.2f} ms[/bold blue]")