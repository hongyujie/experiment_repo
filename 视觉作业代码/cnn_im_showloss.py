# 利用3层卷积神经网络来实现对FashionMNIST 图像分类任务的代码

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import matplotlib.pyplot as plt
import os

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建控制台对象，用于终端显示
console = Console()

# 下载数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(current_dir, "dataset")

# 加载完整训练集和测试集
train_full_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, transform=transforms.ToTensor(),download=False)  
test_dataset = torchvision.datasets.FashionMNIST(root=dataset_root, train=False, transform=transforms.ToTensor(),download=False)  

# 划分训练集和验证集：54,000用于训练，6,000用于验证
train_size = 54000
val_size = 6000
train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

ic("数据集为FashionMNIST")
console.print(f"[bold yellow]训练数据集长度为：{len(train_dataset)}[/bold yellow]")
console.print(f"[bold yellow]验证数据集长度为：{len(val_dataset)}[/bold yellow]")
console.print(f"[bold yellow]测试数据集长度为：{len(test_dataset)}[/bold yellow]")

# 查看数据集图片大小
# 获取第一个训练样本
first_img, first_label = train_dataset[0]
console.print(f"[bold red]图片张量形状：{first_img.shape}[/bold red]")

# 加载数据集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64)
val_data_loader = DataLoader(dataset=val_dataset, batch_size=64)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64)

# 搭建神经网络 
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.model = nn.Sequential(
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
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练次数
epoch = 30
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练loss
train_loss_list = []
# 记录验证loss
val_loss_list = []
# 记录测试loss
test_loss_list = []
# 记录测试准确率
test_accuracy_list = []

# 训练开始
for i in range(epoch):
    my_net.train()  
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
    
    # 计算每轮训练的平均loss和acc
    train_loss = total_train_loss / len(train_data_loader)
    train_accuracy = total_train_accuracy / len(train_dataset)
    # 记录训练loss
    train_loss_list.append(train_loss)

    # 验证步骤开始 
    my_net.eval()  
    total_val_loss = 0
    total_val_accuracy = 0

    with torch.no_grad():  # 去除梯度即不进行权重更新
        for data in val_data_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = my_net(imgs)
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_val_accuracy += accuracy.item()

    val_accuracy = total_val_accuracy / len(val_dataset)
    val_loss = total_val_loss / len(val_data_loader)
    # 记录验证loss
    val_loss_list.append(val_loss)
    
    console.print(f"[bold magenta]第{i+1}轮验证数据集loss:{val_loss}，准确率:{val_accuracy*100:.2f}%[/bold magenta]")
    total_test_step += 1

    # 测试步骤开始 
    my_net.eval()  # 确保模型处于评估模式
    total_test_loss = 0
    total_test_accuracy = 0

    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = my_net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy += accuracy.item()

    test_accuracy = total_test_accuracy / len(test_dataset)
    test_loss = total_test_loss / len(test_data_loader)
    # 记录每轮测试loss和准确率
    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy)
    
    console.print(f"[bold green]第{i+1}轮测试数据集loss:{test_loss}，准确率:{test_accuracy*100:.2f}%[/bold green]")

# 所有训练轮次结束后，输出最终结果
console.print("\n[bold blue]所有训练轮次完成！[/bold blue]")
if len(test_loss_list) > 0:
    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    avg_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list) * 100
    console.print(f"[bold green]测试集平均Loss:{avg_test_loss:.4f}，平均准确率:{avg_test_accuracy:.2f}%[/bold green]")
else:
    console.print("[bold green]训练完成！[/bold green]")

# 绘制训练、验证、测试的loss曲线
plt.figure(figsize=(10, 6))
plt.title('Loss Comparison for CNN', fontweight='bold', fontsize=14)

# 设置坐标轴范围
plt.ylim(bottom=0.0, top=max(max(train_loss_list), max(val_loss_list), max(test_loss_list))*1.1)
plt.xlim(left=0.5, right=epoch+0.5)

# 设置坐标轴标签
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 设置x轴刻度为整数并从1开始
plt.xticks(range(1, epoch+1))

# 设置y轴刻度格式为最多两位小数
plt.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 绘制三条曲线
plt.plot(range(1, epoch+1), train_loss_list, '-', color='blue', label='Train Loss', linewidth=2)
plt.plot(range(1, epoch+1), val_loss_list, '-', color='red', label='Validation Loss', linewidth=2)
plt.plot(range(1, epoch+1), test_loss_list, '-', color='green', label='Test Loss', linewidth=2)

# 添加图例和调整布局
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()

# 添加图表边框
ax = plt.gca()
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# 保存图像到当前目录下的imgs文件夹
imgs_dir = './imgs'
if not os.path.exists(imgs_dir):
    os.makedirs(imgs_dir)

# 使用相对路径保存图像
plt.savefig(os.path.join(imgs_dir, 'all_loss_curves.png'), dpi=300, bbox_inches='tight')
console.print(f"[bold blue]图像已保存到: {os.path.abspath(os.path.join(imgs_dir, 'all_loss_curves.png'))}[/bold blue]")

# 显示图像
plt.show()
plt.close()