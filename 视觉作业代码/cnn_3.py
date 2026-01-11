# 进一步改进的CNN模型 - 基于cnn_2.py进行修改
# 修改内容：
# 1. 学习率调度器优化：
#    - 使用CosineAnnealingLR代替ReduceLROnPlateau
#    - 实现更平滑的学习率余弦衰减曲线
# 2. 增强数据增强：
#    - 添加随机亮度/对比度调整
#    - 添加随机擦除(RandomErasing)增强
#    - 添加高斯模糊
#    - 对数据进行归一化处理
# 3. 优化器参数调整：
#    - 为Adam优化器添加weight_decay参数实现L2正则化
#    - 调整betas参数为(0.9, 0.999)
# 4. 标签平滑(Label Smoothing)：
#    - 在CrossEntropyLoss中添加label_smoothing参数
#    - 减轻模型过度自信，提高泛化能力
# 速度太慢了，效果提升不大，没有等它运行完
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

# 使用绝对路径定位数据集和图像保存目录
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(current_dir, "dataset")
imgs_dir = os.path.join(current_dir, "imgs")
os.makedirs(imgs_dir, exist_ok=True)  # 确保图像目录存在

# 增强数据增强操作
# 训练集转换：包含多种数据增强操作
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomCrop(28, padding=2),     # 随机裁剪
    transforms.RandomRotation(5),             # 随机旋转±5度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机亮度/对比度调整
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),  # 高斯模糊
    transforms.ToTensor(),                    # 转换为张量
    transforms.Normalize(mean=[0.2860], std=[0.3530]),  # 归一化（FashionMNIST的均值和标准差）
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 随机擦除
])

# 测试集转换：仅包含必要的预处理，不进行数据增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530])  # 与训练集相同的归一化参数
])

# 下载和加载数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root=dataset_root, 
    train=True, 
    transform=train_transform, 
    download=False  # 已下载，设置为False
)

test_dataset = torchvision.datasets.FashionMNIST(
    root=dataset_root, 
    train=False, 
    transform=test_transform, 
    download=False  # 已下载，设置为False
)

ic("数据集为FashionMNIST")
console.print(f"[bold yellow]训练数据集长度为：{len(train_dataset)}[/bold yellow]")
console.print(f"[bold yellow]测试数据集长度为：{len(test_dataset)}[/bold yellow]")

# 查看数据集图片大小
first_img, first_label = train_dataset[0]
console.print(f"[bold red]图片张量形状：{first_img.shape}[/bold red]")

# 加载数据集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 神经网络模型结构（保持与cnn_2.py相同的结构）
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
            # 全连接层
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
my_net = my_net.to(device)  # 转移到设置的设备上训练

# 损失函数：添加标签平滑
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑，平滑因子为0.1
loss_fn = loss_fn.to(device)

# 优化器参数调整：为Adam添加weight_decay和调整betas
learning_rate = 1e-3
optimizer = torch.optim.Adam(
    my_net.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,  # 添加L2正则化
    betas=(0.9, 0.999)  # 调整betas参数
)

# 设置训练参数
epoch = 20
total_train_step = 0
total_test_step = 0
test_accuracy_list = []
test_loss_list = []

# 学习率调度器优化：使用CosineAnnealingLR代替ReduceLROnPlateau
# 实现余弦退火学习率调度
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epoch,  # 周期长度等于训练轮数
    eta_min=1e-6  # 最小学习率
)

# 训练开始
for i in range(epoch):
    my_net.train()
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
    previous_lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # 打印学习率变化信息
    console.print(f"[bold cyan]第{i+1}轮学习率: {current_lr:.6f}[/bold cyan]")

    # 测试步骤开始
    my_net.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
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

# 使用绝对路径保存图片
accuracy_img_path = os.path.join(imgs_dir, "test_accuracy_cnn_3.png")
plt.savefig(accuracy_img_path)
plt.show()

# 绘制测试损失随训练次数变化的图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch+1), test_loss_list, 'r-o', label='Test Loss')
plt.title('Test Loss', fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 使用绝对路径保存图片
loss_img_path = os.path.join(imgs_dir, "test_loss_cnn_3.png")
plt.savefig(loss_img_path)
plt.show()

console.print(f"[bold blue]改进后的图像已保存到：{imgs_dir} 文件夹中[/bold blue]")