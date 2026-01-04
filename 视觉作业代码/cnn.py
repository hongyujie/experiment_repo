import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
import matplotlib.pyplot as plt
# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建控制台对象，用于终端显示
console = Console()

# 下载数据集
# train=False表示为训练集，transforms.ToTensor()将图片转换成张量
train_dataset = torchvision.datasets.FashionMNIST(root="./视觉作业代码/dataset", train=True, transform=transforms.ToTensor(),
                                                download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./视觉作业代码/dataset", train=False, transform=transforms.ToTensor(),
                                                download=True)
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

# 搭建神经网络
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.model = nn.Sequential(# 用该方法更方便
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(in_features=576, out_features=64),
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
epoch = 10
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录每次测试的精确度
test_accuracy_list = []
# 记录每次测试的损失
test_loss_list = []

# 训练开始
for i in range(epoch):# 训练和测试一起进行
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
        # console.print(f"[bold magenta]第{total_train_step}次训练，loss为：{loss.item()}[/bold magenta]")

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
    total_test_step += 1

# 绘制测试精确度随训练次数变化的图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch+1), test_accuracy_list, 'b-o', label='Test Accuracy')
plt.title('Test Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig('./视觉作业代码/imgs/test_accuracy.png')
plt.show()

# 绘制测试损失随训练次数变化的图像
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch+1), test_loss_list, 'r-o', label='Test Loss')
plt.title('Test Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('./视觉作业代码/imgs/test_loss.png')
plt.show()

console.print("[bold blue]图像已保存到视觉作业代码/imgs文件夹中[/bold blue]")