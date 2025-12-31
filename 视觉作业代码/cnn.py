import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic
# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建控制台对象，用于终端显示
console = Console()

# 下载数据集
# train=False表示为训练集，transforms.ToTensor()将图片转换成张量
train_dataset = torchvision.datasets.FashionMNIST(root="./dataset", train=True, transform=transforms.ToTensor(),
                                                download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./dataset", train=False, transform=transforms.ToTensor(),
                                                download=True)
ic("数据集为FashionMNIST")
console.print(f"[bold yellow]训练数据集长度为：{len(train_dataset)}[/bold yellow]")
console.print(f"[bold yellow]验证数据集长度为：{len(test_dataset)}[/bold yellow]")
# # 加载数据集
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=64)
# test_data_loader = DataLoader(dataset=test_dataset, batch_size=64)

# # 搭建神经网络
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.model = nn.Sequential(# 用该方法更方便
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Flatten(),
#             nn.Linear(in_features=1024, out_features=64),
#             nn.Linear(in_features=64, out_features=10)
#         )

#     def forward(self, x):
#         return self.model(x)

# # 创建网络模型
# my_net = Net()
# my_net = my_net.to(device)# 转移到我们设置的网络上去训练

# # 损失函数
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = loss_fn.to(device)

# # 优化器
# learning_rate = 1e-2  # 0.01
# optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate)

# # 设置训练网络的一些参数
# # 训练次数
# epoch = 10
# # 记录训练的次数
# total_train_step = 0
# # 记录测试的次数
# total_test_step = 0

# # 添加tensorboard
# writer = SummaryWriter("logs")
# for i in range(epoch):
#     print(f"-----第{i + 1}轮训练开始-----")

#     # 训练开始
#     my_net.train()  # 对于某些特定的层有用，如Dropout层、BatchNorm层等。
#     for data in train_data_loader:
#         imgs, targets = data
#         imgs, targets = imgs.to(device), targets.to(device)
#         outputs = my_net(imgs)
#         loss = loss_fn(outputs, targets)

#         # 优化器优化模型
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_train_step += 1
#         if total_train_step % 100 == 0:
#             print(f"训练次数：{total_train_step}， loss:{loss.item()}")
#             writer.add_scalar("train_loss", loss.item(), total_train_step)

#     # 测试步骤开始
#     my_net.eval()  # 对于某些特定的层有用，如Dropout层、BatchNorm层等。
#     total_test_loss = 0
#     total_accuracy = 0
#     with torch.no_grad():  # 去除梯度即不进行权重更新
#         for data in test_data_loader:
#             imgs, targets = data
#             imgs, targets = imgs.to(device), targets.to(device)
#             outputs = my_net(imgs)
#             loss = loss_fn(outputs, targets)
#             total_test_loss += loss.item()
#             accuracy = (outputs.argmax(1) == targets).sum()
#             total_accuracy += accuracy.item()

#     print(f"整体测试数据集loss:{total_test_loss}")
#     print(f"整体测试数据集accuracy:{total_accuracy / len(test_dataset)}")
#     total_test_step += 1
#     writer.add_scalar("test_loss", total_test_loss, total_test_step)
#     writer.add_scalar("test_accuracy", total_accuracy / len(test_dataset), total_test_step)

# # 保存模型
# torch.save(my_net, "my_net_model.pth")

# writer.close()

