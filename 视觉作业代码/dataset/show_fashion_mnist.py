import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置数据转换，将图像转换为张量并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载FashionMNIST数据集
current_dir = 'd:\\Documents\\Desktop\\demo\\视觉作业代码'
dataset = datasets.FashionMNIST(
    root=current_dir + '\\dataset',
    train=True,
    transform=transform,
    download=False  # 数据集已存在，设置为False
)

# 获取数据集的类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 创建一个4行10列的图像网格
fig, axes = plt.subplots(4, 10, figsize=(15, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 显示40张图片，每张图片下方显示标签
for i, ax in enumerate(axes.flat):
    # 获取第i个图像和标签
    image, label = dataset[i]
    
    # 反归一化图像
    image = image / 2 + 0.5  # 将图像从[-1, 1]转换回[0, 1]
    np_image = image.numpy()
    
    # 显示图像
    ax.imshow(np_image.squeeze(), cmap='gray')
    
    # 设置图片标题
    ax.set_title(class_names[label], fontsize=10)
    
    # 隐藏坐标轴
    ax.axis('off')

# 保存图像
plt.savefig(current_dir + '\\dataset\\fashion_mnist_samples.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
print("已生成并保存40张FashionMNIST样本图片：fashion_mnist_samples.png")