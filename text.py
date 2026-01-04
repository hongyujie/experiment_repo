import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rich.console import Console
from icecream import ic




NetWork = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=64),
            nn.Linear(in_features=64, out_features=10)
)

X = torch.rand(size=(1, 1, 28, 28))
for layer in NetWork:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)