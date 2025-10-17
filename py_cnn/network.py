import torch
from torch import nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self):
        r"""
        CNN init.
        """
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=1)
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1)
        self.pool2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense: nn.Linear = nn.Linear(in_features=4 * 5 * 5, out_features=10, bias=True)

    def forward(self, x) -> torch.Tensor:
        r"""
        CNN forward pass.

        :param x: CNN input.

        :return: CNN output.
        """
        x: torch.Tensor = f.relu(self.conv1(x))
        x = self.pool1(x)
        x = f.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x
