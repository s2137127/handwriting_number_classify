import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # ->(16,28,28)
            nn.ReLU(),  # ->(16,28,28)
            nn.MaxPool2d(kernel_size=2),  # ->(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # ->(32,14,14)
            nn.ReLU(),  # ->(32,14,14)
            nn.MaxPool2d(kernel_size=2),  # ->(32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # TODO
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def name(self):
        return "ConvNet"

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # TODO

    def forward(self, x):
        # TODO
        return out

    def name(self):
        return "MyNet"

