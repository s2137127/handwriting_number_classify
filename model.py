import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=16 * 20 * 20, out_features=120, bias=True)
        self.fc2 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc3 = nn.Linear(in_features=84, out_features=10, bias=True)

    def forward(self, x):
        # TODO
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output

    def name(self):
        return "ConvNet"


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(5, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(5 ,stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256, bias=True)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.fc5 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.drop3 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.fc7 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.drop4 = nn.Dropout(0.2)
        self.fc8 = nn.Linear(in_features=32, out_features=10, bias=True)

    def forward(self, x):
        # TODO
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        x = self.fc3(x)
        # x = self.drop3(x)
        x = self.fc4(x)
        # x = self.drop4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        output = self.fc8(x)
        return output

    def name(self):
        return "MyNet"
