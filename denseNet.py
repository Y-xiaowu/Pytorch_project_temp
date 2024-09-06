import math
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):#in_planes, growth_rate
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        #out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, reduction, num_classes):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_channels = 2 * growth_rate
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dense1 = self._make_dense_layers(num_channels, num_blocks[0])
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = self._make_dense_layers(num_channels, num_blocks[1])
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense3 = self._make_dense_layers(num_channels, num_blocks[2])
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(math.floor(num_channels * reduction))
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense4 = self._make_dense_layers(num_channels, num_blocks[3])
        num_channels += num_blocks[3] * growth_rate

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_dense_layers(self, in_channels, num_block):
        layers = []
        for i in range(num_block):
            layers.append(Bottleneck(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.basic_conv(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet([6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=1000)


def DenseNet169():
    return DenseNet([6, 12, 32, 32], growth_rate=32, reduction=0.5, num_classes=1000)


def DenseNet201():
    return DenseNet([6, 12, 48, 32], growth_rate=32, reduction=0.5, num_classes=1000)


def DenseNet265():
    return DenseNet([6, 12, 64, 48], growth_rate=32, reduction=0.5, num_classes=1000)


# net = DenseNet121()
# print(net)
# x = torch.randn(1, 3, 224, 224)
# y = net(x)
# print(y.size())

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model=DenseNet169().to(device)
    summary(model,(3,224,224))
    print(model)
