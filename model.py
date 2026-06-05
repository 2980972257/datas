import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride == 1:
            assert out_channels % 2 == 0, "out_channels must be even when stride == 1"
            self.branch1 = nn.Sequential(
                DepthwiseSeparableConv(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
                DepthwiseSeparableConv(in_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0)
            )
            self.branch2 = nn.Sequential(
                DepthwiseSeparableConv(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, padding=0),
                DepthwiseSeparableConv(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1),
                DepthwiseSeparableConv(in_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.branch1 = nn.Sequential(
                DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                DepthwiseSeparableConv(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
            )
            self.branch2 = nn.Sequential(
                DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                DepthwiseSeparableConv(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
            )

        self.se = SEModule(out_channels)

    def channel_shuffle(self, x, groups):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out1 = x1 + self.branch1(x2)
            out2 = x2 + self.branch2(x1)
            out = torch.cat((out1, out2), dim=1)
            out = self.channel_shuffle(out, groups=2)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out = torch.cat((out1, out2), dim=1)
        out = self.se(out)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=6):
        super(ShuffleNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = nn.Sequential(
            ShuffleNetV2Block(32, 128, stride=2),
            ShuffleNetV2Block(128, 128, stride=1),
            ShuffleNetV2Block(128, 128, stride=1),
            ShuffleNetV2Block(128, 128, stride=1)
        )
        self.stage3 = nn.Sequential(
            ShuffleNetV2Block(128, 256, stride=2),
            ShuffleNetV2Block(256, 256, stride=1),
            ShuffleNetV2Block(256, 256, stride=1),
            ShuffleNetV2Block(256, 256, stride=1),
            ShuffleNetV2Block(256, 256, stride=1),
            ShuffleNetV2Block(256, 256, stride=1)
        )
        self.stage4 = nn.Sequential(
            ShuffleNetV2Block(256, 512, stride=2),
            ShuffleNetV2Block(512, 512, stride=1),
            ShuffleNetV2Block(512, 512, stride=1),
            ShuffleNetV2Block(512, 512, stride=1),
            ShuffleNetV2Block(512, 512, stride=1),
            ShuffleNetV2Block(512, 512, stride=1)
        )
        self.conv5 = nn.Sequential(
            DepthwiseSeparableConv(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

