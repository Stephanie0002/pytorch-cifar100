""" DensNet Model
author Stephanie
reference https://github.com/weiaicunzai/pytorch-cifar100
"""
import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channels = 4 * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inner_channels, out_channels=growth_rate, kernel_size=3,
                      padding=1, bias=False)
        )

    def forward(self, x):
        output = self.block(x)
        output = torch.cat([x, output], dim=1)
        return output


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


def make_dense_block(block, in_channels, layer_num, growth_rate):
    dense_block = nn.Sequential()
    for i in range(layer_num):
        dense_block.add_module('bottle_neck_layer_{}'.format(i), block(in_channels, growth_rate))
        in_channels += growth_rate
    return dense_block


class DenseNet(nn.Module):
    def __init__(self, block, block_num, growth_rate=32, reduction=0.5, class_num=100):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        inner_channels = growth_rate << 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=inner_channels, kernel_size=3,
                               padding=1, bias=False)
        self.features = nn.Sequential()
        for i in range(len(block_num) - 1):
            self.features.add_module(
                "dense_block_layer_{}".format(i),
                make_dense_block(block, inner_channels, block_num[i], growth_rate)
            )
            inner_channels += growth_rate * block_num[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module(
                "transition_layer_{}".format(i),
                Transition(inner_channels, out_channels)
            )
            inner_channels = out_channels
        self.features.add_module(
            "dense_block_layer_{}".format(len(block_num) - 1),
            make_dense_block(block, inner_channels, block_num[-1], growth_rate)
        )
        inner_channels += growth_rate * block_num[-1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inner_channels, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        output = self.fc(x)
        return output


def densenet121(growth_rate=32):
    return DenseNet(BottleNeck, [6, 12, 24, 16], growth_rate)


def densenet169(growth_rate=32):
    return DenseNet(BottleNeck, [6, 12, 32, 32], growth_rate)


def densenet201(growth_rate=32):
    return DenseNet(BottleNeck, [6, 12, 48, 32], growth_rate)


def densenet161(growth_rate=48):
    return DenseNet(BottleNeck, [6, 12, 36, 24], growth_rate)
