""" ResNet Model
author Stephanie
reference https://github.com/weiaicunzai/pytorch-cifar100
"""
import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsizes, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        # make channels to (g, n)
        x = x.view(batchsizes, self.groups, channels_per_group, height, width)

        # transposing (g, n) to (n, g), then flatten
        x = x.transpose(1, 2).contiguous()
        output = x.view(batchsizes, -1, height, width)

        return output


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, stride=1):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        inner_channels = int(out_channels / 4)
        self.pointwise_group_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_shuffle = ChannelShuffle(groups)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1,
                      stride=stride, groups=inner_channels),
            nn.BatchNorm2d(inner_channels)
        )
        if stride == 2:
            out_channels = out_channels-in_channels
        self.pointwise_group_conv2 = nn.Sequential(
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )
        self.optional_avgpool = nn.AvgPool2d(3, 2, padding=1)

    def forward(self, x):
        origin = x

        x = self.pointwise_group_conv1(x)
        x = self.channel_shuffle(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_group_conv2(x)

        if self.stride == 2:  # stride = 2, need fit shortcut and concat
            origin = self.optional_avgpool(origin)
            output = torch.cat([origin, x], dim=1)
        else:  # stride = 1, just use add as shortcut
            output = torch.add(origin, x)

        output = nn.ReLU(inplace=True)(output)

        return output


def make_stage(stage_no, in_channels, out_channels, block_num, groups=1):
    stage = nn.Sequential()
    if stage_no == 2:
        stage.add_module(f"Stage{stage_no}_ShuffleUnit_layer0",
                         ShuffleUnit(in_channels, out_channels, groups=1, stride=2))
    else:
        stage.add_module(f"Stage{stage_no}_ShuffleUnit_layer0",
                         ShuffleUnit(in_channels, out_channels, groups, stride=2))
    for i in range(1, block_num):
        stage.add_module(f"Stage{stage_no}_ShuffleUnit_layer{i}",
                         ShuffleUnit(in_channels=out_channels, out_channels=out_channels, groups=groups))

    return stage


class ShuffleNet(nn.Module):
    def __init__(self, blocks_num, class_num=100, groups=1, scale=1):
        super(ShuffleNet, self).__init__()
        if groups == 1:
            out_channels = [i*scale for i in [24, 144, 288, 567]]
        elif groups == 2:
            out_channels = [i*scale for i in [24, 200, 400, 800]]
        elif groups == 3:
            out_channels = [i*scale for i in [24, 240, 480, 960]]
        elif groups == 4:
            out_channels = [i*scale for i in [24, 272, 544, 1088]]
        elif groups == 8:
            out_channels = [i*scale for i in [24, 384, 768, 1536]]
        else:
            raise Exception("没有这个group的相关输出通道数据， 仅支持1，2，3，4，8")

        self.conv1_maxpool = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels[0], kernel_size=3,
                      padding=1, stride=2),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.stage2 = make_stage(stage_no=2, in_channels=out_channels[0], out_channels=out_channels[1],
                                 block_num=blocks_num[0], groups=groups)
        self.stage3 = make_stage(stage_no=3, in_channels=out_channels[1], out_channels=out_channels[2],
                                 block_num=blocks_num[1], groups=groups)
        self.stage4 = make_stage(stage_no=4, in_channels=out_channels[2], out_channels=out_channels[3],
                                 block_num=blocks_num[2], groups=groups)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels[3], class_num)

    def forward(self, x):
        x = self.conv1_maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output


# scale=1, group=1
def shufflenet_s1_g1():
    return ShuffleNet([4, 8, 4], groups=1)


# scale=0.5, group=1
def shufflenet_s05_g1():
    return ShuffleNet([4, 8, 4], groups=1, scale=0.5)


# scale=0.25, group=1
def shufflenet_s025_g1():
    return ShuffleNet([4, 8, 4], groups=1, scale=0.25)


# scale=1, group=2
def shufflenet_s1_g2():
    return ShuffleNet([4, 8, 4], groups=2)


# scale=0.5, group=2
def shufflenet_s05_g1():
    return ShuffleNet([4, 8, 4], groups=2, scale=0.5)


# scale=0.25, group=2
def shufflenet_s025_g1():
    return ShuffleNet([4, 8, 4], groups=2, scale=0.25)


# scale=1, group=3
def shufflenet_s1_g3():
    return ShuffleNet([4, 8, 4], groups=3)


# scale=0.5, group=3
def shufflenet_s05_g1():
    return ShuffleNet([4, 8, 4], groups=3, scale=0.5)


# scale=0.25, group=3
def shufflenet_s025_g1():
    return ShuffleNet([4, 8, 4], groups=3, scale=0.25)


# scale=1, group=4
def shufflenet_s1_g4():
    return ShuffleNet([4, 8, 4], groups=4)


# scale=0.5, group=4
def shufflenet_s05_g1():
    return ShuffleNet([4, 8, 4], groups=4, scale=0.5)


# scale=0.25, group=4
def shufflenet_s025_g1():
    return ShuffleNet([4, 8, 4], groups=4, scale=0.25)


# scale=1, group=8
def shufflenet_s1_g8():
    return ShuffleNet([4, 8, 4], groups=8)


# scale=0.5, group=8
def shufflenet_s05_g1():
    return ShuffleNet([4, 8, 4], groups=8, scale=0.5)


# scale=0.25, group=8
def shufflenet_s025_g1():
    return ShuffleNet([4, 8, 4], groups=8, scale=0.25)


