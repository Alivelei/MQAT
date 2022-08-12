# _*_ coding: utf-8 _*_

"""
    @Time : 2021/12/13 9:02 
    @Author : smile 笑
    @File : basicblock.py
    @desc :
"""


import torch
from torch import nn


# 封装标准卷积
def conv3x3(in_planes, out_planes, stride=1,padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
      # 当卷积层后面跟着bn层时，卷积层不需要bias
      # 因为在bn层重新学习均值和方差
      return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # 通道放大倍数
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        # 调用父类的初始化函数
        super(BasicBlock, self).__init__()

        # 如果没有指定Batchnormalization
        if norm_layer is None:
          # 使用标准的BatchNorm
          norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # forward用来调用，x是输入
    def forward(self, x):
        indentity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 下采样
        if self.downsample is not None:
          identity = self.downsample(x)

        # f(x)+x
        out += identity
        # 在加和之后再调用激活函数
        out = self.relu(out)

        return out

