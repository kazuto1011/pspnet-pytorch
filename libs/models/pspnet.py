#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBNReLU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels,
                                             eps=1e-5, momentum=0.95, affine=True))
        if relu:
            self.add_module('relu', nn.ReLU())

    def forward(self, x):
        return super(_ConvBNReLU, self).forward(x)


class _BottleConv(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(_BottleConv, self).__init__()
        self.reduce = _ConvBNReLU(in_channels, mid_channels, 1, stride, 0, 1)  # NOQA
        self.conv3x3 = _ConvBNReLU(mid_channels, mid_channels, 3, 1, dilation, dilation)  # NOQA
        self.increase = _ConvBNReLU(mid_channels, out_channels, 1, 1, 0, 1, relu=False)  # NOQA
        self.proj = _ConvBNReLU(in_channels, out_channels, 1, stride, 0, 1, relu=False)  # NOQA

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        return F.relu(h + self.proj(x))


class _BottleIdentity(nn.Sequential):

    def __init__(self, in_channels, mid_channels, dilation):
        super(_BottleIdentity, self).__init__()
        self.reduce = _ConvBNReLU(in_channels, mid_channels, 1, 1, 0, 1)
        self.conv3x3 = _ConvBNReLU(mid_channels, mid_channels, 3, 1, dilation, dilation)  # NOQA
        self.increase = _ConvBNReLU(mid_channels, in_channels, 1, 1, 0, 1, relu=False)  # NOQA

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        return F.relu(h + x)


class _ResBlock(nn.Sequential):

    def __init__(self, n_layer, in_channels, mid_channels, out_channels, stride, dilate):
        super(_ResBlock, self).__init__()
        self.add_module('block1', _BottleConv(in_channels, mid_channels, out_channels, stride, dilate))  # NOQA
        for i in range(2, n_layer + 1):
            self.add_module('block' + str(i), _BottleIdentity(out_channels, mid_channels, dilate))  # NOQA

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class _DilatedFCN(nn.Module):
    def __init__(self, n_blocks):
        super(_DilatedFCN, self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', _ConvBNReLU(3, 64, 3, 2, 1, 1)),
            ('conv2', _ConvBNReLU(64, 64, 3, 1, 1, 1)),
            ('conv3', _ConvBNReLU(64, 128, 3, 1, 1, 1)),
            ('pool', nn.MaxPool2d(3, 2, 1))
        ]))
        self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1)
        self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h1 = self.layer4(h)
        h2 = self.layer5(h1)
        if self.training:
            return h1, h2
        else:
            return h2


class _PyramidPoolModule(nn.Sequential):

    def __init__(self, in_ch, pyramids=[6, 3, 2, 1]):
        super(_PyramidPoolModule, self).__init__()
        out_ch = in_ch // len(pyramids)
        self.stages = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(output_size=p)),
                ('conv', _ConvBNReLU(in_ch, out_ch, 1, 1, 0, 1)),
            ])) for p in pyramids])

    def forward(self, x):
        hs = [x]
        height, width = x.size()[2:]
        for stage in self.stages:
            h = stage(x)
            h = F.upsample(h, (height, width), mode='bilinear')
            hs.append(h)
        return torch.cat(hs, dim=1)


class PSPNet(nn.Module):
    def __init__(self, n_class, n_blocks, pyramids):
        super(PSPNet, self).__init__()
        self.n_class = n_class
        self.fcn = _DilatedFCN(n_blocks=n_blocks)
        self.ppm = _PyramidPoolModule(in_ch=2048, pyramids=pyramids)
        # Main branch
        self.final = nn.Sequential(OrderedDict([
            ('conv5_4', _ConvBNReLU(4096, 512, 3, 1, 1, 1)),
            ('drop5_4', nn.Dropout2d(p=0.1)),
            ('conv6', nn.Conv2d(512, n_class, 1, stride=1, padding=0))
        ]))
        # Auxiliary branch
        self.aux = nn.Sequential(OrderedDict([
            ('conv4_aux', _ConvBNReLU(1024, 256, 3, 1, 1, 1)),
            ('drop4_aux', nn.Dropout2d(p=0.1)),
            ('conv6_1', nn.Conv2d(256, n_class, 1, stride=1, padding=1)),
        ]))

    def forward(self, x):
        if self.training:
            aux, h = self.fcn(x)
            aux = self.aux(aux)
        else:
            h = self.fcn(x)

        h = self.ppm(h)
        h = self.final(h)

        if self.training:
            return aux, h
        else:
            return h


if __name__ == '__main__':
    model = PSPNet(n_class=150, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])
    print list(model.named_children())

    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 473, 473))
    print model(image).size()
