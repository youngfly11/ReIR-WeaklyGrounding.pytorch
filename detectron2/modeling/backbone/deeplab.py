#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/11/7 20:15

from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import BACKBONE_REGISTRY
from detectron2.config import global_cfg as cfg
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from .backbone import Backbone
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

_BOTTLENECK_EXPANSION = 4

# try:
#     from encoding.nn import SyncBatchNorm
#     _BATCH_NORM = SyncBatchNorm
# except:
#     _BATCH_NORM = nn.BatchNorm2d

_BATCH_NORM = nn.BatchNorm2d


__all__ = ["build_deeplabv2"]

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))

    def freeze_bn(self):
        """
        Fix the batch normalization and change it into evaluate mode
        """

        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()
                for each in m.parameters():
                    each.requires_grad=False


class build_deeplabv2(Backbone):
    def __init__(self):
        super(build_deeplabv2, self).__init__()
        self.deeplab = DeepLabV2(n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        # self.model = self.model.to(torch.device('cuda'))
        pretrain_weight = torch.load(cfg.MODEL.BACKBONE.PRETRAIN_PATH, map_location=torch.device('cpu'))
        model_static = self.deeplab.state_dict()

        load_weight = {}
        for key, value in model_static.items():
            aug_key = 'base.' + key
            if aug_key in pretrain_weight:
                load_weight[key] = pretrain_weight[aug_key]
        self.deeplab.load_state_dict(load_weight)
        print('load pre-trained weight on Pascal successfully')

    def forward(self, input):

        """
        Input is the raw image, batch*3*H*W
        Output is the list [out3, out4, out5]

        """
        layer_out1 = self.deeplab.layer1(input)      ## b*N*(H/4)*(W/4)
        layer_out2 = self.deeplab.layer2(layer_out1) ## b*N*(H/4)*(W/4)
        layer_out3 = self.deeplab.layer3(layer_out2) ## b*N*(H/8)*(W/8)
        layer_out4 = self.deeplab.layer4(layer_out3) ## b*N*(H/8)*(W/8)
        layer_out5 = self.deeplab.layer5(layer_out4) ## b*N*(H/8)*(W/8)
        return [layer_out3, layer_out4, layer_out5]

@BACKBONE_REGISTRY.register()
def build_deeplabv2_backbone(cfg, input_shape: ShapeSpec):

    backbone = build_deeplabv2()
    return backbone
