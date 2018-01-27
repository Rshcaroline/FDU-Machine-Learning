#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/26 下午9:04
# @Author  : Shihan Ran
# @Site    : 
# @File    : Classification.py
# @Software: PyCharm
# @Description: This is a file which loads pytorch pre-trained model.

import torchvision.models as models
from torch import nn

resnet = models.resnet152(pretrained=True)

# resnet第一层卷积接收的通道是3
resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

# 原本为1000类，改为2类
resnet.fc = nn.Linear(2048, 2)

print resnet