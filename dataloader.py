#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/26 下午10:15
# @Author  : Shihan Ran
# @Site    : 
# @File    : dataloader.py
# @Software: PyCharm
# @Description: This is the file which can split a large picture into small images.

import argparse
import skimage.io as io

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

traindir = "./data/train/png"


def my_loader(path):
    return Image.open(path)

train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        loader=my_loader
)

# DataLoader multiprocessing
# 0: shape = [num_of_items, channels, pixels, pixels]
# 1: length = num_of_items, it records labels
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=8, pin_memory=True)

for i in train_loader:
    print i