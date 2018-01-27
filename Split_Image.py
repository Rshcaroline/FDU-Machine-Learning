#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/26 下午10:15
# @Author  : Shihan Ran
# @Site    : 
# @File    : Split_Image.py
# @Software: PyCharm
# @Description: This is the file which can split a large picture into small images.

import os
import tifffile as tif
import argparse

from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

traindir = "./data/train"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# 用ImageFolder来读取dataset
train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

# DataLoader多线程读取
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=5, pin_memory=True)

print train_loader