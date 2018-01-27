#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/27 上午11:33
# @Author  : Shihan Ran
# @Site    : 
# @File    : tif2jpg.py
# @Software: PyCharm
# @Description: This is a file that turn tif image into jpg.

import os
import tifffile as tif

from tqdm import tqdm


def tif2jpg():
    IMG_PATH = './data/train/True'
    DIR_PATH = './data/train/True_jpg'

    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)

    paths = os.listdir(IMG_PATH)

    for path in tqdm(paths):
        img = tif.imread(os.path.join(IMG_PATH, path))
        jpg_path = path.replace('tiff', 'jpg')
        jpg_path = jpg_path.replace('tif', 'jpg')
        jpg_path = jpg_path.replace('TIF', 'jpg')
        tif.imsave(os.path.join(DIR_PATH, jpg_path), img)

tif2jpg()