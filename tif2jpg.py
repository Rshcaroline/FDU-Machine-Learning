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
import numpy as np
import skimage.io

from tqdm import tqdm


def tif2jpg():
    """
    This function helps transfer given tiff image into jpg image hence I can use ImageFolder.
    :return: 
    """
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

def equal():
    """
    This function is aimed to find whether or not tif image == jpg image.
    It only depends on the numeric expression of numpy.
    :return: 
    """
    tif_PATH = './data/train/True/2.TIF'
    jpg_PATH = './data/train/True_jpg/2.jpg'

    img_tif = tif.imread(tif_PATH)
    img_jpg = skimage.io.imread(jpg_PATH)

    print np.equal(img_tif, img_jpg)


if __name__ == '__main__':
    tif2jpg()