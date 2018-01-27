#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/27 上午11:33
# @Author  : Shihan Ran
# @Site    : 
# @File    : tif2png.py
# @Software: PyCharm
# @Description: This is a file that turn tif image into png.

import os
import tifffile as tif
import numpy as np
import skimage.io as io

from tqdm import tqdm


def tif2png():
    """
    This function helps transfer given tiff image into png image hence I can use ImageFolder.
    :return: 
    """
    folder = 'True'
    IMG_PATH = os.path.join('./data/train/', folder)
    DIR_PATH = os.path.join('./data/train/png', folder)

    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)

    paths = os.listdir(IMG_PATH)

    for path in tqdm(paths):
        img = tif.imread(os.path.join(IMG_PATH, path))
        img = img[:, :, 0:3]
        print img.shape
        png_path = path.replace('tiff', 'png')
        png_path = png_path.replace('tif', 'png')
        png_path = png_path.replace('TIF', 'png')
        io.imsave(os.path.join(DIR_PATH, png_path), img)

def equal():
    """
    This function is aimed to find whether or not tif image == png image.
    It only depends on the numeric expression of numpy.
    :return: 
    """
    tif_PATH = './data/train/False/27.tiff'
    png_PATH = './data/train/png//False/27.png'

    img_tif = tif.imread(tif_PATH)
    img_png = io.imread(png_PATH)

    print img_tif
    print img_png

    print np.equal(img_tif[:, :, 0:3], img_png)


if __name__ == '__main__':
    # tif2png()
    equal()