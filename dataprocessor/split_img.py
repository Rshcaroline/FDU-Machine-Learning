#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/27 下午3:38
# @Author  : Shihan Ran
# @Site    : 
# @File    : split_pic.py
# @Software: PyCharm
# @Description: This is the file which can split a large picture into small images.
#               Pay attention to the relative path

from __future__ import division
import os

import skimage.io as io
from tqdm import tqdm


def split_pic(IMG_SIZE= 227):

    IMG_PATH = "../data/images"

    # train_600
    # OUT_PATH = "../data/images_" + str(IMG_SIZE)
    OUT_PATH = "../Paper-Implementations/DiscoGAN/data/our_data_227/B"

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    paths = os.listdir(IMG_PATH)
    print paths
    # paths = paths.sort()

    for path in tqdm(paths):
        count = 0

        img = io.imread(os.path.join(IMG_PATH, path))

        height, width = img.shape[:2]
        # overlap 0.2
        # without 100
        for i in range(IMG_SIZE + 100, height - 100, int(0.8 * IMG_SIZE)):
            for j in range(IMG_SIZE + 100, width - 100, int(0.8 * IMG_SIZE)):
        # IMG_SIZE_i = int((height - 200)/4)
        # IMG_SIZE_j = int((width - 200)/4)

        # for i in range(4):
        #     for j in range(4):
                f = path.replace('.jpeg', '_%d.jpeg' % count)
                f = f.replace('.jpg', '_%d.jpg' % count)
                f = f.replace('.png', '_%d.png' % count)

                io.imsave(os.path.join(OUT_PATH, f),
                          img[i - IMG_SIZE:i, j - IMG_SIZE:j])
                count += 1


if __name__ == '__main__':
    # IMG_SIZE = 1000
    split_pic(IMG_SIZE=100)
