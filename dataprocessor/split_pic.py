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


def split_pic(IMG_SIZE= 227, FOLDER="False"):

    IMG_PATH = "../data/train_raw/png"

    # train_600
    # OUT_FOLDER = "../data/train_" + str(IMG_SIZE)
    # OUT_FOLDER = "../data/train_" + str(IMG_SIZE) + "_NoName"
    OUT_FOLDER = "../data/train_16patch_NewName"

    if not os.path.exists(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)

    # train_600/False
    OUT_PATH = os.path.join(OUT_FOLDER, FOLDER)

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    paths = os.listdir(os.path.join(IMG_PATH, FOLDER))
    print paths
    # paths = paths.sort()

    # count = 0
    count_id = 0

    for path in tqdm(paths):
        count = 0

        folder_path = os.path.join(IMG_PATH, FOLDER)
        img = io.imread(os.path.join(folder_path, path))

        height, width = img.shape[:2]
        # overlap 0.2
        # without 100
        # for i in range(IMG_SIZE + 100, height - 100, int(0.8 * IMG_SIZE)):
        #     for j in range(IMG_SIZE + 100, width - 100, int(0.8 * IMG_SIZE)):
        IMG_SIZE_i = int((height - 200)/4)
        IMG_SIZE_j = int((width - 200)/4)

        for i in range(4):
            for j in range(4):
                # f = path.replace('.jpg', '_%d.jpg' % count)
                # f = f.replace('.png', '_%d.png' % count)
                f = '%d_%d.png' % (count_id, count)

                io.imsave(os.path.join(OUT_PATH, f),
                          # img[i-IMG_SIZE:i, j-IMG_SIZE:j])
                          img[(100 + i * IMG_SIZE_i):(100 + (i + 1) * IMG_SIZE_i), \
                          (100 + j * IMG_SIZE_j):(100 + (j + 1) * IMG_SIZE_j)])
                count += 1

        count_id += 1


if __name__ == '__main__':
    # IMG_SIZE = 1000
    split_pic(FOLDER="False")
    split_pic(FOLDER="True")