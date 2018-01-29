#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/27 下午3:38
# @Author  : Shihan Ran
# @Site    : 
# @File    : split_pic.py
# @Software: PyCharm
# @Description: This is the file which can split a large picture into small images.
#               Pay attention to the relative path.

import os

import skimage.io as io
from tqdm import tqdm


def split_pic(IMG_SIZE= 227, FOLDER="False"):

    IMG_PATH = "../data/train_raw/png"

    # train_600
    OUT_FOLDER = "../data/train_" + str(IMG_SIZE)

    if not os.path.exists(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)

    # train_600/False
    OUT_PATH = os.path.join(OUT_FOLDER, FOLDER)

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    paths = os.listdir(os.path.join(IMG_PATH, FOLDER))

    for path in tqdm(paths):
        count = 0

        folder_path = os.path.join(IMG_PATH, FOLDER)
        img = io.imread(os.path.join(folder_path, path))

        height, width = img.shape[:2]
        # overlap 0.2
        # without 100
        for i in range(IMG_SIZE + 100, height - 100, int(0.8 * IMG_SIZE)):
            for j in range(IMG_SIZE + 100, width - 100, int(0.8 * IMG_SIZE)):
                f = path.replace('.jpg', '_%d.jpg' % count)
                f = f.replace('.png', '_%d.png' % count)
                io.imsave(os.path.join(OUT_PATH, f),
                          img[i-IMG_SIZE:i, j-IMG_SIZE:j])
                count += 1


if __name__ == '__main__':
    IMG_SIZE = 1000
    split_pic(IMG_SIZE, FOLDER="False")
    split_pic(IMG_SIZE, FOLDER="True")