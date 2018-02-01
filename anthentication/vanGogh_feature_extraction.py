# A realization of vanGogh method
# python2.7

from __future__ import print_function
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.signal import convolve2d
import os

print(os.getcwd())
# os.chdir('../')

# 18 filters
filters = dict({
    0: np.dot(1/16, [1, 2, 1, 2, 4, 2, 1, 2, 1]).reshape(3, 3),
    1: np.dot(1/16, [1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3),
    2: np.dot(1/16, [1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape(3, 3),
    3: np.dot(np.sqrt(2)/16, [1, 1, 0, 1, 0, -1, 0, -1, -1]).reshape(3, 3),
    4: np.dot(np.sqrt(2)/16, [0, 1, 1, -1, 0, 1, -1, -1, 0]).reshape(3, 3),
    5: np.dot(np.sqrt(7)/24, [1, 0, -1, 0, 0, 0, -1, 0, 1]).reshape(3, 3),
    6: np.dot(1/48, [-1, 2, -1, -2, 4, -2, -1, 2, -1]).reshape(3, 3),
    7: np.dot(1/48, [-1, -2, -1, 2, 4, 2, -1, -2, -1]).reshape(3, 3),
    8: np.dot(1/12, [0, 0, -1, 0, 2, 0, -1, 0, 0]).reshape(3, 3),
    9: np.dot(1/12, [-1, 0, 0, 0, 2, 0, 0, 0, -1]).reshape(3, 3),
    10: np.dot(np.sqrt(2)/12, [0, 1, 0, -1, 0, -1, 0, 1, 0]).reshape(3, 3),
    11: np.dot(np.sqrt(2)/16, [-1, 0, 1, 2, 0, -2, -1, 0, 1]).reshape(3, 3),
    12: np.dot(np.sqrt(2)/16, [-1, 2, -1, 0, 0, 0, 1, -2, 1]).reshape(3, 3),
    13: np.dot(1/48, [1, -2, 1, -2, 4, -2, 1, -2, 1]).reshape(3, 3),
    14: np.dot(np.sqrt(2)/12, [0, 0, 0, -1, 2, -1, 0, 0, 0]).reshape(3, 3),
    15: np.dot(np.sqrt(2)/24, [-1, 2, -1, 0, 0, 0, -1, 2, -1]).reshape(3, 3),
    16: np.dot(np.sqrt(2)/12, [0, -1, 0, 0, 2, 0, 0, -1, 0]).reshape(3, 3),
    17: np.dot(np.sqrt(2)/24, [-1, 0, -1, 2, 0, 2, -1, 0, -1]).reshape(3, 3),
})
# print(np.ones((3, 3)))
# 18 filters which are rotated 180 degree
filters_180 = dict({
    0: np.dot(1/16, [1, 2, 1, 2, 4, 2, 1, 2, 1]).reshape(3, 3),
    1: np.dot(1/16, [-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3),
    2: np.dot(1/16, [-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3, 3),
    3: np.dot(np.sqrt(2)/16, [-1, -1, 0, -1, 0, 1, 0, 1, 1]).reshape(3, 3),
    4: np.dot(np.sqrt(2)/16, [0, -1, -1, 1, 0, -1, 1, 1, 0]).reshape(3, 3),
    5: np.dot(np.sqrt(7)/24, [1, 0, -1, 0, 0, 0, -1, 0, 1]).reshape(3, 3),
    6: np.dot(1/48, [-1, 2, -1, -2, 4, -2, -1, 2, -1]).reshape(3, 3),
    7: np.dot(1/48, [-1, -2, -1, 2, 4, 2, -1, -2, -1]).reshape(3, 3),
    8: np.dot(1/12, [0, 0, -1, 0, 2, 0, -1, 0, 0]).reshape(3, 3),
    9: np.dot(1/12, [-1, 0, 0, 0, 2, 0, 0, 0, -1]).reshape(3, 3),
    10: np.dot(np.sqrt(2)/12, [0, 1, 0, -1, 0, -1, 0, 1, 0]).reshape(3, 3),
    11: np.dot(np.sqrt(2)/16, [1, 0, -1, -2, 0, 2, 1, 0, -1]).reshape(3, 3),
    12: np.dot(np.sqrt(2)/16, [1, -2, 1, 0, 0, 0, -1, 2, -1]).reshape(3, 3),
    13: np.dot(1/48, [1, -2, 1, -2, 4, -2, 1, -2, 1]).reshape(3, 3),
    14: np.dot(np.sqrt(2)/12, [0, 0, 0, -1, 2, -1, 0, 0, 0]).reshape(3, 3),
    15: np.dot(np.sqrt(2)/24, [-1, 2, -1, 0, 0, 0, -1, 2, -1]).reshape(3, 3),
    16: np.dot(np.sqrt(2)/12, [0, -1, 0, 0, 2, 0, 0, -1, 0]).reshape(3, 3),
    17: np.dot(np.sqrt(2)/24, [-1, 0, -1, 2, 0, 2, -1, 0, -1]).reshape(3, 3),
})


def img2grey2norm(img):
    # Input:
    #      img: PIL.Image file with 'RGB' channels
    # Onput:
    #      img: ndarray with normalization

    img = img.convert('L')
    # img.show()
    img = np.array(img)
    # plt.imshow(img)
    # plt.show()
    img = preprocessing.normalize(img)
    # plt.imshow(img)
    # plt.show()
    # print(img)
    return img


def get_features(TorF, kernel, num_file, patch=16):
    # Input:
    #      TorF: True for the positive training samples and
    #            False for the negative training samples
    # Onput:
    #      features: 54 features for each sample
    print('Start computing features of {} samples......'.format(TorF))
    run = 0  # count for the files done
    FFeatures = []

    for i in range(num_file):
        Features = []
        for ii in range(patch):
            img = Image.open('../data/train_16patch_NewName/{}/{}_{}.png'.format(TorF, i, ii))

            img = img2grey2norm(img)

            features = np.ones((3, 18))
            run += 1

            for j in range(18):
                c1 = convolve2d(img, kernel[j], 'valid')
                # print(c1.shape)
                stat_mean = np.mean(c1)
                stat_std = np.std(c1)
                # stat_tail
                c2 = np.abs(c1 - stat_mean) - stat_std
                stat_tail = np.sum(c2 > 0) / np.sum(img > 0)

                features[0, j] = stat_mean
                features[1, j] = stat_std
                features[2, j] = stat_tail

            Features.extend(features.reshape(1, -1))
            print('{} images done.'.format(run))
        Features = np.array(Features).reshape(patch, 54)
        FFeatures.append(Features)

    return np.array(FFeatures).reshape(num_file, patch, 54)


kernel = filters_180  # / filters
false_features = get_features(False, kernel, 9)
true_features = get_features(True, kernel, 12)
all_features = np.concatenate((false_features, true_features), axis=0)

# standardize
all_features = all_features.reshape(-1, 54)
all_features = preprocessing.scale(all_features, axis=0)
all_features = all_features.reshape(21, 16, 54)

assert all_features.shape == (21, 16, 54), 'output wrong size'

print('Saving features......')
# np.save('false_features.npy', false_features)
# np.save('true_features.npy', true_features)
np.save('patch_features_norm_std.npy', all_features)
print('Mission completed.')
