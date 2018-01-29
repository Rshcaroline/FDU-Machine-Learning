# A realization of vanGogh method

import numpy as np
from PIL import Image
from sklearn import preprocessing

img = Image.open('./data/train_raw/png/True/2_.png')
img = np.array(img)
img = preprocessing.normalize(img)
print(img)
