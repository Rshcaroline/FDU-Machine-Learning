#

from PIL import Image, ImageFilter, ImageDraw
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

style = Image.open('./images/style15.jpg')
content = Image.open('./images/content1.jpg')

contour = content.filter(ImageFilter.CONTOUR)
# plt.figure(1)
# plt.imshow(style)
content.show()
contour.show()
# loader = transforms.ToTensor()
# unloader = transforms.ToPILImage()  # reconvert into PIL image
#
# plt.figure(1)
# plt.imshow(style)
# plt.pause(1)
# style = loader(style)
# style = style[:, :1862, :3264]
# print(style.size())
#
# style = unloader(style)
# plt.cla()
# plt.imshow(style)

