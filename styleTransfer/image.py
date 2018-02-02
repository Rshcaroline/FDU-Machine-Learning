#

from PIL import Image, ImageFilter, ImageDraw
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

style = Image.open('./images/style15.jpg')
content_img = Image.open('./images/content1.jpg')

# use the contour image to remove the colors in content img
contour_img = content_img.filter(ImageFilter.CONTOUR)
# try the detailed contour image
detailed_contour_img = content_img.filter(ImageFilter.DETAIL)
detailed_contour_img = detailed_contour_img.filter(ImageFilter.CONTOUR)
# try the edge enhanced contour image
edge_enhanced_contour_img = content_img.filter(ImageFilter.EDGE_ENHANCE)
edge_enhanced_contour_img = edge_enhanced_contour_img.filter(ImageFilter.CONTOUR)

# plt.figure(1)
# plt.imshow(style)

grey_scale_img = content_img.convert('L')
grey_scale_img.show()
# content_img.show()
# contour_img.show()
# detailed_contour_img.show()
# edge_enhanced_contour_img.show()

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

