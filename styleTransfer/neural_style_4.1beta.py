# -*- coding: utf-8 -*-
# python2.7

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


######################################################################
# Cuda
# ~~~~
#
# If you have a GPU on your computer, it is preferable to run the
# algorithm on it, especially if you want to try larger networks (like
# VGG). For this, we have ``torch.cuda.is_available()`` that returns
# ``True`` if you computer has an available GPU. Then, we can use method
# ``.cuda()`` that moves allocated proccesses associated with a module
# from the CPU to the GPU. When we want to move back this module to the
# CPU (e.g. to use numpy), we use the ``.cpu()`` method. Finally,
# ``.type(dtype)`` will be use to convert a ``torch.FloatTensor`` into
# ``torch.cuda.FloatTensor`` to feed GPU processes.
#

torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


######################################################################
# Load images
# ~~~~~~~~~~~
#
# In order to simplify the implementation, let's start by importing a
# style and a content image of the same dimentions. We then scale them to
# the desired output image size (128 or 512 in the example, depending on gpu
# availablity) and transform them into torch tensors, ready to feed
# a neural network:
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <http://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <http://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images``


# desired size of the output image
imsize = 512 if use_cuda else 128  # use small size if no gpu

loader = transforms.ToTensor()
unloader = transforms.ToPILImage()  # reconvert into PIL image
loader_resize = transforms.Compose([
    transforms.Scale(imsize),
    transforms.ToTensor(),
])


def image2Var(image):
    # image = Image.open(image_name)
    image = Variable(loader_resize(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


######################################################################
# Imported PIL images has values between 0 and 255. Transformed into torch
# tensors, their values are between 0 and 1. This is an important detail:
# neural networks from torch library are trained with 0-1 tensor image. If
# you try to feed the networks with 0-255 tensor images the activated
# feature maps will have no sense. This is not the case with pre-trained
# networks from the Caffe library: they are trained with 0-255 tensor
# images.
#
# Display images
# ~~~~~~~~~~~~~~
#
# We will use ``plt.imshow`` to display images. So we need to first
# reconvert them into PIL images:
#

# plt.ion()


def imsave(tensor, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, image.size()[-2], image.size()[-1])  # remove the fake batch dimension
    image = unloader(image)
    # plt.imshow(image)
    plt.imsave(title, image, format='png')
    if title is not None:
        plt.title(title)
    # plt.pause(0.001) # pause a bit so that plots are updated


# plt.figure()
# imshow(style_img.data, title='Style Image')
#
# plt.figure()
# imshow(content_img.data, title='Content Image')


######################################################################
# Content loss
# ~~~~~~~~~~~~
#
# The content loss is a function that takes as input the feature maps
# :math:`F_{XL}` at a layer :math:`L` in a network fed by :math:`X` and
# return the weigthed content distance :math:`w_{CL}.D_C^L(X,C)` between
# this image and the content image. Hence, the weight :math:`w_{CL}` and
# the target content :math:`F_{CL}` are parameters of the function. We
# implement this function as a torch module with a constructor that takes
# these parameters as input. The distance :math:`\|F_{XL} - F_{YL}\|^2` is
# the Mean Square Error between the two sets of feature maps, that can be
# computed using a criterion ``nn.MSELoss`` stated as a third parameter.
#
# We will add our content losses at each desired layer as additive modules
# of the neural network. That way, each time we will feed the network with
# an input image :math:`X`, all the content losses will be computed at the
# desired layers and, thanks to autograd, all the gradients will be
# computed. For that, we just need to make the ``forward`` method of our
# module returning the input: the module becomes a ''transparent layer''
# of the neural network. The computed loss is saved as a parameter of the
# module.
#
# Finally, we define a fake ``backward`` method, that just call the
# backward method of ``nn.MSELoss`` in order to reconstruct the gradient.
# This method returns the computed loss: this will be useful when running
# the gradient descent in order to display the evolution of style and
# content losses.
#

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


######################################################################
# .. Note::
#    **Important detail**: this module, although it is named ``ContentLoss``,
#    is not a true PyTorch Loss function. If you want to define your content
#    loss as a PyTorch Loss, you have to create a PyTorch autograd Function
#    and to recompute/implement the gradient by the hand in the ``backward``
#    method.
#
# Style loss
# ~~~~~~~~~~
#
# For the style loss, we need first to define a module that compute the
# gram produce :math:`G_{XL}` given the feature maps :math:`F_{XL}` of the
# neural network fed by :math:`X`, at layer :math:`L`. Let
# :math:`\hat{F}_{XL}` be the re-shaped version of :math:`F_{XL}` into a
# :math:`K`\ x\ :math:`N` matrix, where :math:`K` is the number of feature
# maps at layer :math:`L` and :math:`N` the lenght of any vectorized
# feature map :math:`F_{XL}^k`. The :math:`k^{th}` line of
# :math:`\hat{F}_{XL}` is :math:`F_{XL}^k`. We let you check that
# :math:`\hat{F}_{XL} \cdot \hat{F}_{XL}^T = G_{XL}`. Given that, it
# becomes easy to implement our module:
#

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


######################################################################
# The longer is the feature maps dimension :math:`N`, the bigger are the
# values of the gram matrix. Therefore, if we don't normalize by :math:`N`,
# the loss computed at the first layers (before pooling layers) will have
# much more importance during the gradient descent. We dont want that,
# since the most interesting style features are in the deepest layers!
#
# Then, the style loss module is implemented exactly the same way than the
# content loss module, but we have to add the ``gramMatrix`` as a
# parameter:
#

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


######################################################################
# Load the neural network
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Now, we have to import a pre-trained neural network. As in the paper, we
# are going to use a pretrained VGG network with 19 layers (VGG19).
#
# PyTorch's implementation of VGG is a module divided in two child
# ``Sequential`` modules: ``features`` (containing convolution and pooling
# layers) and ``classifier`` (containing fully connected layers). We are
# just interested by ``features``:
#

cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()


######################################################################
# A ``Sequential`` module contains an ordered list of child modules. For
# instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU,
# Maxpool2d, Conv2d, ReLU...) aligned in the right order of depth. As we
# said in *Content loss* section, we wand to add our style and content
# loss modules as additive 'transparent' layers in our network, at desired
# depths. For that, we construct a new ``Sequential`` module, in wich we
# are going to add modules from ``vgg19`` and our loss modules in the
# right order:
#

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


######################################################################
# .. Note::
#    In the paper they recommend to change max pooling layers into
#    average pooling. With AlexNet, that is a small network compared to VGG19
#    used in the paper, we are not going to see any difference of quality in
#    the result. However, you can use these lines instead if you want to do
#    this substitution:
#
#    ::
#
#        # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
#        #                         stride=layer.stride, padding = layer.padding)
#        # model.add_module(name,avgpool)


######################################################################
# Input image
# ~~~~~~~~~~~
#
# Again, in order to simplify the code, we take an image of the same
# dimensions than content and style images. This image can be a white
# noise, or it can also be a copy of the content-image.
#

# input_img = content_img

# if you want to use a white noise instead uncomment the below line:
# input_img = Variable(torch.randn(content_img.data.size())).type(dtype)

# add the original input image to the figure:
# plt.figure()


######################################################################
# Gradient descent
# ~~~~~~~~~~~~~~~~
#
# As Leon Gatys, the author of the algorithm, suggested
# `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__,
# we will use L-BFGS algorithm to run our gradient descent. Unlike
# training a network, we want to train the input image in order to
# minimise the content/style losses. We would like to simply create a
# PyTorch  L-BFGS optimizer, passing our image as the variable to optimize.
# But ``optim.LBFGS`` takes as first argument a list of PyTorch
# ``Variable`` that require gradient. Our input image is a ``Variable``
# but is not a leaf of the tree that requires computation of gradients. In
# order to show that this variable requires a gradient, a possibility is
# to construct a ``Parameter`` object from the input image. Then, we just
# give a list containing this ``Parameter`` to the optimizer's
# constructor:
#

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


######################################################################
# **Last step**: the loop of gradient descent. At each step, we must feed
# the network with the updated input in order to compute the new losses,
# we must run the ``backward`` methods of each loss to dynamically compute
# their gradients and perform the step of gradient descent. The optimizer
# requires as argument a "closure": a function that reevaluates the model
# and returns the loss.
#
# However, there's a small catch. The optimized image may take its values
# between :math:`-\infty` and :math:`+\infty` instead of staying between 0
# and 1. In other words, the image might be well optimized and have absurd
# values. In fact, we must perform an optimization under constraints in
# order to keep having right vaues into our input image. There is a simple
# solution: at each step, to correct the image to maintain its values into
# the 0-1 interval.
#

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)

    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data
######################################################################
# Finally, run the algorithm

# sphinx_gallery_thumbnail_number = 4
# plt.ioff()
# plt.show()


def trick_content_img(trick, content_img):
    if trick == 1:
        # use the contour image to remove the colors in content img
        content_img = content_img.filter(ImageFilter.CONTOUR)
    elif trick == 2:
        # try the detailed contour image
        detailed_contour_img = content_img.filter(ImageFilter.DETAIL)
        content_img = detailed_contour_img.filter(ImageFilter.CONTOUR)
    elif trick == 3:
        # try the edge enhanced contour image
        edge_enhanced_contour_img = content_img.filter(ImageFilter.EDGE_ENHANCE)
        content_img = edge_enhanced_contour_img.filter(ImageFilter.CONTOUR)
    elif trick == 4:
        # try the edge enhanced more contour image
        edge_enhanced_more_contour_img = content_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        content_img = edge_enhanced_more_contour_img.filter(ImageFilter.CONTOUR)
    elif trick == 5:
        # grey scale
        content_img = content_img.convert('L')
    elif trick == 6:
        # black and white
        content_img = content_img.convert('1')

    content_img = image2Var(content_img).type(dtype)

    return content_img


def style_transfer(style_num, content_num, trick, num_steps,
                   style_file='png', content_file='jpg',
                   style_weight=1000, content_weight=1):
    # make sure that the style image size is the same as that of content
    style_img = Image.open("./images/style{}.{}".format(style_num, style_file))
    content_img = Image.open("./images/content{}.{}".format(content_num, content_file))

    trick_img = trick_content_img(tricks[trick], content_img)

    style_img = loader(style_img)
    # !note that the size of content image is always inverse
    style_img = style_img[:channel[trick], :loader(content_img).size()[1], :loader(content_img).size()[2]]
    # style_img = style_img[:3, :1440, :1080]
    style_img = unloader(style_img)

    style_img = image2Var(style_img).type(dtype)
    content_img = image2Var(content_img).type(dtype)

    print(style_img.size(),
          content_img.size(),
          trick_img.size(),
          )
    assert style_img.size() == trick_img.size(), \
        "we need to import style and trick images of the same size"

    input_img = trick_img  # this can either be the content img or style img or even a white noise
    imsave(input_img.data, title='Input_image_{}'.format(trick))

    output = run_style_transfer(cnn,
                                content_img=trick_img,
                                style_img=style_img,
                                input_img=trick_img,  # for initialization
                                num_steps=num_steps,
                                style_weight=style_weight,
                                content_weight=content_weight,
                                )
    imsave(output, title='{}_plus_{}_{}'.format(style_num, content_num, trick))

    # recurrent learning
    # output = Variable(output)
    # output2 = run_style_transfer(cnn,
    #                              content_img=content_img,
    #                              style_img=style_img,
    #                              input_img=output,
    #                              num_steps=100)
    # imshow(output2, title='Output2 Image')

    #

    # plt.figure()


channel = {
    'none': 3,
    'contour': 3,
    'd_contour': 3,  # detailed contour
    'ee_contour': 3,  # edge enhanced contour
    'eem_contour': 3,  # edge enhanced more contour
    'grey_scale': 1,  # grey scale image [colored]
    'black_white': 1,  # black and white
}
tricks = {
    'none': 0,
    'contour': 1,
    'd_contour': 2,  # detailed contour
    'ee_contour': 3,  # edge enhanced contour
    'eem_contour': 4,  # edge enhanced more contour
    'grey_scale': 5,  # grey scale image [colored]
    'black_white': 6,  # black and white
}
# final API
style_transfer(style_num=6,
               content_num=7,
               trick='grey_scale',
               num_steps=300,
               style_file='png',
               content_file='jpg',
               style_weight=1000,
               content_weight=1,
               )
