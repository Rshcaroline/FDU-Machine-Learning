from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# -----Settings-----
torch.cuda.manual_seed(123)

# -----Data Loading-----
datadir = "./data/train"


def my_loader(path):
    return Image.open(path)


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_num=1,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True):
    """
    Refence: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb.js

    Utility function for loading and returning train and valid
    multi-process iterators over the  dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_num: size of validation set, default = 1
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
      
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # load the dataset
    data_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = ImageFolder(data_dir, data_transforms, loader=my_loader)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(valid_num)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset,
                              batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = DataLoader(dataset,
                              batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader


train_loader, valid_loader = get_train_valid_loader(data_dir=datadir, batch_size=20, random_seed=123),

# DataLoader multiprocessing
# 0: shape = [num_of_items, channels, pixels, pixels]
# 1: length = num_of_items, it records labels

# -----Classifier-----
net = models.resnet152(pretrained=True)
net.fc = nn.Linear(2048, 2)

# net = models.resnet18(pretrained=True)
# net.fc = nn.Linear(512, 2)
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

accuracy = []

for iter in range(21):
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # get the inputs
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  # wrap them in Variable
        optimizer.zero_grad()  # zero the parameter gradients

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 50 == 49:
            print('[%d, %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
        running_loss = 0

    correct = 0
    total = 0
    for data in valid_loader:
        inputs, labels = data
    outputs = net(Variable(inputs.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    accuracy.append(correct / total)
    print("---Epoch: %d Dev accuracy:%.3f---" % (epoch + 1, accuracy[-1]))

print('Finished Training')
