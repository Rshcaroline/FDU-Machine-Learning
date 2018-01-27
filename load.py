# Pytorch loader

import torch
from torchvision import transforms, datasets
import numpy as np
import os

os.chdir('../FDU-ML-Final-Project/data/')


class Args:

    def __init__(self):
        self.data = os.getcwd()
        self.batchSize = 0
        self.nThreads = 1


# Data loading code
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

args = Args()
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
train = datasets.ImageFolder(traindir, transform)
val = datasets.ImageFolder(valdir, transform)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)
