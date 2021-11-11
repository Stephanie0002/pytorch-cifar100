""" train and test dataset
author Stephanie
reference https://github.com/weiaicunzai/pytorch-cifar100
"""
import os
import pickle
import numpy
import torch
import torchvision
from torchvision import transforms
from Utils import config

std = config.CIFAR100_TRAIN_STD
mean = config.CIFAR100_TRAIN_MEAN


def cifar100_dataset():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar100_training = torchvision.datasets.CIFAR100(root=config.CIFAR100_PATH, train=True, download=True,
                                                      transform=transform_train)
    train_loader = torch.utils.data.DataLoader(cifar100_training, batch_size=config.BATCH, shuffle=True,
                                               num_workers=config.NUM_WORKERS)

    cifar100_testing = torchvision.datasets.CIFAR100(root=config.CIFAR100_PATH, train=False, download=True,
                                                     transform=transform_test)
    test_loader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100, shuffle=False,
                                             num_workers=config.NUM_WORKERS)

    return train_loader, test_loader
