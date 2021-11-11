""" configurations for this project
author Stephanie
reference https://github.com/weiaicunzai/pytorch-cifar100
"""
import os
from datetime import datetime

# CIFAR100 dataset path (python version)
CIFAR100_PATH = '/home/hzq/pycharm/data/'

# mean and std of cifar100 dataset
CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# Dataload num_workers
NUM_WORKERS = 2

# network type
NET = 'ShuffleNet'
# 'ShuffleNet'
# 'DenseNet'
# 'ResNet'

# start again or not
RESUME = False
# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'
# data format
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = 'run_log'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 100

'''
train 参数
'''
# gpu or cpu
IS_GPU = True
# total training epoches batchs
BATCH = 1024
# for ShuffleNet: 1024
# for ResNet\DenseNet: 128
EPOCH = 1000

# learning rate args
INIT_LR = 0.5
# for ShuffleNet: 0.5
# for ResNet\DenseNet: 0.1
WARM_UP_PHASE = 1

# multiple lr schedule args for ResNet/DenseNet
MILESTONES = [40, 60, 80]
# for DenseNet:[30, 50, 70]
# for ResNet:[40, 50, 60]

# linear lr schedule args for ShuffleNet
lr_range = [0.5, 0]
# for ShuffleNet:[0.5, 0]

# weight decay
Lambda = 0.00004
# for ShuffleNet:0.00004
# for ResNet\DenseNet:0.0001

# momentum
MOMENTUM = 0.9
