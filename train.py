""" train network using pytorch
author Stephanie
reference https://github.com/weiaicunzai/pytorch-cifar100
"""

import os
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import cifar100_dataset
from Utils.utils import *
from Utils import config
from models.myResNet import *
from models.myDenseNet import *
from models.myShuffleNet import *


def train(epoch):
    print("===============Start Training===============")
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if config.IS_GPU:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * config.BATCH + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if config.IS_GPU:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if config.IS_GPU:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':
    net = shufflenet_s1_g8()

    # data preprocessing:
    cifar100_training_loader, cifar100_test_loader = cifar100_dataset()

    # Loss and Optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.INIT_LR, momentum=config.MOMENTUM,
                          weight_decay=config.Lambda)
    train_scheduler_multi = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.MILESTONES,
                                                           gamma=0.1)  # learning rate decay
    train_scheduler_linear = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda cur_epoch: 1 - cur_epoch / (config.EPOCH - config.WARM_UP_PHASE))
    iter_per_epoch = len(cifar100_training_loader)

    # Resume or Restart
    if config.RESUME:
        recent_folder = most_recent_folder(os.path.join(config.CHECKPOINT_PATH, config.NET), fmt=config.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, config.NET, recent_folder)
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, config.NET, config.TIME_NOW)

    # use tensorboard to visualization
    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        config.LOG_DIR, config.NET, config.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if config.IS_GPU:
        net = net.cuda()
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if config.RESUME:
        best_weights = best_acc_weights(os.path.join(config.CHECKPOINT_PATH, config.NET, recent_folder))
        if best_weights:
            weights_path = os.path.join(config.CHECKPOINT_PATH, config.NET, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(config.CHECKPOINT_PATH, config.NET, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(config.CHECKPOINT_PATH, config.NET, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(config.CHECKPOINT_PATH, config.NET, recent_folder))

    for epoch in range(1, config.EPOCH + 1):
        if epoch > config.WARM_UP_PHASE:
            train_scheduler_linear.step(epoch)
            # for ShuffleNet: train_scheduler_linear.step(epoch)
            # for ResNet\DenseNet: train_scheduler_multi.step(epoch)

        if config.RESUME:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > config.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=config.NET, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % config.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=config.NET, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
