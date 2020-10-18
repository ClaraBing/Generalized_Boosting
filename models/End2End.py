from __future__ import print_function
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pickle as pk
import os

from .model_utils import *


class Ensemble(nn.Module):

    def __init__(self, BasicBlock, BasicClfBlock,
                 num_classes=10, input_size = (3,32,32), depth = 8, width = 128):
        super(Ensemble, self).__init__()

        # add blocks to dictionary
        self.blocks = [BasicBlock(input_size, width=width, residual = False, scale=1)]
        for i in range(1,depth):
            self.blocks.append(BasicBlock(self.blocks[i-1].sizeof_output(), width=width, residual = True, scale=1))

        self.layer = nn.Sequential(*self.blocks)

        # add classification layer
        self.cf = BasicClfBlock(self.blocks[depth-1].sizeof_output(), num_classes)

    def forward(self, x):
        x = self.layer(x)
        return self.cf(x)


def train(train_loader, test_loader, BasicBlock, BasicClfBlock,
          opt='SGD', epochs=20,
          lr=0.005, momentum = 0.5, weight_decay=0.0, min_lr=0,
          depth = 30, width=128, scheduler_tolerance=5000,
          use_cuda=True, num_classes = 10, input_size = [3,32,32]):

    model = Ensemble(BasicBlock=BasicBlock, BasicClfBlock = BasicClfBlock, num_classes=num_classes,
                     input_size=input_size, depth=depth, width = width)

    test_loss = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    test_err = np.zeros(epochs)
    train_err = np.zeros(epochs)
    if use_cuda:
        model.cuda()

    optimizer = None
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=scheduler_tolerance, verbose=True)

    for epoch in range(epochs):
        train_helper(model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, data_loader=train_loader, use_cuda=use_cuda)

        test_loss[epoch], test_err[epoch] = test_helper(model=model, data_loader=test_loader, use_cuda=use_cuda, is_test=True)
        train_loss[epoch], train_err[epoch] = test_helper(model=model, data_loader=train_loader, use_cuda=use_cuda, is_test=False)

        # stop training if lr too low
        for param_group in optimizer.param_groups:
          curr_lr = param_group['lr']
          break
        if curr_lr < min_lr:
          break

    return {'model':model, 'train_loss':train_loss, 'train_err':train_err,
            'test_loss':test_loss, 'test_err':test_err}

