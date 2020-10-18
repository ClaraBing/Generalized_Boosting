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
                 num_classes=10, input_size = (3,32,32), iterations = 8, width = 128, width_increment=2, lr=1.):
        super(Ensemble, self).__init__()

        # add layers to dictionary
        self.standby = {0: BasicBlock(input_size, width, residual = False, scale=lr)}
        for i in range(1,iterations):
            self.standby[i] = BasicBlock(self.standby[i-1].sizeof_output(), width=width+width_increment*i, residual = True, scale=lr)


        # add classification layers
        self.standby_cf = {0: BasicClfBlock(self.standby[0].sizeof_output(), num_classes)}
        for i in range(1,iterations): self.standby_cf[i] = BasicClfBlock(self.standby[i].sizeof_output(), num_classes)

        self.frozen_c = []
        self.training_c = None
        self.training_cf = None

    def forward(self, x):
        for c in self.frozen_c:
            x = c(x)
        x = self.training_c(x)
        return self.training_cf(x)


    def add_layer(self, idx):
        if self.training_c:
            self.training_c.requires_grad = False
            self.frozen_c.append(self.training_c)
        if self.training_cf:
            self.training_cf.requires_grad = False
        try:
            self.training_c = self.standby[idx]
            self.training_c.requires_grad = True

            self.training_cf = self.standby_cf[idx]
            self.training_cf.requires_grad = True

            trainable_params = [{'params': self.training_c.parameters()},
                                {'params': self.training_cf.parameters()}
            ]
            return trainable_params
        except:
            print('No more standby layers!')



def train(train_loader, test_loader, BasicBlock, BasicClfBlock,
          opt='SGD', epochs=20,
          lr=0.005, momentum = 0.5, weight_decay=0.0, min_lr=0,
          iterations = 30, width=128,greedy_lr=1., scheduler_tolerance=5000,
          width_increment=16,  use_cuda=True, num_classes = 10, input_size = [3,32,32]):

    model = Ensemble(BasicBlock=BasicBlock, BasicClfBlock = BasicClfBlock, num_classes=num_classes,
                     input_size=input_size, iterations=iterations, width = width, width_increment=width_increment,
                     lr=greedy_lr)

    test_loss = np.zeros(iterations)
    train_loss = np.zeros(iterations)
    test_err = np.zeros(iterations)
    train_err = np.zeros(iterations)
    for i in range(iterations):
        trainable_params = model.add_layer(i)
        if use_cuda:
            model.cuda()

        optimizer = None
        if opt == 'Adam':
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif opt == 'SGD':
            optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=scheduler_tolerance, verbose=True)

        for epoch in range(0, epochs):
            train_helper(model=model, optimizer=optimizer, scheduler=scheduler,
                         epoch=epoch, data_loader=train_loader, use_cuda=use_cuda)
            if (epoch+1) % 5 == 0:
              test_loss[i], test_err[i] = test_helper(model=model, data_loader=test_loader, use_cuda=use_cuda, is_test=True)

            # stop training if lr too low
            for param_group in optimizer.param_groups:
              curr_lr = param_group['lr']
              break
            if curr_lr < min_lr:
              break

        test_loss[i], test_err[i] = test_helper(model=model, data_loader=test_loader, use_cuda=use_cuda, is_test=True)
        train_loss[i], train_err[i] = test_helper(model=model, data_loader=train_loader, use_cuda=use_cuda, is_test=False)

    return {'model':model, 'train_loss':train_loss, 'train_err':train_err,
            'test_loss':test_loss, 'test_err':test_err}


