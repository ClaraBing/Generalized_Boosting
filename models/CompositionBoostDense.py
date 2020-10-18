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
                 num_classes=10, input_size = (3,32,32), iterations = 8, width = 128, lr=1., use_cuda = True):
        super(Ensemble, self).__init__()

        # add layers to dictionary
        self.standby = {0: BasicBlock(input_size, width, residual = False,scale=lr)}
        self.standby_init_c = {0: BasicBlock(input_size, width, residual=False, scale=lr)}
        for i in range(1,iterations):
            self.standby[i] = BasicBlock(self.standby[i-1].sizeof_output(), width=width, residual = False, scale=lr)
            self.standby_init_c[i] = BasicBlock(input_size, width=width, residual=False, scale=lr)
            if i%6==0:
                lr /= 2.0

        # add classification layers
        self.standby_cf = {0: BasicClfBlock(self.standby[0].sizeof_output(), num_classes)}
        for i in range(1,iterations): self.standby_cf[i] = BasicClfBlock(self.standby[i].sizeof_output(), num_classes)

        self.gates = {}
        for idx in range(1,iterations):
            if use_cuda:
                self.gates[idx] = Variable(torch.zeros(idx).cuda(), requires_grad=False)
            else:
                self.gates[idx] = Variable(torch.zeros(idx+1), requires_grad=False)
            self.gates[idx][-1:] = 1

        # copy previous layers weights
        self.init_previous = False

        self.frozen_c = []
        self.frozen_init_c = []

        self.training_c = None
        self.training_cf = None
        self.training_gates = None
        self.training_init_c = None

    def forward(self, x):
        layer_idx = 0
        gs = {}
        for c in self.frozen_c:
            if layer_idx == 0:
                gs[layer_idx] = c(x)
            else:
                curr_gates = self.gates[layer_idx]
                g_sum = curr_gates[layer_idx-1]*gs[layer_idx-1]
                for idx in range(layer_idx-1):
                    g_sum  = g_sum + curr_gates[idx]*gs[idx]
                gs[layer_idx] = gs[layer_idx-1] + c(g_sum) + self.frozen_init_c[layer_idx-1](x)
            layer_idx += 1

        if layer_idx == 0:
            x = self.training_c(x)
        else:
            curr_gates = self.training_gates
            g_sum = curr_gates[layer_idx-1]*gs[layer_idx-1]
            for idx in range(layer_idx-1):
                g_sum = g_sum + curr_gates[idx]*gs[idx]
            x = gs[layer_idx-1] + self.training_c(g_sum) + self.training_init_c(x)

        x = self.training_cf(x)

        return x

    def add_layer(self, idx):
        if self.training_c:
            self.training_c.requires_grad = False
            self.frozen_c.append(self.training_c)
        if self.training_cf:
            self.training_cf.requires_grad = False
        if idx > 1:
            self.training_gates.requires_grad = False
            self.training_init_c.requires_grad = False
            self.frozen_init_c.append(self.training_init_c)
        self.training_c = self.standby[idx]
        self.training_cf = self.standby_cf[idx]

        if idx > 0:
            self.training_gates = self.gates[idx]
            self.training_gates.requires_grad=True
            self.training_init_c = self.standby_init_c[idx]

            if self.init_previous and idx > 4:
                self.training_c.load_state_dict(self.standby[idx - 1].state_dict())
                self.training_cf.load_state_dict(self.standby_cf[idx - 1].state_dict())
                self.training_init_c.load_state_dict(self.standby_init_c[idx - 1].state_dict())

            trainable_params = [{'params': self.training_c.parameters()},
                                {'params': self.training_cf.parameters()},
                                {'params': self.training_gates, 'weight_decay': 0},
                                {'params': self.training_init_c.parameters()}
                                ]
        else:
            trainable_params = [{'params': self.training_c.parameters()},
                                {'params': self.training_cf.parameters()}
                                ]
        return trainable_params


def train(train_loader, test_loader, BasicBlock, BasicClfBlock,
          opt='SGD', epochs=20,
          lr=0.005, momentum = 0.5, weight_decay=0.0, min_lr=0,
          iterations = 30, width=128,greedy_lr=1., scheduler_tolerance=5000,
          use_cuda=True, num_classes = 10, input_size = [3,32,32]):

    model = Ensemble(BasicBlock=BasicBlock, BasicClfBlock = BasicClfBlock, num_classes=num_classes,
                     input_size=input_size, iterations=iterations, width = width, lr=greedy_lr, use_cuda=use_cuda)

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


