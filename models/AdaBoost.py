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
                 num_classes=10, input_size = (3,32,32), iterations = 8, width = 128, lr=1.0, use_cuda=True):
        super(Ensemble, self).__init__()

        # add layers to dictionary
        self.standby = {0: BasicBlock(input_size, width, residual = False,scale=lr)}
        for i in range(1,iterations):
            self.standby[i] = BasicBlock(input_size, width, residual = False,scale=lr)

        # add classification layers
        self.standby_cf = {0: BasicClfBlock(self.standby[0].sizeof_output(), num_classes)}
        for i in range(1,iterations): self.standby_cf[i] = BasicClfBlock(self.standby[i].sizeof_output(), num_classes)

        # weight of each classifier
        self.booster_weights = None
        if use_cuda:
            self.booster_weights = Variable(torch.zeros(iterations).cuda(), requires_grad=False)
        else:
            self.booster_weights = Variable(torch.zeros(iterations), requires_grad=False)

        self.training_c = None
        self.training_cf = None

        # store some additional data
        self.booster_idx = 0
        self.num_classes = num_classes

    def forward(self, x):
        out = self.training_c(x)
        return self.training_cf(out)

    def get_sample_weights(self, x, y):
        output = torch.zeros(y.shape, device = y.device)

        for i in range(self.booster_idx):
            preds = self.standby_cf[i](self.standby[i](x))
            preds = preds.max(1, keepdim=True)[1]
            preds = y.eq(preds.view_as(y))
            output += self.booster_weights[i]*(1-preds.type(output.type()))

        return torch.exp(output)

    def get_predictions(self, x, idx):
        output = torch.zeros((x.shape[0],self.num_classes), device = x.device)

        for i in range(idx):
            preds = self.standby_cf[i](self.standby[i](x))
            preds = preds.max(1, keepdim=True)[1]
            preds_matrix = torch.zeros((x.shape[0],self.num_classes), device = x.device)
            output += self.booster_weights[i] * preds_matrix.scatter_(1, preds, 1)

        return output.max(1, keepdim=True)[1]

    def set_booster_weight(self, weight):
        self.booster_weights[self.booster_idx] = weight

    def add_layer(self, idx):
        if self.training_c:
            self.training_c.requires_grad = False
        if self.training_cf:
            self.training_cf.requires_grad = False
        try:
            self.booster_idx = idx
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

def set_booster_weight(model, data_loader, use_cuda, num_classes):
    model.eval()
    err_num,err_den = 0.0,0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        weights = model.get_sample_weights(data, target)
        with torch.no_grad():
            output = model(data)
            preds = output.max(1, keepdim=True)[1]
            preds = target.eq(preds.view_as(target))
            err_num += torch.sum(weights * (1-preds.type(output.type())))
            err_den += torch.sum(weights)

    err = err_num/err_den
    err = err.cpu().numpy()
    weight = np.log(1.0-err) - np.log(err)  + np.log(num_classes-1)
    model.set_booster_weight(weight)

def get_train_loader(model, data_loader, dataset, use_cuda):
    dummy_features, _ = next(iter(data_loader))
    batch_size = len(dummy_features)
    index = 0
    weights = np.zeros(len(dataset))
    while index <len(dataset):
        ub = min(len(dataset), index+batch_size)
        data_stacked = None
        target_stacked = None
        for i in range(index, ub):
            data = dataset[i]
            target = data[1]
            data = data[0].unsqueeze(0)
            if type(target) is int:
                target = torch.tensor(target)

            if len(target.shape) == 0:
                target = target.unsqueeze(dim = 0)
            if data_stacked is None:
                data_stacked = data
                target_stacked = target
            else:
                data_stacked = torch.cat([data_stacked, data], dim = 0)
                target_stacked = torch.cat([target_stacked, target])

        data, target = data_stacked.type(torch.FloatTensor), target_stacked.type(torch.LongTensor)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            curr_weights = model.get_sample_weights(data, target)
        weights[index:ub] = curr_weights.cpu().numpy()
        index = ub

    sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)


def train(train_loader, test_loader, train_dataset,
          BasicBlock, BasicClfBlock,
          opt='SGD', epochs=20, min_lr=0.0,
          lr=0.005, momentum = 0.5, weight_decay=0.0,
          iterations = 30, width=128,greedy_lr=1., scheduler_tolerance=5000,
          use_cuda=True, num_classes = 10, input_size = [3,32,32]):

    model = Ensemble(BasicBlock=BasicBlock, BasicClfBlock = BasicClfBlock, num_classes=num_classes,
                     input_size=input_size, iterations=iterations, width = width, lr=greedy_lr, use_cuda=use_cuda)

    train_loss = np.zeros(iterations)
    test_loss = np.zeros(iterations)
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

        curr_train_loader = get_train_loader(model=model, data_loader = train_loader, dataset= train_dataset, use_cuda = use_cuda)
        for epoch in range(0, epochs):
            train_helper(model=model, optimizer=optimizer, scheduler=scheduler,
                         epoch=epoch, data_loader=curr_train_loader, use_cuda=use_cuda)

        set_booster_weight(model=model, data_loader=train_loader, use_cuda=use_cuda, num_classes=num_classes)

        test_err[i] = test_helper_adaBoost(model=model, iteration=i, data_loader=test_loader, use_cuda=use_cuda, is_test=True)
        train_err[i] = test_helper_adaBoost(model=model, iteration=i, data_loader=train_loader, use_cuda=use_cuda, is_test=False)

    return {'model':model, 'train_loss':train_loss, 'train_err':train_err,
            'test_loss':test_loss, 'test_err':test_err}







