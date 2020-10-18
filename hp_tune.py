'''
hyper-parameter tuning for greedy training of neural networks.
'''

from __future__ import print_function
import argparse, yaml
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import pickle as pk
import os

from data.data_loader import get_torch_dataset, get_torch_train_val_loaders
from data.data_utils import get_num_classes
from models.shallow_learners import *
from models import AdditiveFeatureBoost, CompositionBoostStd, CompositionBoostDense, \
    CompositionBoostCmplx, AdaBoost, End2End

# Training settings
parser = argparse.ArgumentParser(description='GREEDY TRAINING')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 400)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', nargs='+', type=float, default=[0.005], metavar='LR',
                    help='learning rate (default: [0.005])')
parser.add_argument('--greedy-lr', type=float, default=1, metavar='GLR',
                    help='learning rate of greedy algorithm (default: 1)')
parser.add_argument('--min-lr', type=float, default=0,
                    help='minimum lr, below which training is stopped.')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--iterations', type=int, default=10, metavar='D',
                    help='number of greedy iterations (default: 10)')
parser.add_argument('--width', nargs='+', type=int, default=[512], metavar='D',
                    help='Width of each shallow learner (default: [512])')
parser.add_argument('--width-increment', type=int, default=0, metavar='D',
                    help='with increment of shallow learners '
                         '(useful only for cmplxCompBoost and dcmplxCompBoost; default: 0)')
parser.add_argument('--dataset', type=str, default='cifar10',metavar='DATASET',
                    choices=['cifar10', 'svhn', 'fashionmnist', 'convex', 'mnist_rot', # image datasets
                             'mnist', 'letter', 'covtype',  'connect4', # tabular datasets
                             'sim1', 'sim2', 'sim3'], # simulated datasets
                    help='Dataset to use. (default: "cifar10")')
parser.add_argument('--subsample', type=float, default=0,
                    help='Portion of the dataset on which the model is trained.')
parser.add_argument('--train_val_split_ratio', type=float, default=0.8,
                    help='Portion of the train dataset to use for validation.')
parser.add_argument('--weight-decay', nargs='+', type=float, default=[0.000], metavar='WD',
                    help='weight decay (default: [0])')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['Adam', 'SGD'])
parser.add_argument('--scheduler-tolerance', type=int, default=5000, metavar='ST',
                    help='Tolerance of scheduler (default: 5000)')
parser.add_argument('--transform', type=str, default='none', choices=['none', 'all', 'crop'])
parser.add_argument('--basic-block', type=str, default='fc', choices=['fc', 'conv', 'conv_small'])
parser.add_argument('--algorithm', type=str, default='stdCompBoost',
                    choices=['featureBoost', 'adaBoost', 'stdCompBoost', 'cmplxCompBoost',
                             'denseCompBoost', 'joint'],
                    help='Technique to use. (default: "stdCompBoost")')

# Saving & logging
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for "pth", i.e. the model saving directory.')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

i = 0
output_dir = 'output/dnn/'
os.makedirs(output_dir, exist_ok=True)
while os.path.exists(os.path.join(output_dir, '{}_{}_{}_{}'.format(args.algorithm, args.basic_block, i, args.dataset))):
    i +=1
pth = os.path.join(output_dir,'{}_{}_{}_{}'.format(args.algorithm, args.basic_block, i, args.dataset))
if args.suffix:
  pth += '_' + args.suffix
os.makedirs(pth, exist_ok=True)

args_file_path = os.path.join(pth, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

train_loader, val_loader, train_dataset, val_dataset = get_torch_train_val_loaders(dataset=args.dataset, train_batch_size=args.batch_size,
                            val_batch_size=args.test_batch_size, transform_type=args.transform,
                            split_ratio = args.train_val_split_ratio)

num_classes = get_num_classes(args.dataset)
dummy_features, _ = next(iter(train_loader))
input_size = dummy_features[0].size()

if args.basic_block == 'fc':
    BasicBlock = FullyConnectedBlock
    BasicClfBlock = FullyConnectedClfBlock
elif args.basic_block == 'conv':
    BasicBlock = ConvBlock
    BasicClfBlock = ConvClfBlock
elif args.basic_block == 'conv_small':
    BasicBlock = ConvBlockSmall
    BasicClfBlock = ConvClfBlock

train_err = {}; train_loss = {}
val_err = {}; val_loss = {}


for lr in args.lr:
    for width in args.width:
        if args.basic_block == 'fc':
            width_increment = width // 5
        else:
            width_increment = width // 8
        for weight_decay in args.weight_decay:

            if args.algorithm == 'joint':
                output = End2End.train(train_loader=train_loader, test_loader=val_loader,
                                       BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                       opt=args.optimizer, epochs=args.epochs,
                                       scheduler_tolerance=args.scheduler_tolerance,
                                       lr=lr, momentum=args.momentum, weight_decay=weight_decay, min_lr=args.min_lr,
                                       depth=args.iterations, width=width,
                                       use_cuda=use_cuda, num_classes=num_classes, input_size=input_size)
            elif args.algorithm == 'featureBoost':
                output = AdditiveFeatureBoost.train(train_loader=train_loader, test_loader=val_loader,
                                                    BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                                    opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                                    scheduler_tolerance=args.scheduler_tolerance,
                                                    lr=lr, momentum=args.momentum, weight_decay=weight_decay, min_lr=args.min_lr,
                                                    iterations=args.iterations, width=width,
                                                    use_cuda=use_cuda, num_classes=num_classes, input_size=input_size)
            elif args.algorithm == 'adaBoost':
                output = AdaBoost.train(train_loader=train_loader, test_loader=val_loader,
                                                    train_dataset=train_dataset,
                                                    BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                                    opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                                    scheduler_tolerance=args.scheduler_tolerance,
                                                    lr=lr, momentum = args.momentum, weight_decay=weight_decay, min_lr=args.min_lr,
                                                    iterations = args.iterations, width=width,
                                                    use_cuda=use_cuda, num_classes = num_classes, input_size = input_size)

            elif args.algorithm == 'stdCompBoost':
                output = CompositionBoostStd.train(train_loader=train_loader, test_loader=val_loader,
                                                   BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                                   opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                                   scheduler_tolerance=args.scheduler_tolerance,
                                                   lr=lr, momentum=args.momentum, weight_decay=weight_decay, min_lr=args.min_lr,
                                                   iterations=args.iterations, width=width,
                                                   use_cuda=use_cuda, num_classes=num_classes, input_size=input_size)
            elif args.algorithm == 'denseCompBoost':
                output = CompositionBoostDense.train(train_loader=train_loader, test_loader=val_loader,
                                                     BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                                     opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                                     scheduler_tolerance=args.scheduler_tolerance,
                                                     lr=lr, momentum=args.momentum, weight_decay=weight_decay, min_lr=args.min_lr,
                                                     iterations=args.iterations, width=width,
                                                     use_cuda=use_cuda, num_classes=num_classes, input_size=input_size)
            elif args.algorithm == 'cmplxCompBoost':
                output = CompositionBoostCmplx.train(train_loader=train_loader, test_loader=val_loader,
                                                     BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                                     opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                                     scheduler_tolerance=args.scheduler_tolerance,
                                                     width_increment=width_increment,
                                                     lr=lr, momentum=args.momentum, weight_decay=weight_decay, min_lr=args.min_lr,
                                                     iterations=args.iterations, width=width,
                                                     use_cuda=use_cuda, num_classes=num_classes, input_size=input_size)

            unique_id = 'lr{:.4f}_width{}_wd{:.4f}'.format(lr, width, weight_decay)
            train_err[unique_id] = output['train_err']
            train_loss[unique_id] = output['train_loss']
            val_err[unique_id] = output['test_err']
            val_loss[unique_id] = output['test_loss']

            print('lr: {:.4f}, width: {}, weight decay: {:.4f}'.format(lr, width, weight_decay))
            print(train_err[unique_id])
            print(val_err[unique_id])

with open(os.path.join(pth, 'stats.pk'), 'wb') as f:
    pk.dump([train_err, train_loss, val_err, val_loss], f)


