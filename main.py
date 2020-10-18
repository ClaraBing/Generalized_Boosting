'''
Main file for greedy training of neural networks.
All the following greedy techniques can be called from this file
     1) featureBoost:  additive boosting, where we fit
     models of the form f1(x)+f2(x).... in a greedy fashion
     2) stdCompBoost: standard composition boosting approach of Bengio et al.
     Here we learn models of the form  ft o .... f3 o f2 o f1(x) in a greedy fashion
     3) cmplxCompBoost: composition boosting with increasing complexity of shallow learners
     4) denseCompBoost: dense composition boosting
All these techniques use ''exact greedy update'' rule
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
parser.add_argument('--hp-tune', type=int, default=0, choices=[0,1],
                    help="Whether performing hyperparameter tuning.' \
                    +'Set 0 to use the entire train set;' \
                    +'if set to 1, then the train set will be split into train & val.")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 400)')
parser.add_argument('--subsample', type=float, default=0,
                    help="ratio for subsampling training data.")
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--greedy-lr', type=float, default=1, metavar='GLR',
                    help='learning rate of greedy algorithm (default: 1)')
parser.add_argument('--min-lr', type=float, default=0,
                    help="minimum lr, below which training is stopped.")
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--iterations', type=int, default=30, metavar='D',
                    help='number of greedy iterations (default: 30)')
parser.add_argument('--width', type=int, default=512, metavar='D',
                    help='Width of each shallow learner (default: 512)')
parser.add_argument('--width-increment', type=int, default=0, metavar='D',
                    help='with increment of shallow learners '
                         '(useful only for cmplxCompBoost; default: 0)')
parser.add_argument('--widening-factor', type=int, default=0,
                    help='Factor of increase in width at each layer. Will overwrite width_increment.')
parser.add_argument('--dataset', type=str, default='sim3', metavar='DATASET',
                    choices=['cifar10', 'svhn', 'fashionmnist', 'convex', 'mnist_rot', # image datasets
                             'mnist', 'letter', 'covtype',  'connect4', # tabular datasets
                             'sim1', 'sim2', 'sim3'], # simulated datasets
                    help='Dataset to use. (default: "sim3")')
parser.add_argument('--data-id', type=int, default=0,
                    help="ID for (sub)dataset. Required for the molecular dataset.")
parser.add_argument('--weight-decay', type=float, default=0.000, metavar='WD',
                    help='weight decay (default: 0)')
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

if args.basic_block == 'fc':
  args.width_increment = int(args.width // 5)
else:
  args.width_increment = int(args.width // 8)

# data
if args.hp_tune:
  train_loader, test_loader, train_dataset, val_dataset = get_torch_train_val_loaders(args.dataset,
                     train_batch_size=args.batch_size, val_batch_size=args.test_batch_size,
                     transform_type=args.transform, data_id = args.data_id, subsample=args.subsample)
else:
  train_dataset, test_dataset = get_torch_dataset(dataset=args.dataset, transform_type=args.transform, data_id=args.data_id)
  if args.subsample > 0:
    assert(args.subsample <= 1)
    # train on a (fixed) subset of data.
    k = int(args.subsample * len(train_dataset))
    sub_indices = np.random.choice(len(train_dataset), k, replace=False).tolist()
    sampler = torch.utils.data.SubsetRandomSampler(sub_indices)
    shuffle = False # sampler is mutually exclusive with shuffle
  else:
    sampler = None
    shuffle = True
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

num_classes = get_num_classes(args.dataset)
dummy_features, _ = next(iter(train_loader)) 
input_size = dummy_features[0].size()

# checkpoint
i = 0
output_dir = 'output/'
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

# model
if args.basic_block == 'fc':
    BasicBlock = FullyConnectedBlock
    BasicClfBlock = FullyConnectedClfBlock
elif args.basic_block == 'conv':
    BasicBlock = ConvBlock
    BasicClfBlock = ConvClfBlock
elif args.basic_block == 'conv_small':
    BasicBlock = ConvBlockSmall
    BasicClfBlock = ConvClfBlock

if args.algorithm == 'joint':
    output = End2End.train(train_loader=train_loader, test_loader=test_loader,
                                        BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                        opt=args.optimizer, epochs=args.epochs,
                                        scheduler_tolerance=args.scheduler_tolerance,
                                        lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay, min_lr=args.min_lr,
                                        depth = args.iterations, width=args.width,
                                        use_cuda=use_cuda, num_classes = num_classes, input_size = input_size)
elif args.algorithm == 'featureBoost':
    output = AdditiveFeatureBoost.train(train_loader=train_loader, test_loader=test_loader,
                                        BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                        opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                        scheduler_tolerance=args.scheduler_tolerance,
                                        lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay, min_lr=args.min_lr,
                                        iterations = args.iterations, width=args.width,
                                        use_cuda=use_cuda, num_classes = num_classes, input_size = input_size)
elif args.algorithm == 'adaBoost':
    output = AdaBoost.train(train_loader=train_loader, test_loader=test_loader,
                                        train_dataset=train_dataset,
                                        BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                        opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                        scheduler_tolerance=args.scheduler_tolerance,
                                        lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay, min_lr=args.min_lr,
                                        iterations = args.iterations, width=args.width,
                                        use_cuda=use_cuda, num_classes = num_classes, input_size = input_size)
elif args.algorithm == 'stdCompBoost':
    output = CompositionBoostStd.train(train_loader=train_loader, test_loader=test_loader,
                                        BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                        opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                        scheduler_tolerance=args.scheduler_tolerance,
                                        lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay, min_lr=args.min_lr,
                                        iterations = args.iterations, width=args.width,
                                        use_cuda=use_cuda, num_classes = num_classes, input_size = input_size)
elif args.algorithm == 'denseCompBoost':
    output = CompositionBoostDense.train(train_loader=train_loader, test_loader=test_loader,
                                        BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                        opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                        scheduler_tolerance=args.scheduler_tolerance,
                                        lr=args.lr, momentum = args.momentum, weight_decay=args.weight_decay, min_lr=args.min_lr,
                                        iterations = args.iterations, width=args.width,
                                        use_cuda=use_cuda, num_classes = num_classes, input_size = input_size)
elif args.algorithm == 'cmplxCompBoost':
    output = CompositionBoostCmplx.train(train_loader=train_loader, test_loader=test_loader,
                                         BasicBlock=BasicBlock, BasicClfBlock=BasicClfBlock,
                                         opt=args.optimizer, epochs=args.epochs, greedy_lr=args.greedy_lr,
                                         scheduler_tolerance=args.scheduler_tolerance, width_increment=args.width_increment, min_lr=args.min_lr,
                                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                         iterations=args.iterations, width=args.width,
                                         use_cuda=use_cuda, num_classes=num_classes, input_size=input_size)


with open(os.path.join(pth, 'stats.pk'), 'wb') as f:
    pk.dump([output['test_err'], output['test_loss'], output['train_err'], output['train_loss']], f)

torch.save(output['model'].state_dict(), os.path.join(pth, 'model.pk'))

