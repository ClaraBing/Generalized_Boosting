# data loaders
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import sys
from glob import glob
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from scipy import io

from .binarize_libsvm_data import *
from .preprocess import max_scale, add_bias, normalize_l2
from .simulated_datasets import gen_sim1, gen_sim2, gen_sim3


if sys.version_info[0] == 2:
    import cPickle
else:
    import _pickle as cPickle

DATA_PATH = os.path.join("../datasets/")

def save(url, file):
    os.makedirs(DATA_PATH, exist_ok=True)
    bz2_f = True if os.path.splitext(url)[1] == '.bz2' else False
    print(os.path.splitext(url)[1])

    raw = os.path.basename(url)
    if not os.path.exists(raw):
        os.system('wget %s' % url)

    X_, Y_, xdic, ydic = preprocess(raw, bz2_f=bz2_f);

    X, Y = binarize(X_, Y_, xdic, ydic)
    fout = open(file, 'wb')
    cPickle.dump([X, Y], fout)
    fout.close()

def get_torch_dataset(dataset, transform_type='none', data_id=1, hp_tune=0):
    train_dataset, test_dataset = None, None
    if dataset == 'svhn':
        train_dataset = get_svhn_train_dataset()
        test_dataset = get_svhn_test_dataset()
    elif dataset == 'fashionmnist':
        train_dataset = get_fashionmnist_train_dataset(transform_type)
        test_dataset = get_fashionmnist_test_dataset()
    elif dataset == 'cifar10':
        train_dataset = get_cifar10_train_dataset(transform_type)
        test_dataset = get_cifar10_test_dataset()
    elif dataset == 'mnist_rot':
        train_dataset, test_dataset = get_mnist_rot_datasets()
    elif dataset == 'convex':
        train_dataset, test_dataset = get_convex_datasets()
    elif dataset == 'mnist':
        train_dataset = get_mnist_train_dataset(transform_type)
        test_dataset = get_mnist_test_dataset()
    elif dataset == 'letter':
        train_dataset, test_dataset = get_letter_datasets()
    elif dataset == 'covtype':
        train_dataset, test_dataset = get_covtype_datasets()
    elif dataset == 'connect4':
        train_dataset, test_dataset = get_connect4_datasets()
    elif dataset == 'sim1':
        train_dataset, test_dataset = get_sim1_datasets()
    elif dataset == 'sim2':
        train_dataset, test_dataset = get_sim2_datasets()
    elif dataset == 'sim3':
        train_dataset, test_dataset = get_sim3_datasets()

    return train_dataset, test_dataset


def get_torch_train_val_loaders(dataset, train_batch_size, val_batch_size,
                                transform_type='none', split_ratio = 0.8, data_id=1, subsample=0):
    if dataset in ['cifar10', 'fashionmnist', 'mnist'] and transform_type != 'none':
        if dataset == 'cifar10':
            train_dataset = get_cifar10_train_dataset(transform_type)
            val_dataset = get_cifar10_train_dataset('none')
        elif dataset == 'fashionmnist':
            train_dataset = get_fashionmnist_train_dataset(transform_type)
            val_dataset = get_fashionmnist_train_dataset('none')
        elif dataset == 'cifar100':
            train_dataset = get_cifar100_train_dataset(transform_type)
            val_dataset = get_cifar100_train_dataset('none')
        elif dataset == 'mnist':
            train_dataset = get_mnist_train_dataset(transform_type)
            val_dataset = get_mnist_train_dataset('none')

        num_train = len(train_dataset)
        rand_indices = torch.randperm(num_train)
        split = int(np.floor(split_ratio* num_train))

        train_idx, val_idx = rand_indices[:split], rand_indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, sampler=val_sampler, batch_size=val_batch_size
        )
        return train_loader, val_loader, None, None
    else:
        train_dataset, _ = get_torch_dataset(dataset=dataset, transform_type=transform_type, data_id=data_id)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                   [int(split_ratio*len(train_dataset)),
                                                                    len(train_dataset)-int(split_ratio*len(train_dataset))])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, train_dataset, val_dataset


"""
MNIST:
    * 28x28 grayscale images
    * 60k images for training, 10k for testing
    * 10 classes
    """
def get_mnist_train_dataset(transform_type='none'):

    MNIST_PATH = os.path.join(DATA_PATH, "mnist")
    if transform_type=='none':
        transform = transforms.Compose([transforms.ToTensor()])
    elif transform_type=='crop':
        transform = transforms.Compose(
            [transforms.RandomCrop(size=[28,28], padding=4),
             transforms.ToTensor()])
    elif transform_type=='all':
        transform = transforms.Compose(
            [transforms.RandomCrop(size=[28,28], padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])

    return datasets.MNIST(MNIST_PATH, train=True, download=True,transform= transform)

def get_mnist_test_dataset():

    MNIST_PATH = os.path.join(DATA_PATH, "mnist")
    return datasets.MNIST(MNIST_PATH, train=False, download=True, transform=transforms.ToTensor())


"""
FashionMNIST:
* 28x28 grayscale images
* 60k images for training, 10k for testing
* 10 classes
"""
def get_fashionmnist_train_dataset(transform_type='none'):

    FASHION_MNIST_PATH = os.path.join(DATA_PATH, "fashionmnist")
    if transform_type=='none':
        transform = transforms.Compose([transforms.ToTensor()])
    elif transform_type=='crop':
        transform = transforms.Compose(
                       [transforms.RandomCrop(size=[28,28], padding=4),
                        transforms.ToTensor()])
    elif transform_type=='all':
        transform = transforms.Compose(
                       [transforms.RandomCrop(size=[28,28], padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()])

    return datasets.FashionMNIST(FASHION_MNIST_PATH, train=True, download=True,
                       transform= transform)


def get_fashionmnist_test_dataset():

    FASHION_MNIST_PATH = os.path.join(DATA_PATH, "fashionmnist")
    return datasets.FashionMNIST(FASHION_MNIST_PATH, train=False, download=True,
                       transform=transforms.ToTensor())

"""
CIFAR10:
* 32x32 color images
* 50k for training, 10k for testing
* 10 classes
"""
def get_cifar10_train_dataset(transform_type='none'):
    CIFAR10_PATH = os.path.join(DATA_PATH, "cifar10")

    if transform_type=='none':
        transform = transforms.Compose([transforms.ToTensor()])
    elif transform_type=='crop':
        transform = transforms.Compose(
                       [transforms.RandomCrop(size=[32,32], padding=4),
                        transforms.ToTensor()])
    elif transform_type=='all':
        transform = transforms.Compose(
                       [transforms.RandomCrop(size=[32,32], padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()])
    return datasets.CIFAR10(CIFAR10_PATH, train=True, download=True,
                       transform= transform)

def get_cifar10_test_dataset():
    CIFAR10_PATH = os.path.join(DATA_PATH, "cifar10")

    return datasets.CIFAR10(CIFAR10_PATH, train=False, download=True,
                       transform=transforms.ToTensor())

"""
SVHN:
* 32x32 coloar images
* 73257 digits for training, 26032 digits for testing, 531131 additional less difficult digits
* 10 classes
"""
def get_svhn_train_dataset():
    SVHN_PATH = os.path.join(DATA_PATH, "svhn")
    return datasets.SVHN(SVHN_PATH, split='train', download=True,
                       transform= transforms.Compose(
                        [transforms.ToTensor()]))


def get_svhn_test_dataset():
    SVHN_PATH = os.path.join(DATA_PATH, "svhn")
    return datasets.SVHN(SVHN_PATH, split='test', download=True,
                       transform=transforms.ToTensor())



normalize   = False
bias        = False

"""
LETTER:
* 
"""
def get_letter_datasets():
    train_file = os.path.join(DATA_PATH, "letter.data")
    test_file = os.path.join(DATA_PATH, "letter.t.data")
    split       = False
    standardize = False
    scale       = True

    if not os.path.exists(train_file):
        train_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale'
        save(train_url, train_file)

    if not os.path.exists(test_file):
        test_url  = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t'
        save(test_url, test_file)

    X, Y, Xt, Yt = load(train_file, split, standardize, scale, normalize, bias, test_file_name=test_file)

    X, Y = torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(Y).type(torch.LongTensor)
    Xt, Yt = torch.from_numpy(Xt).type(torch.FloatTensor), torch.from_numpy(Yt).type(torch.LongTensor)

    return torch.utils.data.TensorDataset(X, Y), torch.utils.data.TensorDataset(Xt, Yt)


"""
COVTYPE:
* 
"""
def get_covtype_datasets():
    train_file = os.path.join(DATA_PATH, "covtype.data")
    split       = True
    standardize = True
    scale       = False

    if not os.path.exists(train_file):
        train_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.scale01.bz2'
        save(train_url, train_file)

    X, Y, Xt, Yt = load(train_file, split, standardize, scale, normalize, bias)

    X, Y = torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(Y).type(torch.LongTensor)
    train_data = torch.utils.data.TensorDataset(X, Y)
    Xt, Yt = torch.from_numpy(Xt).type(torch.FloatTensor), torch.from_numpy(Yt).type(torch.LongTensor)
    test_data = torch.utils.data.TensorDataset(Xt, Yt)

    return train_data, test_data



"""
CONVEX:
* 
"""
def get_convex_datasets():
    train_file = os.path.join(DATA_PATH, "convex/convex_train.amat")
    test_file = os.path.join(DATA_PATH, "convex/convex_test.amat")
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    X = train_data[:, :-1] / 1.0
    y = train_data[:, -1:].squeeze()

    Xt = test_data[:, :-1] / 1.0
    yt = test_data[:, -1:].squeeze()

    X = X.reshape(len(X), 1, 28, 28)
    Xt = Xt.reshape(len(Xt), 1, 28, 28)

    X, y = torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(y).type(torch.LongTensor)
    train_data = torch.utils.data.TensorDataset(X, y)
    Xt, yt = torch.from_numpy(Xt).type(torch.FloatTensor), torch.from_numpy(yt).type(torch.LongTensor)
    test_data = torch.utils.data.TensorDataset(Xt, yt)

    return train_data, test_data


"""
MNIST ROTATION + BACKGROUND:
* 
"""


def get_mnist_rot_datasets():
    train_file = os.path.join(DATA_PATH, "mnist_rot/train.amat")
    test_file = os.path.join(DATA_PATH, "mnist_rot/test.amat")
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    X = train_data[:, :-1] / 1.0
    y = train_data[:, -1:].squeeze()

    Xt = test_data[:, :-1] / 1.0
    yt = test_data[:, -1:].squeeze()

    X = X.reshape(len(X), 1, 28, 28)
    Xt = Xt.reshape(len(Xt), 1, 28, 28)

    X, y = torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(y).type(torch.LongTensor)
    train_data = torch.utils.data.TensorDataset(X, y)
    Xt, yt = torch.from_numpy(Xt).type(torch.FloatTensor), torch.from_numpy(yt).type(torch.LongTensor)
    test_data = torch.utils.data.TensorDataset(Xt, yt)

    return train_data, test_data

"""
CONNECT 4
"""
def get_connect4_datasets():
    data_file = os.path.join(DATA_PATH, "connect4/connect.data")
    data = pd.read_csv(data_file, header=None)

    ncols = data.values.shape[1]
    labels = data[ncols-1].values
    Y = np.zeros(labels.shape)
    Y[np.where(labels == 'win')] = 2
    Y[np.where(labels == 'draw')] = 1

    X = pd.get_dummies(data[list(range(ncols-1))]).values

    # split into train and test
    indices = list(range(X.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)

    train_ratio = 0.8
    train_size = int(train_ratio * X.shape[0])
    train_mask = indices[:train_size]
    test_mask = indices[train_size:]
    Xt = X[test_mask]
    Yt = Y[test_mask]
    X = X[train_mask]
    Y = Y[train_mask]

    X, y = torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(Y).type(torch.LongTensor)
    train_data = torch.utils.data.TensorDataset(X, y)
    Xt, yt = torch.from_numpy(Xt).type(torch.FloatTensor), torch.from_numpy(Yt).type(torch.LongTensor)
    test_data = torch.utils.data.TensorDataset(Xt, yt)

    return train_data, test_data


"""
concentric spheres dataset
"""
def get_sim1_datasets():
    fdata = os.path.join(DATA_PATH, 'sim_concentric.pt')
    if not os.path.exists(fdata):
      gen_sim1()
    data = torch.load(fdata)
    train_data = torch.utils.data.TensorDataset(data[0], data[2])
    test_data = torch.utils.data.TensorDataset(data[1], data[3])
    return train_data, test_data

"""
polynomial dataset
"""
def get_sim2_datasets():
    fdata = os.path.join(DATA_PATH, 'sim_deep1.pt')
    if not os.path.exists(fdata):
      gen_sim2()
    data = torch.load(fdata)
    train_data = torch.utils.data.TensorDataset(data[0], data[2])
    test_data = torch.utils.data.TensorDataset(data[1], data[3])
    return train_data, test_data

"""
polynomial dataset
"""
def get_sim3_datasets():
    fdata = os.path.join(DATA_PATH, 'sim_deep3.pt')
    if not os.path.exists(fdata):
      gen_sim3()
    data = torch.load(fdata)
    train_data = torch.utils.data.TensorDataset(data[0], data[2])
    test_data = torch.utils.data.TensorDataset(data[1], data[3])
    return train_data, test_data



def load(file_name, split, standardize, scale, normalize, bias,
         train_ratio=0.8, test_file_name=None):
    fin = open(file_name, u'rb')
    [X, Y] = cPickle.load(fin)
    fin.close()

    if split: # train test splits
        train_size = int(train_ratio * X.shape[0])

        # always use the  same test split
        indices = list(range(X.shape[0]))
        np.random.seed(1)
        np.random.shuffle(indices)

        train_mask = indices[:train_size]
        test_mask = indices[train_size:]
        Xt = X[test_mask]
        Yt = Y[test_mask]
        X = X[train_mask]
        Y = Y[train_mask]
    elif test_file_name is not None:
        fin = open(test_file_name, u'rb')
        [Xt, Yt] = cPickle.load(fin)
        fin.close()
    else:
        Xt = None
        Yt = None
    
    if standardize:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        if Xt is not None:
            Xt = scaler.transform(Xt)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(np.vstack((X,Xt)))
        X = scaler.transform(X)
        if Xt is not None:
            Xt = scaler.transform(Xt)

    if bias:
        X = add_bias(X)
        if Xt is not None:
            Xt = add_bias(Xt)

    if normalize:
        X = normalize_l2(X)
        if Xt is not None:
            Xt = normalize_l2(Xt)

    if Xt is not None:
        return X.astype(np.float32), Y.astype(np.int32), \
               Xt.astype(np.float32), Yt.astype(np.int32)
    else:
        return X.astype(np.float32), Y.astype(np.int32), None, None



