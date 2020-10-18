from __future__ import print_function
import math
import argparse, yaml
import numpy as np
import pickle as pk
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DATA_PATH = '../datasets'
os.makedirs(DATA_PATH, exist_ok=True)

n_tr = 1000000
n_te = 500000

def gen_sim1():
  # concentric ellipsoids
  d = 32
  Sigma = torch.randn(d, d) / d ** 0.5
  Sigma = torch.mm(Sigma, Sigma.t())
  Sigma.add_(0.1 * torch.eye(d))
  
  x_tr1 = torch.randn(int(n_tr / 4), d).type(torch.FloatTensor)
  x_tr2 = 1.5 * torch.randn(int(n_tr / 4), d).type(torch.FloatTensor)
  x_tr3 = 3 * torch.randn(int(n_tr / 4), d).type(torch.FloatTensor)
  x_tr4 = 4.5 * torch.randn(int(n_tr / 4), d).type(torch.FloatTensor)
  x_tr = torch.cat((x_tr1, x_tr2, x_tr3, x_tr4), dim=0)
  x_tr = torch.mm(x_tr, Sigma)
  
  x_te1 = torch.randn(int(n_te / 4), d).type(torch.FloatTensor)
  x_te2 = 1.5 * torch.randn(int(n_te / 4), d).type(torch.FloatTensor)
  x_te3 = 3 * torch.randn(int(n_te / 4), d).type(torch.FloatTensor)
  x_te4 = 4.5 * torch.randn(int(n_te / 4), d).type(torch.FloatTensor)
  x_te = torch.cat((x_te1, x_te2, x_te3, x_te4), dim=0)
  x_te = torch.mm(x_te, Sigma)
  
  y_tr1 = torch.zeros(int(n_tr / 4)).type(torch.LongTensor)
  y_tr2 = torch.ones(int(n_tr / 4)).type(torch.LongTensor)
  y_tr3 = torch.zeros(int(n_tr / 4)).type(torch.LongTensor)
  y_tr4 = torch.ones(int(n_tr / 4)).type(torch.LongTensor)
  y_tr = torch.cat((y_tr1, y_tr2, y_tr3, y_tr4), dim=0)
  
  y_te1 = torch.zeros(int(n_te / 4)).type(torch.LongTensor)
  y_te2 = torch.ones(int(n_te / 4)).type(torch.LongTensor)
  y_te3 = torch.zeros(int(n_te / 4)).type(torch.LongTensor)
  y_te4 = torch.ones(int(n_te / 4)).type(torch.LongTensor)
  y_te = torch.cat((y_te1, y_te2, y_te3, y_te4), dim=0)
  
  torch.save([x_tr, x_te, y_tr, y_te], os.path.join(DATA_PATH, 'sim_concentric.pt'))


def gen_sim2():
  d = 32
  x_tr = torch.randn(n_tr, d).type(torch.FloatTensor)
  x_te = torch.randn(n_te, d).type(torch.FloatTensor)
  tmp1 = torch.zeros(n_tr, d // 4).type(torch.FloatTensor)
  tmp3 = torch.zeros(n_te, d // 4).type(torch.FloatTensor)
  for i in range(d // 4):
      tmp1[:, i] = x_tr[:, 4 * i:4 * i + 1].sum(dim=1) * x_tr[:, 4 * i + 2:4 * i + 3].sum(dim=1)
      tmp3[:, i] = x_te[:, 4 * i:4 * i + 1].sum(dim=1) * x_te[:, 4 * i + 2:4 * i + 3].sum(dim=1)
  
  tmp2 = torch.zeros(n_tr, d // 16).type(torch.FloatTensor)
  tmp4 = torch.zeros(n_te, d // 16).type(torch.FloatTensor)
  for i in range(d // 16):
      tmp2[:, i] = tmp1[:, 4 * i:4 * i + 1].sum(dim=1) * tmp1[:, 4 * i + 2:4 * i + 3].sum(dim=1)
      tmp4[:, i] = tmp3[:, 4 * i:4 * i + 1].sum(dim=1) * tmp3[:, 4 * i + 2:4 * i + 3].sum(dim=1)
  
  tmp2 = tmp2[:, 0] - tmp2[:, 1]
  tmp4 = tmp4[:, 0] - tmp4[:, 1]
  y_tr = (tmp2 >= 0).squeeze().type(torch.LongTensor)
  y_te = (tmp4 >= 0).squeeze().type(torch.LongTensor)
  torch.save([x_tr, x_te, y_tr, y_te], os.path.join(DATA_PATH, 'sim_deep1.pt'))


def gen_sim3():
  d = 32
  x_tr = torch.randn(n_tr, d).type(torch.FloatTensor)
  x_te = torch.randn(n_te, d).type(torch.FloatTensor)
  tmp1 = x_tr[:, :5].sum(dim=1)
  tmp2 = x_tr[:, 5:8].sum(dim=1)
  tmp3 = x_tr[:, 8:11].sum(dim=1)
  tmp4 = x_tr[:, 11:15].sum(dim=1)
  tmp5 = x_tr[:, 15:25].sum(dim=1)
  tmp6 = x_tr[:, 25:].sum(dim=1)
  
  y_tr = (tmp1 * tmp2 * tmp3 * tmp4 * tmp5 * tmp6 >= 0).squeeze().type(torch.LongTensor)
  
  tmp1 = x_te[:, :5].sum(dim=1)
  tmp2 = x_te[:, 5:8].sum(dim=1)
  tmp3 = x_te[:, 8:11].sum(dim=1)
  tmp4 = x_te[:, 11:].sum(dim=1)
  tmp5 = x_te[:, 15:25].sum(dim=1)
  tmp6 = x_te[:, 25:].sum(dim=1)
  y_te = (tmp1 * tmp2 * tmp3 * tmp4 * tmp5 * tmp6 >= 0).squeeze().type(torch.LongTensor)
  
  torch.save([x_tr, x_te, y_tr, y_te], os.path.join(DATA_PATH, 'sim_deep3.pt'))

