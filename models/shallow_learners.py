import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import torch.nn.functional as F

import numpy as np

# helper classes
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

# Basic block for fully connected networks
class FullyConnectedBlock(nn.Module):

    def __init__(self, in_size, width, residual=True, scale=1.0, use_relu = True):
        super(FullyConnectedBlock, self).__init__()
        self.width = width
        self.fc = nn.Linear(np.prod(in_size), width)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Sigmoid()
        self.residual = residual

        self.upsample = Lambda(lambda x: x)
        if width > np.prod(in_size):
            self.upsample = nn.Sequential(LambdaReduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                         Lambda(lambda x: x),
                         Lambda(lambda x: torch.zeros((x.size(0), width-np.prod(in_size)), device=x.device))
                         )
            )
        elif width < np.prod(in_size):
            self.upsample = Lambda(lambda x: x[:, :width])

        self.scale = scale

        for m in self.modules():
            weights_init(m)

    def sizeof_output(self):
        return [self.width]

    def get_norms(self):
        out = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                out.append(torch.norm(m.weight))
        return out

    @staticmethod
    def upsample_fn(input, out_width):
        in_size = input.shape
        upsampler = Lambda(lambda x: x)
        if out_width > np.prod(in_size):
            upsampler = nn.Sequential(LambdaReduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                         Lambda(lambda x: x),
                         Lambda(lambda x: torch.zeros((x.size(0), out_width-np.prod(in_size)), device=x.device))
                         )
            )
        elif out_width < np.prod(in_size):
            upsampler = Lambda(lambda x: x[:, :out_width])

        return upsampler(input)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        out = self.scale*self.relu(out)
        if not self.residual:
            return out

        out += self.upsample(x)

        return out

class FullyConnectedClfBlock(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnectedClfBlock, self).__init__()
        self.layer = nn.Linear(np.prod(input_size),num_classes)

    def forward(self, x):
        return self.layer(x)

# Basic block for conv networks
class ConvBlock(nn.Module):

    def __init__(self, in_size, width, residual=True, scale=1.0, use_relu = True):
        super(ConvBlock, self).__init__()
        self.kernel_size = (3,3)
        self.stride = (1,1)
        self.in_size = in_size
        self.width = width
        if use_relu:
            self.layer = nn.Sequential(  # Sequential,
                nn.Conv2d(in_size[0], width, self.kernel_size, self.stride, (1, 1), 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, width, self.kernel_size, self.stride, (1, 1), 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(  # Sequential,
                nn.Conv2d(in_size[0], width, self.kernel_size, self.stride, (1, 1), 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.Sigmoid(),
                nn.Conv2d(width, width, self.kernel_size, self.stride, (1, 1), 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.Sigmoid()
            )


        self.residual = residual
        self.upsample = Lambda(lambda x: x)
        if width > in_size[0]:
            self.upsample = nn.Sequential(LambdaReduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                         Lambda(lambda x: x),
                         Lambda(lambda x: torch.zeros((x.size(0), width-in_size[0], x.size(2), x.size(3)),
                                                      device=x.device))
                         )
            )
        elif width < in_size[0]:
            self.upsample = Lambda(lambda x: x[:, :width, :, :])

        self.scale = scale


        for m in self.children():
            for m2 in m.modules():
                weights_init(m2)

    @staticmethod
    def upsample_fn(input, out_width):
        in_size = input.shape
        upsampler = Lambda(lambda x: x)
        if out_width > in_size[0]:
            upsampler = nn.Sequential(LambdaReduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                   Lambda(lambda x: x),
                                                   Lambda(lambda x: torch.zeros(
                                                       (x.size(0), out_width - in_size[0], x.size(2), x.size(3)),
                                                       device=x.device))
                                                  )
                                     )
        elif out_width < in_size[0]:
            upsampler = Lambda(lambda x: x[:, :out_width, :, :])

        return upsampler(input)


    def sizeof_output(self):
        h = self.in_size[1]
        w = self.in_size[2]

        h_out = int(np.floor(1+(2+h-self.kernel_size[0])/self.stride[0]))
        w_out = int(np.floor(1+(2+w-self.kernel_size[1])/self.stride[1]))
        return [self.width, h_out, w_out]

    def get_norms(self):
        out = []
        for m in self.children():
            for m2 in m.modules():
                if isinstance(m2, nn.Conv2d) or isinstance(m2, nn.Linear):
                    out.append(torch.norm(m2.weight))
        return out

    def forward(self, x):

        out = self.scale*self.layer(x)
        if not self.residual:
            return out

        out += self.upsample(x)
        return out


# Smaller basic block for conv networks
class ConvBlockSmall(nn.Module):

    def __init__(self, in_size, width, residual=True, scale=1.0, use_relu=True):
        super(ConvBlockSmall, self).__init__()
        self.kernel_size = (3,3)
        self.stride = (1,1)
        self.in_size = in_size
        self.width = width
        if use_relu:
            self.layer = nn.Sequential(  # Sequential,
                nn.Conv2d(in_size[0], width, self.kernel_size, self.stride, (1, 1), 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(  # Sequential,
                nn.Conv2d(in_size[0], width, self.kernel_size, self.stride, (1, 1), 1, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.Sigmoid()
            )

        self.residual = residual
        self.upsample = Lambda(lambda x: x)
        if width > in_size[0]:
            self.upsample = nn.Sequential(LambdaReduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                       Lambda(lambda x: x),
                                                       Lambda(lambda x: torch.zeros((x.size(0), width-in_size[0], x.size(2), x.size(3)),
                                                                       device=x.device))
                                                      )
                                         )
        elif width < in_size[0]:
            self.upsample = Lambda(lambda x: x[:, :width, :, :])

        self.scale = scale


        for m in self.children():
            for m2 in m.modules():
                weights_init(m2)


    @staticmethod
    def upsample_fn(input, out_width):
        in_size = input.shape
        upsampler = Lambda(lambda x: x)
        if out_width > in_size[0]:
            upsampler = nn.Sequential(LambdaReduce(lambda x, y, dim=1: torch.cat((x, y), dim),
                                                   Lambda(lambda x: x),
                                                   Lambda(lambda x: torch.zeros(
                                                       (x.size(0), out_width - in_size[0], x.size(2), x.size(3)),
                                                       device=x.device))
                                                   )
                                      )
        elif out_width < in_size[0]:
            upsampler = Lambda(lambda x: x[:, :out_width, :, :])

        return upsampler(input)

    def sizeof_output(self):
        h = self.in_size[1]
        w = self.in_size[2]

        h_out = int(np.floor(1+(2+h-self.kernel_size[0])/self.stride[0]))
        w_out = int(np.floor(1+(2+w-self.kernel_size[1])/self.stride[1]))
        return [self.width, h_out, w_out]

    def get_norms(self):
        out = []
        for m in self.children():
            for m2 in m.modules():
                if isinstance(m2, nn.Conv2d) or isinstance(m2, nn.Linear):
                    out.append(torch.norm(m2.weight))
        return out

    def forward(self, x):

        out = self.scale*self.layer(x)
        if not self.residual:
            return out

        out += self.upsample(x)
        return out


class ConvClfBlock(nn.Module):

    def __init__(self, input_size, num_classes):
        super(ConvClfBlock, self).__init__()
        num_features = input_size[0]*(input_size[1]//8)*(input_size[2]//8)
        self.layer = nn.Sequential(
                                nn.AvgPool2d(8),
                                Lambda(lambda x: x.view(x.size(0),-1)),
                                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),
                                              nn.Linear(num_features,num_classes))
                             )

    def forward(self, x):
        return self.layer(x)


