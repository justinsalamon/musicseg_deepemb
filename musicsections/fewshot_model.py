# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
# Original code license: https://github.com/wyharveychen/CloserLookFewShot/blob/master/LICENSE.txt
# Reference:
# @inproceedings{chen2019closerfewshot,
#     title={A Closer Look at Few-shot Classification},
#     author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
#     booktitle={International Conference on Learning Representations},
#     year={2019}
# }
import torch.nn as nn
import math


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, in_BN=False, pool=True, pool_size=2, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if in_BN:
            self.in_BN = nn.BatchNorm2d(indim)
            self.parametrized_layers.insert(0, self.in_BN)
        if pool:
            self.pool = nn.MaxPool2d(pool_size)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, in_BN=(i==0), pool=(i < 4))
            # use input BN for the first layer, pool in the first 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1536  # 1024 for win_size=32

    def forward(self, x):
        out = self.trunk(x)
        return out


def Conv4():
    return ConvNet(4)
