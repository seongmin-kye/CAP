#! /usr/bin/python
# -*- encoding: utf-8 -*-
## Re-implementation of prototypical networks (https://arxiv.org/abs/1703.05175).
## Numerically checked against https://github.com/cyvius96/prototypical-network-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from accuracy import accuracy

class ProtoLoss(nn.Module):

    def __init__(self):
        super(ProtoLoss, self).__init__()
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss')

    def forward(self, out_anchor, out_positive, label=None):
        stepsize        = out_anchor.size()[0]
        n_q = out_positive.size(0)//stepsize
        output = torch.matmul(out_positive, F.normalize(out_anchor, dim=-1).T)
        #output      = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1).expand(-1,-1,stepsize*n_q),out_anchor.unsqueeze(-1).expand(-1,-1,stepsize*n_q).transpose(0,2))**2)
        label       = torch.from_numpy(numpy.asarray(range(0,stepsize))).repeat(n_q).cuda()
        nloss       = self.criterion(output, label)
        prec1, _    = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 5))

        return nloss, prec1
