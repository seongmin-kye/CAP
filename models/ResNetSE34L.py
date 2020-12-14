#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *
import pdb

class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', **kwargs):

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(ResNetSE, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(64)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=64)

        if self.encoder_type == "TAP":
            out_dim = num_filters[3] * block.expansion
            self.fc = nn.Linear(out_dim, nOut)

        elif self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
            self.fc = nn.Linear(out_dim, nOut)

        elif self.encoder_type =='CAP':
            # temperature
            self.temperature = torch.tensor(0.05)
            #meta-learner
            self.meta_learner = nn.Sequential(
                                    nn.Conv1d(num_filters[3], num_filters[3], 1),
                                    nn.ReLU())
            # after fc
            self.att_bn = nn.BatchNorm1d(num_filters[3])
            self.fc = nn.Parameter(torch.Tensor(nOut, num_filters[3]))

        else:
            raise ValueError('Undefined encoder')

        self.global_w = nn.Parameter(torch.Tensor(5994, nOut))
        nn.init.xavier_uniform_(self.global_w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        x = self.torchfb(x)+1e-6
        x = self.instancenorm(x.log()).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #average pooling along with freq axis
        x = x.mean(dim=2)

        return x

    def TAP(self, x):
        x = x.mean(dim=2)
        x = self.fc(x)
        return x

    def SAP(self, x):
        x = x.permute(0, 2, 1)  # batch * L * D
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        x = torch.sum(x * w, dim=1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def CAP(self, x_s, x_q):

        feat_q = x_q.flatten(start_dim=2)
        feat_s = x_s.flatten(start_dim=2)

        # Meta-projection layer
        h_q = self.meta_learner(feat_q)
        h_s = self.meta_learner(feat_s)

        h_q = h_q.permute(0, 2, 1).unsqueeze(1)

        # Correlation layer
        cor_q = torch.matmul(F.normalize(h_q, dim=3), F.normalize(h_s, dim=1))
        cor_s = cor_q.permute(0, 1, 3, 2)

        # Context vector
        w_q = cor_q.mean(dim=2, keepdim=True)
        w_s = cor_s.mean(dim=2, keepdim=True)

        # Temperature scaling
        attention_q = (cor_q * w_q / self.temperature).sum(dim=3)
        attention_s = (cor_s * w_s / self.temperature).sum(dim=3)

        # Residual mechanism
        attention_q = torch.softmax(attention_q, dim=2) + 1
        attention_s = torch.softmax(attention_s, dim=2) + 1

        # Temporal average
        feat_q = feat_q.permute(0, 2, 1).unsqueeze(1) * attention_q.unsqueeze(3)
        feat_s = feat_s.permute(0, 2, 1).unsqueeze(0) * attention_s.unsqueeze(3)
        feat_q = feat_q.mean(dim=2)
        feat_s = feat_s.mean(dim=2)

        cat_qs = torch.cat((feat_q, feat_s), dim=0)
        qs, b, d = cat_qs.shape
        cat_qs = cat_qs.reshape(-1, d)
        cat_qs = F.relu(self.att_bn(cat_qs))
        cat_qs = cat_qs.reshape(qs, b, d)

        feat_s = cat_qs[qs // 2:]
        feat_q = cat_qs[:qs // 2]

        # Project to embedding space
        spk_s = F.linear(feat_s, self.fc)
        spk_q = F.linear(feat_q, self.fc)

        return spk_s, spk_q


def ResNetSE34L(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model