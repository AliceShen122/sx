#!/usr/bin/env Python
# coding=utf-8

import torchvision
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from MAB import MultiheadAttentionBlock
import pdb
import settings


class ISABExample(nn.Module):
    def __init__(self, feat_dim, img_dim, txt_dim):  # , context_num=None
        """
        :param feat_dim: feature dimension
        :param img_dim: original image feature dimension
        :param txt_dim: original text feature dimension
        :param context_num: hyper-parameter, the number of context vectors
        :param kwargs:
        """
        super(ISABExample, self).__init__()

        self.feat_dim = feat_dim
        self.modal_emb = nn.Embedding(2, feat_dim)
        self.enc_img = nn.Linear(img_dim, feat_dim)
        self.dec_img = nn.Linear(feat_dim, img_dim)
        self.enc_txt = nn.Linear(txt_dim, feat_dim)
        self.dec_txt = nn.Linear(feat_dim, txt_dim)
        self.mab = MultiheadAttentionBlock(self.feat_dim, 1, self.feat_dim)

    # noinspection PyMethodOverriding
    def forward(self, img_feat, txt_feat):
        batch_size = np.shape(img_feat)[0]
        zeros = torch.zeros((batch_size,), dtype=torch.int32).cuda()
        ones = torch.ones((batch_size,), dtype=torch.int32).cuda()
        # pdb.set_trace()
        modal_emb = self.modal_emb((torch.cat((zeros, ones), dim=0)).type(torch.LongTensor).cuda())  # (64,1024)
        img = self.enc_img(img_feat)  # (32,1024)
        txt = self.enc_txt(txt_feat)  # (32,1024)
        data_emb = torch.cat((img, txt), dim=0)  # (64,1024)
        isab_input = torch.unsqueeze(data_emb + modal_emb, dim=0)  # (1,64,1024)
        isab_output, affinity = self.mab(isab_input, isab_input)  # (1,64,1024)  (1,1,64,64)
        affinity = torch.squeeze(affinity)  # (64,64)
        # pdb.set_trace()
        z_img, z_txt = torch.split(torch.squeeze(isab_output), split_size_or_sections=settings.BATCH_SIZE)  # (32,1024)(32,1024)
        dec_img = self.dec_img(z_img)  # (32,4096)
        dec_txt = self.dec_txt(z_txt)  # (32,512)

        affinity_bias = torch.eye(batch_size).cuda()
        affinity_bias = torch.cat((affinity_bias, affinity_bias), dim=0).cuda()
        affinity_bias = torch.cat((affinity_bias, affinity_bias), dim=1).cuda()
        affinity_mask = torch.ones_like(affinity_bias) - affinity_bias.cuda()
        affinity = affinity * affinity_mask + affinity_bias

        return dec_img, dec_txt, affinity  # , code_img, code_txt


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, feat):
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha_img(self, epoch):
        self.alpha_img = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_dim):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_dim, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha_txt(self, epoch):
        self.alpha_txt = math.pow((1.0 * epoch + 1.0), 0.5)
