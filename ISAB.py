#!/usr/bin/env Python
# coding=utf-8

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from MAB import MultiheadAttentionBlock
import pdb


class ISABExample(nn.Module):
    def __init__(self, feat_dim, img_dim, txt_dim, code_len):  # , context_num=None
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
        self.img_encode = nn.Linear(feat_dim, code_len)
        self.txt_encode = nn.Linear(feat_dim, code_len)
        self.alpha_img = 1.0
        self.alpha_txt = 1.0

    def set_alpha_img(self, epoch):
        self.alpha_img = math.pow((1.0 * epoch + 1.0), 0.5)

    def set_alpha_txt(self, epoch):
        self.alpha_txt = math.pow((1.0 * epoch + 1.0), 0.5)

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
        z_img, z_txt = torch.split(torch.squeeze(isab_output), split_size_or_sections=16)  # (32,1024)(32,1024)
        dec_img = self.dec_img(z_img)  # (32,4096)
        dec_txt = self.dec_txt(z_txt)  # (32,512)

        hid_img = self.img_encode(z_img)  # (32,64)
        code_img = F.tanh(self.alpha_img * hid_img)  # (32,64)
        hid_txt = self.txt_encode(z_txt)  # (32,64)
        code_txt = F.tanh(self.alpha_txt * hid_txt)  # (32,64)

        affinity_bias = torch.eye(batch_size).cuda()
        affinity_bias = torch.cat((affinity_bias, affinity_bias), dim=0).cuda()
        affinity_bias = torch.cat((affinity_bias, affinity_bias), dim=1).cuda()
        affinity_mask = torch.ones_like(affinity_bias) - affinity_bias.cuda()
        affinity = affinity * affinity_mask + affinity_bias

        return z_img, z_txt, dec_img, dec_txt, affinity, code_img, code_txt

# if __name__ == '__main__':
#     _feat_dim = 1024
#     _img_dim = 4096  # 例如我们的图片原始特征是1024维
#     _txt_dim = 512  # 例如我们的文本原始特征是128维
#     model = ISABExample(_feat_dim, _img_dim, _txt_dim)
#
#     _img_feat = torch.ones([32, 4096])  # 假设batch size 64, feature用占位符表示，只用于演示其工作机制
#     _txt_feat = torch.ones([32, 512])
#
#     _dec_img, _dec_txt, _affinity = model(_img_feat, _txt_feat)
#
#     loss = (F.mse_loss(_img_feat, _dec_img) + F.mse_loss(_txt_feat, _dec_txt)) / 2.
#
#     print('this is the additional loss:')
#     print(loss)
#     print('---------------------------')
#     print('this is the similarity matrix we need:')
#     print(_affinity)
