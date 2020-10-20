#!/usr/bin/env Python
# coding=utf-8

from __future__ import division  # 用于/相除的时候,保留真实结果.小数
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, cluster_acc, target_distribution, precision_recall, \
    precision_top_k, \
    optimized_mAP, compress_wiki
import kk
import settings
from ISAB_correct import ISABExample, ImgNet, TxtNet
# from ISAB import ImgNet, TxtNet
import os.path as osp
import os
from sklearn.cluster import KMeans
from similarity_matrix import similarity_matrix
import io
import scipy.io as scio
import numpy as np


# torch.set_default_tensor_type(torch.FloatTensor)


class Session:
    def __init__(self):
        self.logger = settings.logger
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(settings.GPU_ID)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        self.CodeNet_I = ImgNet(code_len=settings.CODE_LEN)
        self.CodeNet_T = TxtNet(txt_dim=512, code_len=settings.CODE_LEN)
        self.CodeNet = ISABExample(feat_dim=settings.feat_dim, img_dim=4096, txt_dim=512)
        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=settings.LR_IMG,
                                          # momentum=settings.MOMENTUM,
                                          weight_decay=settings.WEIGHT_DECAY)
            self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=settings.LR_TXT,
                                          # momentum=settings.MOMENTUM,
                                          weight_decay=settings.WEIGHT_DECAY)
            self.opt_code = torch.optim.Adam(self.CodeNet.parameters(), lr=settings.LR_CODE,
                                             # momentum=settings.MOMENTUM,
                                             weight_decay=settings.WEIGHT_DECAY)
        self.similarity_matrix = similarity_matrix()

    def train(self, epoch):
        self.CodeNet.cuda().train()
        self.CodeNet_I.cuda().train()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha_img(epoch)
        self.CodeNet_T.set_alpha_txt(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha_img, self.CodeNet_T.alpha_txt))

        for batch_idx, (F_I, F_T, _, idx) in enumerate(kk.train_loader):
            F_I = Variable(F_I.cuda())  # (50,4096)
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())  # (50,100)

            self.opt_code.zero_grad()
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            c_img = self.CodeNet_I(F_I.float())  # (50,128)
            c_txt = self.CodeNet_T(F_T)  # (50,128)
            d_img, d_txt, affinity_s = self.CodeNet(F_I.float(), F_T.float())  # (32,4096)(32,512)(64,64)(32,64)(32,64)

            affinity_i_t = affinity_s[settings.BATCH_SIZE:, :settings.BATCH_SIZE]  # (32,32)
            affinity_i_i = affinity_s[:settings.BATCH_SIZE, :settings.BATCH_SIZE]  # (32,32)
            affinity_t_t = affinity_s[settings.BATCH_SIZE:, settings.BATCH_SIZE:]  # (32,32)
            affinity_t_i = affinity_s[:settings.BATCH_SIZE, settings.BATCH_SIZE:]  # (32,32)
            affinity_s = (affinity_t_t + affinity_i_i + affinity_i_t + affinity_t_i) / 4
            S = ((affinity_s + affinity_s.T) / 2) ** 2

            B_I = F.normalize(c_img)  # (32,64)
            B_T = F.normalize(c_txt)  # (32,64)

            BI_BI = B_I.mm(B_I.t())  # (32,32)
            BT_BT = B_T.mm(B_T.t())  # (32,32)
            BI_BT = B_I.mm(B_T.t())  # (32,32)

            loss1 = F.mse_loss(BI_BI, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(BT_BT, S)
            loss_code = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3
            # loss_code = F.mse_loss(BI_BT, affinity_i_t)

            # scio.savemat()
            # latent_loss = F.mse_loss(latent_img, latent_txt)
            # F_I = F.normalize(F_I)  # (32,4096)
            # S_I = F_I.mm(F_I.t())  # (32,32)
            # S_I = S_I * 2 - 1  # (32,32)
            #
            # F_T = F.normalize(F_T)  # (32,1386)
            # S_T = F_T.mm(F_T.t())  # (32,32)
            # S_T = S_T * 2 - 1  # (32,32)
            #
            # B_I = F.normalize(c_img)  # (32,64)
            # B_T = F.normalize(c_txt)  # (32,64)
            #
            # BI_BI = B_I.mm(B_I.t())  # (32,32)
            # BT_BT = B_T.mm(B_T.t())  # (32,32)
            # BI_BT = B_I.mm(B_T.t())  # (32,32)
            #
            # S_tilde = settings.gamma * S_I + (1 - settings.gamma) * S_T  # (32,32)
            # S = (1 - settings.ETA) * S_tilde + settings.ETA * S_tilde.mm(S_tilde) / settings.BATCH_SIZE
            # S = S * settings.MU  # (32,32)
            # S = S.float()

            # loss1 = F.mse_loss(BI_BI, S)
            # loss2 = F.mse_loss(BI_BT, S)
            # loss3 = F.mse_loss(BT_BT, S)
            # loss_code = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3
            # loss = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3

            loss_isab = (F.mse_loss(F_I, d_img) + F.mse_loss(F_T, d_txt)) / 2.
            # loss = (F.mse_loss(F_I, d_img) + F.mse_loss(F_T, d_txt)) / 2.
            loss = settings.alpha * loss_code + settings.beta * loss_isab

            loss.backward()

            self.opt_code.step()
            self.opt_I.step()
            self.opt_T.step()

            if (batch_idx + 1) % (
                    len(kk.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'batch_idx: %d Epoch [%d/%d], Iter [%d/%d] '
                    'loss_code: %.4f '
                    'loss_isab: %.4f '
                    'Total Loss: %.4f '
                    % (batch_idx, epoch + 1, settings.NUM_EPOCH, batch_idx + 1,
                       len(kk.train_dataset) // settings.BATCH_SIZE,
                       loss_code.item(),
                       loss_isab.item(),
                       loss.item()))

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        # self.CodeNet.eval().cuda()

        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(kk.database_loader,
                                                                   kk.test_loader,
                                                                   self.CodeNet_I, self.CodeNet_T)

        MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
        MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        i2t_map = optimized_mAP(qu_BI, re_BT, self.similarity_matrix, 'hash', top=kk.indexDatabase.shape[0])
        i2t_pre_top_k = precision_top_k(qu_BI, re_BT, self.similarity_matrix,
                                        [1, 500, 1000, 1500, 2000], 'hash')
        i2t_pre, i2t_recall = precision_recall(qu_BI, re_BT, self.similarity_matrix)

        t2i_map = optimized_mAP(qu_BT, re_BI, self.similarity_matrix, 'hash', top=kk.indexDatabase.shape[0])
        t2i_pre_top_k = precision_top_k(qu_BT, re_BI, self.similarity_matrix,
                                        [1, 500, 1000, 1500, 2000], 'hash')
        t2i_pre, t2i_recall = precision_recall(qu_BT, re_BI, self.similarity_matrix)

        with io.open('results/results_%d.txt' % settings.CODE_LEN, 'a', encoding='utf-8') as f:
            f.write(u'MAP_I2T: ' + str(MAP_I2T) + '\n')
            f.write(u'MAP_T2I: ' + str(MAP_T2I) + '\n')
            f.write(u'i2t_map: ' + str(i2t_map) + '\n')
            f.write(u't2i_map: ' + str(t2i_map) + '\n')
            f.write(u'i2t_pre_top_k: ' + str(i2t_pre_top_k) + '\n')
            f.write(u't2i_pre_top_k: ' + str(t2i_pre_top_k) + '\n')
            f.write(u'i2t precision: ' + str(i2t_pre) + '\n')
            f.write(u'i2t recall: ' + str(i2t_recall) + '\n')
            f.write(u't2i precision: ' + str(t2i_pre) + '\n')
            f.write(u't2i recall: ' + str(t2i_recall) + '\n\n')

        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')


def main():
    sess = Session()
    if settings.EVAL == True:
        sess.eval()

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # save the model
            # if epoch + 1 == settings.NUM_EPOCH:
            #     sess.save_checkpoints(step=epoch + 1)


if __name__ == '__main__':
    main()
