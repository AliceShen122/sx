#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from metric import compress, calculate_top_map, calculate_map, cluster_acc, target_distribution, precision_recall, \
    precision_top_k, \
    optimized_mAP
import datasets
import settings
from ISAB import ISABExample
import os.path as osp
import os
from sklearn.cluster import KMeans
from similarity_matrix import similarity_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import io
import scipy.io as scio


# torch.set_default_tensor_type(torch.FloatTensor)


class Session:
    def __init__(self):
        self.logger = settings.logger
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(settings.GPU_ID)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        self.CodeNet = ISABExample(feat_dim=settings.feat_dim, img_dim=4096, txt_dim=512, code_len=settings.CODE_LEN)
        if settings.DATASET == "MIRFlickr":
            self.opt_code = torch.optim.Adam(self.CodeNet.parameters(), lr=settings.LR_CODE,
                                             # momentum=settings.MOMENTUM,
                                             weight_decay=settings.WEIGHT_DECAY)
        self.similarity_matrix = similarity_matrix()

    def train(self, epoch):
        self.CodeNet.cuda().train()
        self.CodeNet.set_alpha_img(epoch)
        self.CodeNet.set_alpha_txt(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.CodeNet.alpha_img, self.CodeNet.alpha_txt))

        for batch_idx, (F_I, F_T, _, idx) in enumerate(datasets.train_loader):
            F_I = Variable(F_I.cuda())  # (50,4096)
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())  # (50,100)
            # zeros = torch.zeros((settings.BATCH_SIZE,), dtype=torch.int32).cuda()
            # ones = torch.ones((settings.BATCH_SIZE,), dtype=torch.int32).cuda()

            self.opt_code.zero_grad()

            _, _, d_img, d_txt, affinity_s, c_img, c_txt = self.CodeNet(F_I, F_T.float())  # (32,4096)(32,512)(64,64)(32,64)(32,64)
            affinity_s = affinity_s[:settings.BATCH_SIZE, :settings.BATCH_SIZE]  # (32,32)
            B_I = F.normalize(c_img)  # (32,64)
            B_T = F.normalize(c_txt)  # (32,64)

            BI_BI = B_I.mm(B_I.t())  # (32,32)
            BT_BT = B_T.mm(B_T.t())  # (32,32)
            BI_BT = B_I.mm(B_T.t())  # (32,32)

            loss2 = F.mse_loss(BI_BT, affinity_s)
            loss1 = F.mse_loss(BI_BI, affinity_s)
            loss3 = F.mse_loss(BT_BT, affinity_s)
            loss_code = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3
            # loss_code = F.mse_loss(BI_BT, affinity_s)

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
            #
            # loss1 = F.mse_loss(BI_BI, S)
            # loss2 = F.mse_loss(BI_BT, S)
            # loss3 = F.mse_loss(BT_BT, S)
            # loss_code = settings.LAMBDA1 * loss1 + 1 * loss2 + settings.LAMBDA2 * loss3

            loss_isab = (F.mse_loss(F_I, d_img) + F.mse_loss(F_T, d_txt)) / 2.
            loss = settings.alpha * loss_code + settings.beta * loss_isab

            loss.backward()
            self.opt_code.step()

            if (batch_idx + 1) % (
                    len(datasets.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'batch_idx: %d Epoch [%d/%d], Iter [%d/%d] '
                    'loss_code: %.4f'
                    'loss_isab: %.4f'
                    'Total Loss: %.4f'
                    % (batch_idx, epoch + 1, settings.NUM_EPOCH, batch_idx + 1,
                       len(datasets.train_dataset) // settings.BATCH_SIZE,
                       loss_code.item(),
                       loss_isab.item(),
                       loss.item()))

    def eval(self):
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet.eval().cuda()

        if settings.DATASET == "MIRFlickr":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(datasets.database_loader,
                                                              datasets.test_loader,
                                                              self.CodeNet)

        MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
        MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
        i2t_map = optimized_mAP(qu_BI, re_BT, self.similarity_matrix, 'hash', top=datasets.indexDatabase.shape[0])
        i2t_pre_top_k = precision_top_k(qu_BI, re_BT, self.similarity_matrix,
                                        [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                                         800, 850, 900, 950, 1000], 'hash')
        i2t_pre, i2t_recall = precision_recall(qu_BI, re_BT, self.similarity_matrix)

        t2i_map = optimized_mAP(qu_BT, re_BI, self.similarity_matrix, 'hash', top=datasets.indexDatabase.shape[0])
        t2i_pre_top_k = precision_top_k(qu_BT, re_BI, self.similarity_matrix,
                                        [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                                         800, 850, 900, 950, 1000], 'hash')
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

    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'CodeNet': self.CodeNet.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet.load_state_dict(obj['CodeNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def main():
    sess = Session()
    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval()
            # save the model
            if epoch + 1 == settings.NUM_EPOCH:
                sess.save_checkpoints(step=epoch + 1)


if __name__ == '__main__':
    main()
