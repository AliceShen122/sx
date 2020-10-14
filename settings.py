#!/usr/bin/env Python
# coding=utf-8

import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

DATASET = 'MIRFlickr'

if DATASET == 'MIRFlickr':
    x = './data/mir.mat'
    # test = '/media/test/15996296252/CrossModal/mir_test.mat'
    # train = '/media/test/15996296252/CrossModal/mir_train.mat'
    #
    # LABEL_DIR = '/home/test/桌面/DJSRH/DJSRH_KK(flicker)(correct)/data/MIRFlickr/mirflickr25k-lall.mat'
    # TXT_DIR = '/home/test/桌面/DJSRH/DJSRH_KK(flicker)(correct)/data/MIRFlickr/mirflickr25k-yall.mat'
    # IMG_DIR = '/home/test/桌面/DJSRH/DJSRH_KK(flicker)(correct)/data/flicker_img_4096.mat'

    alpha = 0.3
    beta = 0.3
    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    NUM_EPOCH = 200

    feat_dim = 1024
    LR_CODE = 0.0001

    EVAL_INTERVAL = 2

BATCH_SIZE = 16
CODE_LEN = 64

ETA = 0.4
gamma = 0.3
MU = 1.5
# alpha = 0.3
# beta = 0.3
# Lambda = 100
# gamma = 0.1


MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

GPU_ID = 1
NUM_WORKERS = 0
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'
pretrain_path_flicker_img = './data/ae_flicker_img_%d.pkl' % CODE_LEN
pretrain_path_flicker_txt = './data/ae_flicker_txt_%d.pkl' % CODE_LEN

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('BETA = %.4f' % beta)
# logger.info('LAMBDA1 = %.4f' % LAMBDA1)
# logger.info('LAMBDA2 = %.4f' % LAMBDA2)
logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
# logger.info('LR_IMG = %.4f' % LR_IMG)
# logger.info('LR_TXT = %.4f' % LR_TXT)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('CODE_LEN = %d' % CODE_LEN)
# logger.info('MU = %.4f' % MU)
# logger.info('ETA = %.4f' % ETA)
logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)
# logger.info('GPU_ID =  %d %d' % (GPU_ID[0], GPU_ID[1]))
logger.info('GPU_ID = %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')
