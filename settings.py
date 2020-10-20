#!/usr/bin/env Python
# coding=utf-8

import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

DATASET = 'WIKI'

if DATASET == 'WIKI':
    x = '/media/test/15996296252/CrossModal/wiki.mat'  # image_fea 2568*4096；label 2568*1；text_vec 2568*512 (double)
    y = '/media/test/15996296252/CrossModal/wiki_test.mat'  # id_test 1*500;image_test 500*4096;label_test 500*1；text_test 500*512 (double)
    z = '/media/test/15996296252/CrossModal/wiki_train.mat'  # id_train 1*2000;image_train 2000*4096;label_train 2000*1；text_train 2000*512 (double)

    # LABEL_DIR = '/home/test/桌面/DJSRH/DJSRH_KK(flicker)(correct)/data/MIRFlickr/mirflickr25k-lall.mat'
    # TXT_DIR = '/home/test/桌面/DJSRH/DJSRH_KK(flicker)(correct)/data/MIRFlickr/mirflickr25k-yall.mat'
    # IMG_DIR = '/home/test/桌面/DJSRH/DJSRH_KK(flicker)(correct)/data/flicker_img_4096.mat'
    DATA_DIR = '/media/test/15996296252/wiki/images'
    LABEL_DIR = '/media/test/15996296252/wiki/raw_features.mat'
    TRAIN_LABEL = '/media/test/15996296252/wiki/trainset_txt_img_cat.list'
    TEST_LABEL = '/media/test/15996296252/wiki/testset_txt_img_cat.list'

    alpha = 1
    beta = 0.1
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 600

    # LR_IMG_pretrain_ae = 0.0001
    # LR_TXT_pretrain_ae = 0.0001
    # LR_IMG_cl = 0.000001
    # LR_TXT_cl = 0.000001
    feat_dim = 1024
    LR_CODE = 0.0001
    LR_IMG = 0.001
    LR_TXT = 0.01
    # LR_domain = 0.001
    # LR_fc = 0.00001

    EVAL_INTERVAL = 2

BATCH_SIZE = 50
CODE_LEN = 128

ETA = 0.4
gamma = 0.3
MU = 1.5
# alpha = 0.3
# beta = 0.3
# Lambda = 100
# gamma = 0.1


MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 0
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'

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
# logger.info('GPU_ID = %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')
