#!/usr/bin/env Python
# coding=utf-8

import torch
import settings
import numpy as np
import scipy.io as scio
import h5py

if settings.DATASET == "WIKI":
    set = scio.loadmat(settings.x)
    label_set = np.array(set['label'], dtype=np.int)  # 2568*1
    label_set = np.eye(10)[(label_set-1).reshape(-1)]
    txt_set = np.array(set['text_vec'], dtype=np.float)  # 2568*512
    img_set = np.array(set['image_fea'], dtype=np.float)  # 25688*4096

    first = True
    for label in range(label_set.shape[1]):  # range(10)
        index = np.where(label_set[:, label] == 1)[0]  # label_set每一行中,分别在0,1,2,...,9等位置为1的样例的索引  (从小到大排列)
        N = index.shape[0]  # 数据集中0,1,2,...,9每一类样例的个数 (172,360,340,333,267,236,237,185,285,451,)
        perm = np.random.permutation(N)  # 打乱
        index = index[perm]  # 打乱index

        if first:
            test_index = index[:50]
            train_index = index[50:50 + 255]
            first = False
        else:
            ind = np.array([i for i in list(index) if
                            i not in (list(train_index) + list(test_index))])
            test_index = np.concatenate((test_index, ind[:50]))
            train_index = np.concatenate((train_index, ind[50:50 + 255]))

    if train_index.shape[0] < 2550:
        pick = np.array([i for i in list(range(label_set.shape[0])) if i not in list(train_index)])  # 13118=(18015-4897)
        N = pick.shape[0]
        perm = np.random.permutation(N)
        pick = pick[perm]
        res = 2550 - train_index.shape[0]
        train_index = np.concatenate((train_index, pick[:res]))
    #
    # if test_index.shape[0] < 496:
    #     pick = np.array([i for i in list(range(label_set.shape[0])) if i not in (list(test_index)+list(train_index))])  # 13118=(18015-4897)
    #     N = pick.shape[0]
    #     perm = np.random.permutation(N)
    #     pick = pick[perm]
    #     res = 496 - test_index.shape[0]
    #     test_index = np.concatenate((test_index, pick[:res]))

    # pick_train = np.array([i for i in list(range(label_set.shape[0])) if
    #                  i not in (list(test_index) + list(train_index))])
    # train_index = np.concatenate((train_index, pick_train[:]))
    # database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])
    database_index = train_index
    indexTest = test_index  # ndarray-->496
    indexDatabase = database_index  # ndarray-->2072
    indexTrain = train_index  # ndarray-->2560


    class WIKI(torch.utils.data.Dataset):
        def __init__(self, train=True, database=False):
            if train:
                self.train_labels = label_set[indexTrain]  # (5000,24)
                self.train_index = indexTrain  # 训练数据在整体数据集中的索引号
                self.txt = txt_set[indexTrain]
                self.img = img_set[indexTrain]
            elif database:
                self.train_labels = label_set[indexDatabase]
                self.train_index = indexDatabase
                self.txt = txt_set[indexDatabase]
                self.img = img_set[indexDatabase]
            else:
                self.train_labels = label_set[indexTest]
                self.train_index = indexTest
                self.txt = txt_set[indexTest]
                self.img = img_set[indexTest]

        def __getitem__(self, index):

            # img, target = img_set[self.train_index[index]], self.train_labels[index]
            # img = Image.fromarray(np.transpose(img, (2, 1, 0)))  # array转换成image
            # mirflickr.close()

            target = self.train_labels[index]
            txt = self.txt[index]
            img = self.img[index]
            return img, txt, target, index  # image-->(4096,)(ndarray)  txt-->(1386,)(ndarray)target-->(24,)(ndarray) index-->(0~70)

        def __len__(self):
            return len(self.train_labels)


    train_dataset = WIKI(train=True)
    test_dataset = WIKI(train=False, database=False)
    database_dataset = WIKI(train=False, database=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=settings.NUM_WORKERS,
                                           drop_last=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS,
                                          drop_last=False)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=settings.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=settings.NUM_WORKERS,
                                              drop_last=False)

# all_train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=settings.BATCH_SIZE,
#                                                shuffle=False,
#                                                num_workers=settings.NUM_WORKERS,
#                                                drop_last=False)  # 不抛弃除不尽的数据
