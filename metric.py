#!/usr/bin/env Python
# coding=utf-8
from __future__ import division  # 用于/相除的时候,保留真实结果.小数
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


# def compress(train_loader, test_loader, model_I, model_T):
#     re_BI = list([])
#     re_BT = list([])
#     re_L = list([])
#     for _, (data_I, data_T, target, _) in enumerate(train_loader):
#         var_data_I = Variable(data_I.cuda())
#         _,code_I = model_I(var_data_I)
#         code_I = torch.sign(code_I)
#         re_BI.extend(code_I.cpu().data.numpy())
#         re_L.extend(target.numpy())
#
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         _,code_T = model_T(var_data_T)
#         code_T = torch.sign(code_T)
#         re_BT.extend(code_T.cpu().data.numpy())
#
#     qu_BI = list([])
#     qu_BT = list([])
#     qu_L = list([])
#     for _, (data_I, data_T, target, _) in enumerate(test_loader):
#         var_data_I = Variable(data_I.cuda())
#         _, code_I = model_I(var_data_I)
#         code_I = torch.sign(code_I)
#         qu_BI.extend(code_I.cpu().data.numpy())
#         qu_L.extend(target.numpy())
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         _, code_T = model_T(var_data_T)
#         code_T = torch.sign(code_T)
#         qu_BT.extend(code_T.cpu().data.numpy())
#
#     re_BI = np.array(re_BI)
#     re_BT = np.array(re_BT)
#     re_L = np.array(re_L)
#
#     qu_BI = np.array(qu_BI)
#     qu_BT = np.array(qu_BT)
#     qu_L = np.array(qu_L)
#
#     return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress(train_loader, test_loader, model):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        train_var_data_I = Variable(data_I.cuda())  # (32,4096)
        pre_train_var_data_T = Variable(torch.Tensor(np.zeros((train_var_data_I.shape[0], 512))).cuda())  # (32,512)
        _, _, _, _, _, code_I, _ = model(train_var_data_I, pre_train_var_data_T)
        code_I = torch.sign(code_I)  # (32,64)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target.numpy())

        train_var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        pre_train_var_data_I = Variable(torch.Tensor(np.zeros((train_var_data_T.shape[0], 4096))).cuda())
        _, _, _, _, _, _, code_T = model(pre_train_var_data_I, train_var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        test_var_data_I = Variable(data_I.cuda())
        pre_test_var_data_T = Variable(torch.Tensor(np.zeros((test_var_data_I.shape[0], 512))).cuda())
        _, _, _, _, _, code_I, _ = model(test_var_data_I, pre_test_var_data_T)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target.numpy())

        test_var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        pre_test_var_data_I = Variable(torch.Tensor(np.zeros((test_var_data_T.shape[0], 4096))).cuda())
        _, _, _, _, _, _, code_T = model(pre_test_var_data_I, test_var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)
    # re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)
    # qu_L = np.eye(classes)[np.array(qu_L)]

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]  (2000,16)
    :param B2:  vector [r*n]  (18015,16)
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))  # (2000,18015)
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def precision_recall(q, r, similarity_matrix):
    pre_list = []
    recall_list = []
    query = q.copy()  # (2000,16)
    retrieval = r.copy()  # (18015,16)

    query[query >= 0] = 1
    query[query != 1] = 0
    retrieval[retrieval >= 0] = 1
    retrieval[retrieval != 1] = 0  # 将-1变为0

    query = query.astype(dtype=np.int8)
    retrieval = retrieval.astype(dtype=np.int8)

    distance = hamming_distance(query, retrieval)  # (2000,18015)
    distance_max = np.max(distance)  # 15
    distance_min = np.min(distance)  # 4

    for radius in range(int(distance_min), int(distance_max)):
        temp_distance = distance.copy()

        temp_distance[distance <= radius] = 1
        temp_distance[temp_distance > radius] = 0

        tp = np.sum(similarity_matrix * temp_distance)  # 7790,67581,723031
        precision = 0
        recall = 0
        if tp != 0:
            x = np.sum(temp_distance)  # 109
            y = np.sum(similarity_matrix) # 20848629
            precision = tp / x
            recall = tp / y
        pre_list.append(precision)
        recall_list.append(recall)

    pre_list = [round(i, 4) for i in pre_list]
    recall_list = [round(i, 4) for i in recall_list]

    return pre_list, recall_list


def hamming_distance(X, Y):
    '''
    返回两个矩阵以行为pair的汉明距离
    :param X: (n, hash_len)
    :param Y: (m, hash_len)
    :return: (n, m)
    '''

    res = np.bitwise_xor(np.expand_dims(X, 1), np.expand_dims(Y, 0))
    res = np.sum(res, axis=2)
    return res


def optimized_mAP(q, r, similarity_matrix, dis_metric='hash', top=None):
    query = q.copy()
    retrieval = r.copy()

    query_size = query.shape[0]

    if dis_metric == 'hash':

        query[query >= 0] = 1
        query[query != 1] = 0
        retrieval[retrieval >= 0] = 1
        retrieval[retrieval != 1] = 0

        query = query.astype(dtype=np.int8)
        retrieval = retrieval.astype(dtype=np.int8)
        distance = hamming_distance(query, retrieval)
    elif dis_metric == 'eu':
        distance = euclidean_distances(query, retrieval)
    else:
        distance = cosine_similarity(query, retrieval)

    sorted_index = np.argsort(distance)
    if dis_metric == 'cosine':
        sorted_index = np.flip(sorted_index, axis=1)

    sorted_similarity_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    sorted_similarity_matrix = np.asarray(sorted_similarity_matrix)[:, :top]
    neighbors = np.sum(sorted_similarity_matrix, axis=1)
    one_index = np.argwhere(sorted_similarity_matrix == 1)
    precision = 0
    cnt = 0

    for i in range(query_size):
        precision_at_i = 0
        if neighbors[i] == 0:
            continue
        for j in range(neighbors[i]):
            precision_at_i += np.sum(sorted_similarity_matrix[i, :one_index[cnt, 1] + 1]) / (one_index[cnt, 1] + 1)
            cnt += 1
        precision += precision_at_i / neighbors[i]
    mAP = precision / query_size

    return mAP


def precision_top_k(q, r, similarity_matrix, top_k, dis_metric):
    # calculate top k precision
    query = q.copy()
    retrieval = r.copy()

    if dis_metric == 'hash':

        query[query >= 0] = 1
        query[query != 1] = 0
        retrieval[retrieval >= 0] = 1
        retrieval[retrieval != 1] = 0

        query = query.astype(dtype=np.int8)
        retrieval = retrieval.astype(dtype=np.int8)
        distance = calculate_hamming(query, retrieval)

    sorted_index = np.argsort(distance)

    sorted_simi_matrix = np.array(list(map(lambda x, y: x[y], similarity_matrix, sorted_index)))
    precision = []
    for i in top_k:
        average_precison_top_i = np.mean(np.sum(sorted_simi_matrix[:, :i], axis=1) / i)
        precision.append(average_precison_top_i)

    precision = [round(i, 4) for i in precision]
    return precision