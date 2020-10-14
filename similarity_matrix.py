import datasets
import numpy as np
import scipy.io as scio
import settings


def similarity_matrix():
    labels = datasets.label_set  # (21072,24) -> ndarray
    dim = labels.shape[1]  # 38

    query_label = np.zeros((len(datasets.indexTest), dim))  # (1984, 38)
    retrieval_label = np.zeros((len(datasets.indexDatabase), dim))  # (19088, 38)

    for index, i in enumerate(datasets.indexTest):
        query_label[index] = labels[i]

    for index, i in enumerate(datasets.indexDatabase):
        retrieval_label[index] = labels[i]

    query_label = np.expand_dims(query_label.astype(dtype=np.int), 1)  # (2000, 1, 24)
    retrieval_label = np.expand_dims(retrieval_label.astype(dtype=np.int), 0)  # (1, 18015, 24)
    similarity_matrix = np.bitwise_and(query_label, retrieval_label)  # (2000,18015,24)
    similarity_matrix = np.sum(similarity_matrix, axis=2)  # (2000,18015)
    similarity_matrix[similarity_matrix >= 1] = 1  # (2000,18015)
    return similarity_matrix
