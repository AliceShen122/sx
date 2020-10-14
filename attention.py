#!/usr/bin/env Python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask):
    """

    :param q: query of attention [N H L_q D]  # (1,1,64,1024)
    :param k: key of attention [N H L_k D]  # (1,1,64,1024)
    :param v: value of attention [N H L_v D]  # (1,1,64,1024)
    :param mask: [... L_q L_k]
    :return:
    """

    matmul_qk = torch.matmul(q, k.reshape(1, 1, 1024, -1))  # (..., seq_len_q, seq_len_k)  (1,1,64,64)

    dk = float(k.shape[-1])  # 1024
    scaled_attention_logits = matmul_qk / (dk**2)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiheadAttention(nn.Module):

    def __init__(self, d, n):
        """
        Arguments:
            d: an integer, dimension of queries and values.
                It is assumed that input and
                output dimensions are the same.
            n: an integer, number of heads.
        """
        super(MultiheadAttention, self).__init__()

        self.n = n
        self.d = d
        assert self.d % self.n == 0
        # everything is projected to this dimension
        self.p = self.d // self.n

        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        # self.attention = Attention(temperature=self.p**0.5)

    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.n, self.p))
        return x.transpose(2, 1)

    def forward(self, queries, keys, values, mask):
        """
        Arguments:
            queries: a float tensor with shape [b, n, d].  # img_feat
            keys: a float tensor with shape [b, m, d].     # txt_feat
            values: a float tensor with shape [b, m, d].   # txt_feat
        Returns:
            a float tensor with shape [b, n, d].
        """
        batch_size = queries.shape[0]

        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            queries, keys, values, mask)

        scaled_attention = scaled_attention.transpose(2, 1)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = torch.reshape(scaled_attention,
                                         (batch_size, -1, self.d))  # (batch_size, seq_len_q, d_model)

        output = self.concatenation(concat_attention)  # shape [b, n, d]

        return output, attention_weights
