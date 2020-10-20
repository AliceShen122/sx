#!/usr/bin/env Python
# coding=utf-8

import torch.nn as nn
from attention import MultiheadAttention


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d, n, dff, rate=0.1):
        """
        Arguments:
            d: an integer, input dimension.
            n: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super(MultiheadAttentionBlock, self).__init__()

        self.multihead = MultiheadAttention(d, n)
        self.ffn = nn.Linear(dff, d)
        self.layernorm1 = nn.LayerNorm(d)
        self.layernorm2 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, y, mask=None):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).

        It is invariant to permutations of the
        second dimension of tensor y (`m`).

        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
            :param x:
            :param y:
            :param training:
            :param mask:
        """
        attn1, attn_weights_block1 = self.multihead(x, y, y, mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2, attn_weights_block1
