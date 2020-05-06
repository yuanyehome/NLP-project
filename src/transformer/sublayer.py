"""
@author yy
@date 2020.5.6
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class ScaledProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        """
        :param temperature: scale, sqrt of d_k
        :param dropout: dropout rate
        """
        super(ScaledProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q: query
        :param k: key
        :param v: value
        :param mask: 是否要使用mask，在训练生成模型时候用到，防止模型看到当前后面的内容造成泄露
        :return: output, attention
        q, k, v含义参考`attention is all you need`
        TODO: 需要补充参数的维度
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # transpose:交换维度
        if mask is not None:
            # mask的shape和attn要相同，把某些位置填充value
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        :param n_head:
        :param d_model:
        :param d_k:
        :param d_v:
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.K = nn.Linear(d_model, n_head * d_k, bias=False)
        self.V = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        :param q:
        :param k:
        :param v:
        :param mask:
        :return: q, attn
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(v), v.size(1)
        residual = q
        q = self.layer_norm(q)

        q = self.Q(q).view(batch_size, len_q, n_head, d_k)
        k = self.K(k).view(batch_size, len_k, n_head, d_k)
        v = self.V(v).view(batch_size, len_k, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        return q, attn


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.fc2(F.relu(self.fc1(x)))
        x += residual
        return x

