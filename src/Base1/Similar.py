import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


# 句义相似度模型
class SimilarModel(torch.nn.Module):
    def __init__(self, batch_size, seq_size, embedding, vocab_dim, embed_dim, attn_dim, dropout):
        super(SimilarModel, self).__init__()
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.dropout = dropout
        self.embedding = embedding
        self.attn_dim = attn_dim
        self.vocab_dim = vocab_dim
        self.embed_dim = embed_dim
        # attention相关
        self.Q = nn.Linear(self.embed_dim, self.attn_dim)
        self.K = nn.Linear(self.embed_dim, self.attn_dim)
        self.V = nn.Linear(self.embed_dim, self.attn_dim)
        self.W = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Softmax(dim=1))

    def attn(self, inputs1, inputs2):  # (B,S,E)
        q = self.Q(inputs1)  # (B,S,A)
        k = self.K(inputs2).permute(0, 2, 1)  # (B,A,S)
        v = self.V(inputs1)  # (B,S,A)
        out = torch.bmm(F.softmax(torch.bmm(q, k)), v)  # (B,S,A)
        return out

    # 自相似特征提取
    def s_attn(self, inputs):  # (B,L,H)
        attn_w = torch.tanh(inputs)  # (B,L,H)
        attn_w = self.W(attn_w)  # (B,L,1)
        attn_w = attn_w.permute(0, 2, 1)  # (B,1,L)
        attn = torch.bmm(attn_w, inputs)  # (B,1,E)
        attn = torch.Tensor.reshape(attn, [self.batch_size, self.embed_dim])  # (B,E)
        attn = torch.tanh(attn)  # (B,E)
        return attn

    # 简单特征余弦相似度
    def similar_mul(self, inputs1, inputs2):
        attn1 = self.s_attn(inputs1)
        attn2 = self.s_attn(inputs2)
        cos = torch.cosine_similarity(attn1, attn2, dim=1)  # (B)
        cos = torch.clamp(cos, min=0.0, max=1.0)
        return cos

    def forward(self, inputs1, inputs2):  # (B,S,V)
        inputs1 = self.embedding(inputs1)
        inputs2 = self.embedding(inputs2)
        out = self.similar_mul(inputs1, inputs2)
        return out


