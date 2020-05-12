import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class EncoderRNN(nn.Module):
    def __init__(self, batch_size, seq_size, embedding, embed_dim, layer_size, hidden_size=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.dropout = dropout
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        # gru层
        self.gru = nn.GRU(hidden_size, hidden_size, layer_size,
                          dropout=(0 if layer_size == 1 else dropout), bidirectional=True)

    def forward(self, inputs):  # (B, L)
        output = self.embedding(inputs)
        output = output.permute(1, 0, 2)  # (L, B, E)
        output, hidden = self.gru(output)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden


# attention layer
class Attn(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        attn_energies = torch.sum(hidden * encoder_outputs, dim=2)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# 一次运行获得一个word
class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, layer_size=1, dropout=0.5):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, layer_size, dropout=(0 if layer_size == 1 else dropout))
        # 分类线性层
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        attn_output = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        attn_output = attn_output.squeeze(1)
        output = torch.cat((rnn_output, attn_output), 1)
        output = torch.tanh(self.concat(output))
        output = self.out(output)
        output = F.softmax(output, dim=1)
        return output, hidden



