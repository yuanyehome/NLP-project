import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class EncoderRNN(nn.Module):
    def __init__(self, batch_size, seq_size, embedding, attn_dim, embed_dim, layer_size, hidden_size=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.dropout = dropout
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        # LSTM层
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.layer_size,
                            dropout=self.dropout,
                            bidirectional=True
                            )
        # Attention相关
        self.W = nn.Sequential(nn.Linear(self.attn_dim, 1), nn.Softmax(dim=1))

    # 整句注意力合并
    def attention(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        attn_w = torch.tanh(inputs)  # (B,L,H)
        print(inputs.size())
        attn_w = self.W(attn_w)  # (B,L,1)
        attn_w = attn_w.permute(0, 2, 1)  # (B,1,L)
        attn = torch.bmm(attn_w, inputs)  # (B,1,A)
        attn = torch.Tensor.reshape(attn, [self.batch_size, self.attn_dim])  # (B,A)
        attn = torch.tanh(attn)  # (B,A)
        return attn

    def forward(self, inputs):  # (B, L)
        out = self.embedding(inputs)
        out = out.permute(1, 0, 2)  # (L, B, E)
        out, (hn, cn) = self.lstm(out)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        # print("out", out.size())
        # print("h", hn.size())
        return out, hn


# attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        else:
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# 一次运行获得一个word
class DecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, layer_size=1, dropout=0.5):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, layer_size, dropout=(0 if layer_size == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # print("embedded", embedded.size())
        # print("last_hidden", last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden



