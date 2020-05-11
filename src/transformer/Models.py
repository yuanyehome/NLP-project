"""
@author yy
@date 2020.5.6
"""
import torch
import torch.nn as nn
import numpy as np
from transformer.layers import EncoderLayer, DecoderLayer
from utils import get_pad_mask, get_subsequent_mask


def _get_sin_encoding_table(n_position, d_hid):
    def get_position_angle(position):
        return [position / np.power(10000, 2 * (idx // 2) / d_hid)
                for idx in range(d_hid)]

    sin_table = np.array([get_position_angle(pos_idx)
                          for pos_idx in range(n_position)])
    sin_table[:, 0::2] = np.sin(sin_table[:, 0::2])
    sin_table[:, 1::2] = np.cos(sin_table[:, 1::2])

    return torch.tensor(sin_table, dtype=torch.float).unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # 不将这个table视为参数，而是常量
        # [1, n_position, embed_dim]
        self.register_buffer('pos_table', _get_sin_encoding_table(n_position, d_hid))

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, n_head, d_k, d_v, d_model,
                 d_inner, pad_idx, dropout=0.1, n_position=200):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_encoding = PositionalEncoding(embed_dim, n_position)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, in_seq, mask, return_attn=False):
        enc_self_attn_list = []
        enc_output = self.dropout(self.position_encoding(self.embed(in_seq)))

        for enc_layer in self.layers:
            enc_output, enc_self_attn = enc_layer(enc_output, mask=mask)
            enc_self_attn_list.append(enc_self_attn)
        enc_output = self.layer_norm(enc_output)

        if return_attn:
            return enc_output, enc_self_attn_list
        else:
            return enc_output,


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, n_head, d_k, d_v, d_model,
                 d_inner, pad_idx, dropout=0.1, n_position=200):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_encoding = PositionalEncoding(embed_dim, n_position)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, seq, dec_mask, encoder_output, enc_mask, return_attns=False):
        dec_self_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.position_encoding(self.embed(seq)))
        for dec_layer in self.layers:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_output,
                slf_attn_mask=dec_mask,
                dec_enc_attn_mask=enc_mask
            )
            dec_self_attn_list.append(dec_self_attn)
            dec_enc_attn_list.append(dec_enc_attn)
        dec_output = self.layer_norm(dec_output)
        if return_attns:
            return dec_output, dec_self_attn_list, dec_enc_attn_list
        else:
            return dec_output,


class Transformer(nn.Module):
    def __init__(self, vocab_size, in_pad_idx, out_pad_idx,
                 embed_dim=300, d_model=300, d_inner=2048, n_layers=6,
                 n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                 target_embedding_output_weight_sharing=True,
                 embedding_input_target_weight_sharing=True
                 ):
        """
        :param vocab_size: 词典大小
        :param in_pad_idx: 输入句子的pad的idx值
        :param out_pad_idx: 目标句子的pad的idx值
        :param embed_dim: 词向量嵌入维度
        :param d_model: 模型中间传递的每个词维度（和embed_dim应该相同）
        :param d_inner: 全连接层隐层维度
        :param n_layers: 一个结构中有几个encoder block
        :param n_head: 多头注意力的head数目
        :param d_k: key和query的维度（参考论文）
        :param d_v: value的维度（参考论文）
        :param n_position: 最多有几个位置（即seqlen最多是多少，比最大的大就行）
        """
        # TODO 预训练词向量
        super(Transformer, self).__init__()
        assert (d_model == embed_dim)
        self.in_pad_idx = in_pad_idx
        self.out_pad_idx = out_pad_idx
        self.encoder = Encoder(
            vocab_size, embed_dim, n_layers, n_head,
            d_k, d_v, d_model, d_inner, in_pad_idx, dropout, n_position
        )
        self.decoder = Decoder(
            vocab_size, embed_dim, n_layers, n_head,
            d_k, d_v, d_model, d_inner, in_pad_idx, dropout, n_position
        )
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 以下内容见论文第五页，section 3.4，嵌入层与pre-softmax层权重一致，但是在嵌入层
        # 要*sqrt(d_model)，相当于在pre-softmax除以sqrt(d_model)
        # 但论文并没有说为什么
        self.x_logit_scale = 1
        if target_embedding_output_weight_sharing:
            self.output_layer.weight = self.decoder.embed.weight
            self.x_logit_scale = d_model ** -0.5
        if embedding_input_target_weight_sharing:
            self.encoder.embed.weight = self.decoder.embed.weight

    def forward(self, in_seq, target_seq):
        """
        :param in_seq: [batch_size, seq_len]
        :param target_seq: [batch_size, seq_len]，包含了SOS_token
        :return: [batch_size * target_seq_len, vocab_size]，
                    应该包含EOS_token，不包含SOS_token；每一个地方是一个softmax后的概率
        """
        in_mask = get_pad_mask(in_seq, self.in_pad_idx)
        target_mask = \
            get_pad_mask(target_seq, self.out_pad_idx) & get_subsequent_mask(target_seq)
        enc_output, *_ = self.encoder(in_seq, in_mask)
        dec_output, *_ = self.decoder(target_seq, target_mask, enc_output, in_mask)
        dec_output = self.output_layer(dec_output) * self.x_logit_scale

        return dec_output.view(-1, dec_output.size(2))
