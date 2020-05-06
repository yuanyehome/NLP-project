"""
@author yy
@date 2020.5.6
"""
import torch.nn as nn
from transformer.sublayer import MultiHeadAttention, PositionwiseFeedforward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedforward(d_model, d_inner, dropout=dropout)

    def forward(self, in_seq, mask=None):
        encoder_out, encoder_self_attn = self.slf_attn(in_seq, in_seq, in_seq, mask=mask)
        encoder_out = self.ffn(encoder_out)
        return encoder_out, encoder_self_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedforward(d_model, d_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output,
                slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            decoder_input, decoder_input, decoder_input, mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, encoder_output, encoder_output, mask=dec_enc_attn_mask
        )
        dec_output = self.ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

