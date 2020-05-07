"""
@author yy
@date 2020.5.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer
from utils import get_subsequent_mask, get_pad_mask


class Translator(nn.Module):
    def __init__(self, model, beam_size, max_seq_len, pad_idx, sos_idx, eos_idx):
        """
        :param model:
        :param beam_size:
        :param max_seq_len:
        :param pad_idx:
        :param sos_idx:
        :param eos_idx:
        """
        super(Translator, self).__init__()
        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.model = model
        self.model.eval()
        # 在含有BatchNorm或LayerNorm或dropout时，
        # 在进行测试前要使用eval()，否则会改变权值
        # 即让模型进入测试模式

        self.register_buffer('init_seq', torch.tensor([[sos_idx]], dtype=torch.long))
        self.register_buffer('blank_seqs', torch.full((beam_size, max_seq_len),
                                                      pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.sos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0)
        )

    def _model_decode(self, target_seq, encoder_output, input_mask):
        """
        TODO: 似乎是用来单次解码的
        """
        target_mask = get_subsequent_mask(target_seq)
        decoder_output, *_ = self.model.decoder(target_seq, target_mask,
                                                encoder_output, input_mask)
        return F.softmax(self.model.output_layer(decoder_output), dim=-1)

    def _get_init_state(self, input_seq, input_mask):
        # TODO: 这个函数在做啥
        beam_size = self.beam_size
        encoder_output, *_ = self.model.encoder(input_seq, input_mask)
        decoder_output = self._model_decode(self.init_seq, encoder_output, input_mask)
        best_k_probs, best_k_idx = decoder_output[:, -1, :].topk(beam_size)
        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        encoder_output = encoder_output.repeat(beam_size, 1, 1)
        return encoder_output, gen_seq, scores



