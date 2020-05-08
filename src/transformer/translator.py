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

    def _get_the_best_score_and_idx(self, gen_seq, decoder_output, scores, step):
        """
        :param gen_seq:
        :param decoder_output:
        :param scores:
        :param step:
        :return:
        """
        beam_size = self.beam_size
        best_k2_probs, best_k2_idx = decoder_output[:, -1, :].topk(beam_size)
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        _r, _k = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[_r, _k]
        gen_seq[:, :step] = gen_seq[_r, :step]
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, input_seq):
        assert input_seq.size(0) == 1
        pad_idx = self.pad_idx
        eos_idx = self.eos_idx
        max_seq_len = self.max_seq_len
        beam_size = self.beam_size
        alpha = self.alpha

        with torch.no_grad():
            input_mask = get_pad_mask(input_seq, pad_idx)
            encoder_output, gen_seq, scores = self._get_init_state(input_seq, input_mask)

            ans_idx = 0
            for step in range(2, max_seq_len):
                decoder_output = self._model_decode(gen_seq[:, :step],
                                                    encoder_output, input_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(
                    gen_seq, decoder_output, scores, step
                )
                eos_locs = gen_seq == eos_idx
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # TODO: 上面那句话在干啥
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
