"""
@author yy
@date 2020.5.6
"""

import numpy as np
import torch
import torch.nn.functional as F
import pickle
import random
import torchtext


def get_pad_mask(seq, pad_idx):
    """
    :param seq:
    :param pad_idx:
    :return:
    """
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """
    :param seq:
    :return:
    """
    sz_b, len_s = seq.size()
    subsequent_mask = (-torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1
    ) + 1).type(torch.uint8)
    return subsequent_mask


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (-one_hot + 1) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def printInfo(s, desc='INFO'):
    print("\033[32m[%s]\033[0m %s" % (desc, s))


class Constants:
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


if __name__ == "__main__":
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("...")
