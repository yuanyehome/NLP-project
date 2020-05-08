"""
@author yy
@date 2020.5.6
"""

import numpy as np
import torch


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


if __name__ == "__main__":
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("...")
