"""
@author yy
@date 2020.5.6
"""

import numpy as np


def get_pad_mask(seq, pad_idx):
    """
    :param seq:
    :param pad_idx: 
    :return:
    """
    return (seq != pad_idx).unsqueeze(-2)
