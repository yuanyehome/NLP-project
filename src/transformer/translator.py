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
        super(Translator, self).__init__()

