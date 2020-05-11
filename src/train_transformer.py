"""
@author yy
@date 2020.5.6
"""

import argparse
import math
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

from utils import Constants, printInfo, cal_loss, cal_performance, \
    patch_src, patch_trg
from transformer.Models import Transformer
from transformer.OptimUtils import ScheduleOptim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_pkl', default=None)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.embed_dim = args.d_model
    device = torch.device('cuda' if args.cuda else 'cpu')
    printInfo("Using device %s" % device)


if __name__ == "main":
    main()