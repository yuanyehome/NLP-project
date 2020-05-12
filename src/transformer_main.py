"""
@author yy
@date 2020.5.11
"""

import torch
import argparse
import dill as pickle
from tqdm import tqdm

from utils import Constants, printInfo
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.translator import Translator
import jieba
import torchtext


def load_model(opt, device):
    checkpoint = torch.load(opt.model, map_location=device)
    args = checkpoint['settings']

    model = Transformer(
        args.vocab_size,
        in_pad_idx=args.pad_idx,
        out_pad_idx=args.pad_idx,
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        d_inner=args.d_inner_hid,
        d_k=args.d_k,
        d_v=args.d_v,
        n_head=args.n_head,
        n_layers=args.n_layers,
        dropout=args.dropout,
        target_embedding_output_weight_sharing=True,
        embedding_input_target_weight_sharing=True,
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    printInfo('Trained model state loaded.')
    return model


def main():
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=50)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]
    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        pad_idx=opt.pad_idx,
        eos_idx=opt.eos_idx,
        sos_idx=opt.bos_idx
    ).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]

    while True:
        print("请输入问题: ")
        sentence = input()
        sentence = list(jieba.cut(sentence))
        printInfo(sentence, "debugINFO")
        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in sentence]
        ret = translator.translate_sentence(torch.tensor([src_seq], dtype=torch.long).to(device))
        printInfo(ret, "debugINFO")


if __name__ == "__main__":
    main()
