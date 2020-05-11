"""
@author yy
@date 2020.5.11
"""

import dill as pickle
import os
from utils import printInfo
import random
from utils import Constants
import torchtext


def read_data():
    Q = []
    A = []
    for pair in os.listdir("../data/new_data"):
        if pair.split('.')[-1] != 'tsv':
            continue
        with open("../data/new_data/%s" % pair) as f:
            for line in f.readlines():
                qa = line.split('\t')
                Q.append(qa[0].strip())
                A.append(qa[1].strip())
    printInfo("qa nums: %d" % len(Q))
    zipped_data = list(zip(Q, A))
    random.shuffle(zipped_data)
    Q, A = zip(*zipped_data)

    def split_data_train(all_data):
        length = len(all_data)
        idx1 = length // 10 * 9
        train, val = all_data[0: idx1], all_data[idx1:]
        return train

    def split_data_val(all_data):
        length = len(all_data)
        idx1 = length // 10 * 9
        train, val = all_data[0: idx1], all_data[idx1:]
        return val

    with open('../data/new_data/train.in', 'w') as f:
        f.write('\n'.join(split_data_train(Q)))
    with open('../data/new_data/val.in', 'w') as f:
        f.write('\n'.join(split_data_val(Q)))

    with open('../data/new_data/train.out', 'w') as f:
        f.write('\n'.join(split_data_train(A)))
    with open('../data/new_data/val.out', 'w') as f:
        f.write('\n'.join(split_data_val(A)))


def main():
    # read_data()
    max_len = 50
    SRC = torchtext.data.Field(
        pad_token=Constants.PAD_WORD,
        init_token=Constants.BOS_WORD,
        eos_token=Constants.EOS_WORD
    )
    TRG = torchtext.data.Field(
        pad_token=Constants.PAD_WORD,
        init_token=Constants.BOS_WORD,
        eos_token=Constants.EOS_WORD
    )

    train, val, test = torchtext.datasets.TranslationDataset.splits(
        path='../data/new_data',
        fields=(SRC, TRG),
        exts=('.in', '.out'),
        train='train',
        validation='val',
        test='val',
        filter_pred=lambda x: (len(vars(x)['src']) <= max_len) and
                              (len(vars(x)['trg']) <= max_len)
    )
    SRC.build_vocab(train.src, min_freq=5)
    TRG.build_vocab(train.trg, min_freq=5)
    for w, _ in SRC.vocab.stoi.items():
        if w not in TRG.vocab.stoi:
            TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
    TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
    for w, i in TRG.vocab.stoi.items():
        TRG.vocab.itos[i] = w
    SRC.vocab.stoi = TRG.vocab.stoi
    SRC.vocab.itos = TRG.vocab.itos

    data = {
        'max_len': max_len,
        'vocab': {'src': SRC, "trg": TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': val.examples
    }
    printInfo("Vocab size: %d" % len(SRC.vocab))
    pickle.dump(data, open('../data/new_data/data.pkl', 'wb'))
    printInfo("Torchtext data saved!")


if __name__ == "__main__":
    main()
