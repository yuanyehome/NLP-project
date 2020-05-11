"""
@author yy
@date 2020.5.6
"""

import torchtext
import dill as pickle
from tqdm import tqdm
import random
from utils import Constants
from utils import printInfo


def build_data_files():
    with open("data/preprocessed_data/chosen_word2id.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("data/preprocessed_data/chosen_word_list.pkl", "rb") as f:
        idx2word = pickle.load(f)
    with open("data/preprocessed_data/qa_pair.pkl", "rb") as f:
        qa_pairs = pickle.load(f)
    random.shuffle(qa_pairs)

    def change_back_to_sentence(L):
        ans = []
        for idx in L:
            ans.append(idx2word[idx])
        return ' '.join(ans)

    Q = []
    A = []

    for pair in tqdm(qa_pairs, desc="processing data:"):
        Q.append(change_back_to_sentence(pair[0]))
        A.append(change_back_to_sentence(pair[1]))

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

    with open('data/torchtext_data/train.in', 'w') as f:
        f.write('\n'.join(split_data_train(Q)))
    with open('data/torchtext_data/val.in', 'w') as f:
        f.write('\n'.join(split_data_val(Q)))

    with open('data/torchtext_data/train.out', 'w') as f:
        f.write('\n'.join(split_data_train(A)))
    with open('data/torchtext_data/val.out', 'w') as f:
        f.write('\n'.join(split_data_val(A)))


def main():
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
        path='./data/torchtext_data',
        fields=(SRC, TRG),
        exts=('.in', '.out'),
        train='train',
        validation='val',
        test='val'
    )
    SRC.build_vocab(train.src)
    TRG.build_vocab(train.trg)
    for w, _ in SRC.vocab.stoi.items():
        if w not in TRG.vocab.stoi:
            TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
    TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
    for w, i in TRG.vocab.stoi.items():
        TRG.vocab.itos[i] = w
    SRC.vocab.stoi = TRG.vocab.stoi
    SRC.vocab.itos = TRG.vocab.itos

    data = {
        'vocab': SRC,
        'train': train.examples,
        'valid': val.examples,
        'test': val.examples
    }
    pickle.dump(data, open('data/torchtext_data/data.pkl', 'wb'))
    printInfo("Torchtext data saved!")


if __name__ == '__main__':
    # build_data_files()
    main()
