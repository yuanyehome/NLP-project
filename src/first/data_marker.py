import random
import json
import yaml
import io
import jieba
import numpy as np
import torch
from collections import Counter
from tqdm import*
import pandas as pd


def load_param(path):
    f_open = open(path, encoding='utf-8')
    param = json.load(f_open)
    return param


class GetData:
    def __init__(self):
        self.classes = ['conversations', 'emotion', 'greetings', 'politics', 'ai', 'psychology',
                        'science', 'history', 'botprofile']
        self.stop_words = []  # [word]
        self.vocab_size = 0
        self.vocab = {}  # {word:index}
        self.re_vocab = {}
        self.post_res = []
        self.post = []  # [[post]]
        self.res = []  # [[res]]

    def make_data(self, stop_path, path):
        self.stop_words = []  # [word]
        self.vocab = {}  # {word:index}
        self.re_vocab = {}
        self.post_res = []
        self.post = []  # [[post]]
        self.res = []  # [[res]]

        with open(stop_path, 'r', encoding='UTF-8-sig') as file0:
            self.stop_words = [k.strip() for k in file0.readlines()]
            print("stop words success")

        data = pd.read_csv(path, sep='\t', header=None)
        # print(data[0])
        length = len(data[0])
        print(length)
        for item in range(length):
            self.post_res.append([[word for word in list(jieba.cut(data[0][item]))],
                                  [word for word in list(jieba.cut(data[1][item]))]])

        random.shuffle(self.post_res)

        for item in range(len(self.post_res)):
            self.post.append(self.post_res[item][0])
            self.res.append(self.post_res[item][1])
            self.post[item] = [word if (word != '。' and word != '，') else '.' for word in self.post[item]]
            self.res[item] = [word if (word != '。' and word != '，') else '.' for word in self.res[item]]
            self.post[item] = [word if word != '？' else '?' for word in self.post[item]]
            self.res[item] = [word if word != '？' else '?' for word in self.res[item]]
            self.post[item] = [word for word in self.post[item]
                               if (word != ' ' and word != '（' and word != '）' and word != '`')]
            self.res[item] = [word for word in self.res[item]
                              if (word != ' ' and word != '（' and word != '）' and word != '`')]

        print(self.post[0], self.res[0])
        # 得到词表
        post_words = [word for sentence in self.post for word in sentence]
        res_words = [word for sentence in self.res for word in sentence]

        words = post_words+res_words
        print("all_words", len(words))
        counter = Counter(words)
        print("words_num", len(counter))
        vocabulary = ['<SOS>'] + ['<EOS>'] + ['<PAD>'] + [k[0] for k in counter.most_common(len(counter)) if counter[k[0]] > 2]
        self.vocab_size = len(vocabulary)
        print(vocabulary[self.vocab_size-1])
        print(self.vocab_size)

        self.vocab = dict([(b, a) for a, b in enumerate(vocabulary)])
        self.re_vocab = dict([(a, b) for a, b in enumerate(vocabulary)])

        self.res = np.array(self.res)
        self.post = np.array(self.post)

    def save_data(self):
        with open('resource/dict.json', 'w', encoding='UTF-8-sig') as json_file:
            json.dump(self.vocab, json_file, ensure_ascii=False)
        with open('resource/re_dict.json', 'w', encoding='UTF-8-sig') as json_file:
            json.dump(self.re_vocab, json_file, ensure_ascii=False)
        np.save('resource/post.npy', self.post)
        np.save('resource/res.npy', self.res)

    def load_data(self):
        with open('resource/dict.json', "r", encoding='UTF-8-sig') as json_file:
            self.vocab = json.load(json_file)
        with open('resource/re_dict.json', "r", encoding='UTF-8-sig') as json_file:
            self.re_vocab = json.load(json_file)
        self.res = np.load('resource/res.npy', allow_pickle=True)
        self.post = np.load('resource/post.npy', allow_pickle=True)
        self.vocab_size = len(self.vocab)

    # 获得replier所需数据
    def replier_maker(self, max_length, cut):
        print("similar")
        x = []
        y = []
        for item in tqdm(range(len(self.post))):
            post = [self.vocab[word] for word in self.post[item] if word in self.vocab.keys()]
            res = [self.vocab[word] for word in self.res[item] if word in self.vocab.keys()]
            if len(post) >= max_length:
                post = post[:max_length-1]
            post.append(self.vocab['<EOS>'])
            while len(post) < max_length:
                post.append(self.vocab['<PAD>'])
            if len(res) >= max_length:
                res = res[:max_length-1]
            res.append(self.vocab['<EOS>'])
            while len(res) < max_length:
                res.append(self.vocab['<PAD>'])
            x.append(post)
            y.append(res)
        print(len(x))
        length = len(x)
        test_x = x[int(length * cut):]
        test_y = y[int(length * cut):]
        train_x = x[:int(length * cut)]
        train_y = y[:int(length * cut)]
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return train_x, train_y, test_x, test_y


# some = GetData()
# some.make_data("resource/stop_words.txt", "resource/qingyun.tsv")