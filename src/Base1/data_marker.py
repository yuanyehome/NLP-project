import random
import json
import numpy as np
import torch
from collections import Counter
from tqdm import*


def load_param(path):
    f_open = open(path, encoding='utf-8')
    param = json.load(f_open)
    return param


# 数据集划分训练/测试并转换为tensor表示
def similar_data2id(data, max_length, cut):
    print("similar")
    x = []
    y = []
    random.shuffle(data)
    length = len(data)
    p_bar = tqdm(total=length)
    for (seq1, seq2, label) in data:
        if len(seq1) > max_length:
            seq1 = seq1[:max_length]
        while len(seq1) < max_length:
            seq1.append(0)
        if len(seq2) > max_length:
            seq2 = seq2[:max_length]
        while len(seq2) < max_length:
            seq2.append(0)
        x.append([seq1, seq2])
        y.append(label)
        p_bar.update(1)
    p_bar.close()
    length = len(x)
    test_x = x[:int(length*cut)]
    test_y = y[:int(length * cut)]
    train_x = x[int(length*cut):]
    train_y = y[int(length * cut):]
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y, test_x, test_y


class GetData:
    def __init__(self):
        self.stop_words = []  # [word]
        self.vocab = {}  # {word:index}
        self.post_data = []  # [[post]]
        self.res_data = []  # [[res]]
        self.pairs = []  # [index(post),index(res)...]
        self.post_res = []  # [[post],[res],index(post)]

    def make_data(self, stop_path, post_path, res_path, pair_path, vocab_size):
        self.stop_words = []  # [word]
        self.vocab = {}  # {word:index}
        self.post_data = []  # [[post]]
        self.res_data = []  # [[res]]
        self.pairs = []  # [index(post),index(res)...]
        self.post_res = []  # [[post],[res],index(post)]

        with open(stop_path, 'r', encoding='UTF-8-sig') as file0:
            self.stop_words = [k.strip() for k in file0.readlines()]
            print("stop words success")

        with open(post_path, 'r', encoding='UTF-8-sig') as file1:
            data_list1 = [k.strip() for k in file1.readlines()]
            # 列表推导式，得到data_list列表，元素为文件每一行的内容
            post_seqs = [k.split(maxsplit=1)[1] for k in data_list1]
            for item in range(len(post_seqs)):
                post_seqs[item] = post_seqs[item].split()
            # 得到post_seq，为列表的列表，元素为各个文本的由其各个词组成的列表
            post_words = [word for sentence in post_seqs for word in sentence if word not in self.stop_words]
            print("post success", len(post_seqs))

        with open(res_path, 'r', encoding='UTF-8-sig') as file2:
            data_list2 = [k.strip() for k in file2.readlines()]
            # 列表推导式，得到data_list列表，元素为文件每一行的内容
            res_seqs = [k.split(maxsplit=1)[1] for k in data_list2]
            for item in range(len(res_seqs)):
                res_seqs[item] = res_seqs[item].split()
            # 得到res_seqs，为列表的列表，元素为各个文本的由其各个词组成的列表
            res_words = [word for sentence in res_seqs for word in sentence if word not in self.stop_words]
            print("res success", len(res_seqs))
    
        words = post_words+res_words
        counter = Counter(words)
        vocabulary = ['<UNK>'] + ['<PAD>'] + [k[0] for k in counter.most_common(vocab_size-2)]
        print(vocabulary)

        self.vocab = dict([(b, a) for a, b in enumerate(vocabulary)])
        # 列表推导式，先得到词汇对应id的列表，再强制转换为字典（b为词，a为下标索引即id），之后可据此把词转为对应的id
        text2id_list = lambda text: [self.vocab[word] if word in self.vocab else self.vocab['<UNK>'] for word in text]
        # 使用列表推导式和lambda表达式定义函数text2id_list，将参数text该文本中的每个词转换为id然后构成一个列表
        self.post_data = [text2id_list(text) for text in post_seqs]
        # 对于每个文本调用text2idList把一个个词转为一个个id，元素是每条文本中的词对应的id组成的列表
        self.res_data = [text2id_list(text) for text in res_seqs]
        # 对于每个文本调用text2idList把一个个词转为一个个id，元素是每条文本中的词对应的id组成的列表

        with open(pair_path, 'r', encoding='UTF-8-sig') as file3:
            data_list3 = [k.strip() for k in file3.readlines()]
            self.pairs = [k.split() for k in data_list3]
            num = 0
            for post in range(len(self.pairs)):
                for item in range(1, len(self.pairs[post])):
                    pair = [self.post_data[int(self.pairs[post][0])], self.res_data[int(self.pairs[post][item])]
                            , int(self.pairs[item][0])]
                    self.post_res.append(pair)
                    num += 1
            print("pair success")
        self.pairs = np.array(self.pairs)
        self.post_res = np.array(self.post_res)
        self.res_data = np.array(self.res_data)
        self.post_data = np.array(self.post_data)

    def save_data(self):
        with open('resource/dict.json', 'w', encoding='UTF-8-sig') as json_file:
            json.dump(self.vocab, json_file, ensure_ascii=False)
        np.save('resource/post_res.npy', self.post_res)
        np.save('resource/res_data.npy', self.res_data)
        np.save('resource/post_data.npy', self.post_data)
        np.save('resource/pairs.npy', self.pairs)

    def load_data(self):
        with open('resource/dict.json', "r", encoding='UTF-8-sig') as json_file:
            self.vocab = json.load(json_file)
        self.post_res = np.load('resource/post_res.npy', allow_pickle=True)
        self.res_data = np.load('resource/res_data.npy', allow_pickle=True)
        self.post_data = np.load('resource/post_data.npy', allow_pickle=True)
        self.pairs = np.load('resource/pairs.npy', allow_pickle=True)

    def get_vocab(self):
        return self.vocab

    # 获得replier所需数据
    def replier_maker(self, max_length, cut):
        print("similar")
        x = []
        y = []
        random.shuffle(self.post_res)
        length = len(self.post_res)
        p_bar = tqdm(total=length)
        for (post, res, post_id) in self.post_res:
            if len(post) > max_length:
                post = post[:max_length]
            while len(post) < max_length:
                post.append(0)
            if len(res) > max_length:
                res = res[:max_length]
            while len(res) < max_length:
                res.append(0)
            x.append(post)
            y.append(res)
            p_bar.update(1)
        p_bar.close()
        length = len(x)
        test_x = x[:int(length * cut)]
        test_y = y[:int(length * cut)]
        train_x = x[int(length * cut):]
        train_y = y[int(length * cut):]
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return train_x, train_y, test_x, test_y

    # 获得similar所需数据
    def similar_maker(self, similar_k):
        similar_list = []
        now_similar = 0
        length = len(self.post_data)
        print("make similar")
        for i in tqdm(range(0, length)):
            for j in range(1, len(self.pairs[i])):
                for k in range(1, similar_k):
                    similar_list.append([])
                    similar_list[now_similar].append(self.res_data[int(self.pairs[i][j])])
                    similar_list[now_similar].append(self.res_data[int(self.pairs[i][(j+k) % (len(self.pairs[i])-1)])])
                    similar_list[now_similar].append(1)
                    now_similar += 1
        order = list(range(len(self.post_res)))
        for i in tqdm(range(similar_k)):
            random.shuffle(order)
            for j in range(len(self.post_res)):
                similar_list.append([])
                similar_list[now_similar].append(self.post_res[j][1])
                similar_list[now_similar].append(self.post_res[order[j]][1])
                if self.post_res[j][2] == self.post_res[order[j]][2]:
                    similar_list[now_similar].append(1)
                else:
                    similar_list[now_similar].append(0)
                now_similar += 1
        return similar_list
