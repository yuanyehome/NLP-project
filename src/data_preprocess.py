# -*- coding: utf-8 -*-
"""
@author fjw
modified by yy
"""

import random
from collections import Counter
import pickle

vocabulary_size = 155920
similar_k = 20


with open(r'../data/preprocessed_data/stop_words.txt', encoding='UTF-8-sig') as file0:
    stop = [k.strip() for k in file0.readlines()]

with open(r'../data/preprocessed_data/post.txt', encoding='UTF-8-sig') as file1:
    data_list1 = [k.strip() for k in file1.readlines()]
    # 列表推导式，得到data_list列表，元素为文件每一行的内容
    train_textlist_list1 = [k.split(maxsplit=1)[1] for k in data_list1]
    for i in range(len(train_textlist_list1)):
        train_textlist_list1[i] = train_textlist_list1[i].split()
    # 得到train_textlist_list，为列表的列表，元素为各个文本的由其各个词组成的列表
    words1 = [word for sentence in train_textlist_list1 for word in sentence if word not in stop]

with open(r'../data/preprocessed_data/response.txt', encoding='UTF-8-sig') as file2:
    data_list2 = [k.strip() for k in file2.readlines()]
    # 列表推导式，得到data_list列表，元素为文件每一行的内容
    train_textlist_list2 = [k.split(maxsplit=1)[1] for k in data_list2]
    for i in range(len(train_textlist_list2)):
        train_textlist_list2[i] = train_textlist_list2[i].split()
    # 得到train_textlist_list，为列表的列表，元素为各个文本的由其各个词组成的列表
    words2 = [word for sentence in train_textlist_list2 for word in sentence if word not in stop]

words = words1 + words2


counter = Counter(words)
vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
# print(vocabulary_list)

word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
# 列表推导式，先得到词汇对应id的列表，再强制转换为字典（b为词，a为下标索引即id），
# 之后可据此把词转为对应的id
text2idList = lambda text: [word2id_dict[word] for word in text if word in word2id_dict]
# 使用列表推导式和lambda表达式定义函数text2idlist，
# 将参数text该文本中的每个词转换为id然后构成一个列表
train_idlist_list1 = [text2idList(text) for text in train_textlist_list1]
# 对于每个文本调用text2idList把一个个词转为一个个id，得到train_idlist_list，
# 元素是每条文本中的词对应的id组成的列表
train_idlist_list2 = [text2idList(text) for text in train_textlist_list2]
# 对于每个文本调用text2idList把一个个词转为一个个id，得到train_idlist_list，
# 元素是每条文本中的词对应的id组成的列表

qa_list = []
now_qa = 0

with open(r'../data/preprocessed_data/original.txt', encoding='UTF-8-sig') as file3:
    data_list3 = [k.strip() for k in file3.readlines()]
    train_textlist_list3 = [k.split() for k in data_list3]
    for j in range(len(train_textlist_list3)):
        for i in range(1, len(train_textlist_list3[j])):
            qa_list.append([])
            qa_list[now_qa].append(train_idlist_list1[int(train_textlist_list3[j][0])])
            qa_list[now_qa].append(train_idlist_list2[int(train_textlist_list3[j][i])])
            qa_list[now_qa].append(int(train_textlist_list3[j][0]))
            now_qa += 1

similar_list = []
now_simi = 0

for i in range(0, len(qa_list) - similar_k, similar_k + 1):
    for j in range(1, similar_k + 1):
        similar_list.append([])
        similar_list[now_simi].append(qa_list[i][1])
        similar_list[now_simi].append(qa_list[i + j][1])
        if qa_list[i][2] == qa_list[i + j][2]:
            similar_list[now_simi].append(True)
        else:
            similar_list[now_simi].append(False)
        now_simi += 1

L = list(range(len(qa_list)))

for i in range(similar_k):
    random.shuffle(L)
    for j in range(len(qa_list)):
        similar_list.append([])
        similar_list[now_simi].append(qa_list[j][1])
        similar_list[now_simi].append(qa_list[L[j]][1])
        if qa_list[i][2] == qa_list[L[j]][2]:
            similar_list[now_simi].append(True)
        else:
            similar_list[now_simi].append(False)
        now_simi += 1

print("Some data information: \n \t qa_pair num: %d \n \t chosen_words num: %d"
      % (len(qa_list), len(vocabulary_list)))

with open("../data/preprocessed_data/chosen_word_list.pkl", "wb") as f:
    pickle.dump(vocabulary_list, f)
with open("../data/preprocessed_data/chosen_word2id.pkl", "wb") as f:
    pickle.dump(word2id_dict, f)
with open("../data/preprocessed_data/qa_pair.pkl", "wb") as f:
    pickle.dump(qa_list, f)
