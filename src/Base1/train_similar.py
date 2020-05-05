import json
import random
import numpy as np
from tqdm import*

import torch
import torch.utils.data as dt
from torch import optim

from Similar import SimilarModel
from Embed import Embedding
from data_marker import load_param
from data_marker import GetData
from data_marker import similar_data2id
##################################################################################
# 预设参数
param_path = 'parameter.json'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 是否GPU训练
seed = 2020
torch.manual_seed(seed)
cont = True  # 是否继续训练
time = 0  # 从哪个模型开始


###################################################################################
# 训练过程
def train(train_loader, model, param, begin):
    model.train()
    size = len(train_loader)
    Loss = torch.nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=param['similar']['lr'])
    for epoch in range(begin, param['similar']['epoch']):
        print("epoch", epoch, ":")
        sum_num = 0
        sum_loss = 0
        sum_truth = 0
        for seq, y in train_loader:
            seq = seq.permute(1, 0, 2)
            seq1 = seq[0].to(device)
            seq2 = seq[1].to(device)
            y = y.float().to(device)
            out = model(seq1, seq2)
            out[out < 0.0] = 0.0
            out[out > 1.0] = 1.0
            loss = Loss(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.cpu().detach().numpy()
            sum_num += 1
            if sum_num % 50 == 0:
                print("time:", sum_num, "/", size, " train loss:", sum_loss / sum_num)
        print("train loss:", sum_loss / sum_num)
        if epoch % 2 == 0:
            torch.save(model, 'resource/model_{}.pkl'.format(epoch))


###################################################################################
def main():
    param = load_param(param_path)
    embedding = Embedding(param['vocab_size'], param['embed_dim']).to(device)
    model = SimilarModel(param['batch_size'], param['seq_size'], embedding, param['vocab_size'], param['embed_dim']
                         , param['similar']['attn_dim'], param['similar']['dropout']).to(device)

    data_loader = GetData()
    if cont:
        try:
            data_loader.load_data()
            print("old data_loader get")
        except:
            print("no old data_loader")
            data_loader = GetData()
            data_loader.make_data(stop_path=param['stop_words'], post_path=param['post'], res_path=param['response']
                                  , pair_path=param['original'], vocab_size=param['vocab_size'])
            data_loader.save_data()
    else:
        data_loader.make_data(stop_path=param['stop_words'], post_path=param['post'], res_path=param['response']
                              , pair_path=param['original'], vocab_size=param['vocab_size'])
        data_loader.save_data()

    if cont:
        try:
            train_x = np.load('resource/train_x.npy', allow_pickle=True)
            train_y = np.load('resource/train_y.npy', allow_pickle=True)
            test_x = np.load('resource/test_x.npy', allow_pickle=True)
            test_y = np.load('resource/test_y.npy', allow_pickle=True)
            print("old data get")
        except:
            print("no old data")
            data = data_loader.similar_maker(param['similar']['similar_k'])
            train_x, train_y, test_x, test_y = similar_data2id(data, param['seq_size'], param['similar']['cut'])
            np.save('resource/train_x.npy', train_x)
            np.save('resource/train_y.npy', train_y)
            np.save('resource/test_x.npy', test_x)
            np.save('resource/test_y.npy', test_y)
    else:
        data = data_loader.similar_maker(param['similar']['similar_k'])
        train_x, train_y, test_x, test_y = similar_data2id(data, param['seq_size'], param['similar']['cut'])
        np.save('resource/train_x.npy', train_x)
        np.save('resource/train_y.npy', train_y)
        np.save('resource/test_x.npy', test_x)
        np.save('resource/test_y.npy', test_y)

    # 封装数据集
    print("truth:", train_y.sum()/len(train_y))
    train_x = torch.from_numpy(train_x).long()
    train_y = torch.from_numpy(train_y)
    train_set = dt.TensorDataset(train_x, train_y)
    train_loader = dt.DataLoader(dataset=train_set, batch_size=param["batch_size"], pin_memory=True,
                                 shuffle=True, num_workers=0, drop_last=True)

    if cont:
        try:
            data_loader.load_data()
            model = torch.load('resource/model_{}.pkl'.format(time))
            print("old model get")
            train(train_loader, model, param, time)
        except:
            print("no old model")
            train(train_loader, model, param, 0)
    else:
        train(train_loader, model, param, 0)

    data_loader.save_data()
    torch.save(model, 'resource/similar.pkl')
    torch.save(embedding, 'resource/embedding.pkl')


main()
