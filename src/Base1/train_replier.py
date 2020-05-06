import json
import random
import numpy as np
from tqdm import*

import torch
import torch.utils.data as dt
from torch import optim

from Embed import Embedding
from data_marker import load_param
from data_marker import GetData
from Replier import EncoderRNN
from Replier import DecoderRNN
from Replier import Attn
##################################################################################
# 预设参数
param_path = 'parameter.json'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 是否GPU训练
seed = 2020
torch.manual_seed(seed)
cont = True  # 是否继续
time = 0  # 从哪个模型开始
train_num = 20  # 训练轮数
train_mode = True  # 训练/测试
teacher_forcing_ratio = 1.0  # 用真正回答作为Encoder输入的比例
lr = 0.0001
decoder_learning_ratio = 5.0
clip = 50.0  # 梯度裁剪系数


#################################################################################
# 损失函数
def NLLLoss(inp, target):
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.mean()
    loss = loss.to(device)
    return loss


###################################################################################
# 测试过程
def test(test_loader, model):
    model.eval()
    size = len(test_loader)
    Loss = torch.nn.BCELoss().to(device)
    sum_num = 0
    sum_loss = 0
    sum_truth = 0
    for seq, y in test_loader:
        seq = seq.permute(1, 0, 2)
        seq1 = seq[0].to(device)
        seq2 = seq[1].to(device)
        y = y.float().to(device)
        out = model(seq1, seq2)
        out[out < 0.0] = 0.0
        out[out > 1.0] = 1.0
        loss = Loss(out, y)
        sum_loss += loss.cpu().detach().numpy()
        sum_truth += torch.mean(torch.abs(y-out)).item()
        sum_num += 1
        if sum_num % 50 == 0:
            print("time:", sum_num, "/", size, " test loss:", sum_loss / sum_num)
            print("time:", sum_num, "/", size, " truth loss:", sum_truth / sum_num)
    print("test loss:", sum_loss / sum_num)
    print("truth loss:", sum_truth / sum_num)


###################################################################################
# 训练过程
def train(train_loader, Encoder, Decoder, param, begin, iteration):
    Encoder.train()
    Decoder.train()
    size = len(train_loader)
    Encoder_optimizer = optim.Adam(Encoder.parameters(), lr=lr)
    Decoder_optimizer = optim.Adam(Decoder.parameters(), lr=lr * decoder_learning_ratio)
    for epoch in range(begin, iteration+begin):
        print("epoch", epoch, ":")
        sum_num = 0
        for post, res in train_loader:
            post = post.to(device)
            res = res.permute(1, 0).to(device)
            # Zero gradients
            Encoder_optimizer.zero_grad()
            Decoder_optimizer.zero_grad()
            # Forward pass through encoder
            encoder_outputs, encoder_hidden = Encoder(post)
            # print("encoder_outputs", encoder_outputs.size())
            # print("encoder_hidden", encoder_hidden.size())
            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.from_numpy(np.array([[0 for _ in range(param['batch_size'])]])).long().to(device)
            decoder_input = decoder_input.to(device)
            # print("decoder_input", decoder_input.size())
            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:param['decoder']['layer']]
            # Determine if we are using teacher forcing this iteration
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            # Forward batch of sequences through decoder one time step at a time
            sum_loss = 0
            if use_teacher_forcing:
                for t in range(param["seq_size"]):
                    decoder_output, decoder_hidden = Decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    # print("decoder_output", decoder_output.size())
                    # print("decoder_hidden", decoder_hidden.size())
                    # Teacher forcing: next input is current target
                    decoder_input = res[t].view(1, -1)
                    target = torch.zeros(param['batch_size'], param['vocab_size'])

                    # Calculate and accumulate loss
                    loss = NLLLoss(decoder_output, res[t])
                    sum_loss += loss
            else:
                for t in range(param["seq_size"]):
                    decoder_output, decoder_hidden = Decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    # No teacher forcing: next input is decoder's own current output
                    _, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(param['batch_size'])]]).long().to(device)
                    # Calculate and accumulate loss
                    loss = NLLLoss(decoder_output, res[t])
                    sum_loss += loss

            sum_loss.backward()
            # Clip gradients: gradients are modified in place
            _ = torch.nn.utils.clip_grad_norm_(Encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(Decoder.parameters(), clip)

            # Adjust model weights
            Encoder_optimizer.step()
            Decoder_optimizer.step()

            sum_num += 1
            if sum_num % 50 == 0:
                print("time:", sum_num, "/", size, " train loss:", sum_loss.cpu().detach().numpy() / sum_num)

        if epoch % 5 == 0:
            torch.save(Encoder, 'resource/replier/Encoder_{}.pkl'.format(epoch))
            torch.save(Decoder, 'resource/replier/Decoder_{}.pkl'.format(epoch))


###################################################################################
def main():
    param = load_param(param_path)
    embedding = Embedding(param['vocab_size'], param['embed_dim']).to(device)
    Encoder = EncoderRNN(batch_size=param['batch_size'], seq_size=param['seq_size'], embedding=embedding
                         , attn_dim=param['encoder']['attn_dim'], embed_dim=param['embed_dim']
                         , layer_size=param['encoder']['layer'], hidden_size=param['encoder']['hidden_size']
                         , dropout=param['encoder']['dropout']).to(device)
    Decoder = DecoderRNN(attn_model='dot', embedding=embedding, output_size=param['vocab_size']
                         , layer_size=param['decoder']['layer'], hidden_size=param['decoder']['hidden_size']
                         , dropout=param['decoder']['dropout']).to(device)
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
            train_x = np.load('resource/replier/train_x.npy', allow_pickle=True)
            train_y = np.load('resource/replier/train_y.npy', allow_pickle=True)
            test_x = np.load('resource/replier/test_x.npy', allow_pickle=True)
            test_y = np.load('resource/replier/test_y.npy', allow_pickle=True)
            print("old data get")
        except:
            print("no old data")
            train_x, train_y, test_x, test_y = data_loader.replier_maker(param['seq_size'], param['cut'])
            np.save('resource/replier/train_x.npy', train_x)
            np.save('resource/replier/train_y.npy', train_y)
            np.save('resource/replier/test_x.npy', test_x)
            np.save('resource/replier/test_y.npy', test_y)
    else:
        train_x, train_y, test_x, test_y = data_loader.replier_maker(param['seq_size'], param['cut'])
        np.save('resource/replier/train_x.npy', train_x)
        np.save('resource/replier/train_y.npy', train_y)
        np.save('resource/replier/test_x.npy', test_x)
        np.save('resource/replier/test_y.npy', test_y)

    if train_mode:
        train_x = torch.from_numpy(train_x).long()
        train_y = torch.from_numpy(train_y).long()
        train_set = dt.TensorDataset(train_x, train_y)
        train_loader = dt.DataLoader(dataset=train_set, batch_size=param["batch_size"], pin_memory=True
                                     , shuffle=True, num_workers=0, drop_last=True)
        if cont:
            try:
                data_loader.load_data()
                Encoder = torch.load('resource/replier/Encoder_{}.pkl'.format(time))
                Decoder = torch.load('resource/replier/Decoder_{}.pkl'.format(time))
                print("old model get")
                train(train_loader, Encoder, Decoder, param, time, train_num)
            except:
                print("no old model")
                train(train_loader, Encoder, Decoder, param, 0, train_num)
        else:
            train(train_loader, Encoder, Decoder, param, 0, train_num)
        data_loader.save_data()
        torch.save(Encoder, 'resource/replier/Encoder.pkl')
        torch.save(Encoder, 'resource/replier/Decoder.pkl')
        torch.save(embedding, 'resource/replier/embedding.pkl')


main()
