import json
import random
import numpy as np
from tqdm import*
import jieba

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
time = 400  # 从哪个模型开始
train_num = 30  # 训练轮数
train_mode = False  # 训练/使用
teacher_forcing_ratio = 0.0  # 用真正回答作为Encoder输入的比例
lr = 0.0001
weight_decay = 0.00001
decoder_learning_ratio = 5.0
clip = 50.0  # 梯度裁剪系数


# 只计算<pad>之外的损失和正确率
def NLLLoss(inp, target, pad):
    mask = torch.eq(torch.eq(target, pad), 0)
    nTotal = mask.sum()
    _, predict = inp.topk(1)
    ac = torch.eq(predict, target).float()
    ac = ac.masked_select(mask)
    ac = ac.sum() / nTotal.item()
    ac = ac.to(device)
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, ac, nTotal.item()


###################################################################################
# 使用过程
def play(Encoder, Decoder, param, vocab, re_vocab):
    post = input("请输入：")
    Encoder.eval()
    Decoder.eval()
    # quit退出
    while post != 'quit':
        post = list(jieba.cut(post))
        # print("post:", post)
        post = [vocab[word] for word in post if word in vocab.keys()]
        # print("post:", post)
        if len(post) >= param['seq_size']:
            post = post[:param['seq_size'] - 1]
        post.append(vocab['<EOS>'])
        while len(post) < param['seq_size']:
            post.append(vocab['<PAD>'])
        post = np.array(post)
        post = torch.from_numpy(post).long().to(device).unsqueeze(0)
        post = post.repeat(param['batch_size'], 1)
        encoder_outputs, encoder_hidden = Encoder(post)
        decoder_input = torch.from_numpy(np.array([[vocab['<SOS>'] for _ in range(param['batch_size'])]])).long().to(
            device)
        decoder_input = decoder_input.to(device)
        decoder_hidden = encoder_hidden[:param['decoder']['layer']]
        predict = []
        for t in range(param["seq_size"]):
            decoder_output, decoder_hidden = Decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(param['batch_size'])]]).long().to(device)
            predict = predict + [
                [topi[i][0].cpu().numpy().tolist() for i in range(param['batch_size'])]]
        predict = list(map(list, zip(*predict)))[0]
        res = ''
        for word in predict:
            if word == vocab['<EOS>']:
                break
            res += re_vocab[str(word)]
        print("回复:", res)
        post = input("请输入：")


###################################################################################
# 测试过程
def test(test_loader, Encoder, Decoder, param, vocab, re_vocab):
    Encoder.eval()
    Decoder.eval()
    size = len(test_loader)
    all_num = 0
    all_loss = 0
    all_ac = 0
    for post, res in test_loader:
        post = post.to(device)
        res = res.permute(1, 0).to(device)
        encoder_outputs, encoder_hidden = Encoder(post)
        decoder_input = torch.from_numpy(np.array([[vocab['<SOS>'] for _ in range(param['batch_size'])]])).long().to(
            device)
        decoder_input = decoder_input.to(device)
        decoder_hidden = encoder_hidden[:param['decoder']['layer']]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        sum_loss = 0
        sum_num = 0
        sum_ac = 0
        predict = []
        for t in range(param["seq_size"]):
            decoder_output, decoder_hidden = Decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(param['batch_size'])]]).long().to(device)
            predict = predict + [[re_vocab[str(topi[i][0].cpu().numpy().tolist())] for i in range(param['batch_size'])]]
            target = res[t].view(-1, 1)
            loss, ac, num = NLLLoss(decoder_output, target, vocab['<PAD>'])
            sum_loss += loss * num
            sum_num += num
            sum_ac += ac.item() * num
            all_loss += sum_loss.cpu().detach().numpy() / sum_num
            all_ac += sum_ac / sum_num
            all_num += 1
        predict = list(map(list, zip(*predict)))[0]
        if all_num % 50 == 0:
            res = res.permute(1, 0).cpu().detach().numpy().tolist()
            post = post.cpu().detach().numpy().tolist()
            res = [[re_vocab[str(index)] for index in seq] for seq in res]
            post = [[re_vocab[str(index)] for index in seq] for seq in post]
            print(all_num)
            print("res", res[0])
            print("post", post[0])
            print("predict", predict)
            print("test loss:", all_loss / all_num, "ac:", all_ac / all_num)
    res = res.permute(1, 0).cpu().detach().numpy().tolist()
    post = post.cpu().detach().numpy().tolist()
    res = [[re_vocab[str(index)] for index in seq] for seq in res]
    post = [[re_vocab[str(index)] for index in seq] for seq in post]
    print("res", res[0])
    print("post", post[0])
    print("predict", predict)
    print("test loss:", all_loss / all_num, "ac:", all_ac / all_num)

    Encoder.train()
    Decoder.train()


###################################################################################
# 训练过程
def train(train_loader, test_loader, Encoder, Decoder, param, begin, iteration, vocab, re_vocab):
    Encoder.train()
    Decoder.train()
    size = len(train_loader)
    Encoder_optimizer = optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
    Decoder_optimizer = optim.Adam(Decoder.parameters(), lr=lr * decoder_learning_ratio, weight_decay=weight_decay)
    for epoch in range(begin, iteration+begin):
        all_num = 0
        all_loss = 0
        all_ac = 0
        for post, res in train_loader:
            post = post.to(device)
            res = res.permute(1, 0).to(device)
            Encoder_optimizer.zero_grad()
            Decoder_optimizer.zero_grad()
            # encoder过程
            encoder_outputs, encoder_hidden = Encoder(post)
            decoder_input = torch.from_numpy(np.array([[vocab['<SOS>'] for _ in range(param['batch_size'])]])).long().to(device)
            decoder_input = decoder_input.to(device)
            decoder_hidden = encoder_hidden[:param['decoder']['layer']]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            sum_loss = 0
            print_losses = 0
            sum_num = 0
            sum_ac = 0
            predict = []
            # teacher_forcing下用真实值代替decoder_output
            if use_teacher_forcing:
                for t in range(param["seq_size"]):
                    # decoder过程
                    decoder_output, decoder_hidden = Decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    decoder_input = res[t].view(1, -1)
                    _, topi = decoder_output.topk(1)
                    predict = predict + [[re_vocab[str(topi[i][0].cpu().numpy().tolist())] for i in range(param['batch_size'])]]
                    target = res[t].view(-1, 1)
                    loss, ac, num = NLLLoss(decoder_output, target, vocab['<PAD>'])
                    sum_loss += loss
                    sum_num += num
                    sum_ac += ac.item() * num
                    print_losses += loss.item() * num
            else:
                for t in range(param["seq_size"]):
                    # decoder过程
                    decoder_output, decoder_hidden = Decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    _, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(param['batch_size'])]]).long().to(device)
                    target = res[t].view(-1, 1)
                    loss, ac, num = NLLLoss(decoder_output, target, vocab['<PAD>'])
                    predict = predict + [[re_vocab[str(topi[i][0].cpu().numpy().tolist())] for i in range(param['batch_size'])]]
                    # 结果都是以词为单位平均
                    sum_loss += loss
                    sum_num += num
                    sum_ac += ac.item() * num
                    print_losses += loss.item() * num
            sum_loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(Encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(Decoder.parameters(), clip)

            Encoder_optimizer.step()
            Decoder_optimizer.step()
            all_loss += print_losses / sum_num
            all_ac += sum_ac / sum_num
            all_num += 1
            if all_num % 20 == 0:
                print("time:", all_num, "/", size, " train loss:", all_loss / all_num, "ac:", all_ac / all_num)
                res = res.permute(1, 0).cpu().detach().numpy().tolist()
                post = post.cpu().detach().numpy().tolist()
                res = [[re_vocab[str(index)] for index in seq] for seq in res]
                post = [[re_vocab[str(index)] for index in seq] for seq in post]
                print("res", res[0])
                print("post", post[0])
                print("predict", list(map(list, zip(*predict)))[0])
        print("epoch:", epoch, " train loss:", all_loss / all_num, "ac:", all_ac / all_num)
        res = res.permute(1, 0).cpu().detach().numpy().tolist()
        post = post.cpu().detach().numpy().tolist()
        res = [[re_vocab[str(index)] for index in seq] for seq in res]
        post = [[re_vocab[str(index)] for index in seq] for seq in post]
        print("res", res[0])
        print("post", post[0])
        print("predict", list(map(list, zip(*predict)))[0])
        test(test_loader, Encoder, Decoder, param, vocab, re_vocab)

        # 十次训练存储一次
        if epoch != begin and epoch % 10 == 0:
            torch.save(Encoder, 'model/Encoder_{}.pkl'.format(epoch))
            torch.save(Decoder, 'model/Decoder_{}.pkl'.format(epoch))
            torch.save(Encoder.embedding, 'model/embedding_{}.pkl'.format(epoch))


###################################################################################
def main():
    param = load_param(param_path)
    data_loader = GetData()  # 数据处理类
    # 尝试获取旧数据
    try:
        data_loader.load_data()
        print("old data_loader get")
    except:
        print("no old data_loader")
        data_loader = GetData()
        data_loader.make_data(stop_path=param['stop_words'], path=param['data_path'])
        data_loader.save_data()

    try:
        train_x = np.load('resource/train_x.npy', allow_pickle=True)
        train_y = np.load('resource/train_y.npy', allow_pickle=True)
        test_x = np.load('resource/test_x.npy', allow_pickle=True)
        test_y = np.load('resource/test_y.npy', allow_pickle=True)
        print("old data get")
    except:
        print("no old data")
        train_x, train_y, test_x, test_y = data_loader.replier_maker(param['seq_size'], param['cut'])
        np.save('resource/train_x.npy', train_x)
        np.save('resource/train_y.npy', train_y)
        np.save('resource/test_x.npy', test_x)
        np.save('resource/test_y.npy', test_y)

    embedding = Embedding(data_loader.vocab_size, param['embed_dim']).to(device)
    Encoder = EncoderRNN(batch_size=param['batch_size'], seq_size=param['seq_size'], embedding=embedding
                         , embed_dim=param['embed_dim'], layer_size=param['encoder']['layer']
                         , hidden_size=param['encoder']['hidden_size'], dropout=param['encoder']['dropout']).to(device)
    Decoder = DecoderRNN(embedding=embedding, output_size=data_loader.vocab_size
                         , layer_size=param['decoder']['layer'], hidden_size=param['decoder']['hidden_size']
                         , dropout=param['decoder']['dropout']).to(device)

    test_x = torch.from_numpy(test_x).long()
    test_y = torch.from_numpy(test_y).long()
    test_set = dt.TensorDataset(test_x, test_y)
    test_loader = dt.DataLoader(dataset=test_set, batch_size=param["batch_size"], pin_memory=True, shuffle=True
                                , num_workers=0, drop_last=True)
    # 训练过程
    if train_mode:
        train_x = torch.from_numpy(train_x).long()
        train_y = torch.from_numpy(train_y).long()
        train_set = dt.TensorDataset(train_x, train_y)
        train_loader = dt.DataLoader(dataset=train_set, batch_size=param["batch_size"], pin_memory=True
                                     , shuffle=True, num_workers=0, drop_last=True)
        try:
            if time > 0:
                Encoder = torch.load('model/Encoder_{}.pkl'.format(time))
                Decoder = torch.load('model/Decoder_{}.pkl'.format(time))
            else:
                Encoder = torch.load('model/Encoder.pkl')
                Decoder = torch.load('model/Decoder.pkl')
            Decoder.embedding = Encoder.embedding
            print("old model get")
            train(train_loader, test_loader, Encoder, Decoder, param, time, train_num, data_loader.vocab, data_loader.re_vocab)
        except:
            print("no old model")
            train(train_loader, test_loader, Encoder, Decoder, param, 0, train_num, data_loader.vocab, data_loader.re_vocab)

        data_loader.save_data()
        torch.save(Encoder, 'model/Encoder.pkl')
        torch.save(Decoder, 'model/Decoder.pkl')
        torch.save(embedding, 'model/embedding.pkl')
    # 使用模型
    else:
        if time > 0:
            Encoder = torch.load('model/Encoder_{}.pkl'.format(time))
            Decoder = torch.load('model/Decoder_{}.pkl'.format(time))
        else:
            Encoder = torch.load('model/Encoder.pkl')
            Decoder = torch.load('model/Decoder.pkl')
        # test(test_loader, Encoder, Decoder, param, data_loader.vocab, data_loader.re_vocab)
        play(Encoder, Decoder, param, data_loader.vocab, data_loader.re_vocab)


main()
