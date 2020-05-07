"""
@author yy
@reference https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
@date 2020.5.5
"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
from copy import deepcopy
import random
import numpy as np
import time
import math
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 20
BATCH_SIZE = 32
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5


def seq2tensor(seq):
    seq.append(EOS_token)
    return torch.tensor(seq, dtype=torch.long, device=device).view(-1, 1)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class DataInfo:
    def __init__(self, word2idx, idx2word, qa_pairs):
        self.idx2word = ['SOS_token', 'EOS_token'] + idx2word
        self.word2idx = deepcopy(word2idx)
        for item in self.word2idx.keys():
            self.word2idx[item] += 2
        self.word2idx['SOS_token'] = 0
        self.word2idx['EOS_token'] = 1
        self.qa_pairs = deepcopy(qa_pairs)
        for (i, item) in enumerate(self.qa_pairs):
            self.qa_pairs[i] = [
                list(map(lambda t: t + 2, self.qa_pairs[i][0])),
                list(map(lambda t: t + 2, self.qa_pairs[i][1]))
            ]
        # check
        for (i, word) in enumerate(self.idx2word):
            assert self.word2idx[word] == i
        pairs_num = len(self.qa_pairs)
        self.valid_qa_pairs = self.qa_pairs[0:pairs_num // 10]
        self.train_qa_pairs = self.qa_pairs[pairs_num // 10:]
        print("Build data complete!")

    def change_idxs_to_sentence(self, words):
        ans = []
        for word in words:
            ans.append(self.idx2word[word])
        return ''.join(ans)

    def print_sample(self):
        t = random.randint(0, len(self.qa_pairs) - 1)
        print("Chosen t: %d" % t)
        print("Q: %s\nA: %s" % (self.change_idxs_to_sentence(self.qa_pairs[t][0]),
                                self.change_idxs_to_sentence(self.qa_pairs[t][1])))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_data, hidden):
        embedded = self.embedding(input_data).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        output = self.embedding(input_data).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(in_tensor, out_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    in_length = in_tensor.size(0)
    out_length = out_tensor.size(0)

    loss = 0

    for idx in range(in_length):
        encoder_out, encoder_hidden = encoder(
            in_tensor[idx], encoder_hidden
        )

    decoder_in = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for idx in range(out_length):
            decoder_out, decoder_hidden = decoder(
                decoder_in, decoder_hidden
            )
            loss += criterion(decoder_out, out_tensor[idx])
            decoder_in = out_tensor[idx]
    else:
        for idx in range(out_length):
            decoder_output, decoder_hidden = decoder(
                decoder_in, decoder_hidden
            )
            topv, topi = decoder_output.topk(1)
            decoder_in = topi.squeeze().detach()
            loss += criterion(decoder_output, out_tensor[idx])
            if decoder_in.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / out_length


def trainIters(encoder, decoder, n_iters, data,
               print_every=1000, learning_rate=0.005):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [(lambda item: [seq2tensor(item[0]), seq2tensor(item[1])])
                      (random.choice(data.train_qa_pairs))
                      for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


def evaluate(encoder, decoder, sentence, data, max_length=MAX_LENGTH):
    with torch.no_grad():
        loss = 0
        input_tensor = seq2tensor(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(data.idx2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, loss


def evaluateRandomly(encoder, decoder, data, n=10):
    for i in range(n):
        pair = random.choice(data.valid_qa_pairs)
        print("test case %d")
        print("\tQ: %s" % data.change_idxs_to_sentence(pair[0]))
        print("\tstd A: %s" % data.change_idxs_to_sentence(pair[1]))
        print("\tout A: %s" % data.change_idxs_to_sentence(
            evaluate(encoder, decoder, pair[0], data)[0]
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=None, help="the path of encoder")
    parser.add_argument("--decoder", default=None, help="the path of decoder")
    args = parser.parse_args()
    with open("../data/preprocessed_data/chosen_word2id.pkl", "rb") as f:
        chosen_word2id = pickle.load(f)
    with open("../data/preprocessed_data/chosen_word_list.pkl", "rb") as f:
        word_list = pickle.load(f)
    with open("../data/preprocessed_data/qa_pair.pkl", "rb") as f:
        qa_pairs = pickle.load(f)

    if (args.encoder is not None) and (args.decoder is not None):
        data = DataInfo(chosen_word2id, word_list, qa_pairs)
        encoder = torch.load(args.encoder, map_location=device)
        decoder = torch.load(args.decoder, map_location=device)
        evaluateRandomly(encoder, decoder, data)
        return None

    # prepare data
    data = DataInfo(chosen_word2id, word_list, qa_pairs)
    data.print_sample()
    encoder = EncoderRNN(len(data.idx2word), 300).to(device)
    decoder = DecoderRNN(len(data.idx2word), 300).to(device)
    trainIters(encoder, decoder, 1000000, data, 10000)
    torch.save(encoder, '../models/encoder.pkl')
    torch.save(decoder, '../models/decoder.pkl')
    evaluateRandomly(encoder, decoder, data)


if __name__ == "__main__":
    print("Using device %s" % device)
    main()
