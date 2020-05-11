"""
@author yy
@date 2020.5.6
"""

import argparse
import math
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

from utils import Constants, printInfo, cal_loss, cal_performance, \
    patch_src, patch_trg
from transformer.Models import Transformer
from transformer.OptimUtils import ScheduleOptim
import time


def train_epoch(model, training_data, optimizer, args, device, smoothing):
    model.train()
    total_loss, total_words, total_correct = 0, 0, 0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        src_seq = patch_src(batch.src, args.pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device),
                            patch_trg(batch.trg, args.pad_idx))
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)
        loss, n_correct, n_word = cal_performance(
            pred, gold, args.pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()
        total_words += n_word
        total_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/total_words
    accuracy = total_correct/total_words
    return loss_per_word, accuracy


def eval_epoch(model, valid_data, device, args):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(valid_data, mininterval=2, desc=desc, leave=False):
            src_seq = patch_src(batch.src, args.pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg,
                                                                  args.pad_idx))
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, args.pad_idx, smoothing=False)
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, valid_data, optimizer, device, args):
    log_train_file, log_valid_file = None, None
    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'
        printInfo('Training performance will be written to file: %s and %s' % (
            log_train_file, log_valid_file
        ))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, accu, start_time):
        print('  - \033[32m{header:12}\033[0m ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(header=f"({header})",
                                                 ppl=math.exp(min(loss, 100)),
                                                 accu=100 * accu,
                                                 elapse=(time.time() - start_time) / 60))
    valid_losses = []
    for epoch_i in range(args.epoch):
        printInfo('', 'Epoch %d' % epoch_i)
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, args, device, args.label_smoothing
        )
        print_performances('Training', train_loss, train_accu, start)
        start = time.time()
        valid_loss, valid_accu = eval_epoch(
            model, valid_data, device, args
        )
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses.append(valid_loss)
        checkpoint = {'epoch': epoch_i, 'settings': args, 'model': model.state_dict()}
        model_name = 'models/transformer_best' + '.chkpt'
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, model_name)
            print('    - [Info] The checkpoint file has been updated.')
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, \
                    open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_pkl', default=None)
    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=128000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-log', default="./log/log_%s" % time.strftime(
        "%Y-%m-%d-%H:%M:%S", time.localtime()
    ))
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.embed_dim = args.d_model
    device = torch.device('cuda' if args.cuda else 'cpu')
    printInfo("Using device %s" % device)

    data = pickle.load(open(args.data_pkl, "rb"))
    args.max_seq_len = data["max_len"]
    args.pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    args.vocab_size = len(data['vocab']['src'].vocab)
    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}
    train_ds = Dataset(examples=data['train'], fields=fields)
    val_ds = Dataset(examples=data['valid'], fields=fields)
    printInfo("Training data: %d    Validation data: %d" %
              (len(data['train']), len(data['valid'])))
    train_iterator = BucketIterator(train_ds, batch_size=args.batch_size,
                                    device=device, train=True)
    val_iterator = BucketIterator(val_ds, batch_size=args.batch_size, device=device)

    printInfo("Selected parameters: ")
    print(args)

    # 训练开始
    transformer = Transformer(
        args.vocab_size,
        in_pad_idx=args.pad_idx,
        out_pad_idx=args.pad_idx,
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        d_inner=args.d_inner,
        d_k=args.d_k,
        d_v=args.d_v,
        n_head=args.n_head,
        n_layers=args.n_layers,
        dropout=args.dropout,
        target_embedding_output_weight_sharing=True,
        embedding_input_target_weight_sharing=True,
    ).to(device)

    optimizer = ScheduleOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, args.d_model, args.n_warmup_steps
    )

    train(transformer, train_iterator, val_iterator, optimizer, args)


if __name__ == "__main__":
    main()
