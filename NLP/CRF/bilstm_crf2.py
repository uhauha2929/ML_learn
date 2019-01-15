# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 16:47
# @Author  : uhauha2929
import itertools
import os
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from tqdm import tqdm

TRAIN_DATA = '/home/yzhao/data/icwb2-data/training/pku_training.utf8'
TEST_DATA = '/home/yzhao/data/icwb2-data/testing/pku_test.utf8'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def text2seq(text, char2id):
    # text是分过词的句子，要去除空格
    return [char2id.get(c, 0) for c in ''.join(text.split())]


def text2tag(text, tag2id):
    # text是分过词的句子
    tags = []
    for word in text.split():
        if len(word) == 1:
            tags.append(tag2id['s'])
        elif len(word) == 2:
            tags.extend([tag2id['b'], tag2id['e']])
        else:
            tags.extend([tag2id['b']] + [tag2id['m']] * (len(word) - 2) + [tag2id['e']])
    return tags


def padding(l, pad_id=0):
    # 输入：[[1, 1, 1], [2, 2], [3]]
    # 返回：[(1, 2, 3), (1, 2, 0), (1, 0, 0)] 返回已经是转置后的 [L, B]
    return list(itertools.zip_longest(*l, fillvalue=pad_id))


def binary_matrix(l, pad_id=0):
    # 将targets里非pad部分标记为1，pad部分标记为0
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == pad_id:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def data_generator(raw_data, batch_size, char2id, tag2id):
    X, Y = [], []
    for i, s in enumerate(raw_data, 1):
        x = text2seq(s, char2id)
        if len(x) > 0:
            X.append(x)
            Y.append(text2tag(s, tag2id))
        if len(X) == batch_size or i == len(raw_data):
            X.sort(key=lambda x: len(x), reverse=True)
            X = padding(X)
            mask = binary_matrix(X)
            Y.sort(key=lambda x: len(x), reverse=True)
            Y = padding(Y)
            yield torch.tensor(X, dtype=torch.long).to(DEVICE), \
                  torch.tensor(Y, dtype=torch.long).to(DEVICE), \
                  torch.tensor(mask, dtype=torch.uint8).to(DEVICE)
            X, Y = [], []


class BLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size + 1, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, tag_size)
        self.crf = CRF(tag_size)

    def _get_features(self, x, mask=None):
        x = self.embedding(x)  # [L, B, E]
        if mask is None:
            x, _ = self.lstm(x)  # [L, B, 2H]
        else:
            lengths = [torch.nonzero(mask[:, i]).size(0) for i in range(mask.size(1))]
            packed = pack_padded_sequence(x, lengths)
            x, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(x)
        x = F.selu(self.fc(x))  # [L, B, T]
        return x

    def get_loss(self, x, y, mask=None):
        x = self._get_features(x, mask)
        loss = self.crf(x, y, mask=mask)
        return -loss  # 负对数似然

    def decode(self, x, mask=None):
        x = self._get_features(x)  # [L, B, 2H]
        x = self.crf.decode(x, mask)
        return x


def single_cut(text, model, char2id, tag2id):
    seq = text2seq(text, char2id)
    x = torch.tensor(seq, dtype=torch.long).view(-1, 1).to(DEVICE)
    tags = model.decode(x)[0]
    result = []
    for i, tag in enumerate(tags):
        result.append(text[i])
        if tag == tag2id['s'] or tag == tag2id['e']:
            result.append(' ')
    return ''.join(result)


def train(model, optimizer, trains, char2id, tag2id):
    for _ in range(10):
        bar = tqdm(trains)
        for X, Y, mask in data_generator(bar, 32, char2id, tag2id):
            model.zero_grad()
            loss = model.get_loss(X, Y, mask)
            bar.set_description('loss:{:.4f}'.format(loss.item()))
            loss.backward()
            optimizer.step()

        text = '忠于祖国，忠于中国共产党，有坚定的革命理想、信念，全心全意为人民服务，自觉献身国防事业。'
        seg = single_cut(text, model, char2id, tag2id)
        print(seg)


# def cut_sent(para):
#     para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
#     para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
#     para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
#     para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
#     para = para.rstrip()
#     return para.split("\n")


def generate_test_file(model, tests, char2id, tag2id):
    output = open('test_segmentation.utf8', 'at', encoding='utf-8')
    for line in tqdm(tests):
        line = line.strip()
        if len(line) > 0:
            res = single_cut(line, model, char2id, tag2id)
            output.write(res + '\n')
    output.close()


def main():
    trains = open(TRAIN_DATA, 'rt', encoding='utf-8').readlines()
    counter = Counter(''.join(trains))
    char2id = {c: i for i, (c, _) in enumerate(counter.most_common(None), 1)}  # 0 for pad(mask)
    tag2id = {'s': 0, 'b': 1, 'm': 2, 'e': 3}
    model = BLSTM_CRF(len(char2id), len(tag2id), 128, 128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists('pku.pt'):
        model.load_state_dict(torch.load('pku.pt'))
    else:
        train(model, optimizer, trains, char2id, tag2id)
        torch.save(model.state_dict(), 'pku.pt')

    tests = open(TEST_DATA, 'rt', encoding='utf-8').readlines()
    generate_test_file(model, tests, char2id, tag2id)


if __name__ == '__main__':
    main()
