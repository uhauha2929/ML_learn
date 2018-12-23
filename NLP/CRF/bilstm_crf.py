# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 13:14
# @Author  : uhauha2929
import torch
import torch.nn as nn
import torch.nn.functional as F


# this is a very simple demo, and there may be some minor issues.
# Reference：https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/


def prepare_sequence(seq, word2ix):
    idxs = [word2ix['SOS']] + [word2ix[w] for w in seq] + [word2ix['EOS']]
    return torch.LongTensor(idxs)


def prepare_tags(tags, tag2ix):
    idxs = [tag2ix['S']] + [tag2ix[t] for t in tags] + [tag2ix['E']]
    return torch.LongTensor(idxs)


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # 为了简单起见，初始化全为0，其实我们知道结束标签到任意标签、任意标签到开始标签的转移概率都为0
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sent):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sent).view(len(sent), 1, -1)  # 这里输入的是一个句子
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sent), self.hidden_dim)
        return self.hidden2tag(lstm_out)

    def _real_path_score(self, feats, tags):
        score = feats[0][tags[0]]
        for i in range(len(tags) - 1):
            score = score + self.transitions[tags[i], tags[i + 1]] + feats[i + 1][tags[i + 1]]
        return score

    def _total_path_score(self, feats):
        prev = feats[0]
        for obs in feats[1:]:
            prev = prev.expand(self.tagset_size, self.tagset_size).t()
            obs = obs.expand(self.tagset_size, self.tagset_size)
            scores = prev + obs + self.transitions
            prev = torch.logsumexp(scores, 0)
        score = torch.logsumexp(prev, -1)
        return score

    def neg_log_loss(self, sent, tags):
        feats = self._get_lstm_features(sent)
        real_score = self._real_path_score(feats, tags)
        total_score = self._total_path_score(feats)
        return -(real_score - total_score)

    def _decode(self, feats):
        prev = feats[0]
        alpha0 = []
        alpha1 = []
        for obs in feats[1:]:
            prev = prev.expand(self.tagset_size, self.tagset_size).t()
            obs = obs.expand(self.tagset_size, self.tagset_size)
            scores = prev + obs + self.transitions
            prev, prev_ix = torch.max(scores, 0)
            alpha0.append(prev)
            alpha1.append(prev_ix)

        path = []
        cur_tag = torch.argmax(alpha0[-1])
        path.append(cur_tag.item())
        for indexes in reversed(alpha1):
            pre_tag = indexes[cur_tag]
            path.insert(0, pre_tag.item())
            cur_tag = pre_tag
        return path

    def forward(self, sent):
        with torch.no_grad():
            lstm_feats = self._get_lstm_features(sent)
            tag_seq = self._decode(lstm_feats)
            return tag_seq


EMBEDDING_DIM = 100
HIDDEN_DIM = 128


def main():
    train_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    test_data = [(
        "apple corporation reported that georgia tech made the wall street journal in georgia".split(),
        "B I O O B I O B I I I O B".split()
    )]

    tag2ix = {"S": 0, "E": 1, "B": 2, "I": 3, "O": 4}  # S表示句子开始符的标签，E表示句子结束符的标签
    word2ix = {"SOS": 0, "EOS": 1}  # SOS表示句子开始，EOS表示句子结束
    for sentence, tags in train_data:
        for word in sentence:
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    model = BiLSTM_CRF(len(word2ix), len(tag2ix), EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    precheck_sent = prepare_sequence(test_data[0][0], word2ix)
    precheck_tags = prepare_tags(test_data[0][1], tag2ix)
    print('before training:\n', model(precheck_sent))
    print('real target:\n', precheck_tags.tolist())

    for _ in range(5):
        for sentence, tags in train_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word2ix)
            targets = prepare_tags(tags, tag2ix)
            # 为了简单起见，这里训练时每次输入一个句子
            loss = model.neg_log_loss(sentence_in, targets)
            print(loss.item())
            loss.backward()
            optimizer.step()

    precheck_sent = prepare_sequence(test_data[0][0], word2ix)
    print('after training:\n', model(precheck_sent))


if __name__ == '__main__':
    main()
