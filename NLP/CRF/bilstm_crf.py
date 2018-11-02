# -*- coding: utf-8 -*-
# @Time    : 2018/11/1 13:14
# @Author  : uhauha2929
import torch
import torch.nn as nn
import torch.nn.functional as F


# this is a very simple demo, and there may be some minor issues.
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
# https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/


def prepare_sequence(seq, word2ix):
    idxs = [word2ix[w] for w in seq]
    return torch.LongTensor(idxs)


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sent):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sent).view(len(sent), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sent), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _real_path_score(self, feats, tags):
        score = torch.zeros(1)
        for i in range(len(tags) - 1):
            score = score + self.transitions[tags[i], tags[i + 1]] + feats[i][tags[i]]
        score = score + feats[-1][tags[-1]]
        return score

    def _total_path_score(self, feats):
        prev = feats[0]
        for obs in feats[1:]:
            prev = prev.expand(self.tagset_size, self.tagset_size).t()
            obs = obs.expand(self.tagset_size, self.tagset_size)
            scores = prev + obs + self.transitions
            prev = torch.log(torch.sum(torch.exp(scores), 0))
        score = torch.log(torch.sum(torch.exp(prev)))
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
            path.append(pre_tag.item())
            cur_tag = pre_tag
        path.reverse()
        return path

    def forward(self, sent):
        lstm_feats = self._get_lstm_features(sent)
        tag_seq = self._decode(lstm_feats)
        return tag_seq


EMBEDDING_DIM = 100
HIDDEN_DIM = 100


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

    tag_to_ix = {"B": 0, "I": 1, "O": 2}
    word2ix = {}
    for sentence, tags in train_data:
        for word in sentence:
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    model = BiLSTM_CRF(len(word2ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)

    with torch.no_grad():
        precheck_sent = prepare_sequence(test_data[0][0], word2ix)
        precheck_tags = [tag_to_ix[t] for t in test_data[0][1]]
        print('before training:\n', model(precheck_sent))
        print('real target:\n', precheck_tags)

    for _ in range(100):
        for sentence, tags in train_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word2ix)
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])
            loss = model.neg_log_loss(sentence_in, targets)
            # print(loss.item())
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        precheck_sent = prepare_sequence(test_data[0][0], word2ix)
        print('after training:\n', model(precheck_sent))


if __name__ == '__main__':
    main()
