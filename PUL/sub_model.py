# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 13:09
# @Author  : Xiaoyu Xing
# @File    : wrapper_model.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Flatten(nn.Module):
    def __init__(self, shape):
        super(Flatten, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(-1, self.shape)


class TimeDistributed(nn.Module):
    def __init__(self, module, char2Idx):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.char2Idx = char2Idx

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        if len(ids.size()) <= 2:
            return self.module(x)
        t, n = ids.size(0), ids.size(1)
        # merge batch and seq dimensions
        x_reshape = ids.contiguous().view(t * n, ids.size(2))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, y.size()[1])
        return y, sortedLen, reversedIndices

    def embedding_with_padding(self, x, maxLength, length):
        ids = []
        for s in x:
            charID = []
            for cid in s:
                temp = []
                for id in cid:
                    temp.append(id)
                charID.append(temp)
            padding_vector = [self.char2Idx["PADDING"] for i in range(52)]
            charID += [padding_vector for _ in range(maxLength - len(charID))]
            ids.append(charID)
        ids = Variable(torch.LongTensor(ids))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        ids = ids[indices]
        return ids.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()


class CharCNN(nn.Module):
    def __init__(self, char2Idx):
        super(CharCNN, self).__init__()
        self.char2Idx = char2Idx
        self.embedding = nn.Embedding(len(self.char2Idx), 30)  # b*52*30
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.dropout1 = nn.Dropout(0.5)
        self.conv1 = nn.Sequential(
            nn.Conv1d(30, 30, 3, 1, 1),  # b*30*52
            nn.Tanh(),
            nn.MaxPool1d(52),  # b*30*1
            Flatten(30)
        )
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.embedding(x)
        dropout = self.dropout1(embedding)
        dropout = dropout.permute(0, 2, 1)
        covout = self.conv1(dropout)
        res = self.dropout2(covout)
        return res


class CaseNet(nn.Module):
    def __init__(self, caseEmbeddings, case2Idx):
        super(CaseNet, self).__init__()
        self.caseEmbeddings = caseEmbeddings
        self.case2Idx = case2Idx
        self.embedding = nn.Embedding(caseEmbeddings.shape[0], self.caseEmbeddings.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(self.caseEmbeddings))
        self.embedding.weight.requires_grad = False

        # self.dense = nn.Linear(self.caseEmbeddings.shape[1], self.caseEmbeddings.shape[1])

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        embeddings = self.embedding(ids)
        # embeddings = self.dense(embeddings)
        return embeddings, sortedLen, reversedIndices

    def embedding_with_padding(self, x, maxLength, length):
        ids = []
        for s in x:
            caseID = []
            for id in s:
                caseID.append(id)
            caseID += [self.case2Idx["PADDING_TOKEN"] for _ in range(maxLength - len(caseID))]
            ids.append(caseID)
        ids = Variable(torch.LongTensor(ids))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        ids = ids[indices]
        return ids.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()

class WordNet(nn.Module):
    def __init__(self, wordEmbeddings, word2Idx):
        super(WordNet, self).__init__()
        self.wordEmbeddings = wordEmbeddings
        self.word2Idx = word2Idx

        self.embedding = nn.Embedding(self.wordEmbeddings.shape[0], self.wordEmbeddings.shape[1])
        if wordEmbeddings is None:
            self.embedding.weight.data.normal_(0, 0.01)
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(self.wordEmbeddings))
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        embeddings = self.embedding(ids)
        return embeddings, sortedLen, reversedIndices

    def embedding_with_padding(self, x, maxLength, length):
        ids = []
        for s in x:
            sentenceID = []
            for id in s:
                sentenceID.append(id)
            sentenceID += [self.word2Idx["PADDING_TOKEN"] for _ in range(maxLength - len(sentenceID))]
            ids.append(sentenceID)
        ids = Variable(torch.LongTensor(ids))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        ids = ids[indices]
        return ids.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        return ids, sortedLen, reversedIndices

    def embedding_with_padding(self, feature, maxLength, length):
        feature_ = []
        for sf in feature:
            f = []
            for wf in sf:
                f.append(wf)
            pad = np.zeros(12, dtype=int).tolist()
            f += [pad for _ in range(maxLength - len(f))]
            feature_.append(f)
        feature_ = Variable(torch.LongTensor(feature_))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        feature_ = feature_[indices]
        return feature_.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            target_size: int, target size
            use_cuda: bool, 是否使用gpu, default is True
            average_batch: bool, loss是否作平均, default is True
        """
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if not hasattr(self, 'average_batch'):
            self.__setattr__('average_batch', True)
        if not hasattr(self, 'use_cuda'):
            self.__setattr__('use_cuda', True)

        # init transitions
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        init_transitions = torch.zeros(self.target_size + 2, self.target_size + 2)
        init_transitions[:, self.START_TAG_IDX] = -1000.
        init_transitions[self.END_TAG_IDX, :] = -1000.
        if self.use_cuda:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = list()
        partition_history = list()

        # mask = 1 + (-1) * mask
        mask = (1 - mask.long()).bool()
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()

        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
                      self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = Variable(torch.zeros(batch_size, tag_size)).long()
        if self.use_cuda:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)

        end_energy = torch.gather(end_transition, 1, end_ids)

        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score
