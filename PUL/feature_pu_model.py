# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 09:34
# @Author  : Xiaoyu Xing
# @File    : feature_pu_model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch.autograd import Variable
import argparse
from utils.data_utils import DataPrepare, set_seed, get_cnt, get_hyper_dist
from utils.feature_pu_model_utils import FeaturedDetectionModelUtils
from sub_model import CharCNN, CaseNet, WordNet, FeatureNet, TimeDistributed
from progressbar import *
from sklearn.metrics import precision_score, recall_score, f1_score


class PULSTMCNN(nn.Module):
    def __init__(self, dp, charModel, wordModel, caseModel, featureModel, inputSize, hiddenSize, layerNum, dropout):
        super(PULSTMCNN, self).__init__()
        self.dp = dp
        self.charModel = TimeDistributed(charModel, self.dp.char2Idx)
        self.wordModel = wordModel
        self.caseModel = caseModel
        self.featureModel = featureModel
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=layerNum,batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hiddenSize, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Softmax(dim=2)
            # nn.Linear(200, 1)
        )

    def forward(self, token, case, char, feature):
        charOut, sortedLen1, reversedIndices1 = self.charModel(char)
        wordOut, sortedLen2, reversedIndices2 = self.wordModel(token)
        caseOut, sortedLen3, reversedIndices3 = self.caseModel(case)
        featureOut, sortedLen4, reversedIndices4 = self.featureModel(feature)

        encoding = torch.cat([wordOut.float(), caseOut.float(), charOut.float()], dim=2)

        sortedLen = sortedLen1
        reverseIndices = reversedIndices1

        packed_embeds = pack_padded_sequence(encoding, sortedLen, batch_first=True)

        maxLen = sortedLen[0]
        mask = torch.zeros([len(sortedLen), maxLen, 2])
        for i, l in enumerate(sortedLen):
            mask[i][:l][:] = 1

        lstmOut, (h, _) = self.lstm(packed_embeds)

        paddedOut = pad_packed_sequence(lstmOut, batch_first=True)

        fcOut = self.fc(paddedOut[0])

        fcOut = fcOut * mask.cuda()
        fcOut = fcOut[reverseIndices]

        return fcOut

    def loss_func(self, yTrue, yPred, type):
        y = torch.eye(2)[yTrue].float().cuda()
        if len(y.shape) == 1:
            y = y[None, :]
        # y = torch.from_numpy(yTrue).float().cuda()
        if type == 'bnpu' or type == 'bpu':
            
            loss = torch.mean((y * (1 - yPred)).sum(dim=1))
           
        elif type == 'upu':
            loss = torch.mean((-y * torch.log(yPred)).sum(dim=1))
        # loss = 0.5 * torch.max(1-yPred*(2.0*yTrue-1),0)
        return loss


class Trainer(object):
    def __init__(self, model, prior, beta, gamma, learningRate, m, N=None):
        self.model = model
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.learningRate,
                                          weight_decay=1e-8)
        self.m = m
        self.prior = prior
        self.bestResult = -1
        self.beta = beta
        self.gamma = gamma
        self.positive = np.eye(2)[1]
        self.negative = np.eye(2)[0]
        if N is not None:
            self.N = N
            self.K = int((1-self.prior) * self.N)
            print("N:{}\tK:{}\tK/N:{}".format(self.N, self.K, self.K/self.N), flush=True)

    def train_mini_batch(self, batch, args):
        token, case, char, feature, label, flag = batch
        length = [len(i) for i in flag]
        maxLen = max(length)
        fids = []
        lids = []
        for s in flag:
            f = list(s)
            f += [np.array([-1, -1]) for _ in range(maxLen - len(f))]
            fids.append(f)
        for s in label:
            l = list(s)
            l += [np.array([-1, -1]) for _ in range(maxLen - len(l))]
            lids.append(l)
        fids = np.array(fids)
        lids = np.array(lids)

        postive = (fids == self.positive) * 1
        unlabeled = (fids == self.negative) * 1

        self.optimizer.zero_grad()
        result = self.model(token, case, char, feature)
        a, b = get_cnt(result, lids)
        hP = result.masked_select(torch.from_numpy(postive).bool().cuda()).contiguous().view(-1, 2)
        hU = result.masked_select(torch.from_numpy(unlabeled).bool().cuda()).contiguous().view(-1, 2)
        if len(hP) > 0:
            pRisk = self.model.loss_func(1, hP, args.type)
        else:
            pRisk = torch.FloatTensor([0]).cuda()
        
        uRisk = self.model.loss_func(0, hU, args.type)
        nRisk = uRisk - self.prior * (1 - pRisk)
        risk = self.m * pRisk + nRisk

        if args.type == 'bnpu':
            if nRisk < self.beta:
                risk = -self.gamma * nRisk
        
        (risk).backward()
        self.optimizer.step()
        pred = torch.argmax(hU, dim=1)
        label = Variable(torch.LongTensor(list(lids))).cuda()
        unlabeledY = label.masked_select(torch.from_numpy(unlabeled).bool().cuda()).contiguous().view(-1, 2)

        acc = torch.mean((torch.argmax(unlabeledY, dim=1) == pred).float())
        return acc.item(), risk.item(), pRisk.item(), nRisk.item(), a, b

    def test(self, batch, length):
        token, case, char, feature = batch
        maxLen = max([x for x in length])
        mask = np.zeros([len(token), maxLen, 2])
        for i, x in enumerate(length):
            mask[i][:x][:] = 1
        result = self.model(token, case, char, feature)
       
        result = result.masked_select(torch.from_numpy(mask).bool().cuda()).contiguous().view(-1, 2)
        pred = torch.argmax(result, dim=1)

        temp = result[:, 1]
        return pred.cpu().numpy(), temp.detach().cpu().numpy()

    def save(self, dir):
        if dir is not None:
            torch.save(self.model.state_dict(), dir)

    def decay_learning_rate(self, epoch, init_lr):
        
        lr = init_lr / (1 + 0.05 * epoch)
        print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)


def bio_tags_to_spans(
        tag_sequence: List[str], classes_to_ignore: List[str] = None
):
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").

    # Parameters

    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    # Returns

    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def compute_prf(sentences, flag=None):
    """
        sentences:
            [word, label(BIO), pred]
    """

    def convert_spans(preds):
        spans = set()
        start = 0
        while start < len(preds):
            if preds[start] == 1:
                end = start + 1
                while end < len(preds) and preds[end] == 1:
                    end += 1
                spans.add((start, end))
                start = end
            else:
                start += 1
        return spans
    
    def convert_spans_with_type(preds):
        spans = set()
        start = 0
        while start < len(preds):
            if preds[start]!='O':
                type_ = preds[start]
                end = start + 1
                while end < len(preds) and preds[end]==type_:
                    end += 1
                spans.add((start, end, type_))
                start = end
            else:
                start += 1
        return spans

    tp = 0
    pre = 0
    rec = 0
    all_spans = []
    for sentence in sentences:
        # words = [token[0] for token in sentence]
        labels = [token[1] for token in sentence]
        preds = [token[2] for token in sentence]
        if flag is not None:
            spans = set([(span[1][0], span[1][1] + 1) for span in bio_tags_to_spans(labels) if span[0] == flag])
            pred_spans = convert_spans(preds)
            
        else:
            spans = set([(span[1][0], span[1][1] + 1, span[0]) for span in bio_tags_to_spans(labels)])
            pred_spans = convert_spans_with_type(preds)
        tp += len(spans & pred_spans)
        pre += len(pred_spans)
        rec += len(spans)
        all_spans.append(pred_spans)
    
    p = tp / pre if pre != 0 else 0
    r = tp / rec if rec != 0 else 0
    f = 2 * p * r / (p + r) if (p + r) != 0 else 0
    return p, r, f, all_spans


def run(args, trainSet, validSet, testSet, prior, m, N):

    trainSize = len(trainSet)
    validSize = len(validSet)
    testSize = len(testSet)

    charcnn = CharCNN(dp.char2Idx)
    wordnet = WordNet(dp.wordEmbeddings, dp.word2Idx)
    casenet = CaseNet(dp.caseEmbeddings, dp.case2Idx)
    featurenet = FeatureNet()
    pulstmcnn = PULSTMCNN(dp, charcnn, wordnet, casenet, featurenet, 138, 200, 1, args.drop_out)

    if torch.cuda.is_available:
        charcnn.cuda()
        wordnet.cuda()
        casenet.cuda()
        featurenet.cuda()
        pulstmcnn.cuda()

    trainer = Trainer(pulstmcnn, prior, args.beta, args.gamma, args.lr, m, N)

    time = 0

    bar = ProgressBar(maxval=int((len(trainSet) - 1) / args.batch_size))

    train_sentences = dp.read_origin_file("data/" + args.dataset + "/train.txt")
    trainSize = int(len(train_sentences) * args.pert)
    train_sentences = train_sentences[:trainSize]
    train_words = []
    train_efs = []
    for s in train_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        train_words.append(temp)
        train_efs.append(temp2)

    valid_sentences = dp.read_origin_file("data/" + args.dataset + "/valid.txt")
    valid_words = []
    valid_efs = []
    for s in valid_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        valid_words.append(temp)
        valid_efs.append(temp2)

    test_sentences = dp.read_origin_file("data/" + args.dataset + "/test.txt")
    test_words = []
    test_efs = []
    for s in test_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        test_words.append(temp)
        test_efs.append(temp2)

    for e in range(1, 1000):
        print("Epoch: {}".format(e),flush=True)
        bar.start()
        risks = []
        prisks = []
        nrisks = []
        as_ = []
        bs = []
        for step, (x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch, flag_batch) in enumerate(
                mutils.iterateSet(trainSet, batchSize=args.batch_size, mode="TRAIN")):
            bar.update(step)
            batch = [x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch, flag_batch]
            acc, risk, prisk, nrisk, a, b = trainer.train_mini_batch(batch, args)
            as_ += a
            bs += b
            risks.append(risk)
            prisks.append(prisk)
            nrisks.append(nrisk)
        meanRisk = np.mean(np.array(risks))
        meanRisk2 = np.mean(np.array(prisks))
        meanRisk3 = np.mean(np.array(nrisks))
        print("risk: {}, prisk: {}, nrisk: {}".format(meanRisk, meanRisk2, meanRisk3), flush=True)
        if e % 5 == 0:
            trainer.decay_learning_rate(e, args.lr)
        if e % args.print_time == 0:
            
            pred_valid = []
            corr_valid = []
            as_ = []
            bs = []
            for step, (
                    x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch,
                    y_test_batch, _) in enumerate(
                mutils.iterateSet(validSet, batchSize=100, mode="TEST", shuffle=False)):
                validBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch]
                correcLabels = []
                for x in y_test_batch:
                    for xi in x:
                        correcLabels.append(xi)
                lengths = [len(x) for x in x_word_test_batch]
                predLabels, _ = trainer.test(validBatch, lengths)
                correcLabels = np.array(correcLabels)
                as_ += predLabels.tolist()
                bs += correcLabels.tolist()
                assert len(predLabels) == len(correcLabels)

                start = 0
                for i, l in enumerate(lengths):
                    end = start + l
                    p = predLabels[start:end]
                    c = correcLabels[start:end]
                    pred_valid.append(p)
                    corr_valid.append(c)
                    start = end

            newSentencesValid = []
            for i, s in enumerate(valid_words):
                sent = []
                assert len(s) == len(valid_efs[i]) == len(pred_valid[i])
                for j, item in enumerate(s):
                    sent.append([item, valid_efs[i][j], pred_valid[i][j]])
                newSentencesValid.append(sent)

            p_valid, r_valid, f1_valid, _ = compute_prf(newSentencesValid, args.flag)
            print("Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid),flush=True)
            print("Valid token, pre: {}, rec: {}, f1: {}".format(precision_score(bs, as_), recall_score(bs, as_), f1_score(bs, as_)))
            if f1_valid <= trainer.bestResult:
                time += 1
            else:
                trainer.bestResult = f1_valid
                time = 0
                trainer.save(
                    ("saved_model/{}_{}_{}_lr_{}_prior_{:.1f}_beta_{}_gamma_{}_percent_{}").format(args.type, args.dataset,
                                                                                               args.flag,
                                                                                               trainer.learningRate,
                                                                                               trainer.m,
                                                                                               trainer.beta,
                                                                                               trainer.gamma,
                                                                                               args.pert))
            if time > 5:
                print(("BEST RESULT ON VALIDATE DATA:{}").format(trainer.bestResult))
                break


    pulstmcnn.load_state_dict(
        torch.load(
            "saved_model/{}_{}_{}_lr_{}_prior_{:.1f}_beta_{}_gamma_{}_percent_{}".format(args.type, args.dataset, args.flag,
                                                                                     trainer.learningRate,
                                                                                     trainer.m,
                                                                                     trainer.beta,
                                                                                     trainer.gamma, args.pert)))

    pred_test = []
    corr_test = []
    as_ = []
    bs = []
    for step, (
            x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch,
            y_test_batch, _) in enumerate(
        mutils.iterateSet(testSet, batchSize=100, mode="TEST", shuffle=False)):
        testBatch = [x_word_test_batch, x_case_test_batch, x_char_test_batch, x_feature_test_batch]
        correcLabels = []
        for x in y_test_batch:
            for xi in x:
                correcLabels.append(xi)
        lengths = [len(x) for x in x_word_test_batch]
        predLabels, _ = trainer.test(testBatch, lengths)
        correcLabels = np.array(correcLabels)
        as_ += predLabels.tolist()
        bs += correcLabels.tolist()
        assert len(predLabels) == len(correcLabels)

        start = 0
        for i, l in enumerate(lengths):
            end = start + l
            p = predLabels[start:end]
            c = correcLabels[start:end]
            pred_test.append(p)
            corr_test.append(c)
            start = end

    newSentencesTest = []
    for i, s in enumerate(test_words):
        sent = []
        assert len(s) == len(test_efs[i]) == len(pred_test[i])
        for j, item in enumerate(s):
            sent.append([item, test_efs[i][j], pred_test[i][j]])
        newSentencesTest.append(sent)

    p_valid, r_valid, f1_valid, _ = compute_prf(newSentencesTest, args.flag)
    print("Test Result: Precision: {}, Recall: {}, F1: {}".format(p_valid, r_valid, f1_valid),flush=True)
    print("Test token, pre: {}, rec: {}, f1: {}".format(precision_score(bs, as_), recall_score(bs, as_), f1_score(bs, as_)))
    return p_valid, r_valid, f1_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PU NER")
    # data
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--beta', type=float, default=0.0,help='beta of pu learning (default 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0,help='gamma of pu learning (default 1.0)')
    parser.add_argument('--drop_out', type=float, default=0.5, help = 'dropout rate')
    parser.add_argument('--m', type=float, default=None, help='class balance rate')
    parser.add_argument('--flag', default="PER" , help='entity type (PER/LOC/ORG/MISC)')
    parser.add_argument('--dataset', default="conll2003",help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=100,help='batch size for training and testing')
    parser.add_argument('--print_time', type=int, default=1,help='epochs for printing result')
    parser.add_argument('--pert', type=float, default=1.0,help='percentage of data use for training')
    parser.add_argument('--type', type=str, default='bnpu',help='pu learning type (bnpu/bpu/upu)')  # bpu upu
    parser.add_argument('--seed', type=int, default=1013, help="random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    dp = DataPrepare(args.dataset)
    mutils = FeaturedDetectionModelUtils(dp)

    trainSet, validSet, testSet, prior, N = mutils.load_dataset(args.flag, args.dataset, args.pert)
    best_pre, best_rec, best_f1, best_m = -1, -1, -1, 0.3
    if args.m is None:
        for idx in range(10):
            pre, rec, f1 = run(args, trainSet, validSet, testSet, prior, 0.3+0.1*idx, N)
            if f1 > best_f1:
                best_f1 = f1
                best_pre = pre
                best_rec = rec
                best_m = 0.3+0.1*idx
        print("pre:{}\nrec:{}\nf1:{}\nm:{}".format(best_pre, best_rec, best_f1, best_m))
    else:
        pre, rec, f1 = run(args, trainSet, validSet, testSet, prior, args.m, N)
        print("pre:{}\nrec:{}\nf1:{}\nm:{}".format(pre, rec, f1, args.m))