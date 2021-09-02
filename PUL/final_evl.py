# -*- coding: utf-8 -*-
# @Time    : 2018/10/13 12:21
# @Author  : Xiaoyu Xing
# @File    : final_evl.py

import numpy as np
import argparse
import json
from utils.data_utils import DataPrepare, set_seed
from feature_pu_model import compute_prf


def get_final_result(per, loc, org, misc=None):
    assert len(per) == len(loc) == len(org)
    per_ = np.array(per)
    loc_ = np.array(loc)
    org_ = np.array(org)
    if misc:
        misc_ = np.array(misc)

    lens = len(per)

    column_val = ["PER", 'LOC', 'ORG', "MISC", 'O']

    result = []
    for i in range(lens):
        pers = np.array(per_[i])
        locs = np.array(loc_[i])
        orgs = np.array(org_[i])
        if misc:
            miscs = np.array(misc_[i])

            merge = np.array([pers, locs, orgs, miscs]).transpose()
        else:
            merge = np.array([pers, locs, orgs]).transpose()

        res = []
        for j in range(len(merge)):
            arg_max = np.argmax(merge[j])
            max = np.max(merge[j])

            if max <= 0.5:
                res.append(column_val[-1])
            else:
                res.append(column_val[arg_max])

        result.append(res)

    return result

def get_final_result_types(flags):
    lens = None
    flags_ = {}
    for flag, res in flags.items():
        if lens is None:
            lens = len(res)
        else:
            assert lens==len(res)
        flags_[flag] = np.array(res)

    cols = list(flags.keys())
    cols += ['O']

    result = []
    for i in range(lens):
        curs = {flag:np.array(res[i]) for flag, res in flags_.items()}
        merge = np.array([curs[flag] for flag in types]).transpose()

        res = []
        for j in range(len(merge)):
            arg_max = np.argmax(merge[j])
            max = np.max(merge[j])
            if max <= 0.5:
                res.append(cols[-1])
            else:
                res.append(cols[arg_max])
        result.append(res)
    return result

def get_final_result_types_score(flags):
    lens = None
    flags_ = {}
    for flag, res in flags.items():
        if lens is None:
            lens = len(res)
        else:
            assert lens==len(res)
        flags_[flag] = np.array(res)

    cols = list(flags.keys())
    cols += ['O']

    result = []
    scores = []
    for i in range(lens):
        curs = {flag:np.array(res[i]) for flag, res in flags_.items()}
        merge = np.array([curs[flag] for flag in types]).transpose()

        res = []
        score = []
        for j in range(len(merge)):
            arg_max = np.argmax(merge[j])
            max = np.max(merge[j])
            if max <= 0.5:
                res.append(cols[-1])
                max = 0
            else:
                res.append(cols[arg_max])
            score.append(str(max))
        result.append(res)
        scores.append(score)
    return result, scores


def get_output(filename):
    with open(filename, "r",encoding='utf-8') as fw:
        ress = []
        res = []
        for line in fw:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                if len(res) > 0:
                    ress.append(res)
                    res = []
                continue
            else:
                splits = line.strip().split(' ')
                res.append(float(splits[-1].strip()))

        if len(res) > 0:
            ress.append(res)
            res = []

        return ress


def prf1(labels, preds):
    def compute_precision_recall_f1(labels, preds):
        tp = 0
        np_ = 0
        pp = 0
        for i in range(len(labels)):
            sent_label = labels[i]
            sent_pred = preds[i]
            for j in range(len(sent_label)):
                item1 = np.array(sent_pred[j])
                item2 = np.array(sent_label[j])

                if (item1 == "PER").all() == True or (item1 == "LOC").all() == True or (
                            item1 == "ORG").all() == True :
                    pp += 1

                if (item2 == "PER").all() == True or (item2 == "LOC").all() == True or (
                            item2 == "ORG").all() == True :
                    np_ += 1
                    et_t = item2[0]

                    if (item1 == et_t).all() == True:
                        tp += 1
        if pp == 0:
            p = 0
        else:
            p = float(tp) / float(pp)
        if np_ == 0:
            r = 0
        else:
            r = float(tp) / float(np_)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = float(2 * p * r) / float((p + r))
        return p, r, f1

    p, r, f1 = compute_precision_recall_f1(labels, preds)

    return p, r, f1
    


def get_conflict(per, loc, org, misc=None):
    assert len(per) == len(loc) == len(org)
    per_ = np.array(per)
    loc_ = np.array(loc)
    org_ = np.array(org)
    if misc:
        misc_ = np.array(misc)

    lens = len(per)

    conflict_num = 0
    word_num = 0

    for i in range(lens):
        pers = np.array(per_[i])
        locs = np.array(loc_[i])
        orgs = np.array(org_[i])
        if misc:
            miscs = np.array(misc_[i])

            merge = np.array([pers, locs, orgs, miscs]).transpose()
        else:
            merge = np.array([pers, locs, orgs]).transpose()

        temp = 0
        for j in range(len(merge)):
            word_num += 1

            if np.sum(merge[j] > 0.5) >= 2:
                temp += 1

        if temp >= 2:
            conflict_num += 1

    return conflict_num, word_num, float(conflict_num) / float(word_num)


def get_match_final_result(per, loc, org, misc=None):
    assert len(per) == len(loc) == len(org)
    per_ = np.array(per)
    loc_ = np.array(loc)
    org_ = np.array(org)
    if misc:
        misc_ = np.array(misc)

    lens = len(per)

    column_val = ["PER", 'LOC', 'ORG', "MISC", 'O']

    result = []
    for i in range(lens):
        pers = np.array(per_[i])
        locs = np.array(loc_[i])
        orgs = np.array(org_[i])
        if misc:
            miscs = np.array(misc_[i])
            merge = np.array([pers, locs, orgs, miscs]).transpose()
        else:
            merge = np.array([pers, locs, orgs]).transpose()

        res = []
        for j in range(len(merge)):
            count_one = 0
            for m in merge[j]:
                if m == 1:
                    count_one += 1
            if count_one==1:
                arg_max = np.argmax(merge[j])
                res.append(column_val[arg_max])
            else:
                res.append(column_val[-1])

        result.append(res)

    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PU NER EVL")
    parser.add_argument('--dataset', default="conll2003")
    parser.add_argument('--type', default="bnpu")
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1013)
    parser.add_argument("--nonmisc", dest='nonmisc', type=int, default=0)
    parser.add_argument("--type-file", dest="type_file", default="./types.json")
    parser.add_argument("--result-dir", dest="result_dir", default="result")
    parser.add_argument("--save-file", dest="save_file", default=None)
    parser.add_argument("--save-type", dest="save_type", default="lu")
    args = parser.parse_args()
    with open(args.type_file, "r", encoding="utf-8") as file:
        types = json.load(file)[args.dataset]

    set_seed(args.seed)

    if args.nonmisc==0:
        filenames = [
            args.result_dir+"/"+args.type+"_feature_pu_"+args.dataset+"_{}_{}.txt".format(flag, args.num) 
                for flag in types
        ]
    else:
        filenames = [
            args.result_dir+"/"+args.type+"_feature_pu_"+args.dataset+"_{}_{}.txt".format(flag, args.num) 
                for flag in types if flag!='MISC'
        ]


    origin_file = "data/" + args.dataset + "/{}".format("train.txt" if args.num==2 else "test.txt")
    dp = DataPrepare(args.dataset)

    test_sentences = dp.read_origin_file(origin_file)
    test_words = []
    test_efs = []
    lens = []
    for s in test_sentences:
        temp = []
        temp2 = []
        for word, ef, lf in s:
            temp.append(word)
            temp2.append(ef)
        test_words.append(temp)
        test_efs.append(temp2)
        lens.append(len(s))

    results = {flag: get_output(filename) for flag, filename in zip(types, filenames)}

    final_res = get_final_result_types(results)
    newSentencesTest = []
    tp, pre, rec = 0, 0, 0
    for i, s in enumerate(test_words):
        sent = []
        for j, item in enumerate(s):
            sent.append([item, test_efs[i][j], final_res[i][j]])
            if final_res[i][j]!='O' and test_efs[i][j].split("-")[-1] == final_res[i][j]:
                tp += 1
            if final_res[i][j]!='O':
                pre += 1
            if test_efs[i][j]!='O':
                rec += 1
        newSentencesTest.append(sent)
    print(test_efs[-1], final_res[-1])

    p, r, f1, sents_pred_spans = compute_prf(newSentencesTest)
    if args.save_file is not None:
        if args.save_type == "lu":
            nlines = []
            for i, s in enumerate(test_words):
                spanline = " | ".join(["{},{} {}".format(span[0], span[1], span[2]) for span in sents_pred_spans[i]])
                lines = [" ".join(s), "", spanline, ""]
                nlines.append("\n".join(lines))
        elif args.save_type == "conll":
            final_res, scores = get_final_result_types_score(results)
            nlines = []
            for i, s in enumerate(test_words):
                sent = []
                for j, item in enumerate(s):
                    sent.append(" ".join([item, test_efs[i][j], final_res[i][j], str(scores[i][j])]))
                sent.append("")
                nlines.append("\n".join(sent))
        with open(args.save_file, "w", encoding="utf-8") as file:
                file.write("\n".join(nlines))
                file.write("\n")

    print(p, r, f1)
    print(tp, pre, rec)
    precision = tp / pre if pre!=0 else 0
    recall = tp / rec if rec!=0 else 0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall)!=0 else 0
