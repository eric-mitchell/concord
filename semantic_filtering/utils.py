from __future__ import print_function
from collections import Counter
from math import ceil
import string
import re
import random
import sys

import torch
import torch.nn.functional as F

from hyperopt import hp, tpe, fmin, STATUS_OK
import hyperopt.rand as hyperrandom

sys.path.insert(1, "../nlic")

import nli
import qa
import qa_converter
import solver

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

device = "cuda"

import json
from collections import defaultdict

import tqdm
import argparse
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def calc_acc(ans_chosen, ans):
    f1s = []
    for i in range(len(ans_chosen)):
        if ans[i][0] != "":
            f1s.append(metric_max_over_ground_truths(f1_score, ans_chosen[i], ans[i]))
    return sum(f1s) / len(f1s)



def find_encapsulating_sentence(toks, start, end, is_html):
    stop_pattern = r'[a-zA-Z0-9(),/:&\- ]'
    # stop_pattern = r'\.\!\?'
    while re.search(stop_pattern, toks[start]) and start > 0 and not is_html[start]:
        start = start - 1
    start = start + 1
    while re.search(stop_pattern, toks[end]) and end < len(toks) - 1 and not is_html[end]:
        end = end + 1
    end = end + 1
    return " ".join([toks[x] for x in range(start, end) if not is_html[x]])


def extract_context_from_nq(item):
    toks = item['document']['tokens']['token']
    is_html = item['document']['tokens']['is_html']
    long_answers = item['annotations']['short_answers']
    # print(long_answers)
    all_toks = []
    all_sents = []
    all_pairs = []
    for ans in long_answers:
        p = (ans['start_token'], ans['end_token'])
        if len(ans['start_token']) and p not in all_pairs:
            all_pairs.append(p)
            # for tn in range(ans['start_token'][0], ans['end_token'][0]):
            #     if not is_html[tn]:
            #         all_toks.append(toks[tn])
            para = find_encapsulating_sentence(toks, ans['start_token'][0],
                                               ans['end_token'][0], is_html).replace(" . ", ". ") \
                .replace(" ( edit )", "").replace(" ( Learn how and when to remove this template message )", "")
            all_sents.append(para)
        all_toks = []
    return set(all_sents)


def extract_answer_from_nq(annots):
    all_ans = []
    for x in annots['short_answers']:
        if len(x['text']):
            all_ans.extend(x['text'])
    if not len(all_ans):
        return [""]
    return all_ans


def qualify_context(sent, nli_mod_dummy, quest):
    sent = normalize_answer(sent)
    quest = normalize_answer(quest)
    if "jump" in sent or "wiki" in sent:
        return False

    """
        Check word overlap
    """
    common = Counter(sent.split()) & Counter(quest.split())
    num_same = sum(common.values())
    if num_same == 0:
        return False

    # Not include too long or too short sentences
    if len(nli_mod_dummy.tok(sent, return_tensors="pt")['input_ids'][0]) > 100:
        return False
    if len(nli_mod_dummy.tok(sent, return_tensors="pt")['input_ids'][0]) < 10:
        return False
    return True



"""
    The following functions are from https://worksheets.codalab.org/bundles/0xbcd57bee090b421c982906709c8c27e1
"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
