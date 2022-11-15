from __future__ import print_function
from collections import Counter
from math import ceil
import string
import re
import random
import sys

import torch
import torch.nn.functional as F

from hyperopt import hp, tpe, fmin

sys.path.insert(1, "../nlic")

import nli
import qa
import qa_converter
import solver

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

device = "cuda"

from utils import chunks, find_encapsulating_sentence, extract_context_from_nq, extract_answer_from_nq, qualify_context, metric_max_over_ground_truths, f1_score

import json
from collections import defaultdict
import elasticsearch as es
import elasticsearch.helpers as helpers

import tqdm
import h5py

import time
import argparse
import os


def retrieve_from_gold(i):
    all_set = cont_test[i]
    ret_set = [sent for sent in all_set if qualify_context(sent, nli_mod_dummy, quest_test[i])]
    return ret_set


def calc_acc_eval(ans_chosen, ans):
    f1s = []
    should_flip = []
    for i in range(len(ans_chosen)):
        if ans[i][0] != "":
            # Calculate whether there should be a flip
            f1_cands = [metric_max_over_ground_truths(f1_score, a, ans[i]) for a in final_ans[i]]
            new_f1 = metric_max_over_ground_truths(f1_score, ans_chosen[i], ans[i])
            f1s.append(new_f1)
            if new_f1 < max(f1_cands) and args.mode != "base":
                should_flip.append((ans_chosen[i], final_ans[i][f1_cands.index(max(f1_cands))]))
    return sum(f1s) / len(f1s), should_flip


def calc_bad_flips(ans_chosen_base, ans_chosen_comp):
    good_flips = 0
    bad_flips = 0
    total = 0
    bad_questions = []
    good_questions = []

    for i in range(len(ans_chosen_base)):
        if len(final_ans[i]) > 1:
            # only look at those with multiple answers
            total += 1
            f1_base = metric_max_over_ground_truths(f1_score, ans_chosen_base[i], ans_test[i])
            f1_comp = metric_max_over_ground_truths(f1_score, ans_chosen_comp[i], ans_test[i])
            if f1_base > f1_comp:
                bad_flips += 1
                bad_questions.append({"bad_question": quest_test[i], "gold_answers": ans_test[i],
                                      "gen_answers": statements[i], "bad_context": retrieval_for_store[i],
                                      "original_answer": ans_chosen_base[i], "edited_answer": ans_chosen_comp[i]})
            elif f1_comp > f1_base:
                good_flips += 1
                good_questions.append({"good_question": quest_test[i], "gold_answers": ans_test[i],
                                       "gen_answers": statements[i], "good_context": retrieval_for_store[i],
                                       "original_answer": ans_chosen_base[i], "edited_answer": ans_chosen_comp[i]})
    print("Bad Flips:", bad_flips, "Good Flips:", good_flips, "Total Possible:", total)
    return {"bad": bad_flips, "good": good_flips, "total": total, "bad_details": bad_questions,
            "good_details": good_questions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Test for test, val as default", default="val")
    parser.add_argument("--mode", type=str, help="base, gold", default="base")
    parser.add_argument("--folder", type=str, default="eval_results")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--ret_num", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--nli_model", type=str, default="ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")
    parser.add_argument("--entailment_correction", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--cache_dir", type=str)  # where to cache when using datasets etc
    parser.add_argument("--store_dir", type=str)  # where to store created caches
    parser.add_argument("--entail_only", action="store_true")
    parser.add_argument("--contradiction_only", action="store_true")
    args = parser.parse_args()

    relation_types = []
    if args.entail_only:
        relation_types = ["entailment"]
    elif args.contradiction_only:
        relation_types = ["contradiction"]
    else:
        relation_types = ["entailment", "contradiction"]

    """
        Different parameters for each combination
    """
    if not args.beta and not args.conf:
        if args.mode == "base":
            args.beta = 0
            args.conf = 0
        else:
            # if they are not defined
            if args.model == "t5-small":
                if args.entail_only:
                    args.beta = 0.04261003861208565
                    args.conf = 0.2679288980869049
                    args.normalize = True
                    args.dedup = True
                elif args.contradiction_only:
                    args.beta = 0.06727263159000196
                    args.conf = 0.42597694683939813
                    args.dedup = True
                else:
                    args.beta = 0.112
                    args.conf = 0.540
                    args.entailment_correction = True
            elif args.model == "t5-large":
                if args.entail_only:
                    args.beta = 0.08201305689543481
                    args.conf = 0.27789299933094547
                elif args.contradiction_only:
                    args.beta = 0.069035961077337
                    args.conf = 0.21699031102135544
                else:
                    args.beta = 0.08195625122916185
                    args.conf = 0.4127585581597353
                    args.dedup = True
            elif args.model == "t5-3b":
                if args.entail_only:
                    args.beta = 0.08582894757505771
                    args.conf = 0.1615415420389351
                elif args.contradiction_only:
                    args.beta = 0.10645261397410088
                    args.conf = 0.18461945046745287
                    args.entailment_correction = True
                else:
                    args.beta = 0.0719546124225483
                    args.conf = 0.47686626322806097
                    args.entailment_correction = True
                    args.normalize = True

    # Time counters
    times_qa = []
    times_nli = []
    times_solver = []

    cache_str = "{split}_{temp}_{n}_{model}".format(split=args.split,
                                                    temp=args.temp,
                                                    n=args.n, model=str(args.model))

    add_str = "{c_id}_{beta}_{conf}".format(c_id=cache_str, beta=args.beta, conf=args.conf)
    result_file = open("{folder}/experiment_results_{add_str}.jsonl".format(folder=args.folder, add_str=add_str), "a+")

    result_dict = {}
    result_dict.update(vars(args))
    result_dict.update({"cache_str": cache_str})
    result_dict.update({"add_str": add_str})

    random.seed(42)
    torch.manual_seed(42)
    # Initialize all different modules
    if args.model == "t5-large":
        qa_mod = qa.QA(model_hf_name='google/t5-large-ssm-nq', device=device)
    elif args.model == "t5-small":
        qa_mod = qa.QA(model_hf_name='google/t5-small-ssm-nq', device=device)
    else:
        qa_mod = qa.QA(model_hf_name='google/t5-3b-ssm-nq', device=device)
    if args.nli_model is None:
        nli_mod = nli.NLIInferencer(confidence_threshold=args.conf, dedup_constraints=args.dedup)
    else:
        nli_mod = nli.NLIInferencer(confidence_threshold=args.conf, model_hf_name=args.nli_model,
                                    dedup_constraints=args.dedup)
    qac_mod = qa_converter.QAConverter()

    # Dummy model for tokenization purposes
    nli_mod_dummy = nli.NLIInferencer(model_hf_name=args.nli_model)

    # Load NQ Dataset
    nq_dev = load_dataset('natural_questions', 'dev', beam_runner='DirectRunner', cache_dir=args.cache_dir)

    if args.split == "test":
        nq_dev_inds = json.load(open("readable_indices.json"))["test"]
    else:
        nq_dev_inds = json.load(open("readable_indices.json"))["val"]

    quest_test = [nq_dev['validation'][i]['question']['text'] for i in nq_dev_inds]

    # Check if there is cached file
    CACHE_PATH = os.path.join(args.store_dir, "cache_{cacheid}.jsonl").format(cacheid=cache_str)

    build_cache = False
    print(CACHE_PATH)
    print(f"Attempting to load from cache {CACHE_PATH}")
    if os.path.exists(CACHE_PATH):
        print("Reading from cache ... ")
        data = defaultdict(list)
        with open(CACHE_PATH, "r") as f:
            for idx, l in enumerate(f):
                for k, v in json.loads(l).items():
                    data[k].append(v)
        ans_test = data['ans_test']
        # cont_test = data['gold']
        # Build context
        cont_test = [extract_context_from_nq(nq_dev['validation'][i]) for i in nq_dev_inds]
        assert len(cont_test) == len(nq_dev_inds)
        final_ans = data['final_ans']
        final_probs = data['final_probs']
        statements = data['statements']
    else:
        print("Building cache ... ")
        build_cache = True
        ans_test = [extract_answer_from_nq(nq_dev['validation'][i]['annotations']) for i in nq_dev_inds]

        # Build context
        cont_test = [extract_context_from_nq(nq_dev['validation'][i]) for i in nq_dev_inds]

        # Use the same set
        ans, probs = [], []
        all_quests = chunks(quest_test, 50)
        for qs in quest_test:
            qa_start = time.time()
            p_ans, p_probs = qa_mod(qs, temp=args.temp, n=args.n, num_beam_groups=args.n, num_beams=12,
                                    diversity_penalty=10.0, do_sample=False)
            times_qa.append(time.time() - qa_start)
            # qa_t(questions, n=4, num_beam_groups=4, num_beams=12, diversity_penalty=10.0, do_sample=False)
            ans.extend(p_ans)
            probs.extend(p_probs)
        ans = list(chunks(ans, args.n))
        probs = list(chunks(probs, args.n))

        del qa_mod

        final_ans = [list(dict.fromkeys(a)) for a in ans]
        final_probs = [list(dict.fromkeys(p)) for p in probs]

        all_qs = list(chunks(quest_test, 100))
        all_fans = list(chunks(final_ans, 100))
        assert len(all_qs) == len(all_fans)
        statements = []
        for i in range(len(all_qs)):
            statements.extend(qac_mod(all_qs[i], all_fans[i]))

    # We have retrieved groups per statement group
    # Each group corresponds to one question - exactly one of the statements can ultimately get assigned "True"
    statement_groups = []

    for idx, (s, c) in enumerate(zip(statements, final_probs)):
        statement_groups.append([])
        for ss, cc in zip(s, c):
            if ss not in [x[0] for x in statement_groups[-1]]:
                statement_groups[-1].append((ss, cc))

    sol_mod = solver.Solver(beta=args.beta, normalize=args.normalize)
    nlis = []
    true_states = []
    retrieval_for_store = []

    if build_cache:
        f = open(CACHE_PATH, "a+")

    if args.mode != "base":
        for i, s_s in tqdm.tqdm(enumerate(statements), total=len(statements)):
            if build_cache:
                golds = retrieve_from_gold(i)
                data_store = {
                    "gold": list(golds),
                    "ans_test": ans_test[i],
                    "final_ans": final_ans[i],
                    "final_probs": final_probs[i],
                    "statements": statements[i]
                }
                f.write(json.dumps(data_store) + "\n")
                all_retrieved = data_store[args.mode]
            else:
                all_retrieved = retrieve_from_gold(i)
            true_states.extend(list(all_retrieved))
            retrieval_for_store.append(list(all_retrieved))
            # Build the NLI inference index pairs
            state_inds = list(range(len(s_s)))
            ret_inds = list(range(len(all_retrieved)))
            ret_inds = [x + len(state_inds) for x in ret_inds]
            ind_pairs = []
            for x in state_inds:
                for y in ret_inds:
                    ind_pairs.append((x, y))
            if len(all_retrieved):
                nli_start = time.time()
                nli_relations = nli_mod(list(s_s) + list(all_retrieved), should_evaluate_pairs=ind_pairs,
                                        fp_batch_size=20)
                times_nli.append(time.time() - nli_start)
                nlis.extend(nli_relations)

    nlis = list(set(nlis))
    assert len(statement_groups) == len(final_ans)

    print(Counter([a[2] for a in nlis]))

    sol_mod = solver.Solver(beta=args.beta, normalize=args.normalize)

    sol_start = time.time()
    res = sol_mod(statement_groups, nlis, verbose=False, true_statements=true_states, entailment_correction=args.entailment_correction,
                  normalize=args.normalize)
    times_solver.append(time.time() - sol_start)

    ans_chosen = [final_ans[i][res[i]] for i in range(len(res))]
    ac, sf = calc_acc_eval(ans_chosen, ans_test)
    result_dict.update({"result": ac})
    result_dict.update({"should_flip": sf})
    print(result_dict["result"], vars(args))

    # Base: select the statements with highest probability
    ans_base = [final_ans[i][final_probs[i].index(max(final_probs[i]))] for i in range(len(res))]
    if args.mode != "base":
        result_dict.update(calc_bad_flips(ans_base, ans_chosen))

    result_file.write(json.dumps(result_dict) + "\n")

    # Model result file
    model_result_file = open(args.model + "_results.jsonl", "a+")
    model_store = vars(args)
    model_store.update({"result-f1": result_dict["result"]})
    model_result_file.write(json.dumps(model_store) + "\n")

    # Times
    # print("======= Summary of Running Times =======")
    # num_of_qs = max(len(times_nli), len(times_qa), len(times_solver))
    # print("Average time for QA Forward:", sum(times_qa) / len(times_qa))
    # print("Average time for NLI Forward:", sum(times_nli) / len(times_nli))
    # print("Average time for Solver:", sum(times_solver) / max(len(times_solver), 1))
