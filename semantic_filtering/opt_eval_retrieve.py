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

from utils import chunks, find_encapsulating_sentence, extract_context_from_nq, calc_acc, extract_answer_from_nq

sys.path.insert(1, "../nlic")

import nli
import qa
import qa_converter
import solver

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset

device = "cuda"

# Sentence similarity model
from sentence_transformers import SentenceTransformer

import json
from collections import defaultdict

import tqdm
import argparse
import os


def opt_function(model_name, to_nli, beta, conf, entailment_correction, normalize, dedup):
    nli_mod = nli.NLIInferencer(model_hf_name=model_name, confidence_threshold=conf, dedup_constraints=dedup)
    nlis = []
    for tns in to_nli:
        nli_relations = nli_mod(tns[0], should_evaluate_pairs=tns[1], fp_batch_size=20)
        nlis.extend(nli_relations)

    nlis = list(set(nlis))

    sol_mod = solver.Solver(beta=beta, normalize=normalize)
    res = sol_mod(statement_groups, nlis, verbose=False, true_statements=true_states, entailment_correction=entailment_correction,
                  normalize=normalize, relation_types=relation_types)

    ans_chosen = [final_ans[i][res[i]] for i in range(len(res))]
    avg_acc = calc_acc(ans_chosen, ans_test)

    return {"loss": -avg_acc, "status": STATUS_OK}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, help="Test for test, val as default", default="val")
    parser.add_argument("--folder", type=str, default="hyperparam_search")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.95)
    parser.add_argument("--ret_num", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--model", type=str, default="t5-large")
    parser.add_argument("--entailment_correction", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cache_dir", type=str)  # where to cache when using datasets etc
    parser.add_argument("--store_dir", type=str)  # where to store created caches
    parser.add_argument("--n_max", type=int, default=100)  # max round of hyperopt search
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

    N_MAX_ROUNDS = args.n_max
    SEED = 42

    cache_str = "{split}_{temp}_{n}_{model}".format(split=args.split,
                                                    temp=args.temp,
                                                    n=args.n, model=str(args.model))

    add_str = "{c_id}_{beta}_{conf}".format(c_id=cache_str, beta=args.beta, conf=args.conf)
    result_file = open("{folder}/experiment_results_{add_str}.jsonl".format(folder=args.folder, add_str=add_str), "a+")

    result_dict = {}
    result_dict.update(vars(args))
    result_dict.update({"cache_str": cache_str})
    result_dict.update({"add_str": add_str})

    sent_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    sent_model.to(device)

    random.seed(SEED)
    torch.manual_seed(SEED)
    # Initialize all different modules
    if args.model == "t5-large":
        qa_mod = qa.QA(model_hf_name='google/t5-large-ssm-nq', device=device)
    elif args.model == "t5-small":
        qa_mod = qa.QA(model_hf_name='google/t5-small-ssm-nq', device=device)
    else:
        qa_mod = qa.QA(model_hf_name='google/t5-3b-ssm-nq', device=device)

    qac_mod = qa_converter.QAConverter(model_weights_path=os.path.join(args.store_dir,
                                                                       "qa_converter_models",
                                                                       "t5-statement-conversion-finetune.pt"))

    # Load NQ Dataset
    nq_dev = load_dataset('natural_questions', 'dev', beam_runner='DirectRunner', cache_dir=args.cache_dir)

    if args.split == "test":
        # nq_dev_inds = list(range(len(nq_dev['validation'])))[:5000]
        nq_dev_inds = json.load(open("readable_indices.json"))["test"]
    else:
        # For dev, use N - 5000
        nq_dev_inds = json.load(open("readable_indices.json"))["val"]

    quest_test = [nq_dev['validation'][i]['question']['text'] for i in nq_dev_inds]

    # Check if there is cached file
    CACHE_PATH = os.path.join(args.store_dir, "cache_{cacheid}.jsonl").format(cacheid=cache_str)

    build_cache = False
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
            p_ans, p_probs = qa_mod(qs, temp=args.temp, n=args.n, num_beam_groups=args.n, num_beams=12,
                                    diversity_penalty=10.0, do_sample=False)
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
    to_nli = []

    if build_cache:
        f = open(CACHE_PATH, "a+")

    for i, s_s in tqdm.tqdm(enumerate(statements), total=len(statements)):
        if build_cache:
            golds = cont_test[i]
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
            all_retrieved = cont_test[i]
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
            to_nli.append((list(s_s) + list(all_retrieved), ind_pairs))

    assert len(statement_groups) == len(final_ans)

    rstate = np.random.default_rng(SEED)
    # models = ["ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    #         "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
    #           "roberta-large-mnli"]
    models = ["ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli", ]
    result_file = open(args.model + "_results.json", "a+")
    results = {}

    for model_name in models:
        tuning_info = fmin(fn=lambda opts: opt_function(model_name, to_nli, *opts),
                           space=[hp.uniform("beta", 0.0, 0.5),
                                  hp.uniform('conf', 0.0, 0.6),
                                  hp.choice("entailment_correction", [True, False]),
                                  hp.choice("normalize", [True, False]),
                                  hp.choice("dedup", [True, False])],
                           algo=tpe.suggest,
                           max_evals=N_MAX_ROUNDS,
                           show_progressbar=True,
                           rstate=rstate)
        print(f"info for {model_name}: {tuning_info}")
        # results.update({model_name: tuning_info})
        result_file.write(json.dumps({model_name: tuning_info}) + "\n")
