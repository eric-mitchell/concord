import transformers
from pysat.formula import IDPool, WCNFPlus
from pysat.examples.rc2 import RC2
import numpy as np
import json
from sklearn.metrics import f1_score
import argparse
import pickle

import sys
sys.path.append('../../nlic')

from hyperopt import fmin, tpe, hp, Trials
from functools import partial
from func_timeout import func_timeout, FunctionTimedOut

from solver import Solver

MAX_EVALS = 300
TIMEOUT = 10

def call_solver(solver, statement_groups, compared, normalize, rm_self_entail, entailment_correction):
    return(solver(statement_groups=statement_groups, relations=compared, verbose=False, normalize=normalize, rm_self_entail=rm_self_entail, entailment_correction=entailment_correction))

def return_total_F1(
    beta=0.3,
    confidence_threshold=0.9,
    cache_file=None,
    entailment_correction=True,
    data=None,
):
    print("Beta:",beta,"Confidence:",confidence_threshold)
    normalize = True
    rm_self_entail = False

    qsolver = Solver(beta=beta)

    questions_done = 0
    raw_correct = 0
    bad_change = 0
    good_change = 0
    new_correct = 0
    count = 0

    ground_truth = []
    initial_answer = []
    solved_answer = []

    for key in data.keys():
        img_data = data[key]
        # print(
        #     f"Now working on image: {key}; current NLI detection threshold is: {confidence_threshold}"
        # )

        for group in img_data.keys():
            # print(f"Now in group {group}.")
            group_list = img_data[group]["orig"]
            statement_groups = img_data[group]["nli"]["statement_groups"]
            statement_groups = [
                [
                    tuple(statement_groups[i][j])
                    for j in range(len(statement_groups[i]))
                ]
                for i in range(len(statement_groups))
            ]

            converted_flat = img_data[group]["nli"]["converted_flat"]

            compared = img_data[group]["nli"]["compared"]
            compared = [
                tuple(res) for res in compared if res[3] >= confidence_threshold
            ]

            try:
                result = func_timeout(timeout=TIMEOUT, func=call_solver, args=(qsolver, statement_groups, compared, normalize, rm_self_entail, entailment_correction))
            except FunctionTimedOut:
                result = [0 for i in range(len(statement_groups))]

            for i in range(len(group_list)):
                qs = group_list[i]
                ground_truth.append(qs["answer"])
                initial_answer.append(qs["prediction"][0])
                solved_answer.append(qs["prediction"][result[i]])

                if qs["answer"] == qs["prediction"][0]:
                    raw_correct += 1

                    if qs["answer"] != qs["prediction"][result[i]]:
                        bad_change += 1

                else:
                    if qs["answer"] == qs["prediction"][result[i]]:
                        good_change += 1

                if qs["answer"] == qs["prediction"][result[i]]:
                    new_correct += 1

                count += 1

        questions_done += 1

    orig_f1 = f1_score(ground_truth, initial_answer, average="micro")
    new_f1 = f1_score(ground_truth, solved_answer, average="micro")

    return orig_f1, new_f1


def objective(optargs, cache_file, entailment_correction, data):
    beta, confidence_threshold = optargs

    orig_f1, new_f1 = return_total_F1(beta=beta, confidence_threshold=confidence_threshold, cache_file=cache_file, entailment_correction=entailment_correction, data=data)

    print("Beta:",beta,"Confidence_threshold:",confidence_threshold, "Orig_F1:",orig_f1,"New_F1:",new_f1)

    return 1 - new_f1


def main(cache_file, entailment_correction, trials_out, max_evals):
    trials = Trials()

    data = json.load(cache_file)

    best = fmin(
        fn=partial(
            objective,
            cache_file=cache_file,
            entailment_correction=entailment_correction,
            data=data,
        ),
        space=[
            hp.uniform("beta", 0.05, 1.0),
            hp.uniform("confidence_threshold", (1.0 / 3.0), 1.0),
        ],
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(84567),
        show_progressbar=False,
    )

    pickle.dump(trials, trials_out)

    orig_f1, new_f1 = return_total_F1(
        beta=best["beta"],
        confidence_threshold=best["confidence_threshold"],
        cache_file=cache_file,
        entailment_correction=entailment_correction,
        data=data,
    )

    print(best)

    print(f"Original F1: {orig_f1}, Post-solver F1: {new_f1}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters on one of the vqa-style experiments"
    )

    parser.add_argument(
        "-f",
        "--cache_file",
        help="cache file containing qa/nli inferences",
        type=argparse.FileType("r"),
    )

    parser.add_argument(
        "-o",
        "--trials_out",
        help="file to save trials object from hyperopt run",
        type=argparse.FileType("wb"),
    )

    parser.add_argument(
        "-w",
        "--entailment_correction",
        help="set entailment_correction",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-t", "--max_evals", help="max number of hyperopt trials", type=int,
    )

    args = parser.parse_args()

    if args.max_evals is None:
        max_evals = MAX_EVALS
    else:
        max_evals = args.max_evals

    main(args.cache_file, args.entailment_correction, args.trials_out, max_evals)
