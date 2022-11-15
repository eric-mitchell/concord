import json
import random

from eval_retrieve import metric_max_over_ground_truths, f1_score
import argparse
from collections import defaultdict


def eval_oracle(ans_test, final_ans):
    f1s = []
    oracle_choice = []
    print(len(ans_test))
    for i in range(len(ans_test)):
        if ans_test[i][0] != "":
            f1_ans = []
            for a in final_ans[i]:
                f1_ans.append(metric_max_over_ground_truths(f1_score, a, ans_test[i]))
            f1s.append(max(f1_ans))
            or_c = f1_ans.index(max(f1_ans))
            oracle_choice.append(or_c)
    oc_f1 = sum(f1s) / len(f1s)
    print("ORACLE F1:", oc_f1)
    return oracle_choice, oc_f1


def count_uniques():
    k_dict = {1: 0, 2: 0, 3: 0, 4: 0}
    total = 0
    for i in range(len(final_ans)):
        if ans_test[i][0] != "":
            total += 1
            len_curr = len(final_ans[i])
            for kd in k_dict:
                if len_curr >= kd:
                    k_dict[kd] += 1
    k_dict = {a: (k_dict[a] / total) * 100 for a in k_dict}
    print(k_dict)
    return k_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_id", type=str)
    parser.add_argument("--result_file", type=str)

    args = parser.parse_args()

    CACHE_PATH = args.cache_id
    result_store = open(args.result_file, "a+")
    res = {}

    data = defaultdict(list)
    with open(CACHE_PATH, "r") as f:
        for idx, l in enumerate(f):
            for k, v in json.loads(l).items():
                data[k].append(v)
    ans_test = data['ans_test']
    # cont_test = data['gold']
    final_ans = data['final_ans']

    for k in random.choices(list(range(len(final_ans))), k=30):
        if ans_test[k][0] != '':
            print("=-" * 40)
            print("PREDICTED:", final_ans[k])
            print("GROUND TRUTH:", ans_test[k])

    or_choice, f1_oc = eval_oracle(ans_test, final_ans)
    res.update({"oc_f1": f1_oc})

    res.update({"uniques": count_uniques()})
    result_store.write(json.dumps(res) + "\n")
