from nlic.solver import Solver
import numpy as np
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from sklearn.metrics import f1_score
from argparse import ArgumentParser
from cbqa.metrics import consistency, tau_single_subject
import csv

from hyperopt import fmin, tpe, hp, Trials
from functools import partial

from func_timeout import func_timeout, FunctionTimedOut

import pickle
import time
import yaml

YES = 'Yes.'
NO = 'No.'
int_str_map = lambda ans_int: YES if ans_int else NO
str_int_map = lambda ans_str: 1 if ans_str == YES else 0
solver_ans_map = lambda solver_ans: 1 if solver_ans else 0

TIMEOUT = 30
MAX_EVALS = 300

FILENAME_DICT = {
    'allenai/macaw-large': 'mac-lg',
    'allenai/macaw-3b': 'mac-3b',
    'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli': 'roberta-anli',
    'roberta-large-mnli': 'roberta-mnli',
    'ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli': 'albert',
    'oracle': 'oracle',
}
TUNED_HPARAMS_DIR = 'cbqa/tuned_hparams/'


@dataclass
class BeliefBankQuestion:
    """Houses all the relevant information for a particular question."""
    question: str
    predicate: str
    correct_answer: int
    ptlm_answer: str
    ptlm_confidence: float
    qaconv_positive: str


@dataclass
class Batch:
    id: str
    """Contains all the BeliefBankQuestion for this batch"""
    questions: List[BeliefBankQuestion]
    """The full set of relations (NLI or oracle) for this entity across all statements"""
    statement_relations: List[Tuple[str, str, str, float]] = field(default_factory=list, init=False)


def make_parser():
    parser = ArgumentParser(description='Run CBQA experiment')
    parser.add_argument('--mode', '-m', type=str,
                        default='inference',
                        choices=['hparam', 'inference'])
    parser.add_argument('--qa_model', '-qa', type=str,
                        default='allenai/macaw-large',
                        choices=['allenai/macaw-large', 'allenai/macaw-3b'])
    parser.add_argument('--qa_scores_cached_path', type=str,
                        default='data/cbqa/qa-cache/macaw-large/silver-facts-qa-scored.json',
                        help='Path to cached scored QA facts')
    parser.add_argument('--nli_model', '-nli', type=str,
                        default='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                        choices=[
                            'ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli',
                            'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                            'roberta-large-mnli']
                        )
    parser.add_argument('--nli_scores_cached_path', type=str,
                        default='data/cbqa/nli-cache/roberta-large-anli/nli-scored-silver-facts.csv',
                        help='Path to cached scored NLI pairs')
    parser.add_argument('--oracle', action='store_true',
                        help='If set, the ground truth constraints will be used instead of the NLI as input to the solver')
    parser.add_argument('--oracle_threshold', type=float, default=0.8, help='Confidence in oracle')
    parser.add_argument('--constraints_path', type=str,
                        default='data/cbqa/beliefbank-data-sep2021/constraints_v2.json')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--ablation_keep_relation', type=str, choices=['entailment', 'contradiction'], help='For the ablation, select one relation type to keep', default=None)
    parser.add_argument('--disable_ec', action='store_true', help='Disable entailment correction (enabled by default)')
    return parser


def get_complement(ptlm_answer: str, ptlm_prob: float) -> Tuple[str, float]:
    one_minus_p = np.clip(1 - ptlm_prob + 1e-6, 0, 1)
    if ptlm_answer == YES:
        return NO, one_minus_p
    elif ptlm_answer == NO:
        return YES, one_minus_p
    else:
        raise AssertionError(f'Unexpected PTLM answer: {ptlm_answer}')


def run_solver(batch: Batch,
               nli_threshold: float,
               beta: float,
               ablation_keep_relation: str,
               disable_entailment_correction: bool) -> List[int]:
    """
    Runs the solver for each batch.
    Returns the solver answers as a list of binary integers.
    """
    statement_groups = []
    for question in batch.questions:
        statement_groups.append(
            [(question.qaconv_positive,
              question.ptlm_confidence if question.ptlm_answer == YES else
              get_complement(question.ptlm_answer, question.ptlm_confidence)[
                  1])])
    # Filter out all the NLI relations that are below the threshold
    solver_input_nli_relations = list(
        filter(lambda nli_tuple: nli_tuple[3] >= nli_threshold, batch.statement_relations))
    solver = Solver(beta=beta, normalize=False) if beta is not None else Solver(normalize=False)
    solver_answers = solver(
        statement_groups,
        solver_input_nli_relations,
        normalize=False,
        entailment_correction=not disable_entailment_correction,
        groups_are_unary=True,
        relation_types=[ablation_keep_relation] if ablation_keep_relation else None
    )
    assert all(isinstance(ans, bool) for ans in solver_answers)
    return list(map(lambda ans: solver_ans_map(ans), solver_answers))


def analyze_results(batch: Batch, solver_answers: List[int], args) -> Tuple[
    List[int], List[int], List[int], List[str], int, int]:
    """Computes metrics."""
    correct_answers = []
    original_ptlm_answers = []
    predicates = []
    flipped_count = 0
    good_flips = 0
    bad_flips = 0
    if args.verbose:
        print(f'{batch.id} Results')
    for i, question in enumerate(batch.questions):
        correct_answers.append(question.correct_answer)
        predicates.append(question.predicate)
        ptlm_answer_int = str_int_map(question.ptlm_answer)
        original_ptlm_answers.append(ptlm_answer_int)
        flipped = ptlm_answer_int != solver_answers[i]
        if flipped:
            flipped_count += 1
            if ptlm_answer_int == question.correct_answer:
                # Flipped to incorrect
                bad_flips += 1
            else:
                assert solver_answers[i] == question.correct_answer
                # Flipped to correct
                good_flips += 1
            if args.verbose:
                print(
                    f'Flipped question: {question.question}, correct answer {int_str_map(question.correct_answer)}, orig. answer {question.ptlm_answer}, new answer {int_str_map(solver_answers[i])}')
    original_accuracy = f1_score(correct_answers, original_ptlm_answers)
    solver_accuracy = f1_score(correct_answers, solver_answers)
    with open(args.constraints_path, 'r') as f:
        links = json.load(f)['links']

    def single_subject_consistency(violated, applicable):
        """Since tau_single_subject returns violated, applicable"""
        if not applicable:
            print('Warning: no applicable constraints')
            return 1
        return 1 - (violated / applicable)
    violated, applicable = tau_single_subject(predicates, original_ptlm_answers,
                                              links)
    original_consistency = single_subject_consistency(violated, applicable)
    violated, applicable = tau_single_subject(predicates, solver_answers, links)
    final_consistency = single_subject_consistency(violated, applicable)
    if args.verbose:
        print(
            f'Summary: flipped count: {flipped_count}, original accuracy: {original_accuracy}, post-solver accuracy: {solver_accuracy}, original consistency: {original_consistency}, final consistency: {final_consistency}')
        print('------------------------------------------------------------\n')
    return correct_answers, original_ptlm_answers, solver_answers, predicates, good_flips, bad_flips


def get_filename(args, file_extension: str) -> str:
    nli_model = 'oracle' if args.oracle else args.nli_model
    name = f'{FILENAME_DICT[args.qa_model]}-{FILENAME_DICT[nli_model]}'
    if args.ablation_keep_relation:
        name += f'-{args.ablation_keep_relation}-only'
    if args.disable_ec:
        name += '-no-ec'
    return f'{name}.{file_extension}'


def save_hyperopt_output_to_file(hparams: Dict, args) -> None:
    with open(f'{TUNED_HPARAMS_DIR}{get_filename(args, file_extension="yml")}', 'w') as f:
        yaml.safe_dump(hparams, f)


def read_tuned_parameters_for_inference(args) -> Dict:
    with open(f'{TUNED_HPARAMS_DIR}{get_filename(args, file_extension="yml")}', 'r') as f:
        return yaml.safe_load(f)


def inference(batches: List[Batch], args):
    tuned_hparams = read_tuned_parameters_for_inference(args)
    print('config', {
            'QA model': args.qa_model,
            'NLI model': args.nli_model,
            'NLI threshold': tuned_hparams['nli_threshold'],
            'beta': tuned_hparams['beta']
    })
    if args.oracle:
        assert args.oracle_threshold == tuned_hparams['nli_threshold']
    correct_answers = []
    original_predictions = []
    final_predictions = []
    predicates = []
    subjects = []
    total_good_flips = 0
    total_bad_flips = 0
    solver_runtimes = []
    for batch in batches:
        start_time = time.time()
        try:
            solver_answers = func_timeout(timeout=TIMEOUT, func=run_solver,
                           args=(batch, tuned_hparams['nli_threshold'], tuned_hparams['beta'], args.ablation_keep_relation, args.disable_ec))
        except FunctionTimedOut:
            print(f'{batch.id} entity solver timed out!')
            solver_answers = [str_int_map(q.ptlm_answer) for q in batch.questions]
        entity_solver_runtime = time.time() - start_time
        print(f'{batch.id} solver ran for {entity_solver_runtime} seconds')
        solver_runtimes.append(entity_solver_runtime)
        batch_correct_answers, batch_original_answers, batch_solver_answers, batch_predicates, batch_good_flips, batch_bad_flips = analyze_results(batch, solver_answers, args)
        assert solver_answers == batch_solver_answers
        correct_answers.extend(batch_correct_answers)
        original_predictions.extend(batch_original_answers)
        final_predictions.extend(batch_solver_answers)
        predicates.extend(batch_predicates)
        subjects.extend([batch.id] * len(batch.questions))
        total_good_flips += batch_good_flips
        total_bad_flips += batch_bad_flips

    original_f1 = f1_score(correct_answers, original_predictions)
    final_f1 = f1_score(correct_answers, final_predictions)
    print(
        f'Overall original F1:{original_f1}, overall post-solver F1:{final_f1}, total good flips:{total_good_flips}, total bad flips:{total_bad_flips}, solver average runtime (seconds): {np.mean(solver_runtimes)}')
    with open(args.constraints_path, 'r') as f:
        links = json.load(f)['links']
    original_consistency = consistency(subjects, predicates, original_predictions, links)
    final_consistency = consistency(subjects, predicates, final_predictions, links)
    # This should be 1
    ground_truth_consistency = consistency(subjects, predicates, correct_answers, links)
    print(f'Overall original consistency: {original_consistency}, overall post-solver consistency {final_consistency}, ground truth consistency {ground_truth_consistency}')


def set_up_hyperopt(batches: List[Batch], args):
    trials = Trials()
    space = [
            hp.uniform('beta', 0.05, 1.0),
        ] if args.oracle else [
            hp.uniform('beta', 0.05, 1.0),
            hp.uniform('nli_threshold', 0.5, 1.0),
        ]

    best = fmin(
        fn=partial(opt_wtimeout, args, batches=batches),
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        rstate=np.random.default_rng(224)
    )

    with open(get_filename(args, file_extension='pickle'), 'wb') as outfile:
        # 'wb' argument opens the file in binary mode
        pickle.dump(trials, outfile)

    best = {k: v.item() for k, v in best.items()}
    save_hyperopt_output_to_file(best, args)
    print(best)


def opt_function(external_args, optargs, batches: List[Batch]):
    if external_args.oracle:
        beta, nli_threshold = optargs[0], external_args.oracle_threshold
    else:
        beta, nli_threshold = optargs

    # For calculating F1 over all entities/batches
    labels = []
    preds = []
    for batch in batches:
        batch_labels = [question.correct_answer for question in batch.questions]
        solver_ans = run_solver(batch, nli_threshold, beta, external_args.ablation_keep_relation, external_args.disable_ec)
        assert len(batch_labels) == len(solver_ans)
        labels.extend(batch_labels)
        preds.extend(solver_ans)

    return (1 - f1_score(labels, preds))


def opt_wtimeout(external_args, optargs, batches: List[Batch]):
    try:
        obj = func_timeout(timeout=TIMEOUT, func=opt_function, args=(external_args, optargs, batches))

    except FunctionTimedOut:
        obj = 1

    except Exception as err:
        print(f'Unexpected exception {err}')
        obj = 1

    return obj


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    # in the initial experiment, each batch contains all the facts for one entity
    batches = []
    with open(args.qa_scores_cached_path, 'r') as f:
        facts = json.load(f)

    # Read the QA cache
    for _, entity_facts in facts.items():
        questions = []
        for _, v in entity_facts.items():
            assert {'nl_question', 'relation', 'obj', 'nl_correct_answer', 'qa_ans', 'qa_ans_p', 'qaconv_positive'}.issubset(v.keys())
            questions.append(BeliefBankQuestion(v['nl_question'],
                                          v['relation'] + ',' + v['obj'],
                                          str_int_map(v['nl_correct_answer']),
                                          v['qa_ans'],
                                          v['qa_ans_p'],
                                          v['qaconv_positive']
                                          ))
        # batch ID is the entity name
        batches.append(Batch(entity_facts['0']['subject'], questions))

    if args.oracle:
        # Read the ground truth links from the constraints file
        all_links = {}
        weight_relation_map = lambda weight: 'entailment' if weight == 'yes_yes' else 'contradiction'
        with open(args.constraints_path, 'r') as f:
            links = json.load(f)['links']
            for link in links:
                if link['direction'] != 'forward':
                    continue
                w = link['weight']
                assert w == 'yes_yes' or w == 'yes_no'
                k = (link['source'], link['target'])
                # Should not be any duplicates
                assert k not in all_links
                all_links[k] = weight_relation_map(w)
        
        # Assign oracle relations for each entity
        for batch in batches:
            relations = []
            for idx, q1 in enumerate(batch.questions):
                for jdx, q2 in enumerate(batch.questions):
                    if idx == jdx:
                        continue
                    # predicate = relation + obj, e.g., 'IsA,tree', which is the format that each link 'source' and 'target' has above
                    k = (q1.predicate, q2.predicate)
                    if k not in all_links:
                        continue
                    relations.append((q1.qaconv_positive, q2.qaconv_positive, all_links[k], args.oracle_threshold))
            batch.statement_relations = relations

    else:
        # Read the NLI cache
        with open(args.nli_scores_cached_path, 'r') as f:
            reader = csv.reader(f)
            i = 0
            for entity_entry_count in reader:
                # for this entity, this list contains the input for the solver
                qaconv_positives = [q.qaconv_positive for q in batches[i].questions]
                nli_relations = []
                for _ in range(int(entity_entry_count[0])):
                    s1, s2, t, conf = next(reader)
                    assert s1 in qaconv_positives and s2 in qaconv_positives
                    conf = float(conf)
                    nli_relations.append((s1, s2, t, conf))
                batches[i].statement_relations = nli_relations
                i += 1

    if args.mode == 'inference':
        inference(batches, args)
    else:
        set_up_hyperopt(batches, args)
