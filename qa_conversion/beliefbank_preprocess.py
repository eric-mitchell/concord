import argparse
import csv
import json
import random
from typing import List

import inflect

DATASET_NAME = 'BeliefBank'
BELIEFBANK_SILVER_FACTS_SIZE = 12636
OUT_TRAIN_FILE_NAME = 'train.tsv'
OUT_DEV_FILE_NAME = 'dev.tsv'
_p = inflect.engine()


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and returns an `argparse.Namespace`."""
    parser = argparse.ArgumentParser(
        description="Appends BeliefBank data to the datasets used for fine-tuning T5"
    )
    parser.add_argument(
        "-f", "--facts", help="Fact file path", type=str
    )
    parser.add_argument("--train_fraction",
                        help="Fraction of dataset to use for training",
                        default=0.15, type=float)
    parser.add_argument("--demszky_train_path",
                        help="Path to the Demszky train dataset to use",
                        default='demszky_train.tsv',
                        type=str)
    parser.add_argument("--demszky_dev_path",
                        help="Path to the Demszky dev dataset to use",
                        default='demszky_dev.tsv',
                        type=str)
    parser.add_argument('--templates',
                        help='Templates to use to build questions and statements',
                        default='../../winter2022_projects/jnoh4_madwsa/data/templates.json',
                        type=str)
    parser.add_argument('--uncountables',
                        help='File containing phrases that are not quantifiable',
                        default='../../winter2022_projects/jnoh4_madwsa/data/non_countable.txt',
                        type=str),
    return parser.parse_args()


def copy_file(src_file_path: str, dest_file_path: str) -> None:
    with open(dest_file_path, 'w') as dest, open(src_file_path, 'r') as src:
        for line in src:
            dest.write(line)


def build_and_write_facts(
        fact_file_path: str, uncountables: List[str], templates_file_path: str,
        train_fraction: float) -> None:
    """
    Split BeliefBank facts into train and dev datasets.

    @param fact_file_path: the path to the fact file to read
    @param uncountables: a list of uncountable noun phrases
    @param templates_file_path: path to the templates to use
    @param train_fraction: fraction of dataset to use for training
    """
    facts = json.load(open(fact_file_path, 'r'))
    templates = json.load(open(templates_file_path, "r"))

    id = 0
    num_training_samples = int(BELIEFBANK_SILVER_FACTS_SIZE * train_fraction)
    training_samples_ids = random.sample(range(BELIEFBANK_SILVER_FACTS_SIZE),
                                         num_training_samples)
    with open(OUT_TRAIN_FILE_NAME, 'a', newline='') as train_file, open(OUT_DEV_FILE_NAME,
                                                            'a', newline='') as dev_file:
        train_writer = csv.writer(train_file, delimiter='\t')
        dev_writer = csv.writer(dev_file, delimiter='\t')
        for entity, related_facts in facts.items():
            for predicate, answer in related_facts.items():
                relation, target = predicate.split(",")
                formatted_target = target
                formatted_entity = entity
                if target not in uncountables:
                    if ((relation == 'HasA' or relation == 'HasPart') and not _p.singular_noun(
                            target)) or relation == 'IsA':
                        formatted_target = _p.a(target)
                if entity not in uncountables:
                    # entities are all singular
                    formatted_entity = _p.a(entity)
                chosen_template = random.choice(
                    templates[relation]["templates"])
                nl_question = chosen_template.format(X=formatted_entity,
                                                     Y=formatted_target).capitalize()
                # Unhandled case: converting present participle (-ing verb) into
                # present tense, e.g., the dataset would contain "Can {entity}
                # eating?" instead of "Can {entity} eat?"

                answer = answer.capitalize()
                if answer == "Yes":
                    statement = templates[relation][
                        "assertion_positive"].format(
                        X=formatted_entity, Y=formatted_target).capitalize()
                else:
                    assert answer == "No"
                    statement = templates[relation][
                        "assertion_negative"].format(
                        X=formatted_entity, Y=formatted_target).capitalize()
                row = [DATASET_NAME, id, nl_question, answer, statement,
                       statement]
                if id in training_samples_ids:
                    train_writer.writerow(row)
                else:
                    dev_writer.writerow(row)
                id += 1


if __name__ == "__main__":
    # python beliefbank_preprocess.py -f silver_facts_path
    args = parse_args()
    copy_file(args.demszky_train_path, OUT_TRAIN_FILE_NAME)
    copy_file(args.demszky_dev_path, OUT_DEV_FILE_NAME)
    f = open(args.uncountables, "r")
    uncountables = f.read().split("\n")
    f.close()
    build_and_write_facts(args.facts, uncountables,
                          args.templates, args.train_fraction)
