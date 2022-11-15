import argparse
import json
import random

TEMPLATES_PATH = 'cbqa/templates.json'
UNCOUNTABLES_PATH = 'cbqa/non_countable.txt'


def noun_fluenterer(noun, uncountables, relation=None):
    """
    Make a noun phrase 'fluenter' (more fluent) before putting it in a
    template.  note we only a.) check if the noun is in a list of known
    non-countables or has a relation with a certain type, and b.) look at the
    first letter of the input to decide whether to put a or an.

    :param noun: the noun (phrase) -- subject or object -- to make more fluent
    :param uncountables: the list of uncountables to compare to
    :param relation: BeliefBank relation
    :return: a string with the prettified noun phrase
    """

    if noun in uncountables:
        return noun

    if relation is not None:
        if relation in ['CapableOf', 'MadeOf', 'HasProperty']:
            return noun

    if noun[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an ' + noun

    return 'a ' + noun


def create_datum(subject, predicate, answer, template, uncountables):
    """
    Create a data point (datum) from a subject, predicate, answer, using the
    provided template.

    :param subject: The subject (a string)
    :param predicate: The predicate (a string, relation and entity, comma
            separated)
    :param answer: "yes" or "no"
    :param template: the templates, a dict
    :param uncountables: a list of uncountable noun phrases
    :return: datum, a dict
    """
    relation, obj = predicate.split(',')

    X = noun_fluenterer(subject, uncountables)
    Y = noun_fluenterer(obj, uncountables, relation)

    this_template = template[relation]
    nl_question = random.choice(this_template['templates']).format(X=X, Y=Y)

    if answer == 'yes':
        nl_correct_answer = 'Yes.'
    else:
        nl_correct_answer = 'No.'

    nl_assertion_yes = this_template['assertion_positive'].format(X=X, Y=Y)
    nl_assertion_no = this_template['assertion_negative'].format(X=X, Y=Y)

    unbound_yes = this_template['assertion_positive'].format(X='it', Y=Y)

    return {
        'subject': subject,
        'relation': relation,
        'obj': obj,
        'nl_question': nl_question,
        'nl_correct_answer': nl_correct_answer,
        'Yes.': nl_assertion_yes,
        'No.': nl_assertion_no,
        'unbound_yes': unbound_yes,
    }


def create_dataset_from_full_fact_file(
    infile, uncountables, template_path=TEMPLATES_PATH
):
    """
    Create a dataset from the full set of facts in the fact file given by
    infile, with one subject per batch, and where each batch is the complete
    set of facts.

    :param infile: the path to the fact file to read
    :param uncountables: a list of uncountable noun phrases
    :param template_path:
    :return: a dataset
    """

    facts = json.load(infile)
    template = json.load(open(template_path, 'r'))

    batch_id = 0

    dataset = {}

    subjects = facts.keys()

    for subject in subjects:
        this_batch = {}

        related_facts = facts[subject]
        predicates = related_facts.keys()

        datum_id = 0

        for predicate in predicates:
            datum = create_datum(
                subject=subject,
                predicate=predicate,
                answer=related_facts[predicate],
                template=template,
                uncountables=uncountables,
            )

            this_batch[datum_id] = datum
            datum_id += 1

        dataset[batch_id] = this_batch
        batch_id += 1

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Build a dataset of questions from a fact file'
    )

    parser.add_argument(
        '-f', '--file', help='Fact file to input', type=argparse.FileType('r')
    )

    parser.add_argument(
        '-o', '--output', help='File to write.', type=argparse.FileType('w')
    )

    parser.add_argument(
        '-t', '--test', action='store_true', help='test the batch builder'
    )

    args = parser.parse_args()

    if (args.file is None or args.output is None) and not args.test:
        print('If not testing, need -f and -o.')
        quit()

    f = open(UNCOUNTABLES_PATH, 'r')
    uncountables = f.read().split('\n')
    f.close()

    def main(infile, outfile):
        dataset = create_dataset_from_full_fact_file(infile, uncountables)

        json.dump(dataset, outfile)

        # sets for each thing we want to count

        subjects = set()
        predicates = set()
        relations = set()
        objects = set()

        n_q = 0

        for batch in dataset.keys():
            for datum in dataset[batch].keys():
                relation = dataset[batch][datum]['relation']
                obj = dataset[batch][datum]['obj']
                predicate = relation + ',' + obj

                subjects = subjects.union({dataset[batch][datum]['subject']})
                predicates = predicates.union({predicate})
                relations = relations.union({relation})
                objects = objects.union({obj})

                n_q += 1

        n_s = len(subjects)
        n_p = len(predicates)
        n_r = len(relations)
        n_o = len(objects)

        desc = (
            f'The dataset {outfile.name} has {n_s} subjects, {n_o} objects, \n'
            + f'{n_r} relations, {n_p} predicates, and {n_q} facts.  \n'
        )

        f = open(outfile.name + '.description.txt', 'w')
        f.write(desc)
        f.close()

        return 0

    if args.test:
        new_batch = create_dataset_from_full_fact_file(
            infile=open('data/calibration_facts.json', 'r'),
            uncountables=uncountables,
        )

        print(new_batch)
    else:
        main(args.file, args.output)
