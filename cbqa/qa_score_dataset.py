import json
import argparse
import numpy as np
import time

from typing import List, Dict, Tuple

from nlic.qa import QA
from nlic.qa_converter import QAConverter

YES = 'Yes.'
NO = 'No.'

# input formatter for yes / no questions to Macaw
def in_f_macaw_yn(questions: List[str]) -> List[str]:
    return [
        f'$answer$ ; $mcoptions$ = (A) {YES} (B) {NO} ; $question$ = ' + q
        for q in questions
    ]


def score_batch(batch: Dict, qa: QA, qa_converter: QAConverter, fp_batch_size: int = None) -> Tuple[Dict, float]:

    indices = [int(i) for i in batch.keys()]

    questions = [batch[str(i)]['nl_question'] for i in indices]

    start_time = time.time()
    answers, probs = qa(questions, temp=None, n=1, do_sample=False, fp_batch_size=fp_batch_size)
    runtime = time.time() - start_time

    # qa_converter expects a list of lists of answers; we give one answer
    # per question (b/c all questions are yes / no)
    list_answers = [[YES] for a in answers]

    statement_groups = qa_converter(questions, list_answers)

    for i in indices:
        batch[str(i)]['qa_ans'] = answers[i]
        batch[str(i)]['qa_ans_p'] = probs[i]
        batch[str(i)]['qaconv_positive'] = statement_groups[i][0]


    return batch, runtime


def score_dataset(model_hf_name: str, dset: Dict) -> Dict:

    qa = QA(model_hf_name=model_hf_name, in_format=in_f_macaw_yn)

    qa_converter = QAConverter(
        model_weights_path='data/models/qa-converter/t5-statement-conversion-finetune.pt'
    )

    indices = [i for i in dset.keys()]
    fp_batch_size = 128 if model_hf_name == 'allenai/macaw-3b' else None
    qa_runtimes = []
    for i in indices:
        scored_batch, runtime = score_batch(dset[i], qa, qa_converter, fp_batch_size)
        dset[i] = scored_batch
        qa_runtimes.append(runtime)
        print(f'{dset[i]["0"]["subject"]} QA ran for {runtime} seconds')

    print(f'Average QA runtime per-entity: {np.mean(qa_runtimes)}')
    return dset


def main(model, infile, outfile):
    dataset = json.load(infile)
    scored_dataset = score_dataset(model_hf_name=model, dset=dataset)
    json.dump(scored_dataset, outfile)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Score a pre-processed fact file'
    )

    parser.add_argument(
        '--qa_model', 
        '-m',
        type=str,
        default='allenai/macaw-large',
        choices=['allenai/macaw-large', 'allenai/macaw-3b']
    )

    parser.add_argument(
        '-f',
        '--file',
        help='Pre-processed input file.',
        type=argparse.FileType('r'),
    )

    parser.add_argument(
        '-o', '--output', help='File to write.', type=argparse.FileType('w')
    )

    args = parser.parse_args()

    main(args.qa_model, args.file, args.output)
