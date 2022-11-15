from argparse import ArgumentParser
import csv
import json
from nlic.nli import NLIInferencer
import time
import numpy as np


def make_parser():
    parser = ArgumentParser(
        description='Cache NLI results to save time')
    parser.add_argument('--qa_scores_cached_path', '-f', type=str,
                        default='data/cbqa/qa-cache/macaw-large/silver-facts-qa-scored.json')
    parser.add_argument('--model_hf_name', '-m', type=str,
                        default='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                        choices=[
                            'ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli',
                            'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                            'roberta-large-mnli'])
    parser.add_argument('--output_path', '-o', type=str,
                        default='data/cbqa/nli-cache/roberta-large-snli/nli-scored-silver-facts.csv')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    fp_batch_size = 512 if args.model_hf_name == 'ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli' else 1024

    with open(args.qa_scores_cached_path, 'r') as qa_file, open(args.output_path, 'w') as nli_file:
        # keep everything, since we are tuning confidence_threshold
        model = NLIInferencer(model_hf_name=args.model_hf_name,
                              confidence_threshold=0.0,
                              dedup_constraints=False)
        qa_results = json.load(qa_file)
        writer = csv.writer(nli_file)
        runtimes = []
        for _, fact_set in qa_results.items():
            entity_statements = []
            for _, fact in fact_set.items():
                entity_statements.append(fact['qaconv_positive'])
            start_time = time.time()
            nli_results = model(entity_statements, fp_batch_size=fp_batch_size)
            runtime = time.time() - start_time
            runtimes.append(runtime)
            writer.writerow([len(nli_results)])
            for res in nli_results:
                writer.writerow(res)
            print(fact['subject'] + ' ran for ' + str(runtime) + ' seconds')
        print(f'Average NLI execution time: {np.mean(runtimes)}')
