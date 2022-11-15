import transformers
import torch
from typing import Callable, List
from collections import defaultdict


def filter_c_list(c_list: List, allow_bidirectional_entail: bool = True):
    """
    Max filter within each canonical key so it contains only entailments *or*
    contradictions, but not both.
    """

    c_list = list(set(c_list)) # might have duplicates if a statement appears >1 time

    if len(c_list) <= 1:
        return c_list

    entailments = [c for c in c_list if c[2] == "entailment"]
    contradictions = [c for c in c_list if c[2] == "contradiction"]
    assert len(contradictions) <= 1, "Can't have contradiction between same pair with different probs"

    max_e = max([c[-1] for c in entailments]) if len(entailments) else 0
    max_c = max([c[-1] for c in contradictions]) if len(contradictions) else 0
    if max_e > max_c: # max_e == max_c is basically impossible due to floating point probabilities
        if allow_bidirectional_entail:
            return entailments
        else:
            return [max(entailments, key=lambda c: c[-1])]
    else:
        return contradictions


class NLIInferencer:
    def __init__(
        self,
        model_hf_name="ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
        device=None,
        confidence_threshold: float = 0.5,
        contradiction_prob_combine_fn: str = "min",
        dedup_constraints: bool = True,
        allow_bidirectional_entail: bool = True,
    ):

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_hf_name
        ).to(
            self.device
        )

        self.tok = transformers.AutoTokenizer.from_pretrained(model_hf_name)

        self.confidence_threshold = confidence_threshold
        self.dedup_constraints = dedup_constraints
        self.allow_bidirectional_entail = allow_bidirectional_entail

        if contradiction_prob_combine_fn == "min":
            self.contradiction_reduction = min
        elif contradiction_prob_combine_fn == "max":
            self.contradiction_reduction = max
        elif contradiction_prob_combine_fn == "mean":
            self.contradiction_reduction = lambda x, y: (x + y) / 2
        else:
            raise ValueError(
                f"Invalid combine function {contradiction_prob_combine_fn}; "
                + "expected one of ['min', 'max', 'mean']"
            )

        self.SEP = " "

        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.id2label = {
            k: v.lower() for k, v in self.model.config.id2label.items()
        }
        self.ENT_IDX = label2id["entailment"]
        self.CONT_IDX = label2id["contradiction"]

    def __call__(
        self,
        batch,
        should_evaluate_fn: Callable = None,
        should_evaluate_pairs: list = None,
        fp_batch_size: int = None,
        group_count=0,
        not_redundant: bool = False,
    ):
        """
        `batch` should be a list of strings (statements)
        `fp_batch_size` if not None controls batch size for forward
            passess of the pretrained NLI model
        `should_evaluate_fn` is an optional function that takes as input
            the tuple (statement_1_idx, statement_1, statement_2_idx,
            statement_2) and decides if the NLI model should actually be run on
            the pair of statement_1 and statement_2; the indices passed in are
            the indices of the statements within the `batch`
        `should_evaluate_pairs` is a list of *unordered* pairs of
            indices, which are the pairs of statements in the batch
            that should actually be evaluated
        `group_count` is an integer that, if greater than 0, indicates
            skipping nli comparisons within groups of size group_count
        `not_redundant` is a bool that indicates skipping of nli
            comparisons between entries with identical statements

        RETURN:
            list of tuples, where each tuple is: (statement_1,
            statement_2, RELATION_TYPE, confidence), where RELATION_TYPE
            may be 'entailment' or 'contradiction')
        """

        assert (
            should_evaluate_fn is None or should_evaluate_pairs is None
        ), "Can only take one of eval fn and eval pairs"

        input_idxs = []
        input_strs = []

        for idx, x in enumerate(batch):
            for jdx, x_ in enumerate(batch):
                if idx == jdx:
                    continue

                if group_count > 0:
                    assert len(batch) % group_count == 0, (
                        f"The batch size ({len(batch)}) should be divisible "
                        + f"by the group count ({group_count})"
                    )

                    if int(idx / group_count) == int(jdx / group_count):
                        # Skip within group comparisons
                        continue

                if not_redundant:
                    if x == x_:
                        # Skip identical statements
                        continue

                if should_evaluate_fn:
                    should_eval = should_evaluate_fn(idx, x, jdx, x_)
                elif should_evaluate_pairs:
                    should_eval = ((idx, jdx) in should_evaluate_pairs) or (
                        (jdx, idx) in should_evaluate_pairs
                    )
                else:
                    should_eval = True

                if should_eval:
                    input_idxs.append((idx, jdx))
                    input_strs.append((x, x_))

        result = {"entailment": {}, "contradiction": {}}

        # Set up moving indices for start, end, and size of batches for
        # forward passes
        N = len(input_idxs)
        if fp_batch_size is None:
            fp_batch_size = N

        fp_i_beg = 0

        with torch.inference_mode():
            # 'Batch' forward passes into sub-batches to fit in GPU memory
            while fp_i_beg < N:
                fp_i_end = min(fp_i_beg + fp_batch_size, N)

                input_idxs_batch = input_idxs[fp_i_beg:fp_i_end]
                input_strs_batch = input_strs[fp_i_beg:fp_i_end]

                inputs = self.tok(
                    [x + self.SEP + x_ for x, x_ in input_strs_batch],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                outputs = self.model(**inputs).logits.softmax(-1).cpu()

                for (idx, jdx), pred, prob in zip(
                    input_idxs_batch, outputs.argmax(-1), outputs.max(-1).values
                ):
                    assert idx != jdx, (
                        "We tried to compare a statement with itself; "
                        + "this should never happen"
                    )

                    outcome = self.id2label[pred.item()]

                    if outcome == "entailment":
                        # entailments are ordered
                        if prob > self.confidence_threshold:
                            result[outcome][(idx, jdx)] = prob.item()

                    elif outcome == "contradiction":
                        idx_pair = (min(idx, jdx), max(idx, jdx))

                        # contradictions are not ordered, so we use a
                        # canonical ordering of the nodes
                        if idx_pair not in result[outcome]:
                            result[outcome][idx_pair] = prob.item()
                        else:
                            result[outcome][
                                idx_pair
                            ] = self.contradiction_reduction(
                                result[outcome][idx_pair], prob.item()
                            )

                fp_i_beg += fp_batch_size

            result["contradiction"] = {
                k: v
                for k, v in result["contradiction"].items()
                if v > self.confidence_threshold
            }

            constraint_groups = defaultdict(list)
            for relation in result.keys():
                for (idx, jdx), prob in result[relation].items():
                    a, b = batch[idx], batch[jdx]
                    canonical_key = (min(a, b), max(a, b))
                    if relation == "contradiction":
                        # contradictions are symmetric, so we can canonicalize the statement ordering
                        # we need to do this so we don't get
                        a, b = canonical_key

                    constraint_groups[canonical_key].append((a, b, relation, prob))

            if self.dedup_constraints:
                filtered = [
                    filter_c_list(c_list, self.allow_bidirectional_entail)
                    for c_list in constraint_groups.values()
                ]

                solver_formatted = [c for c_list in filtered for c in c_list]

            else:
                solver_formatted = [
                    c for c_list in constraint_groups.values() for c in c_list
                ]

            return solver_formatted


if __name__ == "__main__":
    inf = NLIInferencer(
        model_hf_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        dedup_constraints=True,
        allow_bidirectional_entail=True,
    )

    statements = [
        "A dog is a mammal.",
        "A dog is a fish.",
        "A dog is a mammal.",
        "A dog is a fish.",
    ]

    outputs = inf(statements)
    print(outputs)