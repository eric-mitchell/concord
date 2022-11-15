import re
import transformers
import torch
import warnings

from typing import Callable


def in_f_macaw(questions):
    return ["$answer$ ; $mcoptions$ ; $question$ = " + q for q in questions]


def out_f_macaw(answers):
    return [(re.split("=|;", a)[1]).strip() for a in answers]


def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels


def gather_log_probs(logits, labels):
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def multiclass_log_probs(pred, targets, should_reduce=True):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    mask, targ = mask_hf_labels(targets)
    unmasked_log_probs = gather_log_probs(pred, targ)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(
            -1
        )  # We want to get the whole sequence right
    acc = correct.float()
    if should_reduce:
        acc = acc.mean()

    log_probs = (unmasked_log_probs * mask.float()).sum(-1)
    if should_reduce:
        log_probs = log_probs.mean()

    return log_probs, acc


class QA:
    def __init__(
        self,
        model_hf_name="allenai/macaw-large",
        device=None,
        in_format: Callable = None,
        out_format: Callable = None,
        mod_factory: Callable = None,
        tok_factory: Callable = None,
    ):
        self.model_hf_name = model_hf_name

        # To extend this wrapper to additional hf models not listed here, add
        # the name to hf_models, and provide a dictionary entry in hf_config
        # whose value is (input formatter, output formatter, tokenizer factory,
        # model factory)

        hf_models = [
            "allenai/macaw-large",
            "allenai/macaw-3b",
            "google/t5-small-ssm-nq",
            "google/t5-large-ssm-nq",
            "google/t5-3b-ssm-nq",
            "unc-nlp/lxmert-base-uncased",
        ]

        hf_mod_config = {
            "allenai/macaw-large": (
                in_format if in_format else in_f_macaw,
                out_format if out_format else out_f_macaw,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "allenai/macaw-3b": (
                in_format if in_format else in_f_macaw,
                out_format if out_format else out_f_macaw,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "google/t5-small-ssm-nq": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "google/t5-large-ssm-nq": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            "google/t5-3b-ssm-nq": (
                lambda x: x,
                lambda x: x,
                transformers.AutoTokenizer.from_pretrained,
                transformers.AutoModelForSeq2SeqLM.from_pretrained,
            ),
            # need to think about how to expand this to images + strings as
            # input
            # "unc-nlp/lxmert-base-uncased": (
            #     lambda x: x,
            #     lambda x: x,
            #     transformers.LxmertTokenizer,
            #     transformers.LxmertForQuestionAnswering,
            # ),
        }

        if model_hf_name not in hf_models:
            self.in_format = lambda x: x
            self.out_format = lambda x: x
            self.tok_factory = tok_factory
            self.mod_factory = mod_factory

            if any(
                factory is None
                for factory in [self.tok_factory, self.mod_factory]
            ):
                warnings.warn(
                    f"""
                    Behaviour may be undefined if model_hf_name is not in
                    {hf_models}, or mod_factory and tok_factory are unset.
                    
                    Defaults are:
                        mod: transformers.AutoModelForSeq2SeqLM.from_pretrained
                        tok: transformers.AutoTokenizer.from_pretrained
                        
                    If these defaults are correct this warning can be safely 
                    ignored.
                """
                )
        else:
            (
                self.in_format,
                self.out_format,
                self.tok_factory,
                self.mod_factory,
            ) = hf_mod_config[model_hf_name]

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.tokenizer = self.tok_factory(self.model_hf_name) if "t5-3b" not in self.model_hf_name \
            else self.tok_factory("google/t5-xl-ssm-nq")
        self.model = self.mod_factory(self.model_hf_name).to(self.device)
        self.model.eval()

    def __call__(self, questions, temp=0.5, n=4,
                 fp_batch_size: int = None, **kwargs):
        N = len(questions)
        if fp_batch_size is None:
            fp_batch_size = N

        fp_i_beg = 0
        answers = []
        probs = []
        with torch.inference_mode():
            do_sample = True
            if "do_sample" in kwargs:
                do_sample = kwargs["do_sample"]
                del kwargs["do_sample"]

            while fp_i_beg < N:
                fp_i_end = min(fp_i_beg + fp_batch_size, N)
                questions_batch = questions[fp_i_beg:fp_i_end]

                qa_inputs = self.tokenizer(
                    self.in_format(questions_batch), padding=True, return_tensors="pt"
                ).to(self.device)

                output = self.model.generate(
                    **qa_inputs,
                    max_new_tokens=25,
                    do_sample=do_sample,
                    temperature=temp,
                    num_return_sequences=n,
                    **kwargs,
                )
                qa_inputs = {
                    k: v.repeat_interleave(n, dim=0) for k, v in qa_inputs.items()
                }

                logits = self.model(**qa_inputs, labels=output).logits

                masked_output = output.masked_fill(
                    output == self.tokenizer.pad_token_id, -100
                )

                log_probs, acc = multiclass_log_probs(
                    logits, masked_output, should_reduce=False
                )

                batch_probs = log_probs.exp()
                probs.extend(batch_probs.cpu().numpy().tolist())
                batch_answers = self.out_format(
                    self.tokenizer.batch_decode(output, skip_special_tokens=True)
                )
                answers.extend(batch_answers)

                fp_i_beg += fp_batch_size

        return answers, probs


if __name__ == "__main__":
    qa_m = QA(model_hf_name="allenai/macaw-large")
    qa_t = QA(model_hf_name="google/t5-small-ssm-nq")

    questions = [
        "What is the capital of Afghanistan?",
        "Tbilisi is the capital of what country?",
        "Kabul is the capital of what country?",
    ]

    out_m = qa_m(questions)
    out_t = qa_t(questions)
    out_t_beam = qa_t(
        questions,
        n=4,
        num_beam_groups=4,
        num_beams=12,
        diversity_penalty=10.0,
        do_sample=False,
    )

    print("macaw:\n", out_m)
    print("t5:\n", out_t)
    print("t5-beam:\n", out_t_beam)
