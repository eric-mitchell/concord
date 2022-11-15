import transformers
import torch
from typing import List
import numpy as np

class QAConverter:
    def __init__(self, model_name="t5-base",
                model_weights_path="/u/scr/nlp/data/nli-consistency/qa_converter_models/t5-statement-conversion-finetune.pt",
                device=None):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tok = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model.load_state_dict(torch.load(model_weights_path, map_location="cpu")["model"])
        self.model.to(self.device)

    def __call__(self, questions: List[str], answers: List[List[str]]) -> List[List[str]]:
        '''
        `questions` should be a list of strings
        `answers` should be a list of answer choices !!!for each question!!!
            in other words, `answers` is a list of lists of strings

        RETURN:
            list of lists of statements, in the same grouping as the input `answers`
        '''
        qa_pairs = []
        for q, as_ in zip(questions, answers):
            for a in as_:
                qa_pairs.append(q + " " + a)
        inputs = self.tok(qa_pairs, padding=True, return_tensors="pt").to(self.device)
        rephrased = self.tok.batch_decode(self.model.generate(**inputs, max_length=128), skip_special_tokens=True)

        ans_cumsum = np.cumsum([0] + [len(a) for a in answers])
        grouped = [rephrased[idx:jdx] for idx, jdx in zip(ans_cumsum[:-1], ans_cumsum[1:])]

        return grouped


if __name__ == "__main__":
    converter = QAConverter()
    questions = ["Who is the UK prime minister?", "What team does Messi play for?"]
    answers = [
        ["Boris Johnson", "Theresa May", "David Cameron"],
        ["PSG", "Barcelona"]
    ]
    outputs = converter(questions, answers)
    print(outputs)
