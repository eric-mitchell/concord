import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pysat.formula import IDPool, WCNFPlus
from pysat.examples.rc2 import RC2
import numpy as np
import matplotlib.pyplot as plt
import json

# custom modules
import sys
sys.path.append('./nlic')
import qa_converter
import nli
import solver

device = "cuda"

### INSTRUCTION FOR USERS : INDICATE APPROPRIATE PATH
data_path = '/data/nli-consistency/vqa/lxmert-test-3pred-40token-1seed_predictions_nli.json'

models = ["ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
              "roberta-large-mnli"]

### INSTRUCTION FOR USERS : INDICATE APPROPRIATE PATH
save_paths = ['/data/nli-consistency/vqa/lxmert-test-3pred-40token-1seed_predictions_nli-xxlarge.json', 
              '/data/nli-consistency/vqa/lxmert-test-3pred-40token-1seed_predictions_nli-mnli.json']

for m_name, save_path in zip(models, save_paths):
    print('Model:', m_name)
    print('Save path:', save_path)
    print('Data path:', data_path)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    nlier = nli.NLIInferencer(model_hf_name=m_name, confidence_threshold=0.0,
                         dedup_constraints=False)

    # Normalize

    num_choices = 2
    not_redundant = True # I.e. self comparisons -- it doesn't add much value since false will still yield truth value
    repeated_comparisons = False # I.e. identical comparisons; as in what if there are multiple answer confidence levels? No because regardless based on single statement
    group_count = num_choices

    raw_correct = 0
    new_correct = 0
    good_change = 0
    bad_change = 0
    count = 0
    questions_done = 0

    for key in data.keys():
        print('image #:', questions_done + 1, 'image number', key)
        img_data = data[key]
        
        for group in img_data.keys():
            group_list = img_data[group]['orig']
            nli_stuff = img_data[group]['nli']
            
            statement_groups = nli_stuff['statement_groups']
            converted_flat = nli_stuff['converted_flat']
            
            compared = nlier(converted_flat, group_count = group_count, not_redundant=not_redundant)
            if not repeated_comparisons:
                compared = list(set(compared))
            
            data[key][group]['nli']['compared'] = compared
        questions_done += 1        

    print(questions_done)
    
    with open(save_path, 'w') as f:
        json.dump(data, f)
