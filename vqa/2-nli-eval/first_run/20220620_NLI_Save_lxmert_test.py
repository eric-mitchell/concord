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

converter = qa_converter.QAConverter()
nlier = nli.NLIInferencer(model_hf_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", confidence_threshold=0.0,
                         dedup_constraints=False)
# qsolver = solver.Solver(beta=0.3)

### INSTRUCTION FOR USERS : INDICATE APPROPRIATE PATH
data_path = '/u/scr/nlp/data/nli-consistency/lxmert_results/lxmert-test-3pred-40token-1seed_predictions.json'

with open(data_path, 'r') as f:
    data = json.load(f)

### INSTRUCTION FOR USERS : INDICATE APPROPRIATE PATH
save_path = '/u/scr/nlp/data/nli-consistency/lxmert_results/lxmert-test-3pred-40token-1seed_predictions_nli.json'

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
        group_list = img_data[group]
        
        qs_list = []
        ans_list = []
        conf_list = []
        for i in range(len(group_list)):
            qs = group_list[i]
            qs_list.append(qs['question'].capitalize())
            ans_list.append(qs['prediction'][0:num_choices])
            conf_list.append(qs['prob'][0:num_choices])
            
        converted = converter(qs_list, ans_list)
        # save
        statement_groups = [[(converted[i][j], conf_list[i][j]) for j in range(len(converted[i]))] for i in range(len(converted))]
        # save
        converted_flat = [qs for list1 in converted for qs in list1]
        # save
        compared = nlier(converted_flat, group_count = group_count, not_redundant=not_redundant, fp_batch_size=64)
        if not repeated_comparisons:
            compared = list(set(compared))
            
        data[key][group] = {'orig':group_list, 'nli':{'statement_groups':statement_groups,'converted_flat':converted_flat,'compared':compared}}
        
    questions_done += 1        

print(questions_done)

with open(save_path, 'w') as f:
    json.dump(data, f)