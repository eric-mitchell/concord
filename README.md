# ConCoRD: Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference

This repository contains a high-level implementation of the system proposed in [our EMNLP 2022 paper](https://ericmitchell.ai/emnlp-2022-concord), as well as steps to reproduce the results presented therein.

TODO: explain downloading of Drive files

## NLI Models
* [RoBERTa Large ANLI](https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli?text=I+like+you.+I+love+you)
* [RoBERTa Large MNLI](https://huggingface.co/roberta-large-mnli)
* [ALBERT xxlarge](https://huggingface.co/ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli?text=I+like+you.+I+love+you)

## Closed-Book Question-Answering (Section 4.1)

The [BeliefBank dataset](https://allenai.org/data/beliefbank) contains:
* `calibration_facts.json`: dev set for hyperparameter tuning
* `silver_facts.json`: test set
* `constraints_v2.json`: golden constraints for evaluating consistency

QA models evaluated include:
* [Macaw Large](https://huggingface.co/allenai/macaw-large?text=%24answer%24+%3B+%24mcoptions%24+%3B+%24question%24+%3D+What+is+the+color+of+a+cloudy+sky%3F)
* [Macaw 3B](https://huggingface.co/allenai/macaw-3b?text=%24answer%24+%3B+%24mcoptions%24+%3B+%24question%24+%3D+What+is+the+color+of+a+cloudy+sky%3F)

Since ConCoRD does not modify the QA or NLI models, for efficiency we cache the inference results from the models on
BeliefBank data. The following sections walk through our full pipeline for generating results, but we have
also uploaded our cached inference results to the Drive folder if you would like to directly experiment with those instead.
All file paths are given relative to the top-level `nli-consistency/` directory.

### Preprocess BeliefBank
Preprocess calibration and silver facts by using pre-written templates to create question and answer pairs.
```commandline
python cbqa/preprocess.py -f data/cbqa/beliefbank-data-sep2021/calibration_facts.json -o {output file path}
```
Repeat for silver facts.

Cached file paths:
`data/cbqa/calibration_facts_preprocessed_by_entity.json`, `data/cbqa/silver_facts_preprocessed_by_entity.json`

### QA Inference
For each of Macaw large and 3B, generate a cache of QA results for each of the calibration and silver facts preprocessed results.

For example, for Macaw large and calibration facts:
```commandline
python -m cbqa.qa_score_dataset -m allenai/macaw-large -f data/cbqa/calibration_facts_preprocessed_by_entity.json -o {output file path}
```
Cached QA results are under `data/cbqa/qa-cache`

### NLI Inference
For each of the NLI models, run NLI inference between each question-answer pair.

For example, for RoBERTa large ANLI:
```commandline
python -m cbqa.nli_score_dataset -m ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli -f data/cbqa/qa-cache/macaw-large/calibration-facts-qa-scored.json -o {output file path}
```
Cached NLI results are under `data/cbqa/nli-cache`

### Hyperparamter Tuning (Appendix H.1.1)
Use the cached QA and NLI calibration facts results to facilitate tuning hyperparameters for the MaxSAT solver with hyperopt. Each QA-NLI model combination, along with QA-oracle,
is evaluated. Results are stored in files under `cbqa/tuned_hparams`, where you can also find our original runs.

For example, to tune Macaw large with RoBERTa large ANLI:
```commandline
python -m cbqa.main -m hparam -qa allenai/macaw-large --qa_scores_cached_path data/cbqa/qa-cache/macaw-large/calibration-facts-qa-scored.json -nli ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli --nli_scores_cached_path data/cbqa/nli-cache/roberta-large-snli/nli-scored-calibration-facts.csv
```

### Inference
Let's put it all together.

Evaluate each QA-NLI model combination using tuned hyperparameters on the silver facts.

For example, to evaluate Macaw large with RoBERTa large ANLI:
```commandline
python -m cbqa.main -qa allenai/macaw-large --qa_scores_cached_path data/cbqa/qa-cache/macaw-large/silver-facts-qa-scored.json -nli ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli --nli_scores_cached_path data/cbqa/nli-cache/roberta-large-snli/nli-scored-silver-facts.csv -v 
```
The `-v` flag enables verbose output that allows you to see exactly which beliefs were flipped or untouched.

The oracle (golden constraints as NLI relations) can be run on Macaw large as follows:
```commandline
python -m cbqa.main -qa allenai/macaw-large --qa_scores_cached_path data/cbqa/qa-cache/macaw-large/silver-facts-qa-scored.json --oracle -v
```
By using the cached QA and NLI results we included under `data/cbqa`, you can reproduce the numbers we report in Tables 1, 6, and 9 in the paper.

### Ablations
#### Relation Type
Add either `--ablation_keep_relation contradiction` or `--ablation_keep_relation entailment` when running `python -m cbqa.main` (as shown above) for hyperparameter tuning and inference.
Our results on the best NLI model for CBQA (RoBERTa ANLI) are reported in Table 5.

#### Entailment Correction
Pass `--disable_ec` when running `python -m cbqa.main` (as shown above) for hyperparameter tuning and inference. 
Our results on the best NLI model for CBQA (RoBERTa ANLI) are reported in Table 8.

## Visual Question-Answering (Section 4.2)

This experiment evaluates ConCoRD on related questions from [ConVQA](https://arijitray1993.github.io/ConVQA/) about images from the [Visual Genome](https://visualgenome.org/api/v0/api_home.html).
* The questions for hyperoptimization (referred to as 'train') is sampled from the file `L-ConVQA`
* The questions for final evaluation (referred to as 'test') is sampled from the file `L-ConVQA Test`

QA models evaluated include:
* Learning Cross-Modality Encoder Representations from Transformers ([LXMERT](https://github.com/airsplay/lxmert))
* Vision-and-Language Transformer ([ViLT](https://github.com/dandelin/ViLT))

Parameters (e.g., paths to datasets, CPU/GPU, etc.) are set by variables within each notebook.  Please make sure that all paths are indicated properly for respective user in sections marked as:

    ### INSTRUCTION FOR USERS : INDICATE APPROPRIATE PATH
    
### [QA Inference](./vqa/1-qa-prep)
For LXMERT, in lieu of the tokenizer provided by HuggingFace, we use the [token-to-text mapping](https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json) from the LXMERT Github repository. 

The QA conversion model that we use has a checkpoint available as a [cached model](https://drive.google.com/drive/u/1/folders/1cVU5mRpQbiwk_fhoGRtS9MzG68MxmDWj), and the [cached data](https://drive.google.com/drive/u/1/folders/1Ok4Cp4qP0gezF1ep4jAR0aFjqMXBLO3E) listed throughout this section are available on-line as well.

Use the notebook [`vg-data-selection.ipynb`](./vqa/1-qa-prep/vg-data-selection.ipynb) to sample images and questions from ConVQA for the 'train' set

QA inference is then performed in the following notebooks:

* [`lxmert-run-train-10000im-3pred-40token-1seed_predictions.ipynb`](./vqa/1-qa-prep/lxmert-run-train-10000im-3pred-40token-1seed.ipynb)
* [`lxmert-test-3pred-40token-1seed_predictions.ipynb`](./vqa/1-qa-prep/lxmert-test-3pred-40token-1seed.ipynb)
* [`vilt-run-train-10000im-3pred-40token-1seed_predictions.ipynb`](./vqa/1-qa-prep/vilt-run-train-10000im-3pred-40token-1seed.ipynb)
* [`vilt-test-3pred-40token-1seed_predictions.ipynb`](./vqa/1-qa-prep/vilt-test-3pred-40token-1seed.ipynb)

Cached data from the data sampling and QA inference available on Google Drive:

* `vg-data-2022-06-13-16:03:37-n=10000-seed=1.txt`
* `lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli.json`
* `lxmert-test-3pred-40token-1seed_predictions_nli.json`
* `vilt-run-train-10000im-3pred-40token-1seed_predictions_nli.json`
* `vilt-test-3pred-40token-1seed_predictions_nli.json`

### [NLI Inference](./vqa/2-nli-eval)
Evaluate the train/test set with various NLI models

Within the [`first_run`](./vqa/2-nli-eval/first_run) directory, evaluate using the ANLI model with:
* Two .ipynb notebooks for the train set:
    * <a href="./vqa/2-nli-eval/first_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) lxmert SAVE_NLI.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) lxmert SAVE_NLI.ipynb`</a>
    * <a href="./vqa/2-nli-eval/first_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vilt SAVE_NLI.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vilt SAVE_NLI.ipynb`</a>
* Two .py files for the test set:
    * [`20220620_NLI_Save_lxmert_test.py`](./vqa/2-nli-eval/first_run/20220620_NLI_Save_lxmert_test.py)
    * [`20220620_NLI_Save_vilt_test.py`](./vqa/2-nli-eval/first_run/20220620_NLI_Save_vilt_test.py)
        
Within the [`second_run`](./vqa/2-nli-eval/second_run) directory, evaluating using the MNLI and XXLARGE models with:
* [`NLI_Save_10000_images_num_answers=2_not_redundant=True_repeated_comparisons=False_vqa_lxmert-models-mnli-xxlarge.py`](./vqa/2-nli-eval/second_run/NLI_Save_10000_images_num_answers=2_not_redundant=True_repeated_comparisons=False_vqa_lxmert-models-mnli-xxlarge.py)
    * See its use in [`vqa_lxmert_models_run.sh`](./vqa/2-nli-eval/second_run/vqa_lxmert_models_run.sh)
* <a href="./vqa/2-nli-eval/second_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa lxmert-test-mnli-EMERGENCY.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa lxmert-test-mnli-EMERGENCY.ipynb`</a>
* <a href="./vqa/2-nli-eval/second_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa lxmert-test-xxlarge-EMERGENCY.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa lxmert-test-xxlarge-EMERGENCY.ipynb`</a>
* <a href="./vqa/2-nli-eval/second_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-test-mnli-EMERGENCY.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-test-mnli-EMERGENCY.ipynb`</a>
* <a href="./vqa/2-nli-eval/second_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-test-xxlarge-EMERGENCY.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-test-xxlarge-EMERGENCY.ipynb`</a>
* <a href="./vqa/2-nli-eval/second_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-val-mnli-EMERGENCY.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-val-mnli-EMERGENCY.ipynb`</a>
* <a href="./vqa/2-nli-eval/second_run/20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-val-xxlarge-EMERGENCY.ipynb">`20220525 NLI Save (10000 images; num_answers = 2; num_choices = 2; not_redundant = True; repeated_comparisons = False, group_count = num_choices) vqa vilt-val-xxlarge-EMERGENCY.ipynb`</a>

Cached data from NLI Inference available on Google Drive:
* `lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli-xxlarge.json`
* `lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json`
* `lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli.json`
* `lxmert-test-3pred-40token-1seed_predictions_nli-xxlarge.json`
* `lxmert-test-3pred-40token-1seed_predictions_nli-mnli.json`
* `lxmert-test-3pred-40token-1seed_predictions_nli.json`
* `vilt-run-train-10000im-3pred-40token-1seed_predictions_nli-xxlarge.json`
* `vilt-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json`
* `vilt-run-train-10000im-3pred-40token-1seed_predictions_nli.json`
* `vilt-test-3pred-40token-1seed_predictions_nli-xxlarge.json`
* `vilt-test-3pred-40token-1seed_predictions_nli-mnli.json`
* `vilt-test-3pred-40token-1seed_predictions_nli.json`

### [Hyperparameter tuning](./vqa/3-hyperparam-opt)
Tune the hyperparameters on the train set, searching for the optimal NLI model, use of entailment correction and λ and β values

The main file that optimizes for the hyperparameters: [`visual_tune_mod.py`](./vqa/3-hyperparam-opt/visual_tune_mod.py)
* The use of the file and its flags are outlined in [`visual_tune_table_6.sh`](./vqa/3-hyperparam-opt/visual_tune_table_6.sh)
* -f is for source of answers & nli outputs
* -o is the trial outputs
* -t is the number of trials
* -w indicates use of entailment correction

Here is an example of the use of [`visual_tune_mode.py`](./vqa/3-hyperparam-opt/visual_tune_mod.py)
```commandline
python3 visual_tune_mod.py -f vilt-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json -o vilt-table6-mnli-nwe.trials -t 100 > vilt-table6-mnli-nwe.log
```

Optimal hyperparameters were manually noted and used for the next (final) step

### [ConCoRD Evaluation](./vqa/4-final-eval)
Evaluate on the test set using the hyperparameters determined from step 3

The first main cell in the notebook <a href="./vqa/4-final-eval/20221019 vqa solve only test with opt_with timeout counter_with ablation and perfect consistency.ipynb">`20221019 vqa solve only test with opt_with timeout counter_with ablation and perfect consistency.ipynb`</a> contains the function for the final evaluation on the test set based on given hyperparameters.

The subsequent four cells contain outputs for the main results in section 4.3

## Test-time Information Injection (Section 4.3)
### QA Models
* [T5-Small-NQ](https://huggingface.co/google/t5-small-ssm-nq)
* [T5-Large-NQ](https://huggingface.co/google/t5-large-ssm-nq)
* [T5-3B-NQ](https://huggingface.co/google/t5-3b-ssm-nq)

### Before you start
In the semantic_filtering directory:
```commandline
mkdir hyperparam_search
mkdir eval_results
```

### Run Base Results
```commandline
export CACHE_DIR=<directory where you want to store cached datasets, this is for huggingface caching>
export STORE_DIR=<your root directory for downloaded data files>/nq/
python3 eval_retrieve.py --mode=base --split={test, val} --model={t5-small, t5-large, t5-3b} --cache_dir=$CACHE_DIR --store_dir=$STORE_DIR
```
These should give you the baseline results reported in Section 4.3.

### Run Oracle Results
Our intermediate data files are stored under the name `cache_{val, test}_0.8_4_t5-{small, large, 3b}.jsonl`. 0.8 and 4 correspond to the temperature and the number of responses we asked the QA model to generate, respectively.

To obtain the oracle results (upper bound of our results), run the following:
```commandline
export CACHE_ID=<path to the intermediate data file of choice>
export RESULT_FILE=<filename for your result file>
python3 eval_answers.py --cache_id=<CACHE_ID> --result_file=<RESULT_FILE>
```

### Run ConCoRD Results
```commandline
export CACHE_DIR=<directory where you want to store cached datasets, this is for huggingface caching>
export STORE_DIR=<your root directory for downloaded data files>/nq/
python3 eval_retrieve.py --mode=gold --split={test, val} --model={t5-small, t5-large, t5-3b} --cache_dir=$CACHE_DIR --store_dir=$STORE_DIR
```

### Run Ablations on Relation Types
Add the flag `entail_only` for entailment-only results, and `contradiction_only` for contradiction-only results to the commands above.

### Running Hyperparam Search
```commandline
export CACHE_DIR=<directory where you want to store cached datasets, this is for huggingface caching>
export STORE_DIR=<your root directory for downloaded data files>/nq/
python3 eval_retrieve.py --model={t5-small, t5-large, t5-3b} --cache_dir=$CACHE_DIR --store_dir=$STORE_DIR
```
The hyperparameter search might take 3 hours or longer depending on the amount of compute available. The results will be printed, or you can find the results stored in the `hyperparam_search` directory.

