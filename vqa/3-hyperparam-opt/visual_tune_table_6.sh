#!/bin/bash

#echo "LXMERT Tuning 10000 normal nli"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli.json -o /data/nli-consistency/vqa/lxmert-table6-normal.trials -w -t 100 > /data/nli-consistency/vqa/lxmert-table6-normal.log

#echo "LXMERT Tuning 10000 xxlarge nli"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli-xxlarge.json -o /data/nli-consistency/vqa/lxmert-table6-xxlarge-add.trials -w -t 100 > /data/nli-consistency/vqa/lxmert-table6-xxlarge-add.log &

#echo "LXMERT Tuning 10000 mnli"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json -o /data/nli-consistency/vqa/lxmert-table6-mnli-add.trials -w -t 100 > /data/nli-consistency/vqa/lxmert-table6-mnli-add.log &

#echo "LXMERT Tuning 10000 normal nli nwe"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli.json -o /data/nli-consistency/vqa/lxmert-table6-normal-nwe.trials -t 100 > /data/nli-consistency/vqa/lxmert-table6-normal-nwe.log &

#echo "LXMERT Tuning 10000 xxlarge nli nwe"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli-xxlarge.json -o /data/nli-consistency/vqa/lxmert-table6-xxlarge-nwe.trials -t 100 > /data/nli-consistency/vqa/lxmert-table6-xxlarge-nwe.log &

#echo "LXMERT Tuning 10000 mnli nwe"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/lxmert-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json -o /data/nli-consistency/vqa/lxmert-table6-mnli-nwe.trials -t 100 > /data/nli-consistency/vqa/lxmert-table6-mnli-nwe.log &

#echo "ViLT Tuning normal we"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/vilt-run-train-10000im-3pred-40token-1seed_predictions_nli.json -o /data/nli-consistency/vqa/vilt-table6-normal.trials -w -t 100 > /data/nli-consistency/vqa/vilt-table6-normal.log

#echo "ViLT Tuning normal nwe"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/vilt-run-train-10000im-3pred-40token-1seed_predictions_nli.json -o /data/nli-consistency/vqa/vilt-table6-normal-nwe.trials -t 100 > /data/nli-consistency/vqa/vilt-table6-normal-nwe.log

#echo "ViLT Tuning mnli we"
#python3 visual_tune_mod.py -f /data/nli-consistency/vqa/vilt-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json -o /data/nli-consistency/vqa/vilt-table6-mnli.trials -w -t 100 > /data/nli-consistency/vqa/vilt-table6-mnli.log

echo "ViLT Tuning mnli nwe"
python3 visual_tune_mod.py -f /u/scr/nlp/data/nli-consistency/grendel_archive/vqa/vilt-run-train-10000im-3pred-40token-1seed_predictions_nli-mnli.json -o /u/scr/nlp/data/nli-consistency/vqa-camera/vilt-table6-mnli-nwe.trials -t 100 > /u/scr/nlp/data/nli-consistency/vqa-camera/vilt-table6-mnli-nwe.log
