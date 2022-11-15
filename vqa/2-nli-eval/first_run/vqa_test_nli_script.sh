#!/bin/bash
nlprun -g 1 -p high -n vqa-lxmert-test-nli -o /u/scr/nlp/data/nli-consistency/lxmert_results/lxmert-test-nli.log 'bash vqa_test_nli_lxmert.sh'

nlprun -g 1 -p high -n vqa-vilt-test-nli -o /u/scr/nlp/data/nli-consistency/vilt_results/vilt-test-nli.log 'bash vqa_test_nli_vilt.sh'
