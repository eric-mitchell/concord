#!/bin/bash

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate nli-consistency-vilt

python3 20220620_NLI_Save_lxmert_test.py
