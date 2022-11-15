#!/bin/bash

echo test1

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate nli-consistency-vilt

echo test2

python3 20220620_NLI_Save_vilt_test.py
