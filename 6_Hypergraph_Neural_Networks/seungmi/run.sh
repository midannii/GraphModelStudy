#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source  ~/.bashrc
conda   activate   parlai

ml purge
ml load cuda/11.0

python -u train.py

echo "###"
echo "### END DATE=$(date)"