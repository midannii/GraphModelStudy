#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source  ~/.bashrc
conda   activate   python-3.6

ml purge
ml load cuda/11.0

python -u experiments/predict_links.py with configs/rgcn/lp-FB-touta.yaml

echo "###"
echo "### END DATE=$(date)"