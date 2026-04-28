#!/bin/bash
# KPG-RL + mini-batch OT for CIFAR-10 generative model
# Baseline: k=2, m=100, method=OT, reg=0 (Table 12 of m-POT paper)
set -e
GPU=${1:-0}
cd "$(dirname "$0")/.."

python main_cifar.py \
    --gpu-id ${GPU} \
    --method OT --reg 0 \
    --k 2 --m 100 --epochs 100 \
    --lr 0.0005 --seed 16 --latent-size 128 \
    --fid-each 5 --L 1000 \
    --use-kpg --n-kp 5 --alpha 0.5 --tau-s 0.1 --tau-t 0.1