#!/bin/bash
# KPG-RL + BoMb-UOT (batch-of-mini-batches, unbalanced) on digits datasets
#
# Uses k=2 mini-batches of size 500 each (total batch = 1000).
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/digits
#   bash sh/exp_KPG_BoMbUOT.sh

set -e
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
echo "Using GPU ${GPU}"

METHOD=jumbot       # unbalanced OT
K=2                 # number of mini-batches in BoMb hierarchy
M=500               # per-mini-batch size
EPOCH=100
TEST_INTERVAL=1
CLASS=10
EPSILON=0.1         # Sinkhorn regularisation (required >0 for unbalanced)
BE=0.0              # exact EMD for outer k*k OT
TAU=1.0             # marginal relaxation
ETA1=0.1
ETA2=0.1
MASS=0.85
LR=4e-4
SEED=1980

# KPG-RL parameters
ALPHA=0.5
TAU_S=0.1
TAU_T=0.1

TASKS=(
    "svhn   mnist"
    "usps   mnist"
    "mnist  usps"
)

for ENTRY in "${TASKS[@]}"; do
    read -r SRC TGT <<< "${ENTRY}"
    echo ""
    echo "=== ${SRC} → ${TGT}  (KPG + BoMb-UOT, k=${K}) ==="
    python train_digits.py \
        --gpu_id    ${GPU} \
        --method    ${METHOD} \
        --source_ds ${SRC} \
        --target_ds ${TGT} \
        --k         ${K} \
        --mbsize    ${M} \
        --n_epochs  ${EPOCH} \
        --test_interval ${TEST_INTERVAL} \
        --nclass    ${CLASS} \
        --epsilon   ${EPSILON} \
        --batch_epsilon ${BE} \
        --tau       ${TAU} \
        --mass      ${MASS} \
        --lr        ${LR} \
        --eta1      ${ETA1} \
        --eta2      ${ETA2} \
        --num_workers 8 \
        --seed      ${SEED} \
        --use_bomb \
        --use_kpg \
        --alpha     ${ALPHA} \
        --tau_s     ${TAU_S} \
        --tau_t     ${TAU_T}
    echo "=== Done: ${SRC} → ${TGT} ==="
done
