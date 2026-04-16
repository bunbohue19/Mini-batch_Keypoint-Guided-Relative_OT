#!/bin/bash
# KPG-RL + mini-batch POT (partial OT) on digits datasets
#
# Hyperparameters follow the baseline exp_mPOT.sh (epsilon=0.1, mass=0.85)
# plus KPG-RL defaults.
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/digits
#   bash sh/exp_KPG_mPOT.sh

set -e
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
echo "Using GPU ${GPU}"

METHOD=jpmbot       # partial OT
K=1
M=500
EPOCH=100
TEST_INTERVAL=1
CLASS=10
EPSILON=0.1         # entropic regularisation
TAU=1.0
ETA1=0.1
ETA2=0.1
MASS=0.85           # fraction of mass to transport
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
    echo "=== ${SRC} → ${TGT}  (KPG + mPOT) ==="
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
        --tau       ${TAU} \
        --mass      ${MASS} \
        --lr        ${LR} \
        --eta1      ${ETA1} \
        --eta2      ${ETA2} \
        --num_workers 8 \
        --seed      ${SEED} \
        --use_kpg \
        --alpha     ${ALPHA} \
        --tau_s     ${TAU_S} \
        --tau_t     ${TAU_T}
    echo "=== Done: ${SRC} → ${TGT} ==="
done
