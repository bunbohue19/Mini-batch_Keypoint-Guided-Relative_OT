#!/bin/bash
# KPG-RL + mini-batch OT (balanced, exact EMD) on digits datasets
#
# Runs all 3 adaptation tasks from Table 1 of the m-POT paper:
#   SVHN→MNIST, USPS→MNIST, MNIST→USPS
#
# Hyperparameters follow the baseline exp_mOT.sh (eta1=0.1, eta2=0.1,
# lr=4e-4, m=500, k=1, epsilon=0) plus KPG-RL defaults (alpha=0.5,
# tau_s=0.1, tau_t=0.1).
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/digits
#   bash sh/train_KPG_mOT.sh

set -e
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
echo "Using GPU ${GPU}"

METHOD=jdot         # balanced OT
K=1
M=500
EPOCH=100
TEST_INTERVAL=1
CLASS=10
EPSILON=0.0         # exact EMD
TAU=1.0
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
    echo "=== ${SRC} → ${TGT}  (KPG + mOT) ==="
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
        --tau_t     ${TAU_T} \
        --data_dir /home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Mini-batch_KPG-RL_OT/data
    echo "=== Done: ${SRC} → ${TGT} ==="
done
