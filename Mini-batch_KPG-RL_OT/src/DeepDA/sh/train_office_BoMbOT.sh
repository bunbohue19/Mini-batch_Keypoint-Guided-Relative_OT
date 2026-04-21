#!/bin/bash
# Office-31  |  Batch-of-Mini-batches OT  (BoMb scheme, no KPG)
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/sh
#   bash train_office31_BoMbOT.sh
#
# Tasks: A→W  A→D  W→A  W→D  D→A  D→W  (all 6 Office-31 transfers)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../office"

# ── Dataset paths ──────────────────────────────────────────────────────────
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE31_IMAGES_ROOT="${DATA_ROOT}/office/images"
LIST_DIR="${DATA_ROOT}/office"

# ── Hyper-parameters ───────────────────────────────────────────────────────
GPU=0
NET=ResNet50
OT_TYPE=balanced
EPSILON=0          # 0 = exact EMD; >0 = Sinkhorn
BE=0               # inter-batch OT regularization (0 = exact EMD)
ETA1=0.1
ETA2=0.1
TAU=1.0
MASS=0.5
K=2                # number of mini-batches in the BoMb hierarchy
M=18               # per-mini-batch size  →  total batch = K * M
BATCH=$(( K * M ))
ITER=20000
TEST_INTERVAL=500
RUN_ID=0

METHOD="BoMbOT"
FINAL_LOG="office31_${METHOD}_run${RUN_ID}_log.txt"

echo "=== Office-31  |  ${METHOD}  |  run ${RUN_ID} ==="

declare -A TASKS
TASKS["A2W"]="amazon_list.txt webcam_list.txt"
TASKS["A2D"]="amazon_list.txt dslr_list.txt"
TASKS["W2A"]="webcam_list.txt amazon_list.txt"
TASKS["W2D"]="webcam_list.txt dslr_list.txt"
TASKS["D2A"]="dslr_list.txt amazon_list.txt"
TASKS["D2W"]="dslr_list.txt webcam_list.txt"

for TASK in A2W A2D W2A W2D D2A D2W; do
    read -r SRC_FILE TGT_FILE <<< "${TASKS[$TASK]}"
    S_PATH="${LIST_DIR}/${SRC_FILE}"
    T_PATH="${LIST_DIR}/${TGT_FILE}"
    OUTPUT_DIR="office31_${TASK}_${METHOD}_k${K}_m${M}_eps${EPSILON}_be${BE}_run${RUN_ID}"

    echo ""
    echo "── ${TASK}  →  ${OUTPUT_DIR}"
    python train.py \
        --gpu_id        ${GPU} \
        --net           ${NET} \
        --dset          office \
        --s_dset_path   "${S_PATH}" \
        --t_dset_path   "${T_PATH}" \
        --stratify_source \
        --batch_size    ${BATCH} \
        --test_interval ${TEST_INTERVAL} \
        --stop_step     ${ITER} \
        --output_dir    "${OUTPUT_DIR}" \
        --final_log     "${FINAL_LOG}" \
        --ot_type       ${OT_TYPE} \
        --eta1          ${ETA1} \
        --eta2          ${ETA2} \
        --epsilon       ${EPSILON} \
        --tau           ${TAU} \
        --mass          ${MASS} \
        --use_bomb \
        --be            ${BE} \
        --k             ${K}
    echo "── Done: ${TASK}"
done

echo ""
echo "=== All Office-31 ${METHOD} tasks finished ==="
