#!/bin/bash
# Office-Home  |  Batch-of-Mini-batches OT  (BoMb scheme, no KPG)
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/sh
#   bash train_home_BoMbOT.sh
#
# Tasks: all 12 Office-Home transfers

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../office"

# ── Dataset paths ──────────────────────────────────────────────────────────
DATA_ROOT="$(cd "${SCRIPT_DIR}/../../../data" && pwd)"
export OFFICE_HOME_IMAGES_ROOT="${DATA_ROOT}/office-home/images"
LIST_DIR="${DATA_ROOT}/office-home"

# ── Hyper-parameters ───────────────────────────────────────────────────────
GPU=0
NET=ResNet50
OT_TYPE=balanced
EPSILON=0
BE=0               # inter-batch OT regularization (0 = exact EMD)
ETA1=0.01
ETA2=0.5
TAU=0.5
MASS=0.5
K=2                # number of mini-batches in the BoMb hierarchy
M=65               # per-mini-batch size  →  total batch = K * M
BATCH=$(( K * M ))
ITER=10000
TEST_INTERVAL=500
RUN_ID=0

METHOD="BoMbOT"
FINAL_LOG="home_${METHOD}_run${RUN_ID}_log.txt"

echo "=== Office-Home  |  ${METHOD}  |  run ${RUN_ID} ==="
echo "    BoMb: k=${K}  m=${M}  batch=${BATCH}  be=${BE}"

TASK_LIST=(
    "Art.txt       Clipart.txt    A2C"
    "Art.txt       Product.txt    A2P"
    "Art.txt       Real_World.txt A2R"
    "Clipart.txt   Art.txt        C2A"
    "Clipart.txt   Product.txt    C2P"
    "Clipart.txt   Real_World.txt C2R"
    "Product.txt   Art.txt        P2A"
    "Product.txt   Clipart.txt    P2C"
    "Product.txt   Real_World.txt P2R"
    "Real_World.txt Art.txt       R2A"
    "Real_World.txt Clipart.txt   R2C"
    "Real_World.txt Product.txt   R2P"
)

for ENTRY in "${TASK_LIST[@]}"; do
    read -r SRC_FILE TGT_FILE TASK <<< "${ENTRY}"
    S_PATH="${LIST_DIR}/${SRC_FILE}"
    T_PATH="${LIST_DIR}/${TGT_FILE}"
    OUTPUT_DIR="home_${TASK}_${METHOD}_k${K}_m${M}_eps${EPSILON}_be${BE}_run${RUN_ID}"

    echo ""
    echo "── ${TASK}  →  ${OUTPUT_DIR}"
    python train.py \
        --gpu_id        ${GPU} \
        --net           ${NET} \
        --dset          office-home \
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
echo "=== All Office-Home ${METHOD} tasks finished ==="
