#!/bin/bash
# Office-Home  |  KPG-RL + mini-batch OT  (averaging scheme WITH KPG guidance)
#
# Usage:
#   cd Mini-batch_KPG-RL_OT/src/DeepDA/sh
#   bash train_home_KPG_mOT.sh
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
OT_TYPE=partial
EPSILON=0          # 0 = exact EMD; >0 = masked Sinkhorn (balanced)
ETA1=0.01
ETA2=0.5
TAU=0.5
MASS=0.5
K=1
M=65
BATCH=$(( K * M ))
ITER=10000
TEST_INTERVAL=500
RUN_ID=0

# KPG-RL-KP parameters  (see keypoint_guided_OT.py kpg_rl_kp)
ALPHA=0.7           # combination coeff: alpha * C_norm + (1 - alpha) * G_norm
TAU_S=0.1           # softmax temperature for source relation profiles
TAU_T=0.1           # softmax temperature for target relation profiles

METHOD="KPG_mOT"
FINAL_LOG="home_${METHOD}_run${RUN_ID}_log.txt"

echo "=== Office-Home  |  ${METHOD}  |  run ${RUN_ID} ==="
echo "    KPG params: alpha=${ALPHA}  tau_s=${TAU_S}  tau_t=${TAU_T}"

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
    OUTPUT_DIR="home_${TASK}_${METHOD}_k${K}_m${M}_eps${EPSILON}_alpha${ALPHA}_run${RUN_ID}"

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
        --k             ${K} \
        --use_kpg \
        --alpha         ${ALPHA} \
        --tau_s         ${TAU_S} \
        --tau_t         ${TAU_T}
    echo "── Done: ${TASK}"
done

echo ""
echo "=== All Office-Home ${METHOD} tasks finished ==="
