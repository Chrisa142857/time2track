#!/usr/bin/env bash
# ============================================================
# Evaluate a trained ECG-conditioned 4-D heart mesh model
# ============================================================
#
# Usage:
#   bash eval_ecg_heart.sh [CHECKPOINT] [CONFIG] [NUM_VAL_BATCHES]
#
# Examples:
#   bash eval_ecg_heart.sh                              # best_model, default config, full val set
#   bash eval_ecg_heart.sh outputs/.../checkpoint-5000  # specific checkpoint
#   bash eval_ecg_heart.sh outputs/.../best_model \
#       src/configs/config_ecg_heart.yaml 10            # 10 batches (quick sanity check)
#   bash eval_ecg_heart.sh "" src/configs/config_ecg_heart_tech.yaml   # TeCh ECG encoder
# ============================================================

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # caller may choose the GPU
CONFIG=${2:-src/configs/config_ecg_heart.yaml}

# Resolve default checkpoint: prefer best_model inside the output_dir encoded in the config
if [ -z "${1:-}" ]; then
    OUTPUT_DIR=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
print(cfg.output_dir)
") || { echo "cannot read output_dir from ${CONFIG}" >&2; exit 1; }
    CHECKPOINT="${OUTPUT_DIR}/best_model"
else
    CHECKPOINT="$1"
fi

NP_DIR="$(dirname "${CHECKPOINT}")/npy_eval"
NUM_VAL_BATCHES=${3:--1}   # -1 = full val set

# Metrics are appended to this CSV (override with RESULTS_CSV=...)
RESULTS_CSV=${RESULTS_CSV:-$(dirname "${CHECKPOINT}")/eval_metrics.csv}

echo "=================================================="
echo " ECG Heart Mesh – Evaluation"
echo "  Config          : ${CONFIG}"
echo "  Checkpoint      : ${CHECKPOINT}"
echo "  Val batches     : ${NUM_VAL_BATCHES} (-1 = all)"
echo "  Per-sample save : ${NP_DIR}"
echo "  Results CSV     : ${RESULTS_CSV}"
echo "=================================================="

mkdir -p results
mkdir -p "${NP_DIR}"

# Change to src/ so relative imports resolve correctly
# cd "$(dirname "$0")/src"


python src/eval_ecg_mesh.py \
    --config             "${CONFIG}"        \
    --checkpoint         "${CHECKPOINT}"    \
    --num_val_batches    "${NUM_VAL_BATCHES}"   \
    --output_csv         "${RESULTS_CSV}"      \
    --npy_dir            "${NP_DIR}"
