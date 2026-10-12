#!/usr/bin/env bash
# ============================================================
# Visualise predicted 4-D heart mesh trajectories
# ============================================================
#
# Usage:
#   bash vis_ecg_heart.sh [CHECKPOINT] [CONFIG] [N_SAMPLES] [OUTPUT_DIR]
#
# Examples:
#   bash vis_ecg_heart.sh                                   # 10 samples, best_model, default output
#   bash vis_ecg_heart.sh outputs/.../checkpoint-5000       # specific checkpoint
#   bash vis_ecg_heart.sh outputs/.../best_model \
#       src/configs/config_ecg_heart.yaml 20 outputs/vis    # 20 samples → outputs/vis
#   bash vis_ecg_heart.sh "" src/configs/config_ecg_heart_tech.yaml    # TeCh ECG encoder
#   bash vis_ecg_heart.sh "" src/configs/config_ecg_heart_delay.yaml   # ECG + 12x12 delay matrix
#
# Add --save_html to the python call below to also generate interactive
# Three.js HTML viewers alongside each GIF.
# ============================================================

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # caller may choose the GPU

ROOT="$(cd "$(dirname "$0")" && pwd)"
# Paths given on the command line are relative to the caller's directory. Config fields
# (output_dir, dataset paths, pretrained_path) are relative to src/, as in training.
CONFIG=$(realpath -m "${2:-src/configs/config_ecg_heart.yaml}")

# Resolve default checkpoint
if [ -z "${1:-}" ]; then
    OUTPUT_DIR=$(cd "${ROOT}/src" && python3 -c "
import os
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
print(os.path.abspath(cfg.output_dir))
") || { echo "cannot read output_dir from ${CONFIG}" >&2; exit 1; }
    CHECKPOINT="${OUTPUT_DIR}/best_model"
else
    CHECKPOINT=$(realpath -m "$1")
fi

N_SAMPLES=${3:-10}

# Default vis output dir: <checkpoint_parent>/vis_eval
if [ -z "${4:-}" ]; then
    VIS_OUTPUT_DIR="$(dirname "${CHECKPOINT}")/vis_eval"
else
    VIS_OUTPUT_DIR=$(realpath -m "$4")
fi

echo "=================================================="
echo " ECG Heart Mesh – Visualisation"
echo "  Config          : ${CONFIG}"
echo "  Checkpoint      : ${CHECKPOINT}"
echo "  Samples         : ${N_SAMPLES}"
echo "  Output dir      : ${VIS_OUTPUT_DIR}"
echo "=================================================="

mkdir -p "${VIS_OUTPUT_DIR}"

# Change to src/ so relative imports resolve correctly
cd "${ROOT}/src"

python vis_ecg_mesh.py \
    --config              "${CONFIG}"           \
    --checkpoint          "${CHECKPOINT}"       \
    --output_dir          "${VIS_OUTPUT_DIR}"   \
    --n_samples           "${N_SAMPLES}"           \
    --split               val 
    # --save_html
    
