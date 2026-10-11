#!/usr/bin/env bash
# ============================================================
# Train ECG-conditioned 4-D heart mesh trajectory model
# ============================================================
#
# Usage:
#   bash train_ecg_heart.sh [NUM_GPUS] [CONFIG]
#
# Examples:
#   bash train_ecg_heart.sh                          # 1 GPU, default config
#   bash train_ecg_heart.sh 4                        # 4 GPUs, default config
#   bash train_ecg_heart.sh 1 src/configs/my.yaml   # custom config
#   bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_tech.yaml  # TeCh ECG encoder
#   bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_delay.yaml # ECG + 12x12 delay matrix
#   bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_delay_add.yaml # ECG + delay with 'add' fusion
#   bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_physiowave.yaml
#   bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_delaywave.yaml
#   bash train_ecg_heart.sh 4 src/configs/config_ecg_heart_delaywavev2.yaml
# ============================================================

set -euo pipefail
NUM_GPUS=${1:-1}
CONFIG=${2:-src/configs/config_ecg_heart.yaml}
# GPU ids come from the caller (CUDA_VISIBLE_DEVICES=0,2,3,6 bash train_ecg_heart.sh 4 ...);
# without one, the first NUM_GPUS devices.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}

# One accelerate config per GPU count: src/configs/acc/<N>gpu.yaml
ACC_CONFIG=src/configs/acc/${NUM_GPUS}gpu.yaml
[ -f "${ACC_CONFIG}" ] || { echo "no accelerate config for ${NUM_GPUS} GPUs: ${ACC_CONFIG}" >&2; exit 1; }

echo "=================================================="
echo " ECG Heart Mesh Trajectory Training"
echo "  GPUs         : ${NUM_GPUS}"
echo "  Config       : ${CONFIG}"
echo "  Accelerate   : ${ACC_CONFIG}"
echo "=================================================="

# Change to the src directory so relative imports work
cd "$(dirname "$0")/src"

accelerate launch \
    --config_file "../${ACC_CONFIG}" \
    train_ecg_mesh.py \
    --config "../${CONFIG}" 
