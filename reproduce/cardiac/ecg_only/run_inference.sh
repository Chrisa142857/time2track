#!/usr/bin/env bash
# Inference of Time2Track on a subject list, optionally with a replaced rest mesh.
#
# Usage: run_inference.sh <subject_list> <npy_dir> [extra args of src/eval_ecg_mesh_e5.py]
#   population rest mesh:  run_inference.sh list.txt <root>/cohort_mean/npy_eval --init_pc_override cohort_mean_template.npy
#   own rest mesh:         run_inference.sh list.txt <root>/subject_specific/npy_eval
#
# The list is split round-robin into NCHUNK (default 4) lists and each one is a separate call at
# batch size 16. DDIM noise is drawn per batch from manual_seed(0), so this partition and batch size
# fix the noise of every subject. Chunks run in parallel, one per GPU in GPUS (default "0").
# Environment: GPUS, NCHUNK, PYTHON (default python), CONFIG, CKPT (defaults below).
set -euo pipefail
if [[ $# -lt 2 || $1 == -h || $1 == --help ]]; then sed -n '2,11p' "$0" | cut -c3-; exit 0; fi
LIST=$(realpath "$1"); NPY=$(realpath -m "$2"); shift 2
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
CONFIG=$(realpath "${CONFIG:-$REPO/src/configs/config_ecg_heart_delaywavev3.yaml}")
CKPT=$(realpath "${CKPT:-$REPO/nips26-physctrl-outputs/ecg_heart_delaywavev3_8L_256d_49frames/best_model}")
PY=${PYTHON:-python}
read -ra GPU_IDS <<< "${GPUS:-0}"
CHUNKS=$(mktemp -d)
trap 'rm -rf "$CHUNKS"' EXIT
split -n "r/${NCHUNK:-4}" -d "$LIST" "$CHUNKS/chunk_"

cd "$REPO/src"   # config paths are relative to src/
pids=()
i=0
for c in "$CHUNKS"/chunk_*; do
  CUDA_VISIBLE_DEVICES=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]} "$PY" eval_ecg_mesh_e5.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --num_inference_steps 50 --guidance_scale 2.0 --eval_batch_size 16 \
    --subject_list "$c" --npy_dir "$NPY" "$@" &
  pids+=($!)
  i=$((i + 1))
  if (( ${#pids[@]} == ${#GPU_IDS[@]} )); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
for p in "${pids[@]}"; do wait "$p"; done
echo "predictions: $(ls "$NPY" | grep -c '_pred\.npy$') in $NPY"
