#!/usr/bin/env bash
# Download the trained weights of Time2Track and of every baseline and ablation (one 6.4 GB archive,
# folder https://drive.google.com/drive/folders/16FqYA1gbopIIr9aIk0Uh8c9OWloG173Y) and extract them to
# ${OUT}/<variant>/{best_model,config.yaml,logs}, which is where the output_dir of each config in
# src/configs points. eval_ecg_heart.sh and vis_ecg_heart.sh then find best_model without arguments.
#
# Usage: bash download_ckpts.sh            (needs uv, or gdown on PATH)
set -euo pipefail
cd "$(dirname "$0")"

FILE_ID=${FILE_ID:-1s26T8W4wqP_FkHy4CndZyO77EQG8TtHT}   # model_weight_log.tar.gz in the folder above
OUT=${OUT:-nips26-physctrl-outputs}                     # output_dir prefix of the configs, relative to the repo root
ARCHIVE="${OUT}/model_weight_log.tar.gz"

mkdir -p "${OUT}"
if command -v gdown > /dev/null; then GDOWN=(gdown); else GDOWN=(uvx gdown); fi
[ -f "${ARCHIVE}" ] || "${GDOWN[@]}" "${FILE_ID}" -O "${ARCHIVE}"
tar xzf "${ARCHIVE}" -C "${OUT}"
echo "extracted $(ls -d "${OUT}"/*/best_model | wc -l) checkpoints under ${OUT}/"
