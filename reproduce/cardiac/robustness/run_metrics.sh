#!/usr/bin/env bash
# Metrics and bootstrap for Tables 7 and 8 (Appendix A.6), after run_inference.sh.
#
# Usage: bash reproduce/cardiac/robustness/run_metrics.sh <lists dir> <out dir>
#   writes <out dir>/table7_noise_dropout.json and <out dir>/table8_taumax.json
#
# Environment: CUDA_VISIBLE_DEVICES, PYTHON, CONFIG as in run_inference.sh, and
#   GT_DIR (default <output_dir of CONFIG>/npy_eval, written by eval_ecg_heart.sh).
# The bootstrap generator is consumed in condition order, so keep the order below.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 4,8p "$0"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
LISTS=$(realpath "$1"); OUT=$(realpath "$2")
PY=${PYTHON:-python}
CONFIG=$(realpath -m "${CONFIG:-${ROOT}/src/configs/config_ecg_heart_delaywavev3.yaml}")
if [ -z "${GT_DIR:-}" ]; then
  GT_DIR=$(cd "${ROOT}/src" && "$PY" -c "
import os
from omegaconf import OmegaConf
print(os.path.abspath(OmegaConf.load('${CONFIG}').output_dir))")/npy_eval
fi
NE() { echo "${OUT}/$1/npy_eval"; }

# Table 7: diseased subgroup (any_cardiac)
"$PY" "${HERE}/robustness_metrics.py" --subject-list "${LISTS}/any_cardiac.txt" --gt-dir "${GT_DIR}" \
  --out "${OUT}/table7_noise_dropout.json" --conditions \
  clean:$(NE ac_clean) noise20db:$(NE ac_noise20) noise10db:$(NE ac_noise10) noise5db:$(NE ac_noise5) \
  drop_II:$(NE ac_drop_II) drop_V2:$(NE ac_drop_V2) drop_chest3:$(NE ac_drop_chest3)

# Table 8: tau_max sweep on conduction_block (tau 50 = cb_clean, the trained default)
"$PY" "${HERE}/robustness_metrics.py" --subject-list "${LISTS}/conduction_block.txt" --gt-dir "${GT_DIR}" \
  --out "${OUT}/table8_taumax.json" --conditions \
  clean:$(NE cb_clean) tau25:$(NE cb_tau25) tau75:$(NE cb_tau75) tau100:$(NE cb_tau100)
