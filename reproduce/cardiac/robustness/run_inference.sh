#!/usr/bin/env bash
# Perturbed inference for Tables 7 and 8 (Appendix A.6), one fresh prediction dir per condition.
#
# Usage (from anywhere):
#   bash reproduce/cardiac/robustness/run_inference.sh <lists dir> <out dir> [condition ...]
#
#   <lists dir>  holds any_cardiac.txt and conduction_block.txt (reproduce/cardiac/pathology)
#   <out dir>    receives <out dir>/<condition>/npy_eval/{subid}_pred.npy
#   condition    any of the names below, default all. Pass a subset to split the sweep over GPUs.
#
# Environment: CUDA_VISIBLE_DEVICES (GPU), PYTHON (interpreter, default python),
#   CONFIG (default src/configs/config_ecg_heart_delaywavev3.yaml),
#   CKPT (default <output_dir of CONFIG>/best_model, as in eval_ecg_heart.sh).
set -euo pipefail
[ $# -ge 2 ] || { sed -n 4,13p "$0"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
LISTS=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
PY=${PYTHON:-python}
CONFIG=$(realpath -m "${CONFIG:-${ROOT}/src/configs/config_ecg_heart_delaywavev3.yaml}")
if [ -z "${CKPT:-}" ]; then
  CKPT=$(cd "${ROOT}/src" && "$PY" -c "
import os
from omegaconf import OmegaConf
print(os.path.abspath(OmegaConf.load('${CONFIG}').output_dir))")/best_model
fi
ALL="ac_clean ac_noise20 ac_noise10 ac_noise5 ac_drop_II ac_drop_V2 ac_drop_chest3 cb_clean cb_tau25 cb_tau75 cb_tau100"

infer() { local name=$1 list=$2; shift 2
  echo "=== $(date +%H:%M:%S) ${name} ==="
  "$PY" "${HERE}/perturbed_inference.py" --config "${CONFIG}" --checkpoint "${CKPT}" \
      --subject-list "${LISTS}/${list}" --npy-dir "${OUT}/${name}/npy_eval" "$@"
}

# any_cardiac = diseased subgroup of Table 7, conduction_block = Table 8.
for name in ${@:-$ALL}; do case $name in
  ac_clean)       infer ac_clean       any_cardiac.txt      --mode clean ;;
  ac_noise20)     infer ac_noise20     any_cardiac.txt      --mode noise   --snr 20 ;;
  ac_noise10)     infer ac_noise10     any_cardiac.txt      --mode noise   --snr 10 ;;
  ac_noise5)      infer ac_noise5      any_cardiac.txt      --mode noise   --snr 5  ;;
  ac_drop_II)     infer ac_drop_II     any_cardiac.txt      --mode dropout --drop-leads 1 ;;
  ac_drop_V2)     infer ac_drop_V2     any_cardiac.txt      --mode dropout --drop-leads 7 ;;
  ac_drop_chest3) infer ac_drop_chest3 any_cardiac.txt      --mode dropout --drop-leads 6,7,8 ;;
  cb_clean)       infer cb_clean       conduction_block.txt --mode clean ;;
  cb_tau25)       infer cb_tau25       conduction_block.txt --mode taumax  --tau-max 25 ;;
  cb_tau75)       infer cb_tau75       conduction_block.txt --mode taumax  --tau-max 75 ;;
  cb_tau100)      infer cb_tau100      conduction_block.txt --mode taumax  --tau-max 100 ;;
  *) echo "unknown condition ${name}" >&2; exit 1 ;;
esac; done
