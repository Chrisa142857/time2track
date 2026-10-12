#!/usr/bin/env bash
# Fig. 7 end to end: session metadata -> gesture-cycle anchors -> vemg2pose and EMGFormer arms
# (control and + delay tokens, seeds 42 43 44) -> per-stage paper metrics -> figure.
#
# Required environment (see README.md):
#   EMG2POSE_DATA   per-session .npy directory (prepare_data.py convert)
#   EMG2POSE_REPO   facebookresearch/emg2pose @5f6f62b      VEMG2POSE_CKPT  tracking_vemg2pose.ckpt
#   EGOEMG_DIR      zhenqis123/EgoEMG @9901726 (+ checkpoints/emg2pose_emgformer_small.ckpt)
#   PHYSIOWAVE_DIR  ForeverBlue816/PhysioWave @6f7bec7
# Optional: PY (python, default python), WORK (outputs, default ./work), FIG (default ./figures),
#   SEEDS (default "42 43 44"), CTRL_DRAW (see step 4).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python}
WORK=${WORK:-work}
FIG=${FIG:-figures}
SEEDS=${SEEDS:-"42 43 44"}
: "${EMG2POSE_DATA:?}" "${EMG2POSE_REPO:?}" "${VEMG2POSE_CKPT:?}" "${EGOEMG_DIR:?}" "${PHYSIOWAVE_DIR:?}"
mkdir -p "$WORK"

# 1. session metadata and the official test index
$PY "$HERE/prepare_data.py" meta --data "$EMG2POSE_DATA" \
  --session-meta "$WORK/session_meta.json" --test-index "$WORK/test_index.json"

# 2. gesture cycles, first usable cycle per session, window indices
#    vemg2pose: 3,838-sample windows, 6,000 steps x 64. EMGFormer: 7,790-sample windows, 1,560 steps x 650.
$PY "$HERE/gesture_cycles.py" detect --data "$EMG2POSE_DATA" --cycles "$WORK/cycle_index.json"
for s in $SEEDS; do
  $PY "$HERE/gesture_cycles.py" anchors --cycles "$WORK/cycle_index.json" --session-meta "$WORK/session_meta.json" \
    --window 3838 --steps 6000 --batch 64 --seed "$s" \
    --anchors "$WORK/first_cycle_anchors.json" --train-index "$WORK/vemg2pose_train_s$s.json"
  $PY "$HERE/gesture_cycles.py" anchors --cycles "$WORK/cycle_index.json" --session-meta "$WORK/session_meta.json" \
    --window 7790 --steps 1560 --batch 650 --seed "$s" \
    --anchors "$WORK/first_cycle_anchors_w7790.json" --train-index "$WORK/emgformer_train_s$s.json"
done
$PY "$HERE/gesture_cycles.py" test_index --anchors "$WORK/first_cycle_anchors.json" \
  --test-index "$WORK/test_index.json" --out "$WORK/test_index_w3838.json"          # 6,151 sessions
$PY "$HERE/gesture_cycles.py" test_index --anchors "$WORK/first_cycle_anchors_w7790.json" \
  --test-index "$WORK/test_index.json" --out "$WORK/test_index_w7790.json"          # 6,149 sessions
for sp in val test; do
  $PY "$HERE/gesture_cycles.py" windows --anchors "$WORK/first_cycle_anchors_w7790.json" --window 7790 \
    --split "$sp" --out "$WORK/emgformer_${sp}_windows.json"
done

# 3. vemg2pose: ctrl vs treat (FiLM delay tokens), final weights after 6,000 steps
$PY "$HERE/vemg2pose_delay.py" val_index --data "$EMG2POSE_DATA" --session-meta "$WORK/session_meta.json" \
  --out "$WORK/vemg2pose_val_windows.json"
for s in $SEEDS; do
  for arm in ctrl treat; do
    $PY "$HERE/vemg2pose_delay.py" train --data "$EMG2POSE_DATA" --arm "$arm" --seed "$s" \
      --index "$WORK/vemg2pose_train_s$s.json" --val-index "$WORK/vemg2pose_val_windows.json" \
      --out "$WORK/vemg2pose/ckpt/${arm}_s$s.pt"
    $PY "$HERE/vemg2pose_delay.py" dump --data "$EMG2POSE_DATA" --arm "$arm" --sd "$WORK/vemg2pose/ckpt/${arm}_s$s.pt" \
      --test-index "$WORK/test_index.json" --anchors "$WORK/first_cycle_anchors.json" \
      --outdir "$WORK/vemg2pose/npy/${arm}_s$s"
  done
done

# 4. EMGFormer: ctrl vs xattn (cross-attention delay tokens), weights with the lowest validation MAE.
#    Controls train on the window draw CTRL_DRAW (default 42, or "seed" for the draw of their own seed),
#    the xattn arms on the draw of their own seed.
CTRL_DRAW=${CTRL_DRAW:-42}
$PY "$HERE/emgformer_delay.py" pack --data "$EMG2POSE_DATA" --anchors "$WORK/first_cycle_anchors_w7790.json" \
  --pack "$WORK/emgformer_pack"
for s in $SEEDS; do
  for arm in ctrl xattn; do
    d=$s
    [ "$arm" = ctrl ] && [ "$CTRL_DRAW" != seed ] && d=$CTRL_DRAW
    $PY "$HERE/emgformer_delay.py" train --data "$EMG2POSE_DATA" --pack "$WORK/emgformer_pack" --arm "$arm" \
      --seed "$s" --index "$WORK/emgformer_train_s$d.json" --val-index "$WORK/emgformer_val_windows.json" \
      --out "$WORK/emgformer/ckpt/${arm}_s$s.pt"
    $PY "$HERE/emgformer_delay.py" dump --data "$EMG2POSE_DATA" --pack "$WORK/emgformer_pack" --arm "$arm" \
      --seed "$s" --sd "$WORK/emgformer/ckpt/${arm}_s$s.pt" --index "$WORK/emgformer_test_windows.json" \
      --outdir "$WORK/emgformer/npy/${arm}_s$s"
  done
done

# 5. per-stage paper metrics, on CPU
CUDA_VISIBLE_DEVICES="" $PY "$HERE/score_stages.py" --npy-root "$WORK/vemg2pose/npy" --arms ctrl treat \
  --seeds $SEEDS --index "$WORK/test_index_w3838.json" --unit user --out "$WORK/vemg2pose_per_stage.json"
CUDA_VISIBLE_DEVICES="" $PY "$HERE/score_stages.py" --npy-root "$WORK/emgformer/npy" --arms ctrl xattn \
  --seeds $SEEDS --index "$WORK/test_index_w7790.json" --unit session --out "$WORK/emgformer_per_stage.json"

# 6. Fig. 7
$PY "$HERE/plot_fig7.py" --emgformer-json "$WORK/emgformer_per_stage.json" \
  --vemg2pose-json "$WORK/vemg2pose_per_stage.json" --seeds $SEEDS --out "$FIG"
