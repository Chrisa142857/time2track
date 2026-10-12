# Delay tokens for sEMG decoders (Fig. 7, Appendix A.8)

This folder reproduces Fig. 7 of the camera-ready paper ("Application to EMG2POSE dataset by adding
delay tokens to the sEMG decoders EMGFormer and vemg2pose on the official test split") and the
experiment described in Appendix A.8, "Delay Tokens for sEMG Decoders". Each decoder is fine-tuned
from its released checkpoint with and without the delay branch, with seeds 42, 43 and 44, on the first
usable gesture cycle of every emg2pose session, and the figure plots the seed-matched change of RMSE,
|PAR−1|, |APCR−1| and |APDR−1| per gesture stage.

| file | role |
|---|---|
| `prepare_data.py` | emg2pose release to per-session `.npy` files, session metadata, official test index |
| `gesture_cycles.py` | gesture-cycle detection, first usable cycle per session, train/val/test window indices |
| `delay_branch.py` | envelope delay branch (FFT cross-correlation, soft-argmax, channel tokens, zero convolution) |
| `vemg2pose_delay.py` | vemg2pose + FiLM injection: validation windows, training, test predictions |
| `emgformer_delay.py` | EMGFormer + cross-attention injection: window pack, training, test predictions |
| `score_stages.py` | paper metrics per gesture stage and over all sessions (uses `reproduce/metrics/paper_metrics.py`) |
| `plot_fig7.py` | Fig. 7 from the two score files |
| `run_all.sh` | the whole pipeline in order |

## Inputs

External code, cloned at the commits used for the paper (not vendored here):

| repository | commit | used for |
|---|---|---|
| https://github.com/facebookresearch/emg2pose | `5f6f62b` | vemg2pose host, loss, metrics (install its `UmeTrack` submodule as in its README) |
| https://github.com/zhenqis123/EgoEMG | `9901726` | EMGFormer host, `config/`, `assets/per_dataset_norm_stats.json` |
| https://github.com/ForeverBlue816/PhysioWave | `6f7bec7` | `TransformerBlock` inside the delay branch |

Checkpoints: `tracking_vemg2pose.ckpt` from the emg2pose release
(`https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz`) and
`emg2pose_emgformer_small.ckpt` from the EgoEMG release (Google Drive link in the EgoEMG README,
placed in `EgoEMG/checkpoints/`).

Data: the full emg2pose dataset (431 GiB,
`https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar`), converted once to one
`{stem}.emg.npy` (T × 16), `{stem}.joints.npy` (T × 20) and `{stem}.meta.json` per session:

```bash
python prepare_data.py convert --emg2pose-dataset /path/to/emg2pose_dataset --data /path/to/emg2pose_npy
```

The paper runs used Python 3.10 with torch 2.11, hydra-core 1.3.2, numpy, scipy 1.15, h5py, einops and
matplotlib 3.10 (PyTorch Lightning is not needed). The figure uses the Arial font.

```bash
export EMG2POSE_DATA=/path/to/emg2pose_npy
export EMG2POSE_REPO=/path/to/emg2pose  VEMG2POSE_CKPT=/path/to/tracking_vemg2pose.ckpt
export EGOEMG_DIR=/path/to/EgoEMG       PHYSIOWAVE_DIR=/path/to/PhysioWave
bash run_all.sh                         # outputs in ./work and ./figures
```

## Steps and settings (`run_all.sh`)

1. `prepare_data.py meta`: session metadata and the test index (6,167 sessions, 178 users, sorted).
2. `gesture_cycles.py`: cycle starts from peaks of the 3 Hz low-pass filtered joint speed of the
   ground truth, then the first usable cycle of every session for the vemg2pose window (3,838 samples,
   24,987 sessions, 6,151 test) and the EMGFormer window (7,790 samples, 24,974 sessions, 6,149 test).
   Training indices per seed: 6,000 × 64 windows (vemg2pose) and 1,560 × 650 windows (EMGFormer), each a
   with-replacement session draw with an electrode roll in {−1, 0, 1}.
3. vemg2pose, arms `ctrl` and `treat`: Adam, learning rate 1e-4, batch 64, 6,000 steps, gradient
   clipping 1.0, chunk 1,024 (K = 2), tau_max 200, beta initialized at 50, final weights evaluated.
   Validation MAE on 2,048 fixed random windows every 1,000 steps (monitoring only).
4. EMGFormer, arms `ctrl` and `xattn`: AdamW, learning rate 1e-4, cosine decay to 5e-5, weight decay
   0.01, batch 650, bf16, gradient clipping 1.0, 60 epochs (1,560 steps), 12 data workers, chunk 2,048
   (K = 3, 48 tokens), tau_max 800, beta initialized at 50, weights with the lowest validation MAE
   evaluated. `run_all.sh` trains the controls on the window draw `CTRL_DRAW` (default 42) and the
   `xattn` arms on the draw of their own seed.
5. `score_stages.py` (CPU): 49 frames over the 2,048 samples after the cycle onset, RMSE and |PAR−1|
   with IK-failure frames masked, |APCR−1| and |APDR−1| on cubic-detrended curves, for all sessions and
   the 29 gesture stages with at least 10 users, with `--unit user` for vemg2pose and `--unit session`
   for EMGFormer.
6. `plot_fig7.py`: delay minus control per seed, mean ± SD (ddof 1) over seeds 42, 43, 44.

Each script prints its options with `--help`. Two settings change the random streams and must stay at
their defaults to repeat the paper runs exactly: the vemg2pose validation cadence (`--val-every 1000`)
and the EMGFormer data workers (`--workers 12`).

## Outputs

`work/vemg2pose_per_stage.json` and `work/emgformer_per_stage.json` hold the metrics per arm, seed and
stage. `figures/emg2pose_delay_minus_base_seeds42-43-44.{pdf,png,json}` is Fig. 7 and its plotted values.

## Runtime

From the paper runs, one NVIDIA H100 NVL (95 GB) per run. vemg2pose: 11 to 19 min per training run
(about 2 GB of GPU memory) and under 1 min per test dump. EMGFormer: 19 to 22 min per training run
(about 19 GB at batch 650) and under 1 min per test dump. The EMGFormer window pack takes 28 GB of
disk. The 12 training runs take about 3.5 h in sequence on one GPU, and each scoring run takes a few
minutes on CPU.
