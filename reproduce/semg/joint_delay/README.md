# Inter-joint delay of hand pose (Fig. 1d)

This folder reproduces panel (d) of Fig. 1 of the paper:

> **(d)** Hand pose delay is diverse by gesture, where the delay of ground truth (GT) hand joint angle cannot be replicated by vemg2pose predictions (Pred).

Each panel is a 20 x 20 matrix over the emg2pose joint angles. The upper triangle is the ground truth and the lower triangle the prediction, and each cell is the mean |inter-joint delay| over the sessions of one gesture stage. The three panels are the emg2pose stages `CountingUpDownFingerWigglingSpreading` (Count, 8 sessions), `HandDeskSeparateClaspedChest` (Chest Stretch, 14 sessions) and `FingerPinchesSingleFingerPinchesMultiple` (Pinch, 10 sessions).

## Inputs

1. The emg2pose code base, which is not vendored here: `git clone https://github.com/facebookresearch/emg2pose` and `git checkout 5f6f62b`. Set up its conda environment (`environment.yml`) and install it with `pip install -e .` as its README describes. Pass the clone with `--emg2pose-repo` or the environment variable `EMG2POSE_REPO`.
2. The emg2pose dataset (full version, 431 GiB, `https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar`), unpacked into one directory of `*.hdf5` session files. The analysis uses the first 444 files in sorted filename order (13 recording sessions). The 600 MiB mini version of the dataset is a different subset and does not reproduce the figure.
3. The released checkpoints (`https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz`), of which `regression_vemg2pose.ckpt` is used.

## Commands

Run from this folder. Step 1 needs the emg2pose environment. Steps 2 and 3 need numpy, scipy, matplotlib and seaborn (the figure was drawn with seaborn 0.13.2 and matplotlib 3.10.8).

```bash
# 1. GT and predicted joint angles: one preds_and_gts_<i>.npz per session (442 files)
python predict_joint_angles.py \
    --emg2pose-repo /path/to/emg2pose \
    --data-dir /path/to/emg2pose_dataset \
    --checkpoint /path/to/emg2pose_model_checkpoints/regression_vemg2pose.ckpt \
    --out-dir predictions

# 2. per-session inter-joint delay matrices of GT and prediction
python joint_delay.py --pred-dir predictions --out joint_delay_matrices.npz

# 3. Fig. 1d
python plot_fig1d.py --matrices joint_delay_matrices.npz --out fig1d_hand_pose_delay.pdf
```

`plot_fig1d.py --stages all` draws every stage instead of the three of the figure.

## Settings (all are the defaults)

- Inference (`predict_joint_angles.py`): samples 0 to 10000 of each session at the native 2 kHz, hydra override `experiment=tracking_vemg2pose` with `checkpoint=regression_vemg2pose.ckpt` as in the emg2pose notebook. The module is rebuilt from the hyperparameters stored in the checkpoint. The network consumes a left context of 1790 samples, so each output covers 8210 samples (4.1 s). Sessions shorter than the left context are skipped (2 of 444).
- Delay (`joint_delay.py`): the first 240 output samples of GT and prediction (0.12 s of the 2 kHz output), `fs=32` as the rate that converts lags to milliseconds, a lag window of ±400 ms and a minimum normalized peak correlation of 0.15. Per joint pair: angular velocity by `np.gradient`, 4th-order Butterworth low-pass at 5 Hz applied with `filtfilt`, cross-correlation, delay at the peak |correlation|. A pair is NaN when either joint has velocity energy below 1e-5, when the peak lies on the edge of the lag window, or when the normalized peak is below the threshold.
- Display (`plot_fig1d.py`): per stage, |delay| with NaN set to −1, cast to int and averaged over sessions. Each panel uses the coolwarm map from the minimum to the maximum of its two stage means, centred on their midpoint.

## Outputs

- `predictions/preds_and_gts_<i>.npz`: `gt` and `pred` (8210 x 20 joint angles, radians) and the session metadata, `<i>` being the index in the sorted file list.
- `joint_delay_matrices.npz`: `gt` and `pred` (442 x 20 x 20 delays in ms), `stage`, `filename` and the delay settings.
- `fig1d_hand_pose_delay.pdf`: the three panels. The published figure was assembled from these panels in a vector editor, which added the GT / Pred labels and the axis annotations.

## Runtime

Step 1 takes about 7 min on one H100 GPU (5.5 min of inference after about 1.5 min of opening the 444 hdf5 files). Steps 2 and 3 take under a minute on CPU.

## Reproducibility

Step 2 run on the prediction files behind the published figure reproduces its delay matrices exactly. Predictions regenerated with step 1 differ from those files by up to about 5e-4 rad (floating-point differences of the hardware and libraries). The ground-truth matrices are unchanged, whereas 106 of the 83,980 predicted cells (0.13 %) change because the thresholds and the argmax of the delay function are discontinuous. In the figure this alters one cell of the Count prediction triangle, and the colour ranges of all three panels stay the same.
