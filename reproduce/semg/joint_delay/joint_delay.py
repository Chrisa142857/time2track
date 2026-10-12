"""Per-session 20 x 20 inter-joint delay matrices of ground-truth and predicted joint angles (Fig. 1d).

For every preds_and_gts_<i>.npz written by predict_joint_angles.py, the first --n-frames samples of
`gt` and `pred` go through get_kinematic_delay_matrix: joint angular velocity, 4th-order Butterworth
low-pass at 5 Hz (zero phase), cross-correlation of every joint pair within +-max_delay_ms, delay at
the peak |correlation|, NaN for flat joints, peaks on the window edge or normalized peaks below
min_corr. Entry [i, j] is the delay of joint i relative to joint j in ms, [j, i] its negative.

The defaults are the settings of the paper's figure. The npz files hold 2 kHz samples, so 240 frames
cover 0.12 s, and --fs is the rate used to convert lags to milliseconds.
"""
import argparse
import glob
import os

import numpy as np
from scipy import signal


def get_kinematic_delay_matrix(pose_data, fs=32, max_delay_ms=400, min_corr_threshold=0.15):
    """pose_data: (num_joints, num_samples) joint angles -> (num_joints, num_joints) delays in ms."""
    num_joints, num_samples = pose_data.shape
    delay_matrix = np.zeros((num_joints, num_joints))

    # 1. Angular Velocity + Low Pass Filter
    velocities = np.gradient(pose_data, axis=1)
    b, a = signal.butter(4, 5, btype='low', fs=fs)
    smoothed_velocities = signal.filtfilt(b, a, velocities, axis=1)

    lags = signal.correlation_lags(num_samples, num_samples)
    max_lag_samples = int((max_delay_ms / 1000.0) * fs)

    for i in range(num_joints):
        for j in range(i + 1, num_joints):
            energy_i = np.sum(smoothed_velocities[i, :] ** 2)
            energy_j = np.sum(smoothed_velocities[j, :] ** 2)
            if energy_i < 1e-5 or energy_j < 1e-5:  # avoid pure zero comparisons
                delay_matrix[i, j] = np.nan
                delay_matrix[j, i] = np.nan
                continue

            corr = signal.correlate(smoothed_velocities[i, :], smoothed_velocities[j, :], mode='full')
            center_idx = len(corr) // 2
            start_idx = center_idx - max_lag_samples
            end_idx = center_idx + max_lag_samples + 1
            valid_corr = corr[start_idx:end_idx]

            optimal_lag_idx = np.argmax(np.abs(valid_corr))  # absolute argmax
            if optimal_lag_idx == 0 or optimal_lag_idx == len(valid_corr) - 1:
                delay_matrix[i, j] = np.nan
                delay_matrix[j, i] = np.nan
                continue

            peak_value = valid_corr[optimal_lag_idx]
            normalized_peak = np.abs(peak_value) / np.sqrt(energy_i * energy_j)
            if normalized_peak < min_corr_threshold:
                delay_matrix[i, j] = np.nan
                delay_matrix[j, i] = np.nan
                continue

            delay_in_samples = lags[start_idx + optimal_lag_idx]
            delay_in_ms = (delay_in_samples / fs) * 1000.0
            delay_matrix[i, j] = int(np.round(delay_in_ms))
            delay_matrix[j, i] = -int(np.round(delay_in_ms))

    return delay_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-dir', required=True, help='directory of preds_and_gts_<i>.npz (predict_joint_angles.py)')
    ap.add_argument('--out', default='joint_delay_matrices.npz', help='output npz')
    ap.add_argument('--n-frames', type=int, default=240, help='leading samples of each window that are used')
    ap.add_argument('--fs', type=float, default=32, help='sampling rate assumed by the delay function (Hz)')
    ap.add_argument('--max-delay-ms', type=float, default=400)
    ap.add_argument('--min-corr', type=float, default=0.15)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, '*.npz')))
    if not files:
        raise SystemExit(f'no npz files in {args.pred_dir}')
    gt, pred, stage, filename = [], [], [], []
    for f in files:
        d = np.load(f)
        gt.append(get_kinematic_delay_matrix(d['gt'][:args.n_frames].T, args.fs, args.max_delay_ms, args.min_corr))
        pred.append(get_kinematic_delay_matrix(d['pred'][:args.n_frames].T, args.fs, args.max_delay_ms, args.min_corr))
        stage.append(str(d['stage']))
        filename.append(str(d['filename']))
    np.savez(args.out, gt=np.stack(gt), pred=np.stack(pred), stage=np.array(stage), filename=np.array(filename),
             n_frames=args.n_frames, fs=args.fs, max_delay_ms=args.max_delay_ms, min_corr=args.min_corr)
    print(f'{len(files)} sessions -> {args.out}')


if __name__ == '__main__':
    main()
