"""Convert the per-subject PLY frames and MRI-aligned ECG CSVs into the .npy
files read by src/dataset/ukbiobank_dataset.py.

Mesh: <mesh_dir>/<subject folder>/frame_*.ply (output of mesh_to_ply.py)
  -> <npy_dir>/<subject folder>/frames.npy   (T, 388, 3) float32, first --seq_len frames
  (dataset.npy_dir). Runs when --mesh_dir exists.

ECG: <ecg_dir>/<eid>_<instance>_<k>.csv (output of align_ecg_to_mri.py)
  -> <ecg_npy_dir>/<eid>.npy                 (12, 600) float32, z-score per lead
  (dataset.ecg_dir). Runs when --ecg_dir is given. All CSVs of one participant
  map to the same <eid>.npy, and the first one written is kept.

Existing outputs are never overwritten.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA = Path(__file__).resolve().parents[1] / 'data' / 'ukb_heartmesh_processed'


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _load_ply_verts(path):
    with open(path) as f:
        lines = f.readlines()
    n_v = 0
    i = 0
    while lines[i].strip() != 'end_header':
        tok = lines[i].split()
        if len(tok) >= 3 and tok[0] == 'element' and tok[1] == 'vertex':
            n_v = int(tok[2])
        i += 1
    i += 1
    return np.array([lines[i + k].split()[:3] for k in range(n_v)], dtype=np.float32)


def _process_subject(subject_dir, npy_root, seq_len=50):
    """
    Convert all frame_*.ply in subject_dir to frames.npy.
    Returns (True, path) on success, (False, error_msg) on failure.
    """
    output_dir = os.path.join(npy_root, os.path.basename(subject_dir))
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, 'frames.npy')
    if os.path.exists(npy_path):
        return True, npy_path   # already done

    try:
        frame_files = sorted(
            f for f in os.listdir(subject_dir) if f.endswith('.ply')
        )[:seq_len]
        if not frame_files:
            return False, f'No PLY files in {subject_dir}'

        frames = np.stack([
            _load_ply_verts(os.path.join(subject_dir, ff))
            for ff in frame_files
        ])  # (T, N, 3)
        np.save(npy_path, frames)
        return True, npy_path
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# ECG helpers
# ---------------------------------------------------------------------------

def _process_ecg(ecg_path, ecg_npy_dir):
    """
    Load an ECG CSV, z-score normalise per lead, and save as <subid>.npy.
    Returns (True, path) on success, (False, error_msg) on failure.
    """
    fname = os.path.basename(ecg_path)
    subid = fname.split('_')[0]
    out_path = os.path.join(ecg_npy_dir, f'{subid}.npy')
    if os.path.exists(out_path):
        return True, out_path

    try:
        ecg = pd.read_csv(ecg_path, index_col=0).values.astype(np.float32)  # (12, 600)
        mu  = ecg.mean(axis=1, keepdims=True)
        std = ecg.std(axis=1,  keepdims=True) + 1e-8
        ecg = (ecg - mu) / std
        np.save(out_path, ecg)
        return True, out_path
    except Exception as e:
        return False, str(e)


def _run(worker, items, num_workers, tag):
    n_done = n_fail = 0
    with Pool(processes=num_workers) as pool:
        for ok, result in tqdm(pool.imap_unordered(worker, items), total=len(items),
                               desc=f'[{tag}] Pre-computing'):
            if ok:
                n_done += 1
            else:
                n_fail += 1
                tqdm.write(f'FAIL: {result}')
    print(f'[{tag}] Done: {n_done} converted, {n_fail} failed.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mesh
    parser.add_argument('--mesh_dir',    default=str(DATA / 'ctrl_mesh_ply'),
                        help='Root directory of control-mesh PLY folders (dataset.mesh_dir)')
    parser.add_argument('--npy_dir',     default=str(DATA / 'ctrl_mesh'),
                        help='Output root for frames.npy files (dataset.npy_dir)')
    parser.add_argument('--seq_len',     default=50, type=int,
                        help='Number of frames per subject to include')
    # ECG
    parser.add_argument('--ecg_dir',     default='',
                        help='Directory of MRI-aligned resting ECG CSV files. '
                             'Leave empty to skip ECG pre-computation.')
    parser.add_argument('--ecg_npy_dir', default=str(DATA / 'ecg_resting'),
                        help='Output directory for per-subject ECG .npy files (dataset.ecg_dir)')
    # Common
    parser.add_argument('--num_workers', default=min(16, os.cpu_count()), type=int,
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    # ── Mesh pre-computation ──────────────────────────────────────────────────
    if os.path.isdir(args.mesh_dir):
        subject_dirs = [
            os.path.join(args.mesh_dir, d)
            for d in os.listdir(args.mesh_dir)
            if os.path.isdir(os.path.join(args.mesh_dir, d))
        ]
        print(f'[Mesh] Found {len(subject_dirs)} subject folders under {args.mesh_dir}')
        print(f'[Mesh] Converting PLY → .npy with {args.num_workers} workers …')
        _run(partial(_process_subject, npy_root=args.npy_dir, seq_len=args.seq_len),
             subject_dirs, args.num_workers, 'Mesh')
    else:
        print(f'[Mesh] mesh_dir not found ({args.mesh_dir}), skipping mesh pre-computation.')

    # ── ECG pre-computation ───────────────────────────────────────────────────
    if args.ecg_dir:
        os.makedirs(args.ecg_npy_dir, exist_ok=True)
        ecg_files = [
            os.path.join(args.ecg_dir, f)
            for f in os.listdir(args.ecg_dir)
            if f.endswith('.csv')
        ]
        print(f'[ECG] Found {len(ecg_files)} CSV files under {args.ecg_dir}')
        print(f'[ECG] Converting CSV → .npy with {args.num_workers} workers …')
        _run(partial(_process_ecg, ecg_npy_dir=args.ecg_npy_dir),
             ecg_files, args.num_workers, 'ECG')
    else:
        print('[ECG] --ecg_dir not set, skipping ECG pre-computation.')


if __name__ == '__main__':
    main()
