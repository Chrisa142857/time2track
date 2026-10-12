"""emg2pose dataset -> per-session .npy layout, session metadata and the test index.

  convert   official emg2pose release (metadata.csv + one HDF5 file per session) -> --data with
            {stem}.emg.npy (T, 16) float32, {stem}.joints.npy (T, 20) float32, {stem}.meta.json.
            The .npy files make every window a real mmap slice.
  meta      --data -> session_meta.json {stem: {split, user, stage}} for all sessions, and
            test_index.json {sids, recs, users} for the official test split (sids sorted, one rec
            per sid with user, stage and the held-out flags, users sorted).

The test index fixes the session ORDER that the scorers use (the first clean session of each user in
this order represents that user in APCR/APDR), so keep it sorted as written here.
"""
import argparse
import glob
import json
import os

import numpy as np

META_KEYS = ('user', 'session', 'stage', 'side', 'moving_hand', 'held_out_user', 'held_out_stage',
             'split', 'generalization')


def convert(a):
    import csv
    import h5py
    os.makedirs(a.data, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(a.emg2pose_dataset, 'metadata.csv'))))
    for i, r in enumerate(rows):
        stem = r['filename']
        with h5py.File(os.path.join(a.emg2pose_dataset, f'{stem}.hdf5'), 'r') as f:
            ts = f['emg2pose']['timeseries']
            np.save(os.path.join(a.data, f'{stem}.emg.npy'), np.ascontiguousarray(ts['emg'], np.float32))
            np.save(os.path.join(a.data, f'{stem}.joints.npy'),
                    np.ascontiguousarray(ts['joint_angles'], np.float32))
        m = {k: r[k] for k in META_KEYS}
        for k in ('held_out_user', 'held_out_stage'):
            m[k] = m[k].strip().lower() == 'true'
        json.dump(m, open(os.path.join(a.data, f'{stem}.meta.json'), 'w'))
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(rows)}', flush=True)
    print(f'converted {len(rows)} sessions -> {a.data}')


def meta(a):
    suf = '.meta.json'
    stems = sorted(os.path.basename(p)[:-len(suf)] for p in glob.glob(os.path.join(a.data, f'*{suf}')))
    M = {s: json.load(open(os.path.join(a.data, s + suf))) for s in stems}
    json.dump({s: {'split': m['split'], 'user': m['user'], 'stage': m['stage']} for s, m in M.items()},
              open(a.session_meta, 'w'))
    sids = sorted(s for s, m in M.items() if m['split'] == 'test')
    recs = [{'sid': s, 'user': M[s]['user'], 'stage': M[s]['stage'],
             'ho_stage': M[s]['held_out_stage'], 'ho_user': M[s]['held_out_user']} for s in sids]
    users = sorted({r['user'] for r in recs})
    json.dump({'sids': sids, 'recs': recs, 'users': users}, open(a.test_index, 'w'))
    print(f'{a.session_meta}: {len(M)} sessions | {a.test_index}: {len(sids)} test sessions, '
          f'{len(users)} users')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['convert', 'meta'])
    ap.add_argument('--emg2pose-dataset', default=os.environ.get('EMG2POSE_DATASET'),
                    help='unpacked emg2pose_dataset.tar (metadata.csv + *.hdf5), convert only')
    ap.add_argument('--data', default=os.environ.get('EMG2POSE_DATA'),
                    help='per-session .npy directory (written by convert, read by meta)')
    ap.add_argument('--session-meta', default='work/session_meta.json')
    ap.add_argument('--test-index', default='work/test_index.json')
    a = ap.parse_args()
    if not a.data or (a.cmd == 'convert' and not a.emg2pose_dataset):
        ap.error('--data (and --emg2pose-dataset for convert) or EMG2POSE_DATA / EMG2POSE_DATASET required')
    for p in (a.session_meta, a.test_index):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    {'convert': convert, 'meta': meta}[a.cmd](a)
