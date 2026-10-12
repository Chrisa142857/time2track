"""First usable gesture cycle of every emg2pose session, and the window indices anchored at it.

  detect      cycle starts per session from the ground-truth joint speed: joints at 100 Hz, summed
              |frame difference| over the 20 joints, 2nd-order Butterworth low-pass at 3 Hz
              (zero phase), peaks at least 0.30 s apart and above half the 75th percentile. A cycle
              runs from one peak to the next, is kept if it lasts 0.30 to 3.0 s and its mean speed is
              at least 25% of the 75th percentile (rest segments dropped). Writes cycle starts in
              samples (2 kHz) and the session length T.
  anchors     first usable cycle per session: the earliest start c with c >= 1790 (left context) and
              (c - 1790) + window <= T. Writes the anchor map ({stem: anchor, offset = anchor - 1790,
              split, T, n_cycles}) and a training index of steps * batch [stem, offset, roll] items,
              a with-replacement session draw with a random electrode roll in {-1, 0, 1}.
  windows     one [stem, offset, 0] item per anchored session of one split (val/test windows).
  test_index  the test index restricted to the anchored sessions, order kept.

Each session contributes exactly one window (its first usable cycle), so the official train/val/test
split and the session counts are kept and only the window position changes.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

SR, JFS, LC = 2000.0, 100.0, 1790
DATA = None                                    # set from --data before the worker pool forks


def cycles_for(stem, min_s=0.30, max_s=3.0, rest_frac=0.25):
    """-> (stem, [cycle_start_sample, ...], T); [] if none, (stem, [], -1, error) on failure."""
    from scipy.signal import butter, filtfilt, find_peaks
    try:
        b, a = butter(2, 3.0 / (JFS / 2), 'low')
        T = int(np.load(f'{DATA}/{stem}.emg.npy', mmap_mode='r').shape[0])
        j = np.load(f'{DATA}/{stem}.joints.npy', mmap_mode='r')
        step = int(SR / JFS)
        v = np.abs(np.diff(np.asarray(j[::step, :20], np.float32), axis=0)).sum(1)
        if v.size < 64:
            return stem, [], T
        v = filtfilt(b, a, v)
        ref = np.percentile(v, 75)
        if ref <= 0:
            return stem, [], T
        pk, _ = find_peaks(v, distance=int(min_s * JFS), height=0.5 * ref)
        out = []
        for s, e in zip(pk[:-1], pk[1:]):
            dur = (e - s) / JFS
            if not (min_s <= dur <= max_s):
                continue
            if v[s:e].mean() < rest_frac * ref:          # rest / idle -> drop
                continue
            out.append(int(s * step))
        return stem, out, T
    except Exception as exc:                              # noqa: BLE001  (reported, not swallowed)
        return stem, [], -1, repr(exc)


def detect(a):
    global DATA
    DATA = a.data
    suf = '.emg.npy'
    stems = sorted(os.path.basename(p)[:-len(suf)] for p in glob.glob(f'{DATA}/*{suf}'))
    assert stems and os.path.exists(f'{DATA}/{stems[0]}.joints.npy'), f'no sessions in {DATA}'
    res, nerr = {}, 0
    with Pool(a.nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(cycles_for, stems, chunksize=16)):
            if len(r) == 4:
                nerr += 1
                if nerr <= 3:
                    print(f'  ERROR {r[0]}: {r[3]}', flush=True)
                continue
            res[r[0]] = {'cycles': r[1], 'T': r[2]}
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(stems)}', flush=True)
    json.dump(res, open(a.cycles, 'w'))
    n = sum(len(v['cycles']) for v in res.values())
    print(f'errors {nerr}/{len(stems)} | wrote {a.cycles}: {len(res)} sessions, {n} cycles')


def first_anchor(cycles, T, window):
    for c in cycles:                                      # ascending
        if c >= LC and (c - LC) + window <= T:
            return int(c)
    return None


def anchors(a):
    meta = json.load(open(a.session_meta))
    cyc = json.load(open(a.cycles))
    anch, by_split = {}, {}
    for stem, v in cyc.items():
        m = meta.get(stem)
        if not m:
            continue
        sp = m.get('split')
        by_split.setdefault(sp, [0, 0])
        by_split[sp][1] += 1
        c = first_anchor(v['cycles'], v['T'], a.window)
        if c is None:
            continue
        anch[stem] = {'anchor': c, 'offset': int(c - LC), 'split': sp, 'T': v['T'],
                      'n_cycles': len(v['cycles'])}
        by_split[sp][0] += 1
    for sp, (ok, tot) in sorted(by_split.items(), key=lambda kv: str(kv[0])):
        print(f'  {str(sp):6s} {ok:6d}/{tot:6d} sessions with a usable cycle (window {a.window})')
    json.dump(anch, open(a.anchors, 'w'))
    print(f'wrote {a.anchors}: {len(anch)} session anchors')
    if not a.train_index:
        return
    stems = sorted(s for s, v in anch.items() if v['split'] == a.split)
    rng = np.random.default_rng(a.seed)
    n = a.steps * a.batch
    pick = rng.integers(0, len(stems), n)
    items = [[stems[pick[i]], anch[stems[pick[i]]]['offset'],
              int(rng.choice([-1, 0, 1])) if a.rotaug else 0] for i in range(n)]
    json.dump({'window': a.window, 'batch': a.batch, 'seed': a.seed, 'split': a.split,
               'rotaug': bool(a.rotaug), 'first_cycle_anchored': True, 'items': items},
              open(a.train_index, 'w'))
    print(f'wrote {a.train_index}: {len(items)} windows from {len(set(i[0] for i in items))} sessions')


def windows(a):
    anch = json.load(open(a.anchors))
    items = [[s, v['offset'], 0] for s, v in sorted(anch.items()) if v['split'] == a.split]
    json.dump({'window': a.window, 'split': a.split, 'first_cycle_anchored': True, 'items': items},
              open(a.out, 'w'))
    print(f'wrote {a.out}: {len(items)} {a.split} windows')


def test_index(a):
    keep = set(json.load(open(a.anchors)))
    ix = json.load(open(a.test_index))
    recs = [r for r in ix['recs'] if r['sid'] in keep]
    out = {'sids': [s for s in ix['sids'] if s in keep], 'recs': recs,
           'users': sorted({r['user'] for r in recs})}
    json.dump(out, open(a.out, 'w'))
    print(f'wrote {a.out}: {len(out["sids"])}/{len(ix["sids"])} test sessions, {len(out["users"])} users')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['detect', 'anchors', 'windows', 'test_index'])
    ap.add_argument('--data', default=os.environ.get('EMG2POSE_DATA'), help='per-session .npy directory')
    ap.add_argument('--nproc', type=int, default=32)
    ap.add_argument('--cycles', default='work/cycle_index.json')
    ap.add_argument('--session-meta', default='work/session_meta.json')
    ap.add_argument('--anchors', default='work/first_cycle_anchors.json')
    ap.add_argument('--train-index', default=None, help='anchors: also write a training index here')
    ap.add_argument('--window', type=int, default=3838, help='full input window in samples')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', default='train')
    ap.add_argument('--rotaug', type=int, default=1)
    ap.add_argument('--test-index', default='work/test_index.json')
    ap.add_argument('--out', default=None, help='windows / test_index output')
    a = ap.parse_args()
    if a.cmd == 'detect' and not a.data:
        ap.error('--data or EMG2POSE_DATA required')
    if a.cmd in ('windows', 'test_index') and not a.out:
        ap.error('--out required')
    for p in (a.cycles, a.anchors, a.train_index, a.out):
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    {'detect': detect, 'anchors': anchors, 'windows': windows, 'test_index': test_index}[a.cmd](a)
