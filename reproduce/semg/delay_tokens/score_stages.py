"""Paper metrics per gesture stage and over all test sessions, per arm and seed.

Reads {npy_root}/{arm}_s{seed}/{sid}_pred.npy and _gt.npy (49, 20, 1) for every sid of --index and
scores them with the shared metric code (reproduce/metrics/paper_metrics.py): RMSE, |PAR-1| (masked
cell, IK-failure frames dropped inside each session) and |APCR-1|, |APDR-1| on the cubic-detrended
curves. Session statistics are computed once per arm and a stage is a row subset of them. Stages with
fewer than --min-users users are skipped.

--unit sets the units of the pairwise diversity sample (APCR, APDR) as in the paper runs:
  user     the first session of each user without IK failure, in index order (vemg2pose)
  session  every session of the subset (EMGFormer)

The ground truth must be bitwise identical across arms (asserted), so every difference between arms
is on the prediction side. Run on CPU, as the paper numbers were.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'metrics'))
import paper_metrics as PM  # noqa: E402


def load(d, sids, kind):
    return np.stack([np.load(os.path.join(d, f'{s}_{kind}.npy')) for s in sids])


def cell(st, cp, cg, idx, clean, user_of, unit):
    """RMSE, PAR, APCR, APDR on the session subset idx (ascending)."""
    _rv, rmse, par, _pj = PM.pointwise_from(st, torch.as_tensor(idx, dtype=torch.long, device=PM.DEV))
    if unit == 'user':
        seen, sel = set(), []
        for i in idx:
            if clean[i] and user_of[i] not in seen:
                seen.add(user_of[i])
                sel.append(i)
        sel = sorted(sel)
    else:
        sel = list(idx)
    mats = (*PM.pair_matrices(cp[sel]), *PM.pair_matrices(cg[sel]))
    apcr, apdr, *_ = PM.diversity_from(mats, torch.arange(len(sel), device=PM.DEV))
    return dict(RMSE=rmse, PAR=par, absPARm1=abs(par - 1.0), APCR=apcr, APDR=apdr,
                absAPCRm1=abs(apcr - 1.0), absAPDRm1=abs(apdr - 1.0), n_div_units=len(sel))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npy-root', required=True, help='directory holding one <arm>_s<seed>/ per run')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
    ap.add_argument('--index', required=True, help='filtered test index (sids, recs) of the scored sessions')
    ap.add_argument('--unit', required=True, choices=['user', 'session'])
    ap.add_argument('--min-users', type=int, default=10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert PM.DEV.type == 'cpu', 'score on CPU (CUDA_VISIBLE_DEVICES=""), as the paper numbers were'

    ix = json.load(open(a.index))
    sids, recs = ix['sids'], ix['recs']
    assert sids == [r['sid'] for r in recs], 'index order mismatch'
    user_of = [r['user'] for r in recs]
    stage_of = [r['stage'] for r in recs]
    tags = [f'{arm}_s{s}' for arm in a.arms for s in a.seeds]

    G0 = load(os.path.join(a.npy_root, tags[0]), sids, 'gt')
    for t in tags[1:]:
        assert np.array_equal(G0, load(os.path.join(a.npy_root, t), sids, 'gt')), f'{t}: GT differs'
    gt = torch.as_tensor(G0, dtype=torch.float64, device=PM.DEV)
    ik_ok = ~(gt.squeeze(-1).abs() < 1e-8).all(-1)                 # [N, T], IK-failure frames
    clean = ik_ok.all(1).cpu().numpy()
    N, T = gt.shape[0], gt.shape[1]
    cg = torch.stack([PM.detrend(gt[i]) for i in range(N)]).reshape(N, T, -1)

    by = {}
    for i, s in enumerate(stage_of):
        by.setdefault(s, []).append(i)
    keep = [(s, idx) for s, idx in sorted(by.items()) if len({user_of[i] for i in idx}) >= a.min_users]
    print(f'{N} sessions, {len(keep)}/{len(by)} stages with >= {a.min_users} users', flush=True)

    overall, rows = {}, [{'stage': s, 'n': len(idx), 'users': len({user_of[i] for i in idx})}
                         for s, idx in keep]
    for t in tags:
        pred = torch.as_tensor(load(os.path.join(a.npy_root, t), sids, 'pred'),
                               dtype=torch.float64, device=PM.DEV)
        st = PM.session_stats(pred, gt, ik_ok)
        cp = torch.stack([PM.detrend(pred[i]) for i in range(N)]).reshape(N, T, -1)
        overall[t] = cell(st, cp, cg, list(range(N)), clean, user_of, a.unit)
        for r, (_, idx) in zip(rows, keep):
            r[t] = cell(st, cp, cg, idx, clean, user_of, a.unit)
        o = overall[t]
        print(f'  {t}: RMSE {o["RMSE"]:.4f} |PAR-1| {o["absPARm1"]:.4f} |APCR-1| {o["absAPCRm1"]:.4f} '
              f'|APDR-1| {o["absAPDRm1"]:.4f} ({o["n_div_units"]} diversity units)', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({'n_sessions': N, 'n_users': len(set(user_of)), 'unit': a.unit, 'arms': a.arms,
               'seeds': a.seeds, 'overall': overall, 'stages': rows}, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
