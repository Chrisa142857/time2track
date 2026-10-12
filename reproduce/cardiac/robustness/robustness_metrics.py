#!/usr/bin/env python
"""Whole-heart metrics and subject bootstrap for each perturbation condition (Appendix A.6).

RV, RMSE, PAR, APCR and APDR as in the whole-heart row of eval_vertex_multiscale.py, from the
per-subject sufficient statistics of reproduce/metrics/cardiac_stats.py. The GT is shared across
conditions and read once from --gt-dir (npy_eval of the unperturbed evaluation). Each condition
supplies only {subid}_pred.npy in its own directory. Metrics are computed on the subjects of the
list that are present in the GT and in every condition, so all comparisons are within-subject.

Conditions are given as name:pred_dir pairs. 'clean' should point at a matched-batching clean
run of perturbed_inference.py (same subset and order) for an exact within-subject baseline.

Tables 7 and 8 report the bootstrap medians of `boot` (RV, RMSE, PAR, and |median - 1| for APCR and
APDR), with the 2.5 and 97.5 percentiles as the 95% intervals of RV and RMSE.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metrics"))
from cardiac_stats import (rv_per_vertex, make_resid_op, Store, metrics, dev)  # noqa: E402

METS = ["RV", "RMSE", "PAR", "APCR", "APDR"]


def pctl(arr):  # arr [ndraws, 5] -> per-metric [median, lo, hi]
    q = np.percentile(arr, [50, 2.5, 97.5], axis=0)  # [3,5]
    return {m: [float(q[0, j]), float(q[1, j]), float(q[2, j])] for j, m in enumerate(METS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--subject-list', required=True)
    ap.add_argument('--gt-dir', required=True, help='npy_eval dir holding {subid}_gt.npy')
    ap.add_argument('--conditions', nargs='+', required=True, help='name:pred_dir ...')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ndraws', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    conds = [(c.split(':', 1)[0], c.split(':', 1)[1]) for c in args.conditions]

    with open(args.subject_list) as f:
        wanted = [ln.strip() for ln in f if ln.strip()]

    gt_dir = args.gt_dir
    # keep only subjects that have GT and are present in EVERY condition dir
    subids = [s for s in wanted if os.path.exists(os.path.join(gt_dir, f'{s}_gt.npy'))]
    for name, d in conds:
        subids = [s for s in subids if os.path.exists(os.path.join(d, f'{s}_pred.npy'))]
    subids = sorted(subids)
    N = len(subids)
    print(f'{os.path.basename(args.subject_list)}: {len(wanted)} requested, '
          f'{N} present in GT and all {len(conds)} conditions')
    if N == 0:
        raise SystemExit('no subjects present in all conditions')

    sample = np.load(os.path.join(gt_dir, f'{subids[0]}_gt.npy'))
    T, V, C = sample.shape
    TT, D = T * T, T * V * C
    resid = make_resid_op(T)
    st = Store(N, V, TT, D, T, C)

    # ---- GT once ----
    t0 = time.time()
    for i, sid in enumerate(subids):
        gt = torch.as_tensor(np.load(os.path.join(gt_dir, f'{sid}_gt.npy')),
                             dtype=torch.float64, device=dev)
        rg = (resid @ gt.reshape(T, -1)).reshape(T, V, C)
        xg = rg.reshape(T, -1); xg = xg - xg.mean(0, keepdim=True)
        st.gt_raw[i] = gt
        st.gtsq[i] = (gt * gt).sum((0, 2))
        st.peakg[i] = (rg.amax(0) - rg.amin(0)).mean(-1)
        st.Gg[i] = (xg @ xg.T).reshape(-1)
        Mg = xg.reshape(-1); st.Mg[i] = Mg; st.sqg[i] = (Mg * Mg).sum()
    print(f'GT loaded {N} in {time.time()-t0:.0f}s')

    idx_all = torch.arange(N, device=dev)
    results = {'meta': {'subject_list': args.subject_list, 'n': N, 'ndraws': args.ndraws,
                        'metrics': METS, 'conditions': dict(conds)},
               'point': {}, 'boot': {}}

    def boot(n, ndraws):
        out = []
        for _ in range(ndraws):
            sel = idx_all[torch.randint(N, (n,), generator=g, device=dev)]
            out.append(metrics(st, sel))
        return np.stack(out)

    for name, d in conds:
        for i, sid in enumerate(subids):
            p = torch.as_tensor(np.load(os.path.join(d, f'{sid}_pred.npy')),
                                dtype=torch.float64, device=dev)
            gt = st.gt_raw[i]
            rp = (resid @ p.reshape(T, -1)).reshape(T, V, C)
            st.rv[i] = rv_per_vertex(p, gt)
            st.l2sq[i] = ((p - gt) ** 2).sum((0, 2))
            st.peakp[i] = (rp.amax(0) - rp.amin(0)).mean(-1)
            xp = rp.reshape(T, -1); xp = xp - xp.mean(0, keepdim=True)
            st.Gp[i] = (xp @ xp.T).reshape(-1)
            Mp = xp.reshape(-1); st.Mp[i] = Mp; st.sqp[i] = (Mp * Mp).sum()
        pt = metrics(st, idx_all)
        results['point'][name] = pt.tolist()
        b = results['boot'][name] = pctl(boot(N, args.ndraws))
        print(f'[{name:14s}] point RV/RMSE/PAR/APCR/APDR = '
              f'{pt[0]:.3f}/{pt[1]:.3f}/{pt[2]:.3f}/{pt[3]:.3f}/{pt[4]:.3f}')
        print(f'[{name:14s}] table row (bootstrap median) RV {b["RV"][0]:.3f} '
              f'[{b["RV"][1]:.3f}, {b["RV"][2]:.3f}]  RMSE {b["RMSE"][0]:.3f} '
              f'[{b["RMSE"][1]:.3f}, {b["RMSE"][2]:.3f}]  PAR {b["PAR"][0]:.3f}  '
              f'|APCR-1| {abs(b["APCR"][0] - 1):.3f}  |APDR-1| {abs(b["APDR"][0] - 1):.3f}')

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
