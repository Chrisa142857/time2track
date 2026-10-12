"""
The paper's metric family (RV, RMSE, PAR, |APCR-1|, |APDR-1|) on generic [S, T, P, C] trajectories,
used for emg2pose joint-angle trajectories and for the bootstrap of the cardiac diversity metrics.

Port map (cardiac -> EMG), all definitions copied from
eval_vertex_multiscale.py (repository root):

  cardiac subject  [49, 388, 3]   ->  EMG session  [49, 20, 1]
  "control point"  = mesh vertex  ->  "control point" = joint DoF (20)
  channels C=3 (x,y,z)            ->  C=1 (the angle itself)
  detrend          = cubic poly residual over time  (identical, degree 3)
  RV   = meanRV        per-point RV coeff on RAW curves, then mean over points
  RMSE = predGtL2N_rom sqrt(sum_s ||p-g||^2) / sqrt(sum_s ||g||^2) per point,
                       then mean over points
  PAR  = peakAmpR_rom  sum_s ptp(detrended pred) / sum_s ptp(detrended gt)
  APCR = xsubjPredRV / xsubjGTRV      (ratio of means, matches paper Sec. 4)
  APDR = xsDistPred   / xsDistGT      (ratio of means, matches paper Sec. 4)

With C=1 the per-point RV coefficient reduces algebraically to the SQUARED
Pearson correlation of the two time series -- stated, not hidden.

Diversity ("inter-subject") unit: reported both per SESSION and per USER.
The pairwise sample is FIXED and MATCHED across all methods (same unit ids,
same order). Bootstrap CIs resample the units with replacement, B >= 1000.
"""
import argparse, json, os, sys
import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-9
_DETREND = {}


def detrend(x, degree=3):
    mat = x.reshape(x.shape[0], -1)
    t = mat.shape[0]
    deg = min(degree, t - 1)
    op = _DETREND.get((t, deg))
    if op is None:
        v = torch.vander(torch.arange(t, dtype=torch.float64, device=DEV),
                         N=deg + 1, increasing=False)
        op = torch.eye(t, dtype=torch.float64, device=DEV) - v @ torch.linalg.pinv(v)
        _DETREND[(t, deg)] = op
    return (op @ mat).reshape(x.shape)


def rv_per_point(x, y):
    """[T,V,C] x [T,V,C] -> [V]; exact port of _rv_per_vertex."""
    x = torch.nan_to_num(x - x.nanmean(0, keepdim=True))
    y = torch.nan_to_num(y - y.nanmean(0, keepdim=True))
    xy = torch.einsum('tvc,tvd->vcd', x, y)
    xx = torch.einsum('tvc,tvd->vcd', x, x)
    yy = torch.einsum('tvc,tvd->vcd', y, y)
    num = (xy * xy).sum((1, 2))
    den = torch.sqrt((xx * xx).sum((1, 2)) * (yy * yy).sum((1, 2)))
    out = torch.full_like(num, float('nan'))
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


# ------------------------------------------------------- per-session stats ---
def session_stats(pred, gt, mask):
    """pred/gt [N,T,J,1] float64, mask [N,T] bool.
    Returns per-session per-joint arrays so any bootstrap is a row resample."""
    N, T, J, C = pred.shape
    rv = torch.full((N, J), float('nan'), dtype=torch.float64, device=DEV)
    pk_p = torch.zeros(N, J, dtype=torch.float64, device=DEV)
    pk_g = torch.zeros(N, J, dtype=torch.float64, device=DEV)
    l2 = torch.zeros(N, J, dtype=torch.float64, device=DEV)
    gn = torch.zeros(N, J, dtype=torch.float64, device=DEV)
    ok = torch.zeros(N, dtype=torch.bool, device=DEV)
    for i in range(N):
        k = mask[i]
        if int(k.sum()) < 5:
            continue
        p, g = pred[i][k], gt[i][k]
        rp, rg = detrend(p), detrend(g)
        rv[i] = rv_per_point(p, g)
        pk_p[i] = (rp.amax(0) - rp.amin(0)).mean(-1)
        pk_g[i] = (rg.amax(0) - rg.amin(0)).mean(-1)
        l2[i] = ((p - g) ** 2).sum((0, 2))
        gn[i] = (g * g).sum((0, 2))
        ok[i] = True
    return dict(rv=rv, pk_p=pk_p, pk_g=pk_g, l2=l2, gn=gn, ok=ok)


def pointwise_from(st, idx):
    """Aggregate per-session stats over a (possibly bootstrap) index set."""
    sel = idx[st['ok'][idx]]
    rv = st['rv'][sel]
    fin = torch.isfinite(rv)
    rv_pt = torch.where(fin, rv, torch.zeros_like(rv)).sum(0) / fin.sum(0).clamp(min=1)
    rmse_pt = torch.sqrt(st['l2'][sel].sum(0)) / torch.sqrt(st['gn'][sel].sum(0)).clamp(min=EPS)
    par_pt = st['pk_p'][sel].sum(0) / st['pk_g'][sel].sum(0).clamp(min=EPS)
    f = lambda a: float(a[torch.isfinite(a)].mean())
    return f(rv_pt), f(rmse_pt), f(par_pt), rv_pt


# -------------------------------------------------------- pairwise matrices --
def pair_matrices(curves):
    """curves: [K, T, D] detrended. Returns full [K,K] RV and normalised
    distance matrices (exact port of _sample_diversity, vectorised)."""
    x = curves - curves.nanmean(1, keepdim=True)
    x = torch.nan_to_num(x)
    K, T, D = x.shape
    gram = torch.bmm(x, x.transpose(1, 2)).reshape(K, -1)     # [K, T*T]
    dot = gram @ gram.T
    nrm = torch.sqrt(torch.diag(dot))
    den = nrm[:, None] * nrm[None, :]
    rv = torch.full_like(dot, float('nan'))
    rv[den > 0] = dot[den > 0] / den[den > 0]

    mat = x.reshape(K, -1)                                    # [K, T*D]
    sq = (mat * mat).sum(1)
    d2 = torch.clamp(sq[:, None] + sq[None, :] - 2.0 * (mat @ mat.T), min=0.0) / mat.shape[1]
    dist = torch.sqrt(d2)
    scale = torch.sqrt(sq / mat.shape[1])                     # per-unit RMS magnitude
    return rv, dist, scale


def _offdiag_mean(M, idx):
    """Mean over unordered pairs of distinct ORIGINAL units in idx."""
    sub = M[idx][:, idx]
    a = idx[:, None].expand_as(sub)
    b = idx[None, :].expand_as(sub)
    keep = (a != b) & torch.isfinite(sub)
    keep = torch.triu(keep, diagonal=1)
    return sub[keep].mean()


def diversity_from(mats, idx):
    """mats = (rvP,dP,scP, rvG,dG,scG). Returns APCR, APDR (ratio of means)."""
    rvP, dP, scP, rvG, dG, scG = mats
    xp = _offdiag_mean(rvP, idx)
    xg = _offdiag_mean(rvG, idx)
    dp = _offdiag_mean(dP, idx) / (scP[idx].mean() + EPS)
    dg = _offdiag_mean(dG, idx) / (scG[idx].mean() + EPS)
    return float(xp / xg), float(dp / dg), float(xp), float(xg), float(dp), float(dg)


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v.size else [float('nan')] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--index', required=True, help='test_index.json of the scored sessions')
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--div-k', type=int, default=1024, help='session-level pairwise sample size')
    args = ap.parse_args()

    ix = json.load(open(args.index))
    sids, users = ix['sids'], ix['users']
    user_of = {r['sid']: r['user'] for r in ix['recs']}

    P = np.stack([np.load(os.path.join(args.dir, f'{s}_pred.npy')) for s in sids])
    G = np.stack([np.load(os.path.join(args.dir, f'{s}_gt.npy')) for s in sids])
    pred = torch.as_tensor(P, dtype=torch.float64, device=DEV)
    gt = torch.as_tensor(G, dtype=torch.float64, device=DEV)
    N, T = pred.shape[0], pred.shape[1]
    print(f'{args.name}: {tuple(pred.shape)}', flush=True)
    assert not torch.isnan(gt).any()

    ik_ok = ~(gt.squeeze(-1).abs() < 1e-8).all(-1)                  # [N,T]
    ones = torch.ones(N, T, dtype=torch.bool, device=DEV)
    clean = ik_ok.all(1)                                            # sessions with 0 IK failures
    print(f'clean sessions (all {T} frames valid): {int(clean.sum())}/{N}', flush=True)

    # subject-level unit ids for the bootstrap (user id per session)
    uidx = torch.as_tensor([users.index(user_of[s]) for s in sids], device=DEV)

    res = {'name': args.name, 'n_sessions': N,
           'n_clean_sessions': int(clean.sum()), 'n_users': len(users)}
    rng = np.random.default_rng(0)

    for tag, msk, pool in (('unmasked', ones, torch.arange(N, device=DEV)),
                           ('masked', ik_ok, torch.nonzero(clean).squeeze(1))):
        # Point-wise metrics use ALL sessions; the mask drops bad FRAMES inside
        # each session (official semantics). Diversity needs equal-length
        # curves, so it uses the `pool` of fully-clean sessions instead.
        st = session_stats(pred, gt, msk)
        allidx = torch.arange(N, device=DEV)
        pt = pointwise_from(st, allidx)
        r = {'RV': pt[0], 'RMSE': pt[1], 'PAR': pt[2],
             'per_joint_RV': pt[3].cpu().numpy().tolist()}

        # ---- bootstrap the point-wise metrics by resampling USERS -----------
        base = allidx
        bu = uidx[base]
        by_user = [torch.nonzero(bu == u).squeeze(1) for u in range(len(users))]
        by_user = [base[g] for g in by_user if g.numel()]
        bs = {'RV': [], 'RMSE': [], 'PAR': []}
        for _ in range(args.boot):
            pick = rng.integers(0, len(by_user), len(by_user))
            idx = torch.cat([by_user[i] for i in pick])
            a, b, c, _ = pointwise_from(st, idx)
            bs['RV'].append(a); bs['RMSE'].append(b); bs['PAR'].append(c)
        for k in bs:
            r[f'{k}_CI'] = ci(bs[k])

        # ---- diversity: session-level and user-level ------------------------
        dpool = pool.cpu().numpy()          # unmasked: all N; masked: clean only
        for unit in ('session', 'user'):
            if unit == 'session':
                sel = np.sort(rng_fixed(dpool, min(args.div_k, len(dpool))))
            else:
                # one deterministic session per user (first in sorted order)
                seen, sel = set(), []
                for i in dpool:
                    u = user_of[sids[i]]
                    if u not in seen:
                        seen.add(u); sel.append(i)
                sel = np.array(sorted(sel))
            sel_t = torch.as_tensor(sel, dtype=torch.long, device=DEV)
            cp = torch.stack([detrend(pred[i]) for i in sel_t]).reshape(len(sel), T, -1)
            cg = torch.stack([detrend(gt[i]) for i in sel_t]).reshape(len(sel), T, -1)
            mats = (*pair_matrices(cp), *pair_matrices(cg))
            loc = torch.arange(len(sel), device=DEV)
            apcr, apdr, xp, xg, dp, dg = diversity_from(mats, loc)
            r[f'{unit}_APCR'] = apcr
            r[f'{unit}_APDR'] = apdr
            r[f'{unit}_absAPCRm1'] = abs(apcr - 1)
            r[f'{unit}_absAPDRm1'] = abs(apdr - 1)
            r[f'{unit}_xsubjPredRV'] = xp; r[f'{unit}_xsubjGTRV'] = xg
            r[f'{unit}_xsDistPred'] = dp; r[f'{unit}_xsDistGT'] = dg
            r[f'{unit}_n_units'] = int(len(sel))
            r[f'{unit}_unit_ids'] = [sids[i] for i in sel]

            # bootstrap over units (resample with replacement)
            ba, bd = [], []
            uu = np.array([users.index(user_of[sids[i]]) for i in sel])
            groups = [np.nonzero(uu == u)[0] for u in np.unique(uu)]
            for _ in range(args.boot):
                pick = rng.integers(0, len(groups), len(groups))
                bidx = torch.as_tensor(np.concatenate([groups[i] for i in pick]),
                                       dtype=torch.long, device=DEV)
                x, y, *_ = diversity_from(mats, bidx)
                ba.append(x); bd.append(y)
            r[f'{unit}_APCR_CI'] = ci(ba)
            r[f'{unit}_APDR_CI'] = ci(bd)
            r[f'{unit}_absAPCRm1_CI'] = ci([abs(v - 1) for v in ba])
            r[f'{unit}_absAPDRm1_CI'] = ci([abs(v - 1) for v in bd])
        res[tag] = r
        print(f'  {tag}: RV={r["RV"]:.3f} RMSE={r["RMSE"]:.3f} PAR={r["PAR"]:.3f} '
              f'APCR_sess={r["session_APCR"]:.3f} APDR_sess={r["session_APDR"]:.3f} '
              f'APCR_user={r["user_APCR"]:.3f} APDR_user={r["user_APDR"]:.3f}', flush=True)

    json.dump(res, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


_FIXED = {}


def rng_fixed(cand, k):
    """Deterministic subsample, identical for every method (seed 0)."""
    key = (len(cand), k)
    if key not in _FIXED:
        _FIXED[key] = np.random.default_rng(0).choice(len(cand), k, replace=False)
    return cand[_FIXED[key]]


if __name__ == '__main__':
    main()
