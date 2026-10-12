"""Whole-heart metrics of the paper from per-subject sufficient statistics.

Replicates eval_vertex_multiscale.py (RV = meanRV, RMSE = predGtL2N_rom, PAR = peakAmpR_rom,
APCR = xsubjPredRV / xsubjGTRV, APDR = xsDistPred / xsDistGT) from per-subject statistics, so that
subjects can be bootstrap-resampled without re-reading the prediction files at every draw.
Used by reproduce/cardiac/pathology and reproduce/cardiac/robustness.
"""
import numpy as np
import torch

EPS = 1e-9

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TRIU = {}


def triu(n):
    if n not in _TRIU:
        _TRIU[n] = torch.triu_indices(n, n, offset=1, device=dev)
    return _TRIU[n]


def make_resid_op(T, deg=3):
    x = torch.arange(T, dtype=torch.float64, device=dev)
    deg = min(deg, T - 1)
    vander = torch.vander(x, N=deg + 1, increasing=False)
    projector = vander @ torch.linalg.pinv(vander)
    return torch.eye(T, dtype=torch.float64, device=dev) - projector


def rv_per_vertex(p, g):  # [T,V,3],[T,V,3] -> [V]
    x = p - p.mean(0, keepdim=True)
    y = g - g.mean(0, keepdim=True)
    xy = torch.einsum("tvc,tvd->vcd", x, y)
    xx = torch.einsum("tvc,tvd->vcd", x, x)
    yy = torch.einsum("tvc,tvd->vcd", y, y)
    num = (xy * xy).sum((1, 2))
    den = torch.sqrt((xx * xx).sum((1, 2)) * (yy * yy).sum((1, 2)))
    out = torch.full_like(num, float("nan"))
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def pairwise_rv_mean(G):  # G [n, TT]
    dot = G @ G.T
    norm = torch.sqrt(torch.clamp(torch.diag(dot), min=0.0))
    den = norm[:, None] * norm[None, :]
    rv = torch.full_like(dot, float("nan"))
    m = den > 0
    rv[m] = dot[m] / den[m]
    r, c = triu(G.shape[0])
    v = rv[r, c]
    v = v[torch.isfinite(v)]
    return v.mean()


def pairwise_dist_mean(M, sq):  # M [n, D], sq [n]
    D = M.shape[1]
    dot = M @ M.T
    d2 = torch.clamp(sq[:, None] + sq[None, :] - 2.0 * dot, min=0.0) / D
    r, c = triu(M.shape[0])
    dist = torch.sqrt(d2[r, c])
    dist = dist[torch.isfinite(dist)]
    denom = torch.sqrt(sq / D).mean() + EPS
    return dist.mean() / denom


class Store:
    """Holds per-subject stats for pred (variant) + gt (shared) on GPU.
    gt_raw, gtsq, peakg, Gg, Mg, sqg are variant-independent (filled once)."""
    def __init__(self, N, V, TT, D, T, C):
        self.gt_raw = torch.empty(N, T, V, C, dtype=torch.float64, device=dev)
        self.rv = torch.empty(N, V, dtype=torch.float64, device=dev)
        self.l2sq = torch.empty(N, V, dtype=torch.float64, device=dev)
        self.gtsq = torch.empty(N, V, dtype=torch.float64, device=dev)
        self.peakp = torch.empty(N, V, dtype=torch.float64, device=dev)
        self.peakg = torch.empty(N, V, dtype=torch.float64, device=dev)
        self.Gp = torch.empty(N, TT, dtype=torch.float64, device=dev)
        self.Gg = torch.empty(N, TT, dtype=torch.float64, device=dev)
        self.Mp = torch.empty(N, D, dtype=torch.float64, device=dev)
        self.Mg = torch.empty(N, D, dtype=torch.float64, device=dev)
        self.sqp = torch.empty(N, dtype=torch.float64, device=dev)
        self.sqg = torch.empty(N, dtype=torch.float64, device=dev)


def metrics(st, idx):
    """idx: LongTensor of subject positions (may repeat for bootstrap)."""
    RVv = torch.nanmean(st.rv[idx], dim=0)
    RV = torch.nanmean(RVv)
    num = torch.sqrt(st.l2sq[idx].sum(0)); den = torch.sqrt(st.gtsq[idx].sum(0))
    RMSEv = num / den
    RMSE = RMSEv[torch.isfinite(RMSEv)].mean()
    pn = st.peakp[idx].sum(0); pd = st.peakg[idx].sum(0)
    PARv = torch.where(pd > 0, pn / pd, torch.full_like(pn, float("nan")))
    PAR = PARv[torch.isfinite(PARv)].mean()
    APCR = pairwise_rv_mean(st.Gp[idx]) / pairwise_rv_mean(st.Gg[idx])
    APDR = pairwise_dist_mean(st.Mp[idx], st.sqp[idx]) / pairwise_dist_mean(st.Mg[idx], st.sqg[idx])
    return torch.stack([RV, RMSE, PAR, APCR, APDR]).cpu().numpy()


