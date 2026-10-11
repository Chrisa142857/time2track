"""Evaluate ECG -> 4D-heart-mesh predictions using 3D vertex trajectories.

Reads <root>/<exp>/{dirname}/<sid>_pred.npy and ..._gt.npy, each shaped
[T, V, 3], and aggregates metrics PER VERTEX across subjects. Whole-heart and
per-material scores are derived by averaging the per-vertex arrays over the
relevant vertex indices.

For ratio metrics we report two flavors:
  - *_rom : ratio-of-means across subjects (robust to per-subject zeros)
  - *_mor : mean-of-per-subject-ratios       (matches the original semantics)

Per-vertex score arrays are saved per experiment as {exp}_per_vertex.npz so
they can be plotted on the mesh afterward.

Cross-subject diversity (xsubjPredRV, xsDistPred, ...) remains subject-level
and is reported once per experiment on the `whole` row only.

Usage
-----
  python eval_vertex_multiscale.py --root ../nips26-physctrl-outputs \
         --mesh-info ../ukb_heart_mesh_visual
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

dirname = "npy_eval"
EPS = 1e-9
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DETREND_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_DTYPE = float

def load_material_info(mesh_info_dir: str) -> Tuple[dict, np.ndarray, List[Tuple[str, np.ndarray]]]:
    with open(os.path.join(mesh_info_dir, "control_material_idmap.json")) as f:
        idmap = json.load(f)
    materials = np.load(os.path.join(mesh_info_dir, "control_materials.npy"))

    material_to_id = {v: k for k, v in idmap.items()}
    groups: List[Tuple[str, np.ndarray]] = []
    for mid in sorted(set(materials) & set(material_to_id.keys())):
        name = f"material_{material_to_id[mid]}"
        idx = np.flatnonzero(materials == mid)
        if idx.size:
            groups.append((name, idx))
    return idmap, materials, groups


def collect_pairs(root: str, max_subjects: Optional[int] = None,
                  subject_list: Optional[set] = None) -> Dict[str, Dict[str, Dict[str, str]]]:
    results: Dict[str, Dict[str, Dict[str, str]]] = {}
    exp_dirs = sorted(glob.glob(os.path.join(root, "*", f"{dirname}/")))
    for d in tqdm(exp_dirs, desc="Experiments"):
        exp = d.split("/")[-3]
        subjects: Dict[str, Dict[str, str]] = {}
        try:
            entries = sorted(e.name for e in os.scandir(d) if e.name.endswith("_pred.npy"))
        except FileNotFoundError:
            continue
        if subject_list is not None:  # restrict to a subgroup (E1/E3); keeps sorted order
            entries = [e for e in entries if e[:-9] in subject_list]
        if max_subjects is not None:
            entries = entries[:max_subjects]
        for fname in entries:
            sid = fname[:-9]
            p_path = os.path.join(d, fname)
            g_path = os.path.join(d, f"{sid}_gt.npy")
            if os.path.exists(g_path):
                subjects[sid] = {"pred": p_path, "gt": g_path}
        if subjects:
            results[exp] = subjects
            tqdm.write(f"  {exp}: {len(subjects)} subjects")
    return results


def _to_tensor(arr) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device=_DEVICE, dtype=torch.float64)
    return torch.as_tensor(np.asarray(arr), dtype=torch.float64, device=_DEVICE)


def _as_matrix_t(traj: torch.Tensor) -> torch.Tensor:
    return traj.reshape(traj.shape[0], -1)


def _center_matrix_t(mat: torch.Tensor) -> torch.Tensor:
    centered = mat - mat.nanmean(dim=0, keepdim=True)
    return torch.nan_to_num(centered)


def _rv_per_vertex(a, b) -> torch.Tensor:
    x = _to_tensor(a)
    y = _to_tensor(b)
    x = torch.nan_to_num(x - x.nanmean(dim=0, keepdim=True))
    y = torch.nan_to_num(y - y.nanmean(dim=0, keepdim=True))
    xy = torch.einsum("tvc,tvd->vcd", x, y)
    xx = torch.einsum("tvc,tvd->vcd", x, x)
    yy = torch.einsum("tvc,tvd->vcd", y, y)
    num = (xy * xy).sum(dim=(1, 2))
    den = torch.sqrt((xx * xx).sum(dim=(1, 2)) * (yy * yy).sum(dim=(1, 2)))
    out = torch.full_like(num, float("nan"))
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _detrend(traj, degree: int = 3):
    arr = _to_tensor(traj)
    mat = arr.reshape(arr.shape[0], -1)
    t = mat.shape[0]
    if t <= 1:
        return torch.zeros_like(arr)

    deg = min(degree, t - 1)
    key = (t, deg)
    resid_op = _DETREND_CACHE.get(key)
    if resid_op is None:
        x = torch.arange(t, dtype=torch.float64, device=_DEVICE)
        vander = torch.vander(x, N=deg + 1, increasing=False)
        projector = vander @ torch.linalg.pinv(vander)
        resid_op = torch.eye(t, dtype=torch.float64, device=_DEVICE) - projector
        _DETREND_CACHE[key] = resid_op
    return (resid_op @ mat).reshape(arr.shape)


def _load_gt_cached(gt_cache: dict, sid: str, gt_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    cached = gt_cache.get(sid)
    if cached is None:
        gt = _to_tensor(np.load(gt_path))
        cached = (gt, _detrend(gt))
        gt_cache[sid] = cached
    return cached


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(num, float("nan"))
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


@dataclass
class VertexRunningSum:
    """Accumulates per-vertex sums of finite values + per-vertex counts."""
    sum: Optional[torch.Tensor] = None
    sum2: Optional[torch.Tensor] = None
    count: Optional[torch.Tensor] = None

    def add(self, values: torch.Tensor, track_sq: bool = False):
        v = values.detach().to(device=_DEVICE, dtype=torch.float64)
        finite = torch.isfinite(v)
        v_clean = torch.where(finite, v, torch.zeros_like(v))
        if self.sum is None:
            self.sum = torch.zeros_like(v_clean)
            self.count = torch.zeros_like(v_clean)
            if track_sq:
                self.sum2 = torch.zeros_like(v_clean)
        self.sum += v_clean
        self.count += finite.to(self.count.dtype)
        if track_sq:
            if self.sum2 is None:
                self.sum2 = torch.zeros_like(v_clean)
            self.sum2 += v_clean * v_clean


@dataclass
class VertexAccumulator:
    """Per-vertex running stats. All arrays end up shaped [V]."""
    raw_rv: VertexRunningSum = field(default_factory=VertexRunningSum)
    detr_rv: VertexRunningSum = field(default_factory=VertexRunningSum)

    # ratio-of-means components (sum numerator / sum denominator across subjects)
    amp_p_sq: VertexRunningSum = field(default_factory=VertexRunningSum)   # sum mean(rp^2) over subj
    amp_g_sq: VertexRunningSum = field(default_factory=VertexRunningSum)
    peak_p: VertexRunningSum = field(default_factory=VertexRunningSum)
    peak_g: VertexRunningSum = field(default_factory=VertexRunningSum)
    rdiff_sq: VertexRunningSum = field(default_factory=VertexRunningSum)   # sum mean((rp-rg)^2) over subj
    l2_sq: VertexRunningSum = field(default_factory=VertexRunningSum)      # sum sum((p-g)^2) over subj
    gt_norm_sq: VertexRunningSum = field(default_factory=VertexRunningSum) # sum sum(g^2) over subj

    # mean-of-per-subject-ratios (one value per subject per vertex)
    amp_ratio_mor: VertexRunningSum = field(default_factory=VertexRunningSum)
    peak_ratio_mor: VertexRunningSum = field(default_factory=VertexRunningSum)
    nrmse_mor: VertexRunningSum = field(default_factory=VertexRunningSum)
    l2_mor: VertexRunningSum = field(default_factory=VertexRunningSum)
    l2_rel_mor: VertexRunningSum = field(default_factory=VertexRunningSum)

    # for subjStdRatio: per-subject |p|.mean(T) and |g|.mean(T) across subjects
    pred_mag: VertexRunningSum = field(default_factory=VertexRunningSum)   # tracks sum + sum2
    gt_mag: VertexRunningSum = field(default_factory=VertexRunningSum)

    def add_subject(self, p: torch.Tensor, g: torch.Tensor,
                    rp: torch.Tensor, rg: torch.Tensor):
        # RV per vertex
        self.raw_rv.add(_rv_per_vertex(p, g))
        self.detr_rv.add(_rv_per_vertex(rp, rg))

        # per-subject per-vertex amplitudes & residuals
        amp_p_sq = (rp * rp).mean(dim=(0, 2))           # [V]
        amp_g_sq = (rg * rg).mean(dim=(0, 2))
        peak_p = (rp.amax(dim=0) - rp.amin(dim=0)).mean(dim=-1)
        peak_g = (rg.amax(dim=0) - rg.amin(dim=0)).mean(dim=-1)
        rdiff_sq = ((rp - rg) ** 2).mean(dim=(0, 2))    # [V]
        l2_sq = ((p - g) ** 2).sum(dim=(0, 2))          # [V]
        gt_norm_sq = (g * g).sum(dim=(0, 2))            # [V]

        amp_p = torch.sqrt(amp_p_sq)
        amp_g = torch.sqrt(amp_g_sq)
        rnrmse = torch.sqrt(rdiff_sq)
        l2 = torch.sqrt(l2_sq)
        gt_norm = torch.sqrt(gt_norm_sq)

        # ratio-of-means accumulators (numerators and denominators separately)
        self.amp_p_sq.add(amp_p_sq)
        self.amp_g_sq.add(amp_g_sq)
        self.peak_p.add(peak_p)
        self.peak_g.add(peak_g)
        self.rdiff_sq.add(rdiff_sq)
        self.l2_sq.add(l2_sq)
        self.gt_norm_sq.add(gt_norm_sq)

        # mean-of-per-subject-ratios
        self.amp_ratio_mor.add(_safe_div(amp_p, amp_g))
        self.peak_ratio_mor.add(_safe_div(peak_p, peak_g))
        self.nrmse_mor.add(_safe_div(rnrmse, amp_g))
        self.l2_mor.add(l2)
        self.l2_rel_mor.add(_safe_div(l2, gt_norm))

        self.pred_mag.add(torch.linalg.norm(p, dim=-1).mean(dim=0), track_sq=True)
        self.gt_mag.add(torch.linalg.norm(g, dim=-1).mean(dim=0), track_sq=True)

    # ---- per-vertex finalizers (each returns [V] tensor on CPU as numpy) ----
    def per_vertex_arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}

        def mean(rs: VertexRunningSum) -> torch.Tensor:
            return _safe_div(rs.sum, rs.count)

        def std(rs: VertexRunningSum) -> torch.Tensor:
            mu = _safe_div(rs.sum, rs.count)
            ex2 = _safe_div(rs.sum2, rs.count)
            var = torch.clamp(ex2 - mu * mu, min=0.0)
            return torch.sqrt(var)

        out["meanRV"] = mean(self.raw_rv).cpu().numpy()
        out["detrRV"] = mean(self.detr_rv).cpu().numpy()

        # ratio-of-means (sum num / sum den, per vertex)
        amp_p_rms = torch.sqrt(_safe_div(self.amp_p_sq.sum, self.amp_p_sq.count))
        amp_g_rms = torch.sqrt(_safe_div(self.amp_g_sq.sum, self.amp_g_sq.count))
        out["ampRatio_rom"] = _safe_div(amp_p_rms, amp_g_rms).cpu().numpy()
        out["peakAmpR_rom"] = _safe_div(self.peak_p.sum, self.peak_g.sum).cpu().numpy()
        rnrmse_rms = torch.sqrt(_safe_div(self.rdiff_sq.sum, self.rdiff_sq.count))
        out["detrNRMSE_rom"] = _safe_div(rnrmse_rms, amp_g_rms).cpu().numpy()
        l2_rms = torch.sqrt(_safe_div(self.l2_sq.sum, self.l2_sq.count))
        out["predGtL2_rom"] = l2_rms.cpu().numpy()
        out["predGtL2N_rom"] = _safe_div(
            torch.sqrt(self.l2_sq.sum), torch.sqrt(self.gt_norm_sq.sum)
        ).cpu().numpy()

        # mean-of-per-subject-ratios
        out["ampRatio_mor"] = mean(self.amp_ratio_mor).cpu().numpy()
        out["peakAmpR_mor"] = mean(self.peak_ratio_mor).cpu().numpy()
        out["detrNRMSE_mor"] = mean(self.nrmse_mor).cpu().numpy()
        out["predGtL2_mor"] = mean(self.l2_mor).cpu().numpy()
        out["predGtL2N_mor"] = mean(self.l2_rel_mor).cpu().numpy()

        # subject-magnitude std ratio
        out["subjStdRatio"] = _safe_div(std(self.pred_mag), std(self.gt_mag)).cpu().numpy()

        out["count"] = self.raw_rv.count.cpu().numpy() if self.raw_rv.count is not None else np.zeros(0)
        return out


# ---------------- cross-subject diversity (subject-level, whole-heart) ---------------

def _sample_diversity(curves: list) -> Tuple[float, float, float, float, float, torch.Tensor, torch.Tensor]:
    if len(curves) < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan, torch.tensor([], device=_DEVICE), torch.tensor([], device=_DEVICE)

    mats = []
    gram = []
    amps = []
    for c in curves:
        x = _center_matrix_t(_to_tensor(c))
        mats.append(x.reshape(-1))
        g = x @ x.T
        gram.append(g.reshape(-1))
        amps.append((x.amax(dim=0) - x.amin(dim=0)).mean())

    gram_mat = torch.stack(gram, dim=0)
    gram_dot = gram_mat @ gram_mat.T
    gram_norm = torch.sqrt(torch.diag(gram_dot))
    rv_den = gram_norm[:, None] * gram_norm[None, :]
    rv = torch.full_like(gram_dot, float("nan"))
    mask = rv_den > 0
    rv[mask] = gram_dot[mask] / rv_den[mask]
    n = len(curves)
    iu_r, iu_c = torch.triu_indices(n, n, offset=1, device=_DEVICE).unbind(0)
    rv_vals = rv[iu_r, iu_c]
    rv_vals = rv_vals[torch.isfinite(rv_vals)]
    mean_rv = float(rv_vals.mean().item()) if rv_vals.numel() else float("nan")
    std_rv = float(rv_vals.std(unbiased=False).item()) if rv_vals.numel() else float("nan")

    mat = torch.stack(mats, dim=0)
    sq = (mat * mat).sum(dim=1)
    dot = mat @ mat.T
    d2 = torch.clamp(sq[:, None] + sq[None, :] - 2.0 * dot, min=0.0) / mat.shape[1]
    dist_vals = torch.sqrt(d2[iu_r, iu_c])
    dist_vals = dist_vals[torch.isfinite(dist_vals)]
    denom = float((torch.sqrt(sq / mat.shape[1]).mean() + EPS).item())
    mean_dist = float((dist_vals.mean().item()) / denom) if dist_vals.numel() else float("nan")
    std_dist = float((dist_vals.std(unbiased=False).item()) / denom) if dist_vals.numel() else float("nan")

    amps_t = torch.stack(amps)
    amp_mean = float(amps_t.mean().item())
    amp_cv = float((amps_t.std(unbiased=False) / (amp_mean + EPS)).item()) if amp_mean > 0 else float("nan")
    return mean_rv, std_rv, amp_cv, mean_dist, std_dist, rv_vals, dist_vals


# ---------------- per-vertex cross-subject diversity ---------------

def _per_vertex_diversity_3d(curves: list) -> Dict[str, np.ndarray]:
    """For a list of S detrended [T, V, 3] tensors, compute per-vertex diversity:
        dCorr  : mean pairwise Pearson across subjects of the centered [T*3] signal
        dAmp   : coefficient of variation across subjects of mean per-axis peak-to-peak
        dDist  : mean pairwise normalized RMSE across subjects
    Each output is a [V] numpy array.
    """
    S = len(curves)
    if S < 2 or curves[0].numel() == 0:
        V = curves[0].shape[1] if S else 0
        nan = np.full((V,), np.nan, dtype=np.float64)
        return {"dCorr": nan, "dAmp": nan.copy(), "dDist": nan.copy()}

    X = torch.stack([c.to(_DEVICE, dtype=_DTYPE) for c in curves], dim=0)  # [S, T, V, 3]
    T, V = X.shape[1], X.shape[2]
    F = T * 3
    Xv = X.permute(2, 0, 1, 3).reshape(V, S, F).contiguous()              # [V, S, F]
    Xc = Xv - Xv.mean(dim=-1, keepdim=True)
    norm = torch.linalg.norm(Xc, dim=-1)                                  # [V, S]
    C = torch.matmul(Xc, Xc.transpose(1, 2))                              # [V, S, S]
    den = norm.unsqueeze(2) * norm.unsqueeze(1) + EPS
    C = C / den
    iu_r, iu_c = torch.triu_indices(S, S, offset=1, device=_DEVICE).unbind(0)
    corr = C[:, iu_r, iu_c].mean(dim=1)                                   # [V]

    pp = (X.amax(dim=1) - X.amin(dim=1)).mean(dim=-1)                     # [S, V]
    pp_mean = pp.mean(dim=0)
    pp_std = pp.std(dim=0, unbiased=False)
    amp_cv = pp_std / (pp_mean.abs() + EPS)                               # [V]

    sq = (Xv * Xv).sum(dim=-1)                                            # [V, S]
    dot = torch.matmul(Xv, Xv.transpose(1, 2))                            # [V, S, S]
    d2 = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * dot).clamp(min=0.0) / F
    dist_pairs = torch.sqrt(d2[:, iu_r, iu_c])                            # [V, npairs]
    denom = torch.sqrt(sq / F).mean(dim=-1) + EPS                         # [V]
    dist = dist_pairs.mean(dim=-1) / denom                                # [V]

    return {
        "dCorr": corr.detach().cpu().numpy().astype(np.float64),
        "dAmp":  amp_cv.detach().cpu().numpy().astype(np.float64),
        "dDist": dist.detach().cpu().numpy().astype(np.float64),
    }


# ---------------- aggregation across vertices for whole / material rows --------------

def _aggregate(per_vertex: Dict[str, np.ndarray], idx: Optional[np.ndarray]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, arr in per_vertex.items():
        if k == "count":
            continue
        sl = arr if idx is None else arr[idx]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            mean = float(sl.mean())
            std = float(sl.std())
            out[k] = f"{mean:.3f} ± {std:.3f}"
        else:
            out[k] = "nan ± nan"
    return out


def compute_metrics(
    results,
    material_groups,
    out_dir: str,
    max_xsubj_subjects: int = 128,
    gt_cache: Optional[dict] = None,
    completed_experiments: Optional[set] = None,
    metrics_path: Optional[str] = None,
):
    rows = []
    gt_cache = {} if gt_cache is None else gt_cache
    completed_experiments = set() if completed_experiments is None else completed_experiments

    for name, subs in sorted(results.items()):
        if name in completed_experiments:
            tqdm.write(f"  {name}: already present in output, skipping")
            continue
        npz_path = os.path.join(out_dir, f"{name}_per_vertex.npz")
        acc = VertexAccumulator()
        sample_pred: list = []
        sample_gt: list = []

        subjects = sorted(subs.items())
        sample_sids = {sid for sid, _ in subjects[:max_xsubj_subjects]}

        for sid, paths in tqdm(subjects, desc=name, mininterval=5.0):
            pred = _to_tensor(np.load(paths["pred"]))
            gt, rgt = _load_gt_cached(gt_cache, sid, paths["gt"])
            if tuple(pred.shape) != tuple(gt.shape):
                raise ValueError(
                    f"shape mismatch for {paths['pred']}: pred {tuple(pred.shape)}, gt {tuple(gt.shape)}"
                )
            if pred.ndim != 3 or pred.shape[-1] != 3:
                raise ValueError(f"expected [T, V, 3] arrays, got {tuple(pred.shape)}")

            rpred = _detrend(pred)
            if not os.path.exists(npz_path):
                acc.add_subject(pred, gt, rpred, rgt)

            if sid in sample_sids:
                # keep [T, V, 3] so we can slice by material before flattening
                sample_pred.append(_to_tensor(rpred))
                sample_gt.append(_to_tensor(rgt))

        if not os.path.exists(npz_path):
            per_vertex = acc.per_vertex_arrays()

            # per-vertex inter-subject diversity (computed from detrended sample curves)
            div_pred = _per_vertex_diversity_3d(sample_pred)
            div_gt = _per_vertex_diversity_3d(sample_gt)
            per_vertex["dPredCorr"] = div_pred["dCorr"]
            per_vertex["dGTCorr"]   = div_gt["dCorr"]
            per_vertex["dPredAmp"]  = div_pred["dAmp"]
            per_vertex["dGTAmp"]    = div_gt["dAmp"]
            per_vertex["dPredDist"] = div_pred["dDist"]
            per_vertex["dGTDist"]   = div_gt["dDist"]

            # save per-vertex arrays for plotting
            np.savez_compressed(npz_path, **per_vertex)
        else:
            per_vertex = np.load(npz_path)
            div_pred = _per_vertex_diversity_3d(sample_pred)
            div_gt = _per_vertex_diversity_3d(sample_gt)
            per_vertex = dict(per_vertex)
            per_vertex["dPredCorr"] = div_pred["dCorr"]
            per_vertex["dGTCorr"]   = div_gt["dCorr"]
            per_vertex["dPredAmp"]  = div_pred["dAmp"]
            per_vertex["dGTAmp"]    = div_gt["dAmp"]
            per_vertex["dPredDist"] = div_pred["dDist"]
            per_vertex["dGTDist"]   = div_gt["dDist"]

        def _xsubj_for(idx: Optional[np.ndarray]) -> Dict[str, str]:
            if idx is None:
                pred_curves = [c.reshape(c.shape[0], -1) for c in sample_pred]
                gt_curves = [c.reshape(c.shape[0], -1) for c in sample_gt]
            else:
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=_DEVICE)
                pred_curves = [c.index_select(1, idx_t).reshape(c.shape[0], -1) for c in sample_pred]
                gt_curves = [c.index_select(1, idx_t).reshape(c.shape[0], -1) for c in sample_gt]
            xp, xp_std, ap, dp, dp_std, xp_vals, dp_vals = _sample_diversity(pred_curves)
            xg, xg_std, ag, dg, dg_std, xg_vals, dg_vals = _sample_diversity(gt_curves)
            gap_vals = xp_vals - xg_vals if xp_vals.numel() and xg_vals.numel() else torch.tensor([], device=_DEVICE)
            ratio_vals = dp_vals / (dg_vals + EPS) if dp_vals.numel() and dg_vals.numel() else torch.tensor([], device=_DEVICE)
            gap_mean = float(gap_vals.mean().item()) if gap_vals.numel() else float("nan")
            gap_std = float(gap_vals.std(unbiased=False).item()) if gap_vals.numel() else float("nan")
            ratio_mean = float(ratio_vals.mean().item()) if ratio_vals.numel() else float("nan")
            ratio_std = float(ratio_vals.std(unbiased=False).item()) if ratio_vals.numel() else float("nan")
            return {
                "xsubjPredRV": f"{xp:.3f} ± {xp_std:.3f}",
                "xsubjGTRV": f"{xg:.3f} ± {xg_std:.3f}",
                "xsubjGapRV": f"{gap_mean:.3f} ± {gap_std:.3f}",
                "xsAmpCVPred": f"{ap:.3f}",
                "xsAmpCVGT": f"{ag:.3f}",
                "xsDistPred": f"{dp:.3f} ± {dp_std:.3f}",
                "xsDistGT": f"{dg:.3f} ± {dg_std:.3f}",
                "xsDistRatio": f"{ratio_mean:.3f} ± {ratio_std:.3f}",
            }

        # build rows: whole + each material
        scales: List[Tuple[str, Optional[np.ndarray]]] = [("whole", None)] + list(material_groups)
        exp_rows = []
        for scale, idx in scales:
            agg = _aggregate(per_vertex, idx)
            row = {"experiment": name, "scale": scale, **agg, **_xsubj_for(idx)}
            exp_rows.append(row)

        rows.extend(exp_rows)
        if metrics_path is not None:
            append_metrics_csv(exp_rows, metrics_path)
        tqdm.write(f"  {name}: wrote {npz_path}")
    return rows


def print_metrics(rows):
    if not rows:
        return
    cols = ("experiment", "scale", "meanRV", "ampRatio_rom", "ampRatio_mor",
            "detrNRMSE_rom", "predGtL2_rom", "predGtL2N_rom",
            "xsubjPredRV", "xsubjGTRV")
    widths = (50, 25, 16, 16, 16, 16, 16, 16, 16, 16)
    print(" ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    for r in rows:
        short = r["experiment"].replace("_8L_256d_49frames", "")
        vals = [short, r["scale"]] + [
            str(r.get(k, "nan")) for k in cols[2:]
        ]
        print(" ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))


def write_metrics_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def append_metrics_csv(rows, path):
    if not rows:
        return
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def completed_experiments_from_csv(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row["experiment"] for row in csv.DictReader(f) if row.get("experiment")}


def preload_gt_cache(results, gt_cache: dict, completed_experiments: Optional[set] = None):
    completed_experiments = set() if completed_experiments is None else completed_experiments
    sid_to_gt = {}
    for exp, subs in results.items():
        if exp in completed_experiments:
            continue
        for sid, paths in subs.items():
            sid_to_gt.setdefault(sid, paths["gt"])

    for sid, gt_path in tqdm(sorted(sid_to_gt.items()), desc="Preload GT", mininterval=5.0):
        _load_gt_cached(gt_cache, sid, gt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default='../nips26-physctrl-outputs',
                    help="Directory containing per-experiment folders with {dirname}/*_{pred,gt}.npy")
    ap.add_argument("--mesh-info", default='../ukb_heart_mesh_visual',
                    help="Directory containing control_material_idmap.json and control_materials.npy")
    ap.add_argument("--out", default="./outputs", help="Output directory for metrics")
    ap.add_argument("--tag", type=str, default="vertex_multiscale")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="Optional debugging limit per experiment")
    ap.add_argument("--subject-list", type=str, default=None,
                    help="Optional path to a txt file (one subid per line) to restrict evaluation "
                         "to a subgroup (E1/E3). Set --max-xsubj-subjects >= subgroup size for "
                         "matched-n pairwise diversity.")
    ap.add_argument("--max-xsubj-subjects", type=int, default=128,
                    help="Subject sample size for cross-subject diversity metrics")
    ap.add_argument("--resume", action="store_true",
                    help="Skip experiments already present in metrics_per_scale.csv")
    ap.add_argument("--preload-gt", action=argparse.BooleanOptionalAction, default=True,
                    help="Load and detrend each unique GT subject once before evaluating experiments")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics-avgstd_per_scale.csv")
    if not args.resume and os.path.exists(metrics_path):
        os.remove(metrics_path)

    _, _, material_groups = load_material_info(args.mesh_info)
    subject_list = None
    if args.subject_list is not None:
        with open(args.subject_list) as f:
            subject_list = {ln.strip() for ln in f if ln.strip()}
        print(f"[subject-list] restricting to {len(subject_list)} subjects from {args.subject_list}")
    results = collect_pairs(args.root, args.max_subjects, subject_list)
    gt_cache = {}
    completed = completed_experiments_from_csv(metrics_path) if args.resume else set()
    if args.preload_gt:
        preload_gt_cache(results, gt_cache, completed)
    rows = compute_metrics(
        results,
        material_groups,
        out_dir=out_dir,
        max_xsubj_subjects=args.max_xsubj_subjects,
        gt_cache=gt_cache,
        completed_experiments=completed,
        metrics_path=metrics_path,
    )
    print_metrics(rows)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
