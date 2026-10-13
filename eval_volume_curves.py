"""Evaluate ECG -> 4D-heart-mesh predictions by comparing convex-hull volume
curves of Pred vs GT for every subject under each output directory.

Reads `<root>/<exp>/{dirname}/<sid>_pred.npy` and `..._gt.npy` (each of shape
[T, V, 3]), computes per-frame convex-hull volume, then reports four metrics
per experiment averaged over subjects:

  meanCorr      : Pearson(Pred, GT) of raw volume curves         (global pattern)
  subjStdRatio  : std(mean Pred per subject) / std(mean GT)      (per-subject discrimination, 1.0=ideal)
  detrCorr      : Pearson(Pred-trend, GT-trend) on cubic-detrended residual (local-peak SHAPE)
  ampRatio      : std(detrended Pred) / std(detrended GT)        (local-peak MAGNITUDE, 1.0=ideal)
  peakAmpR      : ptp(detrended Pred) / ptp(detrended GT)        (peak-to-peak magnitude)
  xsubjPred     : mean pairwise Pearson of detrended curves across subjects (Pred)
  xsubjGT      : mean pairwise Pearson of detrended curves across subjects (GT)
                 -> if xsubjPred >> xsubjGT, the model collapses to a subject-invariant
                    local-peak SHAPE.
  xsAmpCVPred   : coefficient of variation (std/mean) of detrended peak-to-peak across subjects (Pred)
  xsAmpCVGT     : same for GT.  If xsAmpCVPred << xsAmpCVGT, the model gives every subject the
                  same local-peak AMPLITUDE.
  xsDistPred    : mean pairwise RMSE between detrended curves of different subjects, normalized
                  by GT's mean detrended std.
  xsDistGT      : same for GT.  Captures shape + amplitude diversity together; xsDistPred << xsDistGT
                  means predictions for different subjects are nearly identical.
  predGtL2      : per-subject L2 distance ||Pred - GT||_2 averaged across subjects (raw units).
  predGtL2N     : same L2 normalized by ||GT||_2 averaged across subjects (unitless).

Also writes:
  - <out_dir>/all_volumes.png  : Pred and GT panels side-by-side per experiment
  - <out_dir>/overlay.png      : Pred (dashed) vs GT (solid) overlaid per subject
  - <out_dir>/metrics.csv
  - <out_dir>/volumes.json     : raw curves (re-usable cache)

Usage
-----
  python eval_volume_curves.py --root checkpoints
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

dirname = 'npy_eval'
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _hull_volumes(verts_t: np.ndarray) -> List[float]:
    vols: List[float] = []
    for f in range(verts_t.shape[0]):
        try:
            vols.append(float(ConvexHull(verts_t[f]).volume))
        except Exception:
            vols.append(float("nan"))
    return vols


def collect_volumes(root: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Walk `<root>/<exp>/{dirname}/*_pred.npy` and compute volume curves."""
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    exp_dirs = sorted(glob.glob(os.path.join(root, "*", f"{dirname}/")))
    for d in tqdm(exp_dirs, desc="Experiments"):
        exp = d.split("/")[-3]
        subjects: Dict[str, Dict[str, List[float]]] = {}
        subj_files = sorted(glob.glob(os.path.join(d, "*_pred.npy")))
        for p in tqdm(subj_files, desc=f"  {exp}", leave=False):
            sid = os.path.basename(p).replace("_pred.npy", "")
            g_path = p.replace("_pred.npy", "_gt.npy")
            if not os.path.exists(g_path):
                continue
            pred = np.load(p)
            gt = np.load(g_path)
            subjects[sid] = {
                "pred": _hull_volumes(pred),
                "gt": _hull_volumes(gt),
            }
        if subjects:
            results[exp] = subjects
            tqdm.write(f"  {exp}: {len(subjects)} subjects")
    return results


def _mean_pairwise_pearson(curves: np.ndarray) -> float:
    """Mean Pearson over all unique subject pairs of a (S, T) matrix."""
    if curves.shape[0] < 2:
        return float("nan")
    # corrcoef returns SxS; take strictly upper triangle
    C = np.corrcoef(curves)
    iu = np.triu_indices_from(C, k=1)
    vals = C[iu]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _mean_pairwise_rmse(curves: np.ndarray, denom: float) -> float:
    """Mean pairwise RMSE between rows of a (S, T) matrix, normalized by `denom`."""
    s = curves.shape[0]
    if s < 2 or denom <= 0:
        return float("nan")
    diffs = curves[:, None, :] - curves[None, :, :]   # (S, S, T)
    rmse = np.sqrt((diffs ** 2).mean(axis=-1))        # (S, S)
    iu = np.triu_indices(s, k=1)
    return float(rmse[iu].mean() / denom)


def _amp_cv(curves: np.ndarray) -> float:
    """Coefficient of variation of per-subject peak-to-peak amplitude."""
    if curves.shape[0] < 2:
        return float("nan")
    ptp = curves.max(axis=1) - curves.min(axis=1)
    m = ptp.mean()
    return float(ptp.std() / (m + 1e-9)) if m > 0 else float("nan")


def compute_metrics(results: Dict[str, Dict[str, Dict[str, List[float]]]]):
    rows = []
    for name, subs in sorted(results.items()):
        corrs, dcs, amp_ratios, peak_ratios, nrmses = [], [], [], [], []
        pred_means, gt_means = [], []
        det_pred_curves, det_gt_curves = [], []
        l2_abs, l2_rel = [], []
        for _sid, d in subs.items():
            p = np.array(d["pred"], dtype=float)
            g = np.array(d["gt"], dtype=float)
            corrs.append(_pearson(p, g))
            diff = p - g
            l2 = float(np.linalg.norm(diff))
            l2_abs.append(l2)
            l2_rel.append(l2 / (np.linalg.norm(g) + 1e-9))
            x = np.arange(len(p))
            cp = np.polyfit(x, p, 3)
            cg = np.polyfit(x, g, 3)
            rp = p - np.polyval(cp, x)
            rg = g - np.polyval(cg, x)
            dcs.append(_pearson(rp, rg))
            amp_ratios.append(np.std(rp) / (np.std(rg) + 1e-9))
            peak_ratios.append((rp.max() - rp.min()) / (rg.max() - rg.min() + 1e-9))
            nrmses.append(np.sqrt(np.mean((rp - rg) ** 2)) / (np.std(rg) + 1e-9))
            pred_means.append(p.mean())
            gt_means.append(g.mean())
            det_pred_curves.append(rp)
            det_gt_curves.append(rg)
        det_pred_mat = np.stack(det_pred_curves, axis=0)
        det_gt_mat = np.stack(det_gt_curves, axis=0)
        xsubj_pred = _mean_pairwise_pearson(det_pred_mat)
        xsubj_gt = _mean_pairwise_pearson(det_gt_mat)
        # Amplitude diversity across subjects
        xs_amp_cv_pred = _amp_cv(det_pred_mat)
        xs_amp_cv_gt = _amp_cv(det_gt_mat)
        # Combined shape+amplitude diversity across subjects: pairwise RMSE
        # normalized by GT mean detrended std so Pred and GT live on the same scale.
        denom = float(np.mean([np.std(c) for c in det_gt_curves]) + 1e-9)
        xs_dist_pred = _mean_pairwise_rmse(det_pred_mat, denom)
        xs_dist_gt = _mean_pairwise_rmse(det_gt_mat, denom)
        rows.append({
            "experiment": name,
            "meanCorr": float(np.nanmean(corrs)),
            "subjStdRatio": float(np.std(pred_means) / (np.std(gt_means) + 1e-9)),
            "detrCorr": float(np.nanmean(dcs)),
            "ampRatio": float(np.nanmean(amp_ratios)),
            "peakAmpR": float(np.nanmean(peak_ratios)),
            "detrNRMSE": float(np.nanmean(nrmses)),
            "xsubjPred": xsubj_pred,
            "xsubjGT": xsubj_gt,
            "xsubjGap": xsubj_pred - xsubj_gt,
            "xsAmpCVPred": xs_amp_cv_pred,
            "xsAmpCVGT": xs_amp_cv_gt,
            "xsDistPred": xs_dist_pred,
            "xsDistGT": xs_dist_gt,
            "xsDistRatio": xs_dist_pred / (xs_dist_gt + 1e-9),
            "predGtL2": float(np.mean(l2_abs)),
            "predGtL2N": float(np.mean(l2_rel)),
        })
    return rows


def print_metrics(rows):
    hdr = ("experiment", "meanCorr", "ampRatio",
           "xsCorrP", "xsCorrG", "xsAmpCVP", "xsAmpCVG",
           "xsDistP", "xsDistG", "xsDistR", "L2", "L2N")
    print(f"{hdr[0]:50} {hdr[1]:>9} {hdr[2]:>9} "
          f"{hdr[3]:>8} {hdr[4]:>8} {hdr[5]:>9} {hdr[6]:>9} "
          f"{hdr[7]:>8} {hdr[8]:>8} {hdr[9]:>8} {hdr[10]:>10} {hdr[11]:>7}")
    for r in rows:
        short = r["experiment"].replace("_8L_256d_49frames", "")
        print(f"{short:50} {r['meanCorr']:9.3f} {r['ampRatio']:9.3f} "
              f"{r['xsubjPred']:8.3f} {r['xsubjGT']:8.3f} "
              f"{r['xsAmpCVPred']:9.3f} {r['xsAmpCVGT']:9.3f} "
              f"{r['xsDistPred']:8.3f} {r['xsDistGT']:8.3f} {r['xsDistRatio']:8.3f} "
              f"{r['predGtL2']:10.1f} {r['predGtL2N']:7.3f}")


def write_metrics_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_pred_gt_panels(results, out_path, ylim=(200_000, 700_000)):
    names = sorted(results)
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 4, nrows * 4.2), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        r, c = (i // ncols) * 2, i % ncols
        ax_p, ax_g = axes[r, c], axes[r + 1, c]
        for j, (sid, dat) in enumerate(sorted(results[name].items())):
            col = cmap(j % 10)
            ax_p.plot(dat["pred"], color=col, linewidth=0.9, label=sid)
            ax_g.plot(dat["gt"], color=col, linewidth=0.9, label=sid)
        short = name.replace("_8L_256d_49frames", "").replace("ecg_heart_", "") or "baseline"
        ax_p.set_title(f"{short}  Pred", fontsize=9)
        ax_g.set_title(f"{short}  GT", fontsize=9)
        ax_p.tick_params(labelsize=7); ax_g.tick_params(labelsize=7)
        ax_p.grid(True, alpha=0.3); ax_g.grid(True, alpha=0.3)
        ax_p.set_ylim(*ylim); ax_g.set_ylim(*ylim)
    for k in range(n, nrows * ncols):
        r, c = (k // ncols) * 2, k % ncols
        axes[r, c].axis("off"); axes[r + 1, c].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_overlay(results, out_path, ylim=(200_000, 700_000)):
    names = sorted(results)
    n = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        for j, (sid, dat) in enumerate(sorted(results[name].items())):
            col = cmap(j % 10)
            ax.plot(dat["gt"], color=col, linewidth=1.0, linestyle="-", alpha=0.9)
            ax.plot(dat["pred"], color=col, linewidth=1.0, linestyle="--", alpha=0.9)
        short = name.replace("_8L_256d_49frames", "").replace("ecg_heart_", "") or "baseline"
        ax.set_title(short, fontsize=10)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(*ylim)
    axes[0, 0].plot([], [], "k-", label="GT")
    axes[0, 0].plot([], [], "k--", label="Pred")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Directory containing per-experiment folders with {dirname}/*_{pred,gt}.npy")
    ap.add_argument("--out", default='./outputs', help="Output directory for plots/metrics")
    ap.add_argument("--ymin", type=float, default=200_000.0)
    ap.add_argument("--ymax", type=float, default=700_000.0)
    ap.add_argument("--plot", type=int, default=0)
    ap.add_argument("--tag", type=str, default='wholeheart')
    args = ap.parse_args()
    args.out = os.path.join(args.out, args.tag)
    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "volumes.json")

    if os.path.exists(cache_path):
        print(f"loading cached volumes from {cache_path}")
        with open(cache_path) as f:
            results = json.load(f)
    else:
        results = collect_volumes(args.root)
        with open(cache_path, "w") as f:
            json.dump(results, f)
        print(f"wrote {cache_path}")

    rows = compute_metrics(results)
    print_metrics(rows)
    write_metrics_csv(rows, os.path.join(args.out, "metrics.csv"))
    if args.plot == 1:
        ylim = (args.ymin, args.ymax)
        plot_pred_gt_panels(results, os.path.join(args.out, "all_volumes.png"), ylim=ylim)
        plot_overlay(results, os.path.join(args.out, "overlay.png"), ylim=ylim)
        print(f"wrote plots and metrics under {args.out}")


if __name__ == "__main__":
    main()
