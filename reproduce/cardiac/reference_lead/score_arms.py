#!/usr/bin/env python
"""Paper metrics (RV, RMSE, PAR, |APCR-1|, |APDR-1|) for several npy_eval dirs (Table 9).

Metric primitives come from reproduce/metrics/cardiac_stats.py, which mirrors the whole-heart row
of eval_vertex_multiscale.py. Convention of Table 1, matched exactly:
  RV / RMSE / PAR   : over ALL subjects common to every dir
  APCR / APDR       : over the FIRST 128 sorted subjects (--nxsubj)

The bootstrap is paired: every arm is evaluated on the same resampled subject indices in each
draw, so the arm-minus-reference difference (--ref) gets a paired CI.

  python score_arms.py --dir aVR=<...>/reference_lead_aVR/npy_eval \
                       --dir aVL=<...>/reference_lead_aVL/npy_eval --out scores.json
"""
import os, sys, json, glob, time, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metrics"))
from cardiac_stats import (rv_per_vertex, pairwise_rv_mean, pairwise_dist_mean,  # noqa: E402
                           make_resid_op)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = ["RV", "RMSE", "PAR", "APCR", "APDR"]


def load_arm(vdir, subids, resid, T, V, C, gt_cache):
    """Per-subject sufficient statistics for one npy_eval dir."""
    N = len(subids)
    st = {k: torch.empty(N, V, dtype=torch.float64, device=dev)
          for k in ("rv", "l2sq", "peakp")}
    st["Gp"] = torch.empty(N, T * T, dtype=torch.float64, device=dev)
    st["Mp"] = torch.empty(N, T * V * C, dtype=torch.float64, device=dev)
    st["sqp"] = torch.empty(N, dtype=torch.float64, device=dev)
    t0 = time.time()
    for i, sid in enumerate(subids):
        p = torch.as_tensor(np.load(os.path.join(vdir, f"{sid}_pred.npy")),
                            dtype=torch.float64, device=dev)
        gt = gt_cache[i]
        rp = (resid @ p.reshape(T, -1)).reshape(T, V, C)
        st["rv"][i] = rv_per_vertex(p, gt)
        st["l2sq"][i] = ((p - gt) ** 2).sum((0, 2))
        st["peakp"][i] = (rp.amax(0) - rp.amin(0)).mean(-1)
        xp = rp.reshape(T, -1); xp = xp - xp.mean(0, keepdim=True)
        st["Gp"][i] = (xp @ xp.T).reshape(-1)
        Mp = xp.reshape(-1); st["Mp"][i] = Mp; st["sqp"][i] = (Mp * Mp).sum()
        if i % 2000 == 0:
            print(f"    {i}/{N}  {time.time()-t0:.0f}s", flush=True)
    return st


def metrics(st, gt, idx_all, idx_x):
    """RV/RMSE/PAR on idx_all; APCR/APDR on idx_x (Table 1 convention)."""
    RV = torch.nanmean(torch.nanmean(st["rv"][idx_all], dim=0))
    RMSEv = torch.sqrt(st["l2sq"][idx_all].sum(0)) / torch.sqrt(gt["gtsq"][idx_all].sum(0))
    RMSE = RMSEv[torch.isfinite(RMSEv)].mean()
    pn = st["peakp"][idx_all].sum(0); pd = gt["peakg"][idx_all].sum(0)
    PARv = torch.where(pd > 0, pn / pd, torch.full_like(pn, float("nan")))
    PAR = PARv[torch.isfinite(PARv)].mean()
    APCR = pairwise_rv_mean(st["Gp"][idx_x]) / pairwise_rv_mean(gt["Gg"][idx_x])
    APDR = (pairwise_dist_mean(st["Mp"][idx_x], st["sqp"][idx_x])
            / pairwise_dist_mean(gt["Mg"][idx_x], gt["sqg"][idx_x]))
    return torch.stack([RV, RMSE, PAR, APCR, APDR]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", action="append", required=True,
                    help="label=/path/to/npy_eval (repeat; the first one also supplies the GT)")
    ap.add_argument("--ref", default=None, help="label used as the paired baseline")
    ap.add_argument("--nxsubj", type=int, default=128)
    ap.add_argument("--ndraws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = dict(d.split("=", 1) for d in a.dir)
    sets = [{os.path.basename(f)[:-9] for f in glob.glob(os.path.join(v, "*_pred.npy"))}
            for v in arms.values()]
    subids = sorted(set.intersection(*sets))
    print(f"arms={list(arms)}  common subjects={len(subids)}")
    for lab, v in arms.items():
        print(f"  {lab}: {len(glob.glob(os.path.join(v,'*_pred.npy')))} preds  {v}")

    sample = np.load(os.path.join(list(arms.values())[0], f"{subids[0]}_gt.npy"))
    T, V, C = sample.shape
    N = len(subids)
    resid = make_resid_op(T)

    # GT is byte-shared across arms (same subjects, same loader) -> read once.
    print("[GT] loading")
    gt_cache = torch.empty(N, T, V, C, dtype=torch.float64, device=dev)
    gt = {"gtsq": torch.empty(N, V, dtype=torch.float64, device=dev),
          "peakg": torch.empty(N, V, dtype=torch.float64, device=dev),
          "Gg": torch.empty(N, T * T, dtype=torch.float64, device=dev),
          "Mg": torch.empty(N, T * V * C, dtype=torch.float64, device=dev),
          "sqg": torch.empty(N, dtype=torch.float64, device=dev)}
    gdir = list(arms.values())[0]
    for i, sid in enumerate(subids):
        g = torch.as_tensor(np.load(os.path.join(gdir, f"{sid}_gt.npy")),
                            dtype=torch.float64, device=dev)
        rg = (resid @ g.reshape(T, -1)).reshape(T, V, C)
        xg = rg.reshape(T, -1); xg = xg - xg.mean(0, keepdim=True)
        gt_cache[i] = g
        gt["gtsq"][i] = (g * g).sum((0, 2))
        gt["peakg"][i] = (rg.amax(0) - rg.amin(0)).mean(-1)
        gt["Gg"][i] = (xg @ xg.T).reshape(-1)
        Mg = xg.reshape(-1); gt["Mg"][i] = Mg; gt["sqg"][i] = (Mg * Mg).sum()

    # GT identity check: every arm must have byte-identical GT for a spot sample.
    for lab, v in arms.items():
        for sid in subids[:3]:
            assert np.array_equal(np.load(os.path.join(v, f"{sid}_gt.npy")),
                                  np.load(os.path.join(gdir, f"{sid}_gt.npy"))), \
                f"GT mismatch {lab} {sid}"
    print("[GT] identical across arms (spot-checked)")

    stores = {}
    for lab, v in arms.items():
        print(f"[{lab}] loading preds")
        stores[lab] = load_arm(v, subids, resid, T, V, C, gt_cache)
    del gt_cache
    torch.cuda.empty_cache()

    idx_all = torch.arange(N, device=dev)
    idx_x = torch.arange(min(a.nxsubj, N), device=dev)
    point = {lab: metrics(stores[lab], gt, idx_all, idx_x).tolist() for lab in arms}
    for lab, m in point.items():
        print(f"POINT {lab:12s} RV={m[0]:.4f} RMSE={m[1]:.4f} PAR={m[2]:.4f} "
              f"|APCR-1|={abs(m[3]-1):.4f} |APDR-1|={abs(m[4]-1):.4f}")

    # ---- paired bootstrap: same draw indices for every arm ----
    g_ = torch.Generator(device=dev); g_.manual_seed(a.seed)
    draws = {lab: [] for lab in arms}
    for d in range(a.ndraws):
        sel_all = torch.randint(N, (N,), generator=g_, device=dev)
        sel_x = torch.randint(len(idx_x), (len(idx_x),), generator=g_, device=dev)
        for lab in arms:
            draws[lab].append(metrics(stores[lab], gt, sel_all, sel_x))
        if d % 100 == 0:
            print(f"  boot {d}/{a.ndraws}", flush=True)
    draws = {lab: np.stack(v) for lab, v in draws.items()}   # [ndraws, 5]

    def absm1(arr):
        out = arr.copy(); out[:, 3] = np.abs(out[:, 3] - 1); out[:, 4] = np.abs(out[:, 4] - 1)
        return out

    res = {"meta": {"n": N, "nxsubj": int(len(idx_x)), "ndraws": a.ndraws,
                    "seed": a.seed, "dirs": arms, "ref": a.ref,
                    "metric_order": NAMES,
                    "note": "RV/RMSE/PAR over all n; APCR/APDR over first-nxsubj "
                            "sorted subjects (Table 1 convention). Bootstrap paired."},
           "point": {}, "ci": {}, "paired_delta": {}}
    for lab in arms:
        pt = absm1(np.array(point[lab])[None])[0]
        b = absm1(draws[lab])
        res["point"][lab] = dict(zip(NAMES, pt.tolist()))
        res["ci"][lab] = {NAMES[i]: [float(np.percentile(b[:, i], 2.5)),
                                     float(np.percentile(b[:, i], 97.5))]
                          for i in range(5)}
    if a.ref and a.ref in arms:
        bref = absm1(draws[a.ref])
        for lab in arms:
            if lab == a.ref:
                continue
            d = absm1(draws[lab]) - bref            # paired difference per draw
            res["paired_delta"][f"{lab}_minus_{a.ref}"] = {
                NAMES[i]: {"mean": float(d[:, i].mean()),
                           "lo": float(np.percentile(d[:, i], 2.5)),
                           "hi": float(np.percentile(d[:, i], 97.5)),
                           "excludes_zero": bool(np.percentile(d[:, i], 2.5) > 0
                                                 or np.percentile(d[:, i], 97.5) < 0)}
                for i in range(5)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")

    for k, v in res["paired_delta"].items():
        print(f"\nPAIRED {k}")
        for m in NAMES:
            s = v[m]
            flag = "  <-- CI excludes 0" if s["excludes_zero"] else ""
            print(f"  {m:5s} {s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]{flag}")


if __name__ == "__main__":
    main()
