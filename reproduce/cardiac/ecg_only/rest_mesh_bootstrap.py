"""Whole-heart metrics with the population rest mesh (Table 3, column p0-hat), with bootstrap CIs.

The paper's metrics of eval_vertex_multiscale.py (RV = meanRV, PAR = peakAmpR_rom, APCR, APDR) are
computed per condition from per-subject sufficient statistics. RMSE is reported twice:
  RMSE_abs    : ||X_hat - X_gt|| / ||X_gt||                           (predGtL2N_rom)
  RMSE_offrem : ||(X_hat - p0_used) - (X_gt - p0_subj)|| / ||X_gt||   (offset-removed)
where p0_used is the rest mesh supplied at inference (the template for cohort_mean*, the subject's
own rest mesh otherwise) and p0_subj is the subject's own rest mesh. With the own rest mesh the two
coincide. RV, PAR, APCR and APDR are computed on time-centred or cubic-detrended curves, so a
time-constant p0 does not change them.

Bootstrap (B draws, numpy RandomState(--seed)): RV, RMSE and PAR resample all subjects with
replacement, APCR and APDR redraw a sample of --n-xsubj subjects without replacement. Every
condition uses the same draws. The table reports the median of each metric over the draws.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import eval_vertex_multiscale as evm  # noqa: E402


def mesh_dirs_by_subject(mesh_dir):
    """{subid: [entries named {subid}_*]}, from one listing of mesh_dir.
    One listdir instead of a glob per subject, which rescans a ~48k-entry directory."""
    by_sid = {}
    for name in os.listdir(mesh_dir):
        if not name.startswith(".") and "_" in name:
            by_sid.setdefault(name.split("_", 1)[0], []).append(name)
    return by_sid


def load_p0(mesh_dir, by_sid, sid):
    """Subject rest mesh as the loader hands it to the model: frame 0 of frames.npy, centred on
    its own centroid (ukbiobank_dataset.py, scale 1.0, no smoothing, frame 0 is never repaired)."""
    dirs = by_sid.get(sid, [])
    assert len(dirs) == 1, f"{sid}: {len(dirs)} mesh dirs"
    arr = np.load(os.path.join(mesh_dir, dirs[0], "frames.npy"), mmap_mode="r")
    f0 = np.asarray(arr[0], dtype=np.float32)
    return f0 - f0.mean(axis=0, keepdims=True)


def per_subject_stats(pred_dir, sids, gt_cache, p0_used, p0_subj):
    """Per-subject/per-vertex stats, flattened detrended curves, and the offset-removed L2."""
    rv, l2sq, gtsq, pk_p, pk_g, flat, l2sq_d = [], [], [], [], [], [], []
    for i, sid in enumerate(sids):
        p = evm._to_tensor(np.load(os.path.join(pred_dir, f"{sid}_pred.npy")))
        g, rg = gt_cache[sid]
        rp = evm._detrend(p)
        rv.append(evm._rv_per_vertex(p, g))
        l2sq.append(((p - g) ** 2).sum(dim=(0, 2)))
        gtsq.append((g * g).sum(dim=(0, 2)))
        pk_p.append((rp.amax(dim=0) - rp.amin(dim=0)).mean(dim=-1))
        pk_g.append((rg.amax(dim=0) - rg.amin(dim=0)).mean(dim=-1))
        flat.append(evm._center_matrix_t(rp).reshape(-1).float())
        pu = evm._to_tensor(p0_used[sid]).unsqueeze(0)     # (1, V, 3)
        ps = evm._to_tensor(p0_subj[sid]).unsqueeze(0)
        l2sq_d.append((((p - pu) - (g - ps)) ** 2).sum(dim=(0, 2)))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sids)}", flush=True)
    return {"rv": torch.stack(rv), "l2sq": torch.stack(l2sq), "gtsq": torch.stack(gtsq),
            "l2sq_d": torch.stack(l2sq_d), "pk_p": torch.stack(pk_p), "pk_g": torch.stack(pk_g),
            "flat": torch.stack(flat)}


def point_metrics(st, idx):
    """RV / RMSE / PAR over a (bootstrap) subject index set, matching evm."""
    rv = st["rv"][idx]
    pcc = float(np.nanmean(rv.mean(0).cpu().numpy()))
    l2 = torch.sqrt(st["l2sq"][idx].sum(0)) / torch.sqrt(st["gtsq"][idx].sum(0))
    par = st["pk_p"][idx].sum(0) / st["pk_g"][idx].sum(0)
    return pcc, float(np.nanmean(l2.cpu().numpy())), float(np.nanmean(par.cpu().numpy()))


def rmse_variants(st, idx):
    """RMSE_abs and RMSE_offrem, each the mean over vertices of a ratio of sums over subjects."""
    def ratio(num_sq, den_sq):
        r = torch.sqrt(num_sq[idx].sum(0)) / torch.sqrt(den_sq[idx].sum(0))
        return float(np.nanmean(r.cpu().numpy()))
    return ratio(st["l2sq"], st["gtsq"]), ratio(st["l2sq_d"], st["gtsq"])


def diversity_grams(flat, T):
    """gram_dot[i,j] = <G_i, G_j> with G_i = x_i x_i^T (RV numerator), dot[i,j] = <x_i, x_j>."""
    n, D = flat.shape
    x = flat.view(n, T, D // T)
    G = torch.bmm(x, x.transpose(1, 2)).reshape(n, -1)   # (n, T*T)
    return G @ G.T, flat @ flat.T, D


def diversity_for(idx, gram_dot, dot, D):
    """Mean pairwise RV and normalised mean pairwise distance of eval_vertex_multiscale."""
    gd = gram_dot[np.ix_(idx, idx)]
    gn = torch.sqrt(torch.diag(gd))
    rv = gd / (gn[:, None] * gn[None, :])
    m = torch.triu(torch.ones_like(rv, dtype=torch.bool), diagonal=1)
    mean_rv = float(rv[m].mean())

    dd = dot[np.ix_(idx, idx)]
    sq = torch.diag(dd)
    d2 = torch.clamp(sq[:, None] + sq[None, :] - 2 * dd, min=0.0) / D
    denom = float(torch.sqrt(sq / D).mean()) + evm.EPS
    mean_dist = float(torch.sqrt(d2[m]).mean()) / denom
    return mean_rv, mean_dist


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="directory with <condition>/npy_eval/{subid}_{pred,gt}.npy")
    ap.add_argument("--subject-list", required=True, help="val_subsample_2000.txt of build_template.py")
    ap.add_argument("--template", required=True, help="cohort_mean_template.npy of build_template.py")
    ap.add_argument("--conditions", nargs="+", default=["cohort_mean"],
                    help="condition folders under --root; names starting with cohort_mean used the template")
    ap.add_argument("--mesh-dir", default=None,
                    help="directory with {subid}_*/frames.npy (default: dataset.npy_dir of --config)")
    ap.add_argument("--config", default=str(REPO / "src" / "configs" / "config_ecg_heart_delaywavev3.yaml"))
    ap.add_argument("--n-xsubj", type=int, default=128)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output JSON")
    args = ap.parse_args()
    mesh_dir = args.mesh_dir or str(REPO / "src" / OmegaConf.load(args.config).dataset.npy_dir)

    sids = sorted({ln.strip() for ln in open(args.subject_list) if ln.strip()})
    print(f"subjects: {len(sids)}")

    tmpl = np.load(args.template).astype(np.float32)
    if tmpl.ndim == 3:
        tmpl = tmpl[0]
    print("loading rest meshes …", flush=True)
    by_sid = mesh_dirs_by_subject(mesh_dir)
    p0 = {s: load_p0(mesh_dir, by_sid, s) for s in sids}
    p0_used = {c: ({s: tmpl for s in sids} if c.startswith("cohort_mean") else p0)
               for c in args.conditions}

    base_dir = os.path.join(args.root, args.conditions[0], "npy_eval")
    gt_cache = {}
    for sid in sids:
        gt = evm._to_tensor(np.load(os.path.join(base_dir, f"{sid}_gt.npy")))
        gt_cache[sid] = (gt, evm._detrend(gt))
    T = gt_cache[sids[0]][0].shape[0]

    stats, gt_flat = {}, None
    for c in args.conditions:
        d = os.path.join(args.root, c, "npy_eval")
        missing = [s for s in sids if not os.path.exists(os.path.join(d, f"{s}_pred.npy"))]
        assert not missing, f"{c}: {len(missing)} missing preds"
        probe = sids[len(sids) // 2]
        assert np.array_equal(np.load(os.path.join(d, f"{probe}_gt.npy")),
                              np.load(os.path.join(base_dir, f"{probe}_gt.npy"))), f"{c}: GT differs"
        print(f"  {c}: loading …", flush=True)
        stats[c] = per_subject_stats(d, sids, gt_cache, p0_used[c], p0)
        assert torch.isfinite(stats[c]["l2sq_d"]).all(), f"{c}: non-finite offset-removed L2"
        if gt_flat is None:
            gt_flat = torch.stack([evm._center_matrix_t(gt_cache[s][1]).reshape(-1).float()
                                   for s in sids])

    grams = {c: diversity_grams(stats[c]["flat"], T) for c in args.conditions}
    grams["__gt__"] = diversity_grams(gt_flat, T)

    rng = np.random.RandomState(args.seed)
    n = len(sids)
    all_idx = np.arange(n)
    keys = ("RV", "PAR", "APCR", "APDR", "RMSE_abs", "RMSE_offrem")
    out = {c: {k: [] for k in keys} for c in args.conditions}

    point = {}
    for c in args.conditions:
        rv, _, par = point_metrics(stats[c], all_idx)
        a, b = rmse_variants(stats[c], all_idx)
        srv, sd = diversity_for(all_idx[: args.n_xsubj], *grams[c])
        grv, gd = diversity_for(all_idx[: args.n_xsubj], *grams["__gt__"])
        point[c] = {"RV": rv, "PAR": par, "APCR": abs(srv / grv - 1), "APDR": abs(sd / gd - 1),
                    "RMSE_abs": a, "RMSE_offrem": b}

    for b_i in range(args.B):
        boot = rng.randint(0, n, size=n)
        draw = rng.choice(n, size=args.n_xsubj, replace=False)
        grv, gd = diversity_for(draw, *grams["__gt__"])
        for c in args.conditions:
            rv, _, par = point_metrics(stats[c], boot)
            a, bb = rmse_variants(stats[c], boot)
            srv, sdist = diversity_for(draw, *grams[c])
            o = out[c]
            o["RV"].append(rv); o["PAR"].append(par)
            o["APCR"].append(abs(srv / grv - 1)); o["APDR"].append(abs(sdist / gd - 1))
            o["RMSE_abs"].append(a); o["RMSE_offrem"].append(bb)
        if (b_i + 1) % 200 == 0:
            print(f"  bootstrap {b_i+1}/{args.B}", flush=True)

    res = {"n_subjects": n, "B": args.B, "n_xsubj": args.n_xsubj, "point": point, "ci": {}}
    for c in args.conditions:
        res["ci"][c] = {k: {"median": float(np.median(v)),
                            "lo": float(np.percentile(v, 2.5)),
                            "hi": float(np.percentile(v, 97.5))} for k, v in out[c].items()}
        print(f"{c}: " + "  ".join(f"{k} {res['ci'][c][k]['median']:.3f}" for k in keys))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
