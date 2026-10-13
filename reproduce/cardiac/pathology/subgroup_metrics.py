#!/usr/bin/env python
"""Per-subgroup whole-heart metrics with a matched-n healthy bootstrap.

Replicates the whole-heart metrics of eval_vertex_multiscale.py (RV = meanRV, RMSE = predGtL2N_rom,
PAR = peakAmpR_rom, APCR = xsubjPredRV / xsubjGTRV, APDR = xsDistPred / xsDistGT) from per-subject
sufficient statistics (reproduce/metrics/cardiac_stats.py), so subjects can be bootstrap-resampled
without re-reading the prediction files. GT statistics are variant-independent and computed once.

Per subgroup and per variant it reports
  - the point estimate over all subgroup subjects (APCR/APDR over all n subjects pairwise),
  - a bootstrap of the subgroup: n subjects drawn with replacement from the subgroup,
  - a matched-n healthy bootstrap: n subjects drawn with replacement from the healthy pool,
each as [median, 2.5 %, 97.5 %] over the draws. One torch.Generator (seeded by --seed) is consumed
in sequence over variants, subgroups and the two bootstraps.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reproduce" / "metrics"))
from cardiac_stats import Store, dev, make_resid_op, metrics, rv_per_vertex  # noqa: E402

VARIANTS = [  # label=variant directory; the first one also supplies the GT files
    "Ours=ecg_heart_delaywavev3_8L_256d_49frames",
    "CNN=ecg_heart_8L_256d_49frames",
    "TeCh=ecg_heart_tech_8L_256d_49frames",
]
SUBGROUPS = ["any_cardiac", "ischemic", "arrhythmia_cond", "atrial_fib",
             "heart_failure", "conduction_block", "cardiomyopathy"]
METS = ["RV", "RMSE", "PAR", "APCR", "APDR"]


def pctl(arr):  # arr [ndraws, 5] -> per-metric [median, lo, hi]
    q = np.percentile(arr, [50, 2.5, 97.5], axis=0)  # [3,5]
    return {m: [float(q[0, j]), float(q[1, j]), float(q[2, j])] for j, m in enumerate(METS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lists", required=True, help="directory written by build_subgroups.py")
    ap.add_argument("--root", default=str(REPO / "checkpoints"),
                    help="directory with <variant>/npy_eval/{subid}_{pred,gt}.npy")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, metavar="LABEL=VARIANT")
    ap.add_argument("--subgroups", nargs="+", default=SUBGROUPS)
    ap.add_argument("--ndraws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output JSON")
    args = ap.parse_args()
    variants = dict(v.split("=", 1) for v in args.variants)
    gt_variant = next(iter(variants.values()))

    torch.manual_seed(args.seed)
    g = torch.Generator(device=dev); g.manual_seed(args.seed)

    def read_list(name):
        with open(os.path.join(args.lists, f"{name}.txt")) as f:
            return [ln.strip() for ln in f if ln.strip()]

    # subject order = sorted, as in the evaluation pipeline
    healthy = read_list("healthy")
    any_card = read_list("any_cardiac")
    subids = sorted(set(healthy) | set(any_card))
    pos = {s: i for i, s in enumerate(subids)}
    N = len(subids)
    print(f"N subjects = {N}")

    gt_dir = os.path.join(args.root, gt_variant, "npy_eval")
    T, V, C = np.load(os.path.join(gt_dir, f"{subids[0]}_gt.npy")).shape
    TT, D = T * T, T * V * C
    print(f"shape T={T} V={V} C={C}")
    resid = make_resid_op(T)
    st = Store(N, V, TT, D, T, C)

    def load_gt():
        """Read each _gt.npy once -> gt_raw + gt-diversity stats."""
        t0 = time.time()
        for i, sid in enumerate(subids):
            gt = torch.as_tensor(np.load(os.path.join(gt_dir, f"{sid}_gt.npy")),
                                 dtype=torch.float64, device=dev)
            rg = (resid @ gt.reshape(T, -1)).reshape(T, V, C)
            xg = rg.reshape(T, -1); xg = xg - xg.mean(0, keepdim=True)
            st.gt_raw[i] = gt
            st.gtsq[i] = (gt * gt).sum((0, 2))
            st.peakg[i] = (rg.amax(0) - rg.amin(0)).mean(-1)
            st.Gg[i] = (xg @ xg.T).reshape(-1)
            Mg = xg.reshape(-1); st.Mg[i] = Mg; st.sqg[i] = (Mg * Mg).sum()
            if i % 2000 == 0:
                print(f"  [GT] {i}/{N}  {time.time()-t0:.0f}s", flush=True)

    def load_pred(label):
        """Read each _pred.npy once -> pred stats (reuses stored gt_raw)."""
        vdir = os.path.join(args.root, variants[label], "npy_eval")
        t0 = time.time()
        for i, sid in enumerate(subids):
            p = torch.as_tensor(np.load(os.path.join(vdir, f"{sid}_pred.npy")),
                                dtype=torch.float64, device=dev)
            gt = st.gt_raw[i]
            rp = (resid @ p.reshape(T, -1)).reshape(T, V, C)
            st.rv[i] = rv_per_vertex(p, gt)
            st.l2sq[i] = ((p - gt) ** 2).sum((0, 2))
            st.peakp[i] = (rp.amax(0) - rp.amin(0)).mean(-1)
            xp = rp.reshape(T, -1); xp = xp - xp.mean(0, keepdim=True)
            st.Gp[i] = (xp @ xp.T).reshape(-1)
            Mp = xp.reshape(-1); st.Mp[i] = Mp; st.sqp[i] = (Mp * Mp).sum()
            if i % 2000 == 0:
                print(f"  [{label}] {i}/{N}  {time.time()-t0:.0f}s", flush=True)

    def boot(pool_pos, n):
        out = []
        for _ in range(args.ndraws):
            sel = pool_pos[torch.randint(len(pool_pos), (n,), generator=g, device=dev)]
            out.append(metrics(st, sel))
        return np.stack(out)  # [ndraws, 5]

    healthy_pos = torch.as_tensor([pos[s] for s in healthy], device=dev)
    results = {"n": {name: len(read_list(name)) for name in args.subgroups},
               "point": {}, "boot_subgroup": {}, "boot_healthy_matched": {},
               "meta": {"ndraws": args.ndraws, "seed": args.seed,
                        "variants": variants, "subgroups": args.subgroups}}
    results["n"]["healthy"] = len(healthy)

    load_gt()
    for label in variants:
        load_pred(label)
        for kind in ("point", "boot_subgroup", "boot_healthy_matched"):
            results[kind][label] = {}
        for name in args.subgroups:
            sg = read_list(name)
            sg_pos = torch.as_tensor([pos[s] for s in sg], device=dev)
            n = len(sg)
            pt = metrics(st, sg_pos)
            bs = boot(sg_pos, n)
            bh = boot(healthy_pos, n)
            results["point"][label][name] = pt.tolist()
            results["boot_subgroup"][label][name] = pctl(bs)
            results["boot_healthy_matched"][label][name] = pctl(bh)
            print(f"  [{label}:{name} n={n}] point RV/RMSE/PAR/APCR/APDR = "
                  f"{pt[0]:.3f}/{pt[1]:.3f}/{pt[2]:.3f}/{pt[3]:.3f}/{pt[4]:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
