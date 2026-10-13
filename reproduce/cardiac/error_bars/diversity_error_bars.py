"""Bootstrap SD of |APCR-1| and |APDR-1| for the whole-heart table (Table 1 and Appendix A.5).

A paired subject-level bootstrap over the diversity sample of the paper, i.e. the 128 validation
subjects with the smallest subject ids (`sorted(subs.items())[:128]` in eval_vertex_multiscale.py).
Each resample draws 128 subjects with replacement, and the numerator and the denominator of APCR
(xsubjPredRV / xsubjGTRV) and APDR (xsDistPred / xsDistGT) are recomputed on the same resample, so
their correlation is carried. Pairs of the same original subject are excluded (`_offdiag_mean`).
One resample matrix is shared by all methods, because the 128 subjects are the same for every
method. The reported +- is the SD of |APCR-1| and |APDR-1| across the resamples.

Before any bootstrap number is written, the point estimates of the four raw columns are checked
against the metrics CSV of eval_vertex_multiscale.py (to 5e-4).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reproduce" / "metrics"))
import paper_metrics as pm  # noqa: E402

# Table 1 order: label -> variant directory (output_dir of the matching config).
METHODS = [
    "CNN=ecg_heart_8L_256d_49frames",
    "TeCh=ecg_heart_tech_8L_256d_49frames",
    "PhysioWave=ecg_heart_physiowavenopool_8L_256d_49frames",
    "ShiftInvariant=ecg_heart_shift_invariant_8L_256d_49frames",
    "Delay-T=ecg_heart_delay_8L_256d_49frames",
    "Delay-QRS=ecg_heart_delayqrs_8L_256d_49frames",
    "Ours=ecg_heart_delaywavev3_8L_256d_49frames",
]
RAW = ["xsubjPredRV", "xsubjGTRV", "xsDistPred", "xsDistGT"]


def csv_raw(path):
    """Published numerator/denominator means per experiment (whole-heart rows)."""
    out = {}
    for row in csv.DictReader(open(path)):
        if row["scale"] != "whole":
            continue
        out[row["experiment"]] = {k: float(row[k].split("±")[0]) for k in RAW}
    return out


def sample_sids(root, exp, nsamp):
    d = os.path.join(root, exp, "npy_eval")
    sids = sorted(f[:-9] for f in os.listdir(d) if f.endswith("_pred.npy"))
    return sids[:nsamp], d


def curves(root, exp, nsamp):
    """[nsamp, T, V*3] detrended pred and GT, in the paper's subject order."""
    sids, d = sample_sids(root, exp, nsamp)
    cp, cg = [], []
    for s in sids:
        p = torch.as_tensor(np.load(f"{d}/{s}_pred.npy"), dtype=torch.float64, device=pm.DEV)
        g = torch.as_tensor(np.load(f"{d}/{s}_gt.npy"), dtype=torch.float64, device=pm.DEV)
        cp.append(pm.detrend(p).reshape(p.shape[0], -1))
        cg.append(pm.detrend(g).reshape(g.shape[0], -1))
    return sids, torch.stack(cp), torch.stack(cg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO / "checkpoints"),
                    help="directory with <variant>/npy_eval/{subid}_{pred,gt}.npy")
    ap.add_argument("--csv", default=str(REPO / "outputs" / "vertex_multiscale" / "metrics-avgstd_per_scale.csv"),
                    help="metrics CSV of eval_vertex_multiscale.py on the same --root (reproduction check)")
    ap.add_argument("--methods", nargs="+", default=METHODS, metavar="LABEL=VARIANT",
                    help="table rows in order (default: the seven rows of Table 1)")
    ap.add_argument("--nsamp", type=int, default=128, help="diversity sample size")
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output JSON")
    args = ap.parse_args()
    methods = [m.split("=", 1) for m in args.methods]

    ref = csv_raw(args.csv)
    boot = np.random.default_rng(args.seed).integers(0, args.nsamp, (args.B, args.nsamp))
    res, sid_ref = {}, None

    for name, exp in methods:
        sids, cp, cg = curves(args.root, exp, args.nsamp)
        if sid_ref is None:
            sid_ref = sids
        assert sids == sid_ref, f"{name}: different {args.nsamp}-subject sample, not paired"

        mats = (*pm.pair_matrices(cp), *pm.pair_matrices(cg))
        idx0 = torch.arange(args.nsamp, device=pm.DEV)
        apcr, apdr, xp, xg, dp, dg = pm.diversity_from(mats, idx0)

        got = dict(xsubjPredRV=xp, xsubjGTRV=xg, xsDistPred=dp, xsDistGT=dg)
        dev = {k: abs(got[k] - ref[exp][k]) for k in RAW}
        assert max(dev.values()) < 5e-4, (
            f"{name}: does not reproduce {args.csv}: "
            + ", ".join(f"{k} {got[k]:.4f} vs csv {ref[exp][k]:.4f}" for k in RAW))

        bc, bd = [], []
        for row in boot:
            sel = torch.as_tensor(row, dtype=torch.long, device=pm.DEV)
            a, b = pm.diversity_from(mats, sel)[:2]
            bc.append(a); bd.append(b)
        bc, bd = np.asarray(bc), np.asarray(bd)

        res[name] = dict(
            exp=exp, APCR=apcr, APDR=apdr,
            absAPCRm1=abs(apcr - 1), absAPDRm1=abs(apdr - 1),
            se_absAPCRm1=float(np.std(np.abs(bc - 1), ddof=1)),
            se_absAPDRm1=float(np.std(np.abs(bd - 1), ddof=1)),
            csv_dev=max(dev.values()),
        )
        r = res[name]
        print(f'{name:15s} |APCR-1| {r["absAPCRm1"]:.3f}±{r["se_absAPCRm1"]:.3f}   '
              f'|APDR-1| {r["absAPDRm1"]:.3f}±{r["se_absAPDRm1"]:.3f}   '
              f'(reproduces CSV to {r["csv_dev"]:.1e})', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(n_subjects=args.nsamp, n_boot=args.B, seed=args.seed, methods=res),
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
