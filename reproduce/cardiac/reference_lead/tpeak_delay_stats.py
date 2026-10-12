#!/usr/bin/env python
"""Size of the inter-lead T-peak delays in UKB (Appendix A.7): median and IQR of |delay|.

Reads the 12x12 inter-lead T-peak delay matrix of the Fig. 1b analysis (all_ecg_delay_matrix.npz,
keys `filename` = '<subid>.npy' and `T_peak` (N, 12, 12) in ms, T_peak[i, j] = T-peak(j) - T-peak(i))
and pools |T_peak[i, j]| over the 66 lead pairs i < j and over the validation subjects, i.e. the
subjects with a {subid}_gt.npy in --npy-eval.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3] / "src"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delay-matrix", required=True, help="all_ecg_delay_matrix.npz")
    ap.add_argument("--npy-eval", default=None,
                    help="npy_eval dir with {subid}_gt.npy (default: <output_dir of --config>/npy_eval)")
    ap.add_argument("--config", default=str(SRC / "configs" / "config_ecg_heart_delaywavev3.yaml"))
    a = ap.parse_args()
    npy_eval = a.npy_eval
    if npy_eval is None:
        from omegaconf import OmegaConf
        npy_eval = os.path.normpath(os.path.join(SRC, OmegaConf.load(a.config).output_dir, "npy_eval"))

    subs = sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(npy_eval, "*_gt.npy")))
    dz = np.load(a.delay_matrix, allow_pickle=True)
    f2i = {f[:-4]: i for i, f in enumerate(dz["filename"])}
    missing = [s for s in subs if s not in f2i]
    if missing:
        sys.exit(f"{len(missing)} of {len(subs)} subjects have no delay matrix")
    Tp = dz["T_peak"][[f2i[s] for s in subs]].astype(float)
    assert not np.isnan(Tp).any()
    iu = np.triu_indices(12, 1)
    d = np.abs(Tp[:, iu[0], iu[1]]).ravel()
    q25, q50, q75, q95 = np.percentile(d, [25, 50, 75, 95])
    print(f"|inter-lead T-peak delay| over {len(iu[0])} pairs x {len(subs)} subjects: "
          f"median {q50:.0f} ms, IQR {q25:.0f} to {q75:.0f} ms, p95 {q95:.0f} ms, max {d.max():.0f} ms")


if __name__ == "__main__":
    main()
