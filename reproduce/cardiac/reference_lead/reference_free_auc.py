#!/usr/bin/env python
"""ES timing classification from the delay map with and without a reference lead (Appendix A.7).

Step 2, on the output of extract_delay_es.py. Three per-lead timing representations are built
from the same pairwise lag tensor D[r, c, s]:
  leadII  t[c] = D[II, c]                                     (Time2Track, reference lead II)
  mean    t[c] = mean_{r != c} D[r, c],    centred across leads  (no reference lead)
  median  t[c] = median_{r != c} D[r, c],  centred across leads  (no reference lead)
r == c is excluded because D[c, c] = 0 is a tautology, not a measurement. The lead II column of
the leadII representation is identically 0 and is dropped.

Task: early versus late end-systole, the outer tertiles of the GT ES frame (1 = late). Each
representation is standardised and fed to an L2 logistic regression (C = 1.0), and the score is
the ROC-AUC of the out-of-fold predictions of a 5-fold stratified CV (random_state 42).
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

II = 1
OFFDIAG = ~np.eye(12, dtype=bool)          # (12 ref, 12 lead) True where r != c


def rep_leadII(D):
    return D[:, II, :, :].copy()


def rep_median(D):
    Dm = np.where(OFFDIAG[None, :, :, None], D, np.nan)
    t = np.nanmedian(Dm, axis=1)                    # (N,12,3)
    return t - t.mean(axis=1, keepdims=True)


def rep_mean(D):
    Dm = np.where(OFFDIAG[None, :, :, None], D, np.nan)
    t = np.nanmean(Dm, axis=1)
    return t - t.mean(axis=1, keepdims=True)


def features(rep, drop_col=None):
    """(N,12,3) -> (N, F) design matrix; optionally drop an all-zero lead column."""
    if drop_col is None:
        return rep.reshape(len(rep), -1)
    keep = [c for c in range(12) if c != drop_col]
    return rep[:, keep, :].reshape(len(rep), -1)


def oof_auc(X, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def es_split(es):
    """Outer tertiles: 1 = late ES."""
    q33, q67 = np.percentile(es, [33.3333, 66.6667])
    early, late = es <= q33, es >= q67
    mask = early | late
    return mask, late[mask].astype(int), q33, q67


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="npz written by extract_delay_es.py")
    ap.add_argument("--out", default=None, help="optional JSON with the AUCs")
    a = ap.parse_args()

    z = np.load(a.features)
    D = z["delta"].astype(np.float64)                  # (N,12,12,3)
    es = z["es"].astype(float)
    reps = {"leadII": rep_leadII(D), "mean": rep_mean(D), "median": rep_median(D)}

    mk, yy, q33, q67 = es_split(es)
    print(f"ES tertiles <= {q33:.0f} / >= {q67:.0f} frames, n = {mk.sum()} "
          f"({yy.sum()} late / {len(yy) - yy.sum()} early)")
    out = {"n": int(mk.sum())}
    for k, t in reps.items():
        assert not np.isnan(t).any()
        X = features(t[mk], drop_col=II if k == "leadII" else None)
        out[f"auc_{k}"] = float(roc_auc_score(yy, oof_auc(X, yy)))
        print(f"  {k:7s} AUC {out[f'auc_{k}']:.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
