#!/usr/bin/env python
"""Diseased minus matched healthy (Table 2) from the JSON of subgroup_metrics.py.

Each row is that variant's subgroup bootstrap median minus the median of its own matched-n healthy
bootstrap, oriented so that a larger value means a larger loss on the subgroup:
  RMSE (lower is better)  -> subgroup - healthy
  PAR, APCR, APDR (-> 1)  -> |x_subgroup - 1| - |x_healthy - 1|
A negative value means the variant does better on the subgroup than on healthy subjects.
"""
import argparse
import json

COLS = ["RMSE", "PAR", "APCR", "APDR"]
HEAD = ["RMSE", "|PAR-1|", "|APCR-1|", "|APDR-1|"]


def degradation_delta(J, var, sg, met):
    s = J["boot_subgroup"][var][sg][met][0]          # bootstrap median
    h = J["boot_healthy_matched"][var][sg][met][0]
    if met in ("PAR", "APCR", "APDR"):
        return abs(s - 1) - abs(h - 1)
    return s - h                                      # RMSE


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", help="JSON written by subgroup_metrics.py")
    ap.add_argument("--subgroup", default="any_cardiac")
    args = ap.parse_args()

    J = json.load(open(args.results))
    print(f"{args.subgroup}: diseased minus matched healthy (n={J['n'][args.subgroup]})")
    print(f"{'Delta':8s}" + "".join(f"{h:>10s}" for h in HEAD))
    for var in J["meta"]["variants"]:
        print(f"{var:8s}" + "".join(f"{degradation_delta(J, var, args.subgroup, m):+10.3f}" for m in COLS))


if __name__ == "__main__":
    main()
