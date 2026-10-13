#!/usr/bin/env python
"""Build ICD-10 subgroup subject lists for the validation subjects that have predictions.

Each subject's UK Biobank field 41270 (all ICD-10 diagnoses, pipe-separated codes) is tokenized and
matched code by code against the regexes below. A subject without any `any_cardiac` code is healthy
(subjects missing from the table count as healthy). Writes <out>/<subgroup>.txt and
<out>/healthy.txt (one subject id per line) and <out>/counts.json.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# token-level regexes (each token is one ICD-10 code, e.g. "I480")
GROUPS = {
    "any_cardiac":      re.compile(r"^I(2[0-5]|4[2-9]|50)"),
    "ischemic":         re.compile(r"^I2[0-5]"),
    "arrhythmia_cond":  re.compile(r"^I4[4-9]"),
    "atrial_fib":       re.compile(r"^I48"),
    "heart_failure":    re.compile(r"^I50"),
    "conduction_block": re.compile(r"^I4[4-5]"),
    "cardiomyopathy":   re.compile(r"^I4[2-3]"),
}
ANY = GROUPS["any_cardiac"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--icd-tsv", required=True,
                    help="UK Biobank tab-separated extract: first column eid, a column named p41270")
    ap.add_argument("--pred-dir",
                    default=str(REPO / "checkpoints" / "ecg_heart_delaywavev3_8L_256d_49frames" / "npy_eval"),
                    help="npy_eval directory whose *_pred.npy files define the evaluated subjects")
    ap.add_argument("--out", required=True, help="output directory for the lists")
    args = ap.parse_args()

    # 1. evaluated subject ids
    subids = {os.path.basename(p)[:-9] for p in glob.glob(os.path.join(args.pred_dir, "*_pred.npy"))}
    print(f"evaluated subjects: {len(subids)}")

    # 2. join eid -> p41270 cell
    eid_to_cell = {}
    with open(args.icd_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        col = header.index("p41270")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid_to_cell[parts[0]] = parts[col] if len(parts) > col else ""

    # 3. classify each subject
    members = {g: [] for g in GROUPS}
    healthy = []
    missing_icd = 0  # subjects with no row in the ICD table -> no cardiac code (healthy)
    for sid in sorted(subids):
        cell = eid_to_cell.get(sid)
        if cell is None:
            missing_icd += 1
            cell = ""
        tokens = cell.split("|") if cell else []
        for g, rx in GROUPS.items():
            if any(rx.match(t) for t in tokens):
                members[g].append(sid)
        if not any(ANY.match(t) for t in tokens):
            healthy.append(sid)

    # 4. write lists
    os.makedirs(args.out, exist_ok=True)
    for g, sids in members.items():
        with open(os.path.join(args.out, f"{g}.txt"), "w") as fo:
            fo.write("\n".join(sids) + ("\n" if sids else ""))
    with open(os.path.join(args.out, "healthy.txt"), "w") as fo:
        fo.write("\n".join(healthy) + "\n")

    counts = {g: len(sids) for g, sids in members.items()}
    counts["healthy"] = len(healthy)
    counts["_total"] = len(subids)
    counts["_subjects_missing_from_icd_table"] = missing_icd
    print(json.dumps(counts, indent=2))
    with open(os.path.join(args.out, "counts.json"), "w") as fo:
        json.dump(counts, fo, indent=2)
    assert counts["any_cardiac"] + counts["healthy"] == len(subids), "partition mismatch"


if __name__ == "__main__":
    main()
