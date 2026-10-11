"""Radar charts of per-material scores.

For each metric we draw one radar plot whose axes are the heart materials
(LV_ENDOCARDIAL, LV_EPICARDIAL, RV_EPICARDIAL, RV_FREEWALL, RV_SEPTUM),
and one polyline per experiment.

Inputs:
  outputs/facearea_multiscale/metrics_per_scale.csv   (rows: scale=material_*)
  outputs/vertex_multiscale/metrics_per_scale.csv     (rows: scale=material_*)

Outputs (PNG + PDF):
  outputs/radar_facearea/<metric>.{png,pdf}
  outputs/radar_vertex/<metric>.{png,pdf}

The metric labels follow the paper:
  meanCorr / stdRatio / detrCorr / ampRatio / peakAmpR /
  dPredCorr / dGTCorr / dPredAmp / dGTAmp / dPredDist / dGTDist / L2 / L2N
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
})

# label -> (face_num, vertex_num, face_den, vertex_den, direction)
# direction: "higher"=larger is better, "lower"=smaller is better,
#            "target1"=closer to 1.0 is better.
METRICS: Dict[str, Tuple[str, str, Optional[str], Optional[str], str]] = {
    "meanCorr":       ("meanCorr",      "meanRV",        None,        None,       "higher"),
    "stdRatio":       ("subjStdRatio",  "subjStdRatio",  None,        None,       "target1"),
    "detrCorr":       ("detrCorr",      "detrRV",        None,        None,       "higher"),
    "ampRatio":       ("ampRatio_rom",  "ampRatio_rom",  None,        None,       "target1"),
    "peakAmpR":       ("peakAmpR_rom",  "peakAmpR_rom",  None,        None,       "target1"),
    "dPredCorr_dGTCorr": ("xsubjPred",   "xsubjPredRV",  "xsubjGT",   "xsubjGTRV","target1"),
    "dPredAmp_dGTAmp":   ("xsAmpCVPred", "xsAmpCVPred",  "xsAmpCVGT", "xsAmpCVGT","target1"),
    "dPredDist_dGTDist": ("xsDistPred",  "xsDistPred",   "xsDistGT",  "xsDistGT", "target1"),
    "L2":             ("predGtL2_rom",  "predGtL2_rom",  None,        None,       "lower"),
    "L2N":            ("predGtL2N_rom", "predGtL2N_rom", None,        None,       "lower"),
}

MATERIALS = ["LV_ENDOCARDIAL", "LV_EPICARDIAL", "RV_EPICARDIAL", "RV_FREEWALL", "RV_SEPTUM"]

NAME_MAP: Dict[str, str] = {
    "ecg_heart_8L_256d_49frames":              "CNN",
    "ecg_heart_tech_8L_256d_49frames":         "TeCh",
    "ecg_heart_physiowavenopool_8L_256d_49frames": "PhysioWave",
    "ecg_heart_shift_invariant_8L_256d_49frames":  "ShiftInvariant",
    "ecg_heart_delay_8L_256d_49frames":        "Delay-T",
    "ecg_heart_delayqrs_8L_256d_49frames":     "Delay-QRS",
    # Best delaywave variant chosen across whole-heart metrics (rank-sum):
    "ecg_heart_delaywavev3_8L_256d_49frames":  "Ours",
}
ROW_ORDER = ["CNN", "TeCh", "PhysioWave", "ShiftInvariant", "Delay-T", "Delay-QRS", "Ours"]


def short_name(exp: str) -> str:
    return NAME_MAP.get(exp, exp.replace("_8L_256d_49frames", "").replace("ecg_heart_", ""))


EPS = 1e-9


def _parse_metric_value(value: str) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    for sep in ("±", "+-", "+/-"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_material_rows(csv_path: str, num_col: str,
                       den_col: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Return {experiment: {material_name: value}}. If den_col is given, compute num/den."""
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            scale = r.get("scale", "")
            if not scale.startswith("material_"):
                continue
            mat = scale[len("material_"):]
            num = _parse_metric_value(r.get(num_col, ""))
            if den_col is None:
                v = num
            else:
                den = _parse_metric_value(r.get(den_col, ""))
                v = num / den if abs(den) > EPS else float("nan")
            out[r["experiment"]][mat] = v
    return out


def _radar_angles(n: int) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.concatenate([a, a[:1]])


def _periodic_smooth(angles: np.ndarray, vals: np.ndarray, samples: int = 360):
    """Smooth a closed-loop curve via FFT low-pass; preserves periodicity."""
    base = vals[:-1]  # drop the duplicate closing sample
    n = base.shape[0]
    if not np.all(np.isfinite(base)) or n < 3:
        return angles, vals
    spec = np.fft.rfft(base)
    keep = max(2, n // 2 + 1)  # keep all available harmonics for n=5: 3
    if spec.shape[0] > keep:
        spec[keep:] = 0
    dense_n = max(samples, 8 * n)
    upsampled = np.fft.irfft(spec, n=dense_n) * (dense_n / n)
    dense_ang = np.linspace(0, 2 * np.pi, dense_n, endpoint=False)
    dense_ang = np.concatenate([dense_ang, dense_ang[:1]])
    dense_val = np.concatenate([upsampled, upsampled[:1]])
    return dense_ang, dense_val


def _to_goodness(vec: np.ndarray, direction: str) -> np.ndarray:
    """Map raw metric values to a 'higher = better' scale for radial display."""
    if direction == "higher":
        return vec
    if direction == "lower":
        return -vec
    if direction == "target1":
        return -np.abs(vec - 1.0)
    raise ValueError(direction)


def _ticks_for(direction: str, lo_g: float, hi_g: float, n: int = 4):
    """Pick a few raw-value ticks placed at the radial (goodness) coordinates."""
    g_ticks = np.linspace(lo_g, hi_g, n)
    if direction == "higher":
        raw = g_ticks
    elif direction == "lower":
        raw = -g_ticks
    else:  # target1: two raw values per goodness, pick "1 + |g|" arbitrarily; show |1-x|
        raw = 1.0 + np.abs(g_ticks)  # interpretation: distance-from-1 == |raw-1|
    labels = [f"{v:.2g}" for v in raw]
    return g_ticks.tolist(), labels


def plot_one(values: Dict[str, Dict[str, float]], metric_label: str,
             direction: str, out_path: str):
    angles = _radar_angles(len(MATERIALS))
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab20")

    inv_map = {v: k for k, v in NAME_MAP.items()}
    ordered_pairs = []
    for i, display in enumerate(ROW_ORDER):
        exp = inv_map.get(display)
        if exp is not None and exp in values:
            ordered_pairs.append((i, exp, display))

    baseline_palette = plt.get_cmap("tab10")
    series = []
    all_g: List[float] = []
    for slot, exp, display in ordered_pairs:
        raw = np.array([values[exp].get(m, float("nan")) for m in MATERIALS], dtype=float)
        raw_full = np.concatenate([raw, raw[:1]])
        good_full = _to_goodness(raw_full, direction)
        series.append((slot, exp, display, good_full, raw_full))
        all_g += [float(x) for x in good_full[:-1] if np.isfinite(x)]

    for slot, exp, display, good_full, raw_full in series:
        if np.all(np.isfinite(good_full)):
            sm_ang, sm_val = _periodic_smooth(angles, good_full)
        else:
            sm_ang, sm_val = angles, good_full
        is_ours = display == "Ours"
        if is_ours:
            color = "#d62728"   # bold red
            lw = 3.2
            ls = "-"
            zorder = 5
        else:
            color = baseline_palette(slot % 10)
            lw = 1.6
            ls = "--"
            zorder = 2
        ax.plot(sm_ang, sm_val, color=color, linewidth=lw, linestyle=ls,
                zorder=zorder)
        ax.fill(sm_ang, sm_val, color=color, alpha=0.10 if is_ours else 0.05,
                zorder=zorder - 1)
        ax.plot(angles, good_full, "o", color=color,
                markersize=4 if is_ours else 3, zorder=zorder + 1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    # ax.set_title(f"{metric_label}  (outer = better)", y=1.10, fontsize=16)
    ax.set_title("")
    ax.grid(True, alpha=0.4)

    if all_g:
        lo, hi = min(all_g), max(all_g)
        if lo == hi:
            hi = lo + 1.0
        pad = 0.05 * (hi - lo)
        lo_g, hi_g = lo - pad, hi + pad
        ax.set_ylim(lo_g, hi_g)
        ticks, labels = _ticks_for(direction, lo_g, hi_g)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=16)

    hi_r = hi_g if all_g else 1.0
    for i, mat in enumerate(MATERIALS):
        ang = float(angles[i])
        deg = np.degrees(ang)
        rad = hi_r * 1.18
        rot = -deg
        rot = ((rot + 180) % 360) - 180
        if abs(rot) > 90:
            rot += 180
            rot = ((rot + 180) % 360) - 180
        ax.text(ang, rad, mat, fontsize=14, ha="center", va="center",
                rotation=rot, rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def run(csv_path: str, col_key: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for label, cols in METRICS.items():
        num_col = cols[col_key]
        den_col = cols[col_key + 2]
        direction = cols[4]
        data = load_material_rows(csv_path, num_col, den_col)
        if not data:
            print(f"  skip {label} ({num_col} not present)")
            continue
        title = label.replace("_", " / ")
        plot_one(data, title, direction, os.path.join(out_dir, label))
    print(f"wrote radars to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-csv", default="outputs/facearea_multiscale/metrics_per_scale.csv")
    ap.add_argument("--vertex-csv", default="outputs/vertex_multiscale/metrics_per_scale.csv")
    ap.add_argument("--out-face", default="outputs/radar_facearea")
    ap.add_argument("--out-vertex", default="outputs/radar_vertex")
    args = ap.parse_args()
    if os.path.exists(args.face_csv):
        run(args.face_csv, 0, args.out_face)
    if os.path.exists(args.vertex_csv):
        run(args.vertex_csv, 1, args.out_vertex)


if __name__ == "__main__":
    main()
