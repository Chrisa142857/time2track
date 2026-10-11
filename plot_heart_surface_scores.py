"""Render per-vertex scores onto the 3D heart surface.

For every metric, write a single SVG laying out a grid of mesh renderings:
  rows    = anterior / lateral / posterior views
  columns = methods (in the same order as the radar plots)
All subplots share one colorbar; the colormap is computed on a "goodness"
scale (higher = better) just like the radar, while the colorbar tick labels
show the corresponding raw metric values.

Sources
-------
  outputs/vertex_multiscale/<exp>_per_vertex.npz
  outputs/facearea_multiscale/<exp>_per_vertex.npz
  ../ukb_heart_mesh_visual/control_face.ply

Outputs
-------
  outputs/surface_vertex/<metric>.svg
  outputs/surface_facearea/<metric>.svg
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# label -> (vertex_num, face_num, vertex_den, face_den, direction)
# direction: "higher"=larger is better, "lower"=smaller is better,
#            "target1"=closer to 1.0 is better.
# If *_den is None, the raw value is used directly; otherwise we plot num/den.
METRICS: Dict[str, Tuple[str, str, Optional[str], Optional[str], str]] = {
    "meanCorr":          ("meanRV",        "meanCorr",      None,        None,       "higher"),
    "stdRatio":          ("subjStdRatio",  "subjStdRatio",  None,        None,       "target1"),
    "detrCorr":          ("detrRV",        "detrCorr",      None,        None,       "higher"),
    "ampRatio":          ("ampRatio_rom",  "ampRatio_rom",  None,        None,       "target1"),
    "peakAmpR":          ("peakAmpR_rom",  "peakAmpR_rom",  None,        None,       "target1"),
    "dPredCorr_dGTCorr": ("dPredCorr",     "dPredCorr",     "dGTCorr",   "dGTCorr",  "target1"),
    "dPredAmp_dGTAmp":   ("dPredAmp",      "dPredAmp",      "dGTAmp",    "dGTAmp",   "target1"),
    "dPredDist_dGTDist": ("dPredDist",     "dPredDist",     "dGTDist",   "dGTDist",  "target1"),
    "L2":                ("predGtL2_rom",  "predGtL2_rom",  None,        None,       "lower"),
    "L2N":               ("predGtL2N_rom", "predGtL2N_rom", None,        None,       "lower"),
}

EPS = 1e-9

VIEWS = [("anterior", 20, -90), ("lateral", 10, 0), ("posterior", 20, 90)]

NAME_MAP: Dict[str, str] = {
    "ecg_heart_8L_256d_49frames":                  "CNN",
    "ecg_heart_tech_8L_256d_49frames":             "TeCh",
    "ecg_heart_physiowavenopool_8L_256d_49frames": "PhysioWave",
    "ecg_heart_shift_invariant_8L_256d_49frames":  "ShiftInvariant",
    "ecg_heart_delay_8L_256d_49frames":            "Delay-T",
    "ecg_heart_delayqrs_8L_256d_49frames":         "Delay-QRS",
    "ecg_heart_delaywavev3_8L_256d_49frames":      "Ours",
}
ROW_ORDER = ["CNN", "TeCh", "PhysioWave", "ShiftInvariant", "Delay-T", "Delay-QRS", "Ours"]


def load_mesh(mesh_path: str):
    m = trimesh.load(mesh_path, process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def _to_goodness(vec: np.ndarray, direction: str) -> np.ndarray:
    if direction == "higher":
        return vec
    if direction == "lower":
        return -vec
    if direction == "target1":
        return -np.abs(vec - 1.0)
    raise ValueError(direction)


def _cbar_ticks(direction: str, lo_g: float, hi_g: float, n: int = 5):
    g_ticks = np.linspace(lo_g, hi_g, n)
    if direction == "higher":
        raw = g_ticks
    elif direction == "lower":
        raw = -g_ticks
    else:  # target1: show |raw-1| equivalent as 1 + |g|
        raw = 1.0 + np.abs(g_ticks)
    return g_ticks.tolist(), [f"{v:.2g}" for v in raw]


def _face_colors(vert_vals: np.ndarray, faces: np.ndarray, direction: str,
                 vmin_g: float, vmax_g: float, cmap):
    face_vals = vert_vals[faces].mean(axis=1)
    good = _to_goodness(face_vals, direction)
    if vmax_g <= vmin_g:
        vmax_g = vmin_g + 1.0
    norm = (good - vmin_g) / (vmax_g - vmin_g)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    norm = np.clip(norm, 0.0, 1.0)
    return cmap(norm)


def _draw_view(ax, verts, faces, face_colors, elev, azim, ctr, span):
    tri_verts = verts[faces]
    coll = Poly3DCollection(tri_verts, facecolors=face_colors,
                            edgecolors="none", linewidths=0.0)
    ax.add_collection3d(coll)
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_metric_grid(verts: np.ndarray, faces: np.ndarray,
                       per_method: List[Tuple[str, np.ndarray]],
                       metric_label: str, direction: str,
                       out_path: str, cmap_name: str = "viridis"):
    if not per_method:
        return

    pooled_g: List[float] = []
    for _, vals in per_method:
        face_vals = vals[faces].mean(axis=1)
        g = _to_goodness(face_vals, direction)
        pooled_g.extend(float(x) for x in g if np.isfinite(x))
    if not pooled_g:
        return
    lo_g = float(np.percentile(pooled_g, 2))
    hi_g = float(np.percentile(pooled_g, 98))
    if hi_g <= lo_g:
        hi_g = lo_g + 1.0

    cmap = plt.get_cmap(cmap_name)
    n_cols = len(per_method)
    n_rows = len(VIEWS)

    fig = plt.figure(figsize=(2.2 * n_cols + 0.9, 2.2 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1,
                          width_ratios=[1.0] * n_cols + [0.04],
                          wspace=-0.05, hspace=-0.05,
                          left=0.04, right=0.96, top=0.94, bottom=0.02)

    lo = verts.min(axis=0); hi = verts.max(axis=0); ctr = 0.5 * (lo + hi)
    span = float((hi - lo).max()) * 0.5

    for col, (display, vals) in enumerate(per_method):
        fc = _face_colors(vals, faces, direction, lo_g, hi_g, cmap)
        for row, (vname, elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            _draw_view(ax, verts, faces, fc, elev, azim, ctr, span)
            if row == 0:
                title = r"$\bf{Ours}$" if display == "Ours" else display
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.text2D(-0.05, 0.5, vname, transform=ax.transAxes,
                          rotation=90, va="center", ha="right", fontsize=10)

    cax = fig.add_subplot(gs[:, -1])
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lo_g, vmax=hi_g))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    ticks, labels = _cbar_ticks(direction, lo_g, hi_g)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"{metric_label}  (top = better)", fontsize=10)

    fig.suptitle(metric_label, fontsize=13, y=0.995)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _load_per_method(npz_dir: str, key_index: int,
                     n_verts: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {metric_label: {display_name: per_vertex_array}}."""
    out: Dict[str, Dict[str, np.ndarray]] = {label: {} for label in METRICS}
    for exp, display in NAME_MAP.items():
        path = os.path.join(npz_dir, f"{exp}_per_vertex.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        for label, cols in METRICS.items():
            num_key = cols[key_index]
            den_key = cols[key_index + 2]
            if num_key not in data.files:
                continue
            num = np.asarray(data[num_key], dtype=np.float64)
            if num.shape[0] != n_verts:
                print(f"  skip {exp}/{label}: shape {num.shape} vs verts {n_verts}")
                continue
            if den_key is None:
                arr = num
            else:
                if den_key not in data.files:
                    continue
                den = np.asarray(data[den_key], dtype=np.float64)
                if den.shape != num.shape:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    arr = np.where(np.abs(den) > EPS, num / den, np.nan)
            out[label][display] = arr
    return out


def run(npz_dir: str, key_index: int, out_dir: str,
        verts: np.ndarray, faces: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    per_metric = _load_per_method(npz_dir, key_index, verts.shape[0])
    for label, by_display in per_metric.items():
        ordered = [(d, by_display[d]) for d in ROW_ORDER if d in by_display]
        if not ordered:
            print(f"  skip {label}: no methods present")
            continue
        direction = METRICS[label][4]
        out_path = os.path.join(out_dir, f"{label}.svg")
        render_metric_grid(verts, faces, ordered, label, direction, out_path)
        print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="../ukb_heart_mesh_visual/control_face.ply")
    ap.add_argument("--vertex-dir", default="outputs/vertex_multiscale")
    ap.add_argument("--face-dir", default="outputs/facearea_multiscale")
    ap.add_argument("--out-vertex", default="outputs/surface_vertex")
    ap.add_argument("--out-face", default="outputs/surface_facearea")
    args = ap.parse_args()

    verts, faces = load_mesh(args.mesh)
    print(f"mesh: V={verts.shape[0]} F={faces.shape[0]}")
    if os.path.isdir(args.vertex_dir):
        run(args.vertex_dir, 0, args.out_vertex, verts, faces)
    if os.path.isdir(args.face_dir):
        run(args.face_dir, 1, args.out_face, verts, faces)


if __name__ == "__main__":
    main()
