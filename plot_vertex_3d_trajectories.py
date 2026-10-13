"""Plot per-vertex motion summaries (with std envelopes) across subjects.

For each selected vertex we draw a 2-row figure: the top row shows the mean
3D displacement-from-rest trajectory with an axis-aligned std cross at each
timestep, the bottom row shows speed (||velocity||) over time with a mean curve
and a shaded std band. Per-baseline columns share axis limits so a collapsed
method visibly flattens both the trajectory loop and the speed peak relative to
GT and Ours.

Reads <root>/<exp>/npy_eval/<sid>_pred.npy and ..._gt.npy, each shaped
[T, V, 3], matching eval_vertex_multiscale.py.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import pyvista as pv
dirname = "npy_eval"

# Same name map / row order as plot_radar_per_material.py
NAME_MAP: Dict[str, str] = {
    "ecg_heart_8L_256d_49frames":                  "CNN",
    "ecg_heart_tech_8L_256d_49frames":             "TeCh",
    "ecg_heart_physiowavenopool_8L_256d_49frames": "PhysioWave",
    "ecg_heart_shift_invariant_8L_256d_49frames":  "ShiftInvariant",
    "ecg_heart_delay_8L_256d_49frames":            "Delay-T",
    "ecg_heart_delayqrs_8L_256d_49frames":         "Delay-QRS",
    "ecg_heart_delaywavev3_8L_256d_49frames":      "Ours",
}
ROW_ORDER = ["CNN", "TeCh", "PhysioWave", "ShiftInvariant",
             "Delay-T", "Delay-QRS", "Ours"]


def collect_pairs(root: str, exp_filter: Optional[set],
                  max_subjects: Optional[int]) -> Dict[str, Dict[str, Dict[str, str]]]:
    results: Dict[str, Dict[str, Dict[str, str]]] = {}
    exp_dirs = sorted(glob.glob(os.path.join(root, "*", f"{dirname}/")))
    for d in exp_dirs:
        exp = d.split("/")[-3]
        if exp_filter is not None and exp not in exp_filter:
            continue
        try:
            entries = sorted(e.name for e in os.scandir(d) if e.name.endswith("_pred.npy"))
        except FileNotFoundError:
            continue
        if max_subjects is not None:
            entries = entries[:max_subjects]
        subjects: Dict[str, Dict[str, str]] = {}
        for fname in entries:
            sid = fname[:-9]
            p_path = os.path.join(d, fname)
            g_path = os.path.join(d, f"{sid}_gt.npy")
            if os.path.exists(g_path):
                subjects[sid] = {"pred": p_path, "gt": g_path}
        if subjects:
            results[exp] = subjects
    return results


def accumulate_trajectories(paths_by_sid: Dict[str, Dict[str, str]],
                            vertex_idx: np.ndarray,
                            which: str
                            ) -> Tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, int]:
    """Return (disp_mean, disp_std, speed_mean, speed_std, n) across subjects.

    For each subject we subtract the per-vertex temporal mean so the trajectories
    represent displacement-from-rest (otherwise the across-subject mean is just
    a soup of anatomy offsets and the std swamps the motion signal).

    Speed is ||v_t||_2 of the per-step displacement.
    Shapes: disp_*[T, V, 3], speed_*[T-1, V].
    """
    disp_sum: Optional[np.ndarray] = None
    disp_sum2: Optional[np.ndarray] = None
    spd_sum: Optional[np.ndarray] = None
    spd_sum2: Optional[np.ndarray] = None
    n = 0
    for sid, paths in tqdm(sorted(paths_by_sid.items()),
                           desc=which, mininterval=5.0, leave=False):
        arr = np.load(paths[which])  # [T, V, 3]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            continue
        sub = arr[:, vertex_idx, :].astype(np.float64)
        sub = sub - sub.mean(axis=0, keepdims=True)
        vel = np.diff(sub, axis=0)                     # [T-1, V, 3]
        speed = np.linalg.norm(vel, axis=-1)           # [T-1, V]
        if disp_sum is None:
            disp_sum = np.zeros_like(sub)
            disp_sum2 = np.zeros_like(sub)
            spd_sum = np.zeros_like(speed)
            spd_sum2 = np.zeros_like(speed)
        disp_sum += sub
        disp_sum2 += sub * sub
        spd_sum += speed
        spd_sum2 += speed * speed
        n += 1
    if n == 0 or disp_sum is None:
        V = vertex_idx.size
        return (np.zeros((0, V, 3)), np.zeros((0, V, 3)),
                np.zeros((0, V)), np.zeros((0, V)), 0)
    disp_mean = disp_sum / n
    disp_var = np.clip(disp_sum2 / n - disp_mean * disp_mean, 0.0, None)
    disp_std = np.sqrt(disp_var)
    spd_mean = spd_sum / n
    spd_var = np.clip(spd_sum2 / n - spd_mean * spd_mean, 0.0, None)
    spd_std = np.sqrt(spd_var)
    return disp_mean, disp_std, spd_mean, spd_std, n


def pick_vertices(num: int, total_v: int, seed: int = 0,
                  explicit: Optional[List[int]] = None) -> np.ndarray:
    if explicit:
        return np.array(explicit, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_v, size=min(num, total_v), replace=False))


def infer_total_v(results: Dict[str, Dict[str, Dict[str, str]]]) -> int:
    for subs in results.values():
        for paths in subs.values():
            arr = np.load(paths["gt"], mmap_mode="r")
            return int(arr.shape[1])
    return 0


def _color_for(name: str, slot: int):
    if name == "GT":
        return "k"
    if name == "Ours":
        return "#d62728"
    return plt.get_cmap("tab10")(slot % 10)

def load_mesh(mesh_path: str):
    m = trimesh.load(mesh_path, process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def plot_heart_mesh_with_vertex(vertices: np.ndarray,
                                faces: np.ndarray,
                                vertex_id: int,
                                out_path: str,
                                title: str = "Heart mesh with selected vertex") -> None:
    """Plot a triangular heart mesh and highlight one vertex with a red star."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if vertex_id < 0 or vertex_id >= vertices.shape[0]:
        raise IndexError(f"vertex_id {vertex_id} is out of range [0, {vertices.shape[0]})")

    point = vertices[vertex_id]
    fig = plt.figure(figsize=(3.0, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[faces], facecolor=(0.3, 0.3, 0.3, 0.1),
                             edgecolor=(0.35, 0.35, 0.35, 0.6), linewidths=0)
    ax.add_collection3d(mesh)

    ax.scatter([point[0]], [point[1]], [point[2]], color="red",
               marker="*", s=400, edgecolor="k", zorder=10)
    # ax.text(point[0], point[1], point[2], f" {vertex_id}", color="red", fontsize=10)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max()
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_vertex_row(vertex_id: int,
                    panels: List[Tuple[str, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]],
                    out_path: str,
                    std_stride: int = 2):
    """panels: list of (name, disp_mean[T,3], disp_std[T,3],
                        speed_mean[T-1], speed_std[T-1]); GT last."""
    n = len(panels)
    fig = plt.figure(figsize=(2.6 * n, 5.0))

    # Shared 3D bounds: include the std-cross extents so std bars are visible
    # without falling off the axes.
    all_lo = []
    all_hi = []
    for _, dm, ds, _, _ in panels:
        all_lo.append((dm - ds).min(axis=0))
        all_hi.append((dm + ds).max(axis=0))
    lo = np.minimum.reduce(all_lo)
    hi = np.maximum.reduce(all_hi)
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad

    # Shared speed-y bounds across the row
    spd_hi = max(float((sm + ss).max()) for _, _, _, sm, ss in panels)
    spd_hi = spd_hi * 1.05 + 1e-9

    for i, (name, dm, ds, sm, ss) in enumerate(panels, start=1):
        color = _color_for(name, i)
        is_emph = name in ("Ours", "GT")

        # ---- top: 3D displacement trajectory with std cross ----
        ax = fig.add_subplot(2, n, i, projection="3d")
        ax.plot(dm[:, 0], dm[:, 1], dm[:, 2],
                color=color, lw=2.2 if is_emph else 1.5)
        T = dm.shape[0]
        ts = list(range(0, T, max(1, std_stride)))
        if (T - 1) not in ts:
            ts.append(T - 1)
        for t in ts:
            for axis in range(3):
                p_lo = dm[t].copy(); p_hi = dm[t].copy()
                p_lo[axis] -= ds[t, axis]; p_hi[axis] += ds[t, axis]
                ax.plot([p_lo[0], p_hi[0]],
                        [p_lo[1], p_hi[1]],
                        [p_lo[2], p_hi[2]],
                        color=color, alpha=0.55, lw=0.9)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        title = r"$\bf{Ours}$" if name == "Ours" else name
        ax.set_title(title, fontsize=10)

        # ---- bottom: speed (||velocity||) over time, mean ± std ----
        ax2 = fig.add_subplot(2, n, n + i)
        t = np.arange(sm.shape[0])
        ax2.fill_between(t, np.clip(sm - ss, 0, None), sm + ss,
                         color=color, alpha=0.22, linewidth=0)
        ax2.plot(t, sm, color=color, lw=2.0 if is_emph else 1.4)
        ax2.set_ylim(0, spd_hi)
        ax2.set_xlim(t.min(), t.max() if t.size else 1)
        ax2.tick_params(labelsize=7)
        if i == 1:
            ax2.set_ylabel("speed", fontsize=8)
            ax2.set_xlabel("t", fontsize=8)
        else:
            ax2.set_yticklabels([])
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f"vertex {vertex_id}  (mean ± std across subjects)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--out", default="outputs/vertex_3d_trajectories")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="Optional cap on subjects per experiment")
    ap.add_argument("--num-vertices", type=int, default=12,
                    help="How many vertices to plot when --vertices not given")
    ap.add_argument("--vertices", type=int, nargs="*", default=None,
                    help="Explicit vertex indices to plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mesh", default="../ukb_heart_mesh_visual/control_face.ply")
    ap.add_argument("--subj", type=str, default=None)

    args = ap.parse_args()
    
    # verts, faces = load_mesh(args.mesh)
    # plot_heart_mesh_with_vertex(verts, faces, 247, 'heart_mesh_with_vertex_247.svg', '')
    # exit()
    if args.subj is not None: args.out = args.out + '_' + args.subj
    os.makedirs(args.out, exist_ok=True)

    exp_filter = set(NAME_MAP.keys())
    results = collect_pairs(args.root, exp_filter, args.max_subjects)
    
    if not results:
        print("no experiments found")
        return
    if args.subj is not None: 
        results = {exp: {sid: v for sid, v in results[exp].items() if sid == args.subj} for exp in results}
    
    total_v = infer_total_v(results)
    if total_v == 0:
        print("could not infer vertex count")
        return
    vertex_ids = pick_vertices(args.num_vertices, total_v, args.seed, args.vertices)
    print(f"plotting vertices: {vertex_ids.tolist()}")

    # compute per-experiment mean/std for selected vertices
    exp_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for exp, subs in results.items():
        display = NAME_MAP.get(exp, exp)
        dm, ds, sm, ss, n = accumulate_trajectories(subs, vertex_ids, "pred")
        if n == 0:
            continue
        exp_stats[display] = (dm, ds, sm, ss)
        print(f"  {display}: n={n} subjects")

    first_exp = next(iter(results))
    gt_dm, gt_ds, gt_sm, gt_ss, n_gt = accumulate_trajectories(
        results[first_exp], vertex_ids, "gt"
    )
    print(f"  GT (from {NAME_MAP.get(first_exp, first_exp)}): n={n_gt} subjects")

    ordered = [d for d in ROW_ORDER if d in exp_stats]
    for vi, vid in enumerate(vertex_ids):
        panels = []
        for display in ordered:
            dm, ds, sm, ss = exp_stats[display]
            panels.append((display, dm[:, vi, :], ds[:, vi, :],
                           sm[:, vi], ss[:, vi]))
        panels.append(("GT", gt_dm[:, vi, :], gt_ds[:, vi, :],
                       gt_sm[:, vi], gt_ss[:, vi]))
        out_path = os.path.join(args.out, f"vertex_{int(vid):05d}.svg")
        plot_vertex_row(int(vid), panels, out_path)

    print(f"wrote {len(vertex_ids)} svg files to {args.out}/")


if __name__ == "__main__":
    main()
