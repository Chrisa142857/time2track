"""Average Pairwise Distance (APD) across subjects on per-vertex time series.

**Speed APD** (default): temporal mean removal, per-step speed ``||Δx||`` →
``[N, T-1, V]``, then ``APD_v = mean_{i<j} ||speed_i[:,v] - speed_j[:,v]||_2``.

**Location APD** (``--apd-modes location`` or ``both``): same mean-centered
``[T, V, 3]`` displacements as ``plot_vertex_3d_trajectories``, flatten each
vertex trajectory over frames to ``R^{3T}``, then the same pairwise L2 across
subjects per vertex.

Each CSV row lists per-vertex **mean pairwise distance** and **population std**
of those pairwise distances (columns ``Method`` and ``Method_std``; same for
``GT``). Surface plots still use the mean only.

Writes CSV + surface SVG per mode (see ``--out``). Optional GPU:
``--apd-device cuda`` (or ``auto``) uses PyTorch for the pairwise loop.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[misc, assignment]

import matplotlib.pyplot as plt
from matplotlib import cm

from plot_heart_surface_scores import (
    NAME_MAP,
    ROW_ORDER,
    VIEWS,
    load_mesh,
    _to_goodness,
    _face_colors,
    _draw_view,
    _cbar_ticks,
)

DIRNAME = "npy_eval"


def _progress(it, *, desc: str, total: int, unit: str = "it"):
    """tqdm when available; otherwise periodic prints on stderr."""
    if tqdm is not None:
        yield from tqdm(it, desc=desc, total=total, mininterval=0.25,
                        leave=False, unit=unit)
        return
    step = max(1, total // 25) if total else 1
    for i, x in enumerate(it, start=1):
        if i == 1 or i == total or i % step == 0:
            print(f"  [{desc}] {i}/{total}", flush=True)
        yield x


def collect_pairs(root: str, exp_filter: set,
                  max_subjects: Optional[int]) -> Dict[str, Dict[str, Dict[str, str]]]:
    results: Dict[str, Dict[str, Dict[str, str]]] = {}
    for d in sorted(glob.glob(os.path.join(root, "*", f"{DIRNAME}/"))):
        exp = d.split("/")[-3]
        if exp not in exp_filter:
            continue
        try:
            entries = sorted(e.name for e in os.scandir(d) if e.name.endswith("_pred.npy"))
        except FileNotFoundError:
            continue
        if max_subjects is not None:
            entries = entries[:max_subjects]
        subs: Dict[str, Dict[str, str]] = {}
        for f in entries:
            sid = f[:-9]
            p = os.path.join(d, f)
            g = os.path.join(d, f"{sid}_gt.npy")
            if os.path.exists(g):
                subs[sid] = {"pred": p, "gt": g}
        if subs:
            results[exp] = subs
    return results


def load_speeds(paths_by_sid: Dict[str, Dict[str, str]], which: str,
                desc: str = "") -> np.ndarray:
    """Stack speeds across subjects -> [N, T-1, V]."""
    items = sorted(paths_by_sid.items())
    label = desc or which
    out: List[np.ndarray] = []
    for sid, paths in _progress(items, desc=f"{label} ({which})",
                                total=len(items), unit="subj"):
        arr = np.load(paths[which])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            continue
        x = arr.astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        vel = np.diff(x, axis=0)
        spd = np.linalg.norm(vel, axis=-1)  # [T-1, V]
        out.append(spd)
    if not out:
        return np.zeros((0, 0, 0))
    return np.stack(out, axis=0)


def load_location_flat(paths_by_sid: Dict[str, Dict[str, str]], which: str,
                       desc: str = "") -> np.ndarray:
    """Stack mean-centered vertex trajectories flattened over frames -> [N, 3*T, V].

    Same temporal mean removal per vertex as ``plot_vertex_3d_trajectories``;
    each vertex becomes a length ``3*T`` vector (frame-major xyz).
    """
    items = sorted(paths_by_sid.items())
    label = desc or which
    out: List[np.ndarray] = []
    for sid, paths in _progress(items, desc=f"{label} loc ({which})",
                                total=len(items), unit="subj"):
        arr = np.load(paths[which])
        if arr.ndim != 3 or arr.shape[-1] != 3:
            continue
        x = arr.astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        t, v, _ = x.shape
        # [V, T*3] row v = concat over t of x[t,v,:]
        flat = x.transpose(1, 0, 2).reshape(v, t * 3)
        out.append(flat)
    if not out:
        return np.zeros((0, 0, 0))
    # [N, V, T*3] -> [N, T*3, V] for apd_per_vertex
    return np.stack(out, axis=0).transpose(0, 2, 1)


def _resolve_apd_torch_device(apd_device: str):
    """Return ``torch.device`` for CUDA APD, or ``None`` to use NumPy on CPU."""
    s = (apd_device or "auto").lower().strip()
    if s in ("cpu", "numpy", "none"):
        return None
    if torch is None:
        if s == "auto":
            return None
        raise RuntimeError("PyTorch is not installed; use --apd-device cpu or install torch")
    if s == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else None
    if s == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested (--apd-device cuda) but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if s.startswith("cuda:"):
        d = torch.device(s)
        if d.type != "cuda":
            raise ValueError(f"not a CUDA device string: {apd_device!r}")
        return d
    raise ValueError(f"unknown --apd-device {apd_device!r} (use auto|cpu|cuda|cuda:N)")


def _apd_per_vertex_numpy(
    features: np.ndarray, desc: str, pair_chunk: int
) -> Tuple[np.ndarray, np.ndarray]:
    """features [N, D, V] -> (mean pairwise L2, pop. std of pairwise L2) per vertex."""
    N, _, V = features.shape
    if N < 2:
        z = np.zeros(V)
        return z, z
    sp = np.asarray(features, dtype=np.float64)
    sp = np.transpose(sp, (2, 0, 1))  # [V, N, D]
    n_pairs = N * (N - 1) // 2
    sum_d = np.zeros(V, dtype=np.float64)
    sum_d2 = np.zeros(V, dtype=np.float64)
    pc = max(1, int(pair_chunk))
    i_iter = _progress(range(N - 1), desc=f"{desc} (subj i)", total=N - 1, unit="i")
    for i in i_iter:
        row = sp[:, i : i + 1, :]
        j = i + 1
        while j < N:
            jn = min(j + pc, N)
            diff = sp[:, j:jn, :] - row
            d = np.linalg.norm(diff, axis=-1)
            sum_d += d.sum(axis=1)
            sum_d2 += np.square(d).sum(axis=1)
            j = jn
    mean_d = sum_d / n_pairs
    var = sum_d2 / n_pairs - mean_d * mean_d
    var = np.clip(var, 0.0, None)
    std_d = np.sqrt(var)
    return mean_d, std_d


def _apd_per_vertex_torch(
    features: np.ndarray,
    desc: str,
    pair_chunk: int,
    device,
    *,
    fp32_accum: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same math as NumPy path; returns (mean pairwise L2, pop. std) per vertex."""
    if torch is None:
        raise RuntimeError("torch required")
    N, _, V = features.shape
    if N < 2:
        z = np.zeros(V)
        return z, z
    n_pairs = N * (N - 1) // 2
    pc = max(1, int(pair_chunk))
    sp = torch.as_tensor(features, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()
    sum_dtype = torch.float32 if fp32_accum else torch.float64
    sum_d = torch.zeros(V, device=device, dtype=sum_dtype)
    sum_d2 = torch.zeros(V, device=device, dtype=sum_dtype)
    i_iter = _progress(range(N - 1), desc=f"{desc} (subj i, {device})", total=N - 1, unit="i")
    for i in i_iter:
        row = sp[:, i : i + 1, :]
        j = i + 1
        while j < N:
            jn = min(j + pc, N)
            diff = sp[:, j:jn, :] - row
            norms = torch.linalg.vector_norm(diff.to(sum_dtype), dim=-1)
            sum_d += norms.sum(dim=1).to(sum_dtype)
            sum_d2 += (norms * norms).sum(dim=1).to(sum_dtype)
            j = jn
    mean_d = sum_d / n_pairs
    var = sum_d2 / n_pairs - mean_d * mean_d
    var = torch.clamp(var, min=0)
    std_d = torch.sqrt(var)
    return (
        mean_d.detach().cpu().numpy().astype(np.float64, copy=False),
        std_d.detach().cpu().numpy().astype(np.float64, copy=False),
    )


def apd_per_vertex(
    features: np.ndarray,
    desc: str = "APD/vertex",
    pair_chunk: int = 4096,
    apd_device: str = "auto",
    apd_fp32_accum: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """``features`` shaped ``[N, D, V]`` (e.g. speed with ``D=T-1`` or location with ``D=3*T``).

    Returns ``(mean_apd, std_apd)`` per vertex. ``mean_apd[v]`` is
    ``mean_{i<j} ||features[i,:,v] - features[j,:,v]||_2``; ``std_apd[v]`` is
    the population standard deviation of those ``\\binom{N}{2}`` pairwise
    distances.

    Pairwise sums chunk over subject ``j`` (``pair_chunk``); peak memory
    ``O(V * pair_chunk * D)``. **GPU:** ``apd_device='cuda'`` or ``'auto'``.
    """
    tdev = _resolve_apd_torch_device(apd_device)
    if tdev is None:
        return _apd_per_vertex_numpy(features, desc, pair_chunk)
    return _apd_per_vertex_torch(
        features, desc, pair_chunk, tdev, fp32_accum=apd_fp32_accum
    )


def render_apd_surface(verts: np.ndarray, faces: np.ndarray,
                       per_method: List[Tuple[str, np.ndarray]],
                       out_path: str, cmap_name: str = "viridis",
                       *,
                       cbar_label: str = "APD of speed  (top = better)",
                       suptitle: str = "Average Pairwise Distance of per-vertex speed",
                       save_subplots: bool = True):
    if not per_method:
        return
    pooled_g: List[float] = []
    direction = "lower"
    for _, vals in per_method:
        face_vals = vals[faces].mean(axis=1)
        pooled_g.extend(
            float(x) for x in _to_goodness(face_vals, direction) if np.isfinite(x)
        )
    if not pooled_g:
        return
    lo_g = float(np.percentile(pooled_g, 2))
    hi_g = float(np.percentile(pooled_g, 98))
    if hi_g <= lo_g:
        hi_g = lo_g + 1.0

    def _sanitize_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

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

    subplot_metadata: List[Tuple[str, str, np.ndarray, int, int]] = []
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
            subplot_metadata.append((display, vname, fc, elev, azim))

    cax = fig.add_subplot(gs[:, -1])
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lo_g, vmax=hi_g))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    ticks, labels = _cbar_ticks(direction, lo_g, hi_g)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=10)

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if save_subplots:
        base_path, _ = os.path.splitext(out_path)
        for display, vname, fc, elev, azim in subplot_metadata:
            fig2 = plt.figure(figsize=(2.4, 2.4))
            ax2 = fig2.add_subplot(111, projection="3d")
            _draw_view(ax2, verts, faces, fc, elev, azim, ctr, span)
            ax2.set_title("")
            # title = r"$\bf{Ours}$" if display == "Ours" else display
            # ax2.set_title(title, fontsize=11)
            # ax2.text2D(0.01, 0.01, vname, transform=ax2.transAxes,
            #            va="bottom", ha="left", fontsize=9)
            subplot_name = f"{base_path}_{_sanitize_name(display)}_{_sanitize_name(vname)}.svg"
            fig2.savefig(subplot_name, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig2)


def _rmse_per_vertex(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-vertex RMSE across subjects and its population std deviation."""
    if pred.size == 0 or gt.size == 0:
        return np.zeros(pred.shape[-1] if pred.ndim >= 3 else 0), np.zeros(pred.shape[-1] if pred.ndim >= 3 else 0)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}"
        )
    # [N, D, V] -> [N, V] RMSE per vertex per subject
    rmse_by_subject = np.sqrt(np.mean(np.square(pred - gt), axis=1))
    return rmse_by_subject.mean(axis=0), rmse_by_subject.std(axis=0)


def run_rmse_pipeline(
    *,
    results: Dict[str, Dict[str, Dict[str, str]]],
    first_exp: str,
    load_fn,
    data_label: str,
    out_dir: str,
    csv_name: str,
    svg_name: str,
    surface_cbar: str,
    surface_title: str,
    mesh_path: str,
) -> None:
    """Load per-method + GT features, RMSE per vertex, CSV + surface SVG."""
    rmse_by_method: Dict[str, np.ndarray] = {}
    rmse_std_by_method: Dict[str, np.ndarray] = {}
    gt_root = results[first_exp]

    for exp, subs in results.items():
        display = NAME_MAP[exp]
        common_ids = sorted(set(subs).intersection(gt_root))
        if not common_ids:
            print(f"  {display}: no shared subjects with GT, skip")
            continue
        if len(common_ids) != len(subs):
            print(
                f"  {display}: using {len(common_ids)} shared subjects "
                f"(skipping {len(subs) - len(common_ids)} unmatched)",
                flush=True,
            )
        method_subs = {sid: subs[sid] for sid in common_ids}
        gt_subs = {sid: gt_root[sid] for sid in common_ids}

        print(f"  {display}: loading {data_label}…", flush=True)
        pred = load_fn(method_subs, "pred", desc=f"{display} pred")
        gt = load_fn(gt_subs, "gt", desc="GT")
        if pred.size == 0 or gt.size == 0:
            continue
        mean_v, std_v = _rmse_per_vertex(pred, gt)
        rmse_by_method[display] = mean_v
        rmse_std_by_method[display] = std_v
        print(
            f"  {display}: N={pred.shape[0]} RMSE mean={mean_v.mean():.4g}  "
            f"std mean={std_v.mean():.4g}",
            flush=True,
        )

    if not rmse_by_method:
        print(f"  skip {data_label}: no method data")
        return

    ordered = [d for d in ROW_ORDER if d in rmse_by_method]
    header = ["vertex"]
    for c in ordered:
        header.extend([c, f"{c}_std"])
    v_count = next(iter(rmse_by_method.values())).shape[0]
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for v in range(v_count):
            row = [v]
            for c in ordered:
                row.append(f"{rmse_by_method[c][v]:.6g}")
                row.append(f"{rmse_std_by_method[c][v]:.6g}")
            w.writerow(row)
    print(f"wrote {csv_path}")

    verts, faces = load_mesh(mesh_path)
    if verts.shape[0] != v_count:
        print(f"warn: mesh V={verts.shape[0]} but RMSE V={v_count}; skip {svg_name}")
        return
    per_method: List[Tuple[str, np.ndarray]] = [(c, rmse_by_method[c]) for c in ordered]
    out_svg = os.path.join(out_dir, svg_name)
    render_apd_surface(
        verts, faces, per_method, out_svg,
        cbar_label=surface_cbar,
        suptitle=surface_title,
    )
    print(f"wrote {out_svg}")


def run_apd_pipeline(
    *,
    results: Dict[str, Dict[str, Dict[str, str]]],
    first_exp: str,
    load_fn,
    data_label: str,
    out_dir: str,
    csv_name: str,
    svg_name: str,
    surface_cbar: str,
    surface_title: str,
    mesh_path: str,
    pair_chunk: int,
    apd_device: str,
    apd_fp32_accum: bool,
) -> None:
    """Load per-method + GT features, APD per vertex, CSV + optional surface SVG."""
    apd_by_method: Dict[str, np.ndarray] = {}
    apd_std_by_method: Dict[str, np.ndarray] = {}
    for exp, subs in results.items():
        display = NAME_MAP[exp]
        print(f"  {display}: loading {data_label}…", flush=True)
        data = load_fn(subs, "pred", desc=display)
        if data.size == 0:
            continue
        print(f"  {display}: pairwise APD ({data_label})…", flush=True)
        mean_v, std_v = apd_per_vertex(
            data,
            desc=f"{display} {data_label}",
            pair_chunk=pair_chunk,
            apd_device=apd_device,
            apd_fp32_accum=apd_fp32_accum,
        )
        apd_by_method[display] = mean_v
        apd_std_by_method[display] = std_v
        print(
            f"  {display}: N={data.shape[0]} APD mean={mean_v.mean():.4g}  "
            f"pairwise-std mean={std_v.mean():.4g}",
            flush=True,
        )

    print(f"  GT: loading {data_label}…", flush=True)
    gt_data = load_fn(results[first_exp], "gt", desc="GT")
    apd_gt: Optional[np.ndarray] = None
    apd_gt_std: Optional[np.ndarray] = None
    if gt_data.size:
        print(f"  GT: pairwise APD ({data_label})…", flush=True)
        apd_gt, apd_gt_std = apd_per_vertex(
            gt_data,
            desc=f"GT {data_label}",
            pair_chunk=pair_chunk,
            apd_device=apd_device,
            apd_fp32_accum=apd_fp32_accum,
        )
        print(
            f"  GT: N={gt_data.shape[0]} APD mean={apd_gt.mean():.4g}  "
            f"pairwise-std mean={apd_gt_std.mean():.4g}",
            flush=True,
        )

    if not apd_by_method:
        print(f"  skip {data_label}: no method data")
        return

    ordered = [d for d in ROW_ORDER if d in apd_by_method]
    header = ["vertex"]
    for c in ordered:
        header.extend([c, f"{c}_std"])
    if apd_gt is not None and apd_gt_std is not None:
        header.extend(["GT", "GT_std"])
    v_count = next(iter(apd_by_method.values())).shape[0]
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for v in range(v_count):
            row = [v]
            for c in ordered:
                row.append(f"{apd_by_method[c][v]:.6g}")
                row.append(f"{apd_std_by_method[c][v]:.6g}")
            if apd_gt is not None and apd_gt_std is not None:
                row.append(f"{apd_gt[v]:.6g}")
                row.append(f"{apd_gt_std[v]:.6g}")
            w.writerow(row)
    print(f"wrote {csv_path}")

    verts, faces = load_mesh(mesh_path)
    if verts.shape[0] != v_count:
        print(f"warn: mesh V={verts.shape[0]} but APD V={v_count}; skip {svg_name}")
        return
    per_method: List[Tuple[str, np.ndarray]] = [(c, apd_by_method[c]) for c in ordered]
    if apd_gt is not None:
        per_method.append(("GT", apd_gt))
    out_svg = os.path.join(out_dir, svg_name)
    render_apd_surface(
        verts, faces, per_method, out_svg,
        cbar_label=surface_cbar,
        suptitle=surface_title,
    )
    print(f"wrote {out_svg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--mesh", default="../ukb_heart_mesh_visual/control_face.ply")
    ap.add_argument("--out", default="outputs/vertex_apd")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument(
        "--apd-pair-chunk",
        type=int,
        default=4096,
        help="j-chunk size inside APD (smaller uses less GPU/CPU RAM; default 4096)",
    )
    ap.add_argument(
        "--apd-device",
        default="auto",
        help="APD backend: auto (PyTorch CUDA if available else NumPy CPU), cpu, cuda, cuda:N",
    )
    ap.add_argument(
        "--apd-fp32-accum",
        action="store_true",
        help="On GPU, accumulate norm sums in float32 (faster; default uses float64)",
    )
    ap.add_argument(
        "--apd-modes",
        nargs="+",
        choices=("speed", "location"),
        default=["speed"],
        help="speed: APD of ||Δx||; location: APD of mean-centered xyz over frames (R^{3T}); "
        "pass both e.g. `--apd-modes speed location`",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        choices=("apd", "rmse"),
        default=["apd"],
        help="speed: APD of ||Δx||; location: APD of mean-centered xyz over frames (R^{3T}); "
        "pass both e.g. `--apd-modes speed location`",
    )
    args = ap.parse_args()

    apd_dev = _resolve_apd_torch_device(args.apd_device)
    if apd_dev is not None:
        print(f"APD backend: PyTorch {apd_dev}", flush=True)
    else:
        print("APD backend: NumPy (CPU)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    exp_filter = set(NAME_MAP.keys())
    results = collect_pairs(args.root, exp_filter, args.max_subjects)
    if not results:
        print("no experiments found")
        return

    first_exp = next(iter(results))
    modes = list(dict.fromkeys(args.apd_modes))  # preserve order, dedupe

    metrics = list(dict.fromkeys(args.metrics))  # preserve order, dedupe

    if "speed" in modes:
        print("--- APD (speed) ---", flush=True)
        if "rmse" in metrics:
            run_rmse_pipeline(
                results=results,
                first_exp=first_exp,
                load_fn=load_speeds,
                data_label="speeds",
                out_dir=args.out,
                csv_name="rmse_per_vertex.csv",
                svg_name="rmse_surface.svg",
                surface_cbar="RMSE of speed  (lower = better)",
                surface_title="Root Mean Square Error of per-vertex speed",
                mesh_path=args.mesh,
            )
        if "apd" in metrics:
            run_apd_pipeline(
                results=results,
                first_exp=first_exp,
                load_fn=load_speeds,
                data_label="speeds",
                out_dir=args.out,
                csv_name="apd_per_vertex.csv",
                svg_name="apd_surface.svg",
                surface_cbar="APD of speed  (top = better)",
                surface_title="Average Pairwise Distance of per-vertex speed",
                mesh_path=args.mesh,
                pair_chunk=args.apd_pair_chunk,
                apd_device=args.apd_device,
                apd_fp32_accum=args.apd_fp32_accum,
            )

    if "location" in modes:
        print("--- APD (location over frames) ---", flush=True)
        if "rmse" in metrics:
            run_rmse_pipeline(
                results=results,
                first_exp=first_exp,
                load_fn=load_location_flat,
                data_label="locations",
                out_dir=args.out,
                csv_name="rmse_location_per_vertex.csv",
                svg_name="rmse_location_surface.svg",
                surface_cbar="RMSE of location trajectory  (lower = better)",
                surface_title="Root Mean Square Error of per-vertex location (over frames)",
                mesh_path=args.mesh,
            )
        if "apd" in metrics:
            run_apd_pipeline(
                results=results,
                first_exp=first_exp,
                load_fn=load_location_flat,
                data_label="locations",
                out_dir=args.out,
                csv_name="apd_location_per_vertex.csv",
                svg_name="apd_location_surface.svg",
                surface_cbar="APD of location trajectory  (top = better)",
                surface_title="Average Pairwise Distance of per-vertex location (over frames)",
                mesh_path=args.mesh,
                pair_chunk=args.apd_pair_chunk,
                apd_device=args.apd_device,
                apd_fp32_accum=args.apd_fp32_accum,
            )


if __name__ == "__main__":
    main()
