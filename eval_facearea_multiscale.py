"""Evaluate ECG -> 4D-heart-mesh predictions by comparing face-area curves
at multiple scales: whole heart, per material, and per vertex.

Reads <root>/<exp>/{dirname}/<sid>_pred.npy and ..._gt.npy (each of shape
[T, V, 3]), computes per-frame face areas, then reports metrics per experiment
averaged over subjects.

Output scales:
  - whole        : total surface area of entire mesh per frame
  - material_*   : total surface area per material (LV_ENDOCARDIAL, etc.)
  - vertex       : per-vertex face area curves (cached as [T, V] npy arrays)

Usage
-----
  python eval_facearea_multiscale.py --root /path/to/outputs \
         --mesh-info ../ukb_heart_mesh_visual
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import tempfile

# Force multiprocessing temp dirs onto local disk; cleaning up pymp-* under
# tmpfs/RAM mounts can race with worker FDs and raise "Device or resource busy".
os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = os.environ["TMPDIR"]

import multiprocessing as _mp
from typing import Dict, Optional, Tuple  # noqa: E402  (used by _PROJ_CACHE annotation below)

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_MP_CTX = _mp.get_context("spawn")

dirname = 'npy_eval'
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float32
EPS = 1e-9
_PROJ_CACHE: Dict[int, torch.Tensor] = {}


def _to_tensor(arr) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device=_DEVICE, dtype=_DTYPE)
    return torch.as_tensor(np.asarray(arr), dtype=_DTYPE, device=_DEVICE)


def _to_tensor_f64(arr) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device=_DEVICE, dtype=torch.float64)
    return torch.as_tensor(np.asarray(arr), dtype=torch.float64, device=_DEVICE)


def _shape_guard(name: str, t: torch.Tensor, max_elems: int = 5_000_000_000):
    n = t.numel() if isinstance(t, torch.Tensor) else int(np.prod(t))
    if n > max_elems:
        raise RuntimeError(f"[shape-guard] {name} would allocate {n} elements, shape={tuple(t.shape)}")


def load_mesh_info(mesh_info_dir: str):
    """Load mesh info and return mesh plus precomputed material face groups."""
    with open(os.path.join(mesh_info_dir, "control_material_idmap.json")) as f:
        idmap = json.load(f)
    materials = np.load(os.path.join(mesh_info_dir, "control_materials.npy"))
    mesh = trimesh.load(os.path.join(mesh_info_dir, "control_face.ply"))
    face_materials = materials[mesh.faces[:, 0]]
    material_to_id = {v: k for k, v in idmap.items()}
    material_groups = []
    for mid in sorted(set(materials) & set(material_to_id.keys())):
        face_idx = np.flatnonzero(face_materials == mid)
        material_groups.append((f"material_{material_to_id[mid]}", face_idx))
    return mesh, material_groups


class _MeshContext:
    """Precompute face/material indices on GPU once."""
    def __init__(self, mesh, material_groups):
        self.faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long, device=_DEVICE)
        self.material_groups = []
        for mname, face_idx in material_groups:
            idx_t = (torch.as_tensor(face_idx, dtype=torch.long, device=_DEVICE)
                     if face_idx.size > 0 else None)
            self.material_groups.append((mname, idx_t))


def compute_frame_areas_batch(verts_b: torch.Tensor, ctx: _MeshContext):
    """verts_b: [B, T, V, 3] -> whole [B, T], mat_areas {name: [B, T]}, vertex_areas [B, T, V]."""
    verts_b = verts_b.to(device=_DEVICE, dtype=_DTYPE, non_blocking=True)
    B, T, V, _ = verts_b.shape
    faces = ctx.faces
    F = faces.shape[0]

    v0 = verts_b[:, :, faces[:, 0], :]
    v1 = verts_b[:, :, faces[:, 1], :]
    v2 = verts_b[:, :, faces[:, 2], :]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_areas_all = 0.5 * torch.linalg.norm(cross, dim=-1)  # [B, T, F]

    whole = face_areas_all.sum(dim=2)

    mat_areas = {}
    for mname, idx_t in ctx.material_groups:
        if idx_t is not None:
            mat_areas[mname] = face_areas_all[:, :, idx_t].sum(dim=2)
        else:
            mat_areas[mname] = torch.full((B, T), float("nan"),
                                          dtype=_DTYPE, device=_DEVICE)

    vertex_areas = torch.zeros((B, T, V), dtype=_DTYPE, device=_DEVICE)
    share = face_areas_all / 3.0
    for corner in range(3):
        idx = faces[:, corner].view(1, 1, F).expand(B, T, F)
        vertex_areas.scatter_add_(2, idx, share)

    return whole, mat_areas, vertex_areas


# ---------- Dataset / DataLoader for parallel I/O ----------

class _VertsDataset(Dataset):
    """Yields (kind, exp, sid, verts_tensor) for each task."""
    def __init__(self, tasks):
        self.tasks = tasks  # list of (kind, exp, sid, path)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        kind, exp, sid, path = self.tasks[i]
        arr = np.load(path)
        return kind, exp, sid, torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _verts_collate(batch):
    kinds = [b[0] for b in batch]
    exps = [b[1] for b in batch]
    sids = [b[2] for b in batch]
    verts = torch.stack([b[3] for b in batch], dim=0)
    return kinds, exps, sids, verts


def _enumerate_tasks(root, subject_list=None):
    """Return (pred_tasks, gt_tasks) lists; gt tasks dedup'd by sid."""
    pred_tasks = []
    gt_tasks = []
    seen_gt = {}
    exp_dirs = sorted(glob.glob(os.path.join(root, "*", dirname, "")))
    for d in exp_dirs:
        exp = d.split("/")[-3]
        try:
            entries = sorted(e.name for e in os.scandir(d) if e.name.endswith("_pred.npy"))
        except FileNotFoundError:
            continue
        if subject_list is not None:  # restrict to a subgroup
            entries = [e for e in entries if e[:-9] in subject_list]
        for fname in entries:
            sid = fname[:-9]
            p_path = os.path.join(d, fname)
            g_path = os.path.join(d, f"{sid}_gt.npy")
            if not os.path.exists(g_path):
                continue
            pred_tasks.append(("pred", exp, sid, p_path))
            if sid not in seen_gt:
                seen_gt[sid] = g_path
                gt_tasks.append(("gt", "", sid, g_path))
    return pred_tasks, gt_tasks


def _store_batch(results, gt_area_cache, kinds, exps, sids, whole, mat_areas, vertex_areas):
    whole_np = whole.detach().cpu().numpy()
    mat_np = {k: v.detach().cpu().numpy() for k, v in mat_areas.items()}
    va_np = vertex_areas.detach().cpu().numpy()
    for i, (kind, exp, sid) in enumerate(zip(kinds, exps, sids)):
        data = {"whole": whole_np[i]}
        for k, v in mat_np.items():
            data[k] = v[i]
        data["vertex_areas"] = va_np[i]
        if kind == "gt":
            gt_area_cache[sid] = data
        else:
            subs = results.setdefault(exp, {})
            subs[sid] = {"pred": data, "gt": None}


def collect_areas(root, mesh, material_groups, batch_size=32, num_workers=8, subject_list=None):
    """Parallel I/O via DataLoader, batched GPU compute via compute_frame_areas_batch."""
    ctx = _MeshContext(mesh, material_groups)
    pred_tasks, gt_tasks = _enumerate_tasks(root, subject_list)

    results = {}
    gt_area_cache = {}

    def _run(tasks, desc):
        if not tasks:
            return
        loader = DataLoader(
            _VertsDataset(tasks),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=(_DEVICE.type == "cuda"),
            collate_fn=_verts_collate,
            persistent_workers=False,
            multiprocessing_context=_MP_CTX if num_workers > 0 else None,
        )
        for kinds, exps, sids, verts in tqdm(loader, desc=desc):
            whole, mat_areas, vertex_areas = compute_frame_areas_batch(verts, ctx)
            _store_batch(results, gt_area_cache, kinds, exps, sids,
                         whole, mat_areas, vertex_areas)

    _run(gt_tasks, f"GT areas (n={len(gt_tasks)})")
    _run(pred_tasks, f"Pred areas (n={len(pred_tasks)})")

    for exp, subs in results.items():
        for sid, subj in subs.items():
            subj["gt"] = gt_area_cache[sid]
        tqdm.write(f"  {exp}: {len(subs)} subjects")
    return results


def _vertex_cache_complete(results, varea_dir):
    for name, subs in results.items():
        for sid in subs.keys():
            pred_path = os.path.join(varea_dir, f"{name}_{sid}_pred_vertex.npy")
            shared_gt_path = os.path.join(varea_dir, f"{sid}_gt_vertex.npy")
            legacy_gt_path = os.path.join(varea_dir, f"{name}_{sid}_gt_vertex.npy")
            if not os.path.exists(pred_path):
                return False
            if not (os.path.exists(shared_gt_path) or os.path.exists(legacy_gt_path)):
                return False
    return True


def write_area_caches(results, cache_path, manifest_path, varea_dir):
    """Pack per-scale curves into one .npz (binary, dedup'd GT) plus a small JSON manifest."""
    scales = set()
    for subs in results.values():
        for subj in subs.values():
            scales.update(k for k in subj["pred"].keys() if k != "vertex_areas")
            scales.update(k for k in subj["gt"].keys() if k != "vertex_areas")
            break
        if scales:
            break
    scales = sorted(scales)

    pred_index = []
    gt_index = {}
    pred_rows = {s: [] for s in scales}
    gt_rows = {s: [] for s in scales}

    for exp, subs in results.items():
        for sid, subj in subs.items():
            pred_row = len(pred_index)
            pred_index.append([exp, sid, pred_row])
            for s in scales:
                pred_rows[s].append(np.asarray(subj["pred"][s], dtype=np.float32))
            if sid not in gt_index:
                gt_index[sid] = len(gt_index)
                for s in scales:
                    gt_rows[s].append(np.asarray(subj["gt"][s], dtype=np.float32))

    arrays = {}
    for s in scales:
        arrays[f"pred__{s}"] = np.stack(pred_rows[s], axis=0) if pred_rows[s] else np.zeros((0, 0), dtype=np.float32)
        arrays[f"gt__{s}"] = np.stack(gt_rows[s], axis=0) if gt_rows[s] else np.zeros((0, 0), dtype=np.float32)
    np.savez(cache_path, **arrays)
    print(f"wrote {cache_path} ({sum(a.nbytes for a in arrays.values()) / 1e6:.1f} MB)")

    with open(manifest_path, "w") as f:
        json.dump({
            "scales": scales,
            "pred_index": pred_index,
            "gt_index": [[sid, row] for sid, row in gt_index.items()],
        }, f)
    print(f"wrote {manifest_path}")

    print("Saving per-vertex area caches...")
    os.makedirs(varea_dir, exist_ok=True)
    saved_gt_sids = set()
    for name, subs in results.items():
        for sid, subj_data in subs.items():
            if "vertex_areas" in subj_data["pred"]:
                np.save(os.path.join(varea_dir, f"{name}_{sid}_pred_vertex.npy"),
                        subj_data["pred"]["vertex_areas"])
                if sid not in saved_gt_sids:
                    np.save(os.path.join(varea_dir, f"{sid}_gt_vertex.npy"),
                            subj_data["gt"]["vertex_areas"])
                    saved_gt_sids.add(sid)
    print(f"wrote per-vertex areas under {varea_dir}/")


def load_area_caches(cache_path, manifest_path):
    """Load .npz + manifest into the nested results dict expected by compute_metrics."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    scales = manifest["scales"]
    npz = np.load(cache_path)
    pred = {s: npz[f"pred__{s}"] for s in scales}
    gt = {s: npz[f"gt__{s}"] for s in scales}
    gt_row_by_sid = {sid: row for sid, row in manifest["gt_index"]}

    results = {}
    for exp, sid, row in manifest["pred_index"]:
        gt_row = gt_row_by_sid[sid]
        subj = {
            "pred": {s: pred[s][row] for s in scales},
            "gt": {s: gt[s][gt_row] for s in scales},
        }
        results.setdefault(exp, {})[sid] = subj
    return results


def _pearson(a, b):
    a = _to_tensor(a).reshape(-1)
    b = _to_tensor(b).reshape(-1)
    sa = a.std(unbiased=False)
    sb = b.std(unbiased=False)
    if float(sa.item()) == 0 or float(sb.item()) == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    return float(((a * b).mean() / (sa * sb)).item())


def _corrcoef_rows(curves: torch.Tensor) -> torch.Tensor:
    centered = curves - curves.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, unbiased=False)
    norm = centered / (std.unsqueeze(1) + EPS)
    n = curves.shape[1]
    return (norm @ norm.T) / n


def _pairwise_pearson_values(curves):
    curves = _to_tensor(curves)
    if curves.shape[0] < 2:
        return torch.tensor([], device=_DEVICE)
    C = _corrcoef_rows(curves)
    n = C.shape[0]
    iu_r, iu_c = torch.triu_indices(n, n, offset=1, device=_DEVICE).unbind(0)
    vals = C[iu_r, iu_c]
    return vals[torch.isfinite(vals)]


def _pairwise_rmse_values(curves, denom):
    curves = _to_tensor(curves)
    s = curves.shape[0]
    if s < 2 or denom <= 0:
        return torch.tensor([], device=_DEVICE)
    sq = (curves * curves).sum(dim=1)
    dot = curves @ curves.T
    d2 = torch.clamp(sq[:, None] + sq[None, :] - 2.0 * dot, min=0.0) / curves.shape[1]
    rmse = torch.sqrt(d2)
    iu_r, iu_c = torch.triu_indices(s, s, offset=1, device=_DEVICE).unbind(0)
    vals = rmse[iu_r, iu_c]
    return vals[torch.isfinite(vals)] / denom


def _mean_pairwise_pearson(curves):
    vals = _pairwise_pearson_values(curves)
    return float(vals.mean().item()) if vals.numel() else float("nan")


def _mean_pairwise_rmse(curves, denom):
    vals = _pairwise_rmse_values(curves, denom)
    return float(vals.mean().item()) if vals.numel() else float("nan")


def _amp_cv(curves):
    curves = _to_tensor(curves)
    if curves.shape[0] < 2:
        return float("nan")
    ptp = curves.amax(dim=1) - curves.amin(dim=1)
    m = ptp.mean()
    if float(m.item()) <= 0:
        return float("nan")
    return float((ptp.std(unbiased=False) / (m + EPS)).item())


def _detrend_proj(T: int) -> torch.Tensor:
    cached = _PROJ_CACHE.get(T)
    if cached is not None:
        return cached
    x = torch.arange(T, dtype=torch.float64, device=_DEVICE)
    V = torch.stack([x ** 3, x ** 2, x, torch.ones_like(x)], dim=1)  # [T, 4]
    proj = (V @ torch.linalg.pinv(V)).to(_DTYPE)                     # [T, T]
    _PROJ_CACHE[T] = proj
    return proj


def _detrend_poly3(curve: torch.Tensor) -> torch.Tensor:
    if curve.ndim != 1:
        raise ValueError(f"_detrend_poly3 expects 1-D; got {tuple(curve.shape)}")
    proj = _detrend_proj(curve.shape[0])
    return curve - proj @ curve


def _detrend_poly3_batch(curves: torch.Tensor) -> torch.Tensor:
    """curves: [V, T] -> detrended curves [V, T] (cubic OLS per row)."""
    if curves.ndim != 2:
        raise ValueError(f"_detrend_poly3_batch expects [V, T]; got {tuple(curves.shape)}")
    proj = _detrend_proj(curves.shape[1])
    return curves - curves @ proj.T


def _safe_div_t(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(num, float("nan"))
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _pearson_per_vertex(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """p, g: [V, T] -> [V] Pearson correlation per row."""
    p_c = p - p.mean(dim=1, keepdim=True)
    g_c = g - g.mean(dim=1, keepdim=True)
    num = (p_c * g_c).sum(dim=1)
    den = torch.sqrt((p_c * p_c).sum(dim=1) * (g_c * g_c).sum(dim=1))
    return _safe_div_t(num, den)


# ---------------- per-vertex accumulator (face-area variant) ----------------

class VertexRunningSum1D:
    """Per-vertex running sums of finite values + counts."""
    def __init__(self):
        self.sum: Optional[torch.Tensor] = None
        self.sum2: Optional[torch.Tensor] = None
        self.count: Optional[torch.Tensor] = None

    def add(self, values: torch.Tensor, track_sq: bool = False):
        v = values.detach().to(device=_DEVICE, dtype=_DTYPE)
        finite = torch.isfinite(v)
        v_clean = torch.where(finite, v, torch.zeros_like(v))
        if self.sum is None:
            self.sum = torch.zeros_like(v_clean)
            self.count = torch.zeros_like(v_clean)
            if track_sq:
                self.sum2 = torch.zeros_like(v_clean)
        self.sum += v_clean
        self.count += finite.to(self.count.dtype)
        if track_sq:
            if self.sum2 is None:
                self.sum2 = torch.zeros_like(v_clean)
            self.sum2 += v_clean * v_clean


class VertexFaceAreaAccumulator:
    """Per-vertex stats for 1D face-area curves. Each buffer ends up shape [V]."""
    def __init__(self):
        self.raw_corr = VertexRunningSum1D()
        self.detr_corr = VertexRunningSum1D()
        # ratio-of-means components
        self.amp_p_sq = VertexRunningSum1D()
        self.amp_g_sq = VertexRunningSum1D()
        self.peak_p = VertexRunningSum1D()
        self.peak_g = VertexRunningSum1D()
        self.rdiff_sq = VertexRunningSum1D()
        self.l2_sq = VertexRunningSum1D()
        self.gt_norm_sq = VertexRunningSum1D()
        # mean-of-per-subject-ratios
        self.amp_ratio_mor = VertexRunningSum1D()
        self.peak_ratio_mor = VertexRunningSum1D()
        self.nrmse_mor = VertexRunningSum1D()
        self.l2_mor = VertexRunningSum1D()
        self.l2_rel_mor = VertexRunningSum1D()
        # for subjStdRatio
        self.pred_mean = VertexRunningSum1D()
        self.gt_mean = VertexRunningSum1D()

    def add_subject(self, pred_va: torch.Tensor, gt_va: torch.Tensor):
        """pred_va, gt_va: [T, V] face-area curves for one subject."""
        p = pred_va.T.contiguous()  # [V, T]
        g = gt_va.T.contiguous()
        rp = _detrend_poly3_batch(p)
        rg = _detrend_poly3_batch(g)

        self.raw_corr.add(_pearson_per_vertex(p, g))
        self.detr_corr.add(_pearson_per_vertex(rp, rg))

        amp_p_sq = (rp * rp).mean(dim=1)
        amp_g_sq = (rg * rg).mean(dim=1)
        peak_p = rp.amax(dim=1) - rp.amin(dim=1)
        peak_g = rg.amax(dim=1) - rg.amin(dim=1)
        rdiff_sq = ((rp - rg) ** 2).mean(dim=1)
        l2_sq = ((p - g) ** 2).sum(dim=1)
        gt_norm_sq = (g * g).sum(dim=1)

        amp_p = torch.sqrt(amp_p_sq)
        amp_g = torch.sqrt(amp_g_sq)
        rnrmse = torch.sqrt(rdiff_sq)
        l2 = torch.sqrt(l2_sq)
        gt_norm = torch.sqrt(gt_norm_sq)

        self.amp_p_sq.add(amp_p_sq)
        self.amp_g_sq.add(amp_g_sq)
        self.peak_p.add(peak_p)
        self.peak_g.add(peak_g)
        self.rdiff_sq.add(rdiff_sq)
        self.l2_sq.add(l2_sq)
        self.gt_norm_sq.add(gt_norm_sq)

        self.amp_ratio_mor.add(_safe_div_t(amp_p, amp_g))
        self.peak_ratio_mor.add(_safe_div_t(peak_p, peak_g))
        self.nrmse_mor.add(_safe_div_t(rnrmse, amp_g))
        self.l2_mor.add(l2)
        self.l2_rel_mor.add(_safe_div_t(l2, gt_norm))

        self.pred_mean.add(p.mean(dim=1), track_sq=True)
        self.gt_mean.add(g.mean(dim=1), track_sq=True)

    def per_vertex_arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}

        def mean(rs: VertexRunningSum1D) -> torch.Tensor:
            return _safe_div_t(rs.sum, rs.count)

        def std(rs: VertexRunningSum1D) -> torch.Tensor:
            mu = _safe_div_t(rs.sum, rs.count)
            ex2 = _safe_div_t(rs.sum2, rs.count)
            var = torch.clamp(ex2 - mu * mu, min=0.0)
            return torch.sqrt(var)

        out["meanCorr"] = mean(self.raw_corr).cpu().numpy()
        out["detrCorr"] = mean(self.detr_corr).cpu().numpy()

        amp_p_rms = torch.sqrt(_safe_div_t(self.amp_p_sq.sum, self.amp_p_sq.count))
        amp_g_rms = torch.sqrt(_safe_div_t(self.amp_g_sq.sum, self.amp_g_sq.count))
        out["ampRatio_rom"] = _safe_div_t(amp_p_rms, amp_g_rms).cpu().numpy()
        out["peakAmpR_rom"] = _safe_div_t(self.peak_p.sum, self.peak_g.sum).cpu().numpy()
        rnrmse_rms = torch.sqrt(_safe_div_t(self.rdiff_sq.sum, self.rdiff_sq.count))
        out["detrNRMSE_rom"] = _safe_div_t(rnrmse_rms, amp_g_rms).cpu().numpy()
        l2_rms = torch.sqrt(_safe_div_t(self.l2_sq.sum, self.l2_sq.count))
        out["predGtL2_rom"] = l2_rms.cpu().numpy()
        out["predGtL2N_rom"] = _safe_div_t(
            torch.sqrt(self.l2_sq.sum), torch.sqrt(self.gt_norm_sq.sum)
        ).cpu().numpy()

        out["ampRatio_mor"] = mean(self.amp_ratio_mor).cpu().numpy()
        out["peakAmpR_mor"] = mean(self.peak_ratio_mor).cpu().numpy()
        out["detrNRMSE_mor"] = mean(self.nrmse_mor).cpu().numpy()
        out["predGtL2_mor"] = mean(self.l2_mor).cpu().numpy()
        out["predGtL2N_mor"] = mean(self.l2_rel_mor).cpu().numpy()

        out["subjStdRatio"] = _safe_div_t(std(self.pred_mean), std(self.gt_mean)).cpu().numpy()
        out["count"] = self.raw_corr.count.cpu().numpy() if self.raw_corr.count is not None else np.zeros(0)
        return out


def _aggregate_per_vertex(per_vertex: Dict[str, np.ndarray]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, arr in per_vertex.items():
        if k == "count":
            continue
        sl = arr[np.isfinite(arr)]
        if sl.size:
            mean = float(sl.mean())
            std = float(sl.std())
            out[k] = f"{mean:.3f} ± {std:.3f}"
        else:
            out[k] = "nan ± nan"
    return out


def _vertex_paths(varea_dir: str, name: str, sid: str) -> Optional[Tuple[str, str]]:
    pred_path = os.path.join(varea_dir, f"{name}_{sid}_pred_vertex.npy")
    gt_path = os.path.join(varea_dir, f"{sid}_gt_vertex.npy")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(varea_dir, f"{name}_{sid}_gt_vertex.npy")
    if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
        return None
    return pred_path, gt_path


def compute_metrics(results, scales, varea_dir=None, out_dir=None):
    """Compute metrics per experiment per scale.

    For "whole" and "material_*" scales each subject contributes one scalar
    curve, so we accumulate per-subject quantities and report both
    ratio-of-means (*_rom) and mean-of-per-subject-ratios (*_mor).

    For "vertex" scale we aggregate per-vertex across subjects, save the
    per-vertex score arrays as {exp}_per_vertex.npz under out_dir, and report
    a single row whose values are the mean over all vertices.
    """
    rows = []
    for name, subs in sorted(results.items()):
        for scale in scales:
            if scale == "vertex":
                if varea_dir is None or out_dir is None:
                    continue
                acc = VertexFaceAreaAccumulator()
                det_pred_curves, det_gt_curves = [], []
                det_pred_pv, det_gt_pv = [], []
                npz_path = os.path.join(out_dir, f"{name}_per_vertex.npz")
                cache_hit = os.path.exists(npz_path)

                for sid in subs.keys():
                    paths = _vertex_paths(varea_dir, name, sid)
                    if paths is None:
                        continue
                    pred_va = _to_tensor(np.load(paths[0]))  # [T, V]
                    gt_va = _to_tensor(np.load(paths[1]))
                    if pred_va.shape != gt_va.shape:
                        tqdm.write(f"[shape-mismatch] {name}/{sid}: pred={tuple(pred_va.shape)} gt={tuple(gt_va.shape)}")
                    if pred_va.numel() > 100_000_000:
                        raise RuntimeError(f"[shape-guard] {name}/{sid} vertex npy too large: {tuple(pred_va.shape)}")
                    if not cache_hit:
                        acc.add_subject(pred_va, gt_va)
                    # whole-mesh diversity proxy: mean per-vertex curve
                    det_pred_curves.append(_detrend_poly3(pred_va.mean(dim=1)))
                    det_gt_curves.append(_detrend_poly3(gt_va.mean(dim=1)))
                    # per-vertex detrended curves for per-vertex diversity
                    det_pred_pv.append(_detrend_poly3_batch(pred_va.T.contiguous()).cpu())
                    det_gt_pv.append(_detrend_poly3_batch(gt_va.T.contiguous()).cpu())

                if not cache_hit and acc.raw_corr.sum is None:
                    continue

                if cache_hit:
                    per_vertex = dict(np.load(npz_path))
                else:
                    per_vertex = acc.per_vertex_arrays()

                div_pred = _per_vertex_diversity_2d(det_pred_pv)
                div_gt = _per_vertex_diversity_2d(det_gt_pv)
                per_vertex["dPredCorr"] = div_pred["dCorr"]
                per_vertex["dGTCorr"]   = div_gt["dCorr"]
                per_vertex["dPredAmp"]  = div_pred["dAmp"]
                per_vertex["dGTAmp"]    = div_gt["dAmp"]
                per_vertex["dPredDist"] = div_pred["dDist"]
                per_vertex["dGTDist"]   = div_gt["dDist"]
                if out_dir is not None and not cache_hit:
                    np.savez_compressed(npz_path, **per_vertex)
                    tqdm.write(f"  {name}: wrote {npz_path}")

                row = {"experiment": name, "scale": scale, **_aggregate_per_vertex(per_vertex)}
                _attach_xsubj(row, det_pred_curves, det_gt_curves)
                rows.append(row)
                continue

            # whole / material scales: per-subject scalar curves
            row = _compute_subject_scale_row(name, scale, subs)
            if row is not None:
                rows.append(row)
    return rows


def _per_vertex_diversity_2d(curves_pv: list) -> Dict[str, np.ndarray]:
    """For a list of S detrended [V, T] tensors, compute per-vertex diversity:
        dCorr  : mean pairwise Pearson across subjects per vertex
        dAmp   : CV of per-vertex peak-to-peak across subjects
        dDist  : mean pairwise normalized RMSE across subjects per vertex
    """
    S = len(curves_pv)
    if S < 2 or curves_pv[0].numel() == 0:
        V = curves_pv[0].shape[0] if S else 0
        nan = np.full((V,), np.nan, dtype=np.float64)
        return {"dCorr": nan, "dAmp": nan.copy(), "dDist": nan.copy()}

    X = torch.stack([c.to(_DEVICE, dtype=_DTYPE) for c in curves_pv], dim=0)  # [S, V, T]
    S_, V, T = X.shape
    Xv = X.permute(1, 0, 2).contiguous()                                     # [V, S, T]
    Xc = Xv - Xv.mean(dim=-1, keepdim=True)
    norm = torch.linalg.norm(Xc, dim=-1)                                     # [V, S]

    # Vertex chunk size so [C, S, S] float32 fits in ~1 GiB (bmm may need 2-3x workspace)
    chunk = max(1, min(int(1 * (1024 ** 3)) // (S_ * S_ * 4), V))
    corr_parts, dist_parts, amp_parts = [], [], []
    iu_r, iu_c = torch.triu_indices(S_, S_, offset=1, device=_DEVICE).unbind(0)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        xc = Xc[start:end].to(_DTYPE)                                         # [C, S, T]
        nrm = norm[start:end]                                                 # [C, S]
        C = torch.bmm(xc, xc.transpose(1, 2))                                 # [C, S, S]
        den = nrm.unsqueeze(2) * nrm.unsqueeze(1) + EPS
        C = C / den
        corr_parts.append(C[:, iu_r, iu_c].mean(dim=1))

        pp = (xc.amax(dim=-1) - xc.amin(dim=-1))                              # [C, S]
        pp_mean = pp.mean(dim=-1)
        pp_std = pp.std(dim=-1, unbiased=False)
        amp_parts.append(pp_std / (pp_mean.abs() + EPS))

        sq = (xc * xc).sum(dim=-1)                                            # [C, S]
        dot = torch.bmm(xc, xc.transpose(1, 2))
        d2 = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * dot).clamp(min=0.0) / T
        dist_parts.append(torch.sqrt(d2[:, iu_r, iu_c]).mean(dim=-1) / (torch.sqrt(sq / T).mean(dim=-1) + EPS))

    corr = torch.cat(corr_parts, dim=0)
    amp_cv = torch.cat(amp_parts, dim=0)
    dist = torch.cat(dist_parts, dim=0)
    dist = torch.cat(dist_parts, dim=0)

    return {
        "dCorr": corr.detach().cpu().numpy().astype(np.float64),
        "dAmp":  amp_cv.detach().cpu().numpy().astype(np.float64),
        "dDist": dist.detach().cpu().numpy().astype(np.float64),
    }


def _attach_xsubj(row: dict, det_pred_curves: list, det_gt_curves: list):
    if not det_pred_curves:
        for k in ("xsubjPred", "xsubjGT", "xsubjGap", "xsAmpCVPred", "xsAmpCVGT",
                 "xsDistPred", "xsDistGT", "xsDistRatio"):
            row[k] = float("nan")
        return
    det_pred_mat = torch.stack(det_pred_curves, dim=0)
    det_gt_mat = torch.stack(det_gt_curves, dim=0)

    pred_vals = _pairwise_pearson_values(det_pred_mat)
    gt_vals = _pairwise_pearson_values(det_gt_mat)
    pred_mean = float(pred_vals.mean().item()) if pred_vals.numel() else float("nan")
    pred_std = float(pred_vals.std(unbiased=False).item()) if pred_vals.numel() else float("nan")
    gt_mean = float(gt_vals.mean().item()) if gt_vals.numel() else float("nan")
    gt_std = float(gt_vals.std(unbiased=False).item()) if gt_vals.numel() else float("nan")

    gap_vals = pred_vals - gt_vals if pred_vals.numel() and gt_vals.numel() and pred_vals.numel() == gt_vals.numel() else torch.tensor([], device=_DEVICE)
    gap_mean = float(gap_vals.mean().item()) if gap_vals.numel() else float("nan")
    gap_std = float(gap_vals.std(unbiased=False).item()) if gap_vals.numel() else float("nan")

    xs_amp_cv_pred = _amp_cv(det_pred_mat)
    xs_amp_cv_gt = _amp_cv(det_gt_mat)
    denom = float(det_gt_mat.std(dim=1, unbiased=False).mean().item() + EPS)
    dist_pred_vals = _pairwise_rmse_values(det_pred_mat, denom)
    dist_gt_vals = _pairwise_rmse_values(det_gt_mat, denom)
    dist_pred_mean = float(dist_pred_vals.mean().item()) if dist_pred_vals.numel() else float("nan")
    dist_pred_std = float(dist_pred_vals.std(unbiased=False).item()) if dist_pred_vals.numel() else float("nan")
    dist_gt_mean = float(dist_gt_vals.mean().item()) if dist_gt_vals.numel() else float("nan")
    dist_gt_std = float(dist_gt_vals.std(unbiased=False).item()) if dist_gt_vals.numel() else float("nan")

    ratio_vals = dist_pred_vals / (dist_gt_vals + EPS) if dist_pred_vals.numel() and dist_gt_vals.numel() and dist_pred_vals.numel() == dist_gt_vals.numel() else torch.tensor([], device=_DEVICE)
    ratio_mean = float(ratio_vals.mean().item()) if ratio_vals.numel() else float("nan")
    ratio_std = float(ratio_vals.std(unbiased=False).item()) if ratio_vals.numel() else float("nan")

    row.update({
        "xsubjPred": f"{pred_mean:.3f} ± {pred_std:.3f}",
        "xsubjGT": f"{gt_mean:.3f} ± {gt_std:.3f}",
        "xsubjGap": f"{gap_mean:.3f} ± {gap_std:.3f}",
        "xsAmpCVPred": f"{xs_amp_cv_pred:.3f}",
        "xsAmpCVGT": f"{xs_amp_cv_gt:.3f}",
        "xsDistPred": f"{dist_pred_mean:.3f} ± {dist_pred_std:.3f}",
        "xsDistGT": f"{dist_gt_mean:.3f} ± {dist_gt_std:.3f}",
        "xsDistRatio": f"{ratio_mean:.3f} ± {ratio_std:.3f}",
    })


def _compute_subject_scale_row(name: str, scale: str, subs: dict) -> Optional[dict]:
    """Accumulate per-subject quantities for one scalar-curve scale and
    finalize both ratio-of-means and mean-of-per-subject-ratios versions."""
    corrs, dcorrs = [], []
    pred_means, gt_means = [], []
    det_pred_curves, det_gt_curves = [], []

    # ratio-of-means buffers
    amp_p_sq_sum = amp_g_sq_sum = 0.0
    peak_p_sum = peak_g_sum = 0.0
    rdiff_sq_sum = 0.0
    l2_sq_sum = gt_norm_sq_sum = 0.0
    n_pairs = 0
    # mean-of-per-subject-ratios buffers
    amp_ratios, peak_ratios, nrmses = [], [], []
    l2_abs, l2_rel = [], []

    for _sid, subj_data in subs.items():
        if scale not in subj_data["pred"]:
            continue
        p = _to_tensor(subj_data["pred"][scale]).reshape(-1)
        g = _to_tensor(subj_data["gt"][scale]).reshape(-1)
        rp = _detrend_poly3(p)
        rg = _detrend_poly3(g)

        corrs.append(_pearson(p, g))
        dcorrs.append(_pearson(rp, rg))

        amp_p_sq = float((rp * rp).mean().item())
        amp_g_sq = float((rg * rg).mean().item())
        peak_p = float((rp.amax() - rp.amin()).item())
        peak_g = float((rg.amax() - rg.amin()).item())
        rdiff_sq = float(((rp - rg) ** 2).mean().item())
        diff = p - g
        l2_sq = float((diff * diff).sum().item())
        gt_norm_sq = float((g * g).sum().item())

        amp_p_sq_sum += amp_p_sq; amp_g_sq_sum += amp_g_sq
        peak_p_sum += peak_p; peak_g_sum += peak_g
        rdiff_sq_sum += rdiff_sq
        l2_sq_sum += l2_sq; gt_norm_sq_sum += gt_norm_sq
        n_pairs += 1

        amp_p = np.sqrt(amp_p_sq); amp_g = np.sqrt(amp_g_sq)
        l2 = np.sqrt(l2_sq); gt_norm = np.sqrt(gt_norm_sq)
        amp_ratios.append(amp_p / (amp_g + EPS))
        peak_ratios.append(peak_p / (peak_g + EPS))
        nrmses.append(np.sqrt(rdiff_sq) / (amp_g + EPS))
        l2_abs.append(l2)
        l2_rel.append(l2 / (gt_norm + EPS))

        pred_means.append(float(p.mean().item()))
        gt_means.append(float(g.mean().item()))
        det_pred_curves.append(rp)
        det_gt_curves.append(rg)

    if n_pairs == 0:
        return None

    amp_p_rms = np.sqrt(amp_p_sq_sum / n_pairs)
    amp_g_rms = np.sqrt(amp_g_sq_sum / n_pairs)
    rnrmse_rms = np.sqrt(rdiff_sq_sum / n_pairs)
    l2_rms = np.sqrt(l2_sq_sum / n_pairs)

    row = {
        "experiment": name,
        "scale": scale,
        "meanCorr": float(np.nanmean(corrs)),
        "detrCorr": float(np.nanmean(dcorrs)),
        "subjStdRatio": float(np.std(pred_means) / (np.std(gt_means) + EPS)),
        "ampRatio_rom": float(amp_p_rms / (amp_g_rms + EPS)),
        "peakAmpR_rom": float(peak_p_sum / (peak_g_sum + EPS)),
        "detrNRMSE_rom": float(rnrmse_rms / (amp_g_rms + EPS)),
        "predGtL2_rom": float(l2_rms),
        "predGtL2N_rom": float(np.sqrt(l2_sq_sum) / (np.sqrt(gt_norm_sq_sum) + EPS)),
        "ampRatio_mor": float(np.nanmean(amp_ratios)),
        "peakAmpR_mor": float(np.nanmean(peak_ratios)),
        "detrNRMSE_mor": float(np.nanmean(nrmses)),
        "predGtL2_mor": float(np.mean(l2_abs)),
        "predGtL2N_mor": float(np.mean(l2_rel)),
    }
    _attach_xsubj(row, det_pred_curves, det_gt_curves)
    return row


def print_metrics(rows):
    if not rows:
        return
    cols = ("experiment", "scale", "meanCorr", "ampRatio_rom", "ampRatio_mor",
            "detrNRMSE_rom", "predGtL2_rom", "predGtL2N_rom",
            "xsubjPred", "xsubjGT")
    widths = (50, 25, 9, 13, 13, 13, 12, 12, 11, 11)
    print(" ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    for r in rows:
        short = r["experiment"].replace("_8L_256d_49frames", "")
        vals = [short, r["scale"]] + [
            f"{r.get(k, float('nan')):.3f}" if isinstance(r.get(k), float) else str(r.get(k))
            for k in cols[2:]
        ]
        print(" ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))


def write_metrics_csv(rows, path):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float("nan")) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default='checkpoints',
                    help="Directory containing per-experiment folders with {dirname}/*_{pred,gt}.npy")
    ap.add_argument("--mesh-info", default='../ukb_heart_mesh_visual',
                    help="Directory containing control_material_idmap.json, control_materials.npy, control_face.ply")
    ap.add_argument("--out", default='./outputs', help="Output directory for plots/metrics")
    ap.add_argument("--plot", type=int, default=0)
    ap.add_argument("--tag", type=str, default='facearea_multiscale')
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for GPU area computation")
    ap.add_argument("--io-workers", type=int, default=8,
                    help="DataLoader workers for parallel npy loading")
    ap.add_argument("--subject-list", type=str, default=None,
                    help="Optional path to a txt file (one subid per line) to restrict to a "
                         "subgroup. Use a fresh --tag so a stale area cache is not reused.")
    args = ap.parse_args()
    args.out = os.path.join(args.out, args.tag)
    os.makedirs(args.out, exist_ok=True)

    subject_list = None
    if args.subject_list is not None:
        with open(args.subject_list) as f:
            subject_list = {ln.strip() for ln in f if ln.strip()}
        print(f"[subject-list] restricting to {len(subject_list)} subjects from {args.subject_list}")

    mesh, material_groups = load_mesh_info(args.mesh_info)

    material_scales = [name for name, _ in material_groups]
    scales = ["whole"] + material_scales + ["vertex"]

    cache_path = os.path.join(args.out, "areas_per_scale.npz")
    manifest_path = os.path.join(args.out, "areas_manifest.json")
    varea_dir = os.path.join(args.out, "vertex_areas")

    if os.path.exists(cache_path) and os.path.exists(manifest_path):
        print(f"loading cached areas from {cache_path}")
        results = load_area_caches(cache_path, manifest_path)
        if not _vertex_cache_complete(results, varea_dir):
            print("vertex area cache is incomplete; recomputing areas")
            results = collect_areas(args.root, mesh, material_groups,
                                    batch_size=args.batch_size, num_workers=args.io_workers,
                                    subject_list=subject_list)
            write_area_caches(results, cache_path, manifest_path, varea_dir)
    else:
        results = collect_areas(args.root, mesh, material_groups, subject_list=subject_list)
        write_area_caches(results, cache_path, manifest_path, varea_dir)

    rows = compute_metrics(results, scales, varea_dir, out_dir=args.out)
    print_metrics(rows)
    write_metrics_csv(rows, os.path.join(args.out, "metrics-avgstd_per_scale.csv"))

    if args.plot == 1:
        print(f"wrote metrics under {args.out}")


if __name__ == "__main__":
    main()
