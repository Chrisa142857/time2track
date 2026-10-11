"""
UK Biobank 4-D heart mesh dataset for ECG-conditioned trajectory generation.

Produces batches compatible with the ECG-conditioned training loop:
    data_dict = {
        'points_src' : (1, N, 3)      float32  – rest-state mesh (frame 0)
        'points_tgt' : (T, N, 3)      float32  – motion trajectory (frames 1..T)
        'ecg'        : (12, 600)      float32  – 12-lead ECG (pre-normalised per lead)
    }
    info_dict = {'subid': str}

Mesh data expected layout (two alternatives):
    {mesh_dir}/{subid}_{...}/frames.npy          <- fast path: pre-computed (50, 388, 3)
    {mesh_dir}/{subid}_{...}/frame_000.ply …     <- fallback: ASCII PLY files

ECG data expected layout:
    {ecg_dir}/{subid}.npy                        <- fast path: pre-normalised (12, 600) float32
    {ecg_dir}/{subid}_2_0.csv                    <- fallback: raw (12, 600); z-score applied on load

Run  python precompute_npy.py  once to populate the fast-path .npy files.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PLY helpers
# ---------------------------------------------------------------------------

def _parse_ply(path):
    """Parse ASCII PLY. Returns (verts (N,3), edges (E,2), faces (F,?) int64)."""
    with open(path) as f:
        lines = f.readlines()
    n_v = n_e = n_f = 0
    i = 0
    while lines[i].strip() != 'end_header':
        tok = lines[i].split()
        if len(tok) >= 3 and tok[0] == 'element':
            if tok[1] == 'vertex':   n_v = int(tok[2])
            elif tok[1] == 'edge':   n_e = int(tok[2])
            elif tok[1] == 'face':   n_f = int(tok[2])
        i += 1
    i += 1  # skip end_header
    verts = np.array([lines[i + k].split()[:3] for k in range(n_v)], dtype=np.float32)
    i += n_v
    edges = np.array([lines[i + k].split()[:2] for k in range(n_e)], dtype=np.int64)
    i += n_e
    faces = np.array([lines[i + k].split()[1:] for k in range(n_f)], dtype=np.int64)
    return verts, edges, faces


def _load_ply_verts(path):
    with open(path) as f:
        lines = f.readlines()
    n_v = 0
    i = 0
    while lines[i].strip() != 'end_header':
        tok = lines[i].split()
        if len(tok) >= 3 and tok[0] == 'element' and tok[1] == 'vertex':
            n_v = int(tok[2])
        i += 1
    i += 1
    return np.array([lines[i + k].split()[:3] for k in range(n_v)], dtype=np.float32)


def _quad_to_tri(faces_quad):
    f0, f1, f2, f3 = faces_quad[:, 0], faces_quad[:, 1], faces_quad[:, 2], faces_quad[:, 3]
    return np.concatenate([
        np.stack([f0, f1, f2], axis=1),
        np.stack([f0, f2, f3], axis=1),
    ], axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Subject-list JSON cache
# ---------------------------------------------------------------------------

_CACHE_FILENAME = '.subject_pair_cache.json'
_FILTER_CACHE_FILENAME = '.subject_filter_cache.json'


def _load_cache(mesh_dir):
    path = os.path.join(mesh_dir, _CACHE_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return [(item[0], item[1], item[2]) for item in data]
    return None


def _save_cache(mesh_dir, subjects):
    path = os.path.join(mesh_dir, _CACHE_FILENAME)
    with open(path, 'w') as f:
        json.dump(subjects, f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UKbiobankHeartMesh(Dataset):
    """
    Paired (4-D heart mesh trajectory, resting 12-lead ECG) from UK Biobank.

    Config fields required (OmegaConf-compatible dict / object):
        ecg_dir        : str   – directory containing {subid}.npy (pre-normalised) or {subid}_2_0.csv files
        mesh_dir       : str   – directory containing {subid}_xxx/ sub-dirs
        npy_dir        : str   – optional override for frames.npy root
        seq_len        : int   – total frames available per sequence (default 50)
        n_training_frames : int – number of *target* frames to return  (default 49)
        scale          : float – coordinate scale applied after centering (default 1.0)

    Returns (data_dict, info_dict) where:
        data_dict['points_src'] : (1, N, 3)
        data_dict['points_tgt'] : (n_training_frames, N, 3)
        data_dict['ecg']        : (12, 600)
        info_dict['subid']      : str
    """

    def __init__(self, split: str, config):
        self.ecg_dir  = config.ecg_dir
        self.mesh_dir = config.mesh_dir
        self.npy_dir  = getattr(config, 'npy_dir', None) or None
        self.seq_len  = getattr(config, 'seq_len', 50)
        self.n_frames = getattr(config, 'n_training_frames', 49)
        self.scale    = getattr(config, 'scale', 1.0)
        self.delay_mat_path = getattr(config, 'delay_mat_path', None) or None
        self.delay_key      = getattr(config, 'delay_key', 'T_peak')
        # Temporal Gaussian smoothing: sigma=0 disables it
        self.smooth_sigma   = float(getattr(config, 'smooth_sigma', 0.0))
        # Filter subjects whose per-vertex displacement from rest exceeds this
        # value (in mesh units). None disables filtering.
        # Subjects where the fraction of noisy frames exceeds filter_max_noisy_frac
        # are dropped; subjects with fewer noisy frames are kept and those frames
        # are repaired by interpolation from clean neighbors in __getitem__.
        self.filter_max_displacement = getattr(config, 'filter_max_displacement', 50)
        if self.filter_max_displacement is not None:
            self.filter_max_displacement = float(self.filter_max_displacement)
        self.filter_max_noisy_frac = float(getattr(config, 'filter_max_noisy_frac', 0.1))
        # Populated by _filter_subjects: {sid: [frame_indices_in_arr]} for subjects
        # that have ≥1 noisy frame (but below the drop threshold).  Dict lookup in
        # __getitem__ replaces a full per-sample displacement scan.
        self._noisy_frames: dict = {}

        subjects = self._find_paired_subjects()

        if self.filter_max_displacement is not None:
            subjects, self._noisy_frames = self._filter_subjects(subjects)

        # ── Optional ECG delay matrix (12×12 per subject) ────────────────
        self.delay_map = None
        if self.delay_mat_path is not None:
            print(f'[UKbiobankHeartMesh] Loading delay matrix ({self.delay_mat_path})')
            npz = np.load(self.delay_mat_path, allow_pickle=True)
            if self.delay_key not in npz.files:
                raise KeyError(
                    f"Delay key '{self.delay_key}' not in {self.delay_mat_path}; "
                    f"available: {list(npz.files)}"
                )
            mats = npz[self.delay_key].astype(np.float32)   # (K, 12, 12)
            names = npz['filename']                          # (K,) e.g. '1234567.npy'
            if mats.shape[0] != names.shape[0]:
                raise ValueError(
                    f"Delay matrix count ({mats.shape[0]}) does not match "
                    f"filename count ({names.shape[0]}) in {self.delay_mat_path}"
                )
            self.delay_map = {
                str(n).split('.')[0]: mats[i] for i, n in enumerate(names)
            }
            # Strict: every paired subject must have a delay entry
            missing = [sid for sid, _, _ in subjects if sid not in self.delay_map]
            if missing:
                raise ValueError(
                    f"{len(missing)} paired subjects have no delay matrix "
                    f"(e.g. {missing[:3]}). Delay file: {self.delay_mat_path}"
                )
            print(f'[UKbiobankHeartMesh] Loaded delay matrix ({self.delay_key}) '
                  f'for {len(self.delay_map)} subjects.')

        rng   = np.random.RandomState(42)
        idx   = rng.permutation(len(subjects))
        split_frac = getattr(config, 'train_split', 0.8)
        cutoff = int(split_frac * len(subjects))
        if split == 'train':
            self.subjects = [subjects[i] for i in idx[:cutoff]]
        else:
            self.subjects = [subjects[i] for i in idx[cutoff:]]

        print(f'[UKbiobankHeartMesh] {split}: {len(self.subjects)} subjects')

        # Shared mesh topology (edges / faces) – loaded once
        self.edge_index, self.faces_tri = self._load_topology()

    # ------------------------------------------------------------------

    def _filter_subjects(self, subjects):
        """Remove subjects whose mesh trajectory has frequent extreme displacement.

        A frame is considered noisy when its max per-vertex L2 distance from the
        rest frame exceeds ``filter_max_displacement``.  Subjects are dropped only
        when the *fraction* of such frames exceeds ``filter_max_noisy_frac``
        (default 0.1).  Subjects with rare noisy frames are kept; those frames are
        repaired by interpolation in ``__getitem__``.

        Returns
        -------
        filtered        : list of (sid, ecg_file, mesh_d) tuples
        noisy_frames    : dict {sid: [frame_indices]} — only entries with ≥1 noisy frame
        """
        max_noisy_frac  = self.filter_max_noisy_frac
        cache_dir       = self.npy_dir or self.mesh_dir
        stem            = (f'.subject_filter_cache_{self.filter_max_displacement:.4f}'
                           f'_frac{max_noisy_frac:.2f}')
        cache_path      = os.path.join(cache_dir, stem + '.json')
        nf_cache_path   = os.path.join(cache_dir, stem + '_noisy_frames.json')

        if os.path.exists(cache_path) and os.path.exists(nf_cache_path):
            with open(cache_path) as f:
                good_sids = set(json.load(f))
            with open(nf_cache_path) as f:
                noisy_frames = json.load(f)
            filtered  = [s for s in subjects if s[0] in good_sids]
            n_dropped = len(subjects) - len(filtered)
            n_with_nf = sum(1 for v in noisy_frames.values() if v)
            print(
                f'[UKbiobankHeartMesh] Filter cache loaded: kept {len(filtered)} '
                f'({n_with_nf} with repaired frames), '
                f'dropped {n_dropped} '
                f'(noisy_frac>{max_noisy_frac} at disp>{self.filter_max_displacement})'
            )
            return filtered, noisy_frames

        print(
            f'[UKbiobankHeartMesh] Filtering subjects '
            f'(max_disp={self.filter_max_displacement}, '
            f'max_noisy_frac={max_noisy_frac}) …', flush=True
        )
        good, bad        = [], []
        noisy_frames: dict = {}
        for sid, ecg_file, mesh_d in tqdm(subjects, desc='Filtering'):
            npy_root = self.npy_dir or self.mesh_dir
            npy_path = os.path.join(npy_root, mesh_d, 'frames.npy')
            try:
                if os.path.exists(npy_path):
                    arr = np.load(npy_path)
                else:
                    mesh_path = os.path.join(self.mesh_dir, mesh_d)
                    frame_files = sorted(
                        f for f in os.listdir(mesh_path) if f.endswith('.ply')
                    )[:self.seq_len]
                    arr = np.stack([
                        _load_ply_verts(os.path.join(mesh_path, ff))
                        for ff in frame_files
                    ])
                arr = arr[:self.seq_len].astype(np.float32)
                # Per-frame max per-vertex L2 displacement from rest frame
                frame_max_disp = np.sqrt(
                    ((arr[1:] - arr[0:1]) ** 2).sum(-1)
                ).max(axis=1)                                    # (T-1,)
                noisy_mask = frame_max_disp > self.filter_max_displacement
                n_noisy    = int(noisy_mask.sum())
                noisy_frac = n_noisy / max(len(frame_max_disp), 1)
                if noisy_frac <= max_noisy_frac:
                    good.append(sid)
                    if n_noisy:
                        # Pre-compute interpolation recipe for each noisy frame:
                        # (fi, p, n, alpha) — always use (1-alpha)*arr[p] + alpha*arr[n].
                        # Boundary case: p == n, alpha == 0 (just copy the clean frame).
                        n_total   = len(arr)
                        noisy_set = set((np.where(noisy_mask)[0] + 1).tolist())
                        clean_all = [i for i in range(n_total) if i not in noisy_set]
                        recipes   = []
                        for fi in sorted(noisy_set):
                            before = [c for c in clean_all if c < fi]
                            after  = [c for c in clean_all if c > fi]
                            if not before:
                                p, n, a = after[0], after[0], 0.0
                            elif not after:
                                p, n, a = before[-1], before[-1], 0.0
                            else:
                                p, n    = before[-1], after[0]
                                a       = (fi - p) / (n - p)
                            recipes.append([fi, p, n, a])
                        noisy_frames[sid] = recipes
                else:
                    bad.append((sid, float(frame_max_disp.max()), n_noisy))
            except Exception as e:
                print(f'[UKbiobankHeartMesh] Warning: could not load {sid}: {e}')
                bad.append((sid, float('nan'), -1))

        with open(cache_path, 'w') as f:
            json.dump(good, f)
        with open(nf_cache_path, 'w') as f:
            json.dump(noisy_frames, f)

        n_with_nf = len(noisy_frames)
        print(
            f'[UKbiobankHeartMesh] Filter done: kept {len(good)} '
            f'({n_with_nf} with repaired frames), dropped {len(bad)} subjects.'
        )
        if bad:
            top = sorted(
                (b for b in bad if not np.isnan(b[1])), key=lambda x: -x[1]
            )[:5]
            print(
                f'[UKbiobankHeartMesh] Worst dropped '
                f'(subid, max_disp, n_noisy_frames): {top}'
            )

        return [s for s in subjects if s[0] in set(good)], noisy_frames

    def _find_paired_subjects(self):
        cached = _load_cache(self.npy_dir)
        if cached is not None:
            print(f'[UKbiobankHeartMesh] Loaded {len(cached)} paired subjects from cache.')
            return cached

        print('[UKbiobankHeartMesh] Scanning ECG directory …', flush=True)
        ecg_map = {}
        for fname in os.listdir(self.ecg_dir):
            if fname.endswith('.npy'):
                subid = fname[:-4]          # strip .npy
                ecg_map[subid] = fname
            elif fname.endswith('.csv'):
                subid = fname.split('_')[0]
                if subid not in ecg_map or '_2_0' in fname:
                    ecg_map[subid] = fname

        print(f'[UKbiobankHeartMesh] {len(ecg_map)} ECG files found. '
              'Scanning mesh directory …', flush=True)
        mesh_map = {}
        for dname in os.listdir(self.npy_dir):
            if dname.startswith('.'):
                continue
            subid = dname.split('_')[0]
            if subid not in mesh_map:
                mesh_map[subid] = dname

        common   = sorted(set(ecg_map) & set(mesh_map))
        subjects = [(sid, ecg_map[sid], mesh_map[sid]) for sid in common]
        print(f'[UKbiobankHeartMesh] {len(subjects)} paired subjects. Caching …',
              flush=True)
        _save_cache(self.npy_dir, subjects)
        return subjects

    def _load_topology(self):
        _, _, mesh_d = self.subjects[0]
        frame_path = os.path.join(self.mesh_dir, mesh_d, 'frame_000.ply')
        _, edges, faces_quad = _parse_ply(frame_path)
        edge_index = torch.tensor(edges.T.copy(), dtype=torch.long)
        faces_tri  = torch.tensor(_quad_to_tri(faces_quad), dtype=torch.long)
        return edge_index, faces_tri

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        sid, ecg_file, mesh_d = self.subjects[index]

        # ── ECG ───────────────────────────────────────────────────────────
        ecg_path = os.path.join(self.ecg_dir, ecg_file)
        if ecg_file.endswith('.npy'):
            ecg = np.load(ecg_path)                          # (12, 600) float32, already normalised
        else:
            raw = pd.read_csv(ecg_path, index_col=0).values.astype(np.float32)
            mu  = raw.mean(axis=1, keepdims=True)
            std = raw.std(axis=1, keepdims=True) + 1e-8
            ecg = (raw - mu) / std                           # (12, 600) z-score per lead
        ecg = torch.tensor(ecg, dtype=torch.float32)                          # (12, 600)

        # ── Mesh vertices ─────────────────────────────────────────────────
        mesh_path = os.path.join(self.mesh_dir, mesh_d)
        npy_root  = self.npy_dir if self.npy_dir else self.mesh_dir
        npy_path  = os.path.join(npy_root, mesh_d, 'frames.npy')

        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            arr = np.array(arr, copy=True)                    # own writable copy (avoids mmap/read-only storage)
        else:
            frame_files = sorted(
                f for f in os.listdir(mesh_path) if f.endswith('.ply')
            )[:self.seq_len]
            arr = np.stack([
                _load_ply_verts(os.path.join(mesh_path, ff)) for ff in frame_files
            ])

        arr = arr[:self.seq_len]

        # Temporal Gaussian smoothing: cache the result so it is computed once
        # per subject rather than every time the sample is accessed.
        if self.smooth_sigma > 0.0:
            npy_dir = self.npy_dir or self.mesh_dir
            smooth_path = os.path.join(
                npy_dir, mesh_d, f'frames_s{self.smooth_sigma:.1f}.npy'
            )
            if os.path.exists(smooth_path):
                arr = np.load(smooth_path)
            else:
                arr = gaussian_filter1d(arr.astype(np.float32),
                                        sigma=self.smooth_sigma, axis=0)
                np.save(smooth_path, arr)

        # Repair rare noisy frames. Recipes (fi, p, n, alpha) are pre-computed at
        # init — hot path is a dict lookup + small loop, no numpy scan here.
        recipes = self._noisy_frames.get(sid)
        if recipes:
            arr      = arr.astype(np.float32)
            arr_orig = arr.copy()
            fi, p, n, a = np.array(recipes, dtype=np.float32).T
            fi, p, n    = fi.astype(np.intp), p.astype(np.intp), n.astype(np.intp)
            arr[fi] = (1 - a[:, None, None]) * arr_orig[p] + a[:, None, None] * arr_orig[n]

        verts = torch.tensor(arr, dtype=torch.float32)                          # (T, N, 3)
        # Centre around mean of rest frame (frame 0) to keep spatial scale
        verts = verts - verts[0:1].mean(dim=1, keepdim=True)  # (T, N, 3)
        if self.scale != 1.0:
            verts = verts * self.scale

        # ── Split into source (rest) and target (motion) ──────────────────
        T        = min(self.n_frames, verts.shape[0] - 1)
        points_src = verts[0:1].clone()         # (1, N, 3) – independent storage for collation
        points_tgt = verts[1:T + 1].clone()     # (T, N, 3) – independent storage for collation

        data_dict = {
            'points_src': points_src.float(),
            'points_tgt': points_tgt.float(),
            'ecg':        ecg.float(),
        }
        if self.delay_map is not None:
            data_dict['delay_mat'] = torch.tensor(
                self.delay_map[sid], dtype=torch.float32
            )
        info_dict = {'subid': sid}

        return data_dict, info_dict
