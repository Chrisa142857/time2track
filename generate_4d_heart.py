"""Generate 4D heart motion from your own resting 12-lead ECG with a trained Time2Track model.

Two modes:
  ECG only        the rest mesh is the population-average heart of the training cohort
                  (assets/population_rest_mesh.npy)
  ECG + rest mesh your own end-diastolic 388-point control mesh (--rest-mesh)

  python generate_4d_heart.py --ecg my_ecg.xml --out results/
  python generate_4d_heart.py --ecg my_ecg.csv --fs 500 --rest-mesh my_heart.ply --out results/
  python generate_4d_heart.py --ecg ecg_folder/ --rest-mesh mesh_folder/ --out results/

ECG input (one file per subject, or a folder of them):
  .xml          GE MUSE XML as exported by UK Biobank (field 20205): its 500 Hz median beat is used
  .npy/.csv/.txt  a 12 x T array (or T x 12) in the lead order I, II, III, aVR, aVL, aVF, V1 to V6.
                A CSV whose header or index names the leads is reordered by name. Give the
                sampling rate with --fs. A recording of at most 1.4 s is treated as a median beat.
                A longer one is treated as a rhythm strip, and its R-aligned beats are combined into
                a median beat (--ecg-kind overrides this choice).
The ECG is preprocessed as in training (data_prep/): resampled to 500 Hz, 600 samples, rolled so
that the consensus R-peak is sample 0, and z-normalized per lead.

Rest mesh input (--rest-mesh): .npy of shape (388, 3), (1, 388, 3) or (T, 388, 3) (frame 0 is used),
or a .ply with the 388 vertices of the biv-volumetric-meshing control mesh (the format of
data_prep/mesh_to_ply.py), in millimetres and in the coordinate frame of the UK Biobank meshes.
One file for all ECGs, or a folder with one file per ECG named like the ECG.

Outputs per ECG, in --out:
  <name>_motion.npy     (50, 388, 3) float32: frame 0 is the rest mesh, frames 1 to 49 the generated cycle
  <name>.gif            animation of the generated cycle (skip with --no-gif)
  <name>_ply/           one quad-mesh PLY per frame (with --ply)
  <name>_ecg_input.npy  the preprocessed (12, 600) ECG given to the model
  generation_config.json  every resolved setting of the run
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'data_prep'))

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS, N_SAMPLES = 500, 600          # training ECG: 500 Hz, 600 samples (1.2 s), R-peak at sample 0
PRE_R = 200                       # rhythm strips: samples kept before each R-peak (0.4 s)
ECG_EXT = ('.xml', '.npy', '.csv', '.txt')


# ── ECG ─────────────────────────────────────────────────────────────────────────────────────────
def _reorder(sig, names):
    key = {n.strip().upper(): i for i, n in enumerate(names)}
    missing = [l for l in LEADS if l.upper() not in key]
    if missing:
        raise ValueError(f'leads {missing} not found among {list(names)}')
    return sig[[key[l.upper()] for l in LEADS]]


def read_ecg(path, fs):
    """-> (12, T) float array in LEADS order, its sampling rate, and 'median' for UKB XML."""
    ext = path.suffix.lower()
    if ext == '.xml':
        from align_ecg_to_mri import get_ecg_signal          # same reader as data_prep
        sig, names = get_ecg_signal(str(path))
        return _reorder(sig, names), FS, 'median'
    if ext == '.npy':
        sig, names = np.load(path).astype(np.float64), None
    else:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine='python', header=None)
        first_row, first_col = df.iloc[0].astype(str).tolist(), df.iloc[:, 0].astype(str).tolist()
        if {l.upper() for l in LEADS} <= {c.strip().upper() for c in first_row}:
            names, sig = first_row, df.iloc[1:].to_numpy(float).T       # leads in the header row
        elif {l.upper() for l in LEADS} <= {c.strip().upper() for c in first_col}:
            names, sig = first_col, df.iloc[:, 1:].to_numpy(float)      # leads in the first column
        else:
            names, sig = None, df.to_numpy(float)
    if sig.ndim != 2 or 12 not in sig.shape:
        raise ValueError(f'{path.name}: expected a 12 x T array, got shape {sig.shape}')
    if sig.shape[0] != 12:
        sig = sig.T
    if names is not None:
        sig = _reorder(sig, names)
    if fs is None:
        raise ValueError(f'{path.name}: give the sampling rate with --fs')
    return sig, fs, None


def preprocess_ecg(sig, fs, kind):
    """Training preprocessing: 500 Hz, 600 samples, consensus R-peak rolled to 0, z-score per lead."""
    if kind == 'ready':
        if sig.shape != (12, N_SAMPLES):
            raise ValueError(f"--ecg-kind ready expects (12, {N_SAMPLES}), got {sig.shape}")
        return sig.astype(np.float32), kind
    from scipy.signal import resample
    from align_ecg_to_mri import consensus_r_peak, nk
    if nk is None:
        raise SystemExit('neurokit2 is required for the R-peak alignment (pip install neurokit2==0.2.13)')
    if fs != FS:
        sig = resample(sig, int(round(sig.shape[1] * FS / fs)), axis=1)
    if kind == 'auto':
        kind = 'median' if sig.shape[1] <= int(1.4 * FS) else 'rhythm'
    if kind == 'rhythm':
        lead_ii = nk.ecg_clean(sig[1], sampling_rate=FS, method='neurokit')
        r_peaks = nk.ecg_peaks(lead_ii, sampling_rate=FS, method='neurokit')[1]['ECG_R_Peaks']
        beats = [sig[:, r - PRE_R:r - PRE_R + N_SAMPLES] for r in r_peaks
                 if r - PRE_R >= 0 and r - PRE_R + N_SAMPLES <= sig.shape[1]]
        if not beats:
            raise ValueError('no complete beat found in the rhythm strip')
        sig = np.median(np.stack(beats), axis=0)                         # median beat, R at PRE_R
        r_peak = PRE_R
    elif sig.shape[1] != N_SAMPLES:
        print(f'  note: median beat has {sig.shape[1]} samples at 500 Hz, '
              f'{"cropped" if sig.shape[1] > N_SAMPLES else "edge-padded"} to {N_SAMPLES}')
        sig = sig[:, :N_SAMPLES] if sig.shape[1] > N_SAMPLES else \
            np.pad(sig, ((0, 0), (0, N_SAMPLES - sig.shape[1])), mode='edge')
    if kind == 'median':
        r_peak = consensus_r_peak(sig, FS)                               # as data_prep/align_ecg_to_mri.py
    sig = np.roll(sig, -r_peak, axis=1)
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
    return sig.astype(np.float32), kind


# ── rest mesh ───────────────────────────────────────────────────────────────────────────────────
def read_rest_mesh(path, n_points):
    if path.suffix.lower() == '.ply':
        from dataset.ukbiobank_dataset import _parse_ply
        verts = _parse_ply(str(path))[0]
    else:
        verts = np.load(path)
        if verts.ndim == 3:
            verts = verts[0]
    verts = np.asarray(verts, np.float32)
    if verts.shape != (n_points, 3):
        raise ValueError(f'{path.name}: expected a ({n_points}, 3) control mesh, got {verts.shape}')
    return verts - verts.mean(axis=0, keepdims=True)        # centred on the rest centroid, as in training


def rest_mesh_for(ecg_path, rest_arg, template):
    if rest_arg is None:
        return template, 'population average'
    p = Path(rest_arg)
    if p.is_dir():
        hits = [q for q in sorted(p.iterdir()) if q.stem == ecg_path.stem and q.suffix.lower() in ('.npy', '.ply')]
        if not hits:
            raise FileNotFoundError(f'no rest mesh named {ecg_path.stem}.npy/.ply in {p}')
        p = hits[0]
    return read_rest_mesh(p, template.shape[0]), str(p)


# ── outputs ─────────────────────────────────────────────────────────────────────────────────────
def write_ply(path, verts, faces):
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\n'
                f'property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        f.writelines(f'{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n' for v in verts)
        f.writelines(f'{len(q)} ' + ' '.join(map(str, q)) + '\n' for q in faces)


def write_gif(path, motion, faces_tri, fps=24):
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    from utils.visualization import compute_depth
    lo, hi = motion.reshape(-1, 3).min(0), motion.reshape(-1, 3).max(0)
    pad = 0.05 * (hi - lo).max()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    frames = []
    for t, verts in enumerate(motion):
        ax.cla()
        tri = verts[faces_tri]
        depth = compute_depth(tri.mean(axis=1), elev=20, azim=45)
        depth = (depth - depth.min()) / (np.ptp(depth) + 1e-8)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=plt.colormaps['cool'](depth), alpha=0.35,
                                             edgecolors=(0.2, 0.2, 0.2, 0.4), linewidths=0.3))
        for setlim, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo - pad, hi + pad):
            setlim(a, b)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        ax.set_axis_off()
        ax.set_title('rest mesh' if t == 0 else f'frame {t} / {len(motion) - 1}')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


# ── main ────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ecg', nargs='+', required=True, help='ECG file(s) or folder(s)')
    ap.add_argument('--rest-mesh', default=None,
                    help='rest mesh file, or folder of per-ECG files. Omit for the ECG-only mode')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--fs', type=float, default=None, help='sampling rate (Hz) of .npy/.csv/.txt ECGs')
    ap.add_argument('--ecg-kind', choices=('auto', 'median', 'rhythm', 'ready'), default='auto',
                    help="'ready': already preprocessed (12, 600) input, e.g. data_prep's ecg_resting/*.npy")
    ap.add_argument('--config', default=str(REPO / 'src/configs/config_ecg_heart_delaywavev3.yaml'))
    ap.add_argument('--checkpoint', default=None,
                    help='checkpoint dir (default: <output_dir of the config>/best_model, '
                         'which download_ckpts.sh fills)')
    ap.add_argument('--population-mesh', default=str(REPO / 'assets/population_rest_mesh.npy'))
    ap.add_argument('--faces', default=str(REPO / 'assets/control_mesh_faces.npy'))
    ap.add_argument('--steps', type=int, default=None, help='DDIM steps (default: the config, 50)')
    ap.add_argument('--guidance', type=float, default=None, help='CFG scale (default: 2.0, as in the paper)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=1,
                    help='ECGs per sampling batch. The seeded noise is drawn per batch, so a result is '
                         'reproducible for the same seed and batch size (default 1: independent of the others)')
    ap.add_argument('--device', default=None, help='default: cuda if available, else cpu')
    ap.add_argument('--ply', action='store_true', help='also write one PLY mesh per frame')
    ap.add_argument('--no-gif', action='store_true')
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    from diffusers import DDIMScheduler, DDPMScheduler
    from dataset.ukbiobank_dataset import _quad_to_tri
    from model.ecg_mesh_model import ECGCondMDM
    from pipeline_ecg import ECGTrajPipeline

    # inputs
    ecg_paths = []
    for e in map(Path, a.ecg):
        ecg_paths += sorted(q for q in e.iterdir() if q.suffix.lower() in ECG_EXT) if e.is_dir() else [e]
    if not ecg_paths:
        sys.exit('no ECG files found')
    cfg = OmegaConf.load(a.config)
    OmegaConf.set_struct(cfg, False)
    if cfg.model_config.get('delay_mat_path') or cfg.dataset.get('delay_mat_path'):
        sys.exit('this config needs an external delay matrix (Delay-T / Delay-QRS). Use a delaywave config.')
    template = np.load(a.population_mesh).astype(np.float32)
    faces = np.load(a.faces)
    ckpt = Path(a.checkpoint) if a.checkpoint else \
        Path(os.path.abspath(REPO / 'src' / cfg.output_dir)) / 'best_model'
    if not ckpt.is_dir():
        sys.exit(f'checkpoint not found: {ckpt}\nrun "bash download_ckpts.sh" or pass --checkpoint')

    # model: the checkpoint holds every weight, so the PhysioWave pretraining file is not needed
    for enc in cfg.model_config.values():
        if OmegaConf.is_dict(enc) and 'pretrained_path' in enc:
            enc.pretrained_path = None
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    n_points, n_frames = cfg.dataset.n_points, cfg.dataset.n_training_frames
    model = ECGCondMDM(n_points=n_points, n_frames=n_frames, n_feats=3, model_config=cfg.model_config)
    weights = ckpt / 'model.safetensors'
    if weights.is_file():
        from safetensors.torch import load_file
        state = load_file(str(weights), device='cpu')
    else:
        state = torch.load(ckpt / 'pytorch_model.bin', map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        sys.exit(f'checkpoint does not match the config ({len(missing)} missing weights, e.g. {missing[:3]})')
    model.to(device).eval()
    scheduler = DDIMScheduler.from_config(
        DDPMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False).config)
    pipeline = ECGTrajPipeline(model=model, scheduler=scheduler)
    steps = a.steps or cfg.num_inference_steps
    guidance = a.guidance if a.guidance is not None else (2.0 if cfg.condition_drop_rate > 0 else 1.0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = 'ECG + rest mesh' if a.rest_mesh else 'ECG only (population-average rest mesh)'
    print(f'{len(ecg_paths)} ECG(s), mode: {mode}\nmodel: {a.config}\ncheckpoint: {ckpt}\ndevice: {device}')
    records, items = [], []
    for p in ecg_paths:
        sig, fs, kind = read_ecg(p, a.fs if a.ecg_kind != 'ready' else (a.fs or FS))
        ecg, kind = preprocess_ecg(sig, fs, kind or a.ecg_kind)
        rest, rest_src = rest_mesh_for(p, a.rest_mesh, template)
        np.save(out / f'{p.stem}_ecg_input.npy', ecg)
        items.append((p, ecg, rest))
        records.append({'ecg': str(p), 'ecg_kind': kind, 'fs_in': fs, 'rest_mesh': rest_src})

    faces_tri = _quad_to_tri(faces)
    for i in range(0, len(items), a.batch_size):
        chunk = items[i:i + a.batch_size]
        with torch.no_grad():
            traj = pipeline(init_pc=torch.from_numpy(np.stack([r for _, _, r in chunk]))[:, None],
                            ecg=torch.from_numpy(np.stack([e for _, e, _ in chunk])),
                            device=device, batch_size=len(chunk), n_frames=n_frames,
                            num_inference_steps=steps, guidance_scale=guidance,
                            generator=torch.Generator().manual_seed(a.seed)).float().cpu().numpy()
        for (p, _, rest), pred in zip(chunk, traj):
            motion = np.concatenate([rest[None], pred], axis=0).astype(np.float32)   # (50, N, 3)
            np.save(out / f'{p.stem}_motion.npy', motion)
            if a.ply:
                d = out / f'{p.stem}_ply'
                d.mkdir(exist_ok=True)
                for t, verts in enumerate(motion):
                    write_ply(d / f'frame_{t:03d}.ply', verts, faces)
            if not a.no_gif:
                write_gif(out / f'{p.stem}.gif', motion, faces_tri)
            print(f'  {p.name} -> {p.stem}_motion.npy')

    json.dump({'mode': mode, 'config': a.config, 'checkpoint': str(ckpt), 'population_mesh': a.population_mesh,
               'ddim_steps': steps, 'guidance_scale': guidance, 'seed': a.seed, 'batch_size': a.batch_size,
               'device': str(device), 'inputs': records},
              open(out / 'generation_config.json', 'w'), indent=2)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
