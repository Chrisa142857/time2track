"""
Visualise predicted 4-D heart mesh trajectories from a trained ECGCondMDM.

For each sample the script saves a side-by-side GIF (pred | GT) and optionally
an interactive Three.js HTML viewer.

Usage:
    python vis_ecg_mesh.py --config src/configs/config_ecg_heart.yaml \
                           --checkpoint outputs/.../best_model \
                           --output_dir outputs/vis_eval \
                           [--n_samples 10] \
                           [--split val] \
                           [--num_inference_steps 50] \
                           [--guidance_scale 2.0] \
                           [--save_html]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull, QhullError
from tqdm.auto import tqdm

from dataset.ukbiobank_dataset import UKbiobankHeartMesh
from model.ecg_mesh_model import ECGCondMDM, ECGLocalizedDelayEncoder
from pipeline_ecg import ECGTrajPipeline
from utils.visualization import save_mesh_video, save_pointcloud_json, save_threejs_html


def _save_templates_png(encoder, output_dir):
    """Plot learned event-templates (one subplot per template) to templates.png."""
    templates = [t.detach().float().cpu().numpy() for t in encoder.templates]
    K = len(templates)
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 2.5), squeeze=False)
    for k, psi in enumerate(templates):
        axes[0, k].plot(psi)
        axes[0, k].set_title(f'$\\psi^{{({k})}}$  L={len(psi)}')
        axes[0, k].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'templates.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved learned templates → {out_path}')


# ---------------------------------------------------------------------------

def build_segment_masks(rest_verts):
    """Return dict[name] -> boolean (N,) mask. Splits heart by rest-frame
    x-coordinate into left/right halves (anatomical heuristic)."""
    # x = rest_verts[:, 0]
    # mid = np.median(x)
    # return {
    #     'Left (x < median)':  x <  mid,
    #     'Right (x >= median)': x >= mid,
    # }
    return {
        'Whole-heart':  np.ones(len(rest_verts)).astype(bool),
    }


def segment_volumes(traj, mask):
    """Convex-hull volume per frame for the sub-point-cloud selected by mask.
    traj : (T, N, 3)   mask : (N,)   -> returns (T,) volumes."""
    pts = traj[:, mask]
    vols = np.zeros(pts.shape[0], dtype=np.float32)
    for t in range(pts.shape[0]):
        try:
            vols[t] = ConvexHull(pts[t]).volume
        except QhullError:
            vols[t] = np.nan
    return vols


def plot_segment_volumes(vol_records, seg_names, out_path):
    """vol_records : list of dicts {subid, seg_name -> (pred (T,), gt (T,))}.
    Row 0 = predictions, Row 1 = ground truth. Columns = segments."""
    n_seg = len(seg_names)
    fig, axes = plt.subplots(2, n_seg, figsize=(5 * n_seg, 8), squeeze=False)
    cmap = plt.colormaps.get_cmap('tab10')

    for j, seg in enumerate(seg_names):
        ax_pred = axes[0, j]
        ax_gt   = axes[1, j]
        for k, rec in enumerate(vol_records):
            pred, gt = rec['volumes'][seg]
            t = np.arange(len(pred))
            color = cmap(k % 10)
            ax_pred.plot(t, pred, '-', color=color, alpha=0.9,
                         label=rec['subid'])
            ax_gt.plot(t, gt, '-', color=color, alpha=0.9,
                       label=rec['subid'])

        ax_pred.set_title(f'{seg} — Pred')
        ax_gt.set_title(f'{seg} — GT')
        for ax in (ax_pred, ax_gt):
            ax.set_xlabel('Frame')
            ax.set_ylabel('Convex-hull volume')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc='best')

    fig.tight_layout()
    if not out_path.lower().endswith('.svg'):
        out_path = os.path.splitext(out_path)[0] + '.svg'
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)

    # ── Model ────────────────────────────────────────────────────────────
    n_points = cfg.dataset.n_points
    n_frames = cfg.dataset.n_training_frames

    model = ECGCondMDM(
        n_points=n_points,
        n_frames=n_frames,
        n_feats=3,
        model_config=cfg.model_config,
    )

    ckpt_dir  = args.checkpoint
    bin_path  = os.path.join(ckpt_dir, 'pytorch_model.bin')
    safe_path = os.path.join(ckpt_dir, 'model.safetensors')
    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state = load_file(safe_path, device='cpu')
        model.load_state_dict(state, strict=False)
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f'No model weights found in {ckpt_dir}')

    model.to(device)
    model.eval()
    print(f'Loaded checkpoint from {ckpt_dir}')

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = UKbiobankHeartMesh(args.split, cfg.dataset)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,          # one sample at a time for easy per-subject naming
        shuffle=False,
        num_workers=4,
    )

    # ── Pipeline ─────────────────────────────────────────────────────────
    ddpm_cfg = DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type='sample',
        clip_sample=False,
    )
    ddim_scheduler = DDIMScheduler.from_config(ddpm_cfg.config)
    pipeline = ECGTrajPipeline(model=model, scheduler=ddim_scheduler)

    num_inference_steps = args.num_inference_steps or cfg.num_inference_steps
    guidance_scale      = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = 2.0 if cfg.condition_drop_rate > 0 else 1.0

    faces_tri = dataset.faces_tri.numpy()   # (F, 3) shared mesh topology

    os.makedirs(args.output_dir, exist_ok=True)
    n_saved = 0

    vol_records = []
    seg_names   = None

    # Save learned templates (only relevant for ECGLocalizedDelayEncoder)
    if isinstance(model.ecg_encoder, ECGLocalizedDelayEncoder):
        _save_templates_png(model.ecg_encoder, args.output_dir)

    print(f'Generating {args.n_samples} sample(s) from {args.split} split …')
    for batch, info in tqdm(loader):
        if n_saved >= args.n_samples:
            break

        subid = info['subid'][0]

        delay_mat = batch.get('delay_mat')
        if delay_mat is not None:
            delay_mat = delay_mat.to(device)
        with torch.no_grad():
            traj = pipeline(
                init_pc=batch['points_src'].to(device),
                ecg=batch['ecg'].to(device),
                device=device,
                batch_size=1,
                n_frames=n_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(0),
                delay_mat=delay_mat,
            )                                               # (1, T, N, 3)

        traj_np = traj[0].cpu().numpy()                    # (T, N, 3)
        gt_np   = batch['points_tgt'][0].cpu().numpy()     # (T, N, 3)
        np.save(os.path.join(args.output_dir, f'{subid}_pred.npy'), traj_np)
        np.save(os.path.join(args.output_dir, f'{subid}_gt.npy'), gt_np)

        # ── GIF ──────────────────────────────────────────────────────────
        gif_path = os.path.join(args.output_dir, f'{subid}.gif')
        print(traj_np.shape, traj_np.min(), traj_np.max())
        print(gt_np.shape, gt_np.min(), gt_np.max())
        save_mesh_video(traj_np, gt_np, faces_tri, gif_path, vis_flag='heartmesh')
        print(f'  Saved GIF → {gif_path}')

        # ── Segment volume trajectories ──────────────────────────────────
        rest_verts = batch['points_src'][0, 0].cpu().numpy()   # (N, 3)
        seg_masks  = build_segment_masks(rest_verts)
        if seg_names is None:
            seg_names = list(seg_masks.keys())
        vols = {name: (segment_volumes(traj_np, m),
                       segment_volumes(gt_np,   m))
                for name, m in seg_masks.items()}
        vol_records.append({'subid': subid, 'volumes': vols})

        # ── Optional Three.js HTML ────────────────────────────────────────
        if args.save_html:
            pred_json = os.path.join(args.output_dir, f'{subid}_pred.json')
            gt_json   = os.path.join(args.output_dir, f'{subid}_gt.json')
            html_path = os.path.join(args.output_dir, f'{subid}.html')
            save_pointcloud_json(traj_np, pred_json)
            save_pointcloud_json(gt_np,   gt_json)
            save_threejs_html(
                os.path.basename(pred_json),
                os.path.basename(gt_json),
                html_path,
            )
            print(f'  Saved HTML → {html_path}')

        torch.cuda.empty_cache()
        n_saved += 1

    if vol_records:
        vol_plot_path = os.path.join(args.output_dir, 'segment_volumes.png')
        plot_segment_volumes(vol_records, seg_names, vol_plot_path)
        print(f'Saved segment volume plot → {vol_plot_path}')

    print(f'\nDone. {n_saved} sample(s) written to {args.output_dir}')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',              type=str,   required=True)
    parser.add_argument('--checkpoint',          type=str,   required=True,
                        help='Path to accelerate checkpoint dir')
    parser.add_argument('--output_dir',          type=str,   required=True,
                        help='Directory to write GIFs (and optionally HTML viewers)')
    parser.add_argument('--n_samples',           type=int,   default=10,
                        help='Number of samples to visualise')
    parser.add_argument('--split',               type=str,   default='val',
                        choices=['train', 'val'],
                        help='Dataset split to draw samples from')
    parser.add_argument('--num_inference_steps', type=int,   default=None,
                        help='DDIM steps (default: from config)')
    parser.add_argument('--guidance_scale',      type=float, default=None,
                        help='CFG scale (default: 2.0 if CFG-trained, else 1.0)')
    parser.add_argument('--save_html',           action='store_true',
                        help='Also save interactive Three.js HTML viewers')
    args = parser.parse_args()
    main(args)
