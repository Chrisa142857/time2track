"""
Evaluate a trained ECG-conditioned heart mesh trajectory model.

Loads a checkpoint (best_model or checkpoint-N), runs DDIM sampling on the
validation split, and reports:

  Geometric (per frame, averaged over the trajectory):
    • per-vertex L2 error          (mean over vertices & frames)
    • Chamfer distance             (bidirectional mean, mm)
    • Hausdorff distance           (max over vertices, averaged over frames)
    • ASSD                         (average symmetric surface distance)

  Temporal alignment:
    • temporal Pearson r           (correlation of mean-displacement curves)
    • peak-frame timing error      (|predicted peak frame - GT peak frame|)
    • velocity L2 error            (frame-to-frame displacement L2)

  Clinical (using material labels for LV/RV endocardial surfaces):
    • LVEF  (%)
    • RVEF  (%)
    • denoising loss

Usage:
    python eval_ecg_mesh.py --config src/configs/config_ecg_heart.yaml \\
                            --checkpoint outputs/.../best_model \\
                            --material_npy /path/to/control_materials.npy \\
                            [--num_inference_steps 50] \\
                            [--guidance_scale 2.0] \\
                            [--num_val_batches -1] \\
                            [--output_csv results/eval.csv]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dataset.ukbiobank_dataset import UKbiobankHeartMesh
from model.ecg_mesh_model import ECGCondMDM, ECGLocalizedDelayEncoder
from pipeline_ecg import ECGTrajPipeline


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Mean bidirectional Chamfer distance.
    pred, gt: (B, N, 3)
    Returns scalar (mean over batch).
    """
    # (B, N, M) pairwise squared distances
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)   # (B, N, M, 3)
    dist2 = diff.pow(2).sum(-1)                   # (B, N, M)
    d_pred = dist2.min(dim=2).values.sqrt().mean(dim=1)   # (B,)
    d_gt   = dist2.min(dim=1).values.sqrt().mean(dim=1)   # (B,)
    return ((d_pred + d_gt) / 2).mean()


def hausdorff_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Mean directed Hausdorff distance (max of min-distances), averaged over batch.
    pred, gt: (B, N, 3)
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist2 = diff.pow(2).sum(-1)
    h_pred = dist2.min(dim=2).values.sqrt().max(dim=1).values   # (B,)
    h_gt   = dist2.min(dim=1).values.sqrt().max(dim=1).values   # (B,)
    return torch.maximum(h_pred, h_gt).mean()


def assd(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Average Symmetric Surface Distance, averaged over batch.
    pred, gt: (B, N, 3)
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist2 = diff.pow(2).sum(-1)
    d_pred = dist2.min(dim=2).values.sqrt().mean(dim=1)   # (B,)
    d_gt   = dist2.min(dim=1).values.sqrt().mean(dim=1)   # (B,)
    return ((d_pred + d_gt) / 2).mean()


# ---------------------------------------------------------------------------
# Temporal alignment helpers
# ---------------------------------------------------------------------------

def _mean_displacement_curve(traj: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Mean per-vertex L2 displacement from rest frame at each time step.
    traj: (B, T, N, 3),  src: (B, 1, N, 3)
    Returns (B, T).
    """
    return (traj - src).pow(2).sum(-1).sqrt().mean(dim=-1)   # (B, T)


def temporal_pearson(pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                     src: torch.Tensor) -> torch.Tensor:
    """
    Per-sample Pearson r between predicted and GT mean-displacement curves,
    returned as mean over batch.
    """
    curve_p = _mean_displacement_curve(pred_traj, src)   # (B, T)
    curve_g = _mean_displacement_curve(gt_traj,   src)   # (B, T)
    mu_p = curve_p.mean(dim=1, keepdim=True)
    mu_g = curve_g.mean(dim=1, keepdim=True)
    cov  = ((curve_p - mu_p) * (curve_g - mu_g)).mean(dim=1)
    std_p = (curve_p - mu_p).pow(2).mean(dim=1).sqrt().clamp(min=1e-8)
    std_g = (curve_g - mu_g).pow(2).mean(dim=1).sqrt().clamp(min=1e-8)
    return (cov / (std_p * std_g)).mean()


def peak_frame_error(pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                     src: torch.Tensor) -> torch.Tensor:
    """
    Mean |frame of peak mean-displacement in pred - frame in GT|.
    """
    curve_p = _mean_displacement_curve(pred_traj, src)   # (B, T)
    curve_g = _mean_displacement_curve(gt_traj,   src)
    peak_p  = curve_p.argmax(dim=1).float()
    peak_g  = curve_g.argmax(dim=1).float()
    return (peak_p - peak_g).abs().mean()


def velocity_l2_error(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    Mean L2 error on per-frame velocities (finite differences).
    pred_traj, gt_traj: (B, T, N, 3)
    """
    vel_p = pred_traj[:, 1:] - pred_traj[:, :-1]   # (B, T-1, N, 3)
    vel_g = gt_traj[:,   1:] - gt_traj[:,   :-1]
    return (vel_p - vel_g).pow(2).sum(-1).sqrt().mean()


# ---------------------------------------------------------------------------
# Micro-fluctuation metrics
# ---------------------------------------------------------------------------

def temporal_std_ratio(pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                       src: torch.Tensor) -> torch.Tensor:
    """
    Mean over batch of std(pred_curve) / std(gt_curve) for the mean-displacement
    time series. Ideal = 1.0; a temporally flat prediction gives values near 0.
    """
    curve_p = _mean_displacement_curve(pred_traj, src)   # (B, T)
    curve_g = _mean_displacement_curve(gt_traj,   src)
    std_p = curve_p.std(dim=1).clamp(min=0)
    std_g = curve_g.std(dim=1).clamp(min=1e-8)
    return (std_p / std_g).mean()


def hf_power_ratio(pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                   src: torch.Tensor, hf_frac: float = 0.5) -> tuple:
    """
    Power in the high-frequency band (upper `hf_frac` of FFT coefficients) for
    the mean-displacement curve, normalised by total power.

    Returns (hf_power_pred, hf_power_gt) — each is mean over batch.
    A model that predicts flat trajectories gets hf_power_pred ≈ 0 while
    hf_power_gt reflects the true micro-fluctuation content.
    """
    curve_p = _mean_displacement_curve(pred_traj, src).float()   # (B, T)
    curve_g = _mean_displacement_curve(gt_traj,   src).float()

    def _hf_power(curve):
        # rfft returns (B, T//2+1) complex coefficients
        spec  = torch.fft.rfft(curve, dim=1)
        power = spec.abs().pow(2)           # (B, F)
        cutoff = max(1, int(power.shape[1] * (1 - hf_frac)))
        total_power = power.sum(dim=1).clamp(min=1e-12)
        hf = power[:, cutoff:].sum(dim=1)
        return (hf / total_power).mean()

    return _hf_power(curve_p), _hf_power(curve_g)


def detrended_fluctuation_rms(pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                               src: torch.Tensor,
                               smoothing_frames: int = 5) -> tuple:
    """
    RMS of the residual after removing a moving-average trend from the
    mean-displacement curve. Returns (rms_pred, rms_gt) as means over batch.

    smoothing_frames : half-width of the uniform kernel used to estimate trend.
    If the prediction is flat, rms_pred ≈ 0 while rms_gt captures true
    micro-oscillation amplitude.
    """
    curve_p = _mean_displacement_curve(pred_traj, src).float()   # (B, T)
    curve_g = _mean_displacement_curve(gt_traj,   src).float()

    k = smoothing_frames * 2 + 1

    def _residual_rms(curve):
        # 1-D uniform moving average via avg_pool1d
        trend = torch.nn.functional.avg_pool1d(
            curve.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2
        ).squeeze(1)                           # (B, T) approx – may be shorter
        min_t = min(curve.shape[1], trend.shape[1])
        residual = curve[:, :min_t] - trend[:, :min_t]
        return residual.pow(2).mean(dim=1).sqrt().mean()

    return _residual_rms(curve_p), _residual_rms(curve_g)


# ---------------------------------------------------------------------------
# Clinical metrics (LV / RV ejection fraction)
# ---------------------------------------------------------------------------

def _build_chamber_face_masks(material_npy: str, faces_tri: torch.Tensor):
    """
    Load material labels and return boolean index arrays for LV and RV surface faces.

    Material IDs:
        LV_ENDOCARDIAL = 1
        RV_FREEWALL    = 4
        RV_SEPTUM      = 5

    Returns
    -------
    lv_faces : (F_lv, 3) int64 tensor  – faces with all vertices in LV_ENDOCARDIAL
    rv_faces : (F_rv, 3) int64 tensor  – faces with all vertices in {RV_FREEWALL, RV_SEPTUM}
    """
    mats = np.load(material_npy)           # (N,) int64
    lv_mask  = (mats == 1)
    rv_mask  = np.isin(mats, [4, 5])

    f = faces_tri.numpy()                  # (F, 3)
    lv_face_mask = lv_mask[f[:, 0]] & lv_mask[f[:, 1]] & lv_mask[f[:, 2]]
    rv_face_mask = rv_mask[f[:, 0]] & rv_mask[f[:, 1]] & rv_mask[f[:, 2]]

    lv_faces = torch.from_numpy(f[lv_face_mask])
    rv_faces = torch.from_numpy(f[rv_face_mask])
    print(f'[EF] LV faces: {lv_faces.shape[0]}, RV faces: {rv_faces.shape[0]}')
    return lv_faces, rv_faces


def _signed_volume(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Signed volume of a closed triangulated surface via the divergence theorem.
    verts : (N, 3)
    faces : (F, 3)
    Returns scalar.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return (v0 * torch.cross(v1, v2, dim=-1)).sum(-1).sum() / 6.0


def compute_ef(traj_full: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute ejection fraction for a single sample's trajectory.
    traj_full : (T_full, N, 3)  – includes rest frame (index 0)
    faces     : (F, 3) int64
    Returns EF in percent.
    """
    faces = faces.to(traj_full.device)
    vols  = torch.stack([_signed_volume(traj_full[t], faces)
                         for t in range(traj_full.shape[0])]).abs()
    edv = vols.max()
    esv = vols.min()
    if edv < 1e-8:
        return torch.tensor(float('nan'))
    return (edv - esv) / edv * 100.0


# ---------------------------------------------------------------------------
# Main
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

    # Optional override of the delay-token gate before sampling.
    if args.alpha_delay_override is not None:
        enc = model.ecg_encoder
        assert hasattr(enc, 'alpha_delay'), f'{type(enc).__name__} has no alpha_delay'
        old = float(enc.alpha_delay.detach())
        with torch.no_grad():
            enc.alpha_delay.fill_(float(args.alpha_delay_override))
        new = float(enc.alpha_delay.detach())
        print(f'alpha_delay override: {old:.6e} -> {new:.6e}')
        assert abs(new - float(args.alpha_delay_override)) < 1e-5, 'override did not take'  # float32 tol

    model.to(device)
    model.eval()
    print(f'Loaded checkpoint from {ckpt_dir}')

    # ── Dataset ──────────────────────────────────────────────────────────
    val_dataset = UKbiobankHeartMesh('val', cfg.dataset)

    # Optional restriction of inference to a subject subset (keeps dataset order).
    if args.subject_list:
        with open(args.subject_list) as f:
            keep = {ln.strip() for ln in f if ln.strip()}
        before = len(val_dataset.subjects)
        val_dataset.subjects = [s for s in val_dataset.subjects if s[0] in keep]
        print(f'subject_list: kept {len(val_dataset.subjects)}/{before} '
              f'(requested {len(keep)})')

    # Optional fixed template rest mesh substituted for p0 for ALL subjects.
    # A .npz keyed by subject id instead supplies a PER-SUBJECT estimated p0
    # (stratified template / atlas / shape model / ECG retrieval).
    init_pc_override = None
    init_pc_map = None
    tmpl_scale = None
    if args.init_pc_override and args.init_pc_override.endswith('.npz'):
        z = np.load(args.init_pc_override)
        init_pc_map = {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files}
        missing = [s[0] for s in val_dataset.subjects if s[0] not in init_pc_map]
        assert not missing, f'init_pc npz missing {len(missing)} subjects e.g. {missing[:3]}'
        assert args.init_pc_scale_by_subject is False, 'per-subject npz + rescale not supported'
        print(f'init_pc_override (per-subject): {args.init_pc_override} '
              f'n={len(init_pc_map)} shape={tuple(next(iter(init_pc_map.values())).shape)}')
    elif args.init_pc_override:
        t = np.load(args.init_pc_override).astype(np.float32)   # (N,3) or (1,N,3)
        if t.ndim == 3:
            t = t[0]
        init_pc_override = torch.from_numpy(t).to(device)       # (N,3)
        tmpl_scale = init_pc_override.pow(2).sum(-1).mean().sqrt()
        print(f'init_pc_override: {args.init_pc_override} shape={tuple(t.shape)} '
              f'rms_scale={float(tmpl_scale):.4f} '
              f'scale_by_subject={args.init_pc_scale_by_subject}')

    eval_bs = args.eval_batch_size or cfg.eval_batch_size
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
    )

    # ── Material-based face masks for EF computation ─────────────────────
    material_npy = args.material_npy
    lv_faces = rv_faces = None
    if material_npy and os.path.exists(material_npy):
        lv_faces, rv_faces = _build_chamber_face_masks(
            material_npy, val_dataset.faces_tri
        )
    else:
        print('[EF] No material_npy provided or file not found — skipping EF metrics.')

    # ── Schedulers ───────────────────────────────────────────────────────
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type='sample',
        clip_sample=False,
    )
    ddim_scheduler = DDIMScheduler.from_config(ddpm_scheduler.config)
    pipeline = ECGTrajPipeline(model=model, scheduler=ddim_scheduler)

    num_inference_steps = args.num_inference_steps or cfg.num_inference_steps
    guidance_scale      = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = 2.0 if cfg.condition_drop_rate > 0 else 1.0

    if args.npy_dir:
        os.makedirs(args.npy_dir, exist_ok=True)

    # ── Diagnostic dumps (templates, per-subject delay matrices, phi*, …) ─
    if args.diagnostics_dir and isinstance(model.ecg_encoder, ECGLocalizedDelayEncoder):
        os.makedirs(args.diagnostics_dir, exist_ok=True)
        # Save templates once (shared across all subjects)
        tmpl_list = [t.detach().float().cpu().numpy() for t in model.ecg_encoder.templates]
        np.savez(
            os.path.join(args.diagnostics_dir, 'templates.npz'),
            **{f'psi_{k}': arr for k, arr in enumerate(tmpl_list)},
        )
        print(f'[diagnostics] Wrote templates to {args.diagnostics_dir}/templates.npz')
        with torch.no_grad():
            for _, (batch, info) in enumerate(tqdm(val_loader, desc='diagnostics')):
                ecg = batch['ecg'].to(device)
                diag = model.ecg_encoder.dump_diagnostics(ecg)
                bsz = ecg.shape[0]
                for b in range(bsz):
                    sid = info['subid'][b] if 'subid' in info else f'sample{b:05d}'
                    np.savez(
                        os.path.join(args.diagnostics_dir, f'{sid}.npz'),
                        delta_matrix=diag['delta_matrix'][b],
                        per_cycle_std=diag['per_cycle_std'][b],
                        magnitudes=diag['magnitudes'][b],
                        phi_star=diag['phi_star'][b],
                        cycles=diag['cycles'][b],
                        valid=diag['valid'][b],
                    )
        print(f'[diagnostics] Saved per-subject diagnostics to {args.diagnostics_dir}')
    elif args.diagnostics_dir:
        print('[diagnostics] skipped: encoder is not ECGLocalizedDelayEncoder.')

    max_batches = args.num_val_batches

    # ── Denoising loss ───────────────────────────────────────────────────
    # print('Computing denoising loss …')
    # gen_loss = torch.Generator(device=device).manual_seed(0)
    # loss_sum, n_loss = 0.0, 0
    # with torch.no_grad():
    #     for i, (batch, _) in enumerate(tqdm(val_loader)):
    #         if max_batches > 0 and i >= max_batches:
    #             break
    #         latents   = batch['points_tgt'].to(device)
    #         bsz       = latents.shape[0]
    #         noise     = torch.randn_like(latents)
    #         timesteps = torch.randint(
    #             0, ddpm_scheduler.config.num_train_timesteps,
    #             (bsz,), device=device, generator=gen_loss,
    #         ).long()
    #         noisy = ddpm_scheduler.add_noise(latents, noise, timesteps)
    #         delay_mat = batch['delay_mat'].to(device) if 'delay_mat' in batch else None
    #         pred = model(
    #                 noisy, timesteps,
    #                 batch['points_src'].to(device),
    #                 batch['ecg'].to(device),
    #                 delay_mat=delay_mat,
    #             )
    #         loss_sum += F.mse_loss(pred.float(), latents.float()).item() * bsz
    #         n_loss   += bsz

    # val_loss = loss_sum / max(n_loss, 1)
    val_loss = None
    # ── Trajectory metrics ────────────────────────────────────────────────
    print('Sampling trajectories and computing metrics …')

    # Accumulate over batches
    l2_sum   = chf_sum = hd_sum = assd_sum = 0.0
    vel_sum  = pearson_sum = peak_err_sum  = 0.0
    std_ratio_sum = 0.0
    hf_pred_sum = hf_gt_sum = 0.0
    detrend_pred_sum = detrend_gt_sum = 0.0
    lvef_pred_vals, rvef_pred_vals = [], []
    lvef_gt_vals,   rvef_gt_vals   = [], []
    n_samples = 0

    with torch.no_grad():
        for i, (batch, info) in enumerate(tqdm(val_loader)):
            if max_batches > 0 and i >= max_batches:
                break
            
            src = batch['points_src'].to(device)   # (B, 1, N, 3)
            gt  = batch['points_tgt'].to(device)   # (B, T, N, 3)
            bsz = src.shape[0]
            
            domask = torch.ones(bsz, dtype=torch.bool, device=device)  # (B,)
            for b in range(bsz):
                sid = info['subid'][b] if 'subid' in info else f'{n_samples + b:05d}'
                if os.path.exists(os.path.join(args.npy_dir, f'{sid}_pred.npy')):
                    domask[b] = False
            if domask.sum() == 0:
                print(f'[batch {i}] All samples already exist in {args.npy_dir}, skipping …')
                n_samples += bsz
                torch.cuda.empty_cache()
                continue
            
            delay_mat = batch.get('delay_mat')
            if delay_mat is not None:
                delay_mat = delay_mat.to(device)

            # Substitute a fixed template p0 for every subject (ECG/GT unchanged).
            src_used = src
            if init_pc_map is not None:
                N = src.shape[-2]
                sids_b = [info['subid'][b] for b in range(bsz)]
                src_used = torch.stack([init_pc_map[s] for s in sids_b]) \
                                .view(bsz, 1, N, 3).to(device)
            elif init_pc_override is not None:
                N = src.shape[-2]
                tmpl = init_pc_override.view(1, 1, N, 3).expand(bsz, 1, N, 3).clone()
                if args.init_pc_scale_by_subject:
                    subj = src.reshape(bsz, N, 3)
                    subj_scale = subj.pow(2).sum(-1).mean(dim=1).sqrt()   # (B,)
                    factor = (subj_scale / tmpl_scale).view(bsz, 1, 1, 1)
                    tmpl = tmpl * factor
                src_used = tmpl.to(device)

            traj = pipeline(
                    init_pc=src_used,
                    ecg=batch['ecg'].to(device),
                    device=device,
                    batch_size=bsz,
                    n_frames=n_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(0),
                    delay_mat=delay_mat,
                )   # (B, T, N, 3)

            traj = traj.float()
            gt   = gt.float()
            src  = src.float()

            # ── Save trajectories ─────────────────────────────────────────
            if args.npy_dir:
                for b in range(bsz):
                    sid = info['subid'][b] if 'subid' in info else f'{n_samples + b:05d}'
                    np.save(os.path.join(args.npy_dir, f'{sid}_pred.npy'), traj[b].cpu().numpy())
                    np.save(os.path.join(args.npy_dir, f'{sid}_gt.npy'),   gt[b].cpu().numpy())

            # # ── Per-vertex L2 ────────────────────────────────────────────
            # l2 = (traj - gt).pow(2).sum(-1).sqrt().mean()
            # l2_sum += l2.item() * bsz

            # # ── Surface metrics (averaged over frames) ───────────────────
            # T = traj.shape[1]
            # chf_acc = hd_acc = assd_acc = 0.0
            # for t in range(T):
            #     chf_acc  += chamfer_distance(traj[:, t], gt[:, t]).item()
            #     hd_acc   += hausdorff_distance(traj[:, t], gt[:, t]).item()
            #     assd_acc += assd(traj[:, t], gt[:, t]).item()
            # chf_sum  += (chf_acc  / T) * bsz
            # hd_sum   += (hd_acc   / T) * bsz
            # assd_sum += (assd_acc / T) * bsz

            # # ── Temporal alignment ───────────────────────────────────────
            # pearson_sum  += temporal_pearson(traj, gt, src).item() * bsz
            # peak_err_sum += peak_frame_error(traj, gt, src).item() * bsz
            # vel_sum      += velocity_l2_error(traj, gt).item() * bsz

            # # ── Micro-fluctuation metrics ─────────────────────────────────
            # std_ratio_sum += temporal_std_ratio(traj, gt, src).item() * bsz
            # hf_p, hf_g    = hf_power_ratio(traj, gt, src)
            # hf_pred_sum   += hf_p.item() * bsz
            # hf_gt_sum     += hf_g.item() * bsz
            # dt_p, dt_g    = detrended_fluctuation_rms(traj, gt, src)
            # detrend_pred_sum += dt_p.item() * bsz
            # detrend_gt_sum   += dt_g.item() * bsz

            # # ── EF per sample (requires full trajectory incl. rest frame) ─
            # if lv_faces is not None:
            #     for b in range(bsz):
            #         src_b  = src[b]   # (1, N, 3)
            #         # full trajectory: [rest, t1 … tT]
            #         traj_full_pred = torch.cat([src_b, traj[b]], dim=0)   # (T+1, N, 3)
            #         traj_full_gt   = torch.cat([src_b, gt[b]],   dim=0)

            #         lvef_pred_vals.append(compute_ef(traj_full_pred, lv_faces).item())
            #         lvef_gt_vals.append(  compute_ef(traj_full_gt,   lv_faces).item())
            #         rvef_pred_vals.append(compute_ef(traj_full_pred, rv_faces).item())
            #         rvef_gt_vals.append(  compute_ef(traj_full_gt,   rv_faces).item())

            n_samples += bsz
            torch.cuda.empty_cache()
    exit()
    # ── Aggregate ─────────────────────────────────────────────────────────
    N = max(n_samples, 1)
    val_l2       = l2_sum       / N
    val_chamfer  = chf_sum      / N
    val_hd       = hd_sum       / N
    val_assd     = assd_sum     / N
    val_pearson  = pearson_sum  / N
    val_peak_err = peak_err_sum / N
    val_vel      = vel_sum      / N

    val_std_ratio    = std_ratio_sum    / N
    val_hf_pred      = hf_pred_sum      / N
    val_hf_gt        = hf_gt_sum        / N
    val_detrend_pred = detrend_pred_sum / N
    val_detrend_gt   = detrend_gt_sum   / N

    def _safe_mean(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return float(np.mean(clean)) if clean else float('nan')

    lvef_pred = _safe_mean(lvef_pred_vals)
    lvef_gt   = _safe_mean(lvef_gt_vals)
    rvef_pred = _safe_mean(rvef_pred_vals)
    rvef_gt   = _safe_mean(rvef_gt_vals)

    # ── Report ────────────────────────────────────────────────────────────
    print('=' * 60)
    print(f'  Samples evaluated      : {n_samples}')
    print(f'  Val denoising loss     : {val_loss:.6f}')
    print()
    print('  --- Geometric ---')
    print(f'  Per-vertex L2          : {val_l2:.6f}')
    print(f'  Chamfer distance       : {val_chamfer:.6f}')
    print(f'  Hausdorff distance     : {val_hd:.6f}')
    print(f'  ASSD                   : {val_assd:.6f}')
    print()
    print('  --- Temporal alignment ---')
    print(f'  Temporal Pearson r     : {val_pearson:.4f}')
    print(f'  Peak-frame timing err  : {val_peak_err:.2f} frames')
    print(f'  Velocity L2 error      : {val_vel:.6f}')
    print()
    print('  --- Micro-fluctuation (displacement curve) ---')
    print(f'  Temporal std ratio     : {val_std_ratio:.4f}  (pred/GT, ideal=1.0)')
    print(f'  HF power (pred / GT)   : {val_hf_pred:.4f} / {val_hf_gt:.4f}')
    print(f'  Detrended RMS (pred/GT): {val_detrend_pred:.6f} / {val_detrend_gt:.6f}')
    print()
    if lv_faces is not None:
        print('  --- Clinical (EF) ---')
        print(f'  LVEF pred / GT         : {lvef_pred:.1f}% / {lvef_gt:.1f}%')
        print(f'  RVEF pred / GT         : {rvef_pred:.1f}% / {rvef_gt:.1f}%')
    print('=' * 60)

    if args.output_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'checkpoint', 'n_samples',
                    'val_loss', 'val_l2', 'chamfer', 'hausdorff', 'assd',
                    'temporal_pearson', 'peak_frame_err', 'velocity_l2',
                    'temporal_std_ratio', 'hf_power_pred', 'hf_power_gt',
                    'detrended_rms_pred', 'detrended_rms_gt',
                    'lvef_pred', 'lvef_gt', 'rvef_pred', 'rvef_gt',
                    'num_inference_steps', 'guidance_scale',
                ])
            writer.writerow([
                ckpt_dir, n_samples,
                val_loss, val_l2, val_chamfer, val_hd, val_assd,
                val_pearson, val_peak_err, val_vel,
                val_std_ratio, val_hf_pred, val_hf_gt,
                val_detrend_pred, val_detrend_gt,
                lvef_pred, lvef_gt, rvef_pred, rvef_gt,
                num_inference_steps, guidance_scale,
            ])
        print(f'Results appended to {args.output_csv}')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',               type=str,   required=True)
    parser.add_argument('--checkpoint',           type=str,   required=True,
                        help='Path to accelerate checkpoint dir')
    parser.add_argument('--material_npy',         type=str,
                        default='../ukb_heart_mesh_visual/control_materials.npy',
                        help='Path to control_materials.npy with per-vertex material labels')
    parser.add_argument('--num_inference_steps',  type=int,   default=None)
    parser.add_argument('--guidance_scale',       type=float, default=None)
    parser.add_argument('--num_val_batches',      type=int,   default=-1,
                        help='Batches to evaluate; -1 = full val set')
    parser.add_argument('--output_csv',           type=str,   default=None)
    parser.add_argument('--npy_dir',              type=str,   default=None,
                        help='Directory to save per-sample pred/gt .npy files')
    parser.add_argument('--diagnostics_dir',      type=str,   default=None,
                        help='Directory to dump encoder diagnostics (templates, '
                             'delta matrices, phi*, cycles). Only active when '
                             'the encoder is ECGLocalizedDelayEncoder.')
    # ── Rest-mesh and delay-branch options ──────────────────────────────
    parser.add_argument('--subject_list',         type=str,   default=None,
                        help='Optional txt file (one subid per line) to restrict '
                             'inference to a subject subset.')
    parser.add_argument('--init_pc_override',      type=str,   default=None,
                        help='Optional .npy (N,3) or (1,N,3) template rest mesh to '
                             'substitute for the subject-specific p0 for ALL '
                             'subjects. ECG + GT stay subject-specific.')
    parser.add_argument('--alpha_delay_override',  type=float, default=None,
                        help='Force ecg_encoder.alpha_delay to this value '
                             'after loading the checkpoint (0.0 = delay tokens off, '
                             '1.0 = ungated). Default None = keep trained value.')
    parser.add_argument('--init_pc_scale_by_subject', action='store_true',
                        help='If set with --init_pc_override, scale the template '
                             'per-subject by (subject rest RMS-scale / template '
                             'RMS-scale): a cheap size-only p0 proxy.')
    parser.add_argument('--eval_batch_size',       type=int,   default=None,
                        help='Override cfg.eval_batch_size (e.g. 16 to fit a '
                             'shared GPU and avoid OOM).')
    args = parser.parse_args()
    main(args)
