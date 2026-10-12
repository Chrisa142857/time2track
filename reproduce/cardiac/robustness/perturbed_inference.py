#!/usr/bin/env python
"""Inference of the trained Time2Track model under one input perturbation (Appendix A.6).

Loads the trained checkpoint, restricts the validation split to a subject list, applies ONE
inference-time perturbation to batch['ecg'] (or to the encoder's tau_max), runs the DDIM/CFG
pipeline of src/eval_ecg_mesh.py, and saves only {subid}_pred.npy to --npy-dir (skip-if-exists).
The GT is shared across conditions and is read by robustness_metrics.py from the npy_eval of the
unperturbed evaluation, so it is not saved here.

Perturbations (mutually exclusive):
  --mode clean                       no perturbation (matched baseline)
  --mode noise   --snr 10            additive Gaussian electrode noise at SNR dB (per lead)
  --mode dropout --drop-leads 1,7,8  zero those lead channels (0-indexed, lead II = 1)
  --mode taumax  --tau-max 100       override encoder.tau_max at inference (no ECG change)

Batching is fixed (subset order + batch_size, shuffle=False), so the per-batch seed-0 initial
latent noise lands each subject in the same position across every condition and clean versus
perturbed is a matched within-subject comparison. The noise pattern is drawn from a fixed
generator and only rescaled by SNR, so a higher SNR is strictly less of the same noise.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC))

from diffusers import DDIMScheduler, DDPMScheduler  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from dataset.ukbiobank_dataset import UKbiobankHeartMesh  # noqa: E402
from model.ecg_mesh_model import ECGCondMDM  # noqa: E402
from pipeline_ecg import ECGTrajPipeline  # noqa: E402

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
NOISE_SEED = 1234


def perturb_ecg(ecg, mode, snr_db, drop_leads, gen):
    """ecg: (B,12,T). Returns perturbed clone. gen: torch.Generator on ecg.device."""
    if mode == 'noise':
        # per-lead signal power -> per-lead noise std for the target SNR
        p_sig = ecg.pow(2).mean(dim=-1, keepdim=True)                 # (B,12,1)
        sigma = torch.sqrt(p_sig / (10.0 ** (snr_db / 10.0)))         # (B,12,1)
        unit = torch.randn(ecg.shape, generator=gen, device=ecg.device, dtype=ecg.dtype)
        return ecg + unit * sigma
    if mode == 'dropout':
        out = ecg.clone()
        out[:, drop_leads, :] = 0.0
        return out
    return ecg  # clean / taumax (no ECG change)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=str(SRC / 'configs' / 'config_ecg_heart_delaywavev3.yaml'))
    ap.add_argument('--checkpoint', required=True, help='trained model dir with model.safetensors')
    ap.add_argument('--subject-list', required=True, help='one subject id per line')
    ap.add_argument('--npy-dir', required=True, help='fresh output dir for {subid}_pred.npy')
    ap.add_argument('--mode', choices=['clean', 'noise', 'dropout', 'taumax'], default='clean')
    ap.add_argument('--snr', type=float, default=None, help='SNR dB for --mode noise')
    ap.add_argument('--drop-leads', type=str, default='', help='comma lead idx for --mode dropout')
    ap.add_argument('--tau-max', type=int, default=None, help='encoder tau_max for --mode taumax')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-inference-steps', type=int, default=50)
    ap.add_argument('--guidance-scale', type=float, default=2.0)
    args = ap.parse_args()
    for k in ('config', 'checkpoint', 'subject_list', 'npy_dir'):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.chdir(SRC)  # dataset paths and pretrained_path in the config are relative to src/

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)

    model = ECGCondMDM(n_points=cfg.dataset.n_points, n_frames=cfg.dataset.n_training_frames,
                       n_feats=3, model_config=cfg.model_config)
    from safetensors.torch import load_file
    state = load_file(os.path.join(args.checkpoint, 'model.safetensors'), device='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f'loaded {args.checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected')
    print(f'alpha_delay = {float(model.ecg_encoder.alpha_delay):.6g}  '
          f'tau_max(default) = {model.ecg_encoder.tau_max}')

    if args.mode == 'taumax':
        assert args.tau_max is not None
        model.ecg_encoder.tau_max = int(args.tau_max)
        print(f'override tau_max -> {model.ecg_encoder.tau_max}')
    drop_leads = [int(x) for x in args.drop_leads.split(',') if x != ''] if args.mode == 'dropout' else []
    if args.mode == 'dropout':
        print(f'dropping leads {drop_leads} = {[LEAD_NAMES[i] for i in drop_leads]}')
    if args.mode == 'noise':
        assert args.snr is not None
        print(f'additive Gaussian noise @ {args.snr} dB SNR (per lead)')

    with open(args.subject_list) as f:
        wanted = {ln.strip() for ln in f if ln.strip()}

    ds = UKbiobankHeartMesh('val', cfg.dataset)
    idx = [i for i, s in enumerate(ds.subjects) if s[0] in wanted]
    print(f'subject-list {os.path.basename(args.subject_list)}: '
          f'{len(wanted)} requested, {len(idx)} present in val')
    sub = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                                         num_workers=cfg.dataloader_num_workers)

    ddpm = DDPMScheduler(num_train_timesteps=1000, prediction_type='sample', clip_sample=False)
    ddim = DDIMScheduler.from_config(ddpm.config)
    pipe = ECGTrajPipeline(model=model, scheduler=ddim)

    os.makedirs(args.npy_dir, exist_ok=True)
    noise_gen = torch.Generator(device=device).manual_seed(NOISE_SEED)

    n_done = 0
    with torch.no_grad():
        for batch, info in tqdm(loader, desc=f'{args.mode}'):
            src = batch['points_src'].to(device)
            bsz = src.shape[0]
            sids = info['subid']
            todo = [b for b in range(bsz)
                    if not os.path.exists(os.path.join(args.npy_dir, f'{sids[b]}_pred.npy'))]
            # advance noise_gen deterministically even for skipped batches so a resumed
            # run reproduces the same noise realisation per subject
            ecg = batch['ecg'].to(device)
            ecg = perturb_ecg(ecg, args.mode, args.snr, drop_leads, noise_gen)
            if not todo:
                n_done += bsz
                continue
            traj = pipe(init_pc=src, ecg=ecg, device=device, batch_size=bsz,
                        n_frames=cfg.dataset.n_training_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator().manual_seed(0),
                        delay_mat=None).float()
            assert torch.isfinite(traj).all(), 'non-finite prediction!'
            for b in todo:
                np.save(os.path.join(args.npy_dir, f'{sids[b]}_pred.npy'), traj[b].cpu().numpy())
            n_done += bsz
    print(f'done: {n_done} subjects -> {args.npy_dir}')


if __name__ == '__main__':
    main()
