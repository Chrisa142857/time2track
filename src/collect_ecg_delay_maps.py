"""
Export DelayWave delay maps for every validation ECG (same encoder path as training).

Each sample is saved as ``{subid}_delay_map.npz`` with:

  * ``delta`` — (n_leads, max_level) float32, soft-argmax lag vs ref lead (samples).
  * ``entropy`` — (n_leads, max_level) softmax entropy of the windowed xcorr.
  * ``mean_entropy`` — scalar, mean over leads/scales for this sample.
  * ``wavelet_bands`` — (max_level+1, n_leads, T) float32, PhysioWave
    ``wavelet_decomp`` coefficients (approximation + detail scales; T matches
    the encoder, padded for V1).
  * ``magnitude`` — (V1 DelayWave only) log-sum-exp diagnostic.

Also writes ``delay_map_meta.json`` once (ref_lead, tau_max, max_level, …).

Usage (from repo root, same layout as eval_ecg_heart.sh)::

    cd src
    python collect_ecg_delay_maps.py \\
        --config ../src/configs/config_ecg_heart_delaywavev3.yaml \\
        --output_dir /path/to/delay_maps_val

If ``--checkpoint`` is omitted or empty, weights are loaded from
``{output_dir from config}/best_model`` (same rule as ``eval_ecg_heart.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import islice

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dataset.ukbiobank_dataset import UKbiobankHeartMesh
from model.ecg_mesh_model import ECGCondMDM


def _load_model_weights(model: ECGCondMDM, ckpt_dir: str) -> None:
    bin_path = os.path.join(ckpt_dir, 'pytorch_model.bin')
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


def _encoder_meta(enc) -> dict:
    meta = {
        'encoder_class': type(enc).__name__,
        'ref_lead': int(enc.ref_lead),
        'max_level': int(enc.max_level),
        'n_leads': int(enc.n_leads),
        'wavelet_bands_layout': '(max_level+1, n_leads, time) from wavelet_decomp',
    }
    tau = getattr(enc, 'tau_max', None)
    if tau is not None:
        meta['tau_max'] = int(tau)
    if hasattr(enc, 'alpha_delay'):
        meta['alpha_delay'] = float(enc.alpha_delay.detach().cpu())
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Accelerate checkpoint dir (contains model.safetensors or pytorch_model.bin). '
        'Default: {output_dir}/best_model from the config.',
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='DataLoader worker processes. Default 0 avoids multiprocessing shutdown '
        'errors (e.g. "Device or resource busy" on temp dirs). Use -1 to take '
        'dataloader_num_workers from the config.',
    )
    parser.add_argument(
        '--num_val_batches',
        type=int,
        default=-1,
        help='Maximum number of **dataloader batches** to run (not subjects). '
        'Each batch holds eval_batch_size samples (~32). Example: val has ~8093 '
        'subjects → ~253 batches; --num_val_batches 64 stops after 64 batches '
        '(~25%%). Use -1 for the full validation set.',
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)

    ckpt_dir = (args.checkpoint or '').strip()
    if not ckpt_dir:
        out = cfg.get('output_dir')
        if not out:
            raise ValueError(
                'No --checkpoint given and config has no output_dir; '
                'pass --checkpoint or set output_dir in the yaml.'
            )
        ckpt_dir = os.path.join(str(out), 'best_model')
    print(f'Checkpoint: {ckpt_dir}')

    model = ECGCondMDM(
        n_points=cfg.dataset.n_points,
        n_frames=cfg.dataset.n_training_frames,
        n_feats=3,
        model_config=cfg.model_config,
    )
    _load_model_weights(model, ckpt_dir)
    model.to(device)
    model.eval()

    if model.get_ecg_delay_map(torch.zeros(1, 12, 600, device=device)) is None:
        raise RuntimeError(
            'This checkpoint does not use a DelayWave ECG encoder '
            '(delaywave / delaywavev2 / delaywavev3 / delaywavev4).'
        )

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'delay_map_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(_encoder_meta(model.ecg_encoder), f, indent=2)
    print(f'Wrote {meta_path}')

    val_dataset = UKbiobankHeartMesh('val', cfg.dataset)
    num_workers = (
        cfg.dataloader_num_workers if args.num_workers < 0 else args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    max_batches = args.num_val_batches
    n_loader_batches = len(val_loader)
    if max_batches > 0:
        tqdm_total = min(n_loader_batches, max_batches)
        print(
            f'Val loader: {n_loader_batches} batches (batch_size={cfg.eval_batch_size}); '
            f'processing first {tqdm_total} due to --num_val_batches={max_batches}. '
            f'Use --num_val_batches -1 for all {n_loader_batches} batches.'
        )
    else:
        tqdm_total = n_loader_batches
        print(
            f'Val loader: {n_loader_batches} batches (batch_size={cfg.eval_batch_size}); '
            'processing full val set.'
        )
    print(f'DataLoader num_workers={num_workers}')

    n_saved = 0
    loader = val_loader if max_batches <= 0 else islice(val_loader, max_batches)
    with torch.no_grad():
        for batch, info in tqdm(loader, desc='delay maps', total=tqdm_total):
            ecg = batch['ecg'].to(device)
            maps = model.get_ecg_delay_map(ecg)
            assert maps is not None
            bsz = ecg.shape[0]
            delta = maps['delta'].numpy()
            # ent = maps['entropy'].numpy()
            # me = maps['mean_entropy'].numpy()
            wb = maps['wavelet_bands'].numpy()
            mag = maps.get('magnitude')
            mag_np = mag.numpy() if mag is not None else None
            for b in range(bsz):
                sid = info['subid'][b] if 'subid' in info else f'sample_{n_saved:06d}'
                out_npz = os.path.join(args.output_dir, f'{sid}_delay_map.npz')
                payload = {
                    'delta': delta[b].astype(np.float32),
                    # 'entropy': ent[b].astype(np.float32),
                    # 'mean_entropy': np.float32(me[b]),
                    'wavelet_bands': wb[b].astype(np.float32),
                }
                if mag_np is not None:
                    payload['magnitude'] = mag_np[b].astype(np.float32)
                np.savez(out_npz, **payload)
                n_saved += 1

    print(f'Saved {n_saved} delay maps under {args.output_dir}')


if __name__ == '__main__':
    main()
