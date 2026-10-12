#!/usr/bin/env python
"""Pairwise inter-lead delay tensor and ground-truth ES frame of every validation subject.

Step 1 of the reference-free ES timing classification of Appendix A.7.

Delay tensor D[n, r, c, s]: the soft-argmax lag (samples, 2 ms each at 500 Hz) of lead c against
reference lead r at wavelet scale s, from the TRAINED delay branch of Time2Track (wavelet
front-end, FFT cross-correlation, temperature soft-argmax of Eq. 6-7), recomputed for each of the
12 reference leads. The wavelet decomposition is computed once per subject and only the
cross-correlation changes with the reference. D[n, r, r, :] is 0 by construction.

ES frame: the frame of minimum convex-hull volume of the GT trajectory inside the physiological
window of frames 8 to 32 (256 to 1024 ms) on the 50-frame, 32 ms grid whose frame 0 is the rest
frame. The GT trajectories in npy_eval are frames 1..49, and frame 0 lies outside the window, so
it is not needed.

Subjects: every {subid}_gt.npy in --npy-eval (the validation split), in sorted order. The order
matters for the cross-validation folds of reference_free_auc.py.

Output npz: subid (N,), delta (N, 12 ref, 12 lead, 3 scale) float32, es (N,) int16.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.spatial import ConvexHull

SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(SRC))

LO, HI = 8, 32     # ES search window on the 50-frame grid, 256-1024 ms


def hull_vol(pts):
    """Convex-hull volume per frame. pts (T,N,3) -> (T,)."""
    out = np.empty(len(pts))
    for t in range(len(pts)):
        try:
            out[t] = ConvexHull(pts[t]).volume
        except Exception:
            out[t] = np.nan
    return out


def es_frame(gt):
    """gt (49,388,3) = frames 1..49. Windowed argmin of the hull volume, as a 50-frame index.
    The volumes are rounded to float32 first, as they were stored before the argmin."""
    vol = hull_vol(gt).astype(np.float32).astype(float)     # vol[t-1] is frame t
    return int(np.argmin(vol[LO - 1:HI]) + LO)


def load_encoder(cfg, ckpt):
    """DelayWaveECGEncoderV3 built from the config, with the trained ecg_encoder.* weights."""
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from model.ecg_mesh_model import DelayWaveECGEncoderV3

    kw = OmegaConf.to_container(cfg.model_config.delaywavev3_encoder, resolve=True)
    kw["pretrained_path"] = None          # every weight comes from the checkpoint
    enc = DelayWaveECGEncoderV3(latent_dim=cfg.model_config.latent_dim, **kw)
    sd = {}
    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("ecg_encoder."):
                sd[k[len("ecg_encoder."):]] = f.get_tensor(k)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert len([m for m in missing if 'task_head' not in m]) == 0, missing
    enc.eval()
    return enc


@torch.no_grad()
def descriptor_from_bands(enc, W_det, ref_lead):
    """FFT cross-correlation + soft-argmax on detail bands W_det (B,L,C,T) for one reference.
    Mirrors DelayWaveECGEncoderV3._delay_descriptor_wave_spec. Returns delta (B,C,L) in samples."""
    B, L, C, T = W_det.shape
    W_det = W_det.float()
    F_det = torch.fft.rfft(W_det, dim=-1)
    F_ref = F_det[:, :, ref_lead:ref_lead + 1, :]
    S = F_det * F_ref.conj()
    R = torch.fft.irfft(S, n=T, dim=-1)
    eps = 1e-6
    norm_c = W_det.pow(2).sum(-1, keepdim=True).clamp(min=eps).sqrt()
    norm_r = norm_c[:, :, ref_lead:ref_lead + 1, :]
    R = R / (norm_c * norm_r + eps)
    R_shift = torch.fft.fftshift(R, dim=-1)
    center = T // 2
    tau_max = min(enc.tau_max, center - 1)
    s, e = center - tau_max, center + tau_max + 1
    Rw = R_shift[..., s:e]
    W_len = Rw.shape[-1]
    beta = (Fn.softplus(enc.beta_raw) + 1e-3).view(1, L, 1, 1)
    Rb = Rw * beta
    logp = Fn.log_softmax(Rb, dim=-1)
    p = logp.exp()
    pos = torch.arange(-tau_max, tau_max + 1, dtype=Rw.dtype).view(1, 1, 1, W_len)
    tau = (p * pos).sum(-1)                       # (B,L,C)
    tau = tau.transpose(1, 2).contiguous()        # (B,C,L)
    delta = tau - tau[:, ref_lead:ref_lead + 1, :]
    return delta.cpu().numpy()


@torch.no_grad()
def pairwise_delta(enc, ecg):
    """ecg (B,12,600) float32 -> D (B, 12 ref, 12 lead, 3 scale)."""
    wave = enc.backbone.wavelet_decomp(torch.as_tensor(ecg, dtype=torch.float32))
    Wd = wave.reshape(len(ecg), 4, 12, 600)[:, :3]
    return np.stack([descriptor_from_bands(enc, Wd, r) for r in range(12)], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(SRC / "configs" / "config_ecg_heart_delaywavev3.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="trained model dir (default: <output_dir of --config>/best_model)")
    ap.add_argument("--npy-eval", default=None,
                    help="npy_eval dir with {subid}_gt.npy (default: <output_dir of --config>/npy_eval)")
    ap.add_argument("--ecg-dir", default=None,
                    help="dir with {subid}.npy resting ECGs (default: dataset.ecg_dir of --config)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    a = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(a.config)
    rel = lambda p: os.path.normpath(os.path.join(SRC, p))   # config paths are relative to src/
    ckpt = a.checkpoint or rel(os.path.join(cfg.output_dir, "best_model"))
    npy_eval = a.npy_eval or rel(os.path.join(cfg.output_dir, "npy_eval"))
    ecg_dir = a.ecg_dir or rel(cfg.dataset.ecg_dir)
    torch.set_num_threads(a.threads)

    subs = np.array(sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(npy_eval, "*_gt.npy"))))
    N = len(subs)
    print(f"{N} subjects in {npy_eval}", flush=True)

    t0 = time.time()
    es = np.empty(N, np.int16)
    for i, s in enumerate(subs):
        es[i] = es_frame(np.load(os.path.join(npy_eval, f"{s}_gt.npy")))
        if i % 1000 == 0:
            print(f"ES {i}/{N}  {time.time()-t0:.0f}s", flush=True)

    enc = load_encoder(cfg, ckpt)    # runs on the CPU, as in the paper run
    delta = np.empty((N, 12, 12, 3), np.float32)
    for i in range(0, N, a.batch_size):
        sl = subs[i:i + a.batch_size]
        ecg = np.stack([np.load(os.path.join(ecg_dir, f"{s}.npy")).astype(np.float32) for s in sl])
        delta[i:i + len(sl)] = pairwise_delta(enc, ecg)
        if i % (a.batch_size * 10) == 0:
            print(f"delay {i}/{N}  {time.time()-t0:.0f}s", flush=True)

    assert not np.isnan(delta).any()
    for r in range(12):                      # self-lag of the reference must be ~0
        assert np.abs(delta[:, r, r]).max() < 1e-2, r
    np.savez_compressed(a.out, subid=subs, delta=delta, es=es)
    print(f"saved {a.out} {delta.shape} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
