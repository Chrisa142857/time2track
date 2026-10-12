"""Envelope delay branch for sEMG (Appendix A.8, shared by both decoders).

sEMG channels record asynchronous motor-unit interference patterns rather than delayed copies of one
propagating wave, so the branch cross-correlates the activation ENVELOPE of each channel instead of
its wavelet detail bands: |x_c| after mean removal, FFT low-pass at 15 Hz, FFT cross-correlation
against the reference channel 0, soft-argmax over |tau| <= tau_max with a learnable temperature
(beta_raw, initialized at 50). The per-channel lag is embedded by a two-layer MLP (1 -> 1024 -> 256,
GELU), summed with a learnable channel embedding and mixed by one PhysioWave TransformerBlock
(4 heads, MLP ratio 4, dropout 0.1).

Gate: zero convolution (ControlNet). The output head is zero-initialized and alpha_delay is pinned at
1, so the decoder equals the released one at initialization while the head still receives gradient.

This base class carries the FiLM head used by vemg2pose (vemg2pose_delay.py). emgformer_delay.py
subclasses it and replaces the head and the chunking.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Clone of https://github.com/ForeverBlue816/PhysioWave @6f7bec7. The entry scripts set it from
# --physiowave, the environment variable is the fallback.
PHYSIOWAVE_DIR = os.environ.get('PHYSIOWAVE_DIR')


def _load_tech_block():
    """PhysioWave's TransformerBlock, loaded from transformer_modules.py by file path."""
    if 'transformer_modules' not in sys.modules:
        if not PHYSIOWAVE_DIR:
            raise SystemExit('PhysioWave location unknown: pass --physiowave or set PHYSIOWAVE_DIR')
        spec = importlib.util.spec_from_file_location(
            'transformer_modules', os.path.join(PHYSIOWAVE_DIR, 'transformer_modules.py'))
        m = importlib.util.module_from_spec(spec)
        sys.modules['transformer_modules'] = m
        spec.loader.exec_module(m)
    return sys.modules['transformer_modules'].TransformerBlock


class EnvelopeDelayBranch(nn.Module):
    """Per-channel activation-envelope lag descriptor -> FiLM (gain, shift)."""

    def __init__(self, n_channels=16, ref_channel=0, cutoff_hz=15.0, tau_max=200,
                 beta_init=50.0, latent_dim=256, out_dim=64, chunk=1790,
                 sample_rate=2000.0, dropout=0.1):
        super().__init__()
        TransformerBlock = _load_tech_block()
        self.n_channels, self.ref_channel = n_channels, ref_channel
        self.cutoff_hz, self.sample_rate = float(cutoff_hz), float(sample_rate)
        self.tau_max, self.chunk = int(tau_max), int(chunk)

        # one scalar lag per channel -> latent. beta_raw has shape (1,), single envelope scale.
        self.beta_raw = nn.Parameter(torch.full((1,), float(beta_init)))
        self.delay_mlp = nn.Sequential(
            nn.Linear(1, 4 * latent_dim), nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim))
        self.channel_pos_embed = nn.Parameter(torch.randn(1, n_channels, latent_dim) * 0.02)
        self.delay_block = TransformerBlock(dim=latent_dim, num_heads=4, mlp_ratio=4.0,
                                            dropout=dropout, rope_dim=None)
        self.out = nn.Linear(latent_dim, 2 * out_dim)      # FiLM (gain, shift)

        # zero convolution: out(u) = 0 at step 0 (decoder unchanged), dL/dW_out = dL/dv * u != 0
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.alpha_delay = nn.Parameter(torch.ones(()), requires_grad=False)
        self.last_ent = float('nan')
        self.last_lag_abs = float('nan')

    # ---------------------------------------------------------------- envelope
    def _envelope(self, x):
        """|x| low-passed at cutoff_hz by a zero-phase brick-wall filter in the rFFT domain."""
        T = x.shape[-1]
        a = x.abs()
        a = a - a.mean(-1, keepdim=True)
        Fx = torch.fft.rfft(a, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate, device=x.device)
        Fx = Fx * (freqs <= self.cutoff_hz).to(Fx.dtype)
        return torch.fft.irfft(Fx, n=T, dim=-1)

    # -------------------------------------------------------------- descriptor
    def _lag_descriptor(self, env):
        """(N, C, T) -> (delta (N, C, 1), mean entropy)."""
        N, C, T = env.shape
        s = env - env.mean(-1, keepdim=True)
        Fs = torch.fft.rfft(s.float(), dim=-1)
        R = torch.fft.irfft(Fs * Fs[:, self.ref_channel:self.ref_channel + 1].conj(),
                            n=T, dim=-1)
        nrm = s.float().pow(2).sum(-1, keepdim=True).clamp(min=1e-9).sqrt()
        R = R / (nrm * nrm[:, self.ref_channel:self.ref_channel + 1] + 1e-12)
        R = torch.fft.fftshift(R, dim=-1)
        c = T // 2
        tm = int(min(self.tau_max, c - 1))
        Rw = R[..., c - tm:c + tm + 1]
        beta = F.softplus(self.beta_raw) + 1e-3
        logp = F.log_softmax(Rw * beta, dim=-1)
        p = logp.exp()
        pos = torch.arange(-tm, tm + 1, device=R.device, dtype=R.dtype)
        tau = (p * pos).sum(-1)
        ent = -(p * logp).sum(-1)
        delta = tau - tau[:, self.ref_channel:self.ref_channel + 1]
        return delta.unsqueeze(-1), ent.mean()

    # ------------------------------------------------------------------- lags
    def lags(self, x, lc, t_out):
        """Causal chunks: K = round(t_out / chunk) chunks of `chunk` samples, chunk k ends at
        lc + k * (t_out // K), so output frame t never depends on EMG after t. With the vemg2pose
        setting (t_out 2048, chunk 1024) this gives K = 2 chunks ending at the start and at the
        middle of the output window."""
        B, C, _ = x.shape
        K = max(1, int(round(t_out / self.chunk)))
        Tc = t_out // K
        win = []
        for k in range(K):
            end = lc + k * Tc                            # exclusive => strictly causal
            w = x[..., max(0, end - self.chunk):end]
            if w.shape[-1] < self.chunk:
                w = F.pad(w, (self.chunk - w.shape[-1], 0))
            win.append(w)
        xs = torch.stack(win, 1).reshape(B * K, C, self.chunk)
        d, ent = self._lag_descriptor(self._envelope(xs))
        return d.reshape(B, K, C, 1), ent

    # ---------------------------------------------------------------- forward
    def forward(self, x, lc, t_out):
        """(B, C, lc+t_out) -> FiLM (gain, shift), each (B, out_dim, t_out)."""
        delta, ent = self.lags(x, lc, t_out)
        B, K = delta.shape[0], delta.shape[1]
        self.last_ent = float(ent.detach())
        self.last_lag_abs = float(delta.detach().abs().mean())
        u = self.delay_mlp(delta.reshape(B * K, self.n_channels, 1))
        u = u + self.channel_pos_embed
        u = self.delay_block(u)
        # channel average per chunk, then linear interpolation over the K chunks to every frame
        v = self.out(u).mean(1).reshape(B, K, -1).permute(0, 2, 1)
        v = v.expand(-1, -1, t_out) if K == 1 else \
            F.interpolate(v, size=t_out, mode='linear', align_corners=True)
        v = self.alpha_delay * v
        g, b = v.chunk(2, dim=1)
        return g, b, ent
