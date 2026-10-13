"""
ECG-conditioned heart mesh trajectory diffusion model.

ECGEncoder    – 1-D temporal conv that maps (B, 12, 600) ECG signals to
                (B, n_ecg_tokens, latent_dim) conditioning tokens.

ECGCondMDM    – Wraps SpaitalTemporalTransformer from spacetime.py.
                Replaces physics-based conditioning (force / E / nu) with
                ECG tokens, while keeping frame conditioning (rest-state
                mesh prepended as frame 0).

Forward signature:
    ECGCondMDM.forward(x, timesteps, init_pc, ecg, null_emb=None, delay_mat=None)
        x          : (B, T, N, 3)   noisy trajectory
        timesteps  : (B,)
        init_pc    : (B, 1, N, 3)   rest-state mesh
        ecg        : (B, 12, 600)   z-scored 12-lead ECG
        null_emb   : (B, 1, 1)      classifier-free guidance mask (0=null,1=keep)
        delay_mat  : (B, 12, 12)    optional ECG delay matrix (required when
                                    using ECGDelayEncoder)
    Returns (B, T, N, 3) denoised trajectory.
"""

import math
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.spacetime import SpaitalTemporalTransformer, PointEmbed

# TeCh modules use absolute imports (`from layers...`, `from models...`).
# We add TeCh dir to sys.path only while importing, then remove it so
# its `utils/` package does not shadow the project-local `src/utils/`.
_TECH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'TeCh')
)
_PHYSIOWAVE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'PhysioWave')
)


@contextmanager
def _tech_on_path():
    added = False
    if _TECH_DIR not in sys.path:
        sys.path.insert(0, _TECH_DIR)
        added = True
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(_TECH_DIR)
            except ValueError:
                pass


@contextmanager
def _physiowave_on_path():
    added = False
    if _PHYSIOWAVE_DIR not in sys.path:
        sys.path.insert(0, _PHYSIOWAVE_DIR)
        added = True
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(_PHYSIOWAVE_DIR)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# ECG encoder
# ---------------------------------------------------------------------------

class ECGEncoder(nn.Module):
    """
    Lightweight 1-D temporal CNN that converts a 12-lead ECG to a fixed set
    of conditioning tokens.

    Architecture  (default latent_dim=256, n_tokens=8):
        Conv1d(12  → 64,  k=7, s=4)  600 → 150
        Conv1d(64  → 128, k=5, s=3)  150 → 50
        Conv1d(128 → 256, k=5, s=5)   50 → 10
        AdaptiveAvgPool1d(8)           10 → 8
        Transpose  →  (B, 8, 256)
    """

    def __init__(self, latent_dim: int = 256, n_tokens: int = 8):
        super().__init__()
        mid = latent_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(12,  mid // 2, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(mid // 2, mid,       kernel_size=5, stride=3, padding=2),
            nn.GELU(),
            nn.Conv1d(mid,      latent_dim, kernel_size=5, stride=5, padding=2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.n_tokens = n_tokens

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600)  →  (B, n_tokens, latent_dim)"""
        x = self.conv(ecg)      # (B, latent_dim, T_reduced)
        x = self.pool(x)        # (B, latent_dim, n_tokens)
        return x.transpose(1, 2)  # (B, n_tokens, latent_dim)


# ---------------------------------------------------------------------------
# ECG + delay-matrix encoder
# ---------------------------------------------------------------------------

class ECGDelayEncoder(nn.Module):
    """
    ECG + delay-matrix encoder using TeCh-style transformers for both branches.

    ECG branch  : TeChECGEncoder → 1 token  (B, 1, latent_dim)
    Delay branch: linear projection of each lead row to d_model, then
                  Encoder/EncoderLayer (CoTAR-based) transformer, aggregated
                  via mean-pooling → 1 token  (B, 1, latent_dim)

    Output: (B, 2, latent_dim)
    """

    def __init__(
        self,
        latent_dim:     int   = 256,
        n_tokens:       int   = 8,    # kept for API compatibility; ignored internally
        n_leads:        int   = 12,
        delay_hidden:   int   = 128,  # kept for API compatibility; ignored internally
        seq_len:        int   = 600,
        enc_in:         int   = 12,
        v_layer:        int   = 2,
        t_layer:        int   = 2,
        patch_len:      int   = 10,
        n_heads:        int   = 4,
        dropout:        float = 0.1,
        augmentations:  str   = "none",
        n_delay_layers: int   = 2,
        cond_type:        str   = 'concat',  # how to combine ECG tokens + delay tokens ('concat' or 'add')
    ):
        super().__init__()
        self.n_leads = n_leads
        self.cond_type = cond_type

        # ECG branch: TeCh encoder → (B, 1, D)
        self.ecg_encoder = TeChECGEncoder(
            latent_dim=latent_dim, seq_len=seq_len, enc_in=enc_in,
            v_layer=v_layer, t_layer=t_layer, patch_len=patch_len,
            n_heads=n_heads, dropout=dropout, augmentations=augmentations,
        )

        # Delay branch: project each lead row, then Transformer, aggregate to 1 token
        self.delay_proj = nn.Linear(n_leads, latent_dim)
        with _tech_on_path():
            from layers.Transformer_EncDec import Encoder as TechEncoder, EncoderLayer
            from models.TeCh import PositionalEmbedding
        delay_cfg = SimpleNamespace(d_model=latent_dim, dropout=dropout)
        self.delay_pos_embed = PositionalEmbedding(d_model=enc_in)
        self.delay_encoder = TechEncoder(
            [EncoderLayer(delay_cfg) for _ in range(n_delay_layers)]
        )

        self.n_tokens = 2 if self.cond_type == 'concat' else 1  # 1 ECG token + 1 delay token

    def forward(self, ecg: torch.Tensor,
                delay_mat: torch.Tensor) -> torch.Tensor:
        """
        ecg:       (B, 12, 600)
        delay_mat: (B, 12, 12)
        returns:   (B, 2, latent_dim)
        """
        if delay_mat is None:
            raise ValueError("ECGDelayEncoder requires 'delay_mat' input.")
        ecg_tokens = self.ecg_encoder(ecg)                         # (B, 1, D)
        x = self.delay_proj(delay_mat+self.delay_pos_embed(delay_mat))                             # (B, 12, D)
        x = self.delay_encoder(x)                                  # (B, 12, D)
        delay_token = x.mean(dim=1, keepdim=True)                  # (B, 1, D)
        if self.cond_type == 'add':
            return delay_token + ecg_tokens  # combine tokens by addition
        else:
            return torch.cat([ecg_tokens, delay_token], dim=1)         # (B, 2, D)


# ---------------------------------------------------------------------------
# Shift-invariant ECG encoder (FFT magnitude + inter-channel phase delay)
# ---------------------------------------------------------------------------

class ECGShiftInvariantEncoder(nn.Module):
    """
    Two-branch shift-invariant ECG encoder.

    Branch 1 – Waveform (shift-invariant):
        FFT magnitude of raw ECG → FCN-style 1-D CNN → n_wave_tokens conditioning
        tokens.  Because |X[k]| is unaffected by a temporal shift, these tokens
        encode waveform shape independently of recording-time offset.
        Inspired by FCN_controller from Shift-Invariant_Deep_Learning_on_Time_Series.

    Branch 2 – Inter-channel delay:
        FFT complex output → extract low-frequency phase bins 1..n_delay_freqs
        (skipping DC) → pairwise per-lead phase differences → sin/cos encoded
        to handle circular wrapping → per-lead MLP → 12 delay tokens.

    No external delay matrix required — timing is derived directly from the ECG.

    Output: (B, n_wave_tokens + n_leads, latent_dim)
    """

    def __init__(
        self,
        latent_dim:    int  = 256,
        n_wave_tokens: int  = 8,
        n_leads:       int  = 12,
        n_delay_freqs: int  = 16,
        delay_hidden:  int  = 128,
        canonicalize:  bool = False,
    ):
        super().__init__()
        self.n_leads       = n_leads
        self.n_delay_freqs = n_delay_freqs
        self.canonicalize  = canonicalize
        self.n_tokens      = n_wave_tokens + n_leads

        # Branch 1: CNN on FFT magnitude
        # Input: (B, n_leads, 301)  [rfft of 600-point signal]
        # 301 → 150 → 75 → 37 → pool to n_wave_tokens
        self.wave_cnn = nn.Sequential(
            nn.Conv1d(n_leads, 32,        kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64,             kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, latent_dim,     kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(latent_dim),   nn.ReLU(),
        )
        self.wave_pool = nn.AdaptiveAvgPool1d(n_wave_tokens)

        # Branch 2: per-lead phase-difference MLP
        # sin/cos doubles features: n_leads * n_delay_freqs * 2
        delay_in = n_leads * n_delay_freqs * 2
        self.delay_mlp = nn.Sequential(
            nn.Linear(delay_in, delay_hidden),
            nn.GELU(),
            nn.Linear(delay_hidden, latent_dim),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600)  →  (B, n_wave_tokens + 12, latent_dim)"""
        B = ecg.shape[0]
        # Ensure float32 for FFT: rfft on bf16 promotes automatically, but
        # doing it explicitly avoids any precision loss in angle() near ±π.
        X = torch.fft.rfft(ecg.float(), dim=-1)   # (B, n_leads, 301) complex64

        # Optional: rotate all leads so lead-0 bin-1 has zero phase.
        # No effect on |X|; only used as an ablation for the delay branch.
        if self.canonicalize:
            ref = X[:, 0:1, 1:2].angle()          # (B, 1, 1)
            X = X * torch.exp(-1j * ref)

        # Branch 1: magnitude → CNN → wave tokens
        w = self.wave_cnn(X.abs())                 # (B, D, ~37)
        wave_tokens = self.wave_pool(w).transpose(1, 2)  # (B, n_wave_tokens, D)

        # Branch 2: pairwise phase differences → delay tokens
        phase = X[:, :, 1:self.n_delay_freqs + 1].angle()  # (B, 12, 16)
        phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1)        # (B, 12, 12, 16)
        feat = torch.cat([phase_diff.sin(), phase_diff.cos()], dim=-1)  # (B, 12, 12, 32)
        feat = feat.reshape(B, self.n_leads, -1)                    # (B, 12, 384)
        delay_tokens = self.delay_mlp(
            feat.reshape(B * self.n_leads, -1)
        ).reshape(B, self.n_leads, -1).to(ecg.dtype)                # (B, 12, D)

        return torch.cat([wave_tokens, delay_tokens], dim=1)        # (B, n_wave+12, D)


# ---------------------------------------------------------------------------
# TeCh-based ECG encoder
# ---------------------------------------------------------------------------

class TeChECGEncoder(nn.Module):
    """
    Wraps the TeCh classification encoder (TeCh/models/TeCh.py) and skips
    its final projector. Returns the `channel + temporal` embedding from
    TeCh as a single conditioning token of shape (B, 1, latent_dim).

    Defaults reproduce the TeCh ECG hyper-parameters:
        v_layer=2, t_layer=2, patch_len=10, n_heads=4, dropout=0.1,
        augmentations="none"
    """

    def __init__(
        self,
        latent_dim: int = 256,
        seq_len:    int = 600,
        enc_in:     int = 12,
        v_layer:    int = 2,
        t_layer:    int = 2,
        patch_len:  int = 10,
        n_heads:    int = 4,
        dropout:    float = 0.1,
        augmentations: str = "none",
    ):
        super().__init__()
        # Lazy import so TeCh is only required when this encoder is used
        with _tech_on_path():
            from models.TeCh import Model as TeChModel  # type: ignore

        cfg = SimpleNamespace(
            v_layer=v_layer,
            t_layer=t_layer,
            d_model=latent_dim,
            num_class=1,            # projector built but unused
            augmentations=augmentations,
            seq_len=seq_len,
            enc_in=enc_in,
            patch_len=patch_len,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.tech = TeChModel(cfg)
        # Remove the classification projector – we use channel+temporal directly.
        # Keeping unused parameters causes DDP to hang at
        # `_verify_param_shape_across_processes` / fail in backward.
        if hasattr(self.tech, 'projector'):
            del self.tech.projector
        self.v_layer = v_layer
        self.t_layer = t_layer
        self.n_tokens = 1

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600) → (B, 1, latent_dim)"""
        # TeCh expects (B, seq_len, enc_in)
        x_enc = ecg.transpose(1, 2)
        channel = self.tech.channel_encoder(x_enc).mean(1) if self.v_layer > 0 else 0
        temporal = self.tech.temporal_encoder(x_enc).mean(1) if self.t_layer > 0 else 0
        emb = channel + temporal                      # (B, D)
        return emb.unsqueeze(1)                       # (B, 1, D)


# ---------------------------------------------------------------------------
# Localized-delay ECG encoder (shift-invariant, multi-branch)
# ---------------------------------------------------------------------------


class CoTARWithCore(nn.Module):
    """CoTAR variant that exposes the pooled core token alongside the refined
    channel tokens. Identical math to
    TeCh.layers.Transformer_EncDec.CoTAR; only the return signature differs.

    Input : x (B, N, D)
    Output: (refined (B, N, D), core_pooled (B, d_core))
    """

    def __init__(self, d_model: int, d_core: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_core)
        self.lin3 = nn.Linear(d_model + d_core, d_model)
        self.lin4 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        core = F.gelu(self.lin1(x))
        core = self.lin2(core)                                    # (B, N, d_core)
        weight = F.softmax(core, dim=1)                           # (B, N, d_core)
        core_pooled = (core * weight).sum(dim=1)                  # (B, d_core)
        core_broadcast = core_pooled.unsqueeze(1).expand(-1, x.shape[1], -1)
        core_cat = torch.cat([x, core_broadcast], dim=-1)         # (B, N, D+d_core)
        refined = F.gelu(self.lin3(core_cat))
        refined = self.lin4(refined)                              # (B, N, D)
        return refined, core_pooled


def _ricker_init(length: int) -> torch.Tensor:
    """Unit-norm zero-mean Ricker-like wavelet of given length."""
    sigma = max(length / 6.0, 1.0)
    t = torch.arange(length, dtype=torch.float32) - (length - 1) / 2.0
    w = (1.0 - (t / sigma).pow(2)) * torch.exp(-(t / sigma).pow(2) / 2.0)
    w = w - w.mean()
    w = w / (w.norm() + 1e-8)
    return w


@torch.no_grad()
def _detect_rpeak_cycles(
    ecg: torch.Tensor,
    ref_lead: int = 1,
    k_cycles: int = 2,
    min_dist: int = 200,
    smooth_k: int = 11,
):
    """Detect up to ``k_cycles`` R-peak-centred windows in the reference lead.

    Parameters
    ----------
    ecg       : (B, C, T) float tensor (any dtype; promoted to float32 internally)
    ref_lead  : channel index used for detection (default lead II = index 1)
    k_cycles  : number of cycle slots to return per sample
    min_dist  : minimum spacing (samples) between accepted peaks, also used as
                the window half-width for cycle bounds (total width = min_dist)
    smooth_k  : uniform smoothing kernel size for the energy trace

    Returns
    -------
    cycles : (B, K, 2) long tensor of (start, end) positions in [0, T-1]
    valid  : (B, K) bool tensor (True where a real peak was accepted)
    """
    B, _, T = ecg.shape
    x = ecg[:, ref_lead, :].float()                                # (B, T)

    xp = F.pad(x.unsqueeze(1), (1, 1), mode='replicate').squeeze(1)  # (B, T+2)
    d = xp[:, 2:] - 2.0 * xp[:, 1:-1] + xp[:, :-2]                 # (B, T)
    energy = d.pow(2)

    pad = smooth_k // 2
    e_pad = F.pad(energy.unsqueeze(1), (pad, pad), mode='replicate')
    e_smooth = F.avg_pool1d(e_pad, kernel_size=smooth_k, stride=1).squeeze(1)  # (B, T)

    cycles = torch.zeros(B, k_cycles, 2, dtype=torch.long, device=ecg.device)
    valid = torch.zeros(B, k_cycles, dtype=torch.bool, device=ecg.device)
    half = min_dist // 2

    for b in range(B):
        order = torch.argsort(e_smooth[b], descending=True).tolist()
        selected = []
        for idx in order:
            if len(selected) >= k_cycles:
                break
            if all(abs(idx - s) >= min_dist for s in selected):
                selected.append(idx)
        for k, peak in enumerate(selected):
            s = max(0, peak - half)
            e_pos = min(T - 1, peak + half)
            cycles[b, k, 0] = s
            cycles[b, k, 1] = e_pos
            valid[b, k] = True
        # Fallback: evenly spaced windows for unfilled slots
        for k in range(len(selected), k_cycles):
            center = int(((k + 0.5) * T) / k_cycles)
            s = max(0, center - half)
            e_pos = min(T - 1, center + half)
            cycles[b, k, 0] = s
            cycles[b, k, 1] = e_pos
    return cycles, valid


class ECGLocalizedDelayEncoder(nn.Module):
    """
    Shift-invariant ECG encoder with three output token groups:

      1. Waveform-core token (1)   – CoTAR core extracted from the canonicalized
                                     waveform branch.
      2. Delay-core token    (1)   – 12 per-lead delay descriptors projected
                                     through a TeCh CoTAR encoder stack, then
                                     mean-pooled to one token (mirroring the
                                     pattern in ``ECGDelayEncoder``).
      3. Waveform-pool tokens (S_w) – learnable queries attention-pool the
                                     refined 12 channel tokens.

    Total tokens: 1 + 1 + S_w  (default 10).

    Inter-lead delays are computed by circular normalized cross-correlation of
    the ECG against learnable event templates, with per-cycle soft-argmax over
    R-peak-centred windows detected on the reference lead. Relative delays
    (against ``ref_lead``) are exactly invariant to circular shifts of the input.
    """

    def __init__(
        self,
        latent_dim:        int   = 256,
        n_leads:           int   = 12,
        ref_lead:          int   = 1,
        n_templates:       int   = 4,
        template_lengths:  list  = (20, 40, 80, 160),
        k_cycles:          int   = 2,
        min_dist:          int   = 200,
        seq_len:           int   = 600,
        n_wave_tokens:     int   = 1,
        d_core:            int   = 64,
        n_delay_layers:    int   = 2,
        n_heads:           int   = 4,
        dropout:           float = 0.1,
        guidance_hidden:   int   = 128,
        freeze_templates_steps: int = 0,
        cond_type:        str   = 'concat',  # how to combine ECG tokens + delay tokens ('concat' or 'add')
        ablation:           Optional[str] = None,  
    ):
        super().__init__()
        template_lengths = list(template_lengths)
        assert len(template_lengths) == n_templates, (
            f"template_lengths ({len(template_lengths)}) must match "
            f"n_templates ({n_templates})"
        )
        assert 0 <= ref_lead < n_leads

        D = latent_dim
        self.latent_dim        = D
        self.n_leads           = n_leads
        self.ref_lead          = ref_lead
        self.n_templates       = n_templates
        self.template_lengths  = template_lengths
        self.k_cycles          = k_cycles
        self.min_dist          = min_dist
        self.seq_len           = seq_len
        self.n_wave_tokens     = n_wave_tokens
        self.d_core            = d_core
        self.cond_type         = cond_type
        self.ablation          = ablation
        self.freeze_templates_steps = freeze_templates_steps

        # ── Learnable event templates (Ricker init, log-spaced lengths) ─
        self.templates = nn.ParameterList([
            nn.Parameter(_ricker_init(L)) for L in template_lengths
        ])
        # Per-template soft-argmax temperature (softplus-ed to ensure positivity)
        self.beta_raw = nn.Parameter(torch.zeros(n_templates))

        # ── Per-lead delay descriptor → latent ──────────────────────────
        delay_in = 3 * n_templates   # [delta, magnitude, cycle-std] per template
        self.delay_mlp = nn.Sequential(
            nn.Linear(delay_in, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D),
        )
        self.lead_embed = nn.Embedding(n_leads, D)
        self.type_embed = nn.Embedding(3, D)  # 0=wave-core, 1=delay-core, 2=wave-pool

        # ── TeCh CoTAR stack for delay centralization ───────────────────
        with _tech_on_path():
            from layers.Transformer_EncDec import Encoder as TechEncoder, EncoderLayer
        delay_cfg = SimpleNamespace(d_model=D, dropout=dropout)
        self.delay_encoder = TechEncoder(
            [EncoderLayer(delay_cfg) for _ in range(n_delay_layers)]
        )

        # ── Guidance net: |FFT(x_ref)| → canonicalization angle ────────
        n_freq = seq_len // 2 + 1
        self.guidance_net = nn.Sequential(
            nn.Linear(n_freq, guidance_hidden),
            nn.GELU(),
            nn.Linear(guidance_hidden, 1),
        )

        # ── Waveform branch: per-lead time-domain embed → CoTAR → pool ─
        self.wave_proj = nn.Linear(seq_len, D)
        self.cotar = CoTARWithCore(d_model=D, d_core=d_core)
        self.core_proj = nn.Linear(d_core, D)
        if self.ablation != 'nowavepool':
            self.wave_queries = nn.Parameter(torch.randn(n_wave_tokens, D) * 0.02)
            self.wave_pool_attn = nn.MultiheadAttention(
                embed_dim=D, num_heads=n_heads, batch_first=True
            )

        self.n_tokens = 1 if self.cond_type == 'add' else 1 + 1 + n_wave_tokens
        if self.ablation == 'nowavepool':
            self.n_tokens -= n_wave_tokens
    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def _circular_xcorr(self, ecg_f32: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Normalized circular cross-correlation r(tau) per (lead, position).

        ecg_f32 : (B, C, T) float32
        psi     : (L,)      float32
        returns : (B, C, T) float32
        """
        B, C, T = ecg_f32.shape
        L = psi.shape[0]
        eps = 1e-6

        # Right-pad with circular copy so conv1d output length equals T.
        x_pad = F.pad(ecg_f32, (0, L - 1), mode='circular')          # (B, C, T+L-1)
        kernel = psi.view(1, 1, L).expand(C, 1, L).contiguous()      # (C, 1, L)
        num = F.conv1d(x_pad, kernel, groups=C)                      # (B, C, T)

        # Sliding L2 norm of x over length-L windows (circular).
        x2_pad = F.pad(ecg_f32.pow(2), (0, L - 1), mode='circular')
        ones = torch.ones_like(kernel)
        win_energy = F.conv1d(x2_pad, ones, groups=C)                # (B, C, T)
        x_norm = win_energy.clamp(min=eps).sqrt()
        psi_norm = psi.norm() + eps
        return num / (x_norm * psi_norm)

    def _soft_delay_from_cycles(
        self,
        r: torch.Tensor,        # (B, C, T) float32
        cycles: torch.Tensor,   # (B, K, 2) long
        beta: torch.Tensor,     # scalar float32
    ):
        """Per-cycle soft-argmax over windowed correlation map.

        Returns
        -------
        tau       : (B, C, K) soft-argmax position
        magnitude : (B, C, K) log-sum-exp magnitude over window, divided by beta
        entropy   : scalar    mean softmax entropy (for sparsity regularizer)
        """
        B, C, T = r.shape
        K = cycles.shape[1]
        pos = torch.arange(T, device=r.device, dtype=r.dtype)        # (T,)

        starts = cycles[..., 0].to(r.dtype).unsqueeze(-1)            # (B, K, 1)
        ends   = cycles[..., 1].to(r.dtype).unsqueeze(-1)            # (B, K, 1)
        pos_bkt = pos.view(1, 1, T)                                  # (1, 1, T)

        no_wrap = starts <= ends
        in_nowrap = (pos_bkt >= starts) & (pos_bkt <= ends)
        in_wrap   = (pos_bkt >= starts) | (pos_bkt <= ends)
        in_window = torch.where(no_wrap, in_nowrap, in_wrap)         # (B, K, T)

        # (B, K, C, T): replicate r along K, mask outside-window with -inf-ish
        r_bkt = (r * beta).unsqueeze(1).expand(B, K, C, T)
        mask  = in_window.unsqueeze(2).expand(B, K, C, T)
        r_masked = r_bkt.masked_fill(~mask, -1e4)

        log_probs = F.log_softmax(r_masked, dim=-1)
        probs = log_probs.exp()

        tau = (probs * pos.view(1, 1, 1, T)).sum(dim=-1)             # (B, K, C)
        m = torch.logsumexp(r_masked, dim=-1) / beta                 # (B, K, C)

        # Entropy for sparsity loss
        entropy = -(probs * log_probs).sum(dim=-1)                   # (B, K, C)
        entropy_mean = entropy.mean()

        return tau.transpose(1, 2).contiguous(), m.transpose(1, 2).contiguous(), entropy_mean

    def _canonicalize(self, ecg_f32: torch.Tensor):
        """Reference-lead FFT-phase canonicalization applied uniformly to all
        leads.  Returns (x_tilde (B, C, T) float32, phi_star (B,) float32).
        """
        T = ecg_f32.shape[-1]
        X = torch.fft.rfft(ecg_f32, dim=-1)                          # (B, C, F) complex64
        mag_ref = X[:, self.ref_lead, :].abs()                       # (B, F)
        phi_star = self.guidance_net(mag_ref).squeeze(-1)            # (B,)

        freq_idx = torch.arange(X.shape[-1], device=X.device, dtype=ecg_f32.dtype)
        # Cancel a global time shift: multiply bin k by exp(-1j * k * phi_star)
        angle = -phi_star[:, None] * freq_idx[None, :]               # (B, F)
        rotation = torch.polar(torch.ones_like(angle), angle)        # (B, F) complex
        X_canon = X * rotation.unsqueeze(1)                          # (B, C, F)
        x_tilde = torch.fft.irfft(X_canon, n=T, dim=-1)              # (B, C, T)
        return x_tilde, phi_star

    def _template_diversity(self) -> torch.Tensor:
        """Rectified-cosine² pairwise template diversity, zero-padding shorter
        templates to the longest length."""
        K = len(self.templates)
        if K < 2:
            return torch.tensor(0.0, device=self.templates[0].device)
        L_max = max(p.shape[0] for p in self.templates)
        padded = []
        for p in self.templates:
            diff = L_max - p.shape[0]
            if diff > 0:
                left = diff // 2
                right = diff - left
                pp = F.pad(p, (left, right), mode='constant', value=0.0)
            else:
                pp = p
            padded.append(pp / (pp.norm() + 1e-8))
        stacked = torch.stack(padded, dim=0)                         # (K, L_max)
        sim = stacked @ stacked.t()                                  # (K, K)
        off_diag = sim - torch.eye(K, device=sim.device, dtype=sim.dtype)
        rect = F.relu(off_diag)
        # Average over K*(K-1) off-diagonal entries
        return rect.pow(2).sum() / (K * (K - 1))

    # ------------------------------------------------------------------
    # Main encoding path
    # ------------------------------------------------------------------

    def _encode_given_cycles(
        self,
        ecg: torch.Tensor,         # (B, C, T)
        cycles: torch.Tensor,      # (B, K, 2)
        valid: torch.Tensor,       # (B, K)
        return_diag: bool = False,
    ):
        """Deterministic encoding path (no random shift).
        Returns tokens (B, n_tokens, D) plus optional diagnostic dict.
        """
        B, C, T = ecg.shape
        D = self.latent_dim
        out_dtype = ecg.dtype
        ecg32 = ecg.float()

        # ── Delay branch: soft-argmax per (lead, template, cycle) ─────
        taus_per_tmpl, mags_per_tmpl, stds_per_tmpl = [], [], []
        entropies = []
        for k, psi_k in enumerate(self.templates):
            r_k = self._circular_xcorr(ecg32, psi_k)                 # (B, C, T)
            beta_k = F.softplus(self.beta_raw[k]) + 1e-3
            tau, m, ent = self._soft_delay_from_cycles(r_k, cycles, beta_k)
            taus_per_tmpl.append(tau)                                # (B, C, K_cyc)
            mags_per_tmpl.append(m)
            stds_per_tmpl.append(tau)  # placeholder; std computed below
            entropies.append(ent)

        # Stack across templates: (B, C, K_tmpl, K_cyc)
        tau_bcktk = torch.stack(taus_per_tmpl, dim=2)                # (B, C, K_tmpl, K_cyc)
        mag_bcktk = torch.stack(mags_per_tmpl, dim=2)
        # Cross-cycle aggregation
        tau_mean = tau_bcktk.mean(dim=-1)                            # (B, C, K_tmpl)
        mag_mean = mag_bcktk.mean(dim=-1)
        if tau_bcktk.shape[-1] > 1:
            tau_std = tau_bcktk.std(dim=-1)
        else:
            tau_std = torch.zeros_like(tau_mean)

        # Relative delay vs reference lead
        tau_ref = tau_mean[:, self.ref_lead:self.ref_lead + 1, :]    # (B, 1, K_tmpl)
        delta = tau_mean - tau_ref                                   # (B, C, K_tmpl)

        # Per-lead descriptor → latent
        descriptor = torch.cat([delta, mag_mean, tau_std], dim=-1)   # (B, C, 3K)
        u = self.delay_mlp(descriptor)                               # (B, C, D)
        lead_ids = torch.arange(C, device=u.device)
        u = u + self.lead_embed(lead_ids)                            # (B, C, D)
        u = self.delay_encoder(u)                                    # (B, C, D)
        t_delay_core = u.mean(dim=1, keepdim=True) + self.type_embed.weight[1]
        # shape (B, 1, D)

        # ── Waveform branch: canonicalize → per-lead embed → CoTAR ────
        x_tilde, phi_star = self._canonicalize(ecg32)                # (B, C, T), (B,)
        H = self.wave_proj(x_tilde) + self.lead_embed(lead_ids)      # (B, C, D)
        H_refined, core_pooled = self.cotar(H)                       # (B, C, D), (B, d_core)
        t_wave_core = self.core_proj(core_pooled).unsqueeze(1) + self.type_embed.weight[0]
        if self.ablation != 'nowavepool':
            queries = self.wave_queries.unsqueeze(0).expand(B, -1, -1)   # (B, S_w, D)
            t_wave_pool, _ = self.wave_pool_attn(queries, H_refined, H_refined)
            t_wave_pool = t_wave_pool + self.type_embed.weight[2]        # (B, S_w, D)
        else:
            t_wave_pool = None
        
        if self.cond_type == 'add':
            if t_wave_pool is not None:
                tokens = t_wave_core + t_delay_core + t_wave_pool
            else:
                tokens = t_wave_core + t_delay_core 
        else:
            tokens = [t_wave_core, t_delay_core, t_wave_pool] if t_wave_pool is not None else [t_wave_core, t_delay_core]
            tokens = torch.cat(tokens, dim=1)  # (B, n_tokens, D)
        tokens = tokens.to(out_dtype)

        diag = None
        if return_diag:
            diag = {
                'delta_matrix':  delta.detach(),                # (B, C, K_tmpl)
                'per_cycle_std': tau_std.detach(),              # (B, C, K_tmpl)
                'magnitudes':    mag_mean.detach(),             # (B, C, K_tmpl)
                'phi_star':      phi_star.detach(),             # (B,)
                'cycles':        cycles.detach(),               # (B, K_cyc, 2)
                'valid':         valid.detach(),                # (B, K_cyc)
                'sparsity_entropy': torch.stack(entropies).mean().detach(),
            }
            # Templates: return as a list of 1-D tensors (they have different lengths)
            diag['templates'] = [p.detach() for p in self.templates]
        return tokens, diag, entropies, phi_star

    def forward(self, ecg: torch.Tensor, return_aux: bool = False):
        """ecg: (B, 12, seq_len) → (B, n_tokens, latent_dim)
        If ``return_aux`` is True, also returns a dict of auxiliary loss tensors
        with keys: ``diversity``, ``sparsity``, ``core_var``, ``shift_inv``.
        """
        cycles, valid = _detect_rpeak_cycles(
            ecg, ref_lead=self.ref_lead,
            k_cycles=self.k_cycles, min_dist=self.min_dist,
        )
        tokens, _, entropies, phi_star = self._encode_given_cycles(
            ecg, cycles, valid, return_diag=False,
        )

        if not return_aux:
            return tokens

        # ── Auxiliary losses (all float32 scalars with grad) ────────────
        L_div = self._template_diversity()
        L_sparsity = torch.stack(entropies).mean()
        L_core_var = phi_star.std() if phi_star.numel() > 1 else phi_star.new_zeros(())

        # Shift-invariance: roll ECG & cycles by the same random offset.
        T = ecg.shape[-1]
        shift = int(torch.randint(0, T, (1,)).item())
        ecg_shifted = torch.roll(ecg, shifts=shift, dims=-1)
        cycles_shifted = (cycles + shift) % T
        tokens_shifted, _, _, _ = self._encode_given_cycles(
            ecg_shifted, cycles_shifted, valid, return_diag=False,
        )
        L_shift = F.mse_loss(tokens.float(), tokens_shifted.float())

        aux = {
            'diversity':  L_div,
            'sparsity':   L_sparsity,
            'core_var':   L_core_var,
            'shift_inv':  L_shift,
        }
        return tokens, aux

    @torch.no_grad()
    def dump_diagnostics(self, ecg: torch.Tensor) -> dict:
        """Return CPU numpy-ready diagnostics for offline analysis. Preserves
        dtypes as float32 / int64 where applicable."""
        cycles, valid = _detect_rpeak_cycles(
            ecg, ref_lead=self.ref_lead,
            k_cycles=self.k_cycles, min_dist=self.min_dist,
        )
        _, diag, _, _ = self._encode_given_cycles(
            ecg, cycles, valid, return_diag=True,
        )
        out = {
            'delta_matrix':     diag['delta_matrix'].float().cpu().numpy(),
            'per_cycle_std':    diag['per_cycle_std'].float().cpu().numpy(),
            'magnitudes':       diag['magnitudes'].float().cpu().numpy(),
            'phi_star':         diag['phi_star'].float().cpu().numpy(),
            'cycles':           diag['cycles'].cpu().numpy(),
            'valid':            diag['valid'].cpu().numpy(),
            'sparsity_entropy': float(diag['sparsity_entropy'].item()),
            'templates':        [t.float().cpu().numpy() for t in diag['templates']],
        }
        return out


# ---------------------------------------------------------------------------
# PhysioWave-based ECG encoder
# ---------------------------------------------------------------------------

class PhysioWaveECGEncoder(nn.Module):
    """ECG encoder backed by the pretrained PhysioWave BERTWaveletTransformer.

    Pads ECG to a multiple of the temporal patch size, runs the wavelet
    transformer feature extractor, projects token dim to ``latent_dim`` and
    pools the token sequence to ``n_tokens`` conditioning tokens.

    Defaults match the PhysioWave ECG pretraining recipe:
        in_channels=12, max_level=3, wave_kernel_size=24,
        wavelet_names=['db4','db6','sym4','coif2'], use_separate_channel=True,
        patch_size=64, embed_dim=384, depth=8, num_heads=12.
    """

    def __init__(
        self,
        latent_dim:           int   = 256,
        n_tokens:             int   = 8,
        seq_len:              int   = 600,
        in_channels:          int   = 12,
        max_level:            int   = 3,
        wave_kernel_size:     int   = 24,
        wavelet_names:        list  = ('db4', 'db6', 'sym4', 'coif2'),
        use_separate_channel: bool  = True,
        patch_size:           int   = 64,
        embed_dim:            int   = 384,
        depth:                int   = 8,
        num_heads:            int   = 12,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        use_pos_embed:        bool  = True,
        pos_embed_type:       str   = '2d',
        pretrained_path:      Optional[str] = None,
        freeze_backbone:      bool  = False,
    ):
        super().__init__()
        # Local `model/` package shadows PhysioWave's top-level `model.py`,
        # so load PhysioWave modules via importlib under unique names.
        import importlib.util as _ilu
        with _physiowave_on_path():
            def _load(mod_name, file_name):
                if mod_name in sys.modules:
                    return sys.modules[mod_name]
                spec = _ilu.spec_from_file_location(
                    mod_name, os.path.join(_PHYSIOWAVE_DIR, file_name)
                )
                mod = _ilu.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod

            # PhysioWave/model.py imports its siblings unqualified
            # (`from wavelet_modules import ...`); preload them under their
            # bare names while the dir is on sys.path.
            _load('wavelet_modules', 'wavelet_modules.py')
            _load('transformer_modules', 'transformer_modules.py')
            _load('head_modules', 'head_modules.py')
            pw_model = _load('physiowave_model', 'model.py')
            BERTWaveletTransformer = pw_model.BERTWaveletTransformer

        self.patch_size = patch_size
        # Pad seq_len up to nearest multiple of patch_size
        self.padded_len = ((seq_len + patch_size - 1) // patch_size) * patch_size
        self.pad_amount = self.padded_len - seq_len

        self.backbone = BERTWaveletTransformer(
            in_channels=in_channels,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=list(wavelet_names),
            use_separate_channel=use_separate_channel,
            patch_size=(1, patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type=None,
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[PhysioWaveECGEncoder] loaded {pretrained_path}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

        # Drop unused pretrain head (otherwise its params hang DDP)
        if hasattr(self.backbone, 'task_heads'):
            for k in list(self.backbone.task_heads.keys()):
                del self.backbone.task_heads[k]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(embed_dim, latent_dim)
        # self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.n_tokens = 480 # hard coded according to the actual shape torch.Size([10, 480, 256])

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, in_channels, T) → (B, n_tokens, latent_dim)"""
        if self.pad_amount > 0:
            ecg = F.pad(ecg, (0, self.pad_amount))
        feats = self.backbone.forward_features(ecg)   # (B, L, embed_dim)
        feats = self.proj(feats)                      # (B, L, latent_dim)
        # AdaptiveAvgPool1d expects (B, C, L)
        # pooled = self.pool(feats.transpose(1, 2)).transpose(1, 2)     # (B, latent_dim, n_tokens)
        pooled = feats                      # (B, latent_dim, L)
        assert pooled.shape[1] == self.n_tokens, (
            f"Expected pooled shape (B, ..., {self.n_tokens}), got {pooled.shape}"
        )
        return pooled                 # (B, n_tokens, latent_dim)


# ---------------------------------------------------------------------------
# DelayWave ECG encoder: explicit inter-channel-delay objective on top of
# PhysioWave's learnable wavelet front-end + transformer morphology branch.
# ---------------------------------------------------------------------------


class DelayWaveECGEncoder(nn.Module):
    """ECG encoder that adds an explicit inter-channel delay branch to the
    PhysioWave ``BERTWaveletTransformer`` morphology backbone.

    Pipeline (single-cycle ECG, no R-peak gating):

        ecg (B, C, T) ─┬─► wavelet_decomp ─► wave_spec (B, (L+1)C, T_pad)
                       │      │
                       │      └─► detail bands (B, L, C, T_pad)
                       │             │
                       │             ├─► circular xcorr vs ref-lead (FFT)
                       │             │     soft-argmax over T_pad
                       │             └─► τ, mag, ent  per (scale, lead)
                       │
                       │   delta = τ - τ[ref];   desc = [delta, mag]
                       │   delay_mlp → PhysioWave TransformerEncoder → mean-pool
                       │      → 1 delay-core token
                       │
                       └─► prepare_tokens ─► encoder
                              proj + AdaptiveAvgPool ─► n_morph_tokens
                              (mirrors ``PhysioWaveECGEncoder``)

    Output: (B, 1 + n_morph_tokens, latent_dim)  if cond_type == 'concat'
            (B, n_morph_tokens,     latent_dim)  if cond_type == 'add'
    """

    def __init__(
        self,
        latent_dim:           int   = 256,
        n_morph_tokens:       int   = 8,
        seq_len:              int   = 600,
        n_leads:              int   = 12,
        ref_lead:             int   = 1,                       # lead II
        # Wavelet front-end (matches PhysioWave ECG recipe)
        max_level:            int   = 3,
        wave_kernel_size:     int   = 24,
        wavelet_names:        list  = ('db4', 'db6', 'sym4', 'coif2'),
        use_separate_channel: bool  = True,
        # Morphology backbone
        patch_size:           int   = 64,
        embed_dim:            int   = 384,
        depth:                int   = 8,
        num_heads:            int   = 12,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        use_pos_embed:        bool  = True,
        pos_embed_type:       str   = '2d',
        pretrained_path:      Optional[str] = None,
        freeze_backbone:      bool  = False,
        # Delay branch
        n_delay_layers:       int   = 2,
        delay_num_heads:      int   = 4,
        delay_mlp_ratio:      float = 4.0,
        cond_type:            str   = 'concat',
    ):
        super().__init__()
        import importlib.util as _ilu
        with _physiowave_on_path():
            def _load(mod_name, file_name):
                if mod_name in sys.modules:
                    return sys.modules[mod_name]
                spec = _ilu.spec_from_file_location(
                    mod_name, os.path.join(_PHYSIOWAVE_DIR, file_name)
                )
                mod = _ilu.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod

            _load('wavelet_modules', 'wavelet_modules.py')
            tx_mod = _load('transformer_modules', 'transformer_modules.py')
            _load('head_modules', 'head_modules.py')
            pw_model = _load('physiowave_model', 'model.py')
            BERTWaveletTransformer = pw_model.BERTWaveletTransformer
            TransformerEncoder = tx_mod.TransformerEncoder

        assert 0 <= ref_lead < n_leads
        self.n_leads      = n_leads
        self.ref_lead     = ref_lead
        self.max_level    = max_level
        self.cond_type    = cond_type
        self.latent_dim   = latent_dim
        self.n_morph_tokens = n_morph_tokens

        self.patch_size = patch_size
        self.padded_len = ((seq_len + patch_size - 1) // patch_size) * patch_size
        self.pad_amount = self.padded_len - seq_len

        self.backbone = BERTWaveletTransformer(
            in_channels=n_leads,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=list(wavelet_names),
            use_separate_channel=use_separate_channel,
            patch_size=(1, patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type=None,
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[DelayWaveECGEncoder] loaded {pretrained_path}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

        if hasattr(self.backbone, 'task_heads'):
            for k in list(self.backbone.task_heads.keys()):
                del self.backbone.task_heads[k]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Morphology head (mirrors PhysioWaveECGEncoder) ────────────────
        self.morph_proj = nn.Linear(embed_dim, latent_dim)
        # self.morph_pool = nn.AdaptiveAvgPool1d(n_morph_tokens)
        n_morph_tokens = 480 # hard coded according to the actual shape torch.Size([10, 480, 256])

        # ── Delay branch ──────────────────────────────────────────────────
        # Per-scale soft-argmax temperature (softplus-ed)
        self.beta_raw = nn.Parameter(torch.zeros(max_level))
        # Per-lead descriptor → latent
        delay_in = 2 * max_level   # [delta, mag] per detail scale
        self.delay_mlp = nn.Sequential(
            nn.Linear(delay_in, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.lead_pos_embed = nn.Parameter(torch.randn(1, n_leads, latent_dim) * 0.02)
        self.delay_encoder = TransformerEncoder(
            embed_dim=latent_dim,
            depth=n_delay_layers,
            num_heads=delay_num_heads,
            mlp_ratio=delay_mlp_ratio,
            dropout=dropout,
            rope_dim=None,
        )

        self.n_tokens = (1 + n_morph_tokens) if cond_type == 'concat' else n_morph_tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _wave_spec_and_morph(self, ecg_padded: torch.Tensor):
        """Run wavelet decomposition once; return (wave_spec, morph_tokens).

        wave_spec    : (B, (L+1)*C, T_pad) signed real
        morph_tokens : (B, n_morph_tokens, latent_dim)
        """
        wave_spec = self.backbone.wavelet_decomp(ecg_padded)        # (B, (L+1)*C, T_pad)
        wave_2d = wave_spec.unsqueeze(1)                            # (B, 1, (L+1)*C, T_pad)
        tokens = self.backbone.prepare_tokens(wave_2d)              # (B, L_patch, embed_dim)
        feats  = self.backbone.encoder(tokens)                      # (B, L_patch, embed_dim)
        feats  = self.morph_proj(feats)                             # (B, L_patch, latent_dim)
        # morph  = self.morph_pool(feats.transpose(1, 2)).transpose(1, 2)
        morph = feats
        return wave_spec, morph

    def _delay_descriptor(self, wave_spec: torch.Tensor):
        """Compute (delta, mag, ent) from wave_spec.

        wave_spec : (B, (L+1)*C, T)
        returns   : delta (B, C, L), mag (B, C, L), ent (B, C, L)
        """
        B, _, T = wave_spec.shape
        L, C = self.max_level, self.n_leads
        # Reshape to (B, L+1, C, T); use only L detail bands
        W = wave_spec.reshape(B, L + 1, C, T)
        W_det = W[:, :L].float()                                    # (B, L, C, T)

        # Cross-spectrum vs ref-lead → circular cross-correlation
        F_det = torch.fft.rfft(W_det, dim=-1)                        # (B, L, C, F)
        F_ref = F_det[:, :, self.ref_lead:self.ref_lead + 1, :]      # (B, L, 1, F)
        S = F_det * F_ref.conj()
        R = torch.fft.irfft(S, n=T, dim=-1)                          # (B, L, C, T)

        # Normalize by per-window L2 norms to get a coherence-like map
        eps = 1e-6
        norm_c = W_det.pow(2).sum(-1, keepdim=True).clamp(min=eps).sqrt()  # (B, L, C, 1)
        norm_r = norm_c[:, :, self.ref_lead:self.ref_lead + 1, :]          # (B, L, 1, 1)
        R = R / (norm_c * norm_r + eps)

        # Soft-argmax over the full (single-cycle) window
        beta = (F.softplus(self.beta_raw) + 1e-3).view(1, L, 1, 1)         # (1, L, 1, 1)
        Rb = R * beta
        logp = F.log_softmax(Rb, dim=-1)
        p = logp.exp()
        pos = torch.arange(T, device=R.device, dtype=R.dtype).view(1, 1, 1, T)
        tau = (p * pos).sum(-1)                                            # (B, L, C)
        mag = torch.logsumexp(Rb, dim=-1) / beta.squeeze(-1)               # (B, L, C)
        ent = -(p * logp).sum(-1)                                          # (B, L, C)

        # (B, L, C) → (B, C, L)
        tau = tau.transpose(1, 2).contiguous()
        mag = mag.transpose(1, 2).contiguous()
        ent = ent.transpose(1, 2).contiguous()

        delta = tau - tau[:, self.ref_lead:self.ref_lead + 1, :]            # (B, C, L)
        return delta, mag, ent

    def _delay_token(self, delta: torch.Tensor, mag: torch.Tensor):
        """Project (delta, mag) → 1 delay-core token (B, 1, latent_dim)."""
        desc = torch.cat([delta, mag], dim=-1)                              # (B, C, 2L)
        u = self.delay_mlp(desc)                                            # (B, C, D)
        u = u + self.lead_pos_embed                                         # broadcast
        u = self.delay_encoder(u)                                           # (B, C, D)
        return u.mean(dim=1, keepdim=True)                                  # (B, 1, D)

    def _wavelet_diversity(self, ecg_padded: torch.Tensor) -> torch.Tensor:
        """Negative entropy of the AdaptiveWaveletSelector's softmax weights,
        averaged over the batch. Penalises collapse to a single base."""
        sel = self.backbone.wavelet_decomp.selector.selector  # nn.Sequential ending in Softmax
        with torch.no_grad():
            pass  # selector itself is trainable; just want grad through it
        weights = sel(ecg_padded)                                  # (B, M)
        wbar = weights.mean(dim=0).clamp(min=1e-8)                  # (M,)
        H = -(wbar * wbar.log()).sum()
        return -H  # minimising −H ⇒ maximising entropy ⇒ diverse base usage

    def get_delay_map(self, ecg: torch.Tensor) -> dict:
        """Soft-argmax delay descriptor vs ref lead (same path as conditioning).

        Returns CPU tensors: delta, magnitude, entropy (each (B, n_leads, max_level)),
        mean_entropy (B,) per sample, wavelet_bands (B, max_level+1, n_leads, T_pad).
        """
        if self.pad_amount > 0:
            ecg_pad = F.pad(ecg, (0, self.pad_amount))
        else:
            ecg_pad = ecg
        wave_spec = self.backbone.wavelet_decomp(ecg_pad)
        delta, mag, ent = self._delay_descriptor(wave_spec)
        B, _, T = wave_spec.shape
        bands = wave_spec.reshape(B, self.max_level + 1, self.n_leads, T)
        return {
            'delta': delta.detach().float().cpu(),
            'magnitude': mag.detach().float().cpu(),
            'entropy': ent.detach().float().cpu(),
            'mean_entropy': ent.mean(dim=(1, 2)).detach().float().cpu(),
            'wavelet_bands': bands.detach().float().cpu(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ecg: torch.Tensor, return_aux: bool = False):
        """ecg: (B, n_leads, seq_len)  →  (B, n_tokens, latent_dim).

        If ``return_aux`` is True, returns (tokens, aux_dict) with keys
        ``shift_inv``, ``sparsity``, ``wavelet_diversity``, ``delay_subject_var``.
        """
        if self.pad_amount > 0:
            ecg_pad = F.pad(ecg, (0, self.pad_amount))
        else:
            ecg_pad = ecg

        wave_spec, morph_tokens = self._wave_spec_and_morph(ecg_pad)
        delta, mag, ent = self._delay_descriptor(wave_spec)
        ent_mean = ent.mean()
        delay_token = self._delay_token(delta, mag).to(morph_tokens.dtype)

        if self.cond_type == 'add':
            tokens = morph_tokens + delay_token  # broadcast over n_morph_tokens
        else:
            tokens = torch.cat([delay_token, morph_tokens], dim=1)

        if not return_aux:
            return tokens

        # # ── Auxiliary losses ─────────────────────────────────────────────
        # # 1. Sparsity (sharpness of the soft-argmax distributions)
        # L_sparsity = ent_mean

        # # 2. Wavelet diversity (entropy of the per-base selection weights)
        # L_div = self._wavelet_diversity(ecg_pad)

        # # 3. Per-subject variance of the relative delay matrix (NEGATIVE log)
        # if delta.shape[0] > 1:
        #     var_b = delta.float().var(dim=0, unbiased=False).mean()
        #     L_subject_var = -torch.log(var_b + 1e-6)
        # else:
        #     L_subject_var = delta.new_zeros(())

        # # 4. Shift-invariance: delays of a circularly-shifted ECG must match
        # T = ecg.shape[-1]
        # shift = int(torch.randint(1, T, (1,)).item())
        # ecg_shifted = torch.roll(ecg, shifts=shift, dims=-1)
        # if self.pad_amount > 0:
        #     ecg_shifted_pad = F.pad(ecg_shifted, (0, self.pad_amount))
        # else:
        #     ecg_shifted_pad = ecg_shifted
        # wave_spec_s = self.backbone.wavelet_decomp(ecg_shifted_pad)
        # delta_s, _, _ = self._delay_descriptor(wave_spec_s)
        # L_shift = F.mse_loss(delta.float(), delta_s.float())

        aux = {
            # 'shift_inv':         L_shift,
            # 'sparsity':          L_sparsity,
            # 'wavelet_diversity': L_div,
            # 'delay_subject_var': L_subject_var,
        }
        return tokens, aux


# ---------------------------------------------------------------------------
# DelayWave V2: gated delay token + windowed cross-correlation +
# subject-alignment contrastive loss. Strictly additive on top of the
# PhysioWave morphology backbone (alpha_delay=0 at init ⇒ equivalent to
# PhysioWaveECGEncoder).
# ---------------------------------------------------------------------------


class DelayWaveECGEncoderV2(nn.Module):
    """V2 of DelayWave with three core changes versus V1:

      1. **Gated delay token (zero-init).** A learnable scalar
         ``alpha_delay`` is multiplied onto the delay-core token and
         starts at 0. At init the encoder is exactly equivalent to
         ``PhysioWaveECGEncoder``; the delay branch can only contribute
         once gradients pull ``alpha_delay`` away from 0.
      2. **Windowed cross-correlation.** Soft-argmax restricted to
         ``[-tau_max, +tau_max]`` (default 50 samples ≈ 100 ms at 500 Hz).
         Eliminates wraparound ambiguity and yields informative
         descriptors from step 0.
      3. **No padding for the delay branch.** Wavelet front-end is
         applied to the un-padded ECG for the delay branch, avoiding
         zero-pad-induced fake peaks. Morphology branch keeps the
         padded path so patch embed is divisible.

    Loss surface: V1's ``shift_inv`` and ``delay_subject_var`` are
    replaced by a task-aligned ``delay_subject_align`` InfoNCE term that
    pushes two augmented views of the same ECG together while pushing
    different subjects apart. ``sparsity`` and ``wavelet_diversity`` are
    kept as stabilizers.
    """

    def __init__(
        self,
        latent_dim:           int   = 256,
        n_morph_tokens:       int   = 8,
        seq_len:              int   = 600,
        n_leads:              int   = 12,
        ref_lead:             int   = 1,
        # Wavelet front-end
        max_level:            int   = 3,
        wave_kernel_size:     int   = 24,
        wavelet_names:        list  = ('db4', 'db6', 'sym4', 'coif2'),
        use_separate_channel: bool  = True,
        # Morphology backbone (mirrors PhysioWaveECGEncoder)
        patch_size:           int   = 64,
        embed_dim:            int   = 384,
        depth:                int   = 8,
        num_heads:            int   = 12,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        use_pos_embed:        bool  = True,
        pos_embed_type:       str   = '2d',
        pretrained_path:      Optional[str] = None,
        freeze_backbone:      bool  = False,
        # Delay branch
        tau_max:              int   = 50,
        beta_init:            float = 2.0,
        delay_num_heads:      int   = 4,
        delay_mlp_ratio:      float = 4.0,
        # Contrastive aug knobs
        align_noise_std:      float = 0.05,
        align_amp_std:        float = 0.1,
        align_temperature:    float = 0.1,
    ):
        super().__init__()
        import importlib.util as _ilu
        with _physiowave_on_path():
            def _load(mod_name, file_name):
                if mod_name in sys.modules:
                    return sys.modules[mod_name]
                spec = _ilu.spec_from_file_location(
                    mod_name, os.path.join(_PHYSIOWAVE_DIR, file_name)
                )
                mod = _ilu.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod

            _load('wavelet_modules', 'wavelet_modules.py')
            tx_mod = _load('transformer_modules', 'transformer_modules.py')
            _load('head_modules', 'head_modules.py')
            pw_model = _load('physiowave_model', 'model.py')
            BERTWaveletTransformer = pw_model.BERTWaveletTransformer
            TransformerBlock = tx_mod.TransformerBlock

        assert 0 <= ref_lead < n_leads
        self.n_leads        = n_leads
        self.ref_lead       = ref_lead
        self.max_level      = max_level
        self.latent_dim     = latent_dim
        self.n_morph_tokens = n_morph_tokens
        self.tau_max        = int(tau_max)
        self.align_noise_std   = align_noise_std
        self.align_amp_std     = align_amp_std
        self.align_temperature = align_temperature

        self.patch_size = patch_size
        self.padded_len = ((seq_len + patch_size - 1) // patch_size) * patch_size
        self.pad_amount = self.padded_len - seq_len

        self.backbone = BERTWaveletTransformer(
            in_channels=n_leads,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=list(wavelet_names),
            use_separate_channel=use_separate_channel,
            patch_size=(1, patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type=None,
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[DelayWaveECGEncoderV2] loaded {pretrained_path}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

        if hasattr(self.backbone, 'task_heads'):
            for k in list(self.backbone.task_heads.keys()):
                del self.backbone.task_heads[k]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Morphology head (mirrors PhysioWaveECGEncoder) ────────────────
        self.morph_proj = nn.Linear(embed_dim, latent_dim)
        # self.morph_pool = nn.AdaptiveAvgPool1d(n_morph_tokens)
        n_morph_tokens = 480

        # ── Delay branch (delta-only descriptor) ──────────────────────────
        self.beta_raw = nn.Parameter(torch.full((max_level,), float(beta_init)))
        self.delay_mlp = nn.Sequential(
            nn.Linear(max_level, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.lead_pos_embed = nn.Parameter(torch.randn(1, n_leads, latent_dim) * 0.02)
        self.delay_block = TransformerBlock(
            dim=latent_dim,
            num_heads=delay_num_heads,
            mlp_ratio=delay_mlp_ratio,
            dropout=dropout,
            rope_dim=None,
        )
        # Zero-init gate: V2 ≡ PhysioWaveECGEncoder at step 0.
        self.alpha_delay = nn.Parameter(torch.zeros(()))

        self.n_tokens = n_leads + n_morph_tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _morph_tokens(self, ecg: torch.Tensor) -> torch.Tensor:
        if self.pad_amount > 0:
            ecg_pad = F.pad(ecg, (0, self.pad_amount))
        else:
            ecg_pad = ecg
        feats = self.backbone.forward_features(ecg_pad)            # (B, L, embed_dim)
        feats = self.morph_proj(feats)                              # (B, L, D)
        # feats = self.morph_pool(feats.transpose(1, 2)).transpose(1, 2)
        return feats

    def _delay_descriptor_wave_spec(self, wave_spec: torch.Tensor):
        """Soft-argmax delays from PhysioWave ``wavelet_decomp`` output."""
        B, flat, T = wave_spec.shape
        L = self.max_level
        C = self.n_leads
        if flat != (L + 1) * C:
            raise ValueError(
                f'wave_spec shape {wave_spec.shape} incompatible with '
                f'max_level={L}, n_leads={C}'
            )
        W = wave_spec.reshape(B, L + 1, C, T)
        W_det = W[:, :L].float()                                    # (B, L, C, T)

        F_det = torch.fft.rfft(W_det, dim=-1)
        F_ref = F_det[:, :, self.ref_lead:self.ref_lead + 1, :]
        S = F_det * F_ref.conj()
        R = torch.fft.irfft(S, n=T, dim=-1)                          # (B, L, C, T)

        eps = 1e-6
        norm_c = W_det.pow(2).sum(-1, keepdim=True).clamp(min=eps).sqrt()
        norm_r = norm_c[:, :, self.ref_lead:self.ref_lead + 1, :]
        R = R / (norm_c * norm_r + eps)

        R_shift = torch.fft.fftshift(R, dim=-1)
        center = T // 2
        tau_max = min(self.tau_max, center - 1)
        s, e = center - tau_max, center + tau_max + 1
        Rw = R_shift[..., s:e]
        W_len = Rw.shape[-1]

        beta = (F.softplus(self.beta_raw) + 1e-3).view(1, L, 1, 1)
        Rb = Rw * beta
        logp = F.log_softmax(Rb, dim=-1)
        p = logp.exp()
        pos = torch.arange(
            -tau_max, tau_max + 1,
            device=wave_spec.device, dtype=wave_spec.dtype,
        ).view(1, 1, 1, W_len)
        tau = (p * pos).sum(-1)
        ent = -(p * logp).sum(-1)

        tau = tau.transpose(1, 2).contiguous()
        ent = ent.transpose(1, 2).contiguous()
        delta = tau - tau[:, self.ref_lead:self.ref_lead + 1, :]
        return delta, ent

    def _delay_descriptor(self, ecg: torch.Tensor):
        """Windowed soft-argmax delay descriptor on the un-padded ECG.

        Returns delta (B, n_leads, max_level), entropy grid (same shape).
        """
        wave_spec = self.backbone.wavelet_decomp(ecg)
        return self._delay_descriptor_wave_spec(wave_spec)

    def _delay_repr(self, ecg: torch.Tensor):
        """Project descriptor → 1 residual block → (per-lead, pooled, ent)."""
        delta, ent = self._delay_descriptor(ecg)
        ent_mean = ent.mean()
        u = self.delay_mlp(delta)                                    # (B, C, D)
        u = u + self.lead_pos_embed
        u = self.delay_block(u)                                      # (B, C, D)
        delay_token = u#.mean(dim=1, keepdim=True)                    # (B, 1, D)
        return u, delay_token, ent_mean

    def _wavelet_diversity(self, ecg: torch.Tensor) -> torch.Tensor:
        sel = self.backbone.wavelet_decomp.selector.selector
        weights = sel(ecg)                                           # (B, M)
        wbar = weights.mean(dim=0).clamp(min=1e-8)
        H = -(wbar * wbar.log()).sum()
        return -H

    def _augment_for_align(self, ecg: torch.Tensor) -> torch.Tensor:
        B, C, T = ecg.shape
        amp = 1.0 + self.align_amp_std * torch.randn(B, C, 1, device=ecg.device, dtype=ecg.dtype)
        noise = self.align_noise_std * torch.randn_like(ecg)
        return ecg * amp + noise

    def _delay_subject_align(self, ecg: torch.Tensor) -> torch.Tensor:
        B = ecg.shape[0]
        if B < 2:
            return ecg.new_zeros(())
        v1 = self._augment_for_align(ecg)
        v2 = self._augment_for_align(ecg)
        u1, _, _ = self._delay_repr(v1)
        u2, _, _ = self._delay_repr(v2)
        z1 = F.normalize(u1.mean(dim=1).float(), dim=-1)
        z2 = F.normalize(u2.mean(dim=1).float(), dim=-1)
        logits = (z1 @ z2.t()) / self.align_temperature
        targets = torch.arange(B, device=ecg.device)
        return F.cross_entropy(logits, targets)

    def get_delay_map(self, ecg: torch.Tensor) -> dict:
        """Per-lead delay stats plus wavelet bands (B, max_level+1, n_leads, T)."""
        wave_spec = self.backbone.wavelet_decomp(ecg)
        delta, ent = self._delay_descriptor_wave_spec(wave_spec)
        B, _, T = wave_spec.shape
        bands = wave_spec.reshape(B, self.max_level + 1, self.n_leads, T)
        return {
            'delta': delta.detach().float().cpu(),
            'entropy': ent.detach().float().cpu(),
            'mean_entropy': ent.mean(dim=(1, 2)).detach().float().cpu(),
            'wavelet_bands': bands.detach().float().cpu(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ecg: torch.Tensor, return_aux: bool = False):
        """ecg: (B, n_leads, seq_len) → (B, 1 + n_morph_tokens, latent_dim)."""
        morph_tokens = self._morph_tokens(ecg)
        # If shuffle_delay is set, compute the delay branch from other subjects'
        # ECG (cyclic within-batch roll) while morphology stays subject-correct — an
        # on-distribution test of whether the delay tokens carry subject-specific info.
        ecg_delay = torch.roll(ecg, 1, 0) if getattr(self, 'shuffle_delay', False) else ecg
        _, delay_token, ent_mean = self._delay_repr(ecg_delay)

        delay_token = (self.alpha_delay * delay_token).to(morph_tokens.dtype)
        tokens = torch.cat([delay_token, morph_tokens], dim=1)

        if not return_aux:
            return tokens

        # L_sparsity   = ent_mean
        # L_div        = self._wavelet_diversity(ecg)
        # L_align      = self._delay_subject_align(ecg)
        # L_delay_norm = delay_token.float().pow(2).mean()

        aux = {
            # 'sparsity':            L_sparsity,
            # 'wavelet_diversity':   L_div,
            # 'delay_subject_align': L_align,
            # 'delay_norm':          L_delay_norm,
        }
        return tokens, aux



class DelayWaveECGEncoderV3(nn.Module):
    """V3 of DelayWave with three core changes versus V1 and V2:
    """

    def __init__(
        self,
        latent_dim:           int   = 256,
        n_morph_tokens:       int   = 8,
        seq_len:              int   = 600,
        n_leads:              int   = 12,
        ref_lead:             int   = 1,
        # 'single' (paper): each lead's delay is measured against ref_lead only, so a
        # token carries max_level numbers. 'full': every lead is cross-correlated
        # against EVERY other, so lead c's token carries its lag against all n_leads
        # references at all max_level scales -- the whole n_leads x n_leads x max_level
        # map rather than one row of it. Token COUNT is unchanged; only what each token
        # knows changes. ref_lead still fixes the gauge and the warm-start slot.
        ref_mode:             str   = 'single',
        # Learning-rate reparameterisation for the columns that ref_mode='full' adds.
        # They start at exactly 0, and Adam's per-step displacement is ~lr regardless of
        # gradient size, so at lr 1e-5 over 12k steps such a column travels ~0.06 while
        # the warm-started lead-II block sits at per-entry RMS ~0.33. Measured gradients
        # on the new columns are 1.14x lead II's, so the blocker is runway, not signal.
        # Scaling those INPUT columns by c makes the effective contribution travel c*lr
        # per step. c*0 = 0, so the step-0 identity with the shipped model is preserved.
        extra_scale:          float = 1.0,
        # Wavelet front-end
        max_level:            int   = 3,
        wave_kernel_size:     int   = 24,
        wavelet_names:        list  = ('db4', 'db6', 'sym4', 'coif2'),
        use_separate_channel: bool  = True,
        # Morphology backbone (mirrors PhysioWaveECGEncoder)
        patch_size:           int   = 64,
        embed_dim:            int   = 384,
        depth:                int   = 8,
        num_heads:            int   = 12,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        use_pos_embed:        bool  = True,
        pos_embed_type:       str   = '2d',
        pretrained_path:      Optional[str] = None,
        freeze_backbone:      bool  = False,
        # Delay branch
        tau_max:              int   = 50,
        beta_init:            float = 2.0,
        delay_num_heads:      int   = 4,
        delay_mlp_ratio:      float = 4.0,
        # Contrastive aug knobs
        align_noise_std:      float = 0.05,
        align_amp_std:        float = 0.1,
        align_temperature:    float = 0.1,
    ):
        super().__init__()
        import importlib.util as _ilu
        with _physiowave_on_path():
            def _load(mod_name, file_name):
                if mod_name in sys.modules:
                    return sys.modules[mod_name]
                spec = _ilu.spec_from_file_location(
                    mod_name, os.path.join(_PHYSIOWAVE_DIR, file_name)
                )
                mod = _ilu.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod

            _load('wavelet_modules', 'wavelet_modules.py')
            tx_mod = _load('transformer_modules', 'transformer_modules.py')
            _load('head_modules', 'head_modules.py')
            pw_model = _load('physiowave_model', 'model.py')
            BERTWaveletTransformer = pw_model.BERTWaveletTransformer
            TransformerBlock = tx_mod.TransformerBlock

        assert 0 <= ref_lead < n_leads
        assert ref_mode in ('single', 'full'), ref_mode
        self.n_leads        = n_leads
        self.ref_lead       = ref_lead
        self.ref_mode       = ref_mode
        self.max_level      = max_level
        self.latent_dim     = latent_dim
        n_morph_tokens = n_leads
        self.n_morph_tokens = n_morph_tokens
        self.tau_max        = int(tau_max)
        self.align_noise_std   = align_noise_std
        self.align_amp_std     = align_amp_std
        self.align_temperature = align_temperature

        self.patch_size = patch_size
        self.padded_len = ((seq_len + patch_size - 1) // patch_size) * patch_size
        self.pad_amount = self.padded_len - seq_len

        self.backbone = BERTWaveletTransformer(
            in_channels=n_leads,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=list(wavelet_names),
            use_separate_channel=use_separate_channel,
            patch_size=(1, patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type=None,
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[DelayWaveECGEncoderV3] loaded {pretrained_path}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

        if hasattr(self.backbone, 'task_heads'):
            for k in list(self.backbone.task_heads.keys()):
                del self.backbone.task_heads[k]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Morphology head (mirrors PhysioWaveECGEncoder) ────────────────
        self.morph_proj = nn.Linear(embed_dim, latent_dim)
        self.morph_pool = nn.AdaptiveAvgPool1d(n_morph_tokens)

        # ── Delay branch (delta-only descriptor) ──────────────────────────
        self.beta_raw = nn.Parameter(torch.full((max_level,), float(beta_init)))
        # 'full' widens ONLY this first projection: max_level -> n_leads * max_level.
        # Everything downstream (lead_pos_embed, delay_block, alpha_delay, token count)
        # is untouched, so the two modes differ by exactly one weight matrix.
        d_delay_in = max_level * (n_leads if ref_mode == 'full' else 1)
        self.d_delay_in = d_delay_in
        self.extra_scale = float(extra_scale)
        if ref_mode == 'full' and self.extra_scale != 1.0:
            _s = torch.full((d_delay_in,), self.extra_scale)
            _s[ref_lead * max_level:(ref_lead + 1) * max_level] = 1.0
        else:
            _s = torch.ones(d_delay_in)
        # non-persistent: stays out of state_dict so warm-start stays strict-clean
        self.register_buffer('delay_in_scale', _s, persistent=False)
        self.delay_mlp = nn.Sequential(
            nn.Linear(d_delay_in, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.lead_pos_embed = nn.Parameter(torch.randn(1, n_leads, latent_dim) * 0.02)
        self.delay_block = TransformerBlock(
            dim=latent_dim,
            num_heads=delay_num_heads,
            mlp_ratio=delay_mlp_ratio,
            dropout=dropout,
            rope_dim=None,
        )
        # Zero-init gate: V2 ≡ PhysioWaveECGEncoder at step 0.
        self.alpha_delay = nn.Parameter(torch.zeros(()))

        self.n_tokens = n_leads + n_morph_tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _morph_tokens(self, ecg: torch.Tensor) -> torch.Tensor:
        if self.pad_amount > 0:
            ecg_pad = F.pad(ecg, (0, self.pad_amount))
        else:
            ecg_pad = ecg
        feats = self.backbone.forward_features(ecg_pad)            # (B, L, embed_dim)
        feats = self.morph_proj(feats)                              # (B, L, D)
        feats = self.morph_pool(feats.transpose(1, 2)).transpose(1, 2)
        return feats

    def _delay_descriptor_wave_spec(self, wave_spec: torch.Tensor):
        """Soft-argmax delays from PhysioWave ``wavelet_decomp`` output."""
        B, flat, T = wave_spec.shape
        L = self.max_level
        C = self.n_leads
        if flat != (L + 1) * C:
            raise ValueError(
                f'wave_spec shape {wave_spec.shape} incompatible with '
                f'max_level={L}, n_leads={C}'
            )
        W = wave_spec.reshape(B, L + 1, C, T)
        W_det = W[:, :L].float()

        F_det = torch.fft.rfft(W_det, dim=-1)
        eps = 1e-6
        nrm = W_det.pow(2).sum(-1).clamp(min=eps).sqrt()             # (B, L, C)

        if self.ref_mode == 'full':
            # All-pairs cross-spectrum: S[b,l,r,c,f] = F[c] * conj(F[r]).
            S = F_det.unsqueeze(2) * F_det.unsqueeze(3).conj()       # (B,L,R,C,Fq)
            R = torch.fft.irfft(S, n=T, dim=-1)                      # (B,L,R,C,T)
            den = nrm.unsqueeze(2) * nrm.unsqueeze(3)                # (B,L,R,C)
            R = R / (den.unsqueeze(-1) + eps)
            bshape = (1, L, 1, 1, 1)
        else:
            F_ref = F_det[:, :, self.ref_lead:self.ref_lead + 1, :]
            S = F_det * F_ref.conj()
            R = torch.fft.irfft(S, n=T, dim=-1)                      # (B,L,C,T)
            norm_c = nrm.unsqueeze(-1)
            norm_r = norm_c[:, :, self.ref_lead:self.ref_lead + 1, :]
            R = R / (norm_c * norm_r + eps)
            bshape = (1, L, 1, 1)

        R_shift = torch.fft.fftshift(R, dim=-1)
        center = T // 2
        tau_max = min(self.tau_max, center - 1)
        s, e = center - tau_max, center + tau_max + 1
        Rw = R_shift[..., s:e]
        W_len = Rw.shape[-1]

        beta = (F.softplus(self.beta_raw) + 1e-3).view(*bshape)
        Rb = Rw * beta
        logp = F.log_softmax(Rb, dim=-1)
        p = logp.exp()
        pos = torch.arange(
            -tau_max, tau_max + 1,
            device=wave_spec.device, dtype=wave_spec.dtype,
        ).view(*([1] * (Rw.dim() - 1)), W_len)
        tau = (p * pos).sum(-1)
        ent = -(p * logp).sum(-1)

        if self.ref_mode == 'full':
            # (B,L,R,C) -> (B,R,C,L); gauge-fix each reference row by its own self-lag,
            # exactly as the single-reference path subtracts tau[ref].
            tau = tau.permute(0, 2, 3, 1).contiguous()               # (B,R,C,L)
            ent = ent.permute(0, 2, 3, 1).contiguous()
            di = torch.arange(C, device=tau.device)
            tau = tau - tau[:, di, di, :].unsqueeze(2)               # (B,R,C,L)
            # Token axis is the LEAD c; its features are (reference, scale), r-major,
            # so reference r occupies columns [r*L : (r+1)*L] -- the slot the
            # single-reference weights are scattered into at warm start.
            delta = tau.permute(0, 2, 1, 3).reshape(B, C, C * L)
            ent = ent.permute(0, 2, 1, 3).reshape(B, C, C * L)
            return delta, ent

        tau = tau.transpose(1, 2).contiguous()
        ent = ent.transpose(1, 2).contiguous()
        delta = tau - tau[:, self.ref_lead:self.ref_lead + 1, :]
        return delta, ent

    def _delay_descriptor(self, ecg: torch.Tensor):
        """Windowed soft-argmax delay descriptor on the un-padded ECG.

        Returns delta (B, n_leads, max_level), entropy grid (same shape).
        """
        wave_spec = self.backbone.wavelet_decomp(ecg)
        return self._delay_descriptor_wave_spec(wave_spec)

    def _delay_repr(self, ecg: torch.Tensor):
        """Project descriptor → 1 residual block → (per-lead, pooled, ent)."""
        delta, ent = self._delay_descriptor(ecg)
        ent_mean = ent.mean()
        delta = delta * self.delay_in_scale.to(delta.dtype)
        u = self.delay_mlp(delta)                                    # (B, C, D)
        u = u + self.lead_pos_embed
        u = self.delay_block(u)                                      # (B, C, D)
        delay_token = u#.mean(dim=1, keepdim=True)                    # (B, 1, D)
        return u, delay_token, ent_mean

    def _wavelet_diversity(self, ecg: torch.Tensor) -> torch.Tensor:
        sel = self.backbone.wavelet_decomp.selector.selector
        weights = sel(ecg)                                           # (B, M)
        wbar = weights.mean(dim=0).clamp(min=1e-8)
        H = -(wbar * wbar.log()).sum()
        return -H

    def _augment_for_align(self, ecg: torch.Tensor) -> torch.Tensor:
        B, C, T = ecg.shape
        amp = 1.0 + self.align_amp_std * torch.randn(B, C, 1, device=ecg.device, dtype=ecg.dtype)
        noise = self.align_noise_std * torch.randn_like(ecg)
        return ecg * amp + noise

    def _delay_subject_align(self, ecg: torch.Tensor) -> torch.Tensor:
        B = ecg.shape[0]
        if B < 2:
            return ecg.new_zeros(())
        v1 = self._augment_for_align(ecg)
        v2 = self._augment_for_align(ecg)
        u1, _, _ = self._delay_repr(v1)
        u2, _, _ = self._delay_repr(v2)
        z1 = F.normalize(u1.mean(dim=1).float(), dim=-1)
        z2 = F.normalize(u2.mean(dim=1).float(), dim=-1)
        logits = (z1 @ z2.t()) / self.align_temperature
        targets = torch.arange(B, device=ecg.device)
        return F.cross_entropy(logits, targets)

    def get_delay_map(self, ecg: torch.Tensor) -> dict:
        """Per-lead delay stats plus wavelet bands (B, max_level+1, n_leads, T)."""
        wave_spec = self.backbone.wavelet_decomp(ecg)
        delta, ent = self._delay_descriptor_wave_spec(wave_spec)
        B, _, T = wave_spec.shape
        bands = wave_spec.reshape(B, self.max_level + 1, self.n_leads, T)
        return {
            'delta': delta.detach().float().cpu(),
            'entropy': ent.detach().float().cpu(),
            'mean_entropy': ent.mean(dim=(1, 2)).detach().float().cpu(),
            'wavelet_bands': bands.detach().float().cpu(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ecg: torch.Tensor, return_aux: bool = False):
        """ecg: (B, n_leads, seq_len) → (B, 1 + n_morph_tokens, latent_dim)."""
        morph_tokens = self._morph_tokens(ecg)
        # If shuffle_delay is set, compute the delay branch from other subjects'
        # ECG (cyclic within-batch roll) while morphology stays subject-correct — an
        # on-distribution test of whether the delay tokens carry subject-specific info.
        ecg_delay = torch.roll(ecg, 1, 0) if getattr(self, 'shuffle_delay', False) else ecg
        _, delay_token, ent_mean = self._delay_repr(ecg_delay)

        delay_token = (self.alpha_delay * delay_token).to(morph_tokens.dtype)
        tokens = torch.cat([delay_token, morph_tokens], dim=1)

        if not return_aux:
            return tokens

        # L_sparsity   = ent_mean
        # L_div        = self._wavelet_diversity(ecg)
        # L_align      = self._delay_subject_align(ecg)
        # L_delay_norm = delay_token.float().pow(2).mean()

        aux = {
            # 'sparsity':            L_sparsity,
            # 'wavelet_diversity':   L_div,
            # 'delay_subject_align': L_align,
            # 'delay_norm':          L_delay_norm,
        }
        return tokens, aux


class DelayWaveECGEncoderV4(nn.Module):
    """V4 of DelayWave: per-(lead, scale) delay tokens with Fourier-feature
    embedding of the continuous lag value.

    Differences vs. V3 (descriptor and morphology branch are byte-identical):
      1. **Per-(lead, scale) tokenization.** Instead of mixing the L scales
         into one per-lead vector via an MLP and running attention over the
         C lead tokens, V4 emits one token per (lead, scale), giving a
         token grid of shape (B, C*L, D). Cross-scale interactions become
         data-dependent via the delay block's attention.
      2. **Fourier-feature lag embedding.** The continuous lag value
         ``delta`` is lifted with sinusoidal features (log-spaced, scaled
         to ``tau_max``), then a small MLP maps to ``latent_dim``. This is
         continuous, differentiable through ``beta`` and the wavelet
         front-end, and gives a multi-resolution view of the lag without
         needing the network to learn ordinality.
      3. **Additive lead + scale embeddings.** Two learned positional
         embeddings, broadcast across the other axis, label each grid
         cell.
    """

    def __init__(
        self,
        latent_dim:           int   = 256,
        n_morph_tokens:       int   = 8,
        seq_len:              int   = 600,
        n_leads:              int   = 12,
        ref_lead:             int   = 1,
        # Wavelet front-end
        max_level:            int   = 3,
        wave_kernel_size:     int   = 24,
        wavelet_names:        list  = ('db4', 'db6', 'sym4', 'coif2'),
        use_separate_channel: bool  = True,
        # Morphology backbone (mirrors PhysioWaveECGEncoder)
        patch_size:           int   = 64,
        embed_dim:            int   = 384,
        depth:                int   = 8,
        num_heads:            int   = 12,
        mlp_ratio:            float = 4.0,
        dropout:              float = 0.1,
        use_pos_embed:        bool  = True,
        pos_embed_type:       str   = '2d',
        pretrained_path:      Optional[str] = None,
        freeze_backbone:      bool  = False,
        # Delay branch
        tau_max:              int   = 50,
        beta_init:            float = 2.0,
        delay_num_heads:      int   = 4,
        delay_mlp_ratio:      float = 4.0,
        num_freq_bands:       int   = 8,
        # Contrastive aug knobs
        align_noise_std:      float = 0.05,
        align_amp_std:        float = 0.1,
        align_temperature:    float = 0.1,
    ):
        super().__init__()
        import importlib.util as _ilu
        with _physiowave_on_path():
            def _load(mod_name, file_name):
                if mod_name in sys.modules:
                    return sys.modules[mod_name]
                spec = _ilu.spec_from_file_location(
                    mod_name, os.path.join(_PHYSIOWAVE_DIR, file_name)
                )
                mod = _ilu.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod

            _load('wavelet_modules', 'wavelet_modules.py')
            tx_mod = _load('transformer_modules', 'transformer_modules.py')
            _load('head_modules', 'head_modules.py')
            pw_model = _load('physiowave_model', 'model.py')
            BERTWaveletTransformer = pw_model.BERTWaveletTransformer
            TransformerBlock = tx_mod.TransformerBlock

        assert 0 <= ref_lead < n_leads
        self.n_leads        = n_leads
        self.ref_lead       = ref_lead
        self.max_level      = max_level
        self.latent_dim     = latent_dim
        n_morph_tokens = n_leads*max_level
        self.n_morph_tokens = n_morph_tokens
        self.tau_max        = int(tau_max)
        self.num_freq_bands = int(num_freq_bands)
        self.align_noise_std   = align_noise_std
        self.align_amp_std     = align_amp_std
        self.align_temperature = align_temperature

        self.patch_size = patch_size
        self.padded_len = ((seq_len + patch_size - 1) // patch_size) * patch_size
        self.pad_amount = self.padded_len - seq_len

        self.backbone = BERTWaveletTransformer(
            in_channels=n_leads,
            max_level=max_level,
            wave_kernel_size=wave_kernel_size,
            wavelet_names=list(wavelet_names),
            use_separate_channel=use_separate_channel,
            patch_size=(1, patch_size),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            task_type=None,
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[DelayWaveECGEncoderV4] loaded {pretrained_path}: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

        if hasattr(self.backbone, 'task_heads'):
            for k in list(self.backbone.task_heads.keys()):
                del self.backbone.task_heads[k]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Morphology head (mirrors PhysioWaveECGEncoder) ────────────────
        self.morph_proj = nn.Linear(embed_dim, latent_dim)
        self.morph_pool = nn.AdaptiveAvgPool1d(n_morph_tokens)

        # ── Delay branch ───────────────────────────────────────────────────
        self.beta_raw = nn.Parameter(torch.full((max_level,), float(beta_init)))

        # Log-spaced Fourier frequencies, scaled so the highest band's period
        # ≈ 2 samples and the lowest spans ~tau_max samples. Keeps the
        # embedding window-aware: changing tau_max rescales frequencies.
        freqs = (2.0 ** torch.arange(num_freq_bands).float()) * math.pi / float(max(tau_max, 1))
        self.register_buffer('delay_freqs', freqs)                # (F,)

        self.delay_value_embed = nn.Sequential(
            nn.Linear(2 * num_freq_bands, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.lead_embed  = nn.Parameter(torch.randn(1, n_leads, 1, latent_dim) * 0.02)
        self.scale_embed = nn.Parameter(torch.randn(1, 1, max_level, latent_dim) * 0.02)

        self.delay_block = TransformerBlock(
            dim=latent_dim,
            num_heads=delay_num_heads,
            mlp_ratio=delay_mlp_ratio,
            dropout=dropout,
            rope_dim=None,
        )
        # Zero-init gate: V4 ≡ PhysioWaveECGEncoder at step 0.
        self.alpha_delay = nn.Parameter(torch.zeros(()))

        self.n_tokens = n_leads * max_level + n_morph_tokens

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _morph_tokens(self, ecg: torch.Tensor) -> torch.Tensor:
        if self.pad_amount > 0:
            ecg_pad = F.pad(ecg, (0, self.pad_amount))
        else:
            ecg_pad = ecg
        feats = self.backbone.forward_features(ecg_pad)            # (B, L, embed_dim)
        feats = self.morph_proj(feats)                              # (B, L, D)
        feats = self.morph_pool(feats.transpose(1, 2)).transpose(1, 2)
        return feats

    def _delay_descriptor_wave_spec(self, wave_spec: torch.Tensor):
        """Soft-argmax delays from PhysioWave ``wavelet_decomp`` output."""
        B, flat, T = wave_spec.shape
        L = self.max_level
        C = self.n_leads
        if flat != (L + 1) * C:
            raise ValueError(
                f'wave_spec shape {wave_spec.shape} incompatible with '
                f'max_level={L}, n_leads={C}'
            )
        W = wave_spec.reshape(B, L + 1, C, T)
        W_det = W[:, :L].float()

        F_det = torch.fft.rfft(W_det, dim=-1)
        F_ref = F_det[:, :, self.ref_lead:self.ref_lead + 1, :]
        S = F_det * F_ref.conj()
        R = torch.fft.irfft(S, n=T, dim=-1)

        eps = 1e-6
        norm_c = W_det.pow(2).sum(-1, keepdim=True).clamp(min=eps).sqrt()
        norm_r = norm_c[:, :, self.ref_lead:self.ref_lead + 1, :]
        R = R / (norm_c * norm_r + eps)

        R_shift = torch.fft.fftshift(R, dim=-1)
        center = T // 2
        tau_max = min(self.tau_max, center - 1)
        s, e = center - tau_max, center + tau_max + 1
        Rw = R_shift[..., s:e]
        W_len = Rw.shape[-1]

        beta = (F.softplus(self.beta_raw) + 1e-3).view(1, L, 1, 1)
        Rb = Rw * beta
        logp = F.log_softmax(Rb, dim=-1)
        p = logp.exp()
        pos = torch.arange(
            -tau_max, tau_max + 1,
            device=wave_spec.device, dtype=wave_spec.dtype,
        ).view(1, 1, 1, W_len)
        tau = (p * pos).sum(-1)
        ent = -(p * logp).sum(-1)

        tau = tau.transpose(1, 2).contiguous()
        ent = ent.transpose(1, 2).contiguous()
        delta = tau - tau[:, self.ref_lead:self.ref_lead + 1, :]
        return delta, ent

    def _delay_descriptor(self, ecg: torch.Tensor):
        """Windowed soft-argmax delay descriptor on the un-padded ECG.

        Returns delta (B, n_leads, max_level), entropy grid (same shape).
        """
        wave_spec = self.backbone.wavelet_decomp(ecg)
        return self._delay_descriptor_wave_spec(wave_spec)

    def _fourier_features(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: (B, C, L) → (B, C, L, 2*F) sin/cos features."""
        phi = delta.unsqueeze(-1) * self.delay_freqs                # (B, C, L, F)
        return torch.cat([phi.sin(), phi.cos()], dim=-1)            # (B, C, L, 2F)

    def _delay_repr(self, ecg: torch.Tensor):
        """Build per-(lead, scale) tokens, run one delay block.

        Returns (grid_tokens, delay_tokens, ent_mean) where both grid_tokens
        and delay_tokens have shape (B, C*L, D). The two are aliased to
        match V2/V3's 3-tuple signature; downstream uses delay_tokens.
        """
        delta, ent = self._delay_descriptor(ecg)                     # (B, C, L)
        ent_mean = ent.mean()
        feat = self._fourier_features(delta)                         # (B, C, L, 2F)
        u = self.delay_value_embed(feat)                             # (B, C, L, D)
        u = u + self.lead_embed + self.scale_embed                   # broadcast
        B, C, L, D = u.shape
        u = u.reshape(B, C * L, D)
        u = self.delay_block(u)                                      # (B, C*L, D)
        return u, u, ent_mean

    def _wavelet_diversity(self, ecg: torch.Tensor) -> torch.Tensor:
        sel = self.backbone.wavelet_decomp.selector.selector
        weights = sel(ecg)                                           # (B, M)
        wbar = weights.mean(dim=0).clamp(min=1e-8)
        H = -(wbar * wbar.log()).sum()
        return -H

    def _augment_for_align(self, ecg: torch.Tensor) -> torch.Tensor:
        B, C, T = ecg.shape
        amp = 1.0 + self.align_amp_std * torch.randn(B, C, 1, device=ecg.device, dtype=ecg.dtype)
        noise = self.align_noise_std * torch.randn_like(ecg)
        return ecg * amp + noise

    def _delay_subject_align(self, ecg: torch.Tensor) -> torch.Tensor:
        B = ecg.shape[0]
        if B < 2:
            return ecg.new_zeros(())
        v1 = self._augment_for_align(ecg)
        v2 = self._augment_for_align(ecg)
        u1, _, _ = self._delay_repr(v1)
        u2, _, _ = self._delay_repr(v2)
        z1 = F.normalize(u1.mean(dim=1).float(), dim=-1)
        z2 = F.normalize(u2.mean(dim=1).float(), dim=-1)
        logits = (z1 @ z2.t()) / self.align_temperature
        targets = torch.arange(B, device=ecg.device)
        return F.cross_entropy(logits, targets)

    def get_delay_map(self, ecg: torch.Tensor) -> dict:
        """Per-lead delay stats plus wavelet bands (B, max_level+1, n_leads, T)."""
        wave_spec = self.backbone.wavelet_decomp(ecg)
        delta, ent = self._delay_descriptor_wave_spec(wave_spec)
        B, _, T = wave_spec.shape
        bands = wave_spec.reshape(B, self.max_level + 1, self.n_leads, T)
        return {
            'delta': delta.detach().float().cpu(),
            'entropy': ent.detach().float().cpu(),
            'mean_entropy': ent.mean(dim=(1, 2)).detach().float().cpu(),
            'wavelet_bands': bands.detach().float().cpu(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ecg: torch.Tensor, return_aux: bool = False):
        """ecg: (B, n_leads, seq_len) → (B, n_leads*max_level + n_morph_tokens, latent_dim)."""
        morph_tokens = self._morph_tokens(ecg)
        _, delay_tokens, ent_mean = self._delay_repr(ecg)

        delay_tokens = (self.alpha_delay * delay_tokens).to(morph_tokens.dtype)
        tokens = torch.cat([delay_tokens, morph_tokens], dim=1)

        if not return_aux:
            return tokens

        aux = {}
        return tokens, aux


# ---------------------------------------------------------------------------
# Additional temporal foundation models as conditioning encoders
#
# Every encoder below emits **12 tokens x latent_dim**: one per ECG lead where
# the architecture keeps lead identity, otherwise an adaptive average pool of
# the model's own native token axis down to 12.  Nothing else in the pipeline
# changes.  See experiments/2026.07.25-run04_e17-temporal-transformer-baselines/results.md.
# ---------------------------------------------------------------------------

def _cfg_kwargs(model_config, key):
    """Sub-config block -> plain dict of kwargs (OmegaConf-safe)."""
    sub = getattr(model_config, key, None) or {}
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(sub, DictConfig):
            return OmegaConf.to_container(sub, resolve=True)
    except ImportError:
        pass
    return dict(sub)


_MEDFORMER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Medformer')
)
_MOMENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'moment')
)


@contextmanager
def _repo_on_path(repo_dir, shadowed=('layers', 'models', 'utils')):
    """Import from a vendored research repo that uses absolute top-level imports.

    TeCh and Medformer both ship packages called ``layers`` / ``models`` /
    ``utils``.  Whichever is imported first wins for the whole process, so we
    stash and purge the colliding ``sys.modules`` entries around the import.
    Already-constructed objects keep their own class references, so restoring
    the old entries afterwards is safe.
    """
    saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in shadowed}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, repo_dir)
    try:
        yield
    finally:
        try:
            sys.path.remove(repo_dir)
        except ValueError:
            pass
        for k in [k for k in sys.modules if k.split('.')[0] in shadowed]:
            del sys.modules[k]
        sys.modules.update(saved)


class MedformerFamilyECGEncoder(nn.Module):
    """PatchTST / iTransformer / Medformer, taken verbatim from the official
    Medformer MedTS-benchmark repo (https://github.com/DL4mHealth/Medformer) —
    the same codebase TeCh benchmarks against — and tapped at the
    pre-classifier embedding instead of the class projector.

    model_name:
        'patchtst'     — channel-INDEPENDENT patch transformer (ICLR'23).
                         Native tokens (12 leads x n_patches); mean-pooled over
                         patches to 12 per-lead tokens.
        'itransformer' — inverted transformer (ICLR'24); one token per variate,
                         so natively exactly 12 lead tokens.
        'medformer'    — multi-granularity patching + cross-granularity
                         attention (NeurIPS'24 D&B), prior MedTS SOTA; its
                         encoder returns one token per granularity, so the
                         published 12-entry PTB `patch_len_list` gives 12 tokens.
    """

    _MODULE = {'patchtst': 'PatchTST',
               'itransformer': 'iTransformer',
               'medformer': 'Medformer'}

    def __init__(
        self,
        model_name:   str,
        latent_dim:   int = 256,
        seq_len:      int = 600,
        enc_in:       int = 12,
        e_layers:     int = 6,
        d_ff:         int = 512,
        n_heads:      int = 8,
        dropout:      float = 0.1,
        patch_len:    int = 16,     # PatchTST default (Medformer repo run.py)
        stride:       int = 8,      # PatchTST default
        patch_len_list: str = "2,4,8,8,16,16,16,32,32,32,32,32",  # published PTB list
        augmentations: str = "none",
        n_tokens:     int = 12,
    ):
        super().__init__()
        model_name = model_name.lower()
        if model_name not in self._MODULE:
            raise ValueError(f"unknown Medformer-family model '{model_name}'")
        self.model_name = model_name
        self.n_tokens = n_tokens

        cfg = SimpleNamespace(
            task_name='classification', seq_len=seq_len, pred_len=0,
            enc_in=enc_in, e_layers=e_layers, d_model=latent_dim, d_ff=d_ff,
            n_heads=n_heads, dropout=dropout, factor=1, activation='gelu',
            output_attention=False, embed='timeF', freq='h', num_class=1,
            patch_len=patch_len, stride=stride, patch_len_list=patch_len_list,
            augmentations=augmentations, single_channel=False, no_inter_attn=False,
        )
        import importlib
        with _repo_on_path(_MEDFORMER_DIR):
            mod = importlib.import_module(f'models.{self._MODULE[model_name]}')
            self.net = mod.Model(cfg)
        # The class projector is unused; unused parameters make DDP hang in
        # `_verify_param_shape_across_processes` (same reason TeCh's is deleted).
        if hasattr(self.net, 'projection'):
            del self.net.projection

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600) → (B, 12, latent_dim)"""
        x = ecg.transpose(1, 2)                       # (B, seq_len, enc_in)
        net = self.net
        if self.model_name == 'patchtst':
            # Reproduces Model.classification() up to (not including) the head.
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            enc_out, n_vars = net.patch_embedding(x.permute(0, 2, 1))
            enc_out, _ = net.encoder(enc_out)         # (B*n_vars, patch_num, D)
            enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
            tokens = enc_out.mean(dim=2)              # (B, 12, D) — one per lead
        elif self.model_name == 'itransformer':
            enc_out = net.enc_embedding(x, None)
            enc_out, _ = net.encoder(enc_out, attn_mask=None)
            tokens = enc_out[:, :12]                  # (B, 12, D) — one per variate
        else:  # medformer
            enc_out = net.enc_embedding(x)
            enc_out, _ = net.encoder(enc_out, attn_mask=None)
            tokens = enc_out                          # (B, n_granularities, D)
        if tokens.shape[1] != self.n_tokens:
            tokens = F.adaptive_avg_pool1d(
                tokens.transpose(1, 2), self.n_tokens).transpose(1, 2)
        return tokens


class MomentECGEncoder(nn.Module):
    """MOMENT-1 (Goswami et al., ICML 2024) — a *pretrained* general-purpose
    time-series foundation model (T5 encoder over 8-sample patches, channels
    processed independently).  We take its patch embeddings with
    ``reduction='none'`` and mean-pool over patches, giving one token per lead.

    Design cell: pretrained (general TS, not ECG) · fixed 8-sample patches ·
    strictly channel-independent — it never models inter-lead structure.
    """

    def __init__(self, latent_dim: int = 256, seq_len: int = 600,
                 model_id: str = "AutonLab/MOMENT-1-large",
                 freeze_backbone: bool = False):
        super().__init__()
        if _MOMENT_DIR not in sys.path:
            sys.path.insert(0, _MOMENT_DIR)
        from momentfm import MOMENTPipeline           # type: ignore
        self.moment = MOMENTPipeline.from_pretrained(
            model_id, model_kwargs={"task_name": "embedding", "seq_len": seq_len})
        self.moment.init()
        # Only the reconstruction head is pretrained and we never use any head.
        for attr in ('head',):
            if hasattr(self.moment, attr):
                delattr(self.moment, attr)
        if freeze_backbone:
            for p in self.moment.parameters():
                p.requires_grad_(False)
        d = self.moment.config.d_model
        self.proj = nn.Linear(d, latent_dim)
        self.n_tokens = 12

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600) → (B, 12, latent_dim)"""
        out = self.moment.embed(x_enc=ecg.float(), reduction="none")
        emb = out.embeddings                          # (B, 12, n_patches, d)
        return self.proj(emb.mean(dim=2).to(self.proj.weight.dtype))


class ChronosECGEncoder(nn.Module):
    """Chronos-T5 (Ansari et al., TMLR 2024) — a *pretrained* time-series
    foundation model that **quantizes amplitudes into a fixed token vocabulary**
    and runs a T5 language model over the resulting token sequence.  We take the
    encoder hidden states and mean-pool per lead.

    Design cell: pretrained (general TS) · discrete amplitude tokenization ·
    channel-independent (each lead is an independent "sentence").  The tokenizer
    is not differentiable, so the ECG reaches the model only through a fixed
    4096-bin quantization; the T5 encoder itself is fine-tuned.
    """

    def __init__(self, latent_dim: int = 256, seq_len: int = 600,
                 model_id: str = "amazon/chronos-t5-small",
                 n_leads: int = 12, freeze_backbone: bool = False):
        super().__init__()
        from chronos import BaseChronosPipeline    # type: ignore
        pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", dtype=torch.float32)
        self.tokenizer = pipe.tokenizer            # not an nn.Module (no params)
        self.model = pipe.model                    # registered -> trainable
        self.ctx_len = pipe.model.config.context_length
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.n_leads = n_leads
        self.proj = nn.Linear(pipe.model.model.config.d_model, latent_dim)
        self.n_tokens = n_leads

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600) → (B, 12, latent_dim)"""
        B = ecg.shape[0]
        x = F.interpolate(ecg.float(), size=self.ctx_len,
                          mode='linear', align_corners=False)      # (B, 12, ctx)
        ctx = x.reshape(B * self.n_leads, self.ctx_len)
        # Chronos' bin boundaries live on CPU and the quantization is
        # non-differentiable anyway, so tokenize on CPU and move the ids back.
        ids, mask, _ = self.tokenizer.context_input_transform(ctx.detach().cpu())
        h = self.model.encode(input_ids=ids.to(ecg.device),
                              attention_mask=mask.to(ecg.device))  # (B*12, S, d)
        h = h.mean(dim=1).reshape(B, self.n_leads, -1)
        return self.proj(h.to(self.proj.weight.dtype))


class HubertECGEncoder(nn.Module):
    """HuBERT-ECG (Coppola et al., 2024) — a *pretrained, ECG-specific*
    self-supervised foundation model (HuBERT over 8.9 M 12-lead ECGs).

    Design cell: pretrained · ECG-domain · conv feature extractor over the 12
    leads **concatenated into one 1-D stream**, so lead identity is only
    implicit in position.  Following the released preprocessing we resample each
    lead to 100 Hz before concatenation.
    """

    def __init__(self, latent_dim: int = 256, seq_len: int = 600,
                 fs_in: int = 500, fs_model: int = 100, n_leads: int = 12,
                 model_id: str = "Edoardo-BS/hubert-ecg-base",
                 freeze_backbone: bool = False):
        super().__init__()
        from transformers import AutoModel            # type: ignore
        self.hubert = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        if freeze_backbone:
            for p in self.hubert.parameters():
                p.requires_grad_(False)
        self.n_leads = n_leads
        self.len_model = int(round(seq_len * fs_model / fs_in))
        self.proj = nn.Linear(self.hubert.config.hidden_size, latent_dim)
        self.n_tokens = 12

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        """ecg: (B, 12, 600) → (B, 12, latent_dim)"""
        x = F.interpolate(ecg.float(), size=self.len_model,
                          mode='linear', align_corners=False)   # (B, 12, L)
        x = x.reshape(x.shape[0], -1)                           # (B, 12*L)
        h = self.hubert(x).last_hidden_state                    # (B, S, H)
        h = F.adaptive_avg_pool1d(h.transpose(1, 2), self.n_tokens).transpose(1, 2)
        return self.proj(h.to(self.proj.weight.dtype))


# ---------------------------------------------------------------------------
# ECG-conditioned mesh diffusion model
# ---------------------------------------------------------------------------

class ECGCondMDM(nn.Module):
    """
    Diffusion model for 4-D cardiac mesh trajectory conditioned on 12-lead ECG.

    model_config fields (OmegaConf-compatible):
        n_layers          : int   transformer depth            (default 8)
        latent_dim        : int   hidden / embedding dimension (default 256)
        n_ecg_tokens      : int   ECG condition token count    (default 8)
        pred_offset       : bool  predict offset from rest     (default True)
        transformer_block : str   block class name             (default SpatialTemporalTransformerBlock)
    """

    def __init__(self, n_points: int, n_frames: int, n_feats: int = 3,
                 model_config=None):
        super().__init__()

        self.n_points   = n_points
        self.n_feats    = n_feats
        self.latent_dim = model_config.latent_dim
        self.n_ecg_tokens = getattr(model_config, 'n_ecg_tokens', 8)
        self.pred_offset  = getattr(model_config, 'pred_offset', True)
        self.cond_frame   = 1  # always prepend rest-state mesh as frame 0

        # Point-cloud embedding (same as in MDM_ST)
        self.input_encoder = PointEmbed(dim=self.latent_dim)

        # ECG conditioning encoder ─ selectable backbone
        ecg_encoder_type = getattr(model_config, 'ecg_encoder_type', 'cnn')
        if ecg_encoder_type == 'cnn':
            self.ecg_encoder = ECGEncoder(
                latent_dim=self.latent_dim,
                n_tokens=self.n_ecg_tokens,
            )
        elif ecg_encoder_type == 'delay':
            n_leads = getattr(model_config, 'n_leads', 12)
            delay_hidden = getattr(model_config, 'delay_hidden', 128)
            cond_type = getattr(model_config, 'cond_type', 'concat')
            self.ecg_encoder = ECGDelayEncoder(
                latent_dim=self.latent_dim,
                n_tokens=self.n_ecg_tokens,
                n_leads=n_leads,
                delay_hidden=delay_hidden,
                cond_type=cond_type,
            )
            # Total conditioning tokens = ECG tokens + per-lead delay tokens
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'tech':
            tech_cfg = getattr(model_config, 'tech_encoder', {}) or {}
            tech_kwargs = dict(tech_cfg) if not isinstance(tech_cfg, dict) else tech_cfg
            # OmegaConf DictConfig → plain dict
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(tech_cfg, DictConfig):
                    tech_kwargs = OmegaConf.to_container(tech_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = TeChECGEncoder(
                latent_dim=self.latent_dim,
                **tech_kwargs,
            )
            # TeCh emits a single (channel+temporal) token
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'shift_invariant':
            self.ecg_encoder = ECGShiftInvariantEncoder(
                latent_dim    = self.latent_dim,
                n_wave_tokens = self.n_ecg_tokens,
                n_leads       = getattr(model_config, 'n_leads',       12),
                n_delay_freqs = getattr(model_config, 'n_delay_freqs', 16),
                delay_hidden  = getattr(model_config, 'delay_hidden',  128),
                canonicalize  = getattr(model_config, 'canonicalize',  False),
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'localized_delay':
            loc_cfg = getattr(model_config, 'localized_delay_encoder', {}) or {}
            cond_type = getattr(model_config, 'cond_type', 'concat')
            ablation = getattr(model_config, 'ablation', None)
            loc_kwargs = dict(loc_cfg) if not isinstance(loc_cfg, dict) else loc_cfg
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(loc_cfg, DictConfig):
                    loc_kwargs = OmegaConf.to_container(loc_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = ECGLocalizedDelayEncoder(
                latent_dim=self.latent_dim,
                cond_type=cond_type,
                ablation=ablation,
                **loc_kwargs,
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'physiowave':
            pw_cfg = getattr(model_config, 'physiowave_encoder', {}) or {}
            pw_kwargs = dict(pw_cfg) if not isinstance(pw_cfg, dict) else pw_cfg
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(pw_cfg, DictConfig):
                    pw_kwargs = OmegaConf.to_container(pw_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = PhysioWaveECGEncoder(
                latent_dim=self.latent_dim,
                n_tokens=self.n_ecg_tokens,
                **pw_kwargs,
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'delaywave':
            dw_cfg = getattr(model_config, 'delaywave_encoder', {}) or {}
            dw_kwargs = dict(dw_cfg) if not isinstance(dw_cfg, dict) else dw_cfg
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(dw_cfg, DictConfig):
                    dw_kwargs = OmegaConf.to_container(dw_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = DelayWaveECGEncoder(
                latent_dim=self.latent_dim,
                **dw_kwargs,
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'delaywavev2':
            dw_cfg = getattr(model_config, 'delaywavev2_encoder', {}) or {}
            dw_kwargs = dict(dw_cfg) if not isinstance(dw_cfg, dict) else dw_cfg
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(dw_cfg, DictConfig):
                    dw_kwargs = OmegaConf.to_container(dw_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = DelayWaveECGEncoderV2(
                latent_dim=self.latent_dim,
                **dw_kwargs,
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'delaywavev3':
            dw_cfg = getattr(model_config, 'delaywavev3_encoder', {}) or {}
            dw_kwargs = dict(dw_cfg) if not isinstance(dw_cfg, dict) else dw_cfg
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(dw_cfg, DictConfig):
                    dw_kwargs = OmegaConf.to_container(dw_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = DelayWaveECGEncoderV3(
                latent_dim=self.latent_dim,
                **dw_kwargs,
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'delaywavev4':
            dw_cfg = getattr(model_config, 'delaywavev4_encoder', {}) or {}
            dw_kwargs = dict(dw_cfg) if not isinstance(dw_cfg, dict) else dw_cfg
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(dw_cfg, DictConfig):
                    dw_kwargs = OmegaConf.to_container(dw_cfg, resolve=True)
            except ImportError:
                pass
            self.ecg_encoder = DelayWaveECGEncoderV4(
                latent_dim=self.latent_dim,
                **dw_kwargs,
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type in ('patchtst', 'itransformer', 'medformer'):
            # Additional temporal transformers from the Medformer MedTS
            # benchmark (the very set TeCh is benchmarked against).
            self.ecg_encoder = MedformerFamilyECGEncoder(
                ecg_encoder_type,
                latent_dim=self.latent_dim,
                **_cfg_kwargs(model_config, f'{ecg_encoder_type}_encoder'),
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'moment':
            self.ecg_encoder = MomentECGEncoder(
                latent_dim=self.latent_dim,
                **_cfg_kwargs(model_config, 'moment_encoder'),
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'chronos':
            self.ecg_encoder = ChronosECGEncoder(
                latent_dim=self.latent_dim,
                **_cfg_kwargs(model_config, 'chronos_encoder'),
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        elif ecg_encoder_type == 'hubert_ecg':
            self.ecg_encoder = HubertECGEncoder(
                latent_dim=self.latent_dim,
                **_cfg_kwargs(model_config, 'hubert_ecg_encoder'),
            )
            self.n_ecg_tokens = self.ecg_encoder.n_tokens
        else:
            raise ValueError(f"Unknown ecg_encoder_type: {ecg_encoder_type}")

        transformer_block = getattr(
            model_config, 'transformer_block', 'SpatialTemporalTransformerBlock'
        )

        # Spatio-temporal transformer backbone
        # sample_frames includes the conditioning rest-state frame
        self.dit = SpaitalTemporalTransformer(
            sample_points=n_points,
            sample_frames=n_frames + self.cond_frame,
            in_channels=n_feats,
            num_layers=model_config.n_layers,
            num_attention_heads=self.latent_dim // 64,
            time_embed_dim=self.latent_dim,
            cond_seq_length=self.n_ecg_tokens,
            cond_seq_length_t=self.cond_frame,
            transformer_block=transformer_block,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        x:          torch.Tensor,          # (B, T, N, 3)  noisy trajectory
        timesteps:  torch.Tensor,          # (B,)
        init_pc:    torch.Tensor,          # (B, 1, N, 3) or (B, N, 3)
        ecg:        torch.Tensor,          # (B, 12, 600)
        null_emb:   Optional[torch.Tensor] = None,  # (B, 1, 1) CFG mask
        delay_mat:  Optional[torch.Tensor] = None,  # (B, 12, 12) for delay encoder
        return_aux: bool = False,
    ):
        """Returns denoised trajectory (B, T, N, 3).
        If ``return_aux`` is True, returns (trajectory, aux_dict). Only the
        ECGLocalizedDelayEncoder populates aux; other encoders return {}.
        """

        B, T, N, C = x.shape
        init_pc = init_pc.reshape(B, N, C)

        # ── ECG conditioning tokens ──────────────────────────────────────
        aux: dict = {}
        if isinstance(self.ecg_encoder, ECGDelayEncoder):
            encoder_hidden_states = self.ecg_encoder(ecg, delay_mat)
        elif isinstance(self.ecg_encoder, ECGLocalizedDelayEncoder):
            if return_aux:
                encoder_hidden_states, aux = self.ecg_encoder(ecg, return_aux=True)
            else:
                encoder_hidden_states = self.ecg_encoder(ecg)
        elif isinstance(self.ecg_encoder, DelayWaveECGEncoder):
            if return_aux:
                encoder_hidden_states, aux = self.ecg_encoder(ecg, return_aux=True)
            else:
                encoder_hidden_states = self.ecg_encoder(ecg)
        elif isinstance(self.ecg_encoder, DelayWaveECGEncoderV2):
            if return_aux:
                encoder_hidden_states, aux = self.ecg_encoder(ecg, return_aux=True)
            else:
                encoder_hidden_states = self.ecg_encoder(ecg)
        elif isinstance(self.ecg_encoder, DelayWaveECGEncoderV3):
            if return_aux:
                encoder_hidden_states, aux = self.ecg_encoder(ecg, return_aux=True)
            else:
                encoder_hidden_states = self.ecg_encoder(ecg)
        elif isinstance(self.ecg_encoder, DelayWaveECGEncoderV4):
            if return_aux:
                encoder_hidden_states, aux = self.ecg_encoder(ecg, return_aux=True)
            else:
                encoder_hidden_states = self.ecg_encoder(ecg)
        else:
            encoder_hidden_states = self.ecg_encoder(ecg)  # (B, n_ecg_tokens, D)

        # Classifier-free guidance: zero out conditioning for null batch
        if null_emb is not None:
            encoder_hidden_states = encoder_hidden_states * null_emb

        # ── Prepend rest-state frame ─────────────────────────────────────
        x_full = torch.cat([init_pc.unsqueeze(1), x], dim=1)  # (B, T+1, N, 3)

        # ── Encode point positions ───────────────────────────────────────
        hidden_states = self.input_encoder(
            x_full.reshape(-1, N, C)
        ).reshape(B, T + 1, N, self.latent_dim)

        # ── Transformer forward ──────────────────────────────────────────
        output = self.dit(
            hidden_states, encoder_hidden_states, timesteps
        ).reshape(B, T + 1, N, 3)

        # Remove conditioning frame slot (index 0)
        output = output[:, 1:]  # (B, T, N, 3)

        # Predict offset from rest state (improves convergence)
        if self.pred_offset:
            output = output + init_pc.unsqueeze(1)

        if return_aux:
            return output, aux
        return output

    def get_ecg_delay_map(self, ecg: torch.Tensor) -> Optional[dict]:
        """Run the DelayWave delay descriptor on ``ecg`` (same as conditioning path).

        ``ecg`` : (B, n_leads, T) on any device; returned tensors are CPU float32.

        Returns ``None`` if the encoder is not a DelayWave variant.
        """
        E = self.ecg_encoder
        if isinstance(
            E,
            (
                DelayWaveECGEncoder,
                DelayWaveECGEncoderV2,
                DelayWaveECGEncoderV3,
                DelayWaveECGEncoderV4,
            ),
        ):
            return E.get_delay_map(ecg)
        return None
