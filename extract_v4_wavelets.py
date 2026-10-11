"""Extract learned wavelet templates from a trained DelayWaveECGEncoderV4 checkpoint
and save each as an SVG.

For each (wavelet basis, low/high, lead-channel) we save one SVG showing the 24-tap
filter coefficients. We also save one overlay SVG per (wavelet, low/high) showing all
12 channels together for quick comparison against the initialization.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open

CKPT = '../nips26-physctrl-outputs/ecg_heart_delaywavev4_8L_256d_49frames/best_model/model.safetensors'
OUT  = '../physctrl/v4_wavelet_templates'
WAVELET_NAMES = ['db4', 'db6', 'sym4', 'coif2']  # from config
KINDS = ['low', 'high']

os.makedirs(OUT, exist_ok=True)

f = safe_open(CKPT, 'pt')
prefix = 'ecg_encoder.backbone.wavelet_decomp.selector.wavelet_filters'

for w_idx, wname in enumerate(WAVELET_NAMES):
    for kind in KINDS:
        key = f'{prefix}.{w_idx}.{kind}_filter.weight'
        w = f.get_tensor(key).float().squeeze(1).cpu().numpy()  # (C=12, K=24)
        C, K = w.shape

        fig, ax = plt.subplots(figsize=(6, 3))
        for c in range(C):
            ax.plot(range(K), w[c], linewidth=1.0, alpha=0.7, label=f'lead {c}')
        ax.set_title(f'{wname} — {kind} filter (all 12 leads)')
        ax.set_xlabel('tap')
        ax.set_ylabel('coef')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=3, loc='best')
        fig.tight_layout()
        path = os.path.join(OUT, f'{wname}_{kind}_all_leads.svg')
        fig.savefig(path, format='svg')
        plt.close(fig)

        for c in range(C):
            fig, ax = plt.subplots(figsize=(4, 2.2))
            ax.plot(range(K), w[c], color='C0', linewidth=1.4)
            ax.set_title(f'{wname} {kind} — lead {c}')
            ax.set_xlabel('tap')
            ax.set_ylabel('coef')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(OUT, f'{wname}_{kind}_lead{c:02d}.svg')
            fig.savefig(path, format='svg')
            plt.close(fig)

n = len(WAVELET_NAMES) * len(KINDS) * (1 + 12)
print(f'wrote {n} svg files to {OUT}')
