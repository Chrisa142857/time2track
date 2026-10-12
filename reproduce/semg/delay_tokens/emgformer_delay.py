"""EMGFormer with delay tokens injected by cross-attention (Appendix A.8, blue in Fig. 7).

EMGFormer (EgoEMG release, emg2pose_emgformer_small.ckpt) decodes hand pose with a TDS featurizer
(tds_slim, left context 510) followed by a Transformer encoder (3 layers, width 256, 4 heads, RoPE).
The delay branch (delay_branch.py, non-causal like the host) reads K = 3 chunks of 2,048 samples
spread evenly over the 7,280 output samples of the 7,790-sample window, tau_max = 800 samples
(+-400 ms), and emits 3 x 16 = 48 per-channel delay tokens U. After the last encoder layer every frame
token queries them, z'_t = z_t + W_o MHA(z_t, LN(U), LN(U)), with W_o zero-initialized (the zero
convolution), so the `xattn` arm equals the released EMGFormer at initialization.

  pack   materialize the cycle-anchored 7,790-sample window of every session once (emg, joints,
         IK mask), so every arm reads identical bytes.
  train  fine-tune `ctrl` (no branch) or `xattn` from the released checkpoint with the official
         recipe: AdamW lr 1e-4, weight decay 0.01, cosine decay to 5e-5 over 60 epochs (1,560 steps),
         batch 650, bf16 autocast, gradient clipping 1.0, per-dataset input normalization, the
         official train augmentations with the electrode roll taken from the index. Validation MAE
         after every epoch, the weights with the lowest validation MAE are saved (final weights kept
         as *_final.pt).
  dump   predictions on the cycle-anchored test windows in the scored geometry: 49 frames over the
         2,048 samples after the anchor, {sid}_pred.npy / {sid}_gt.npy of shape (49, 20, 1).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delay_branch  # noqa: E402
from delay_branch import EnvelopeDelayBranch  # noqa: E402

WINDOW = 7790          # official EMGFormer window
LC = 510               # tds_slim left context (asserted at build time)
ANCHOR_LC = 1790       # anchors are stored as offset = anchor - 1790
SCORE_W, SEQ = 2048, 50
TAU_MAX, CHUNK, CUTOFF_HZ, BETA_INIT = 800, 2048, 15.0, 50.0

# set in setup() from the CLI, before any DataLoader forks
EGO = DATA = PACK = CKPT = None
EMG_MEAN = EMG_STD = None
NORM_EPS = 1e-6


def setup(a):
    """Locations from the CLI; EgoEMG's own `emg2pose` package is imported only here (it is a
    different package from facebookresearch/emg2pose, so never import both in one process)."""
    global EGO, DATA, PACK, CKPT, EMG_MEAN, EMG_STD
    global compose, initialize_config_dir, instantiate, get_ik_failures_mask, AngleMAE, LandmarkDistances
    if not a.egoemg:
        raise SystemExit('--egoemg or EGOEMG_DIR required (clone of zhenqis123/EgoEMG @9901726)')
    EGO, DATA, PACK = a.egoemg, a.data, a.pack
    CKPT = a.ckpt or os.path.join(EGO, 'checkpoints', 'emg2pose_emgformer_small.ckpt')
    delay_branch.PHYSIOWAVE_DIR = a.physiowave or delay_branch.PHYSIOWAVE_DIR
    sys.path.insert(0, EGO)
    from emg2pose.utils import get_ik_failures_mask                  # noqa: F811
    if a.cmd == 'pack':
        return
    from hydra import compose, initialize_config_dir                 # noqa: F811
    from hydra.utils import instantiate                              # noqa: F811
    from emg2pose.metrics import AngleMAE, LandmarkDistances         # noqa: F811
    # The released checkpoint was trained with per-dataset normalization (its hyper_parameters say
    # norm_mode 'per-dataset'), for emg2pose mean 0 / std 18.49, applied before the augmentations.
    ns = json.load(open(os.path.join(EGO, 'assets', 'per_dataset_norm_stats.json')))['emg2pose']
    EMG_MEAN, EMG_STD = ns['mean'], ns['std']


# ------------------------------------------------------------------ branch ---
class DelayTokenBranch(EnvelopeDelayBranch):
    """EnvelopeDelayBranch with the output stage replaced: K x C per-channel tokens of width
    model_dim instead of a FiLM (gain, shift) pair. Envelope, cross-correlation, soft-argmax and beta
    are unchanged. The token head is not zeroed here, the zero convolution of the injection is
    TokenFormer.xattn_out (two zeros in series would block the gradient)."""

    def __init__(self, model_dim=256, latent_dim=256, **kw):
        super().__init__(latent_dim=latent_dim, **kw)
        self.out = nn.Linear(latent_dim, model_dim)   # token head replaces the FiLM head
        self.last_K = 0

    def chunk_starts(self, lc, t_out):
        """K chunks of exactly `chunk` samples spread across [lc, lc + t_out)."""
        K = max(1, t_out // self.chunk)
        step = 0 if K == 1 else (t_out - self.chunk) // (K - 1)
        return K, [lc + k * step for k in range(K)]

    def lags(self, x, lc, t_out):
        """Non-causal, matching the host decoder (official small config: decoder.causal false)."""
        B, C, T = x.shape
        K, starts = self.chunk_starts(lc, t_out)
        assert starts[-1] + self.chunk <= T, (starts, self.chunk, T)
        xs = torch.stack([x[..., s:s + self.chunk] for s in starts], 1)
        env = self._envelope(xs.reshape(B * K, C, self.chunk))
        d, ent = self._lag_descriptor(env)
        self.last_K = K
        return d.reshape(B, K, C, 1), ent

    def forward(self, x, lc, t_out):
        """(B, C, T) -> (B, K * C, model_dim) per-channel delay tokens."""
        delta, ent = self.lags(x, lc, t_out)                     # (B, K, C, 1)
        B, K = delta.shape[0], delta.shape[1]
        self.last_ent = float(ent.detach())
        self.last_lag_abs = float(delta.detach().abs().mean())
        u = self.delay_mlp(delta.reshape(B * K, self.n_channels, 1))
        u = u + self.channel_pos_embed
        u = self.delay_block(u)
        tok = self.out(u).reshape(B, K * self.n_channels, -1)    # no channel averaging
        return self.alpha_delay * tok, ent


class TokenFormer(nn.Module):
    """EMGFormer forward written out (the host calls its decoder on features only), with a
    cross-attention from every frame token to the delay tokens after the last encoder layer."""

    def __init__(self, host, branch=None, model_dim=256):
        super().__init__()
        self.host, self.branch = host, branch
        self.left_context, self.right_context = host.left_context, host.right_context
        if branch is not None:
            d = model_dim
            self.xattn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
            self.xattn_norm = nn.LayerNorm(d)
            self.xattn_out = nn.Linear(d, d)               # zero convolution: step 0 equals ctrl
            nn.init.zeros_(self.xattn_out.weight)
            nn.init.zeros_(self.xattn_out.bias)

    def forward(self, emg):
        h, dec = self.host, self.host.decoder
        feats = h.featurizer(emg)                                  # (B, 256, T)
        tgt = dec.input_proj(feats.transpose(1, 2))                # (B, T, 256)
        if dec.pos_encoder is not None:
            tgt = dec.pos_encoder(tgt)
        # the branch runs before the encoder: in training both draw dropout masks from the global
        # RNG, so this order is part of the recipe
        tok = None
        if self.branch is not None:
            tok, _ = self.branch(emg, h.left_context, emg.shape[-1] - h.left_context)
        enc = dec.encoder(src=tgt, mask=None)
        if tok is not None:
            kv = self.xattn_norm(tok.to(enc.dtype))                 # (B, K*C, d)
            r, _ = self.xattn(query=enc, key=kv, value=kv, need_weights=False)
            enc = enc + self.xattn_out(r)
        return h.head(dec.output_proj(enc).transpose(1, 2))        # (B, 20, T)


def official_cfg():
    with initialize_config_dir(config_dir=os.path.join(EGO, 'config'), version_base=None):
        return compose(config_name='base', overrides=[f'data_location={DATA}'])


def build_model(arm, seed=0):
    cfg = official_cfg()
    host = instantiate(cfg.module, _convert_='all')
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
    host.load_state_dict(sd, strict=True)                    # pretrained_strict: true
    assert int(host.left_context) == LC and int(host.right_context) == 0
    branch = None
    if arm == 'xattn':
        torch.manual_seed(seed)
        branch = DelayTokenBranch(model_dim=cfg.module.decoder.model_dim, n_channels=16,
                                  tau_max=TAU_MAX, beta_init=BETA_INIT, cutoff_hz=CUTOFF_HZ, chunk=CHUNK)
        K, starts = branch.chunk_starts(LC, WINDOW - LC)
        print(f'  branch: tau_max +/-{branch.tau_max} samples, cutoff {branch.cutoff_hz:g} Hz, '
              f'chunk {branch.chunk} -> K={K} chunks at {starts}, {K * 16} tokens', flush=True)
    return TokenFormer(host, branch, model_dim=cfg.module.decoder.model_dim)


# -------------------------------------------------------------------- data ---
def pack_one(args):
    i, stem, off = args
    e = np.load(f'{DATA}/{stem}.emg.npy', mmap_mode='r')
    j = np.load(f'{DATA}/{stem}.joints.npy', mmap_mode='r')
    T = min(e.shape[0], j.shape[0])
    off = min(max(0, off), max(0, T - WINDOW))
    ee = np.asarray(e[off:off + WINDOW, :16], np.float32)
    jj = np.asarray(j[off:off + WINDOW, :20], np.float32)
    if ee.shape[0] < WINDOW:
        ee = np.pad(ee, ((0, WINDOW - ee.shape[0]), (0, 0)))
        jj = np.concatenate([jj, np.repeat(jj[-1:], WINDOW - jj.shape[0], 0)], 0)
    E = np.load(f'{PACK}/emg.npy', mmap_mode='r+')
    J = np.load(f'{PACK}/jnt.npy', mmap_mode='r+')
    M = np.load(f'{PACK}/nof.npy', mmap_mode='r+')
    E[i] = ee.T
    J[i] = jj.T
    M[i] = get_ik_failures_mask(jj)
    return i


def pack(a):
    from multiprocessing import Pool
    A = json.load(open(a.anchors))
    stems = sorted(A)
    os.makedirs(PACK, exist_ok=True)
    N = len(stems)
    json.dump({'stems': stems, 'n': N, 'window': WINDOW, 'row': {s: i for i, s in enumerate(stems)}},
              open(f'{PACK}/meta.json', 'w'))
    for name, shape, dt in (('emg', (N, 16, WINDOW), np.float32), ('jnt', (N, 20, WINDOW), np.float32),
                            ('nof', (N, WINDOW), np.bool_)):
        p = f'{PACK}/{name}.npy'
        if not os.path.exists(p):
            np.lib.format.open_memmap(p, mode='w+', dtype=dt, shape=shape).flush()
    done = 0
    with Pool(a.nproc) as pool:
        for _ in pool.imap_unordered(pack_one, [(i, s, A[s]['offset']) for i, s in enumerate(stems)],
                                     chunksize=8):
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{N}', flush=True)
    print(f'DONE pack: {N} windows -> {PACK}', flush=True)


def _make_transforms():
    """The official train augmentation stack minus RotationAugmentation (the electrode roll comes
    from the index, so every arm sees the identical roll) and the tensor conversion."""
    cfg = official_cfg()
    return [instantiate(t) for t in cfg.transforms.train
            if 'Rotation' not in t._target_ and 'Extract' not in t._target_
            and 'ToFloatTensor' not in t._target_]


class WinDS(torch.utils.data.Dataset):
    """Cycle-anchored windows served from the pack, one row per session."""

    def __init__(self, index_path, train=True):
        d = json.load(open(index_path))
        m = json.load(open(f'{PACK}/meta.json'))
        assert m['window'] == d.get('window', WINDOW) == WINDOW
        self.rows = np.array([m['row'][i[0]] for i in d['items']], np.int32)
        self.rolls = np.array([i[2] for i in d['items']], np.int8)
        self.tfm = _make_transforms() if train else []
        self.E = None

    def _open(self):                                   # per worker, after fork
        self.E = np.load(f'{PACK}/emg.npy', mmap_mode='r')
        self.J = np.load(f'{PACK}/jnt.npy', mmap_mode='r')
        self.M = np.load(f'{PACK}/nof.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.E is None:
            self._open()
        r, roll = int(self.rows[i]), int(self.rolls[i])
        ee = np.array(self.E[r], np.float32)                        # (C, T), writable
        jj = np.array(self.J[r], np.float32)                        # (20, T)
        nof = np.array(self.M[r])
        ee = (ee - EMG_MEAN) / (EMG_STD + NORM_EPS)                 # per-dataset norm
        if roll:
            ee = np.roll(ee, roll, axis=0)                          # electrode roll
        x = torch.from_numpy(np.ascontiguousarray(ee.T))            # (T, C)
        for t in self.tfm:
            x = t(x)
        return x.T.contiguous(), torch.from_numpy(jj), torch.from_numpy(nof)


# -------------------------------------------------------------------- loss ---
class Loss:
    """Official objective: joint angle MAE + 0.01 * fingertip distance."""

    def __init__(self, device):
        self.mae, self.lmd = AngleMAE(), LandmarkDistances()
        self.lmd.hand_model.to(device)

    def __call__(self, pred, tgt, mask):
        m = {}
        m.update(self.mae(pred, tgt, mask, 'train'))
        m.update(self.lmd(pred, tgt, mask, 'train'))
        return (m['train_mae'] + 0.01 * m['train_fingertip_distance'],
                float(m['train_mae'].detach()))


def align(model, pred, jnt, nof):
    """Official alignment: slice targets to the prediction span, upsample predictions to it."""
    h = model.host
    tgt = jnt[..., LC:]
    msk = nof[..., LC:]
    n = tgt.shape[-1]
    return h.align_predictions(pred, n), tgt, h.align_mask(msk, n)


@torch.no_grad()
def val_mae(model, dl, dev):
    """Masked MAE in degrees over the validation windows (official AngleMAE semantics)."""
    was = model.training
    model.eval()
    tot = n = 0.0
    for emg, jnt, nof in dl:
        emg, jnt, nof = (emg.to(dev, non_blocking=True), jnt.to(dev, non_blocking=True),
                         nof.to(dev, non_blocking=True))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, t, m = align(model, model(emg), jnt, nof)
        mk = m[:, None, :].expand_as(t)
        tot += float(((p.float() - t.float()).abs() * mk).sum())
        n += float(mk.sum())
    model.train(was)
    return tot / max(n, 1) * 180 / math.pi


# ------------------------------------------------------------------- train ---
def train(a):
    dev = torch.device('cuda')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_float32_matmul_precision('high')
    model = build_model(a.arm, seed=a.seed).to(dev).train()
    nb = sum(p.numel() for n, p in model.named_parameters() if n.startswith('branch'))
    print(f'arm={a.arm} seed={a.seed} params={sum(p.numel() for p in model.parameters())/1e6:.3f}M '
          f'(branch {nb/1e6:.3f}M)', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=5e-5)
    lossfn = Loss(dev)
    tds = WinDS(a.index)
    dl = torch.utils.data.DataLoader(tds, batch_size=a.micro, shuffle=False, num_workers=a.workers,
                                     pin_memory=False, drop_last=True, prefetch_factor=2,
                                     persistent_workers=a.workers > 0)
    vdl = torch.utils.data.DataLoader(WinDS(a.val_index, train=False), batch_size=a.micro,
                                      shuffle=False, num_workers=4, drop_last=False)
    # one epoch = one pass over the anchored train sessions (one window each)
    accum = max(1, a.batch // a.micro)
    n_micro, n_sess = len(dl), int(np.unique(tds.rows).size)
    steps_per_epoch = max(1, n_sess // (a.micro * accum))
    total_steps = a.epochs * steps_per_epoch
    assert total_steps * accum <= n_micro, f'index too short: need {total_steps*accum} micro-batches'
    print(f'batch {a.micro}x{accum}={a.micro*accum}, {n_sess} train sessions -> {steps_per_epoch} '
          f'steps/epoch x {a.epochs} epochs = {total_steps} steps, val {len(vdl.dataset)}', flush=True)

    hist, val, t0, step, k = [], [], time.time(), 0, 0
    best = {'val': float('inf'), 'ep': -1, 'step': -1, 'sd': None}
    opt.zero_grad(set_to_none=True)
    for emg, jnt, nof in dl:
        emg, jnt, nof = (emg.to(dev, non_blocking=True), jnt.to(dev, non_blocking=True),
                         nof.to(dev, non_blocking=True))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, t, m = align(model, model(emg), jnt, nof)
            loss, mae = lossfn(p.float(), t.float(), m)
        (loss / accum).backward()
        k += 1
        if k % accum:
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {'step': step, 'loss': float(loss), 'mae': mae, 'gnorm': gn,
               'lr': opt.param_groups[0]['lr'], 't': time.time() - t0}
        if model.branch is not None:
            rec['beta'] = float(F.softplus(model.branch.beta_raw[0]) + 1e-3)
            rec['ent'], rec['lag_abs'] = model.branch.last_ent, model.branch.last_lag_abs
        hist.append(rec)
        if step % a.log_every == 0 or step == 1:
            w = hist[-a.log_every:]
            print(f'  step {step:5d} ep {step // steps_per_epoch:3d} '
                  f'loss {np.mean([h["loss"] for h in w]):.4f} '
                  f'mae {np.mean([h["mae"] for h in w]) * 180 / math.pi:.3f}deg '
                  f'gn {gn:.2f} lr {rec["lr"]:.2e} {rec["t"]:.0f}s', flush=True)
        if step % steps_per_epoch == 0:
            sch.step()
            ep = step // steps_per_epoch
            if ep % a.val_every == 0 or ep == 1:
                v = val_mae(model, vdl, dev)
                val.append({'epoch': ep, 'step': step, 'val_mae_deg': v, 't': time.time() - t0})
                if v < best['val']:                     # model selection on validation MAE
                    best['val'], best['ep'], best['step'] = v, ep, step
                    best['sd'] = {k2: t2.detach().to('cpu', copy=True)
                                  for k2, t2 in model.state_dict().items()}
                print(f'  [val] ep {ep:3d} step {step:5d}  val_mae {v:.4f} deg', flush=True)
        if step >= total_steps:
            break
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(model.state_dict(), a.out.replace('.pt', '_final.pt'))
    torch.save(best['sd'] if best['sd'] is not None else model.state_dict(), a.out)
    print(f'  [select] best val_mae {best["val"]:.4f} deg @ ep {best["ep"]} step {best["step"]} -> {a.out}',
          flush=True)
    json.dump({'selected': {kk: best[kk] for kk in ('val', 'ep', 'step')}, 'hist': hist, 'val': val,
               'arm': a.arm, 'seed': a.seed, 'epochs': a.epochs, 'steps': step,
               'batch': a.micro * accum, 'steps_per_epoch': steps_per_epoch, 'n_train_sessions': n_sess},
              open(a.out.replace('.pt', '_hist.json'), 'w'))


# -------------------------------------------------------------------- dump ---
@torch.no_grad()
def dump(a):
    dev = torch.device('cuda')
    model = build_model(a.arm, seed=a.seed).to(dev).eval()
    model.load_state_dict(torch.load(a.sd, map_location='cpu'), strict=True)
    ix = json.load(open(a.index))
    ds = WinDS(a.index, train=False)
    os.makedirs(a.outdir, exist_ok=True)
    # prediction index 0 is absolute sample offset + LC, the anchor is offset + ANCHOR_LC
    sub = np.linspace(0, SCORE_W - 1, SEQ).round().astype(np.int64) + (ANCHOR_LC - LC)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.micro, shuffle=False, num_workers=a.workers,
                                     drop_last=False)
    sids = [it[0] for it in ix['items']]
    tot = n = 0.0
    t0, i0, n_nan = time.time(), 0, 0
    for emg, jnt, nof in dl:
        emg, jnt, nof = emg.to(dev), jnt.to(dev), nof.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(emg)
        p, t, m = align(model, pred, jnt, nof)
        p, t = p.float(), t.float()
        mk = m[:, None, :].expand_as(t)
        tot += float(((p - t).abs() * mk).sum())
        n += float(mk.sum())
        p49 = p[:, :, sub][:, :, 1:SEQ].permute(0, 2, 1).unsqueeze(-1).cpu().numpy()
        g49 = t[:, :, sub][:, :, 1:SEQ].permute(0, 2, 1).unsqueeze(-1).cpu().numpy()
        for kk in range(p49.shape[0]):
            s = sids[i0 + kk]
            n_nan += int(not np.isfinite(p49[kk]).all())
            np.save(os.path.join(a.outdir, f'{s}_pred.npy'), p49[kk])
            np.save(os.path.join(a.outdir, f'{s}_gt.npy'), g49[kk])
        i0 += p49.shape[0]
    print(f'DONE dump {a.outdir}: {i0} sessions NaN {n_nan} MAE over the full prediction span '
          f'{tot / max(n, 1) * 180 / math.pi:.3f} deg {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['pack', 'train', 'dump'])
    ap.add_argument('--egoemg', default=os.environ.get('EGOEMG_DIR'),
                    help='clone of zhenqis123/EgoEMG @9901726 with checkpoints/ and assets/')
    ap.add_argument('--ckpt', default=None, help='default: <egoemg>/checkpoints/emg2pose_emgformer_small.ckpt')
    ap.add_argument('--physiowave', default=os.environ.get('PHYSIOWAVE_DIR'), help='PhysioWave clone @6f7bec7')
    ap.add_argument('--data', default=os.environ.get('EMG2POSE_DATA'), help='per-session .npy directory')
    ap.add_argument('--pack', default='work/emgformer_pack', help='packed 7,790-sample windows')
    ap.add_argument('--arm', default='ctrl', choices=['ctrl', 'xattn'])
    ap.add_argument('--seed', type=int, default=42)
    # pack
    ap.add_argument('--anchors', default='work/first_cycle_anchors_w7790.json')
    ap.add_argument('--nproc', type=int, default=24)
    # train
    ap.add_argument('--index', default='work/emgformer_train_s42.json', help='train or test window index')
    ap.add_argument('--val-index', default='work/emgformer_val_windows.json')
    ap.add_argument('--out', default=None, help='checkpoint .pt (best validation MAE)')
    ap.add_argument('--epochs', type=int, default=60, help='cosine T_max, in epochs')
    ap.add_argument('--val-every', type=int, default=1)
    ap.add_argument('--batch', type=int, default=650, help='official effective batch')
    ap.add_argument('--micro', type=int, default=650, help='per-step micro batch')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=12,
                    help='official value; the augmentation RNG streams depend on it')
    ap.add_argument('--log-every', type=int, default=10)
    # dump
    ap.add_argument('--sd', default=None, help='fine-tuned state dict to dump')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()
    if not a.data:
        ap.error('--data or EMG2POSE_DATA required')
    if a.cmd == 'train' and not a.out:
        ap.error('--out required')
    if a.cmd == 'dump' and not (a.sd and a.outdir):
        ap.error('--sd and --outdir required')
    setup(a)
    {'pack': pack, 'train': train, 'dump': dump}[a.cmd](a)
