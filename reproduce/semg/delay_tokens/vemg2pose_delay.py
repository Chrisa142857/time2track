"""vemg2pose with delay tokens injected by FiLM (Appendix A.8, orange in Fig. 7).

vemg2pose (official emg2pose tracking checkpoint) decodes hand pose with a TDS convolutional
featurizer followed by an LSTM. The delay branch (delay_branch.py) reads K = 2 causal chunks of 1,024
samples, tau_max = 200 samples (+-100 ms), and modulates the 64 featurizer channels,
h'_t = h_t * (1 + g_t) + b_t, with (g_t, b_t) linearly interpolated over the chunks. The FiLM head is
a zero convolution, so the `treat` arm equals the released vemg2pose at initialization.

  val_index  fixed list of random validation windows (seed 1234), monitored every --val-every steps.
  train      fine-tune `ctrl` (no branch) or `treat` (+ delay branch) from the released checkpoint:
             Adam, lr 1e-4, batch 64 windows of 3,838 samples (1,790 left context + 2,048 output),
             electrode roll from the index, gradient clipping 1.0, 6,000 steps, final weights saved.
  dump       predictions on the cycle-anchored test windows: 49 frames over the 2,048 samples after
             the anchor, {sid}_pred.npy / {sid}_gt.npy of shape (49, 20, 1) in radians.

The validation pass draws one number from the global torch RNG when its DataLoader starts, so the
validation cadence is part of the training recipe: keep --val-every 1000 to reproduce the paper runs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delay_branch  # noqa: E402
from delay_branch import EnvelopeDelayBranch  # noqa: E402

LC = 1790            # TdsNetwork left_context, asserted at build time
W, SEQ = 2048, 50    # scored window after the anchor, and its 50 sub-sampled frames (frame 0 = init pose)
DATA = None          # per-session .npy directory, set in setup() before any DataLoader forks


def setup(a):
    """Locations from the CLI; emg2pose (facebookresearch, @5f6f62b) is imported only here."""
    global DATA, instantiate, get_ik_failures_mask, AngleMAE, LandmarkDistances, NUM_JOINTS
    DATA = a.data
    delay_branch.PHYSIOWAVE_DIR = a.physiowave or delay_branch.PHYSIOWAVE_DIR
    if a.cmd == 'val_index':
        return
    if not (a.emg2pose_repo and a.ckpt):
        raise SystemExit('--emg2pose-repo and --ckpt (or EMG2POSE_REPO / VEMG2POSE_CKPT) required')
    sys.path.insert(0, a.emg2pose_repo)
    from hydra.utils import instantiate                              # noqa: F811
    from emg2pose.utils import get_ik_failures_mask                  # noqa: F811
    from emg2pose.metrics import AngleMAE, LandmarkDistances         # noqa: F811
    from emg2pose.constants import NUM_JOINTS                        # noqa: F811


# ------------------------------------------------------------------- model ---
class DelayTdsNetwork(nn.Module):
    """Wraps the host TDS featurizer and applies the delay FiLM to its feature stream. The pose
    module, LSTM decoder, rollout, loss and the host state dict are untouched."""

    def __init__(self, tds, delay):
        super().__init__()
        self.tds, self.delay = tds, delay
        self.left_context = tds.left_context
        self.right_context = tds.right_context

    def forward(self, x):
        feats = self.tds(x)                                         # (B, 64, T_out)
        g, b, ent = self.delay(x, self.left_context, feats.shape[-1])
        return feats * (1.0 + g.to(feats.dtype)) + b.to(feats.dtype)


def build_model(arm, ckpt, chunk, beta_init, sd_path=None):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    hp = ck['hyper_parameters']
    model = instantiate(hp['network_conf'], _convert_='all')
    sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(sd, strict=True)
    assert int(model.left_context) == LC and int(model.right_context) == 0
    if arm == 'treat':
        model.network = DelayTdsNetwork(model.network,
                                        EnvelopeDelayBranch(chunk=chunk, beta_init=beta_init))
    if sd_path:
        model.load_state_dict(torch.load(sd_path, map_location='cpu'), strict=True)
    return model, bool(hp.get('provide_initial_pos', False))


# -------------------------------------------------------------------- data ---
def val_index(a):
    """Random-offset windows of one split, shared by every arm."""
    meta = json.load(open(a.session_meta))
    stems = sorted(k for k, v in meta.items() if v['split'] == a.split)
    rng = np.random.default_rng(a.seed)
    lens, items = {}, []
    pick = rng.integers(0, len(stems), a.n_windows)
    for i in range(a.n_windows):
        s = stems[pick[i]]
        if s not in lens:
            lens[s] = int(np.load(f'{DATA}/{s}.emg.npy', mmap_mode='r').shape[0])
        hi = max(1, lens[s] - a.window)
        items.append([s, int(rng.integers(0, hi)), 0])
    json.dump({'window': a.window, 'seed': a.seed, 'split': a.split, 'rotaug': False, 'items': items},
              open(a.out, 'w'))
    print(f'wrote {a.out}: {len(items)} windows from {len(set(i[0] for i in items))} sessions')


class StreamDS(torch.utils.data.Dataset):
    """Fixed list of [stem, offset, roll] windows. If a window's initial pose is an IK failure the
    offset is walked forward deterministically, a pure function of the item, so every arm sees the
    identical windows."""

    def __init__(self, index_path, lc=LC):
        d = json.load(open(index_path))
        self.items, self.window, self.lc = d['items'], d['window'], lc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, off, roll = self.items[i]
        e = np.load(f'{DATA}/{stem}.emg.npy', mmap_mode='r')
        j = np.load(f'{DATA}/{stem}.joints.npy', mmap_mode='r')
        T = min(e.shape[0], j.shape[0])
        W_ = self.window
        off = min(off, max(0, T - W_))
        for _ in range(8):                       # walk to a valid initial pose
            jj = np.asarray(j[off:off + W_], np.float32)
            if jj.shape[0] < W_:
                jj = np.concatenate([jj, np.repeat(jj[-1:], W_ - jj.shape[0], 0)], 0)
            if get_ik_failures_mask(jj)[self.lc]:
                break
            off = (off + W_ // 4) % max(1, T - W_)
        ee = np.asarray(e[off:off + W_], np.float32)
        if ee.shape[0] < W_:
            ee = np.concatenate([ee, np.zeros((W_ - ee.shape[0], ee.shape[1]), np.float32)], 0)
        ee = ee[:, :16]
        if roll:
            ee = np.roll(ee, roll, axis=-1)
        nof = get_ik_failures_mask(jj)
        return (torch.from_numpy(ee.T.copy()), torch.from_numpy(jj[:, :20].T.copy()),
                torch.from_numpy(nof.copy()))


def anchored_window(stem, lc, anchor):
    """(emg [C, lc+W], joints [J, lc+W], gt49 [49, 20, 1]) with the output span starting at the
    gesture-cycle anchor."""
    e = np.load(os.path.join(DATA, f'{stem}.emg.npy'), mmap_mode='r')
    j = np.load(os.path.join(DATA, f'{stem}.joints.npy'), mmap_mode='r')
    T = min(e.shape[0], j.shape[0])
    start = int(anchor)
    if start < lc or start + W > T:                # keep the window inside the session
        start = min(max(lc, start), max(lc, T - W))
    emg_lo = start - lc
    emg = np.asarray(e[emg_lo:start + W], np.float32)
    idxs = np.linspace(start, start + W - 1, SEQ).round().astype(np.int64)
    idxs = np.clip(idxs, 0, T - 1)
    jsub = np.asarray(j[idxs, :20], np.float32)
    jfull = np.asarray(j[emg_lo:start + W, :20], np.float32)
    gt49 = jsub[1:SEQ][:, :, None]
    return emg.T, jfull.T, gt49


# -------------------------------------------------------------------- loss ---
class Loss:
    """Official emg2pose objective: joint angle MAE + 0.01 * fingertip distance."""

    def __init__(self, device):
        self.mae, self.lmd = AngleMAE(), LandmarkDistances()
        self.lmd.hand_model.to(device)

    def __call__(self, pred, target, mask):
        m = {}
        m.update(self.mae(pred, target, mask, 'train'))
        m.update(self.lmd(pred, target, mask, 'train'))
        loss = m['train_mae'] + 0.01 * m['train_fingertip_distance']
        return loss, float(m['train_mae'].detach()), float(m['train_landmark_distance'].detach())


def masked(no_ik, lc):
    """Drop every frame of a window whose initial pose (frame lc) is an IK failure."""
    m = no_ik.clone()
    m[~m[:, lc]] = False
    return m


@torch.no_grad()
def val_mae(model, pip, val_index_path, batch, workers, dev):
    """Mean AngleMAE (radians) over the fixed validation windows."""
    was = model.training
    model.eval()
    dl = torch.utils.data.DataLoader(StreamDS(val_index_path), batch_size=batch, shuffle=False,
                                     num_workers=workers, pin_memory=True, drop_last=True)
    tot = n = 0.0
    for emg, jnt, nof in dl:
        emg, jnt, nof = emg.to(dev), jnt.to(dev), nof.to(dev)
        m = masked(nof, LC)
        if m.sum() == 0:
            continue
        pred, tgt, msk = model({'emg': emg, 'joint_angles': jnt, 'no_ik_failure': m}, pip)
        mk = msk[:, None, :].expand(-1, NUM_JOINTS, -1)
        tot += float(((pred - tgt).abs() * mk).sum())
        n += float(mk.sum())
    model.train(was)
    return tot / max(n, 1)


# ------------------------------------------------------------------- train ---
def train(a):
    dev = torch.device('cuda')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model, pip = build_model(a.arm, a.ckpt, a.chunk, a.beta_init)
    model = model.to(dev).train()
    n_new = sum(p.numel() for n, p in model.named_parameters() if 'delay' in n)
    print(f'arm={a.arm} seed={a.seed} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M '
          f'(delay branch {n_new/1e6:.2f}M) provide_initial_pos={pip}', flush=True)
    # alpha_delay is pinned at 1 and never receives a gradient (zero convolution), so it is left out
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if not n.endswith('alpha_delay')],
                           lr=a.lr)
    lossfn = Loss(dev)
    dl = torch.utils.data.DataLoader(StreamDS(a.index), batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, pin_memory=True,
                                     drop_last=True, persistent_workers=a.workers > 0)
    hist, t0, step = [], time.time(), 0
    for emg, jnt, nof in dl:
        emg, jnt, nof = (emg.to(dev, non_blocking=True), jnt.to(dev, non_blocking=True),
                         nof.to(dev, non_blocking=True))
        m = masked(nof, LC)
        if m.sum() == 0:
            continue
        pred, tgt, msk = model({'emg': emg, 'joint_angles': jnt, 'no_ik_failure': m}, pip)
        loss, mae, lm = lossfn(pred, tgt, msk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        step += 1
        rec = {'step': step, 'loss': float(loss), 'mae': mae, 'landmark': lm, 'gnorm': gn,
               't': time.time() - t0}
        if a.arm != 'ctrl':
            d = model.network.delay
            rec['beta'] = float(torch.nn.functional.softplus(d.beta_raw)[0] + 1e-3)
            rec['ent'], rec['lag_abs'] = d.last_ent, d.last_lag_abs
        hist.append(rec)
        if step % a.log_every == 0 or step == 1:
            w = hist[-a.log_every:]
            extra = '' if a.arm == 'ctrl' else \
                f'beta {rec["beta"]:.3f} ent {rec["ent"]:.3f} |lag| {rec["lag_abs"]:.3f} '
            print(f'  step {step:5d}  loss {np.mean([h["loss"] for h in w]):.4f}  '
                  f'mae {np.mean([h["mae"] for h in w]):.4f}  gn {gn:.2f}  {extra}{rec["t"]:.0f}s',
                  flush=True)
        if a.val_every and step % a.val_every == 0:
            hist[-1]['val_mae'] = v = val_mae(model, pip, a.val_index, a.batch, a.workers, dev)
            print(f'  [val] step {step}: val_mae {v:.5f}', flush=True)
        if step >= a.steps:
            break
    hist[-1]['val_mae_final'] = v = val_mae(model, pip, a.val_index, a.batch, a.workers, dev)
    print(f'  [val] FINAL step {step}: val_mae {v:.5f}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(model.state_dict(), a.out)
    json.dump(hist, open(a.out.replace('.pt', '_hist.json'), 'w'))
    print(f'DONE {a.arm} s{a.seed}: {step} steps in {time.time()-t0:.0f}s -> {a.out}', flush=True)


# -------------------------------------------------------------------- dump ---
@torch.no_grad()
def dump(a):
    dev = torch.device('cuda')
    model, pip = build_model(a.arm, a.ckpt, a.chunk, a.beta_init, sd_path=a.sd)
    model = model.to(dev).eval()
    lc = int(model.left_context)
    anch = json.load(open(a.anchors))
    all_sids = json.load(open(a.test_index))['sids']
    sids = [s for s in all_sids if s in anch]
    os.makedirs(a.outdir, exist_ok=True)
    print(f'{len(sids)} cycle-anchored test sessions ({len(all_sids) - len(sids)} without a usable '
          f'cycle, excluded)', flush=True)
    sub = np.linspace(0, W - 1, SEQ).round().astype(np.int64)
    t0, n_nan = time.time(), 0
    for b0 in range(0, len(sids), a.batch):
        ch = sids[b0:b0 + a.batch]
        E, J, G = zip(*(anchored_window(s, lc, anch[s]['anchor']) for s in ch))
        emg = torch.as_tensor(np.stack(E), device=dev)
        jnt = torch.as_tensor(np.stack(J), device=dev)
        nof = torch.as_tensor(np.stack([get_ik_failures_mask(x.T) for x in J]), device=dev)
        pred, ja, _ = model({'emg': emg, 'joint_angles': jnt, 'no_ik_failure': nof}, pip)
        p49 = pred[:, :, sub][:, :, 1:SEQ].permute(0, 2, 1).unsqueeze(-1).float().cpu().numpy()
        for k, sid in enumerate(ch):
            n_nan += int(not np.isfinite(p49[k]).all())
            np.save(os.path.join(a.outdir, f'{sid}_pred.npy'), p49[k])
            np.save(os.path.join(a.outdir, f'{sid}_gt.npy'), G[k])
        if b0 % (a.batch * 40) == 0:
            print(f'  {b0 + len(ch)}/{len(sids)} {time.time()-t0:.0f}s nan={n_nan}', flush=True)
    print(f'DONE dump {a.outdir}: {len(sids)} sessions {time.time()-t0:.0f}s NaN {n_nan}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['val_index', 'train', 'dump'])
    ap.add_argument('--emg2pose-repo', default=os.environ.get('EMG2POSE_REPO'),
                    help='clone of facebookresearch/emg2pose @5f6f62b (with UmeTrack)')
    ap.add_argument('--ckpt', default=os.environ.get('VEMG2POSE_CKPT'),
                    help='released tracking_vemg2pose.ckpt')
    ap.add_argument('--physiowave', default=os.environ.get('PHYSIOWAVE_DIR'), help='PhysioWave clone @6f7bec7')
    ap.add_argument('--data', default=os.environ.get('EMG2POSE_DATA'), help='per-session .npy directory')
    ap.add_argument('--arm', default='ctrl', choices=['ctrl', 'treat'])
    ap.add_argument('--chunk', type=int, default=1024, help='delay-branch chunk in samples')
    ap.add_argument('--beta-init', type=float, default=50.0)
    ap.add_argument('--batch', type=int, default=64)
    # val_index
    ap.add_argument('--session-meta', default='work/session_meta.json')
    ap.add_argument('--n-windows', type=int, default=2048)
    ap.add_argument('--window', type=int, default=3838)
    ap.add_argument('--split', default='val')
    # train
    ap.add_argument('--index', default='work/vemg2pose_train_s42.json')
    ap.add_argument('--val-index', default='work/vemg2pose_val_windows.json')
    ap.add_argument('--out', default=None, help='train: checkpoint .pt, val_index: index .json')
    ap.add_argument('--seed', type=int, default=None, help='train default 42, val_index default 1234')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--val-every', type=int, default=1000)
    # dump
    ap.add_argument('--sd', default=None, help='fine-tuned state dict to dump')
    ap.add_argument('--test-index', default='work/test_index.json')
    ap.add_argument('--anchors', default='work/first_cycle_anchors.json')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()
    if a.seed is None:
        a.seed = 1234 if a.cmd == 'val_index' else 42
    if not a.data:
        ap.error('--data or EMG2POSE_DATA required')
    if a.cmd in ('val_index', 'train') and not a.out:
        ap.error('--out required')
    if a.cmd == 'dump' and not (a.sd and a.outdir):
        ap.error('--sd and --outdir required')
    setup(a)
    {'val_index': val_index, 'train': train, 'dump': dump}[a.cmd](a)
