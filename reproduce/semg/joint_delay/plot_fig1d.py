"""Fig. 1d: mean |inter-joint delay| per gesture stage, ground truth (upper triangle) vs prediction (lower).

Per stage, the display statistic is the mean over sessions of |delay| with NaN set to -1 and cast to
int (the notebook statistic behind the figure). Each panel has its own colour range, from the minimum to
the maximum of the two stage means, centred on its midpoint.
"""
import argparse

import numpy as np

# the three stages of the paper's figure and their panel titles
FIG1D_PANELS = {'CountingUpDownFingerWigglingSpreading': 'Count',
                'HandDeskSeparateClaspedChest': 'Chest Stretch',
                'FingerPinchesSingleFingerPinchesMultiple': 'Pinch'}


def stage_mean(mats):
    x = np.abs(mats)
    x[np.isnan(x)] = -1
    return x.astype(int).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--matrices', default='joint_delay_matrices.npz', help='output of joint_delay.py')
    ap.add_argument('--out', default='fig1d_hand_pose_delay.pdf')
    ap.add_argument('--stages', nargs='+', default=list(FIG1D_PANELS), help='emg2pose stage names, or "all"')
    ap.add_argument('--labels', nargs='+',
                    help='panel titles, one per stage (default: the figure titles for its three stages, '
                         'otherwise stage name and n)')
    ap.add_argument('--cmap', default='coolwarm')
    ap.add_argument('--font', default='Arial')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    z = np.load(args.matrices)
    stage = z['stage']
    stages = list(np.unique(stage)) if args.stages == ['all'] else args.stages
    labels = args.labels or [FIG1D_PANELS.get(s) for s in stages]
    if len(labels) != len(stages):
        raise SystemExit('--labels needs one title per stage')

    plt.rcParams['font.family'] = args.font
    fig, axes = plt.subplots(1, len(stages), figsize=(2.2 * len(stages), 2.6), squeeze=False)
    keep_upper = np.tril(np.ones((20, 20), dtype=bool), k=0)   # seaborn mask: hide lower triangle and diagonal
    keep_lower = np.triu(np.ones((20, 20), dtype=bool), k=0)   # seaborn mask: hide upper triangle and diagonal
    for ax, s, label in zip(axes[0], stages, labels):
        sel = stage == s
        if not sel.any():
            raise SystemExit(f'stage {s} not in {args.matrices}')
        gt, pred = stage_mean(z['gt'][sel]), stage_mean(z['pred'][sel])
        vmax = max(gt.max(), pred.max())
        vmin = min(gt.min(), pred.min())
        kw = dict(cmap=args.cmap, center=(vmin + vmax) / 2, vmin=vmin, vmax=vmax, square=True,
                  xticklabels=False, yticklabels=False, ax=ax)
        sns.heatmap(pred, mask=keep_lower, cbar=False, **kw)
        sns.heatmap(gt, mask=keep_upper, cbar=True,
                    cbar_kws=dict(orientation='horizontal', label='Delay (ms)', pad=0.05), **kw)
        ax.text(0.65, 0.85, 'GT', transform=ax.transAxes, ha='center', va='center')
        ax.text(0.35, 0.15, 'Pred', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(label or f'{s} (n={sel.sum()})')
        print(f'{s}: n={sel.sum()} vmin={vmin:.2f} vmax={vmax:.2f}')
    axes[0, 0].set_ylabel('Joint angle')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
