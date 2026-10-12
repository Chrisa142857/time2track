"""Fig. 7: change of the hand pose metrics from adding delay tokens to EMGFormer and vemg2pose.

Reads the two outputs of score_stages.py. Per metric (RMSE, |PAR-1|, |APCR-1|, |APDR-1|) and subset
(all stages and the four stages of the figure) it draws the seed-matched difference delay minus
control for each seed (dots) and their mean +- SD (marker and bar, SD with ddof 1). Lower is better
for every metric, so the shaded negative half is a gain. Row labels give the vemg2pose test session
counts. Writes <stem>.pdf, <stem>.png and <stem>.json (the plotted values).
"""
import argparse
import json
from pathlib import Path

import numpy as np

MET = ['RMSE', '|PAR−1|', '|APCR−1|', '|APDR−1|']
KEY = ['RMSE', 'absPARm1', 'absAPCRm1', 'absAPDRm1']
ALL = 'ALL'
LAB = {ALL: 'All stages', 'HandOverHandCountingUpDownFingerWigglingSpreading': 'Hand over hand',
       'ShakaVulcanPeace': 'Shaka Vulcan Peace', 'HookEmHornsOKScissors': 'HookEm OK Scissors',
       'FingerFreeform': 'Finger freeform'}
DDOF = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emgformer-json', default='work/emgformer_per_stage.json')
    ap.add_argument('--vemg2pose-json', default='work/vemg2pose_per_stage.json')
    ap.add_argument('--emgformer-arms', nargs=2, default=['ctrl', 'xattn'], metavar=('BASE', 'DELAY'))
    ap.add_argument('--vemg2pose-arms', nargs=2, default=['ctrl', 'treat'], metavar=('BASE', 'DELAY'))
    ap.add_argument('--stages', nargs='+', default=['HandOverHandCountingUpDownFingerWigglingSpreading',
                                                    'ShakaVulcanPeace', 'HookEmHornsOKScissors',
                                                    'FingerFreeform'])
    ap.add_argument('--seeds', nargs='+', default=['42', '43', '44'])
    ap.add_argument('--width-in', type=float, default=5.5, help='NeurIPS text width')
    ap.add_argument('--height-in', type=float, default=1.62)
    ap.add_argument('--pad-in', type=float, default=0.02, help='tight bounding box padding')
    ap.add_argument('--font', default='Arial')
    ap.add_argument('--font-family', default='sans-serif')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()

    src = {'EMGFormer': (json.load(open(a.emgformer_json)), a.emgformer_arms),
           'vemg2pose': (json.load(open(a.vemg2pose_json)), a.vemg2pose_arms)}
    subsets = [ALL, *a.stages]
    val, nses = {}, {}        # val[model][role][subset] -> (seeds, 4)
    for m, (d, arms) in src.items():
        st = {r['stage']: r for r in d['stages']}
        nses[m] = {s: st[s]['n'] for s in a.stages}
        nses[m][ALL] = d['n_sessions']
        val[m] = {}
        for role, arm in zip(('base', 'delay'), arms):
            val[m][role] = {ALL: np.array([[d['overall'][f'{arm}_s{s}'][k] for k in KEY] for s in a.seeds])}
            for s_ in a.stages:
                val[m][role][s_] = np.array([[st[s_][f'{arm}_s{s}'][k] for k in KEY] for s in a.seeds])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': a.font_family, f'font.{a.font_family}': [a.font], 'font.size': 7.5,
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 0, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'mathtext.fontset': 'custom'})
    LABEL_FS = 6.8                                         # row labels and x-axis titles share one size
    COL = {'EMGFormer': '#2f6bd8', 'vemg2pose': '#d0632b'}
    MK = {'EMGFormer': 'o', 'vemg2pose': 's'}
    fig, axs = plt.subplots(1, 4, figsize=(a.width_in, a.height_in), sharey=True, gridspec_kw={'wspace': 0.07})
    ys = np.arange(len(subsets))[::-1]
    summary = {}
    for j, ax in enumerate(axs):
        lim = 0
        for k, m in enumerate(COL):
            for y, s in zip(ys, subsets):
                d = val[m]['delay'][s][:, j] - val[m]['base'][s][:, j]
                mu, sd = d.mean(), d.std(ddof=DDOF)
                yy = y + (0.19 if k == 0 else -0.19)
                ax.errorbar(mu, yy, xerr=sd, fmt=MK[m], ms=3.2, color=COL[m], mfc=COL[m], mec='white', mew=0.4,
                            elinewidth=0.9, capsize=1.6, capthick=0.7, zorder=3)
                ax.scatter(d, np.full(len(d), yy), s=4, color=COL[m], alpha=0.45, lw=0, zorder=2)
                lim = max(lim, abs(mu) + sd, *np.abs(d))
                summary.setdefault(m, {}).setdefault(s, {})[MET[j]] = {
                    'base_mean': float(val[m]['base'][s][:, j].mean()), 'delta_mean': float(mu),
                    'delta_sd': float(sd), 'delta_per_seed': d.round(6).tolist()}
        lim *= 1.06
        ax.set_xlim(-lim, lim)
        ax.axvspan(-lim, 0, color='#2a9d55', alpha=0.07, lw=0, zorder=0)
        ax.axvline(0, color='0.25', lw=0.6, zorder=1)
        for y in ys[:-1]:
            ax.axhline(y - 0.5, color='0.88', lw=0.5, zorder=0)
        ax.set_xlabel(f'Δ {MET[j]}', fontsize=LABEL_FS, labelpad=1.5)
        ax.text(0.03, 1.0, 'better', transform=ax.transAxes, ha='left', va='top', fontsize=6.5, color='#2a7d48',
                style='italic')
        ax.ticklabel_format(axis='x', style='plain')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        ax.tick_params(axis='x', labelsize=6.3)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([LAB.get(s, s) + (f" ({nses['vemg2pose'][s]:,})" if s == ALL
                                             else f" ({nses['vemg2pose'][s]})") for s in subsets], fontsize=LABEL_FS)
    axs[0].set_ylim(-0.5, len(subsets) - 0.5)
    h = [plt.Line2D([], [], marker=MK[m], color=COL[m], ls='', ms=3.6, label=m) for m in COL]
    # the corner under the row labels is empty: the legend goes there, level with the x tick labels
    axs[0].legend(handles=h, loc='upper right', bbox_to_anchor=(-0.02, -0.005), frameon=False, fontsize=6.4,
                  handletextpad=0.05, handlelength=1.0, ncol=2, columnspacing=0.35, borderaxespad=0)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.97, bottom=0.2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'emg2pose_delay_minus_base_seeds{"-".join(a.seeds)}'
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', pad_inches=a.pad_in)
    fig.savefig(out / f'{stem}.png', dpi=300, bbox_inches='tight', pad_inches=a.pad_in)
    json.dump({'ddof': DDOF, 'n_sessions': nses, 'values': summary}, open(out / f'{stem}.json', 'w'), indent=1)
    print('wrote', out / f'{stem}.pdf')
    for m in summary:
        for s in subsets:
            print(f'{m:9s} {s[:24]:24s}',
                  '  '.join(f"{k} {v['delta_mean']:+.4f}±{v['delta_sd']:.4f}" for k, v in summary[m][s].items()))


if __name__ == '__main__':
    main()
