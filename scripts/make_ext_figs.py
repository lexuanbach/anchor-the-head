"""fig_frontier + fig_drift from metrics_align.json / metrics_diag.json.

fig_frontier: novel-IoU vs seen-IoU, one curve per interpolation knob (distillation lambda,
output gate g, residual scale eta, trust region tau). The frozen baseline and the collapsed
seen-only head are the two endpoints of every curve.
fig_drift: alignment drift vs collapse index across measured corrections; fit and Pearson r use class-specific settings.
Run: python scripts/make_ext_figs.py --align results/extended/metrics_align.json --diag results/extended/metrics_diag.json
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np


def get(res, k, f):
    return res[k][f]['mean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', default='results/extended/metrics_align.json')
    ap.add_argument('--diag', default='results/extended/metrics_diag.json')
    ap.add_argument('--lposs', default='results/extended/metrics_lposs.json')
    ap.add_argument('--outdir', default='figures')
    a = ap.parse_args()
    import os; os.makedirs(a.outdir, exist_ok=True)
    A = json.load(open(a.align))['summary']
    D = json.load(open(a.diag))['summary']
    import os
    L = json.load(open(a.lposs))['summary'] if os.path.exists(a.lposs) else {}

    # ---------------- fig_frontier ----------------
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    curves = {
        'B1 logit distillation ($\\lambda$)':
            (['decoder_seen_only'] + [f'B1_logitKD_lam{l}' for l in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]], A, 'o-', '#1f77b4'),
        'B3 output gate ($g$)':
            ([f'B3_gate{g}' for g in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]], A, 's-', '#ff7f0e'),
        'B4 residual scale ($\\eta$)':
            ([f'B4_winterp{w}' for w in [1.0, 0.75, 0.5, 0.25]], A, '^-', '#2ca02c'),
        'C4 trust region ($\\tau$)':
            ([f'C4_tau{t}' for t in [4.0, 2.0, 1.0, 0.5]], D, 'd-', '#9467bd'),
    }
    for label, (keys, src, style, color) in curves.items():
        xs = [get(src, k, 'seenIoU') for k in keys if k in src]
        ys = [get(src, k, 'novelIoU') for k in keys if k in src]
        ax.plot(xs, ys, style, label=label, color=color, ms=4, lw=1.4, alpha=0.9)
    bx, by = get(A, 'uncal', 'seenIoU'), get(A, 'uncal', 'novelIoU')
    ax.scatter([bx], [by], marker='*', s=170, color='black', zorder=5, label='frozen baseline')
    ax.scatter([get(A, 'B6_learned_affinity', 'seenIoU')], [get(A, 'B6_learned_affinity', 'novelIoU')],
               marker='P', s=70, color='#d62728', zorder=5, label='B6 learned affinity (class-sym.)')
    ax.axhline(by, color='gray', lw=0.6, ls=':')
    ax.axvline(bx, color='gray', lw=0.6, ls=':')
    ax.set_xlabel('seen-class IoU')
    ax.set_ylabel('novel-class IoU')
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)
    ax.set_title('Novel- and seen-class IoU under anchoring', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{a.outdir}/fig_frontier.pdf')
    print(f'wrote {a.outdir}/fig_frontier.pdf')

    # ---------------- fig_drift ----------------
    fig2, ax2 = plt.subplots(figsize=(4.6, 3.4))
    sym_keys = {'uncal', 'globalT', 'diffusion', 'laplacian', 'B6_learned_affinity', 'LPOSS'}
    xs_s, ys_s, xs_c, ys_c = [], [], [], []
    pts = []
    Lsub = {k: v for k, v in L.items() if k in ('LPOSS', 'LPOSS+head', 'base+head')}
    for src in (A, D, Lsub):
        for k, v in src.items():
            ci = v.get('collapse_index')
            dr = v['drift']['mean'] if 'drift' in v else None
            if ci is None or dr is None or dr != dr:
                continue
            pts.append((dr, ci))
            if k in sym_keys:
                xs_s.append(dr); ys_s.append(ci)
            else:
                xs_c.append(dr); ys_c.append(ci)
    x = np.asarray(xs_c); y = np.asarray(ys_c)
    assert len(x) == 38, f'Expected 38 class-specific settings, got {len(x)}'
    r = float(np.corrcoef(x, y)[0, 1])
    ax2.scatter(xs_c, ys_c, s=26, color='#1f77b4', alpha=0.8, label='class-specific corrections')
    ax2.scatter(xs_s, ys_s, s=42, color='#d62728', marker='^', alpha=0.9, label='class-symmetric corrections')
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(0, x.max(), 50)
    ax2.plot(xx, m * xx + b, color='gray', lw=1, ls='--', label=f'class-specific fit ($r={r:.2f}$)')
    ax2.set_xlabel('alignment drift ($100\\,\\mathrm{mean}[1-\\cos(p,p^0)]$)')
    ax2.set_ylabel('collapse index (novel-IoU / baseline)')
    ax2.set_title('Novel-class retention and alignment drift', fontsize=9)
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(f'{a.outdir}/fig_drift.pdf')
    print(f'wrote {a.outdir}/fig_drift.pdf  (n_class_specific={len(x)}, pearson r={r:.3f})')


if __name__ == '__main__':
    main()
