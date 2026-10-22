"""Generate all report figures (PDF) from data/campaign.json and data/aux.json."""
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
camp = json.loads((HERE / 'data/campaign.json').read_text())
aux = json.loads((HERE / 'data/aux.json').read_text())
rows = camp['per_case']

plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25})
CARD_LABEL = {'a100': 'A100 80GB (cuTENSOR)', 'v100': 'V100 32GB (cuTENSOR)', 'a100_cupy': 'A100 80GB (CuPy)',
              'v100_cupy': 'V100 32GB (CuPy)', 'p100_16': 'P100 16GB (CuPy)', 'p100_12': 'P100 12GB (CuPy)'}
COLOR = {'a100': '#1b9e77', 'v100': '#2c6fbb', 'a100_cupy': '#8bc34a', 'v100_cupy': '#7aa6e0', 'p100_16': '#e08a1e', 'p100_12': '#c0392b'}
FAMILY_MARK = {'scf': 'o', 'solvent': 's', 'td': '^', 'hessian': 'D', 'correlation': 'v', 'properties': 'P', 'optimization': 'X', 'pbc': '*'}


def by_card():
    d = defaultdict(list)
    for r in rows:
        d[r['card']].append(r)
    return d


CARDS = by_card()


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    plt.close(fig)
    print('wrote', name)


def ratio_rows(card, min_g1=0.0):
    out = []
    for r in CARDS.get(card, []):
        if r.get('G0_over_G1') and r.get('nao') and (r['G1'].get('median_s') or 0) >= min_g1:
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------

PREFERRED = {'v100': ['v100-review', 'v100'], 'a100': ['a100-inplace', 'a100'], 'p100-16gb-gram': ['p100-16gb-gram'], 'p100-12gb-gram': ['p100-12gb-gram']}


def tier_record(T, card, size, version, tier):
    """Storage-tier record for the final G1 where one exists; original-code runs are unchanged by the fixes.
    Returns (record, is_final_code)."""
    labels = PREFERRED.get(card, [card]) if version == 'current' else [card.replace('-gram', '')]
    for k, lab in enumerate(labels):
        rec = T.get(f'{lab}/{size}/{version}/{tier}')
        if rec:
            return rec, (version == 'original' or k == 0)
    return None, False


# 1. G0/G1 time ratio versus system size
def fig_speed_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, group, title in ((axes[0], ('a100', 'v100'), 'cuTENSOR engine'),
                             (axes[1], ('a100_cupy', 'v100_cupy', 'p100_16', 'p100_12'), 'CuPy fallback engine')):
        for card in group:
            for fam, mark in FAMILY_MARK.items():
                pts = [(r['nao'], r['G0_over_G1']) for r in ratio_rows(card) if r['family'] == fam]
                if pts:
                    ax.scatter(*zip(*pts), s=14, marker=mark, color=COLOR[card], alpha=.75, linewidths=.3, edgecolors='k',
                               label=CARD_LABEL[card] if fam == 'scf' else None)
        ax.axhline(1, color='k', lw=.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_yticks([.8, 1, 1.5, 2, 3])
        ax.set_yticklabels(['0.8', '1', '1.5', '2', '3'])
        ax.set_xlabel('basis functions')
        ax.set_title(title)
        ax.legend(loc='upper left', frameon=False)
    axes[0].set_ylabel('time original / time new  (>1: new faster)')
    handles = [plt.Line2D([], [], marker=m, ls='', color='gray', markersize=4, label=f) for f, m in FAMILY_MARK.items()]
    fig.legend(handles=handles, loc='lower center', ncol=8, frameon=False, bbox_to_anchor=(.5, -.09))
    save(fig, 'fig_speed_ratio_vs_size')


# 2. Distribution of G0/G1 for non-trivial cases
def fig_speed_violin():
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    order = ['a100', 'v100', 'a100_cupy', 'v100_cupy', 'p100_16', 'p100_12']
    data = [[r['G0_over_G1'] for r in ratio_rows(c, min_g1=5.0)] for c in order]
    pos = np.arange(len(order))
    parts = ax.violinplot([d for d in data], positions=pos, widths=.75, showmedians=True, showextrema=True)
    for body, c in zip(parts['bodies'], order):
        body.set_facecolor(COLOR[c])
        body.set_alpha(.6)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-.12, .12, len(d)), d, s=6, color='k', alpha=.5)
        ax.text(i, ax.get_ylim()[1] if False else 2.55, f'n={len(d)}', ha='center', fontsize=7)
    ax.axhline(1, color='k', lw=.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([CARD_LABEL[c].replace(' (', '\n(') for c in order])
    ax.set_ylabel('time original / time new')
    ax.set_title('Cases where the new code takes at least 5 s')
    ax.set_ylim(.7, 2.7)
    save(fig, 'fig_speed_violin')


# 3. memory
def fig_memory():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    for card in ('v100', 'a100', 'p100_16'):
        pts = [(r['G0']['gpu_peak_gb'], r['G1']['gpu_peak_gb']) for r in CARDS.get(card, [])
               if r['G0'].get('gpu_peak_gb') and r['G1'].get('gpu_peak_gb') and r['G0']['gpu_peak_gb'] > .2]
        if pts:
            ax.scatter(*zip(*pts), s=12, color=COLOR[card], alpha=.7, edgecolors='k', linewidths=.3, label=CARD_LABEL[card])
    lim = [.2, 80]
    ax.plot(lim, lim, 'k-', lw=.8, label='equal')
    for f in (2, 4):
        ax.plot(lim, [x / f for x in lim], 'k:', lw=.7)
        ax.text(lim[1] / 1.15, lim[1] / f / 1.25, f'{f}x less', fontsize=7, ha='right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('original: tracked GPU peak (GB)')
    ax.set_ylabel('new: tracked GPU peak (GB)')
    ax.set_title('Per-case GPU memory (memory-mode run)')
    ax.legend(loc='upper left', frameon=False)
    ax = axes[1]
    cases = [('core-vitamin_c-b3lyp-svp', 'vitamin C'), ('core-tamoxifen-b3lyp-svp', 'tamoxifen'), ('core-tamoxifen-b3lyp-tzvp', 'tamoxifen\nTZVP'),
             ('core-taxol-pbe-svp', 'Taxol\nPBE'), ('core-taxol-hf-svp', 'Taxol\nHF'), ('ext-vitamin_c-b3lyp-hessian', 'vitamin C\nHessian')]
    x = np.arange(len(cases))
    for k, (v, hatch, lab) in enumerate((('G0', '', 'original'), ('G1', '//', 'new'))):
        vals = []
        for cid, _ in cases:
            r = next((x_ for x_ in CARDS['v100'] if x_['case'] == cid), None)
            vals.append((r[v].get('gpu_peak_gb') or 0) if r else 0)
        ax.bar(x + (k - .5) * .38, vals, .38, label=lab, color='#888888' if v == 'G0' else COLOR['v100'], hatch=hatch, edgecolor='k', linewidth=.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1].replace('\n', ' ') for c in cases], fontsize=6.5, rotation=35, ha='right')
    ax.set_ylabel('tracked GPU peak (GB), V100')
    ax.legend(frameon=False)
    ax.set_title('Selected cases')
    save(fig, 'fig_memory')


# 4. CPU vs GPU
def fig_cpu_gpu():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, key, title in ((axes[0], 'CPUcpu48_over_G1', 'CPU baseline: 48 threads, whole node'),
                           (axes[1], 'CPUbest_over_G1', 'CPU baseline: best of the CPU lanes')):
        for card in ('a100', 'v100', 'p100_16'):
            pts = [(r['nao'], r[key]) for r in CARDS.get(card, []) if r.get(key) and r.get('nao')]
            if pts:
                ax.scatter(*zip(*pts), s=12, color=COLOR[card], alpha=.7, edgecolors='k', linewidths=.3, label=CARD_LABEL[card])
        ax.axhline(1, color='k', lw=.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('basis functions')
        ax.set_title(title, fontsize=8)
        ax.legend(frameon=False, loc='lower right', fontsize=6.5)
    axes[0].set_ylabel('CPU time / GPU time (new code)')
    save(fig, 'fig_cpu_vs_gpu')


# 5. CPU thread scan
def fig_cpu_scan():
    scan = aux['cpu_scan']
    cases = [('core-h2o-b3lyp-svp', 'H$_2$O'), ('core-benzene-b3lyp-svp', 'benzene'), ('core-vitamin_c-b3lyp-svp', 'vitamin C'),
             ('core-tamoxifen-b3lyp-svp', 'tamoxifen'), ('core-taxol-pbe-svp', 'Taxol PBE')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for cid, lab in cases:
        pts = sorted((s['omp'], s['seconds']) for s in scan if s['case'] == cid and s['blas'] == s['omp'] and s['bind'] == 'none')
        if pts:
            best = min(p[1] for p in pts)
            ax.plot([p[0] for p in pts], [p[1] / best for p in pts], 'o-', ms=3, label=lab)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('threads (OpenMP = BLAS)')
    ax.set_ylabel('time / best time of that case')
    ax.set_title('Thread scaling, BLAS threads = OpenMP threads')
    ax.legend(frameon=False)
    ax = axes[1]
    names = ['t24', 't24_blas1', 't24_pinned', 't48', 't48_blas1', 't48_pinned', 't12']
    x = np.arange(3)
    wid = .11
    sel = [c for c in cases if c[0] in ('core-vitamin_c-b3lyp-svp', 'core-tamoxifen-b3lyp-svp', 'core-taxol-pbe-svp')]
    for k, name in enumerate(names):
        vals = []
        for cid, _ in sel:
            runs = [s['seconds'] for s in scan if s['case'] == cid and s['name'] == name]
            best = min(s['seconds'] for s in scan if s['case'] == cid)
            vals.append(runs[0] / best if runs else np.nan)
        ax.bar(x + (k - 3) * wid, vals, wid, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in sel])
    ax.set_ylabel('time / best time of that case')
    ax.set_title('Threading configurations (24/48/12 threads)')
    ax.legend(ncol=2, frameon=False, fontsize=6.5)
    save(fig, 'fig_cpu_scan')


# 6. Taxol stage profile
def fig_taxol_stages():
    ts = aux['taxol_stages']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    labels, scf_parts, grad_parts = [], [], []
    for key in ('a100/original', 'a100/current', 'v100/original', 'v100/current'):
        if key not in ts:
            continue
        s = ts[key]['stages']
        jk = s['scf.jk'] - s['scf.df_build_nested_in_jk']
        scf_parts.append([jk, s['scf.df_build_nested_in_jk'], s['scf.xc_quadrature'], s['scf.diagonalization'],
                          max(0, ts[key]['scf'] - s['scf.jk'] - s['scf.xc_quadrature'] - s['scf.diagonalization'])])
        grad_parts.append([s['gradient.df_jk_total'], s['gradient.xc_full_response'],
                           max(0, ts[key]['gradient'] - s['gradient.df_jk_total'] - s['gradient.xc_full_response'])])
        labels.append(key.replace('/', '\n').replace('original', 'orig.').replace('current', 'new'))
    names_scf = ['J/K per cycle', 'DF build', 'XC', 'diagonalization', 'other']
    names_grad = ['DF J/K gradient', 'XC gradient', 'other']
    for ax, parts, names, title in ((axes[0], scf_parts, names_scf, 'SCF (20 cycles)'), (axes[1], grad_parts, names_grad, 'Gradient')):
        bottom = np.zeros(len(labels))
        for i, nm in enumerate(names):
            vals = np.array([p[i] for p in parts])
            ax.bar(range(len(labels)), vals, bottom=bottom, label=nm)
            bottom += vals
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('seconds')
        ax.set_title(f'Taxol B3LYP/def2-SVP: {title}')
        ax.legend(frameon=False, fontsize=6.5)
    save(fig, 'fig_taxol_stages')


# 7. storage tiers
def fig_storage():
    t = aux['storage_tiers']
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex='col')
    systems = [('large', 'Taxol (1123 AOs)'), ('valinomycin', 'valinomycin (1542 AOs)')]
    cols = [('v100', 'V100\n32 GB'), ('a100', 'A100\n80 GB'), ('p100-16gb-gram', 'P100\n16 GB'), ('p100-12gb-gram', 'P100\n12 GB')]
    tiers = [('original', 'gpu', 'original, GPU-resident', '#bbbbbb'), ('original', 'host', 'original, host RAM', '#777777'),
             ('current', 'gpu', 'new, GPU-resident', COLOR['v100']), ('current', 'host', 'new, host RAM', COLOR['a100']),
             ('current', 'disk', 'new, disk', COLOR['p100_12'])]
    for j, (sk, sl) in enumerate(systems):
        for i, (key, ylabel) in enumerate((('seconds', 'workflow time (s)'), ('gpu_peak_gradient_gb', 'GPU peak, gradient phase (GB)'))):
            ax = axes[i][j]
            x = np.arange(len(cols))
            wid = .16
            for k, (ver, tier, lab, colr) in enumerate(tiers):
                vals, mark = [], []
                for ck, _ in cols:
                    rec, _final = tier_record(t, ck, sk, ver, tier)
                    completed = bool(rec and rec['status'] in ('passed', 'capacity_observation', 'failed_accuracy'))
                    have = completed and bool(rec.get(key))
                    vals.append(rec[key] if have else np.nan)
                    mark.append('' if have else ('?' if completed else 'x'))
                pos = x + (k - 2) * wid
                ax.bar(pos, vals, wid, label=lab if (i == 0 and j == 0) else None, color=colr, edgecolor='k', linewidth=.3)
                for p, m in zip(pos, mark):
                    if m:
                        ax.text(p, 0.012 if key != 'seconds' else 0.012, m, ha='center', va='bottom', fontsize=6.5,
                                color='#aa0000' if m == 'x' else '#555555', transform=ax.get_xaxis_transform())
            ax.set_xticks(x)
            ax.set_xticklabels([c[1] for c in cols], fontsize=7)
            ax.set_ylabel(ylabel)
            if key == 'seconds':
                ax.set_yscale('log')
            if i == 0:
                ax.set_title(sl)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(.5, -.045))
    fig.text(.5, -.075, 'red x: did not complete on that card;  ?: completed, but no memory measurement', ha='center', fontsize=7, color='#aa0000')
    save(fig, 'fig_storage_tiers')


# 8. capacity
def fig_capacity():
    cap = aux['capacity']
    nao = {'small': 208, 'medium': 537, 'large': 1123, 'valinomycin': 1542, 'water008': 1824, 'gly30': 2154, 'water009': 2400,
           'gly40': 2864, 'water010': 3336, 'gly50': 3574}
    cards = ['p100_12', 'p100_16', 'v100_16', 'v100', 'a100']
    cols = [('original', 'gpu', 'orig. GPU-resident'), ('original', 'host', 'orig. host RAM'), ('current', 'gpu', 'new GPU-resident'),
            ('current', 'host', 'new host RAM'), ('current', 'disk', 'new disk')]
    colors = ['#bbbbbb', '#777777', COLOR['v100'], COLOR['a100'], COLOR['p100_12']]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    x = np.arange(len(cards))
    for k, (ver, tier, lab) in enumerate(cols):
        vals = []
        for c in cards:
            done = [nao[r.split('/')[-1]] for key, r in ((k2, k2) for k2 in cap) if key.startswith(f'{c}/{ver}/{tier}/')
                    and cap[key]['status'] in ('passed', 'capacity_observation', 'failed_accuracy') and key.split('/')[-1] in nao]
            vals.append(max(done) if done else 0)
        ax.bar(x + (k - 2) * .16, vals, .16, label=lab, color=colors[k], edgecolor='k', linewidth=.3)
    ax.set_xticks(x)
    ax.set_xticklabels(['P100\n12GB', 'P100\n16GB', 'V100\n16GB', 'V100\n32GB', 'A100\n80GB'])
    ax.set_ylabel('largest completed system (basis functions)')
    ax.set_title('Capacity: B3LYP/def2-SVP energy + gradient')
    ax.legend(frameon=False, fontsize=6.3, loc='upper left')
    ax = axes[1]
    for (ver, tier, lab), col in zip(cols[2:], colors[2:]):
        pts = sorted((nao[key.split('/')[-1]], cap[key]['seconds'] / 60) for key in cap
                     if key.startswith(f'a100/{ver}/{tier}/') and cap[key]['status'] in ('passed', 'capacity_observation')
                     and cap[key].get('seconds') and key.split('/')[-1] in nao)
        if pts:
            ax.plot(*zip(*pts), 'o-', color=col, ms=3, label=lab)
    pts = sorted((nao[key.split('/')[-1]], cap[key]['seconds'] / 60) for key in cap
                 if key.startswith('a100/original/host/') and cap[key]['status'] in ('passed', 'capacity_observation')
                 and cap[key].get('seconds') and key.split('/')[-1] in nao)
    if pts:
        ax.plot(*zip(*pts), 's--', color='#777777', ms=3, label='orig. host RAM')
    ax.set_xlabel('basis functions')
    ax.set_ylabel('workflow time (min), A100')
    ax.set_yscale('log')
    ax.set_title('Time to solution on the A100')
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, 'fig_capacity')


# 9. accuracy
def fig_accuracy():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for card in ('a100', 'v100', 'p100_16', 'p100_12', 'a100_cupy', 'v100_cupy'):
        pts = []
        for r in CARDS.get(card, []):
            for v in ('G0', 'G1'):
                e = r[v].get('errors') or {}
                if 'energy_hartree' in e and 'gradient_hartree_per_bohr' in e and e['energy_hartree'] and e['gradient_hartree_per_bohr']:
                    pts.append((max(e['energy_hartree'], 1e-14), max(e['gradient_hartree_per_bohr'], 1e-14), v))
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, color=COLOR[card], alpha=.55, label=CARD_LABEL[card].split(' (')[0] + (' CuPy' if 'cupy' in card else ''))
    ax.axvline(1e-8, color='r', lw=.8)
    ax.axhline(1e-6, color='r', lw=.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('energy error vs CPU (Eh)')
    ax.set_ylabel('max gradient error (Eh/Bohr)')
    ax.set_title('SCF energy + gradient cases (red: gates)')
    ax.legend(frameon=False, fontsize=6, ncol=2, loc='lower right')
    ax = axes[1]
    obs = defaultdict(list)
    for card in ('a100', 'v100', 'p100_16', 'p100_12'):
        for r in CARDS.get(card, []):
            for v in ('G0', 'G1'):
                for k, val in (r[v].get('errors') or {}).items():
                    if val:
                        obs[k].append(val)
    names = [k for k in ('energy_hartree', 'gradient_hartree_per_bohr', 'hessian_hartree_per_bohr2', 'excitation_energy_hartree', 'polarizability_au', 'shielding_ppm') if k in obs]
    ax.boxplot([np.log10(obs[k]) for k in names], tick_labels=[n.replace('_hartree', '').replace('_per_bohr2', '').replace('_per_bohr', '').replace('_', '\n') for n in names], showfliers=True,
               flierprops=dict(markersize=2))
    gates = {'energy_hartree': -8, 'gradient_hartree_per_bohr': -6, 'hessian_hartree_per_bohr2': -5, 'polarizability_au': -5, 'shielding_ppm': -1}
    for i, n in enumerate(names, 1):
        if n in gates:
            ax.plot([i - .35, i + .35], [gates[n]] * 2, color='r', lw=1.4)
    ax.set_ylabel('log10 max error (red: gate)')
    ax.set_title('Error by observable (A100, V100, P100 cards)')
    ax.tick_params(axis='x', labelsize=6)
    save(fig, 'fig_accuracy')


# 10. coverage
def fig_status():
    order = ['a100', 'v100', 'p100_16', 'p100_12', 'a100_cupy', 'v100_cupy']
    statuses = ['qualified', 'accuracy_failed', 'nonconverged', 'resource_limited', 'infrastructure_failed', 'none']
    colors = ['#2e7d32', '#ef6c00', '#fbc02d', '#c62828', '#6a1b9a', '#bdbdbd']
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for i, card in enumerate(order):
        cnt = defaultdict(int)
        for r in CARDS.get(card, []):
            for v in ('G0', 'G1'):
                s = r[v].get('memory_status')
                cnt[s or 'none'] += 1
        left = 0
        total = sum(cnt.values())
        for s, c in zip(statuses, colors):
            ax.barh(i, cnt[s] / total * 100, left=left, color=c, label=(s if s != 'none' else 'no record') if i == 0 else None)
            left += cnt[s] / total * 100
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([CARD_LABEL[c] for c in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('% of (case, variant) memory-run outcomes')
    ax.legend(ncol=3, frameon=False, fontsize=6.5, loc='lower center', bbox_to_anchor=(.45, 1.0))
    save(fig, 'fig_status')


# 11. GEMM microbenchmark
def fig_gemm():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, card in zip(axes, ('a100', 'v100')):
        g = aux['gemm'].get(card)
        if not g:
            continue
        blocks = [b for b in g['blocks'] if 'skipped' not in b]
        x = np.arange(len(blocks))
        ax.bar(x - .2, [b['K_per_cycle_cutensor_s'] for b in blocks], .4, label='cuTENSOR path', color=COLOR[card])
        ax.bar(x + .2, [b['K_per_cycle_gemm_s'] for b in blocks], .4, label='cuBLAS GEMM/SYRK path', color='#999999')
        ax.set_xticks(x)
        ax.set_xticklabels([b['nL'] for b in blocks])
        ax.set_xlabel('auxiliary functions per block')
        ax.set_ylabel('exchange build per cycle (s)')
        ax.set_title(f"{g['device'].replace('NVIDIA ', '').replace('Tesla ', '')}\ncuBLAS DGEMM ceiling {g['dgemm_ceiling_tflops']:.1f} TFLOPS", fontsize=8)
        ax.legend(frameon=False)
    save(fig, 'fig_gemm')


# 12. cold start
def fig_cold():
    cold = aux['cold_start']
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7), sharey=True)
    for ax, (card, rows_) in zip(axes, cold.items()):
        combos = sorted({(r['engine'], r['variant']) for r in rows_})
        x = np.arange(len(combos))
        for k, (state, colr) in enumerate((('cold', '#e08a1e'), ('warm', '#2c6fbb'))):
            vals = [next((r['seconds'] for r in rows_ if r['engine'] == e and r['variant'] == v and r['state'] == state
                          and r['case'] == 'core-h2o-b3lyp-svp'), np.nan) for e, v in combos]
            ax.bar(x + (k - .5) * .38, vals, .38, color=colr, label=('first process (empty kernel cache)' if state == 'cold' else 'repeat (cache populated)') if card == 'a100' else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{'cuT.' if e == 'cutensor' else 'CuPy'}\n{'orig.' if v == 'G0' else 'new'}" for e, v in combos], fontsize=6.5)
        ax.set_title({'a100': 'A100', 'v100': 'V100', 'p100_16': 'P100 16GB', 'p100_12': 'P100 12GB'}[card], fontsize=8)
    axes[0].set_ylabel('H$_2$O B3LYP workflow time (s)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.08))
    save(fig, 'fig_cold_start')


# 13. SCF cycles
def fig_cycles():
    cyc = aux['scf_cycles']
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    strategies = ['base', 'atom', 'huckel', 'vsap', 'diis12', 'twostage']
    sizes = ['small', 'medium', 'large']
    x = np.arange(len(strategies))
    for k, size in enumerate(sizes):
        vals = []
        for s in strategies:
            rec = cyc.get(f'a100_{size}_{s}')
            vals.append(rec['full_quality_cycles'] if rec and rec['status'] == 'ok' else np.nan)
        ax.bar(x + (k - 1) * .26, vals, .26, label={'small': 'vitamin C (208 AOs)', 'medium': 'tamoxifen (537)', 'large': 'Taxol (1123)'}[size])
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('full-quality SCF cycles (A100)')
    ax.set_title('SCF starting guess and convergence strategy (same thresholds)')
    ax.legend(frameon=False)
    save(fig, 'fig_scf_cycles')


# 14. Tamoxifen engine stage profile
def fig_engine_profile():
    ep = aux['tamoxifen_engine_profile']
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    keys = [k for k in ep]
    parts = []
    for k in keys:
        rows_ = ep[k]
        med = lambda f: st.median(f(r) for r in rows_)
        jk = med(lambda r: r['stages'].get('scf.jk', 0))
        xc = med(lambda r: r['stages'].get('scf.xc_quadrature', 0))
        gj = med(lambda r: r['stages'].get('gradient.df_jk_total', 0))
        gx = med(lambda r: r['stages'].get('gradient.xc_full_response', 0))
        tot = med(lambda r: r['total'])
        parts.append([jk, xc, gj, gx, max(0, tot - jk - xc - gj - gx)])
    names = ['SCF J/K', 'SCF XC', 'gradient J/K', 'gradient XC', 'other']
    bottom = np.zeros(len(keys))
    for i, nm in enumerate(names):
        vals = np.array([p[i] for p in parts])
        ax.bar(range(len(keys)), vals, bottom=bottom, label=nm)
        bottom += vals
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace('/', '\n').replace('original', 'orig.').replace('current', 'new') for k in keys], fontsize=6)
    ax.set_ylabel('seconds (median of 3)')
    ax.set_title('Tamoxifen B3LYP/def2-SVP stage times by engine')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    save(fig, 'fig_engine_profile')


if __name__ == '__main__':
    for f in (fig_speed_ratio, fig_speed_violin, fig_memory, fig_cpu_gpu, fig_cpu_scan, fig_taxol_stages, fig_storage, fig_capacity,
              fig_accuracy, fig_status, fig_gemm, fig_cold, fig_cycles, fig_engine_profile):
        try:
            f()
        except Exception as error:   # keep going: a figure with missing data must not block the report build
            import traceback
            print('FAILED', f.__name__, repr(error))
            traceback.print_exc()
