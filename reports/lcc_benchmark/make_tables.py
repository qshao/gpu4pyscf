"""Generate LaTeX table fragments and number macros from data/campaign.json and data/aux.json."""
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAB = HERE / 'tables'
TAB.mkdir(exist_ok=True)
camp = json.loads((HERE / 'data/campaign.json').read_text())
aux = json.loads((HERE / 'data/aux.json').read_text())
rows = camp['per_case']
audits = camp['audits']
CARDS = defaultdict(dict)
for r in rows:
    CARDS[r['card']][r['case']] = r


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

LABEL = {'a100': 'A100', 'v100': 'V100', 'a100_cupy': 'A100 CuPy', 'v100_cupy': 'V100 CuPy', 'p100_16': 'P100-16', 'p100_12': 'P100-12'}
ORDER = ['a100', 'v100', 'a100_cupy', 'v100_cupy', 'p100_16', 'p100_12']
macros = {}
KEY = {'a100': 'AHund', 'v100': 'VHund', 'a100_cupy': 'AHundCupy', 'v100_cupy': 'VHundCupy', 'p100_16': 'PSixteen', 'p100_12': 'PTwelve'}


def tex(s):
    return str(s).replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')


def tt(s):
    return '\\texttt{' + tex(s) + '}'


def f_time(s):
    if s is None:
        return '--'
    return f'{s:.2f}' if s < 10 else f'{s:.1f}' if s < 1000 else f'{s:.0f}'


def f_ratio(x):
    return '--' if x is None else f'{x:.2f}'


def geomean(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def write(name, text):
    (TAB / name).write_text(text)
    print('wrote', name)


def table(header, body, align, caption=None, label=None, small=True, star=False):
    env = 'table*' if star else 'table'
    fit = len(header) >= 6                     # wide tables are scaled to the text width
    lines = [f'\\begin{{{env}}}[!htbp]', '\\centering']
    if small:
        lines.append('\\small')
    if fit:
        lines.append('\\resizebox{\\textwidth}{!}{%')
    lines += [f'\\begin{{tabular}}{{{align}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    lines += [' & '.join(r) + ' \\\\' if r != ['\\midrule'] else '\\midrule' for r in body]
    lines += ['\\bottomrule', '\\end{tabular}' + ('}' if fit else '')]
    if caption:
        lines.append(f'\\caption{{{caption}}}')
    if label:
        lines.append(f'\\label{{{label}}}')
    lines.append(f'\\end{{{env}}}')
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------------------------------------------------------------
# macros: audited record counts and headline numbers
total_records = sum(a['record_count'] for a in audits.values())
macros['NAuditedRecords'] = f'{total_records:,}'.replace(',', '\\,')
macros['NStrata'] = str(len(audits) - 1)
cases_all = {r['case'] for r in rows}
macros['NCasesRun'] = str(len(cases_all))
for c in ORDER:
    if c in audits:
        macros[f'NRec{KEY[c]}'] = str(audits[c]['record_count'])


def ratios(card, min_g1=0.0):
    return [r['G0_over_G1'] for r in CARDS.get(card, {}).values() if r.get('G0_over_G1') and (r['G1'].get('median_s') or 0) >= min_g1]


for c in ORDER:
    for tag, thr in (('All', 0.0), ('Big', 5.0)):
        rr = ratios(c, thr)
        key = KEY[c]
        macros[f'Median{tag}{key}'] = f'{st.median(rr):.2f}' if rr else '--'
        macros[f'N{tag}{key}'] = str(len(rr))
        macros[f'Max{tag}{key}'] = f'{max(rr):.2f}' if rr else '--'
        macros[f'Min{tag}{key}'] = f'{min(rr):.2f}' if rr else '--'
# memory ratios (V100)
mr = [r['G0']['gpu_peak_gb'] / r['G1']['gpu_peak_gb'] for r in CARDS['v100'].values()
      if r['G0'].get('gpu_peak_gb') and r['G1'].get('gpu_peak_gb') and r['G0']['gpu_peak_gb'] > .3]
macros['MemMedian'] = f'{st.median(mr):.1f}'
macros['MemMax'] = f'{max(mr):.1f}'
macros['MemMin'] = f'{min(mr):.2f}'
macros['MemN'] = str(len(mr))

# ------------------------------------------------------------------------------------------------------------------------------
# coverage table
body = []
for c in ORDER:
    cnt = Counter()
    for r in CARDS.get(c, {}).values():
        for v in ('G0', 'G1'):
            if r[v].get('memory_status'):
                cnt[r[v]['memory_status']] += 1
    n = sum(cnt.values())
    body.append([LABEL[c], str(n), str(cnt['qualified']), str(cnt['accuracy_failed']), str(cnt['nonconverged']),
                 str(cnt['resource_limited']), str(cnt['infrastructure_failed']), str(audits[c]['record_count'])])
write('tab_coverage.tex', table(['stratum', 'outcomes', 'qualified', 'accuracy failed', 'non-converged', 'resource limited', 'infrastructure', 'audited records'],
                                body, 'lrrrrrrr',
                                'Outcomes of the memory-mode run of each (case, code version) pair on each stratum, and the number of audited records. '
                                '`Infrastructure\' on the P100 cards includes plain CUDA out-of-memory errors that the runner labelled generically (see text).',
                                'tab:coverage'))

# ------------------------------------------------------------------------------------------------------------------------------
# family table
fams = sorted({r['family'] for r in rows})
hdr = ['family', 'cases'] + [LABEL[c] for c in ORDER]
body = []
for f in fams:
    ncase = len({r['case'] for r in rows if r['family'] == f})
    line = [tex(f), str(ncase)]
    for c in ORDER:
        rr = [r['G0_over_G1'] for r in CARDS.get(c, {}).values() if r['family'] == f and r.get('G0_over_G1')]
        g = geomean(rr)
        line.append(f'{g:.2f} ({len(rr)})' if g else '--')
    body.append(line)
write('tab_families.tex', table(hdr, body, 'lr' + 'r' * len(ORDER),
                                'Geometric mean of the time ratio original/new per job family (number of cases in brackets); only cases where both versions '
                                'passed the accuracy gate are included. Values above 1 mean the new code is faster.', 'tab:families', star=True))

# ------------------------------------------------------------------------------------------------------------------------------
# large cases
LARGE = ['core-vitamin_c-b3lyp-svp', 'core-inosine-b3lyp-svp', 'core-tamoxifen-lda-svp', 'core-tamoxifen-pbe-svp', 'core-tamoxifen-tpss-svp',
         'core-tamoxifen-b3lyp-svp', 'core-tamoxifen-wb97x-svp', 'core-tamoxifen-pbe-tzvp', 'core-tamoxifen-b3lyp-tzvp',
         'core-taxol-lda-svp', 'core-taxol-pbe-svp', 'core-taxol-tpss-svp', 'core-taxol-hf-svp', 'core-taxol-b3lyp-svp', 'core-taxol-wb97x-svp',
         'ext-tamoxifen-b3lyp-iefpcm', 'ext-tamoxifen-b3lyp-smd', 'ext-tamoxifen-wb97mv', 'ext-vitamin_c-b3lyp-hessian', 'ext-inosine-b3lyp-tddft']


def case_cell(card, cid):
    r = CARDS.get(card, {}).get(cid)
    if not r:
        return '--'
    g0, g1 = r['G0'], r['G1']
    if r.get('G0_over_G1'):
        return f"{f_time(g1['median_s'])} / {r['G0_over_G1']:.2f}"
    s1 = g1.get('memory_status')
    if g1.get('median_s') and g0.get('memory_status') == 'accuracy_failed':
        return f"{f_time(g1['median_s'])} / (G0 acc.)"
    if s1 in ('resource_limited', 'infrastructure_failed'):
        return 'mem.'
    if s1 == 'nonconverged':
        return 'nc'
    return '--'


for name, group, cap in (('tab_large_cutensor.tex', ['a100', 'v100'], 'cuTENSOR engine'),
                         ('tab_large_cupy.tex', ['a100_cupy', 'v100_cupy', 'p100_16', 'p100_12'], 'CuPy fallback engine')):
    body = []
    for cid in LARGE:
        nao = next((r.get('nao') for c in ORDER for r in [CARDS.get(c, {}).get(cid)] if r and r.get('nao')), '')
        body.append([tt(cid.replace('core-', '').replace('ext-', '')), str(nao)] + [case_cell(c, cid) for c in group])
    write(name, table(['case', 'AOs'] + [LABEL[c] for c in group], body, 'lr' + 'r' * len(group),
                      f'Larger cases, {cap}: new-code time in seconds / ratio original-over-new. `mem.\' = did not fit in GPU memory or was refused by the '
                      'memory forecast; `nc\' = reference or GPU solver did not converge; `(G0 acc.)\' = the original code fails the CPU accuracy gate, so no '
                      'equal-accuracy ratio exists.', 'tab:large_' + ('cutensor' if 'cuTENSOR' in cap else 'cupy')))

# ------------------------------------------------------------------------------------------------------------------------------
# memory table (V100, cuTENSOR) + P100 notes
MEM = ['core-vitamin_c-b3lyp-svp', 'core-inosine-b3lyp-svp', 'core-tamoxifen-lda-svp', 'core-tamoxifen-b3lyp-svp', 'core-tamoxifen-wb97x-svp',
       'core-tamoxifen-b3lyp-tzvp', 'core-taxol-lda-svp', 'core-taxol-pbe-svp', 'core-taxol-tpss-svp', 'core-taxol-hf-svp', 'core-taxol-b3lyp-svp',
       'core-taxol-wb97x-svp', 'ext-tamoxifen-b3lyp-smd', 'ext-vitamin_c-b3lyp-hessian', 'ext-inosine-b3lyp-tddft', 'ext-vitamin_c-dfmp2']
body = []
for cid in MEM:
    r = CARDS['v100'].get(cid)
    if not r:
        continue
    g0, g1 = r['G0'].get('gpu_peak_gb'), r['G1'].get('gpu_peak_gb')
    body.append([tt(cid.replace('core-', '').replace('ext-', '')), str(r.get('nao', '')), f'{g0:.2f}' if g0 else '--', f'{g1:.2f}' if g1 else '--',
                 f'{g0 / g1:.2f}' if g0 and g1 else '--'])
write('tab_memory.tex', table(['case', 'AOs', 'original (GB)', 'new (GB)', 'ratio'], body, 'lrrrr',
                              'Tracked GPU memory peak of the whole workflow on the V100 (memory-mode run; excludes the CuPy pool and native workspaces).',
                              'tab:memory'))

# ------------------------------------------------------------------------------------------------------------------------------
# CPU vs GPU table
CPUSEL = ['core-h2o-b3lyp-svp', 'core-benzene-b3lyp-svp', 'core-vitamin_c-b3lyp-svp', 'core-inosine-b3lyp-svp', 'core-tamoxifen-b3lyp-svp',
          'core-tamoxifen-b3lyp-tzvp', 'core-taxol-pbe-svp', 'core-taxol-tpss-svp', 'ext-vitamin_c-b3lyp-hessian', 'ext-inosine-b3lyp-tddft',
          'ext-tamoxifen-b3lyp-iefpcm', 'ext-vitamin_c-dfmp2']
body = []
for cid in CPUSEL:
    r = CARDS['a100'].get(cid)
    if not r:
        continue
    cfg = r.get('CPU_by_config') or {}
    c48 = cfg.get('cpu48')
    best = r['CPU'].get('median_s')
    cell = lambda card: (f_time(CARDS[card][cid]['G1'].get('median_s')) if cid in CARDS.get(card, {}) else '--')
    body.append([tt(cid.replace('core-', '').replace('ext-', '')), str(r.get('nao', '')), f_time(c48), f_time(best) + (f" ({tex(r.get('CPU_best_config'))})" if r.get('CPU_best_config') and len(cfg) > 1 else ''),
                 cell('a100'), cell('v100'), cell('p100_16'),
                 f"{best / r['G1']['median_s']:.0f}" if best and r['G1'].get('median_s') else '--'])
write('tab_cpu_gpu.tex', table(['case', 'AOs', 'CPU 48 thr (s)', 'CPU best (s)', 'A100 new (s)', 'V100 new (s)', 'P100-16 new (s)', 'speed-up A100'],
                               body, 'lrrrrrrr',
                               'CPU wall time against the new GPU code. CPU 48 thr: whole 48-core node as requested; CPU best: the faster of the available CPU lanes '
                               '(configuration in brackets); speed-up = CPU best / A100.', 'tab:cpugpu', star=True))

# ------------------------------------------------------------------------------------------------------------------------------
# storage tiers
T = aux['storage_tiers']
body = []
for gi, (size, sl) in enumerate((('large', 'Taxol (1123 AOs)'), ('valinomycin', 'valinomycin (1542 AOs)'))):
    if gi:
        body.append(['\\midrule'])
    for ck, cl in (('v100', 'V100 32GB'), ('a100', 'A100 80GB'), ('p100-16gb-gram', 'P100 16GB'), ('p100-12gb-gram', 'P100 12GB')):
        for ver, tier, tl in (('original', 'gpu', 'orig. GPU'), ('original', 'host', 'orig. host'), ('current', 'gpu', 'new GPU'), ('current', 'host', 'new host'),
                              ('current', 'disk', 'new disk')):
            rec, final = tier_record(T, ck, size, ver, tier)
            if not rec:
                continue
            ok = rec['status'] in ('passed', 'capacity_observation', 'failed_accuracy')
            dag = '' if final or not ok else '$^\\dagger$'
            body.append([sl if tl == 'orig. GPU' and ck == 'v100' else '', cl if tl == 'orig. GPU' or tl == 'orig. host' else '', tl,
                         (f_time(rec['seconds']) + dag) if ok else 'out of memory',
                         (f"{rec['gpu_peak_gradient_gb']:.1f}" + dag) if ok and rec.get('gpu_peak_gradient_gb') else '--',
                         f"{rec['host_rss_gb']:.0f}" if ok and rec.get('host_rss_gb') else '--'])
write('tab_storage.tex', table(['system', 'card', 'version / tier', 'time (s)', 'GPU peak, gradient (GB)', 'host RSS (GB)'], body, 'lllrrr',
                               'Storage tiers for the fitted integrals. Taxol has a CPU reference and passes the accuracy gate for the new code in every tier '
                               '(the original code fails its gradient gate); valinomycin is a capacity observation. The original code has no disk tier. Rows marked $\\dagger$ were run before '
                               'the final gradient-memory fixes (their gradient peaks are 6.6 and 14.8 GB instead of 4.1 and 8.7 GB); all other new-code rows use the final code.',
                               'tab:storage'))

# ------------------------------------------------------------------------------------------------------------------------------
# capacity
cap = aux['capacity']
atoms = {'small': 20, 'medium': 57, 'large': 113, 'valinomycin': 168, 'water008': 228, 'gly30': 213, 'water009': 300, 'gly40': 283,
         'water010': 417, 'gly50': 353}
nao = {'small': 208, 'medium': 537, 'large': 1123, 'valinomycin': 1542, 'water008': 1824, 'gly30': 2154, 'water009': 2400, 'gly40': 2864,
       'water010': 3336, 'gly50': 3574}
nice = {'small': 'vitamin C', 'medium': 'tamoxifen', 'large': 'Taxol', 'valinomycin': 'valinomycin', 'water008': '(H$_2$O)$_{76}$', 'gly30': 'gly30',
        'water009': '(H$_2$O)$_{100}$', 'gly40': 'gly40', 'water010': '(H$_2$O)$_{139}$', 'gly50': 'gly50'}
done = ('passed', 'capacity_observation', 'failed_accuracy')
body = []
biggest = (0, None)
for card, cl in (('p100_12', 'P100 12GB'), ('p100_16', 'P100 16GB'), ('v100_16', 'V100 16GB'), ('v100', 'V100 32GB'), ('a100', 'A100 80GB')):
    first = True
    for ver, tier, tl in (('original', 'gpu', 'orig. GPU'), ('original', 'host', 'orig. host'), ('current', 'gpu', 'new GPU'), ('current', 'host', 'new host'),
                          ('current', 'disk', 'new disk')):
        best = [(nao[k.split('/')[-1]], k.split('/')[-1]) for k in cap if k.startswith(f'{card}/{ver}/{tier}/') and cap[k]['status'] in done and k.split('/')[-1] in nao]
        if best:
            n, rung = max(best)
            rec = cap[f'{card}/{ver}/{tier}/{rung}']
            if n > biggest[0] and ver == 'current':
                biggest = (n, (card, tier, rung))
            body.append([cl if first else '', tl, nice[rung], str(atoms[rung]), str(n), f_time(rec['seconds'] / 60) if rec.get('seconds') else '--'])
        else:
            body.append([cl if first else '', tl, '--', '--', '--', '--'])
        first = False
    body.append(['\\midrule'])
body.pop()
write('tab_capacity.tex', table(['card', 'version / tier', 'largest system', 'atoms', 'AOs', 'time (min)'], body, 'llcrrr',
                                'Largest system completed with B3LYP/def2-SVP energy and full-response gradient. These are capacity observations: no CPU reference '
                                'exists, so only finite values and convergence are checked.', 'tab:capacity'))
if biggest[1]:
    macros['BiggestAOs'] = str(biggest[0])
    macros['BiggestAtoms'] = str(atoms[biggest[1][2]])
    macros['BiggestName'] = nice[biggest[1][2]]

# ------------------------------------------------------------------------------------------------------------------------------
# CPU scan table
scan = aux['cpu_scan']
body = []
names = [('t1', '1'), ('t2', '2'), ('t4', '4'), ('t8', '8'), ('t12', '12'), ('t24', '24'), ('t48', '48'), ('t24_blas1', '24/B1'), ('t48_blas1', '48/B1'),
         ('t24_pinned', '24 pin'), ('t48_pinned', '48 pin')]
for cid, cl in (('core-h2o-b3lyp-svp', 'H$_2$O'), ('core-benzene-b3lyp-svp', 'benzene'), ('core-vitamin_c-b3lyp-svp', 'vitamin C'),
                ('core-tamoxifen-b3lyp-svp', 'tamoxifen'), ('core-taxol-pbe-svp', 'Taxol PBE')):
    line = [cl]
    vals = {s['name']: s['seconds'] for s in scan if s['case'] == cid}
    best = min(vals.values()) if vals else None
    for n, _ in names:
        v = vals.get(n)
        line.append('--' if v is None else (f'\\textbf{{{f_time(v)}}}' if v == best else f_time(v)))
    body.append(line)
write('tab_cpu_scan.tex', table(['case'] + [h for _, h in names], body, 'l' + 'r' * len(names),
                                'CPU thread scan, wall time in seconds on one 48-core node. Columns: OpenMP threads with BLAS threads equal to OpenMP threads; '
                                '`B1\' = BLAS threads set to 1; `pin\' = OpenMP threads pinned to cores. Bold: fastest for that case.', 'tab:cpuscan', star=True))

# ------------------------------------------------------------------------------------------------------------------------------
# cold start
cold = aux['cold_start']
body = []
for card, cl in (('a100', 'A100'), ('v100', 'V100'), ('p100_16', 'P100-16'), ('p100_12', 'P100-12')):
    first = True
    for eng in sorted({r['engine'] for r in cold.get(card, [])}):
        for ver in ('G0', 'G1'):
            def get(state, case='core-tamoxifen-b3lyp-svp'):
                return next((r['seconds'] for r in cold[card] if r['engine'] == eng and r['variant'] == ver and r['state'] == state and r['case'] == case), None)
            c, w = get('cold'), get('warm')
            c2, w2 = get('cold', 'core-h2o-b3lyp-svp'), get('warm', 'core-h2o-b3lyp-svp')
            body.append([cl if first else '', eng, 'original' if ver == 'G0' else 'new', f_time(c2), f_time(w2), f_time(c), f_time(w),
                         f_time(c2 - w2) if c2 and w2 else '--'])
            first = False
write('tab_cold.tex', table(['card', 'engine', 'code', 'H$_2$O cold (s)', 'H$_2$O warm (s)', 'tamoxifen cold (s)', 'tamoxifen warm (s)', 'first-process overhead (s)'],
                            body, 'lllrrrrr',
                            'Cold start: the first process with an empty CuPy kernel cache against a repeat with the cache populated (workflow time inside the '
                            'process, imports excluded). Only the first process of each pass is truly cold, so the first-process overhead is the H$_2$O cold time minus its warm time; '
                            'the later, larger cases of the cold pass reuse kernels compiled by H$_2$O.', 'tab:cold'))

# ------------------------------------------------------------------------------------------------------------------------------
# per-case appendix
cases_sorted = sorted(cases_all)
lines = ['\\scriptsize\\setlength{\\tabcolsep}{3pt}', '\\begin{longtable}{lr' + 'r' * 6 + 'rrrr}', '\\caption{Per-case results. Columns A100 to P100-12: time ratio original/new (``nc\'\': a solver or reference did not '
         'converge; ``mem\'\': resource limit; ``acc\'\': the original code fails the accuracy gate; ``--\'\': not run or not qualified). G1: new-code time on the V100; '
         'mem: V100 GPU memory original/new in GB; CPU: best CPU time.}\\label{tab:percase}\\\\', '\\toprule',
         'case & AOs & A100 & V100 & A100c & V100c & P100-16 & P100-12 & G1 V100 (s) & mem (GB) & CPU (s) \\\\', '\\midrule', '\\endfirsthead',
         '\\toprule case & AOs & A100 & V100 & A100c & V100c & P100-16 & P100-12 & G1 V100 (s) & mem (GB) & CPU (s) \\\\ \\midrule \\endhead',
         '\\bottomrule \\endfoot']


def rcell(card, cid):
    r = CARDS.get(card, {}).get(cid)
    if not r:
        return '--'
    if r.get('G0_over_G1'):
        return f"{r['G0_over_G1']:.2f}"
    s0, s1 = r['G0'].get('memory_status'), r['G1'].get('memory_status')
    if s1 == 'nonconverged' or s0 == 'nonconverged':
        return 'nc'
    if s1 in ('resource_limited', 'infrastructure_failed'):
        return 'mem'
    if s0 == 'accuracy_failed':
        return 'acc'
    return '--'


for cid in cases_sorted:
    r = CARDS['v100'].get(cid) or next((CARDS[c][cid] for c in ORDER if cid in CARDS.get(c, {})), None)
    memcell = '--'
    rv = CARDS['v100'].get(cid)
    if rv and rv['G0'].get('gpu_peak_gb') and rv['G1'].get('gpu_peak_gb'):
        memcell = f"{rv['G0']['gpu_peak_gb']:.1f}/{rv['G1']['gpu_peak_gb']:.1f}"
    cpu = (CARDS['a100'].get(cid) or {}).get('CPU', {}).get('median_s')
    lines.append(' & '.join([tt(cid.replace('core-', 'c-').replace('ext-', 'e-')), str(r.get('nao', '')) if r else ''] + [rcell(c, cid) for c in ORDER]
                            + [f_time(rv['G1'].get('median_s')) if rv else '--', memcell, f_time(cpu)]) + ' \\\\')
lines.append('\\end{longtable}')
write('tab_percase.tex', '\n'.join(lines) + '\n')

# ------------------------------------------------------------------------------------------------------------------------------
# SCF cycles table
cyc = aux['scf_cycles']
body = []
for s, sl in (('base', 'MINAO (baseline)'), ('atom', 'superposition of atomic densities'), ('huckel', 'extended H\\"uckel'), ('vsap', 'superposition of atomic potentials'),
              ('diis12', 'MINAO, DIIS space 12'), ('twostage', 'two-stage (coarse grid first)')):
    line = [sl]
    for size in ('small', 'medium', 'large'):
        rec = cyc.get(f'a100_{size}_{s}')
        if rec and rec['status'] == 'ok':
            line.append(f"{rec['full_quality_cycles']} / {rec['scf_total_seconds']:.1f}")
        else:
            line.append('--')
    body.append(line)
body.append(['second-order (SOSCF)', 'failed', 'failed', 'failed'])
write('tab_cycles.tex', table(['strategy', 'vitamin C', 'tamoxifen', 'Taxol'], body, 'lrrr',
                              'SCF cycles at full quality / total SCF seconds on the A100 for different starting guesses and convergence strategies (all reach '
                              'the same energy within $10^{-8}$ Eh).', 'tab:cycles'))


# ------------------------------------------------------------------------------------------------------------------------------
# earlier GB10 study (context)
g = aux.get('gb10')
if g:
    body = []
    for c in g['cases']:
        o, n = c['versions']['original'], c['versions']['current']
        body.append([tex(c['molecule']), str(c['natm']), str(c['nao']), f_time(o['medians']['workflow_seconds']), f_time(n['medians']['workflow_seconds']),
                     f"{o['medians']['workflow_seconds'] / n['medians']['workflow_seconds']:.2f}", f"{o['peak_tracked_cupy_bytes'] / 1e9:.1f}",
                     f"{n['peak_tracked_cupy_bytes'] / 1e9:.1f}"])
    write('tab_gb10.tex', table(['system', 'atoms', 'AOs', 'original (s)', 'new (s)', 'ratio', 'orig. GB', 'new GB'], body, 'lrrrrrrr',
                                'Earlier study on one NVIDIA GB10 (unified memory, CuPy fallback contractions, B3LYP/def2-SVP, three timing repetitions); shown for context. '
                                'The original code fails the CPU gradient gate on Taxol, so that ratio is not an equal-accuracy speed-up.', 'tab:gb10'))


# CPU speed-up macros (CPU time / new GPU time) over cases whose CPU time is at least 20 s
for tag, key in (('Best', 'CPUbest_over_G1'), ('Wide', 'CPUcpu48_over_G1')):
    for c in ('a100', 'v100', 'p100_16'):
        vals = [r[key] for r in CARDS.get(c, {}).values() if r.get(key) and (r.get('CPU') or {}).get('median_s', 0) >= 20]
        k = KEY[c]
        macros[f'CPUMed{tag}{k}'] = f'{st.median(vals):.0f}' if vals else '--'
        macros[f'CPUMax{tag}{k}'] = f'{max(vals):.0f}' if vals else '--'
        macros[f'CPUN{tag}{k}'] = str(len(vals))


# provisional-CPU-baseline note: shown only while the data contain a single CPU lane
lanes = max((len([v for v in (r.get('CPU_by_config') or {}).values() if v]) for r in rows), default=0)
macros['CPULaneNote'] = ('' if lanes >= 2 else
                         '\\paragraph{Provisional.} This build of the report contains only the 48-thread whole-node lane: the 24-thread, one-BLAS-thread lane '
                         'was still running, so the CPU baseline columns and the CPU speed-ups below equal the 48-thread values and overstate the speed-up of '
                         'the GPU, most strongly for mid-size systems (Table~\\ref{tab:cpuscan}). They are replaced by the best-of-lanes values when that lane is complete.')

# ------------------------------------------------------------------------------------------------------------------------------
(HERE / 'macros.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(macros.items())))
print('macros:', len(macros))
