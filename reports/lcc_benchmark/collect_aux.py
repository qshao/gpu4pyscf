"""Collect the auxiliary datasets of the LCC benchmark report into data/aux.json.

Every number in the report comes from a JSON file written by a benchmark job; this script only gathers them.
"""
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SG = Path('/scratch/qsh226/gpu4pyscf/benchmarks/single_gpu')
PROFILE = SG / 'lcc_profile_results'
RESULTS = SG / 'qm_suite/results'
OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)


def load(path):
    return json.loads(Path(path).read_text())


def maybe(path):
    return load(path) if os.path.exists(path) else None


aux = {}

# ---- 1. stage profile of Taxol (cuTENSOR engine): SCF and gradient stages -------------------------------------------------
stages = {}
for card in ('a100', 'v100'):
    for version in ('original', 'current'):
        d = maybe(PROFILE / f'{card}_large_{version}_default_time.json')
        if d and d.get('status') in ('passed', 'failed_accuracy'):
            stages[f'{card}/{version}'] = dict(scf=d['scf_seconds'], gradient=d['gradient_seconds'], total=d['workflow_seconds'],
                                                stages={k: v['seconds'] for k, v in d['stages'].items()}, status=d['status'])
aux['taxol_stages'] = stages

# ---- 2. gradient workspace sweep -------------------------------------------------------------------------------------------
sweep = {}
for card in ('a100', 'v100'):
    for ws in ('default', '1', '4', '8', '16', 'unlimited'):
        d = maybe(PROFILE / f'{card}_large_current_{ws}_time.json')
        if d and d.get('status') == 'passed':
            sweep[f'{card}/{ws}'] = dict(gradient=d['gradient_seconds'], total=d['workflow_seconds'],
                                         aux_batch=(d.get('workspace_plans') or [{}])[-1].get('batch_aux'))
aux['workspace_sweep'] = sweep

# ---- 3. storage tiers (Taxol, valinomycin) ----------------------------------------------------------------------------------
tiers = {}
for card in ('v100', 'a100', 'v100-review', 'a100-inplace', 'p100-12gb-gram', 'p100-16gb-gram', 'p100-12gb', 'p100-16gb'):
    for size in ('large', 'valinomycin'):
        for version in ('original', 'current'):
            for tier in ('gpu', 'host', 'disk'):
                suf = '' if tier == 'gpu' else '_' + tier
                t = maybe(PROFILE / f'{card}_{size}_{version}_default_time{suf}.json')
                if not t:
                    continue
                m = maybe(PROFILE / f'{card}_{size}_{version}_default_memory{suf}.json') or {}
                mem = {k: v['peak_live_bytes'] / 1e9 for k, v in (m.get('memory') or {}).items()}
                tiers[f'{card}/{size}/{version}/{tier}'] = dict(
                    status=t['status'], seconds=t.get('workflow_seconds'), scf=t.get('scf_seconds'), gradient=t.get('gradient_seconds'),
                    gpu_peak_scf_gb=mem.get('scf'), gpu_peak_gradient_gb=mem.get('gradient'),
                    host_rss_gb=(m.get('host_peak_rss_bytes') or t.get('host_peak_rss_bytes') or 0) / 1e9 or None,
                    error=(t.get('error') or '')[:120],
                    gradient_error=(t.get('comparison') or {}).get('gradient_max_error'))
aux['storage_tiers'] = tiers

# ---- 4. Tamoxifen stage profile by engine -----------------------------------------------------------------------------------
engine = {}
for card in ('a100', 'v100'):
    for eng in ('cutensor', 'cupy'):
        for version in ('original', 'current'):
            rows = []
            for r in (1, 2, 3):
                d = maybe(PROFILE / f'{card}-{eng}-r{r}_medium_{version}_default_time.json')
                if d and d.get('status') == 'passed':
                    rows.append(dict(total=d['workflow_seconds'], scf=d['scf_seconds'], gradient=d['gradient_seconds'],
                                     stages={k: v['seconds'] for k, v in d['stages'].items()}))
            if rows:
                engine[f'{card}/{eng}/{version}'] = rows
aux['tamoxifen_engine_profile'] = engine

# ---- 5. SCF cycle experiments, grid split, GEMM microbenchmark -----------------------------------------------------------------
cycles = {}
for f in sorted(glob.glob(str(SG / 'lcc_scf_cycle_results/*.json'))):
    d = load(f)
    key = Path(f).stem
    cycles[key] = {k: d.get(k) for k in ('status', 'full_quality_cycles', 'total_cycles', 'scf_total_seconds', 'guess_seconds',
                                         'energy_error_hartree', 'accepted', 'prestage', 'error')}
aux['scf_cycles'] = cycles
grid = {}
for f in sorted(glob.glob(str(PROFILE / '*-grid-r*_current_default_time.json'))):
    d = load(f)
    if d.get('grid_info'):
        grid[Path(f).stem] = dict(grid=d['grid_info'], xc_seconds=d['stages']['scf.xc_quadrature']['seconds'],
                                  xc_calls=d['stages']['scf.xc_quadrature']['calls'], scf=d['scf_seconds'])
aux['grid_split'] = grid
aux['gemm'] = {c: maybe(SG / f'lcc_gemm_results/{c}.json') for c in ('a100', 'v100')}

# ---- 6. CPU thread-scaling experiment -----------------------------------------------------------------------------------------
tasks = {}
for fn in ('campaign1-cpuscan-tasks.txt', 'campaign1-cpuscan-small-tasks.txt'):
    for line in open(RESULTS / fn):
        case, idx, omp, blas, bind, name = line.split()
        tasks[int(idx)] = dict(case=case, omp=int(omp), blas=int(blas), bind=bind, name=name)
scan = []
for p in glob.glob(str(RESULTS / 'campaign1-lcc-cpuscan/jobs/*__CPU__timing__*/result.json')):
    d = load(p)
    t = tasks.get(d['pair_index'])
    if t and d['status'] in ('cpu_timed', 'accuracy_failed'):
        scan.append(dict(t, seconds=d['result']['workflow_seconds'], status=d['status']))
aux['cpu_scan'] = scan

# ---- 7. Cold start ---------------------------------------------------------------------------------------------------------------
cold = {}
cases = ['core-h2o-b3lyp-svp', 'core-vitamin_c-b3lyp-svp', 'core-tamoxifen-b3lyp-svp']
for card, engines in (('a100', ['cutensor', 'cupy']), ('v100', ['cutensor', 'cupy']), ('p100_16', ['cupy']), ('p100_12', ['cupy'])):
    rows = []
    for p in glob.glob(str(RESULTS / f'campaign1-lcc-coldstart-{card}/jobs/*__timing__*/result.json')):
        d = load(p)
        if d['mode'] != 'timing' or d['status'] != 'qualified':
            continue
        i = d['pair_index']
        c = i % 3
        s = (i // 3) % 2
        v = (i // 6) % 2
        e = i // 12
        rows.append(dict(engine=engines[e], variant=('G0', 'G1')[v], state=('cold', 'warm')[s], case=cases[c],
                         seconds=d['result']['workflow_seconds'], process_seconds=d['process_elapsed_seconds']))
    cold[card] = rows
aux['cold_start'] = cold

# ---- 8. Capacity ladders -----------------------------------------------------------------------------------------------------------
cap = {}
import re
pattern = re.compile(r'^cap-(a100|v100_16|v100|p100_16|p100_12)_([a-z0-9]+)_(original|current)_default_time(?:_(host|disk))?$')
for f in glob.glob(str(PROFILE / 'cap-*_default_time*.json')):
    m = pattern.match(Path(f).stem)
    if not m:
        continue
    card, rung, version, tier = m.group(1), m.group(2), m.group(3), m.group(4) or 'gpu'
    d = load(f)
    mem = maybe(Path(f).with_name(Path(f).name.replace('_time', '_memory')))
    peaks = {k: v['peak_live_bytes'] / 1e9 for k, v in ((mem or {}).get('memory') or {}).items()}
    cap[f'{card}/{version}/{tier}/{rung}'] = dict(
        status=d['status'], seconds=d.get('workflow_seconds'), scf=d.get('scf_seconds'), gradient=d.get('gradient_seconds'),
        cycles=d.get('cycles'), peaks_gb=peaks, host_rss_gb=((mem or d).get('host_peak_rss_bytes') or 0) / 1e9 or None,
        error=(d.get('error') or '')[:140], build_budget=d.get('build_budget_bytes'), staging_budget=d.get('staging_budget_bytes'))
aux['capacity'] = cap

# ---- 9. Earlier GB10 comparison (context only) -------------------------------------------------------------------------------------
gb10 = maybe(SG / 'original_comparison/summary.json')
aux['gb10'] = gb10

(OUT / 'aux.json').write_text(json.dumps(aux, indent=1, default=str))
print({k: (len(v) if hasattr(v, '__len__') else v) for k, v in aux.items()})
