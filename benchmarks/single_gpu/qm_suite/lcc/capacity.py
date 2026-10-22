"""Capacity ladder for one card: largest system each (variant, storage tier) completes. Resumable.

usage: capacity.py CARD [--dry-run]     CARD in a100 v100 p100_16 p100_12 v100_16
Rungs ascend in AO count; a (variant, tier) column stops after two consecutive resource-limited rungs.
Per rung: one timing-mode run (no allocator hook), then one memory-mode run if it completed. Results are the
lcc_profile_results/cap-<card>_<rung>_<version>_default_<mode>[_<tier>].json records of benchmarks/single_gpu/lcc_profile.py.
Exit code 3: not enough walltime left for another rung (the sbatch wrapper requeues itself).
"""
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_DIR = HERE.parents[1]                      # benchmarks/single_gpu
RESULTS = PROFILE_DIR / 'lcc_profile_results'
PACKAGES = '/scratch/qsh226/g4p_lcc/capacity_pkg'   # current -> campaign G1 snapshot, original -> campaign G0 snapshot
# rung -> (profile size name, AOs for def2-SVP, unscreened triangular CDERI GB). Screened tensors measured at 44-55% of this.
RUNGS = [('small', 208, 0.5), ('medium', 537, 3.0), ('large', 1123, 28.3), ('valinomycin', 1542, 71.6),
         ('water008', 1824, 114.4), ('gly30', 2154, 201.5), ('water009', 2400, 260.5), ('gly40', 2864, 473.7),
         ('water010', 3336, 699.4), ('gly50', 3574, 920.6), ('gly60', 4284, 1585.5), ('gly70', 4994, 2511.8),
         ('gly80', 5704, 3742.6), ('gly90', 6414, 5321.5), ('gly100', 7124, 7291.6), ('gly110', 7834, 9696.2),
         ('gly120', 8544, 12578.8)]
COLUMNS = [('original', 'gpu'), ('original', 'host'), ('current', 'gpu'), ('current', 'host'), ('current', 'disk')]
CARDS = {  # host RAM available to the job (GB) for admission; scratch is the shared filesystem
    'a100': 240, 'v100': 180, 'v100_16': 180, 'p100_16': 115, 'p100_12': 180}
# Screened/unscreened CDERI size, measured from host RSS of completed host-tier runs; later rungs of a family reuse the last value.
SCREENED_BY_RUNG = {'small': 1.0, 'medium': 0.8, 'large': 0.59, 'valinomycin': 0.46, 'water008': 0.29, 'gly30': 0.12, 'water009': 0.24}
FAMILY_DEFAULT = {'water': 0.24, 'gly': 0.12}
RETRY_LIMIT = 3             # budget raises per rung
BUDGET_FRACTION_OF_DEVICE = 0.35
RUN_TIMEOUT = 24 * 3600
DISK_RUN_CAP_GB = 1500      # one disk-tier tensor (account quota is 25 TB)
DISK_TOTAL_CAP_GB = 4000    # all temporary integral files at once; /scratch/qsh226 already holds 9.2 TB of the 25 TB quota
TMP = Path(os.environ.get('QM_SUITE_SCRATCH', '/scratch/qsh226/tmp_cderi'))
FAIL_WORDS = ('out of memory', 'Not enough GPU memory', 'MemoryError', 'cudaErrorMemoryAllocation', 'Cannot allocate',
              'Gradient workspace budget', 'minimum build staging requires', 'minimum shell-pair build workspace requires')


def result_path(card, rung, version, mode, tier):
    suffix = '' if tier == 'gpu' else '_' + tier
    return RESULTS / f'cap-{card}_{rung}_{version}_default_{mode}{suffix}.json'


def classify(path):
    """-> (status, detail). 'running' is a record left by a killed process."""
    if not path.exists():
        return None, ''
    data = json.loads(path.read_text())
    status, error = data.get('status'), data.get('error') or ''
    if status == 'running':
        return 'killed', 'process died before writing a result (host memory or walltime)'
    if status in ('passed', 'capacity_observation', 'failed_accuracy'):
        return status, ''
    if status == 'out_of_memory' or any(w in error for w in FAIL_WORDS):
        return 'resource_limited', error[:120]
    return 'failed', error[:120]


def remaining_seconds():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return float('inf')
    out = subprocess.run(['squeue', '-h', '-j', job, '-o', '%L'], capture_output=True, text=True).stdout.strip()
    days, _, rest = out.partition('-') if '-' in out else ('0', '', out)
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def screened_ratio(rung):
    return SCREENED_BY_RUNG.get(rung, FAMILY_DEFAULT['water' if rung.startswith('water') else 'gly'])


def admission(card, tier, gb_unscreened, rung):
    need = screened_ratio(rung) * gb_unscreened
    if tier == 'host' and need + 8 > 0.85 * CARDS[card]:
        return f'estimated host tensor {need:.0f} GB + overhead exceeds 85% of the node memory ({CARDS[card]} GB)'
    if tier == 'disk':
        if need > DISK_RUN_CAP_GB:
            return f'estimated scratch file {need:.0f} GB exceeds the {DISK_RUN_CAP_GB} GB per-run cap (25 TB account quota)'
        out = subprocess.run(['du', '-s', '--block-size=1G', str(TMP)], capture_output=True, text=True).stdout.split()
        used = float(out[0]) if out else 0.
        if used + need > DISK_TOTAL_CAP_GB:
            return f'temporary integral files already use {used:.0f} GB; adding {need:.0f} GB exceeds {DISK_TOTAL_CAP_GB} GB'
    return None


def run(cmd, env, timeout):
    process = subprocess.Popen(cmd, env=env, cwd=PROFILE_DIR, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 'timeout'


def run_with_budget_retry(card, rung, version, tier, env):
    """Run one timing-mode rung; if it fails only because a build/staging budget is below that rung's stated minimum, raise
    that budget (1.2x the stated need, at most BUDGET_FRACTION_OF_DEVICE of the card) and retry. Failed attempts are kept
    as <record>.attemptN.json. Returns (status, detail, env_used)."""
    tpath = result_path(card, rung, version, 'time', tier)
    env = dict(env)
    status, detail, code = None, '', 0
    for attempt in range(RETRY_LIMIT + 1):
        cmd = [sys.executable, 'lcc_profile.py', f'cap-{card}', rung, version, 'default', 'time', tier]
        code = run(cmd, env, RUN_TIMEOUT)
        status, detail = classify(tpath)
        if code == 'timeout':
            status, detail = 'timeout', f'exceeded {RUN_TIMEOUT // 3600} h'
        elif status is None:
            status, detail = 'failed', f'no result record (exit {code})'
        data = json.loads(tpath.read_text()) if tpath.exists() else {}
        match = re.search(r'(build staging|shell-pair build workspace) requires (\d+) bytes', data.get('error') or '')
        if status in ('failed', 'resource_limited') and match and attempt < RETRY_LIMIT:
            new = math.ceil(1.2 * int(match.group(2)) / 1024**3) * 1024**3
            if new > BUDGET_FRACTION_OF_DEVICE * (data.get('device_total_bytes') or 0):
                detail = f'{match.group(1)} needs {int(match.group(2)) / 1e9:.1f} GB: more than {BUDGET_FRACTION_OF_DEVICE:.0%} of the card'
                status = 'resource_limited'
                break
            env['LCC_STAGING_BYTES' if 'staging' in match.group(1) else 'LCC_BUILD_BYTES'] = str(new)
            tpath.rename(tpath.with_suffix(f'.attempt{attempt}.json'))
            print(f'RETRY {rung}/{version}/{tier} with {match.group(1)} budget {new / 1e9:.1f} GB', flush=True)
            continue
        break
    return status, detail, env


def main():
    card = sys.argv[1]
    dry = '--dry-run' in sys.argv
    assert card in CARDS
    env = dict(os.environ, LCC_PACKAGES=PACKAGES, OMP_NUM_THREADS='4')
    plan = {}
    for version, tier in COLUMNS:
        consecutive = 0
        for rung, nao, gb in RUNGS:
            key = f'{version}/{tier}/{rung}'
            tpath = result_path(card, rung, version, 'time', tier)
            status, detail = classify(tpath)
            if status is None:
                reason = admission(card, tier, gb, rung)
                if reason:
                    status, detail = 'admission_rejected', reason
                elif dry:
                    plan[key] = 'would run'
                    continue
                else:
                    if remaining_seconds() < RUN_TIMEOUT + 3600:
                        print('not enough walltime left for another rung; requeue', flush=True)
                        sys.exit(3)
                    print('RUN', key, flush=True)
                    status, detail, used_env = run_with_budget_retry(card, rung, version, tier, env)
                    if status in ('passed', 'capacity_observation', 'failed_accuracy'):
                        mpath = result_path(card, rung, version, 'memory', tier)
                        if classify(mpath)[0] is None and remaining_seconds() > RUN_TIMEOUT + 3600:
                            run([sys.executable, 'lcc_profile.py', f'cap-{card}', rung, version, 'default', 'memory', tier],
                                used_env, RUN_TIMEOUT)
            plan[key] = f'{status} {detail}'.strip()
            if status in ('resource_limited', 'killed', 'timeout', 'admission_rejected'):
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= 2:
                plan[f'{version}/{tier}/(stop)'] = 'two consecutive resource-limited rungs'
                break
    print(json.dumps(plan, indent=1))
    (RESULTS / f'cap-{card}_ladder_status.json').write_text(json.dumps(plan, indent=1))


if __name__ == '__main__':
    main()
