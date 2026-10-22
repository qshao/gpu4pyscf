"""Audit each card's campaign fork and build the CPU / G0 / G1 comparison tables.

usage: compare_variants.py MASTER_NAME OUT_PREFIX [--card a100=campaign1-lcc-a100 ...] [--cpu campaign1-lcc-cpu]
Every fork is audited first (hashes, identities, recomputed accuracy); numbers come only from audited records.
Per (card, case): G0 and G1 qualified timing medians, tracked-GPU-memory peaks, accuracy errors, terminal statuses;
CPU timing medians from the CPU fork. Family summaries use geometric means over cases where both variants are qualified,
with the denominator and the excluded cases listed. No cross-family average.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = SUITE / 'results'


def load_fork(name):
    from audit_tolerant import audit_campaign_tolerant
    root = RESULTS / name
    audit = audit_campaign_tolerant(root)
    records = [json.loads(p.read_text()) for p in sorted((root / 'jobs').glob('*/result.json'))]
    campaign = json.loads((root / 'campaign.json').read_text())
    cases = {c['id']: c for c in campaign['sources']['cases']}
    return audit, records, cases


def median(values):
    return statistics.median(values) if values else None


def summarize_fork(records, cases):
    out = {}
    for cid, case in cases.items():
        row = dict(family=case['family'], task=case['task'], group=case['group'], system=case.get('system'),
                   basis=case.get('basis'), charge=case.get('charge'), spin=case.get('spin'))
        for r in records:                       # size information is stored by every successful run in result.metadata.state
            state = ((r.get('result') or {}).get('metadata') or {}).get('state') if r['case_id'] == cid else None
            if state:
                row.update({k: state[k] for k in ('nao', 'nocc', 'naux', 'nelectron') if k in state})
                break
        for variant in ('G0', 'G1', 'CPU'):
            timing = [r for r in records if r['case_id'] == cid and r['variant'] == variant and r['mode'] == 'timing'
                      and r['status'] in ('qualified', 'cpu_timed')]
            samples = [r['result']['workflow_seconds'] for r in timing]
            memory = [r for r in records if r['case_id'] == cid and r['variant'] == variant and r['mode'] == 'memory']
            status = memory[-1]['status'] if memory else None
            row[variant] = dict(n=len(samples), median_s=median(samples), samples_s=samples,
                                min_s=min(samples) if samples else None, max_s=max(samples) if samples else None,
                                mad_s=median([abs(x - median(samples)) for x in samples]) if samples else None,
                                memory_status=status)
            statuses = [r['status'] for r in records if r['case_id'] == cid and r['variant'] == variant and r['mode'] == 'memory']
            row[variant]['memory_statuses'] = statuses
            if memory and memory[-1].get('error'):
                row[variant]['error'] = str(memory[-1]['error'])[:160]
            if memory and memory[-1].get('environment', {}).get('contract_engine'):
                row[variant]['engine'] = memory[-1]['environment']['contract_engine']
            if memory and memory[-1].get('environment', {}).get('device'):
                row[variant]['device'] = memory[-1]['environment']['device']
            if memory and memory[-1].get('phase_memory'):
                row[variant]['gpu_peak_gb'] = memory[-1]['phase_memory']['peak_live_bytes'] / 1e9
            if memory and memory[-1].get('comparison'):
                errs = memory[-1]['comparison'].get('errors', {})
                row[variant]['errors'] = {k: v.get('max_abs') for k, v in errs.items()}
        out[cid] = row
    return out


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def family_table(per_card):
    rows = []
    for card, cases in per_card.items():
        fam = defaultdict(list)
        excluded = defaultdict(list)
        for cid, row in cases.items():
            g0, g1 = row.get('G0', {}), row.get('G1', {})
            if g0.get('median_s') and g1.get('median_s'):
                fam[row['family']].append((cid, g0['median_s'] / g1['median_s']))
            else:
                excluded[row['family']].append(cid)
        for family in sorted(set(fam) | set(excluded)):
            ratios = [r for _, r in fam[family]]
            rows.append(dict(card=card, family=family, shared_qualified=len(ratios),
                             geomean_G0_over_G1=geomean(ratios), min=min(ratios) if ratios else None,
                             max=max(ratios) if ratios else None, excluded=excluded[family]))
    return rows


def main():
    master, prefix = sys.argv[1], sys.argv[2]
    forks = {}
    args = sys.argv[3:]
    for i, a in enumerate(args):
        if a == '--card':
            k, v = args[i + 1].split('=')
            forks[k] = v
        if a == '--cpu':        # --cpu [label=]fork, repeatable; label defaults to cpu48 (the whole-node 48-thread lane)
            lab, _, fk = args[i + 1].rpartition('=')
            forks['cpu_' + (lab or 'cpu48')] = fk
    per_card, audits = {}, {}
    for card, name in forks.items():
        audits[card], records, cases = load_fork(name)
        per_card[card] = summarize_fork(records, cases)
    gpu = {c: v for c, v in per_card.items() if not c.startswith('cpu_')}
    cpus = {c[4:]: v for c, v in per_card.items() if c.startswith('cpu_')}
    table = []
    for card, cases in gpu.items():
        for cid, row in cases.items():
            samples = {lab: cpus[lab].get(cid, {}).get('CPU', {}) for lab in cpus}
            timed = {lab: v for lab, v in samples.items() if v.get('median_s')}
            best_lab = min(timed, key=lambda lab: timed[lab]['median_s']) if timed else None
            c = timed[best_lab] if best_lab else {}
            entry = dict(card=card, case=cid, family=row['family'], task=row['task'],
                         G0=row['G0'], G1=row['G1'], CPU=c, CPU_best_config=best_lab,
                         CPU_by_config={lab: v.get('median_s') for lab, v in samples.items()})
            entry.update({k: row.get(k) for k in ('system', 'basis', 'charge', 'spin', 'nao', 'nocc', 'naux', 'nelectron')})
            if row['G0'].get('median_s') and row['G1'].get('median_s'):
                entry['G0_over_G1'] = row['G0']['median_s'] / row['G1']['median_s']
            for ref_label, ref in (('best', c), ('cpu48', samples.get('cpu48', {}))):
                if ref.get('median_s') and row['G1'].get('median_s'):
                    entry[f'CPU{ref_label}_over_G1'] = ref['median_s'] / row['G1']['median_s']
                if ref.get('median_s') and row['G0'].get('median_s'):
                    entry[f'CPU{ref_label}_over_G0'] = ref['median_s'] / row['G0']['median_s']
            table.append(entry)
    result = dict(master=master, audits={k: {kk: v[kk] for kk in ('campaign_sha256', 'record_count', 'statuses')}
                                         for k, v in audits.items()},
                  per_case=table, families=family_table(gpu))
    Path(prefix + '.json').write_text(json.dumps(result, indent=1))
    lines = ['# CPU vs original GPU (G0) vs new GPU (G1)', '']
    for card in gpu:
        lines += [f'## {card}', '', '| family | shared qualified | geomean G0/G1 | range | excluded |', '|---|---:|---:|---|---|']
        for r in (x for x in result['families'] if x['card'] == card):
            lines.append('| %s | %d | %s | %s | %s |' % (
                r['family'], r['shared_qualified'],
                '%.2fx' % r['geomean_G0_over_G1'] if r['geomean_G0_over_G1'] else '-',
                '%.2f-%.2f' % (r['min'], r['max']) if r['min'] else '-', ', '.join(r['excluded'][:4]) + ('...' if len(r['excluded']) > 4 else '')))
        lines.append('')
    Path(prefix + '.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
