"""Recompute scientific gates and verify immutable records without a GPU."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def _integer(value):
    return type(value) is int and value >= 0


def _request(record, path, campaign_hash, case):
    from run import job_id, signature
    request_path = path.with_name('request.json')
    if not request_path.is_file():
        raise ValueError('missing immutable request: ' + str(path))
    request = json.loads(request_path.read_text())
    keys = {'case_id', 'variant', 'mode', 'attempt', 'campaign_sha256',
            'scientific_signature', 'pair_index'}
    if set(request) != keys or any(record.get(k) != request[k] for k in keys):
        raise ValueError('request/result identity mismatch: ' + str(path))
    if not _integer(request['attempt']):
        raise ValueError('invalid attempt')
    if (request['campaign_sha256'] != campaign_hash or request['case_id'] != case['id']
            or request['scientific_signature'] != signature(case)):
        raise ValueError('campaign/scientific request identity mismatch')
    if path.parent.name != job_id(case['id'], request['variant'], request['mode'], request['attempt']):
        raise ValueError('request/directory identity mismatch')
    if request['mode'] not in {'reference', 'capability', 'timing', 'memory', 'profile'}:
        raise ValueError('invalid measurement mode')


def _provenance(record, case, campaign, require_converged=True):
    from run import signature
    result = record['result']
    if (result.get('case') != case or result.get('case_id') != case['id']
            or result.get('scientific_signature') != signature(case)):
        raise ValueError('scientific input identity mismatch')
    expected_seed = campaign.get('initial_densities', {}).get(case['id'], {}).get('array_sha256')
    if not expected_seed or result.get('initial_density_sha256') != expected_seed:
        raise ValueError('initial density identity mismatch')
    metadata = result.get('metadata', {})
    environment = record.get('environment', {})
    version = environment.get('versions', {}).get('pyscf')
    if not isinstance(version, str) or not version.strip():
        raise ValueError('missing CPU PySCF implementation version')
    if not metadata.get('effective_settings'):
        raise ValueError('missing effective scientific settings')
    classes = metadata.get('classes', {})
    if not str(classes.get('molecule', '')).startswith('pyscf.'):
        raise ValueError('missing molecule implementation identity')
    if record['variant'] == 'CPU':
        if record['mode'] not in ('reference', 'timing') or record.get('backend') != 'cpu':
            raise ValueError('CPU reference identity mismatch')
        expected_backend = 'cpu'
        if not str(classes.get('scf', '')).startswith('pyscf.'):
            raise ValueError('CPU reference used non-CPU implementation')
        if 'task' in classes and not str(classes['task']).startswith('pyscf.'):
            raise ValueError('CPU task implementation mismatch')
    else:
        expected_backend = 'gpu'
        if record['variant'] not in ('G0', 'G1') or record.get('backend') != 'gpu':
            raise ValueError('GPU variant identity mismatch')
        if not str(classes.get('scf', '')).startswith('gpu4pyscf.'):
            raise ValueError('GPU record used non-GPU SCF')
        task_class = str(classes.get('task', ''))
        if task_class and not task_class.startswith('gpu4pyscf.'):
            allowed_hybrid = case['task'] == 'optimization' and task_class == 'pyscf.geomopt.geometric_solver'
            if not allowed_hybrid:
                raise ValueError('GPU task implementation mismatch')
        snapshot = Path(campaign['sources']['variants'][record['variant']]['root']).resolve()
        package = snapshot if snapshot.name == 'gpu4pyscf' else snapshot / 'gpu4pyscf'
        if Path(record.get('gpu_source', '')).resolve() != package / '__init__.py':
            raise ValueError('GPU import path differs from frozen snapshot')
    if metadata.get('backend') != expected_backend or environment.get('backend') != expected_backend:
        raise ValueError('environment/observable backend mismatch')
    if require_converged and (result.get('converged') is not True or result.get('task_converged') is not True):
        raise ValueError('nonconverged successful record')
    seconds = result.get('workflow_seconds')
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid workflow time')


def audit_campaign(output):
    from run import check_campaign, digest
    from compare import compare_results
    output = Path(output).resolve()
    campaign = json.loads((output / 'campaign.json').read_text())
    check_campaign(campaign)
    campaign_hash = digest(output / 'campaign.json')
    cases = {c['id']: c for c in campaign['sources']['cases']}
    loaded = {}
    counts, records, pairs = Counter(), [], set()
    for path in sorted((output / 'jobs').glob('*/result.json')):
        record = json.loads(path.read_text())
        if record.get('case_id') not in cases:
            raise ValueError('unknown recorded case')
        case = cases[record['case_id']]
        _request(record, path, campaign_hash, case)
        seconds = record.get('process_elapsed_seconds')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid elapsed time')
        successful = record['status'] in ('qualified', 'reference_ready', 'cpu_timed')
        if successful or 'comparison' in record:
            _provenance(record, case, campaign, require_converged=successful)
        if record['status'] == 'reference_ready':
            if record['variant'] != 'CPU' or record['mode'] != 'reference':
                raise ValueError('reference must be independently computed on CPU')
            comparison = compare_results(case, record['result'], record['result'])
            if not comparison['passed']:
                raise ValueError('CPU reference observable validation failed')
            if 'reference_self_check' in record and record['reference_self_check'] != comparison:
                raise ValueError('CPU reference self-check changed')
        if record['status'] == 'cpu_timed':
            if record['variant'] != 'CPU' or record['mode'] != 'timing':
                raise ValueError('cpu_timed result must be a CPU timing sample')
            if any(k in record for k in ('phase_memory', 'host_samples', 'profile')):
                raise ValueError('instrumented timing record')
            if not _integer(record.get('pair_index')):
                raise ValueError('CPU timing requires nonnegative integer repetition index')
            slot = (case['id'], 'CPU', record['pair_index'])
            if slot in pairs:
                raise ValueError('duplicate CPU timing slot')
            pairs.add(slot)
            if 'comparison' not in record:
                raise ValueError('missing reproducibility comparison with the independent reference')
        if record['status'] == 'qualified':
            if record['variant'] not in ('G0', 'G1'):
                raise ValueError('qualified result must be a GPU variant')
            if record['mode'] == 'timing':
                if any(k in record for k in ('phase_memory', 'host_samples', 'profile')):
                    raise ValueError('instrumented timing record')
                if not _integer(record.get('pair_index')):
                    raise ValueError('qualified timing requires nonnegative integer pair index')
                pair = (case['id'], record['variant'], record['pair_index'])
                if pair in pairs:
                    raise ValueError('duplicate qualified timing pair slot')
                pairs.add(pair)
            if record['mode'] == 'memory':
                peak = record.get('phase_memory', {}).get('peak_live_bytes')
                if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak < 0:
                    raise ValueError('missing allocation evidence')
            if 'comparison' not in record:
                raise ValueError('missing independent CPU comparison')
        loaded[path.resolve()] = record
    for path, record in loaded.items():
        if 'comparison' in record:
            reference_path = (output / record['reference_path']).resolve()
            reference = loaded.get(reference_path)
            if reference is None or reference['status'] != 'reference_ready':
                raise ValueError('reference is not an audited independent CPU job')
            if digest(reference_path) != record['reference_sha256']:
                raise ValueError('CPU reference changed')
            case = cases[record['case_id']]
            if reference['case_id'] != case['id']:
                raise ValueError('reference case differs')
            comparison = compare_results(case, reference['result'], record['result'])
            if comparison != record['comparison']:
                raise ValueError('stored comparison statistics changed')
            if (record['status'] in ('qualified', 'cpu_timed')) != bool(comparison['passed']):
                raise ValueError('comparison disagrees with qualification status')
            if not comparison['passed'] and record['status'] != comparison['qualification_status']:
                raise ValueError('failed control classification changed')
        counts[record['status']] += 1
        records.append({'path': str(path.relative_to(output)), 'sha256': digest(path), 'status': record['status']})
    return {'status': 'audit_passed',
            'scientific_completion': 'not_established' if not counts['qualified'] else 'partial_or_complete_per_coverage',
            'campaign_sha256': campaign_hash, 'case_count': len(cases), 'record_count': len(records),
            'statuses': dict(counts), 'records': records,
            'note': 'Audit success is not GPU benchmark success. Missing/unsupported/resource-limited cases remain explicit; CPU references are not GPU timing samples.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(audit_campaign(parser.parse_args().output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
