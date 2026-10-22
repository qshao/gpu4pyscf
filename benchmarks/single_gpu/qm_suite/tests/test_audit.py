"""Audit synthetic records without loading either scientific backend."""
import copy
import json
from pathlib import Path
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import audit
import run
import summarize
from compare import compare_results


@pytest.fixture
def saved(tmp_path, monkeypatch):
    # Source hashing has its own tests. These records exercise audit/summary gates.
    monkeypatch.setattr(run, 'check_campaign', lambda campaign: None)
    case = {'id': 'water', 'task': 'energy_gradient', 'task_options': {}}
    campaign = {'sources': {'cases': [case], 'variants': {
        v: {'root': str(tmp_path / 'snapshots' / v)} for v in ('G0', 'G1')}},
        'initial_densities': {'water': {'array_sha256': 'atomic-seed'}}}
    (tmp_path / 'campaign.json').write_text(json.dumps(campaign))
    result = {'case': case, 'case_id': 'water', 'scientific_signature': run.signature(case),
        'initial_density_sha256': 'atomic-seed', 'energy_hartree': -1.,
        'gradient_hartree_per_bohr': [[0., 0., 0.]], 'converged': True,
        'task_converged': True, 'workflow_seconds': 2., 'metadata': {
            'backend': 'cpu', 'classes': {'molecule': 'pyscf.gto.mole.Mole', 'scf': 'pyscf.dft.rks.RKS',
                                        'task': 'pyscf.grad.rks.Gradients'},
            'effective_settings': {'conv_tol': 1e-10}}}
    def save(variant='CPU', pair=None, attempt=0, status=None, seconds=2.):
        mode = 'reference' if variant == 'CPU' else 'timing'
        request = {'case_id': 'water', 'variant': variant, 'mode': mode, 'attempt': attempt,
            'campaign_sha256': run.digest(tmp_path / 'campaign.json'),
            'scientific_signature': run.signature(case), 'pair_index': pair}
        d = dict(request, status=status or ('reference_ready' if variant == 'CPU' else 'qualified'),
                 backend='cpu' if variant == 'CPU' else 'gpu', process_elapsed_seconds=3.,
                 environment={'backend': 'cpu' if variant == 'CPU' else 'gpu', 'versions': {'pyscf': '2.10.0'}},
                 result=copy.deepcopy(result))
        d['result']['workflow_seconds'] = seconds
        if variant != 'CPU':
            d['result']['metadata']['backend'] = 'gpu'
            d['result']['metadata']['classes'].update(scf='gpu4pyscf.dft.rks.RKS', task='gpu4pyscf.grad.rks.Gradients')
            d['gpu_source'] = str(tmp_path / 'snapshots' / variant / 'gpu4pyscf' / '__init__.py')
            ref = tmp_path / 'jobs' / run.job_id('water', 'CPU', 'reference', 0) / 'result.json'
            d.update(reference_path=str(ref.relative_to(tmp_path)), reference_sha256=run.digest(ref),
                     comparison=compare_results(case, json.loads(ref.read_text())['result'], d['result']))
        directory = tmp_path / 'jobs' / run.job_id('water', variant, mode, attempt)
        directory.mkdir(parents=True)
        (directory / 'request.json').write_text(json.dumps(request))
        (directory / 'result.json').write_text(json.dumps(d))
        return directory / 'result.json'
    return tmp_path, save


def alter(path, mutate):
    d = json.loads(path.read_text()); mutate(d); path.write_text(json.dumps(d))


def test_valid_independent_reference_and_pair(saved):
    root, save = saved
    save(); save('G0', 0, seconds=4.); save('G1', 0)
    assert audit.audit_campaign(root)['record_count'] == 3
    entry = summarize.summarize(root)['cases'][0]
    assert entry['observed_speed_ratio'] == 2.
    assert entry['paired_ratios'] == [{'pair_index': 0, 'G0_over_G1': 2.}]


@pytest.mark.parametrize('change', [
    lambda d: d.update(backend='gpu'),
    lambda d: d['environment'].update(backend='gpu'),
    lambda d: d['environment']['versions'].pop('pyscf'),
    lambda d: d['result']['metadata']['classes'].update(scf='gpu4pyscf.dft.rks.RKS'),
    lambda d: d['result']['metadata'].pop('effective_settings'),
    lambda d: d['result'].update(initial_density_sha256='other'),
])
def test_reference_provenance_required(saved, change):
    root, save = saved
    alter(save(), change)
    with pytest.raises(ValueError): audit.audit_campaign(root)


@pytest.mark.parametrize('change', [
    lambda d: d['environment'].update(backend='cpu'),
    lambda d: d.update(gpu_source='/other/gpu4pyscf/__init__.py'),
    lambda d: d['result']['metadata'].update(backend='cpu'),
    lambda d: d['result']['metadata']['classes'].update(scf='pyscf.dft.rks.RKS'),
])
def test_gpu_provenance_required(saved, change):
    root, save = saved
    save(); alter(save('G0', 0), change)
    with pytest.raises(ValueError): audit.audit_campaign(root)


@pytest.mark.parametrize('pair', [None, -1, True, 0.5])
def test_qualified_timing_requires_integer_pair(saved, pair):
    root, save = saved
    save(); save('G0', pair)
    with pytest.raises(ValueError): audit.audit_campaign(root)


def test_duplicate_pair_rejected(saved):
    root, save = saved
    save(); save('G0', 0); save('G0', 0, attempt=1)
    with pytest.raises(ValueError): audit.audit_campaign(root)


def test_request_identity_must_match_result_and_directory(saved):
    root, save = saved
    p = save()
    request = p.with_name('request.json')
    alter(request, lambda d: d.update(attempt=99))
    with pytest.raises(ValueError): audit.audit_campaign(root)


def test_missing_request_rejected(saved):
    root, save = saved
    save().with_name('request.json').unlink()
    with pytest.raises(ValueError): audit.audit_campaign(root)


def test_unmatched_samples_never_form_speed_ratio(saved):
    root, save = saved
    save(); save('G0', 0); save('G1', 1)
    entry = summarize.summarize(root)['cases'][0]
    assert 'observed_speed_ratio' not in entry
    assert not entry['headline_eligible']
    assert entry['coverage_status'] == 'unmatched_timing'
    assert len(entry['unmatched_timings']) == 2


def test_missing_gpu_is_explicit_coverage_gap(saved):
    root, save = saved
    save()
    entry = summarize.summarize(root)['cases'][0]
    assert entry['coverage_status'] == 'gpu_not_measured'
    assert not entry['headline_eligible']
    assert audit.audit_campaign(root)['scientific_completion'] == 'not_established'


def test_expected_accuracy_failure_is_auditable(saved):
    root, save = saved
    save(); p = save('G0', 0, status='accuracy_failed')
    def fail(d):
        d['result']['gradient_hartree_per_bohr'] = [[1., 0., 0.]]
        ref = json.loads((root / d['reference_path']).read_text())
        d['comparison'] = compare_results(d['result']['case'], ref['result'], d['result'])
    alter(p, fail)
    assert audit.audit_campaign(root)['statuses']['accuracy_failed'] == 1
