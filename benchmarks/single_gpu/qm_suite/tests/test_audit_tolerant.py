"""The tolerant audit equality: accepts last-digit noise, rejects every real change."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'lcc'))
from audit_tolerant import close, ToleranceDict


def test_accepts_last_digit_noise():
    a = {'passed': True, 'diag': {'overlap': 0.9999999999999993, 'rms': 1.599e-16}, 'map': [0, 1, 2]}
    b = {'passed': True, 'diag': {'overlap': 0.9999999999999996, 'rms': 1.618e-16}, 'map': [0, 1, 2]}
    assert ToleranceDict(a) == b and b == ToleranceDict(a) and not (ToleranceDict(a) != b)


def test_rejects_real_changes():
    base = {'passed': True, 'err': 1e-9, 'map': [0, 1], 'name': 'x'}
    for changed in ({'passed': False, 'err': 1e-9, 'map': [0, 1], 'name': 'x'},
                    {'passed': True, 'err': 1.1e-9, 'map': [0, 1], 'name': 'x'},
                    {'passed': True, 'err': 1e-9, 'map': [1, 0], 'name': 'x'},
                    {'passed': True, 'err': 1e-9, 'map': [0, 1], 'name': 'y'},
                    {'passed': True, 'err': 1e-9, 'map': [0, 1]}):
        assert ToleranceDict(base) != changed
        assert changed != ToleranceDict(base)
    assert not close({'v': float('nan')}, {'v': 1.0}) and close({'v': float('nan')}, {'v': float('nan')})


def test_admission_rejection_without_request_is_accepted_only_when_it_carries_a_forecast(tmp_path, monkeypatch):
    import json
    import audit
    import run
    import audit_tolerant
    monkeypatch.setattr(run, 'check_campaign', lambda campaign: None)
    case = {'id': 'water', 'task': 'energy_gradient', 'task_options': {}}
    (tmp_path / 'campaign.json').write_text(json.dumps({'sources': {'cases': [case], 'variants': {}}, 'initial_densities': {}}))
    chash = run.digest(tmp_path / 'campaign.json')

    def write(name, record):
        d = tmp_path / 'jobs' / name
        d.mkdir(parents=True)
        (d / 'result.json').write_text(json.dumps(record))

    base = {'status': 'resource_limited', 'case_id': 'water', 'variant': 'G1', 'mode': 'memory', 'attempt': 0, 'pair_index': None,
            'campaign_sha256': chash, 'process_elapsed_seconds': 0., 'admission': {'admitted': False}}
    write(run.job_id('water', 'G1', 'memory', 0), base)
    assert audit_tolerant.audit_campaign_tolerant(tmp_path)['statuses'] == {'resource_limited': 1}
    # a record that is not an admission rejection still needs its request
    bad = tmp_path / 'jobs' / run.job_id('water', 'G1', 'memory', 1)
    bad.mkdir()
    (bad / 'result.json').write_text(json.dumps(dict(base, attempt=1, admission=None) | {'status': 'qualified'}))
    import pytest
    with pytest.raises(ValueError):
        audit_tolerant.audit_campaign_tolerant(tmp_path)
