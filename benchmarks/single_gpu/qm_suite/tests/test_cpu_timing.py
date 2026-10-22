"""CPU timing lane: audit, summary and run-level gating without either scientific backend."""
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
from test_audit import saved, alter  # noqa: F401  (fixtures reused from the GPU audit tests)


def save_cpu_timing(root, repetition, seconds=100., attempt=None, status='cpu_timed', threads='48'):
    case = json.loads((root / 'campaign.json').read_text())['sources']['cases'][0]
    ref = root / 'jobs' / run.job_id('water', 'CPU', 'reference', 0) / 'result.json'
    reference = json.loads(ref.read_text())
    attempt = repetition if attempt is None else attempt
    request = {'case_id': 'water', 'variant': 'CPU', 'mode': 'timing', 'attempt': attempt,
               'campaign_sha256': run.digest(root / 'campaign.json'),
               'scientific_signature': run.signature(case), 'pair_index': repetition}
    d = copy.deepcopy(reference)
    d.update(request, status=status, process_elapsed_seconds=seconds + 2,
             reference_path=str(ref.relative_to(root)), reference_sha256=run.digest(ref))
    d.pop('reference_self_check', None)
    d['environment']['threads'] = {'OMP_NUM_THREADS': threads}
    d['result']['workflow_seconds'] = seconds
    d['comparison'] = compare_results(case, reference['result'], d['result'])
    directory = root / 'jobs' / run.job_id('water', 'CPU', 'timing', attempt)
    directory.mkdir(parents=True)
    (directory / 'request.json').write_text(json.dumps(request))
    (directory / 'result.json').write_text(json.dumps(d))
    return directory / 'result.json'


def test_cpu_timing_samples_are_audited_and_summarized(saved):
    root, save = saved
    save(); save('G0', 0, seconds=4.); save('G1', 0, seconds=2.)
    for rep, seconds in enumerate((100., 110., 90.)):
        save_cpu_timing(root, rep, seconds)
    assert audit.audit_campaign(root)['statuses']['cpu_timed'] == 3
    entry = summarize.summarize(root)['cases'][0]
    assert entry['timing']['CPU']['median_seconds'] == 100. and entry['timing']['CPU']['n'] == 3
    assert entry['timing']['CPU']['threads'] == ["{'OMP_NUM_THREADS': '48'}"]
    assert entry['CPU_over_G0'] == 25. and entry['CPU_over_G1'] == 50.
    assert entry['observed_speed_ratio'] == 2.      # CPU samples never enter the G0/G1 pairing


def test_cpu_timing_cannot_masquerade_as_gpu_control(saved):
    root, save = saved
    save(); save('G0', 0, seconds=4.); save('G1', 0, seconds=2.); save_cpu_timing(root, 0)
    entry = summarize.summarize(root)['cases'][0]
    assert entry['paired_indices'] == [0] and 'CPU' not in {r.get('variant') for r in entry['unmatched_timings']}


@pytest.mark.parametrize('change', [
    lambda d: d['result']['gradient_hartree_per_bohr'][0].__setitem__(0, 1e-3),   # corrupted force
    lambda d: d.update(pair_index=1.5),
    lambda d: d.update(phase_memory={'peak_live_bytes': 1}),                       # instrumented timing
    lambda d: d.pop('comparison'),
])
def test_corrupted_cpu_timing_is_rejected(saved, change):
    root, save = saved
    save()
    path = save_cpu_timing(root, 0)
    alter(path, change)
    with pytest.raises(ValueError):
        audit.audit_campaign(root)


def test_duplicate_cpu_repetition_slot_is_rejected(saved):
    root, save = saved
    save(); save_cpu_timing(root, 0, attempt=0); save_cpu_timing(root, 0, attempt=1)
    with pytest.raises(ValueError, match='duplicate CPU timing slot'):
        audit.audit_campaign(root)


def test_cpu_variant_may_time_but_not_profile_or_memory():
    for mode in ('reference', 'capability', 'timing'):
        run.validate_mode(mode)
