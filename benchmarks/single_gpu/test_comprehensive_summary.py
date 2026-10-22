"""The independent validator must reject forged passed numerical comparisons."""
import copy
import json

import pytest


def test_candidate_version_requires_saved_source():
    import summarize_comprehensive as validator
    archived=validator.BASE/'runs/2026-10-17-pre-comprehensive/occupied-pair-lower-only.py.source'
    assert validator.candidate_version(validator.digest(archived))=='lower_triangle_v1'
    assert validator.candidate_version(validator.digest(validator.BASE/'occupied_pair.py'))=='symmetric_average_factored_v3'
    with pytest.raises(AssertionError): validator.candidate_version('unknown')


@pytest.mark.parametrize('corruption',[None,'energy','gradient'])
def test_recompute_scientific_gate(tmp_path,monkeypatch,corruption):
    import summarize_comprehensive as validator
    from molecular import _metadata,compare_cases
    monkeypatch.setattr(validator,'RESULTS',tmp_path)
    case='vitamin_svp_pbe'
    config=validator.settings(case)
    metadata=_metadata('cpu')
    result=dict(metadata,method='PBE',settings=config,converged=True,
                energy_hartree=-1.,gradient_hartree_per_bohr=[[0.,0.,0.]],
                scf_seconds=1.,gradient_seconds=1.,iterations=1)
    common=dict(metadata,status='passed',settings=config,
                driver_sha256=validator.digest(validator.BASE/'comprehensive.py'),
                geometry_sha256=validator.digest(config['atom']))
    cpu_path=tmp_path/(case+'_cpu.json')
    cpu_path.write_text(json.dumps(dict(common,result=result)))
    actual=copy.deepcopy(result)
    comparison=compare_cases(result,actual)
    if corruption=='energy': actual['energy_hartree']+=1e-4
    if corruption=='gradient': actual['gradient_hartree_per_bohr'][0][0]=1e-3
    report=dict(common,result=actual,comparison=comparison,
                cpu_reference_sha256=validator.digest(cpu_path))
    (tmp_path/(case+'_gpu.json')).write_text(json.dumps(report))
    if corruption:
        with pytest.raises(AssertionError): validator.summarize()
    else:
        summary=validator.summarize()
        assert summary['status']=='partial'
        assert summary['cases'][0]['modes']['gpu']['comparison']['passed']


def test_large_failed_run_stays_failed(tmp_path,monkeypatch):
    import summarize_comprehensive as validator
    from molecular import _metadata
    monkeypatch.setattr(validator,'RESULTS',tmp_path)
    case='taxol_svp_b3lyp'
    config=validator.settings(case)
    metadata=_metadata('cpu')
    result=dict(metadata,method='B3LYP',settings=config,converged=True,
                energy_hartree=-1.,gradient_hartree_per_bohr=[[0.,0.,0.]],
                scf_seconds=1.,gradient_seconds=1.,iterations=1)
    common=dict(metadata,settings=config,geometry_sha256=validator.digest(config['atom']),result=result)
    (tmp_path/(case+'_seeded_cpu.json')).write_text(json.dumps(dict(common,status='passed')))
    (tmp_path/(case+'_capacity_disk_packed.json')).write_text(json.dumps(dict(common,
        status='failed',error='strict differential gate',configuration=dict(scientific=config))))
    summary=validator.summarize()
    entry=summary['large_cases'][0]
    assert entry['status']=='failed'
    assert entry['cpu_comparison']['passed']
    assert entry['error']=='strict differential gate'
    assert summary['status']=='partial'


def test_replaced_gpu_reference_rejected(tmp_path,monkeypatch):
    import summarize_comprehensive as validator
    from molecular import _metadata
    monkeypatch.setattr(validator,'RESULTS',tmp_path)
    case='taxol_svp_b3lyp'; config=validator.settings(case)
    metadata=_metadata('cpu')
    result=dict(metadata,method='B3LYP',settings=config,converged=True,
                energy_hartree=-1.,gradient_hartree_per_bohr=[[0.,0.,0.]],
                scf_seconds=1.,gradient_seconds=1.,iterations=1)
    common=dict(metadata,case=case,result=result,
                geometry_sha256=validator.digest(config['atom']),
                configuration=dict(scientific=config,storage='resident',buffer_bytes=1,cache_bytes=0),
                native_libraries={},algorithm_source_hashes={},cupy='fixture',
                memory_profiler_sha256=validator.digest(validator.BASE/'memory_profile.py'),
                candidate_sha256=validator.digest(validator.BASE/'occupied_pair.py'))
    baseline=tmp_path/(case+'_capacity_resident_baseline.json')
    baseline.write_text(json.dumps(dict(common,status='captured_original_gpu',variant='baseline')))
    candidate=dict(common,status='failed',variant='packed',gpu_reference_storage='resident',
                   gpu_reference_sha256=validator.digest(baseline))
    (tmp_path/(case+'_capacity_disk_packed.json')).write_text(json.dumps(candidate))
    # Even a semantically identical replacement must not acquire the old identity.
    baseline.write_text(baseline.read_text()+'\n')
    with pytest.raises(AssertionError,match='GPU reference report changed'):
        validator.summarize()
