"""Reject tampering with references, settings, numerical data and phase peaks."""
import json

import pytest


@pytest.mark.parametrize('corruption',[None,'reference','geometry','force','phase','time','missing_helper'])
def test_resident_audit_checks_raw_record(tmp_path,monkeypatch,corruption):
    import summarize_resident_workspace as audit
    name='large_control_profile_0.json'
    record=json.loads((audit.OUT/name).read_text())
    if corruption=='reference': record['cpu_reference_sha256']='forged'
    if corruption=='geometry': record['geometry_sha256']='forged'
    if corruption=='force': record['result']['gradient_hartree_per_bohr'][0][0]+=.001
    if corruption=='phase': record['phase_memory']['phases']['scf']['peak_live_bytes']=-1
    if corruption=='time': record['workflow_seconds']=-1
    if corruption=='missing_helper': del record['helpers']['test_resident_workspace_run.py']
    (tmp_path/name).write_text(json.dumps(record))
    monkeypatch.setattr(audit,'OUT',tmp_path)
    if corruption:
        with pytest.raises(AssertionError): audit.load('control','large','profile',0)
    else:
        assert audit.load('control','large','profile',0)['comparison']['passed']


@pytest.mark.parametrize('budget',[4*1024**3,1024**3,None])
def test_candidate_audit_requires_effective_runtime_budget(tmp_path,monkeypatch,budget):
    import summarize_resident_workspace as audit
    record=json.loads((audit.OUT/'large_control_profile_0.json').read_text())
    record.update(version='candidate',mode='memory',
                  algorithm_sources=audit.python_sources(audit.ROOT/'gpu4pyscf'))
    record['helpers']['test_resident_workspace_run.py']=audit.digest(
        audit.BASE/'test_resident_workspace_run.py')
    if budget is not None:
        record['effective_build_workspace_bytes']=budget
    (tmp_path/'large_candidate_memory_0.json').write_text(json.dumps(record))
    monkeypatch.setattr(audit,'OUT',tmp_path)
    if budget==4*1024**3:
        assert audit.load('candidate','large','memory',0)['comparison']['passed']
    else:
        with pytest.raises((AssertionError,KeyError)):
            audit.load('candidate','large','memory',0)


@pytest.mark.parametrize('corruption',['missing_metadata','forged_metadata','stale_configuration'])
def test_candidate_audit_checks_benchmark_dependencies(tmp_path,monkeypatch,corruption):
    import summarize_resident_workspace as audit
    record=json.loads((audit.OUT/'large_control_profile_0.json').read_text())
    record.update(version='candidate',mode='memory',
                  algorithm_sources=audit.python_sources(audit.ROOT/'gpu4pyscf'),
                  effective_build_workspace_bytes=4*1024**3)
    record['helpers']['test_resident_workspace_run.py']=audit.digest(
        audit.BASE/'test_resident_workspace_run.py')
    source='benchmarks/single_gpu/molecular.py'
    if corruption=='missing_metadata': del record['source_hashes'][source]
    if corruption=='forged_metadata': record['source_hashes'][source]='forged'
    if corruption=='stale_configuration':
        reference=audit.cpu_reference
        def stale(size):
            path,cpu=reference(size)
            return path,dict(cpu,configuration_sha256='stale')
        monkeypatch.setattr(audit,'cpu_reference',stale)
    (tmp_path/'large_candidate_memory_0.json').write_text(json.dumps(record))
    monkeypatch.setattr(audit,'OUT',tmp_path)
    with pytest.raises(AssertionError):
        audit.load('candidate','large','memory',0)
