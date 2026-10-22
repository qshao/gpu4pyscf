"""CPU-only invariants for fresh-process benchmark execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))

def module(name):
    spec=importlib.util.spec_from_file_location('qm_'+name,BASE/(name+'.py'))
    assert spec is not None and spec.loader is not None
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def test_record_refuses_overwrite(tmp_path):
    r=module('run');p=tmp_path/'r.json'
    r.write_once(p,{'status':'passed'})
    import pytest
    with pytest.raises(FileExistsError):r.write_once(p,{'status':'failed'})
    assert json.loads(p.read_text())['status']=='passed'

def test_source_identity_change_rejected(tmp_path):
    r=module('run');p=tmp_path/'a.py';p.write_text('x=1\n')
    identities={str(p):r.digest(p)}
    r.verify_identities(identities)
    p.write_text('x=2\n')
    import pytest
    with pytest.raises(RuntimeError,match='identity'):r.verify_identities(identities)

def test_signature_deterministic_and_sensitive():
    r=module('run')
    assert r.signature({'spin':0,'charge':0})==r.signature({'charge':0,'spin':0})
    assert r.signature({'spin':0})!=r.signature({'spin':1})

def test_pair_schedule_balanced_and_reproducible():
    m=module('schedule')
    order=m.paired_order(5,42)
    assert order==m.paired_order(5,42)
    assert order.count(['G0','G1'])==3
    assert order.count(['G1','G0'])==2

def test_budget_charges_only_its_lane():
    m=module('schedule');b=m.Budget(gpu_hours=1,cpu_hours=2)
    b.charge('cpu',1800);b.charge('gpu',100)
    assert b.remaining('cpu')==5400 and b.remaining('gpu')==3500
    assert not b.admits('gpu',3501)

def test_extension_rule_symmetric():
    m=module('schedule')
    assert m.needs_more([10]*5,[10.1]*5)
    assert not m.needs_more([20]*5,[10]*5)
    assert m.needs_more([10]*5,[9.9]*5)

def test_helpers_import_without_cuda():
    for name in ['run','schedule','audit','metrics']:
        path=str(BASE/(name+'.py'))
        code=f"import runpy,sys;runpy.run_path({path!r},run_name='safe_import');assert 'cupy' not in sys.modules"
        cp=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        assert cp.returncode==0,cp.stderr

def test_nested_instrumentation_prohibited_in_timing():
    r=module('run')
    import pytest
    with pytest.raises(ValueError,match='timing'):r.validate_mode('timing',True)
    r.validate_mode('memory',True)

def test_missing_gpu_result_is_failure_not_pass():
    r=module('run')
    out=r.classify_exception(RuntimeError('cudaErrorNoDevice: no CUDA-capable device'))
    assert out=='infrastructure_failed'

def test_memory_estimates_not_added_on_unified():
    m=module('metrics')
    x=m.host_observation()
    assert 'physical_total_from_gpu_plus_rss' not in x
    assert 'host_address_space_peak_rss_bytes' in x

def test_resource_headroom_is_minimum_not_sum():
    m=module('metrics')
    obs={'memavailable_bytes':1000,'memtotal_bytes':2000,'cgroup_memory_max':900,'cgroup_memory_current':300}
    assert m.available_headroom(obs,800)==600
    assert not m.resource_admission(500,obs,800)['admitted']
    assert m.resource_admission(400,obs,800)['admitted']

def test_missing_record_after_attempt_is_not_resumable_pass(tmp_path):
    m=module('schedule')
    directory=tmp_path/'jobs'/'water__G0__timing__0000';directory.mkdir(parents=True)
    assert m.completed(tmp_path,'water','G0','timing')==[]

def test_pair_slots_are_not_shifted_by_failures(tmp_path):
    m=module('schedule')
    for i,status in enumerate(['qualified','accuracy_failed']):
        d=tmp_path/'jobs'/f'water__G0__timing__{i:04d}';d.mkdir(parents=True)
        (d/'result.json').write_text(json.dumps(dict(status=status,pair_index=i)))
    assert m.pair_attempted(tmp_path,'water','G0',0)
    assert m.pair_attempted(tmp_path,'water','G0',1)
    assert not m.pair_attempted(tmp_path,'water','G0',2)
