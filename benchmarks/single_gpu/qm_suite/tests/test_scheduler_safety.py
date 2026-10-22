"""CPU-only scheduler safety tests; subprocess fixtures never import CUDA."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import schedule
import metrics
import run


def test_module_dry_run_counts_extension_ceiling():
    cp=subprocess.run([sys.executable,'-m','benchmarks.single_gpu.qm_suite.schedule','--campaign','pilot','--dry-run'],capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
    assert json.loads(cp.stdout)['gpu_timing_jobs_upper_bound']==180


def test_capacity_fails_before_preparation(monkeypatch):
    monkeypatch.setattr(sys,'argv',['schedule','--campaign','capacity','--dry-run'])
    with pytest.raises(NotImplementedError,match='adaptive'):
        schedule.main()


@pytest.mark.parametrize('failure',[FileNotFoundError('missing'),subprocess.TimeoutExpired(['probe'],30)])
def test_preflight_archives_probe_errors(tmp_path,monkeypatch,failure):
    def fail(*args,**kwargs):raise failure
    monkeypatch.setattr(subprocess,'run',fail)
    result=schedule.preflight(tmp_path)
    assert result['status']=='infrastructure_failed'
    saved=list((tmp_path/'preflights').glob('*.json'))
    assert len(saved)==1
    assert json.loads(saved[0].read_text())['probe_errors']


def test_resume_rejects_changed_policy_before_work(tmp_path,monkeypatch):
    (tmp_path/'campaign.json').write_text(json.dumps({'sources':{'cases':[{'id':'water'}]},'schedule_policy':{'seed':0}}))
    monkeypatch.setattr(schedule,'selected_cases',lambda name:[{'id':'water'}])
    monkeypatch.setattr(run,'check_campaign',lambda campaign:None)
    monkeypatch.setattr(sys,'argv',['schedule','--references-only','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='scheduling policy'):
        schedule.main()


def test_group_cleanup_kills_term_ignoring_descendant(tmp_path):
    childfile=tmp_path/'child.pid'
    code="""import subprocess,sys,time
p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
open(sys.argv[1],'w').write(str(p.pid))
time.sleep(60)
"""
    process=subprocess.Popen([sys.executable,'-c',code,str(childfile)],start_new_session=True)
    try:
        deadline=time.monotonic()+5
        while not childfile.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert childfile.exists()
        time.sleep(.1)
        result=schedule.stop_owned_process_group(process,grace=.1)
        assert result['verified_no_live_children']
        assert result['sigkill_sent']
        assert not schedule.group_has_live_processes(process.pid)
    finally:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def fixture_launch(tmp_path,monkeypatch,variant='CPU',storage='resident'):
    case={'id':'water','task_options':{'storage':storage},'settings':{}}
    (tmp_path/'campaign.json').write_text(json.dumps({'sources':{'cases':[case]}}))
    monkeypatch.setattr(metrics,'estimate_storage',lambda *a:{'estimated_peak_bytes':100,'full_triangular_cderi_bytes':100})
    observation={'memavailable_bytes':10000,'memtotal_bytes':20000,'cgroup_memory_max':10000,'cgroup_memory_current':0}
    monkeypatch.setattr(metrics,'host_observation',lambda:dict(observation))
    return observation


def test_gpu_device_headroom_rejects_before_launch(tmp_path,monkeypatch):
    fixture_launch(tmp_path,monkeypatch)
    monkeypatch.setattr(schedule,'device_free_bytes',lambda:50,raising=False)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('device admission must reject'))
    result=schedule.launch(tmp_path,'water','G1','memory',schedule.Budget(1,1),timeout=1)
    assert result['status']=='resource_limited'
    assert result['admission']['tightest_available_headroom_bytes']==50


def test_disk_scratch_admission_is_separate(tmp_path,monkeypatch):
    fixture_launch(tmp_path,monkeypatch,storage='disk')
    monkeypatch.setattr(schedule,'device_free_bytes',lambda:10000,raising=False)
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda p:type('Usage',(),{'free':10})())
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('scratch admission must reject'))
    result=schedule.launch(tmp_path,'water','G1','memory',schedule.Budget(1,1),timeout=1)
    assert result['status']=='resource_limited'
    assert not result['scratch_admission']['admitted']


@pytest.mark.parametrize('failure',[KeyboardInterrupt(),RuntimeError('monitor failed')])
def test_monitor_errors_cleanup_and_preserve_record(tmp_path,monkeypatch,failure):
    fixture_launch(tmp_path,monkeypatch)
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:child)
    calls=0
    def observe():
        nonlocal calls
        calls+=1
        if calls>1:raise failure
        return {'memavailable_bytes':10000,'memtotal_bytes':20000}
    monkeypatch.setattr(metrics,'host_observation',observe)
    try:
        with pytest.raises(type(failure)):
            schedule.launch(tmp_path,'water','CPU','reference',schedule.Budget(1,1),timeout=1)
        assert child.poll() is not None
        results=list((tmp_path/'jobs').glob('*/result.json'))
        assert len(results)==1
        assert json.loads(results[0].read_text())['status']=='infrastructure_failed'
        assert json.loads((results[0].parent/'cleanup.json').read_text())['verified_no_live_children']
    finally:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def test_cgroup_pressure_stops_child(tmp_path,monkeypatch):
    observation=fixture_launch(tmp_path,monkeypatch)
    real_popen=subprocess.Popen
    def start(*args,**kwargs):
        observation['cgroup_memory_current']=9900
        return real_popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    monkeypatch.setattr(subprocess,'Popen',start)
    result=schedule.launch(tmp_path,'water','CPU','reference',schedule.Budget(1,1),timeout=2)
    assert result['status']=='resource_limited'


def test_device_forecast_is_advisory_on_request(tmp_path,monkeypatch):
    fixture_launch(tmp_path,monkeypatch)
    monkeypatch.setenv('QM_SUITE_DEVICE_ADMISSION','advisory')
    monkeypatch.setattr(schedule,'device_free_bytes',lambda:50,raising=False)
    started=[]
    class Stop(Exception):pass
    def popen(*a,**k):
        started.append(True);raise Stop()
    monkeypatch.setattr(subprocess,'Popen',popen)
    with pytest.raises(Stop):
        schedule.launch(tmp_path,'water','G1','memory',schedule.Budget(1,1),timeout=1)
    assert started  # admitted although the 50-byte device headroom is below the 100-byte estimate
