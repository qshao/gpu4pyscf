"""Bounded, resumable fresh-process campaign scheduler. One job at a time."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))  # also works under python -m benchmarks.single_gpu.qm_suite.schedule
EXTENSION_CEILING=9  # samples per variant after the symmetric near-unity/noisy extension rule


def paired_order(repetitions,seed):
    orders=[['G0','G1'] for _ in range((repetitions+1)//2)]+[['G1','G0'] for _ in range(repetitions//2)]
    random.Random(seed).shuffle(orders)
    return orders


class Budget:
    def __init__(self,gpu_hours,cpu_hours):
        if not all(math.isfinite(x) and x>=0 for x in (gpu_hours,cpu_hours)):raise ValueError('nonnegative finite budgets required')
        self.limits={'gpu':gpu_hours*3600,'cpu':cpu_hours*3600}
        self.used={'gpu':0.,'cpu':0.}
    def charge(self,lane,seconds):
        if seconds<0 or not math.isfinite(seconds):raise ValueError('invalid elapsed time')
        self.used[lane]+=seconds
    def remaining(self,lane):return max(0.,self.limits[lane]-self.used[lane])
    def admits(self,lane,expected):return expected<=self.remaining(lane)


def needs_more(left,right):
    if not left or not right:return True
    def norm_mad(x):
        med=statistics.median(x)
        if med<=0:return float('inf')
        return statistics.median(abs(v-med) for v in x)/med
    ratio=statistics.median(left)/statistics.median(right)
    return max(norm_mad(left),norm_mad(right))>.03 or .95<=ratio<=1.05


def selected_cases(name):
    from manifest import load_cases,pilot_cases
    if name=='pilot':return pilot_cases()
    if name=='all':return load_cases()  # every registry case; callers narrow it with --cases
    group={'core':'core','extensions':'extension','direct':'direct','periodic':'periodic','capacity':'capacity'}[name]
    return load_cases(group)


def preflight(output):
    """One probe, preserved verbatim. Missing devices never become timing samples."""
    from run import write_once
    code="import sys,json;sys.path.insert(0,"+repr(str(BASE))+ ");from run import environment;print(json.dumps(environment('gpu')))"
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0'
    errors=[];cp=smi=None
    try:cp=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env,timeout=30)
    except (OSError,subprocess.SubprocessError) as error:errors.append('python probe: '+repr(error))
    try:smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.SubprocessError) as error:errors.append('nvidia-smi probe: '+repr(error))
    ok=cp is not None and cp.returncode==0
    data={'status':'available' if ok else 'infrastructure_failed','probe_errors':errors,
          'python_exit':None if cp is None else cp.returncode,'python_stdout':'' if cp is None else cp.stdout,
          'python_stderr':'' if cp is None else cp.stderr,
          'nvidia_smi_exit':None if smi is None else smi.returncode,'nvidia_smi_stdout':'' if smi is None else smi.stdout,
          'nvidia_smi_stderr':'' if smi is None else smi.stderr,
          'device_nodes':[str(p) for p in Path('/dev').glob('nvidia*')],
          'utc':datetime.now(timezone.utc).isoformat()}
    if ok:data['environment']=json.loads(cp.stdout.splitlines()[-1])
    dest=Path(output)/'preflights';dest.mkdir(exist_ok=True)
    write_once(dest/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),data)
    return data


def device_free_bytes():
    """Free memory of visible GPU 0 from the driver, or None when it cannot be observed."""
    try:
        cp=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],
                          capture_output=True,text=True,timeout=30)
        return int(cp.stdout.split()[0])*1024**2 if cp.returncode==0 else None
    except (OSError,subprocess.SubprocessError,ValueError,IndexError):return None


def group_has_live_processes(pgid):
    """True when any non-zombie process belongs to process group `pgid`."""
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
        except (OSError,IndexError):continue
        if fields[0]!='Z' and int(fields[2])==pgid:return True
    return False


def stop_owned_process_group(process,grace=10.):
    """TERM then KILL the process group this scheduler started (never another group), and verify it is empty."""
    pgid=process.pid;result={'pgid':pgid,'sigterm_sent':False,'sigkill_sent':False}
    def signal_group(sig):
        try:os.killpg(pgid,sig);return True
        except ProcessLookupError:return False
    result['sigterm_sent']=signal_group(signal.SIGTERM)
    deadline=time.monotonic()+grace
    while time.monotonic()<deadline and group_has_live_processes(pgid):time.sleep(.02)
    if group_has_live_processes(pgid):
        result['sigkill_sent']=signal_group(signal.SIGKILL)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and group_has_live_processes(pgid):time.sleep(.02)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    result['verified_no_live_children']=not group_has_live_processes(pgid)
    return result


def next_attempt(output,case,variant,mode):
    numbers=[]
    for path in (Path(output)/'jobs').glob(f'{case}__{variant}__{mode}__*'):
        try:numbers.append(int(path.name.rsplit('__',1)[-1]))
        except ValueError:continue
    return max(numbers,default=-1)+1


def completed(output,case,variant,mode):
    data=[]
    for path in (Path(output)/'jobs').glob(f'{case}__{variant}__{mode}__*/result.json'):
        d=json.loads(path.read_text())
        if d['status'] in ('qualified','reference_ready','cpu_timed'):data.append((path,d))
    return sorted(data,key=lambda x:str(x[0]))


def pair_attempted(output,case,variant,pair_index):
    for path in (Path(output)/'jobs').glob(f'{case}__{variant}__timing__*/result.json'):
        if json.loads(path.read_text()).get('pair_index')==pair_index:return True
    return False


def launch(output,case,variant,mode,budget,timeout=1800,pair_index=None):
    from run import job_id,write_once,digest
    lane='cpu' if variant=='CPU' else 'gpu'
    if not budget.admits(lane,timeout):return None
    attempt=next_attempt(output,case,variant,mode)
    directory=Path(output)/'jobs'/job_id(case,variant,mode,attempt)
    cmd=[sys.executable,str(BASE/'run.py'),'--campaign-root',str(output),'--case',case,'--variant',variant,'--mode',mode,'--attempt',str(attempt)]
    if pair_index is not None:cmd+=['--pair-index',str(pair_index)]
    threads=os.environ.get('QM_SUITE_CPU_THREADS','4') if variant=='CPU' else '4'  # CPU lane: declared whole-node policy
    env=dict(os.environ,OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,MKL_NUM_THREADS=threads,
             CUDA_VISIBLE_DEVICES='' if variant=='CPU' else '0')
    logs=Path(output)/'logs';logs.mkdir(exist_ok=True)
    from metrics import host_observation,estimate_storage,resource_admission,available_headroom
    campaign_path=Path(output)/'campaign.json'
    case_config=next(c for c in json.loads(campaign_path.read_text())['sources']['cases'] if c['id']==case)
    estimate=estimate_storage(case_config,variant)
    # On a dedicated discrete GPU (LCC) the pre-launch device forecast is advisory: a real allocation failure is classified
    # resource_limited by the runner, whereas the forecast over-rejects (measured Taxol G0 peaks are well below it).
    advisory=os.environ.get('QM_SUITE_DEVICE_ADMISSION')=='advisory'
    device_free=device_free_bytes() if lane=='gpu' and not advisory else None
    admission=resource_admission(estimate['estimated_peak_bytes'],host_observation(),device_free)
    admission['device_admission']='advisory (exclusive discrete GPU; allocation failure is recorded as resource_limited)' if advisory else 'enforced'
    storage=(case_config.get('task_options') or {}).get('storage') or (case_config.get('settings') or {}).get('storage')
    scratch=None
    if storage=='disk':
        scratch_dir=Path(os.environ.get('QM_SUITE_SCRATCH',output))
        free=shutil.disk_usage(scratch_dir).free;need=estimate.get('full_triangular_cderi_bytes',estimate['estimated_peak_bytes'])
        scratch={'admitted':need<=.7*free,'required_bytes':need,'free_bytes':free,'directory':str(scratch_dir),'fraction_limit':.7,
                 'note':'Scratch capacity is admitted separately from device and host memory.'}
    def terminal(status,elapsed,**extra):
        directory.mkdir(parents=True,exist_ok=True)
        data={'status':status,'case_id':case,'variant':variant,'mode':mode,'attempt':attempt,'pair_index':pair_index,
              'campaign_sha256':digest(campaign_path),'process_elapsed_seconds':elapsed}
        data.update(extra);write_once(directory/'result.json',data);return data
    if not admission['admitted'] or (scratch and not scratch['admitted']):
        extra={'admission':admission,'estimate':estimate}
        if scratch:extra['scratch_admission']=scratch
        return terminal('resource_limited',0.,**extra)
    start=time.monotonic();cleanup=None;reason=None
    with (logs/(directory.name+'.log')).open('x') as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-start>timeout:reason='timeout';break
                obs=host_observation()
                capacity=min(x for x in (obs['memtotal_bytes'],obs.get('cgroup_memory_max')) if x)  # the binding limit
                if available_headroom(obs)<.2*capacity:reason='resource_limited';break
                time.sleep(.1)
        except BaseException as error:
            # Monitor failure or interrupt: never leave the child running, and keep an explicit record.
            cleanup=stop_owned_process_group(process)
            directory.mkdir(parents=True,exist_ok=True)
            write_once(directory/'cleanup.json',cleanup)
            if not (directory/'result.json').exists():
                terminal('infrastructure_failed',time.monotonic()-start,error=repr(error),cleanup=cleanup)
            budget.charge(lane,time.monotonic()-start)
            raise
        if reason:
            cleanup=stop_owned_process_group(process)
            directory.mkdir(parents=True,exist_ok=True)
            write_once(directory/'cleanup.json',cleanup)
            if not (directory/'result.json').exists():
                terminal(reason,time.monotonic()-start,cleanup=cleanup)
    elapsed=time.monotonic()-start;budget.charge(lane,elapsed)
    result=directory/'result.json'
    if not result.exists():
        terminal('infrastructure_failed',elapsed,returncode=process.returncode)
    return json.loads(result.read_text())


def main():
    from run import write_once,check_campaign
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',choices=['pilot','all','core','extensions','direct','periodic','capacity'],default='pilot')
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--references-only',action='store_true')
    p.add_argument('--cpu-timing',action='store_true',help='CPU timing lane: repeated fresh-process CPU workflows (no GPU)')
    p.add_argument('--cases',nargs='+');p.add_argument('--gpu-hours',type=float,default=8);p.add_argument('--cpu-hours',type=float,default=24)
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repetitions',type=int);a=p.parse_args()
    if a.timeout<=0:raise ValueError('positive timeout required')
    if a.campaign=='capacity':
        raise NotImplementedError('adaptive capacity ladder is not implemented; it needs the AO inventory and a resource-limited size ladder')
    cases=selected_cases(a.campaign)
    if a.cases:
        wanted=set(a.cases);cases=[c for c in cases if c['id'] in wanted]
        if {c['id'] for c in cases}!=wanted:raise ValueError('unknown case ID for selected campaign')
    reps=a.repetitions or (3 if a.campaign=='pilot' else 5)
    if reps<=0:raise ValueError('positive repetition count required')
    budget=Budget(a.gpu_hours,a.cpu_hours)
    policy={'seed':a.seed,'repetitions':reps,'extension_ceiling':EXTENSION_CEILING}
    if a.dry_run:
        print(json.dumps({'campaign':a.campaign,'cases':[c['id'] for c in cases],'case_count':len(cases),
              'gpu_timing_jobs_upper_bound':0 if a.references_only else len(cases)*2*max(reps,EXTENSION_CEILING),
              'separate_gpu_memory_jobs_upper_bound':0 if a.references_only else len(cases)*2,
              'cpu_references_upper_bound':len(cases),'cpu_timing_jobs_upper_bound':len(cases)*reps,'paired_order':paired_order(reps,a.seed),
              'gpu_hour_limit':a.gpu_hours,'cpu_hour_limit':a.cpu_hours},indent=2));return 0
    output=a.output or BASE/'results'/datetime.now(timezone.utc).strftime(a.campaign+'-%Y%m%dT%H%M%S')
    if not (output/'campaign.json').exists():
        from campaign import prepare
        campaign=prepare(output,cases,policy)
    else:
        campaign=json.loads((output/'campaign.json').read_text());check_campaign(campaign)
        if [c['id'] for c in campaign['sources']['cases']]!=[c['id'] for c in cases]:raise ValueError('resume cases differ from frozen campaign')
        if campaign.get('schedule_policy')!=policy:raise ValueError('resume scheduling policy differs from the frozen campaign: %s != %s'%(campaign.get('schedule_policy'),policy))
    # Budgets apply to the campaign across resumes, not separately to each invocation.
    for path in (output/'jobs').glob('*/result.json'):
        d=json.loads(path.read_text());budget.charge('cpu' if d['variant']=='CPU' else 'gpu',d['process_elapsed_seconds'])
    if not a.references_only and not a.cpu_timing:
        pre=preflight(output)
        if pre['status']!='available':
            print('GPU unavailable to this execution environment; preflight archived at '+str(output/'preflights'))
            return 2
    deferred=[]
    for case in cases:
        cid=case['id']
        if not completed(output,cid,'CPU','reference'):
            result=launch(output,cid,'CPU','reference',budget,a.timeout)
            if result is None:deferred.append({'case':cid,'reason':'cpu_budget'});continue
            if result['status']!='reference_ready':continue
        if a.references_only:continue
        if a.cpu_timing:
            # Needs the independent reference of the same campaign; pair_index counts CPU repetitions.
            for index in range(reps):
                if pair_attempted(output,cid,'CPU',index):continue
                if launch(output,cid,'CPU','timing',budget,a.timeout,index) is None:
                    deferred.append({'case':cid,'variant':'CPU','reason':'cpu_budget'})
            continue
        eligible=[]
        for variant in ('G0','G1'):
            if completed(output,cid,variant,'memory'):eligible.append(variant);continue
            result=launch(output,cid,variant,'memory',budget,a.timeout)
            if result and result['status']=='qualified':eligible.append(variant)
        for pair_index,order in enumerate(paired_order(reps,a.seed)):
            for variant in order:
                if variant not in eligible:continue
                if pair_attempted(output,cid,variant,pair_index):continue
                if launch(output,cid,variant,'timing',budget,a.timeout,pair_index) is None:
                    deferred.append({'case':cid,'variant':variant,'reason':'gpu_budget'})
        # Near-unity/noisy comparisons extend both controls symmetrically.
        observations={v:completed(output,cid,v,'timing') for v in ('G0','G1')}
        def times(v):return [d['result']['workflow_seconds'] for _,d in observations[v]]
        if all(len(observations[v])>=reps for v in ('G0','G1')) and needs_more(times('G0'),times('G1')):
            for pair_index,order in enumerate(paired_order(max(0,9-reps),a.seed+1),start=reps):
                for v in order:
                    if not pair_attempted(output,cid,v,pair_index):
                        if launch(output,cid,v,'timing',budget,a.timeout,pair_index) is None:
                            deferred.append({'case':cid,'variant':v,'reason':'extension_budget'})
    ledger={'limits_seconds':budget.limits,'used_seconds':budget.used,'deferred':deferred,
            'note':'Instrumented measurements are separate from timings; failed controls excluded from qualified ratios.'}
    write_once(output/('ledger-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),ledger)
    print(json.dumps(ledger,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
