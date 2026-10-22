"""One task in one process. Timing, allocation and profiling are separate lanes."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def signature(case):
    return hashlib.sha256(json.dumps(case,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_once(path,value):
    """Publish an immutable record; existing records cannot be replaced."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(content)


def verify_identities(identities):
    for p,expected in identities.items():
        if not Path(p).is_file() or digest(p)!=expected:
            raise RuntimeError('source identity changed: '+str(p))


def validate_mode(mode,instrumented=False):
    if mode not in ('reference','capability','timing','memory','profile'):
        raise ValueError('unknown measurement mode '+mode)
    if mode=='timing' and instrumented:
        raise ValueError('timing must exclude allocation and profiling instrumentation')


def classify_exception(exc):
    name=type(exc).__name__;message=str(exc)
    if name=='DependencyMissing' or isinstance(exc,ModuleNotFoundError):return 'dependency_missing'
    if name in ('UnsupportedTask','AdapterNotImplemented') or isinstance(exc,NotImplementedError):return 'unsupported'
    if isinstance(exc,MemoryError) or 'OutOfMemory' in name:return 'resource_limited'
    if 'NoDevice' in message or 'CUDA-capable device' in message or 'driver' in message.lower():return 'infrastructure_failed'
    if 'converg' in message.lower():return 'nonconverged'
    return 'infrastructure_failed'


def environment(backend):
    versions={}
    for name in ['numpy','pyscf','cupy-cuda13x','cupy-cuda12x','gpu4pyscf-libxc-cuda12x','cutensor-cu12']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    data={'machine':platform.machine(),'python':sys.version,'interpreter':sys.executable,
          'versions':versions,'backend':backend,
          'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    try:data['cpu_affinity_count']=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):pass
    try:
        data['cpu_model']=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    except (OSError,StopIteration):pass
    if backend=='gpu':
        try:
            from gpu4pyscf.lib import cutensor as _ct
            data['contract_engine']=_ct.contract_engine or 'cutensor'
        except Exception as error:data['contract_engine']='unknown: '+repr(error)
        import cupy as cp
        count=cp.cuda.runtime.getDeviceCount()
        if count!=1:raise RuntimeError('exactly one visible CUDA device required; observed '+str(count))
        prop=cp.cuda.runtime.getDeviceProperties(0)
        name=prop['name'];data.update(device=name.decode() if isinstance(name,bytes) else name,
             compute_capability=cp.cuda.Device().compute_capability,
             cuda_driver=cp.cuda.runtime.driverGetVersion(),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
             free_and_total_device_bytes=list(cp.cuda.runtime.memGetInfo()))
    return data


def check_campaign(campaign):
    from manifest import verify_manifest
    verify_manifest(campaign['sources'])
    verify_identities(campaign['helper_identities'])
    verify_identities(campaign.get('native_identities',{}))
    verify_identities(campaign.get('input_identities',{}))


def job_id(case_id,variant,mode,attempt):
    value=f'{case_id}__{variant}__{mode}__{attempt:04d}'
    if not re.fullmatch('[A-Za-z0-9_.-]+',value):raise ValueError('unsafe job identifier')
    return value


def run_one(campaign_root,case_id,variant,mode,attempt,pair_index=None):
    validate_mode(mode)
    campaign_root=Path(campaign_root).resolve()
    campaign=json.loads((campaign_root/'campaign.json').read_text())
    cases={c['id']:c for c in campaign['sources']['cases']}
    case=cases[case_id];backend='cpu' if variant=='CPU' else 'gpu'
    if mode=='reference' and backend!='cpu':raise ValueError('references must use independent CPU')
    if backend=='cpu' and mode not in ('reference','capability','timing'):raise ValueError('CPU runs are references or CPU timing samples only')
    directory=campaign_root/'jobs'/job_id(case_id,variant,mode,attempt)
    directory.mkdir(parents=True,exist_ok=False)
    request={'case_id':case_id,'variant':variant,'mode':mode,'attempt':attempt,
             'campaign_sha256':digest(campaign_root/'campaign.json'),'scientific_signature':signature(case),'pair_index':pair_index}
    write_once(directory/'request.json',request)
    report=dict(request,status='running',backend=backend,started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    started=time.monotonic();mf_result=None
    try:
        check_campaign(campaign)
        if backend=='gpu':
            path=Path(campaign['sources']['variants'][variant]['root'])
            package=path if path.name=='gpu4pyscf' else path/'gpu4pyscf'
            if any(n=='gpu4pyscf' or n.startswith('gpu4pyscf.') for n in sys.modules):
                raise RuntimeError('GPU package imported before snapshot isolation')
            sys.path.insert(0,str(package.parent))
        report['environment']=environment(backend)
        if backend=='gpu':
            import cupy as cp
            import gpu4pyscf
            if Path(gpu4pyscf.__file__).resolve()!=package/'__init__.py':raise RuntimeError('wrong GPU snapshot imported')
            cp.matmul(cp.ones((1,1)),cp.ones((1,1)));cp.cuda.Device().synchronize()
            report['gpu_source']=str(Path(gpu4pyscf.__file__).resolve())
        from adapters import execute
        from metrics import host_observation,host_samples
        report['host_before']=host_observation()
        if mode=='memory':
            sys.path.insert(0,str(BASE.parent))
            from phase_memory import phase_memory
            with host_samples() as host,phase_memory() as tracker:
                mf_result=execute(case,backend,mode=mode,tracker=tracker)
            report.update(phase_memory=tracker.data,host_samples=host)
        else:
            mf_result=execute(case,backend,mode=mode)
        mf_result.update(case=case,case_id=case_id,scientific_signature=signature(case))
        report['result']=mf_result
        report['host_after']=host_observation()
        check_campaign(campaign)
        if not mf_result.get('converged') or not mf_result.get('task_converged'):
            report['status']='nonconverged'
        elif backend=='gpu':
            from compare import compare_results
            reference=find_reference(campaign_root,case_id)
            if reference is None:
                report.update(status='unqualified',reason='independent CPU reference unavailable')
            else:
                ref=json.loads(reference.read_text())
                report['reference_path']=str(reference.relative_to(campaign_root));report['reference_sha256']=digest(reference)
                report['comparison']=compare_results(case,ref['result'],mf_result)
                report['status']='qualified' if report['comparison']['passed'] else report['comparison'].get('qualification_status','accuracy_failed')
                if report['status']=='qualified' and not report['comparison']['passed']:report['status']='accuracy_failed'
        elif mode=='timing':
            # CPU timing sample (separate lane, declared whole-node thread policy). Its accuracy is the reproducibility of
            # the independent reference; it is never a GPU control and never enters GPU-versus-GPU ratios.
            from compare import compare_results
            reference=find_reference(campaign_root,case_id)
            if reference is None:
                report.update(status='unqualified',reason='independent CPU reference unavailable')
            else:
                ref=json.loads(reference.read_text())
                report['reference_path']=str(reference.relative_to(campaign_root));report['reference_sha256']=digest(reference)
                report['comparison']=compare_results(case,ref['result'],mf_result)
                report['status']='cpu_timed' if report['comparison']['passed'] else report['comparison'].get('qualification_status','accuracy_failed')
        else:
            # Successful CPU execution is a reference, not a CPU/GPU accuracy pass.
            from compare import compare_results
            reference_check=compare_results(case,mf_result,mf_result)
            report['reference_self_check']=reference_check
            report['status']='reference_ready' if reference_check['passed'] else 'nonconverged'
    except Exception as exc:
        report.update(status=classify_exception(exc),error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        report['process_elapsed_seconds']=time.monotonic()-started
        write_once(directory/'result.json',report)
    print(json.dumps({'job':directory.name,'status':report['status'],'seconds':report['process_elapsed_seconds']}),flush=True)
    return report


def find_reference(campaign_root,case_id):
    candidates=[]
    for p in (Path(campaign_root)/'jobs').glob(case_id+'__CPU__reference__*/result.json'):
        d=json.loads(p.read_text())
        if d['status']=='reference_ready':candidates.append(p)
    return sorted(candidates)[-1] if candidates else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',required=True);p.add_argument('--case',required=True)
    p.add_argument('--variant',choices=['CPU','G0','G1','G0a'],required=True)
    p.add_argument('--mode',choices=['reference','capability','timing','memory','profile'],required=True)
    p.add_argument('--attempt',type=int,default=0);p.add_argument('--pair-index',type=int);a=p.parse_args()
    out=run_one(a.campaign_root,a.case,a.variant,a.mode,a.attempt,a.pair_index)
    return 0 if out['status'] in ('qualified','reference_ready','cpu_timed') else 2


if __name__=='__main__':raise SystemExit(main())
