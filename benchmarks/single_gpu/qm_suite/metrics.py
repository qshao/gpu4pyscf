"""Host accounting for dedicated diagnostic runs; no CUDA import at collection."""
from contextlib import contextmanager
from pathlib import Path
import resource
import threading
import time


def host_observation():
    data={'host_address_space_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemTotal','MemAvailable'):
            data[key.lower()+'_bytes']=int(value.split()[0])*1024
    for name in ['memory.current','memory.max']:  # cgroup v2
        path=Path('/sys/fs/cgroup')/name
        if path.exists():
            value=path.read_text().strip()
            data['cgroup_'+name.replace('.','_')]=int(value) if value!='max' else None
    if 'cgroup_memory_max' not in data:  # cgroup v1 (LCC): the job's own memory controller group
        try:
            group=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                       if 'memory' in line.split(':',2)[1].split(','))
            base=Path('/sys/fs/cgroup/memory')/group.lstrip('/')
            limit=int((base/'memory.limit_in_bytes').read_text());usage=int((base/'memory.usage_in_bytes').read_text())
            # An unconstrained group reports an effectively unlimited value; treat it as no cgroup limit.
            data['cgroup_memory_max']=limit if limit<data.get('memtotal_bytes',0) else None
            data['cgroup_memory_current']=usage
        except (OSError,StopIteration,ValueError):pass
    return data


@contextmanager
def host_samples(interval=0.1):
    """Sample system/cgroup state; RSS VmHWM itself is a kernel high-water mark."""
    stop=threading.Event();samples=[]
    def sample():
        while not stop.is_set():
            samples.append(dict(monotonic=time.monotonic(),**host_observation()))
            stop.wait(interval)
    worker=threading.Thread(target=sample,daemon=True);worker.start()
    data={'interval_seconds':interval,'samples':samples,
          'note':'Sampled host/system state is not an exact physical allocation peak; never add RSS and CuPy storage on unified memory.'}
    try:yield data
    finally:
        stop.set();worker.join();data['final']=host_observation()


def available_headroom(observation,device_free=None):
    limits=[observation['memavailable_bytes']]
    maximum=observation.get('cgroup_memory_max')
    if maximum is not None:
        limits.append(max(0,maximum-observation.get('cgroup_memory_current',0)))
    if device_free is not None:limits.append(device_free)
    return min(limits)


def resource_admission(estimate,observation,device_free=None):
    headroom=available_headroom(observation,device_free)
    return {'admitted':estimate<=.7*headroom,'estimated_peak_bytes':estimate,
            'tightest_available_headroom_bytes':headroom,'fraction_limit':.7,
            'note':'Conservative admission forecast, not a physical-RAM bound; unified memory metrics are not added.'}


def estimate_storage(case,variant):
    """Dimension-based admission estimate only; measured allocations remain separate."""
    from pyscf import gto,df,lib
    root=Path(__file__).resolve().parents[3]
    if case['task']=='pbc_energy':
        return {'estimated_peak_bytes':2*1024**3,'policy':'small periodic feasibility floor; actual FFT/native storage separately monitored'}
    with lib.with_omp_threads(4):
        mol=gto.M(atom=str(root/case['geometry']),basis=case['basis'],charge=case['charge'],spin=case['spin'],
                  ecp=case.get('task_options',{}).get('ecp',{}),verbose=0)
        nao=mol.nao_nr();naux=df.addons.make_auxmol(mol,case['auxbasis']).nao_nr() if case['df'] else 0
    cderi=naux*nao*(nao+1)//2*8
    metric=naux*naux*16
    buffers=8*nao*nao*256+8*nao*256*max(mol.nelectron//2,1)
    base=512*1024**2+40*nao*nao*8
    if case['df']:
        factor=3 if variant in ('G0','CPU') else 1
        peak=base+factor*cderi+metric+buffers+case['settings'].get('build_budget_bytes',4*1024**3)
    elif case['task']=='ccsd':peak=base+3*nao**4*8
    else:peak=base
    return {'nao':nao,'naux':naux,'full_triangular_cderi_bytes':cderi,'estimated_peak_bytes':peak,
            'policy':'Unscreened integral size plus conservative factors/buffers; not a measured or guaranteed peak.'}
