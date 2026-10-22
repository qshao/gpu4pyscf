"""One small real QM calculation to diagnose current GPU access; no fallback."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import sys
import time
import traceback

ROOT=Path('/home/qshao/gpu4pyscf')
sys.path.insert(0,str(ROOT))
OUT=ROOT/'benchmarks/single_gpu/qm_suite/access_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT.mkdir(parents=True,exist_ok=False)
(OUT/'probe.py').write_text(Path(__file__).read_text())
report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'target':'DGX Spark / NVIDIA GB10',
        'task':'H2O B3LYP/def2-SVP density-fitted SCF and full-response analytic gradient',
        'scope':'Single application access test, not a repeated performance benchmark. No GPU-to-CPU fallback.',
        'settings':{'atom':'O 0 0 0; H 0 -.757 .587; H 0 .757 .587','basis':'def2-svp',
                    'auxbasis':'def2-universal-jkfit','xc':'B3LYP','grid_level':3,
                    'conv_tol':1e-10,'conv_tol_grad':1e-7,'cpu_threads':4,'max_cycle':100},
        'device_nodes':[str(p) for p in Path('/dev').glob('nvidia*')],
        'gpu_calculation_completed':False}
from pyscf import gto,dft,lib,scf
import numpy as np
lib.num_threads(4)
s=report['settings']
mol=gto.M(atom=s['atom'],basis=s['basis'],verbose=0)
initial=scf.RHF(mol).get_init_guess(key='minao')
report['initial_density_sha256']=hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()

def configure(mf):
    mf.conv_tol=s['conv_tol'];mf.conv_tol_grad=s['conv_tol_grad'];mf.max_cycle=s['max_cycle']
    mf.grids.level=s['grid_level'];mf.small_rho_cutoff=0
    return mf

start=time.perf_counter()
cpu=configure(dft.RKS(mol,xc=s['xc']).density_fit(auxbasis=s['auxbasis']))
energy=cpu.kernel(dm0=initial)
grad=cpu.nuc_grad_method();grad.grid_response=True;grad.auxbasis_response=True
reference=grad.kernel()
report['cpu_reference']={'converged':bool(cpu.converged),'implementation':type(cpu).__module__+'.'+type(cpu).__name__,
                         'energy_hartree':float(energy),'gradient_hartree_per_bohr':reference.tolist(),
                         'elapsed_seconds':time.perf_counter()-start,'nao':mol.nao_nr()}
print('CPU application reference completed: energy',energy,'Eh; converged',cpu.converged,flush=True)
start=time.perf_counter();stage='importing native GPU4PySCF'
try:
    from gpu4pyscf.dft import rks
    import cupy as cp
    stage='constructing GPU density-fitted RKS'
    gpu=configure(rks.RKS(mol,xc=s['xc']).density_fit(auxbasis=s['auxbasis']))
    report['gpu_implementation']=type(gpu).__module__+'.'+type(gpu).__name__
    stage='uploading common initial density and running GPU SCF'
    ge=gpu.kernel(dm0=cp.asarray(initial))
    stage='running GPU analytic gradient'
    gg=gpu.nuc_grad_method();gg.grid_response=True;gg.auxbasis_response=True
    actual=cp.asnumpy(gg.kernel());cp.cuda.Device().synchronize()
    ee=abs(float(ge)-float(energy));gerr=float(np.max(np.abs(actual-reference)))
    report['gpu_result']={'converged':bool(gpu.converged),'energy_hartree':float(ge),'gradient_hartree_per_bohr':actual.tolist(),
                          'elapsed_seconds':time.perf_counter()-start,'energy_error_hartree':ee,'gradient_max_error':gerr,
                          'accuracy_passed':bool(gpu.converged and ee<=1e-8 and gerr<=1e-6)}
    report['gpu_calculation_completed']=True
    report['status']='gpu_application_passed' if report['gpu_result']['accuracy_passed'] else 'accuracy_failed'
except Exception as exc:
    report.update(status='gpu_application_failed',failed_stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                  gpu_attempt_elapsed_seconds=time.perf_counter()-start)
    print('GPU application attempt failed at:',stage,flush=True)
    print(type(exc).__name__+': '+str(exc),flush=True)
# Direct CUDA driver diagnostic is independent of nvidia-smi and CuPy's allocator.
try:
    driver=ctypes.CDLL('libcuda.so.1')
    code=int(driver.cuInit(0));report['cuda_driver_cuInit_return_code']=code
    if code==0:
        count=ctypes.c_int();code2=int(driver.cuDeviceGetCount(ctypes.byref(count)))
        report['cuda_driver_device_count']=count.value;report['cuda_driver_device_count_return_code']=code2
    print('CUDA driver cuInit return code:',code,flush=True)
except Exception as exc:report['cuda_driver_diagnostic_error']=str(exc)
(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('Evidence:',OUT/'result.json',flush=True)
print('GPU calculation completed:',report['gpu_calculation_completed'],flush=True)
