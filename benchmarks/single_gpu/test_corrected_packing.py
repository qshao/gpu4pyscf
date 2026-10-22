"""Packing qualification at identical, independently checked corrected orbitals."""
from contextlib import ExitStack,nullcontext
import statistics
import time
import traceback
from unittest.mock import patch

import pytest

from corrected_optimization import BASE,CASE,new_report,gradient_inputs,check_gradient,digest,save


def setup(script,name,variant):
    profile,identity=gradient_inputs()
    output,report=new_report(name,script)
    report.update(variant=variant,profile_sha256=identity,orbitals=profile['orbitals'],
        orbital_helper_sha256=digest(BASE/'gradient_snapshot.py'),
        note='Gradient only at identical corrected, CPU-qualified orbitals. Original CPU1e-8Eh/1e-6 force gates and strict corrected-GPU1e-9 force differential. Fresh-process allocation observations are separate from uninstrumented alternating timing. Gradient speed is not whole-workflow speed; no repeated SCF in this experiment.')
    return output,report,profile


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('variant',['baseline','packed'])
def test_corrected_gradient_memory(variant):
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf.df.grad import rhf
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    from memory_profile import peak_memory
    output,report,profile=setup(__file__,f'gradient_{variant}_memory.json',variant)
    report['memory_profiler_sha256']=digest(BASE/'memory_profile.py')
    save(output,report)
    try:
        assert cp.get_default_memory_pool().used_bytes()==0
        with lib.with_omp_threads(4):
            mf=restore(CASE,profile['orbitals']['path'])
            scopes=[]
            with ExitStack() as stack:
                if variant=='packed': stack.enter_context(packed_gradient())
                original=rhf._jk_energy_per_atom
                def jk(*args,**kwargs):
                    with peak_memory() as observed: value=original(*args,**kwargs)
                    scopes.append(observed)
                    return value
                stack.enter_context(patch.object(rhf,'_jk_energy_per_atom',jk))
                grad=mf.nuc_grad_method(); grad.grid_response=grad.auxbasis_response=True
                with peak_memory() as observed: actual=grad.kernel()
        comparison=check_gradient(profile['result'],actual)
        report.update(memory=observed,jk_memory=scopes,comparison=comparison,status='passed')
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('corrected gradient memory',variant,observed['peak_tracked_live_bytes'],comparison['error_to_corrected_gpu'],flush=True)


@pytest.mark.special
@pytest.mark.benchmark
def test_corrected_gradient_pairs():
    import cupy as cp
    from pyscf import lib
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    output,report,profile=setup(__file__,'gradient_pairs.json','paired')
    report.update(warmups=[],pairs=[])
    save(output,report)
    try:
        with lib.with_omp_threads(4):
            mf=restore(CASE,profile['orbitals']['path'])
            def run(version):
                with packed_gradient() if version=='packed' else nullcontext():
                    grad=mf.nuc_grad_method(); grad.grid_response=grad.auxbasis_response=True
                    cp.cuda.Device().synchronize(); start=time.perf_counter()
                    actual=grad.kernel()
                    cp.cuda.Device().synchronize(); elapsed=time.perf_counter()-start
                return dict(seconds=elapsed,**check_gradient(profile['result'],actual))
            for version in ['baseline','packed']:
                report['warmups'].append(dict(variant=version,**run(version)))
                save(output,report)
            for repeat in range(3):
                versions=['baseline','packed'] if repeat%2==0 else ['packed','baseline']
                report['pairs'].append({version:run(version) for version in versions})
                save(output,report)
                print('corrected gradient pair',repeat+1,flush=True)
        medians={version:statistics.median(pair[version]['seconds'] for pair in report['pairs'])
                 for version in ['baseline','packed']}
        report.update(medians=medians,speedup=medians['baseline']/medians['packed'],
            paired_ratios=[p['baseline']['seconds']/p['packed']['seconds'] for p in report['pairs']],status='passed')
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('corrected gradient medians',report['medians'],report['speedup'],flush=True)
