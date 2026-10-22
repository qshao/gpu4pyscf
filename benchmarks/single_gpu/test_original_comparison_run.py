"""One complete original/current workflow per fresh pytest process."""
from contextlib import nullcontext
import gc
import hashlib
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import pytest

from original_comparison import OUT,BASE,manifest,digest,save,case_settings,configure_cpu,cpu_reference


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('version,size,mode,repeat',[
    (version,size,mode,repeat)
    for size,version in [(s,v) for s in ['small','medium','large'] for v in ['original','current']]+[('large','disk')]
    for mode,repeat in [('time',0),('time',1),('time',2),('memory',0)]])
def test_original_comparison_run(tmp_path,version,size,mode,repeat):
    assert version in ['original','current','disk'] and mode in ['time','memory']
    if version=='disk': assert size=='large'
    frozen=manifest();package=frozen['packages']['original' if version=='original' else 'current']
    assert not any(name=='gpu4pyscf' or name.startswith('gpu4pyscf.') for name in sys.modules)
    sys.path.insert(0,str(Path(package['path']).parent))
    import cupy as cp
    import gpu4pyscf
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    from gpu4pyscf.df import df
    from pyscf import lib
    from molecular import _metadata
    from memory_profile import peak_memory
    from seeded_reference import compare_execution_cases
    assert Path(gpu4pyscf.__file__).resolve()==Path(package['path'])/'__init__.py'
    assert cp.cuda.runtime.getDeviceCount()==1
    output=OUT/'results'/f'{size}_{version}_{mode}_{repeat}.json'
    if output.exists(): raise FileExistsError('Raw result already exists: '+str(output))
    reference_path,reference=cpu_reference(size)
    config=case_settings(size)
    metadata=_metadata('gpu')
    metadata['metadata_helper_source_hashes']=metadata.pop('source_hashes')
    report=dict(metadata,status='running',size=size,version=version,mode=mode,repeat=repeat,
        settings=config,manifest_sha256=digest(OUT/'manifest.json'),
        benchmark_sha256=digest(__file__),configuration_sha256=digest(BASE/'original_comparison.py'),
        cpu_reference_sha256=digest(reference_path),package=package['path'],
        geometry_sha256=digest(config['atom']),native_memory_before=list(cp.cuda.runtime.memGetInfo()),
        scope='Fresh molecule/atomic initial guess, fresh DF construction, original-threshold SCF and full grid/auxiliary-response gradient. Excludes imports/provenance and final cleanup. No converged seed or CDERI reuse. Three independent timing processes; memory runs use allocator instrumentation and have diagnostic timing only. RSS and requested CuPy storage are distinct on unified-memory GB10; native/preexisting storage and unmapped OS file cache are excluded from the CuPy ledger.')
    save(output,report)
    stores=[];refs=[];paths=[];iterations=[]
    import weakref
    build=df._cholesky_eri
    def observe(*args,**kwargs):
        value=build(*args,**kwargs)
        for store in value[0]:
            refs.append(weakref.ref(store))
            is_disk=hasattr(store,'stats')
            stores.append(dict(shape=list(store.shape),nbytes=store.nbytes,is_disk=is_disk,
                gpu_array=isinstance(store,cp.ndarray),stats=store.stats if is_disk else None))
            if is_disk: paths.append(store.path)
        return value
    mf=None
    try:
        with lib.with_omp_threads(4),patch.object(df,'_cholesky_eri',observe):
            cp.cuda.Device().synchronize()
            with peak_memory() if mode=='memory' else nullcontext({}) as memory:
                start=time.perf_counter()
                cpu=configure_cpu(size)
                dm=cpu.get_init_guess()
                dm_sha=hashlib.sha256(np.ascontiguousarray(dm).tobytes()).hexdigest()
                mf=cpu.to_gpu()
                mf.grids.prune=nwchem_prune
                assert mf.grids.prune is nwchem_prune and mf.grids.level==3
                assert mf.max_cycle==100 and mf.conv_tol==1e-10 and mf.conv_tol_grad==1e-7
                assert mf.small_rho_cutoff==0.
                mf.with_df.use_gpu_memory=None if version=='disk' else True
                if version!='original':
                    mf.with_df.cderi_storage='disk' if version=='disk' else None
                    mf.with_df.cderi_directory=tmp_path
                    mf.with_df.cderi_buffer_bytes=4*1024**3
                    mf.with_df.cderi_cache_bytes=0
                setup=time.perf_counter()-start
                def callback(environment):
                    cp.cuda.Device().synchronize()
                    iterations.append(dict(cycle=int(environment['cycle']),energy_hartree=float(environment['e_tot'])))
                    if len(iterations)%5==0:
                        print('SCF progress',size,version,mode,repeat,len(iterations),flush=True)
                mf.callback=callback
                scf_start=time.perf_counter();energy=mf.kernel(dm0=dm)
                cp.cuda.Device().synchronize();scf_seconds=time.perf_counter()-scf_start
                if not mf.converged: raise RuntimeError('SCF failed to converge')
                grad=mf.nuc_grad_method()
                grad.grid_response=grad.auxbasis_response=True
                grad_start=time.perf_counter();gradient=grad.kernel()
                cp.cuda.Device().synchronize();gradient_seconds=time.perf_counter()-grad_start
                seconds=time.perf_counter()-start
                actual=np.asarray(gradient.get() if hasattr(gradient,'get') else gradient)
                result=dict(method='B3LYP',settings=config,converged=bool(mf.converged),
                    energy_hartree=float(energy),gradient_hartree_per_bohr=actual.tolist())
            report.update(result=result,setup_seconds=setup,scf_seconds=scf_seconds,
                gradient_seconds=gradient_seconds,workflow_seconds=seconds,iterations=iterations,
                initial_density_sha256=dm_sha,nao=mf.mol.nao,natm=mf.mol.natm,
                actual_factory_settings=dict(grid_pruning='nwchem',grid_level=mf.grids.level,
                    max_cycle=mf.max_cycle,conv_tol=mf.conv_tol,conv_tol_grad=mf.conv_tol_grad,small_rho_cutoff=mf.small_rho_cutoff),
                integrals=stores,memory=memory)
        comparison=compare_execution_cases(reference['result'],result)
        report['comparison']=comparison
        save(output,report)
        assert np.isfinite(actual).all() and np.isfinite(energy)
        assert stores and all(store['is_disk']==(version=='disk') for store in stores)
        if version!='disk': assert all(store['gpu_array'] for store in stores)
        else:
            for store in stores:
                stats=store['stats']
                assert stats['peak_host_buffer_bytes']<=4*1024**3
                assert stats['peak_device_buffer_bytes']<=4*1024**3
                assert stats['cache_device_bytes']==0 and stats['d2h_bytes']==store['nbytes']
        mf.with_df.reset()
        del grad,mf;mf=None;gc.collect();cp.cuda.Device().synchronize()
        assert all(reference() is None for reference in refs)
        assert all(not path.exists() for path in paths)
        assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))
        for name,module in list(sys.modules.items()):
            if name=='gpu4pyscf' or name.startswith('gpu4pyscf.'):
                path=getattr(module,'__file__',None)
                if path and path.endswith('.py'):
                    relative=Path(path).resolve().relative_to(package['path'])
                    assert digest(path)==package['sources'][str(relative)]
        report.update(status='passed' if comparison['passed'] else 'failed_accuracy',scratch_clean=True)
        save(output,report)
        print('Original/current workflow',size,version,mode,repeat,seconds,comparison,flush=True)
        assert comparison['passed'],comparison
    except BaseException as error:
        if report['status']!='failed_accuracy': report['status']='failed_execution'
        report.update(error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    finally:
        if mf is not None: mf.with_df.reset()
