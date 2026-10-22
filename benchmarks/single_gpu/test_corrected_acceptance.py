"""Full production SCF/force checks after selective optimization promotion."""
import traceback
import weakref
from unittest.mock import patch

import pytest

from corrected_optimization import BASE,ROOT,CASE,new_report,seed_density,oracle,digest,save
from seeded_reference import compare_execution_cases


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('storage',['resident','disk'])
def test_corrected_optimization_acceptance(storage,tmp_path):
    import molecular
    from comprehensive import configured
    from gpu4pyscf.df import df
    from gpu4pyscf.df.disk import DiskTensor
    density,seed,parent=seed_density()
    cpu_path,cpu=oracle()
    output,report=new_report(f'production_{storage}_qualified_acceptance.json',__file__)
    report.update(storage=storage,seed=seed,seed_parent_sha256=parent,
                  xc_source_sha256=digest(ROOT/'gpu4pyscf/grad/rks.py'),
                  disk_source_sha256=digest(ROOT/'gpu4pyscf/df/disk.py'),
                  buffer_bytes=4*1024**3,cache_bytes=0,
                  note='Fresh production SCF/integrals/grids at original scientific settings; saved orbitals supply only the initial density. Independent CPU1e-8Eh/1e-6Eh/Bohr gates. Diagnostic single timings are not a paired workflow speed claim.')
    save(output,report)
    refs=[]; stores=[]
    original_build=df._cholesky_eri
    def observe(*args,**kwargs):
        value=original_build(*args,**kwargs)
        for store in value[0]:
            refs.append(weakref.ref(store))
            stores.append(dict(shape=list(store.shape),nbytes=store.nbytes,
                               disk=isinstance(store,DiskTensor),
                               stats=store.stats if isinstance(store,DiskTensor) else None))
        return value
    try:
        with configured(CASE):
            factory=molecular._mean_field
            def initialize(method,backend):
                from gpu4pyscf.dft.gen_grid import nwchem_prune
                mf,_=factory(method,backend)
                assert mf.grids.prune is nwchem_prune
                assert mf.max_cycle==100 and mf.grids.level==3
                assert mf.small_rho_cutoff==0. and mf.conv_tol==1e-10 and mf.conv_tol_grad==1e-7
                report['actual_factory_settings']=dict(grid_pruning='nwchem',grid_level=mf.grids.level,
                    max_cycle=mf.max_cycle,small_rho_cutoff=mf.small_rho_cutoff,
                    conv_tol=mf.conv_tol,conv_tol_grad=mf.conv_tol_grad)
                mf.with_df.use_gpu_memory=True if storage=='resident' else None
                mf.with_df.cderi_storage=None if storage=='resident' else 'disk'
                mf.with_df.cderi_directory=tmp_path
                mf.with_df.cderi_buffer_bytes=report['buffer_bytes']
                mf.with_df.cderi_cache_bytes=0
                save(output,report)
                return mf,density.copy()
            with patch.object(molecular,'_mean_field',initialize),patch.object(df,'_cholesky_eri',observe):
                result=molecular.run_case('B3LYP','gpu')
        comparison=compare_execution_cases(cpu['result'],result)
        report.update(result=result,comparison=comparison,integrals=stores)
        save(output,report)
        assert comparison['passed'],comparison
        assert all(reference() is None for reference in refs)
        assert stores and all(store['disk']==(storage=='disk') for store in stores)
        if storage=='disk':
            for store in stores:
                stats=store['stats']
                assert stats['peak_device_buffer_bytes']<=report['buffer_bytes']
                assert stats['peak_host_buffer_bytes']<=report['buffer_bytes']
                assert stats['d2h_bytes']==store['nbytes'] and stats['cache_device_bytes']==0
        assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))
        report.update(status='passed',scratch_clean=True)
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    finally:
        for reference in refs:
            live=reference()
            if isinstance(live,DiskTensor): live.close()
    save(output,report)
    print('Production acceptance',storage,comparison,flush=True)
