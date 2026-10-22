"""Disk capacity and selected CPU contraction checks without a full SCF run."""

import numpy as np
import pytest

from comprehensive import BASE,ROOT,SOURCES,RESULTS,configured,settings,save
from molecular import _metadata,_file_identity


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case,mode',[('vitamin_svp_b3lyp','disk'),
                                    ('valinomycin_svp_b3lyp','disk'),
                                    ('taxol_svp_b3lyp','host')])
def test_jk_capacity(case,mode,tmp_path):
    import cupy as cp
    from pyscf import lib
    import molecular
    from memory_profile import peak_memory
    from selected_jk import selected_jk
    from gpu4pyscf.df.disk import DiskTensor
    output=RESULTS/(case+'_'+mode+'_jk_capacity.json')
    if output.exists(): raise FileExistsError('Archive previous report explicitly')
    budget=(128*1024**2 if 'vitamin' in case else 4*1024**3)
    config=settings(case)
    report=dict(_metadata('gpu'),status='running',case=case,mode=mode,buffer_bytes=budget,
        staging_budget_applies=(mode=='disk'),
        cache_bytes=0,settings=config,
        benchmark_sha256=_file_identity(__file__)['sha256'],
        oracle_sha256=_file_identity(BASE/'selected_jk.py')['sha256'],
        configuration_sha256=_file_identity(BASE/'comprehensive.py')['sha256'],
        memory_profiler_sha256=_file_identity(BASE/'memory_profile.py')['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES+['gpu4pyscf/df/disk.py']},
        geometry_sha256=_file_identity(config['atom'])['sha256'],
        note='CAPACITY/CONTRACTION CHECK ONLY: one J/K evaluation at a non-self-consistent closed-shell one-electron orbital guess. No converged SCF energy/forces or production-grid qualification claimed. CPU oracle contracts the same GPU-generated DF integrals for selected J/K elements; it is not a CPU integral oracle. Raw CPU oracle row buffer<=256MiB, with additional density/factor/lookup/selected-row/BLAS storage. Allocation scope includes orbital initialization, DF construction and one GPU J/K; excludes subsequent CPU oracle. Zero experimental cache. Timings are not a performance acceptance benchmark.')
    save(output,report)
    store=reader=mf=None
    try:
        assert cp.get_default_memory_pool().used_bytes()==0
        with lib.with_omp_threads(4),configured(case):
            with peak_memory() as memory:
                mf,_=molecular._mean_field('B3LYP','gpu')
                mf.with_df.cderi_storage='disk' if mode=='disk' else None
                mf.with_df.use_gpu_memory=None if mode=='disk' else False
                mf.with_df.cderi_buffer_bytes=budget
                mf.with_df.cderi_cache_bytes=0
                mf.with_df.cderi_directory=tmp_path
                mf.mo_energy,mf.mo_coeff=mf.eig(mf.get_hcore(),mf.get_ovlp())
                mf.mo_occ=mf.get_occ(mf.mo_energy,mf.mo_coeff)
                dm=mf.make_rdm1()
                mf.with_df.build()
                store=mf.with_df._cderi[0]
                j,k=mf.with_df.get_jk(dm,hermi=1)
            store=mf.with_df._cderi[0]
            assert isinstance(store,DiskTensor) if mode=='disk' else isinstance(store,np.ndarray)
            stats=store.stats.copy() if mode=='disk' else None
            if mode=='disk':
                assert stats['d2h_bytes']==store.nbytes and stats['cache_device_bytes']==0
                assert stats['peak_device_buffer_bytes']<=budget and stats['peak_host_buffer_bytes']<=budget
            class HostRows:
                def __init__(self,array):
                    self.array=array; self.shape=array.shape; self.read_bytes=0
                def read_rows(self,a,b,out):
                    out[:]=self.array[a:b]; self.read_bytes+=out.nbytes; return out
            reader=store if mode=='disk' else HostRows(store)
            nao=mf.mol.nao
            selected=np.unique(np.linspace(0,nao-1,8,dtype=int))
            mask=mf.mo_occ>0
            factor=cp.asnumpy(mf.mo_coeff[:,mask]*cp.sqrt(mf.mo_occ[mask]))
            pair_addresses,diagonal=map(cp.asnumpy,mf.with_df._cderi_idx)
            expected_j,expected_k=selected_jk(reader,pair_addresses,diagonal,cp.asnumpy(dm),factor,selected)
            index=cp.asarray(selected)
            actual_j=cp.asnumpy(j[index[:,None],index])
            actual_k=cp.asnumpy(k[index[:,None],index])
            errors=dict(j_max=float(np.max(np.abs(actual_j-expected_j))),
                        k_max=float(np.max(np.abs(actual_k-expected_k))))
            assert errors['j_max']<=1e-8 and errors['k_max']<=1e-8,errors
            oracle_read_bytes=(store.stats['read_bytes']-stats['read_bytes'] if mode=='disk' else reader.read_bytes)
            assert oracle_read_bytes==store.nbytes
            report.update(memory=memory,nao=nao,naux=store.shape[0],nocc=int(mask.sum()),
                cderi_shape=list(store.shape),cderi_bytes=store.nbytes,selected=selected.tolist(),
                gpu_jk_stats=stats,oracle_read_bytes=oracle_read_bytes,
                oracle_buffer_bytes=256*1024**2,errors=errors,
                selected_gpu_j=actual_j.tolist(),selected_gpu_k=actual_k.tolist(),
                selected_cpu_j=expected_j.tolist(),selected_cpu_k=expected_k.tolist())
            if mode=='disk':
                file_path=store.path
                store.close()
                assert not file_path.exists()
            reader=None
            mf.with_df.reset()
            report.update(cderi_file_removed=(mode=='disk'),scratch_clean=True,status='passed')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save(output,report)
        raise
    finally:
        if isinstance(store,DiskTensor): store.close()
        if mf is not None: mf.with_df.reset()
        store=reader=None
        assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))
    save(output,report)
    print(case,report['nao'],report['cderi_bytes'],report['memory']['peak_tracked_live_bytes'],report['errors'],flush=True)
