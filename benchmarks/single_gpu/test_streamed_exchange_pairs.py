"""Paired molecular J/K at a deliberately bounded post-build streaming budget."""
from contextlib import nullcontext
import statistics
import time
import traceback

import numpy as np
import pytest

from corrected_optimization import BASE,new_report,digest,save


@pytest.mark.special
@pytest.mark.benchmark
def test_streamed_exchange_pairs(tmp_path):
    import cupy as cp
    import molecular
    from pyscf import lib
    from comprehensive import configured,settings
    from gpu4pyscf.df.disk import DiskTensor
    from short_exchange import version
    from selected_jk import selected_jk
    output,report=new_report('streamed_exchange_pairs.json',__file__)
    case='valinomycin_svp_b3lyp'
    report.update(case=case,settings=settings(case),geometry_sha256=digest(settings(case)['atom']),
        cpu_reference_sha256=None,candidate_sha256=digest(BASE/'short_exchange.py'),
        transform_sha256=digest(BASE/'exchange_layout.py'),oracle_sha256=digest(BASE/'selected_jk.py'),
        build_buffer_bytes=4*1024**3,warmups=[],pairs=[],
        note='Single molecular J/K, not converged SCF/forces. Build uses4GiB, then existing streaming-buffer setting is reduced to exactly two32-row sparse buffers. Both candidates use identical32-row blocks and storage. Three alternating pairs after two warmups, no allocation instrumentation. Full J/K checked against original GPU and selected NumPy contractions of same native GPU integrals; CPU is not an independent integral oracle. Cache zero; final memory counters are iteration-stage staging, explicitly reset after preserving build-stage peaks. No whole-SCF speed or total-RAM claim.')
    save(output,report)
    store=mf=None
    try:
        with lib.with_omp_threads(4),configured(case):
            mf,_=molecular._mean_field('B3LYP','gpu')
            mf.with_df.cderi_storage='disk'; mf.with_df.use_gpu_memory=None
            mf.with_df.cderi_directory=tmp_path
            mf.with_df.cderi_buffer_bytes=report['build_buffer_bytes']; mf.with_df.cderi_cache_bytes=0
            mf.mo_energy,mf.mo_coeff=mf.eig(mf.get_hcore(),mf.get_ovlp())
            mf.mo_occ=mf.get_occ(mf.mo_energy,mf.mo_coeff)
            dm=mf.make_rdm1(); mf.with_df.build()
            store=mf.with_df._cderi[0]; assert isinstance(store,DiskTensor)
            report.update(build_stats=dict(store.stats),cderi_shape=list(store.shape),cderi_bytes=store.nbytes)
            budget=2*32*store.shape[1]*8
            mf.with_df.cderi_buffer_bytes=budget
            assert mf.with_df.get_blksize(mem_fraction=.4)==32
            report['stream_buffer_bytes']=budget
            store.stats['peak_host_buffer_bytes']=store.stats['peak_device_buffer_bytes']=0
            expected=None
            def run(candidate):
                nonlocal expected
                before=dict(store.stats)
                with version() if candidate=='strided' else nullcontext():
                    cp.cuda.Device().synchronize(); start=time.perf_counter()
                    j,k=mf.with_df.get_jk(dm,hermi=1)
                    cp.cuda.Device().synchronize(); elapsed=time.perf_counter()-start
                arrays=[cp.asnumpy(j),cp.asnumpy(k)]
                assert all(np.isfinite(a).all() for a in arrays)
                if expected is None: expected=arrays
                errors=[float(np.max(np.abs(a-b))) for a,b in zip(arrays,expected)]
                assert all(np.isfinite(e) and e<=1e-9 for e in errors),errors
                transferred=store.stats['h2d_bytes']-before['h2d_bytes']
                assert transferred==store.nbytes
                return dict(seconds=elapsed,errors=errors,h2d_bytes=transferred,
                            h2d_calls=store.stats['h2d_calls']-before['h2d_calls'])
            for candidate in ['baseline','strided']:
                report['warmups'].append(dict(variant=candidate,**run(candidate)))
                save(output,report)
            for repeat in range(3):
                order=['baseline','strided'] if repeat%2==0 else ['strided','baseline']
                report['pairs'].append({candidate:run(candidate) for candidate in order})
                save(output,report); print('molecular streamed exchange pair',repeat+1,flush=True)
            selected=np.unique(np.linspace(0,mf.mol.nao-1,8,dtype=int))
            factor=cp.asnumpy(mf.mo_coeff[:,mf.mo_occ>0]*cp.sqrt(mf.mo_occ[mf.mo_occ>0]))
            addresses,diagonal=map(cp.asnumpy,mf.with_df._cderi_idx)
            j,k=selected_jk(store,addresses,diagonal,cp.asnumpy(dm),factor,selected)
            errors=[float(np.max(np.abs(a-b[np.ix_(selected,selected)]))) for a,b in zip([j,k],expected)]
            assert all(np.isfinite(e) and e<=1e-8 for e in errors),errors
            assert store.stats['cache_device_bytes']==0
            assert store.stats['peak_host_buffer_bytes']<=budget and store.stats['peak_device_buffer_bytes']<=budget
            medians={candidate:statistics.median(pair[candidate]['seconds'] for pair in report['pairs'])
                     for candidate in ['baseline','strided']}
            report.update(medians=medians,speedup=medians['baseline']/medians['strided'],
                paired_ratios=[pair['baseline']['seconds']/pair['strided']['seconds'] for pair in report['pairs']],
                cpu_contraction_errors=errors,stream_stats=dict(store.stats),status='passed')
            path=store.path;store.close();assert not path.exists()
            mf.with_df.reset(); report['scratch_clean']=True
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    finally:
        if isinstance(store,DiskTensor): store.close()
        if mf is not None: mf.with_df.reset()
    save(output,report)
    print('molecular streamed J/K',report['medians'],report['speedup'],report['cpu_contraction_errors'],flush=True)
