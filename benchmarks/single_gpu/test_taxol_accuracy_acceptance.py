"""Full SCF/force acceptance for the corrected resident and disk workflows."""
import json
import traceback
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from comprehensive import BASE, ROOT, RESULTS, SOURCES, configured, settings, save
from molecular import _file_identity, _metadata, run_case
from seeded_reference import compare_execution_cases


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('storage',['resident','disk'])
def test_taxol_accuracy(storage,tmp_path):
    import molecular
    from gpu4pyscf.df import df
    from gpu4pyscf.df.disk import DiskTensor
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    cpu_path=RESULTS/'taxol_svp_b3lyp_seeded_cpu.json'
    parent=json.loads(parent_path.read_text()); cpu=json.loads(cpu_path.read_text())
    assert cpu['status']=='passed'
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    assert cpu['geometry_sha256']==_file_identity(settings('taxol_svp_b3lyp')['atom'])['sha256']
    with np.load(parent['orbitals']['path'],allow_pickle=False) as archive:
        density=(archive['mo_coeff']*archive['mo_occ'])@archive['mo_coeff'].T
    output=BASE/'accuracy_results'/('taxol_'+storage+'_acceptance.json')
    if output.exists(): raise FileExistsError('Archive existing acceptance report explicitly')
    budget=4*1024**3
    report=dict(_metadata('gpu'),status='running',storage=storage,
        benchmark_sha256=_file_identity(__file__)['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES+['gpu4pyscf/df/disk.py']},
        configuration_sha256=_file_identity(BASE/'comprehensive.py')['sha256'],
        comparison_sha256=_file_identity(BASE/'seeded_reference.py')['sha256'],
        cpu_reference_sha256=_file_identity(cpu_path)['sha256'],
        seed_report_sha256=_file_identity(parent_path)['sha256'],seed=parent['orbitals'],
        buffer_bytes=budget,cache_bytes=0,
        note='Full production SCF and full grid/auxiliary-response gradients. Prior GPU orbitals form only the initial density; fresh DF integrals/Fock/grid are rebuilt and SCF must converge at original1e-10 energy/1e-7 orbital thresholds. Original independent CPU gates1e-8Eh/1e-6Eh/Bohr remain unchanged. CPU thread count is the only excluded execution-setting difference. No packing prototype or tightened-SCF workaround; no speed-ratio claim.')
    save(output,report)
    stores=[]; refs=[]
    build=df._cholesky_eri
    def observe(*args,**kwargs):
        result=build(*args,**kwargs)
        for store in result[0]:
            refs.append(weakref.ref(store))
            stores.append(dict(shape=list(store.shape),nbytes=store.nbytes,
                               disk=isinstance(store,DiskTensor),
                               stats=store.stats if isinstance(store,DiskTensor) else None))
        return result
    try:
        with configured('taxol_svp_b3lyp'):
            factory=molecular._mean_field
            def initialize(method,backend):
                mf,_=factory(method,backend)
                mf.with_df.use_gpu_memory=True if storage=='resident' else None
                mf.with_df.cderi_storage=None if storage=='resident' else 'disk'
                mf.with_df.cderi_directory=tmp_path
                mf.with_df.cderi_buffer_bytes=budget
                mf.with_df.cderi_cache_bytes=0
                return mf,density.copy()
            with patch.object(molecular,'_mean_field',initialize),patch.object(df,'_cholesky_eri',observe):
                result=run_case('B3LYP','gpu')
        comparison=compare_execution_cases(cpu['result'],result)
        report.update(result=result,comparison=comparison,integrals=stores)
        assert comparison['passed'],comparison
        assert all(reference() is None for reference in refs)
        assert stores and all(store['disk']==(storage=='disk') for store in stores)
        if storage=='disk':
            for store in stores:
                stats=store['stats']
                assert stats['peak_device_buffer_bytes']<=budget
                assert stats['peak_host_buffer_bytes']<=budget
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
    print('Taxol corrected',storage,comparison,result['iterations'],result['scf_seconds'],result['gradient_seconds'],flush=True)
