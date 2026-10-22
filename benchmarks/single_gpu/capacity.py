"""Large-system GPU differential qualification with bounded disk CDERI."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import weakref
from unittest.mock import patch

import numpy as np

import molecular
from comprehensive import BASE,ROOT,RESULTS,SOURCES,CASES,settings,configured,save
from molecular import _metadata,_file_identity,run_case,compare_cases

LARGE_CASES=['tamoxifen_tzvp_b3lyp','taxol_svp_b3lyp',
             'valinomycin_svp_pbe','valinomycin_svp_b3lyp']


def configuration(case):
    if case not in LARGE_CASES: raise ValueError('Unknown capacity case: '+case)
    return dict(scientific=settings(case),storage='disk' if 'valinomycin' in case else 'resident',
                buffer_bytes=4*1024**3,cache_bytes=0)


def execute(case,variant,storage=None,reference_storage=None):
    if variant not in ['baseline','packed']: raise ValueError('Unknown variant: '+variant)
    config=configuration(case)
    if storage is not None:
        if storage not in ['resident','disk']: raise ValueError('Unknown storage: '+storage)
        config['storage']=storage
    if reference_storage is not None and (variant!='packed' or reference_storage not in ['resident','disk']):
        raise ValueError('Reference storage is only supported for explicit candidate comparisons')
    path=RESULTS/(case+'_capacity_'+config['storage']+'_'+variant+'.json')
    if path.exists(): raise FileExistsError('Archive report explicitly before rerunning: '+str(path))
    orbital_path=path.with_suffix('.npz')
    if orbital_path.exists(): raise FileExistsError('Archive orbital snapshot explicitly: '+str(orbital_path))
    import cupy as cp
    from gpu4pyscf.df import df
    from gpu4pyscf.df.grad import rhf
    from gpu4pyscf.grad import rhf as rhf_grad
    from gpu4pyscf.df.disk import DiskTensor
    from occupied_pair import packed_gradient
    from memory_profile import peak_memory
    from gradient_snapshot import write_orbitals
    from pyscf import gto
    from pyscf.df.addons import make_auxmol
    report=dict(_metadata('gpu'),status='running',case=case,variant=variant,configuration=config,
        benchmark_sha256=_file_identity(__file__)['sha256'],
        candidate_sha256=_file_identity(BASE/'occupied_pair.py')['sha256'],
        configuration_sha256=_file_identity(BASE/'comprehensive.py')['sha256'],
        memory_profiler_sha256=_file_identity(BASE/'memory_profile.py')['sha256'],
        orbital_helper_sha256=_file_identity(BASE/'gradient_snapshot.py')['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES+['gpu4pyscf/df/disk.py']},
        geometry_sha256=_file_identity(config['scientific']['atom'])['sha256'],
        cpu_qualification='pending',packed_unpack_calls=0,
        note='Fresh-process allocation run; timings diagnostic, not repeated speed qualification. Includes a small host orbital snapshot at gradient entry. Original GPU baseline is an algorithm-preservation reference, not an independent CPU oracle. Candidate gates1e-10Eh/1e-9Eh/Bohr relative to original GPU, same iterations. CPU gates1e-8Eh/1e-6Eh/Bohr checked separately when available. Full grid/auxiliary response. Zero experimental cache. Memory requested through CuPy excludes native/preexisting storage and global OS file cache; process VmHWM separate. Triangular CDERI size is an estimate: within-shell full blocks can add entries. Full-square CDERI size is an upper bound on stored AO pairs, not build workspaces or hardware capacity.')
    save(path,report)
    stores=[]
    try:
        mol=gto.M(atom=config['scientific']['atom'],basis=config['scientific']['basis'],verbose=0)
        aux=make_auxmol(mol,config['scientific']['auxbasis'])
        report.update(atoms=mol.natm,nao=mol.nao,naux=aux.nao,nocc=mol.nelectron//2,
                      dense_triangle_cderi_estimate_bytes=8*aux.nao*mol.nao*(mol.nao+1)//2,
                      full_square_cderi_upper_bytes=8*aux.nao*mol.nao**2)
        if variant=='baseline':
            archive=BASE/'runs/2026-10-17-pre-comprehensive/gpu4pyscf__df__grad__rhf.py.source'
            assert _file_identity(archive)['sha256']==report['algorithm_source_hashes']['gpu4pyscf/df/grad/rhf.py']
        else:
            reference_storage=reference_storage or config['storage']
            reference=RESULTS/(case+'_capacity_'+reference_storage+'_baseline.json')
            baseline=json.loads(reference.read_text())
            assert baseline['status']=='captured_original_gpu'
            for key in ['geometry_sha256','native_libraries','algorithm_source_hashes',
                        'memory_profiler_sha256','pyscf','numpy','cpu_libxc','cupy']:
                assert baseline[key]==report[key],key
            for key in ['scientific','buffer_bytes','cache_bytes']:
                assert baseline['configuration'][key]==config[key],key
            assert baseline['configuration']['storage']==reference_storage
            report['gpu_reference_storage']=reference_storage
            report['gpu_reference_sha256']=_file_identity(reference)['sha256']
        original_build=df._cholesky_eri
        def build(*args,**kwargs):
            result=original_build(*args,**kwargs)
            stores.extend(dict(shape=list(s.shape),nbytes=s.nbytes,
                               stats=s.stats if isinstance(s,DiskTensor) else None,
                               reference=weakref.ref(s)) for s in result[0])
            return result
        scopes=[]
        with ExitStack() as stack:
            stack.enter_context(configured(case))
            factory=molecular._mean_field
            def mean_field(method,backend):
                mf,dm0=factory(method,backend)
                mf.with_df.cderi_storage='disk' if config['storage']=='disk' else None
                mf.with_df.use_gpu_memory=None if config['storage']=='disk' else True
                mf.with_df.cderi_buffer_bytes=config['buffer_bytes']
                mf.with_df.cderi_cache_bytes=0
                return mf,dm0
            stack.enter_context(patch.object(molecular,'_mean_field',mean_field))
            stack.enter_context(patch.object(df,'_cholesky_eri',build))
            original_kernel=rhf_grad.GradientsBase.kernel
            def gradient_kernel(grad,*args,**kwargs):
                if 'orbitals' not in report:
                    write_orbitals(grad.base,orbital_path)
                    report['orbitals']=_file_identity(orbital_path)
                return original_kernel(grad,*args,**kwargs)
            stack.enter_context(patch.object(rhf_grad.GradientsBase,'kernel',gradient_kernel))
            if variant=='packed':
                stack.enter_context(packed_gradient())
                unpack=rhf._occupied_unpack
                def counted(*args,**kwargs):
                    report['packed_unpack_calls']+=1
                    return unpack(*args,**kwargs)
                stack.enter_context(patch.object(rhf,'_occupied_unpack',counted))
            original=rhf._jk_energy_per_atom
            def jk(*args,**kwargs):
                with peak_memory() as memory: value=original(*args,**kwargs)
                scopes.append(memory)
                return value
            stack.enter_context(patch.object(rhf,'_jk_energy_per_atom',jk))
            assert cp.get_default_memory_pool().used_bytes()==0
            with peak_memory() as memory:
                result=run_case(CASES[case][2],'gpu')
        report.update(result=result,memory=memory,jk_memory=scopes,
                      storage=[dict(shape=s['shape'],nbytes=s['nbytes'],stats=s['stats']) for s in stores])
        assert all(s['reference']() is None for s in stores)
        report['cderi_released_by_gradient']=True
        if config['storage']=='disk':
            for store in report['storage']:
                stats=store['stats']
                assert stats['peak_host_buffer_bytes']<=config['buffer_bytes']
                assert stats['peak_device_buffer_bytes']<=config['buffer_bytes']
                assert stats['d2h_bytes']==store['nbytes'] and stats['cache_device_bytes']==0
        cpu_path=RESULTS/(case+'_cpu.json')
        if cpu_path.exists():
            cpu=json.loads(cpu_path.read_text())
            if cpu['status']=='passed':
                assert cpu['geometry_sha256']==report['geometry_sha256']
                report['cpu_reference_sha256']=_file_identity(cpu_path)['sha256']
                report['cpu_comparison']=compare_cases(cpu['result'],result)
                assert report['cpu_comparison']['passed'],report['cpu_comparison']
                report['cpu_qualification']='passed'
        if variant=='packed':
            left=baseline['result']
            error=dict(energy_hartree=abs(result['energy_hartree']-left['energy_hartree']),
                       gradient_max=float(np.max(np.abs(np.array(result['gradient_hartree_per_bohr'])-np.array(left['gradient_hartree_per_bohr'])))))
            assert error['energy_hartree']<=1e-10 and error['gradient_max']<=1e-9,error
            assert result['iterations']==left['iterations']
            if CASES[case][2]=='B3LYP': assert report['packed_unpack_calls']>0
            report.update(gpu_comparison=error,status='passed_differential')
        else:
            report['status']='captured_original_gpu'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        save(path,report)
        raise
    finally:
        for item in stores:
            live=item['reference']()
            if isinstance(live,DiskTensor): live.close()
    save(path,report)
    return report
