"""Large gradient-only allocation and paired timing from verified orbitals."""
from contextlib import ExitStack,nullcontext
import json
import statistics
import time
from unittest.mock import patch

import numpy as np
import pytest

from comprehensive import BASE,ROOT,RESULTS,SOURCES,settings,save
from molecular import _metadata,_file_identity,compare_cases


def inputs(case):
    original_path=RESULTS/(case+'_capacity_resident_baseline.json')
    original=json.loads(original_path.read_text())
    assert original['status']=='captured_original_gpu'
    snapshot_path=original_path
    snapshot=original
    if 'orbitals' not in snapshot:
        snapshot_path=RESULTS/(case+'_capacity_disk_packed.json')
        snapshot=json.loads(snapshot_path.read_text())
        assert snapshot['status']=='passed_differential'
        assert snapshot['gpu_reference_sha256']==_file_identity(original_path)['sha256']
    identity=snapshot['orbitals']
    assert _file_identity(identity['path'])==identity
    metadata=_metadata('gpu')
    for key in ['pyscf','numpy','cpu_libxc','cupy','native_libraries','source_hashes']:
        assert metadata[key]==original[key]==snapshot[key],key
    assert snapshot.get('orbital_helper_sha256')==_file_identity(BASE/'gradient_snapshot.py')['sha256']
    for path,sha in original['algorithm_source_hashes'].items():
        assert _file_identity(ROOT/path)['sha256']==sha,path
    assert _file_identity(settings(case)['atom'])['sha256']==original['geometry_sha256']
    assert original['result']['settings']==settings(case)
    report=dict(metadata,case=case,status='running',
        benchmark_sha256=_file_identity(__file__)['sha256'],
        baseline_sha256=_file_identity(original_path)['sha256'],
        snapshot_report_sha256=_file_identity(snapshot_path)['sha256'],
        orbitals=identity,settings=settings(case),
        helper_hashes={p:_file_identity(BASE/p)['sha256'] for p in
            ['comprehensive.py','gradient_snapshot.py','occupied_pair.py','memory_profile.py']},
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        note='GRADIENT ONLY at saved converged orbitals; SCF is not rerun/timed and no whole-workflow speedup is inferred. Full grid/auxiliary response. CPU energy uses unchanged saved converged SCF energy; gradient recomputed here. Reference energy/forces are original GPU results. Memory tracks new requested CuPy storage after orbitals restored, excluding preexisting/native storage. J/K scope measures new allocations only. Paired timing and fresh-process allocation observations are separate.')
    return report,original['result'],identity['path']


def checked(case,reference,array):
    actual=np.asarray(array.get() if hasattr(array,'get') else array)
    error=float(np.max(np.abs(actual-np.asarray(reference['gradient_hartree_per_bohr']))))
    assert error<=1e-9,error
    result=dict(error_to_original_gpu=error,gradient_hartree_per_bohr=actual.tolist())
    cpu_path=RESULTS/(case+'_cpu.json')
    if cpu_path.exists():
        oracle=json.loads(cpu_path.read_text())
        if oracle['status']=='passed':
            comparison=compare_cases(oracle['result'],dict(reference,gradient_hartree_per_bohr=actual.tolist()))
            assert comparison['passed'],comparison
            result.update(cpu_comparison=comparison,cpu_reference_sha256=_file_identity(cpu_path)['sha256'])
    return result


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case',['tamoxifen_tzvp_b3lyp','taxol_svp_b3lyp'])
@pytest.mark.parametrize('variant',['baseline','packed'])
def test_gradient_memory(case,variant):
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf.df.grad import rhf
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    from memory_profile import peak_memory
    output=RESULTS/(case+'_gradient_'+variant+'_memory.json')
    if output.exists(): raise FileExistsError('Archive existing report explicitly')
    report,reference,orbital_path=inputs(case)
    report.update(variant=variant,mode='memory')
    save(output,report)
    try:
        assert cp.get_default_memory_pool().used_bytes()==0
        with lib.with_omp_threads(4):
            mf=restore(case,orbital_path)
            scopes=[]
            with ExitStack() as stack:
                if variant=='packed': stack.enter_context(packed_gradient())
                original=rhf._jk_energy_per_atom
                def jk(*args,**kwargs):
                    with peak_memory() as memory: value=original(*args,**kwargs)
                    scopes.append(memory)
                    return value
                stack.enter_context(patch.object(rhf,'_jk_energy_per_atom',jk))
                grad=mf.nuc_grad_method()
                grad.grid_response=grad.auxbasis_response=True
                with peak_memory() as memory:
                    actual=grad.kernel()
        report.update(memory=memory,jk_memory=scopes,comparison=checked(case,reference,actual),status='passed')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save(output,report)
        raise
    save(output,report)
    print(case,variant,'gradient memory',memory['peak_tracked_live_bytes'],
          max(x['peak_tracked_live_bytes'] for x in scopes),flush=True)


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case',['tamoxifen_tzvp_b3lyp','taxol_svp_b3lyp'])
def test_gradient_pairs(case):
    import cupy as cp
    from pyscf import lib
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    output=RESULTS/(case+'_gradient_pairs.json')
    if output.exists(): raise FileExistsError('Archive existing report explicitly')
    report,reference,orbital_path=inputs(case)
    report.update(mode='paired_gradient_timing',warmups=[],pairs=[])
    save(output,report)
    try:
        with lib.with_omp_threads(4):
            mf=restore(case,orbital_path)
            def run(version):
                with packed_gradient() if version=='packed' else nullcontext():
                    grad=mf.nuc_grad_method()
                    grad.grid_response=grad.auxbasis_response=True
                    cp.cuda.Device().synchronize(); start=time.perf_counter()
                    actual=grad.kernel()
                    cp.cuda.Device().synchronize(); elapsed=time.perf_counter()-start
                return dict(seconds=elapsed,**checked(case,reference,actual))
            for version in ['baseline','packed']:
                report['warmups'].append(dict(variant=version,**run(version)))
                save(output,report)
            for repeat in range(3):
                versions=['baseline','packed'] if repeat%2==0 else ['packed','baseline']
                report['pairs'].append({v:run(v) for v in versions})
                save(output,report)
                print(case,'gradient pair',repeat+1,'passed',flush=True)
        report['medians']={v:statistics.median(p[v]['seconds'] for p in report['pairs']) for v in ['baseline','packed']}
        report['speedup']=report['medians']['baseline']/report['medians']['packed']
        report['paired_ratios']=[p['baseline']['seconds']/p['packed']['seconds'] for p in report['pairs']]
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save(output,report)
        raise
    save(output,report)
    print(case,'gradient speedup',report['speedup'],flush=True)


@pytest.mark.special
@pytest.mark.benchmark
def test_large_gradient_profile():
    from pyscf import lib
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    from test_stage_profile import stage_profile
    from test_fine_stage_profile import fine_profile
    case='taxol_svp_b3lyp'
    output=RESULTS/(case+'_gradient_packed_profile.json')
    if output.exists(): raise FileExistsError('Archive existing report explicitly')
    report,reference,orbital_path=inputs(case)
    report['mode']='inclusive_gradient_profile'
    report['helper_hashes'].update({p:_file_identity(BASE/p)['sha256'] for p in
                                  ['test_stage_profile.py','test_fine_stage_profile.py']})
    save(output,report)
    try:
        with lib.with_omp_threads(4):
            mf=restore(case,orbital_path)
            with packed_gradient(),stage_profile() as coarse,fine_profile() as fine:
                grad=mf.nuc_grad_method()
                grad.grid_response=grad.auxbasis_response=True
                actual=grad.kernel()
        assert fine['bindings_restored']
        report.update(coarse=coarse,fine=fine,comparison=checked(case,reference,actual),status='passed')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save(output,report)
        raise
    save(output,report)
    print(case,'packed gradient profile passed',coarse,flush=True)
