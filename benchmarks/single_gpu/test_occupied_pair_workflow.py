"""Fresh-process gradient symmetry qualification; timing and memory are separate."""
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comprehensive import CASES,RESULTS,SOURCES,ROOT,configured,save
from molecular import _metadata,_file_identity,run_case,compare_cases


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case',[c for c in CASES if 'b3lyp' in c])
@pytest.mark.parametrize('variant',['baseline','packed'])
@pytest.mark.parametrize('mode',['control','memory'])
def test_workflow(case,variant,mode):
    from gpu4pyscf.df.grad import rhf
    from occupied_pair import packed_gradient
    from memory_profile import peak_memory
    import cupy as cp
    output=RESULTS/(case+'_occupied_'+variant+'_'+mode+'.json')
    if output.exists():
        raise FileExistsError('Archive previous report explicitly: '+str(output))
    cpu_path=RESULTS/(case+'_cpu.json')
    oracle=json.loads(cpu_path.read_text())
    assert oracle['status']=='passed'
    cpu=oracle['result']
    report=dict(_metadata('gpu'),status='running',case=case,variant=variant,mode=mode,
                reference_sha256=_file_identity(cpu_path)['sha256'],
                benchmark_sha256=_file_identity(__file__)['sha256'],
                candidate_sha256=_file_identity(Path(__file__).with_name('occupied_pair.py'))['sha256'],
                configuration_sha256=_file_identity(Path(__file__).with_name('comprehensive.py'))['sha256'],
                memory_profiler_sha256=_file_identity(Path(__file__).with_name('memory_profile.py'))['sha256'],
                algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
                geometry_sha256=_file_identity(cpu['settings']['atom'])['sha256'],
                note='Single control is not repeated speed qualification. Memory is requested CuPy storage, not total RAM; J/K scope counts new allocations only and excludes preexisting storage. Full grid/aux response; default cache disabled.')
    assert report['geometry_sha256']==oracle['geometry_sha256']
    for key in ['pyscf','numpy','cpu_libxc']: assert report[key]==cpu[key]
    save(output,report)
    try:
        measured=[]
        with configured(case) as settings:
            assert settings==cpu['settings']
            candidate=packed_gradient() if variant=='packed' else nullcontext()
            with candidate:
                original=rhf._jk_energy_per_atom
                def observed(*args,**kwargs):
                    if mode=='memory':
                        with peak_memory() as memory:
                            result=original(*args,**kwargs)
                        measured.append(memory)
                        return result
                    return original(*args,**kwargs)
                with patch.object(rhf,'_jk_energy_per_atom',observed):
                    if mode=='memory':
                        assert cp.get_default_memory_pool().used_bytes()==0
                        with peak_memory() as full:
                            result=run_case(CASES[case][2],'gpu')
                        report.update(memory=full,jk_memory=measured)
                    else:
                        result=run_case(CASES[case][2],'gpu')
        comparison=compare_cases(cpu,result)
        report.update(result=result,comparison=comparison)
        assert comparison['passed'],comparison
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save(output,report)
        raise
    save(output,report)
    print(case,variant,mode,result['iterations'],result['scf_seconds'],result['gradient_seconds'],flush=True)
