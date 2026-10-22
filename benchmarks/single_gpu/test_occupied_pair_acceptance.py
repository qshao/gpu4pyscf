"""Repeated paired whole workflows for the benchmark-only packed gradient."""
from contextlib import nullcontext
import json
from pathlib import Path
import statistics

import pytest

from comprehensive import CASES,RESULTS,SOURCES,ROOT,configured,save
from molecular import _metadata,_file_identity,run_case,compare_cases


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('case',['tamoxifen_svp_b3lyp','tamoxifen_tzvp_b3lyp','taxol_svp_b3lyp'])
def test_paired(case):
    from occupied_pair import packed_gradient
    from gpu4pyscf.df.grad import rhf
    output=RESULTS/(case+'_occupied_pairs.json')
    if output.exists(): raise FileExistsError('Archive report explicitly: '+str(output))
    cpu_path=RESULTS/(case+'_cpu.json')
    oracle=json.loads(cpu_path.read_text())
    assert oracle['status']=='passed'
    cpu=oracle['result']
    report=dict(_metadata('gpu'),case=case,status='running',warmups=[],pairs=[],
        reference_sha256=_file_identity(cpu_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        candidate_sha256=_file_identity(Path(__file__).with_name('occupied_pair.py'))['sha256'],
        configuration_sha256=_file_identity(Path(__file__).with_name('comprehensive.py'))['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        geometry_sha256=_file_identity(cpu['settings']['atom'])['sha256'],
        note='Three alternating pairs after both warmups on shared GB10. Full FP64, grid and auxiliary response. Same source/native build; only gradient occupied-pair storage changes. No concurrent CPU reference jobs. No instrumentation inside timing other than existing iteration synchronization.')
    assert report['geometry_sha256']==oracle['geometry_sha256']
    for key in ['pyscf','numpy','cpu_libxc']: assert report[key]==cpu[key]
    original=rhf._jk_energy_per_atom
    def run(version):
        with packed_gradient() if version=='packed' else nullcontext():
            result=run_case(CASES[case][2],'gpu')
        assert rhf._jk_energy_per_atom is original
        comparison=compare_cases(cpu,result)
        assert comparison['passed'],comparison
        return dict(result=result,comparison=comparison)
    save(output,report)
    try:
        with configured(case) as settings:
            assert settings==cpu['settings']
            for version in ['baseline','packed']:
                report['warmups'].append(dict(variant=version,**run(version)))
                save(output,report)
            for repeat in range(3):
                versions=['baseline','packed'] if repeat%2==0 else ['packed','baseline']
                pair={version:run(version) for version in versions}
                assert pair['baseline']['result']['iterations']==pair['packed']['result']['iterations']
                report['pairs'].append(pair); save(output,report)
                print(case,'pair',repeat+1,'passed',flush=True)
        report['medians']={v:{phase:statistics.median(p[v]['result'][phase] for p in report['pairs'])
                              for phase in ['scf_seconds','gradient_seconds']} for v in ['baseline','packed']}
        report['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+p[v]['result']['gradient_seconds'] for p in report['pairs']) for v in ['baseline','packed']}
        report['speedup']=report['combined_medians']['baseline']/report['combined_medians']['packed']
        report['paired_ratios']=[(p['baseline']['result']['scf_seconds']+p['baseline']['result']['gradient_seconds'])/(p['packed']['result']['scf_seconds']+p['packed']['result']['gradient_seconds']) for p in report['pairs']]
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        save(output,report)
        raise
    save(output,report)
    print(case,'speedup',report['speedup'],'paired range',min(report['paired_ratios']),max(report['paired_ratios']),flush=True)
