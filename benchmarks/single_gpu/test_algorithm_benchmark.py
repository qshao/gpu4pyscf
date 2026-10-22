"""Algorithm components and paired whole workflows; no persistent substitutions."""
import json
from pathlib import Path
import statistics
import time
from contextlib import ExitStack
from types import FunctionType
from unittest.mock import patch

import cupy as cp
import pytest

from algorithm_candidates import adaptive_rho, algorithm_version, symmetric_gram
from molecular import SETTINGS, _file_identity, _metadata, compare_cases, run_case


@pytest.fixture(autouse=True)
def pre_algorithm_baseline():
    """Keep isolated experiments reproducible after production promotion."""
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft import numint
    snapshot=json.loads(Path(__file__).with_name('algorithm_baseline.json').read_text())
    with ExitStack() as stack:
        for label,module in [('df_jk',df_jk),('numint',numint)]:
            for name,source in snapshot[label]['functions'].items():
                namespace=dict(module.__dict__)
                exec(compile(source,'pre_algorithm_'+label+'.py','exec'),namespace)
                function=namespace[name]
                # Live globals let the exchange candidate patch contract inside
                # this saved get_jk, just as it did before promotion.
                restored=FunctionType(function.__code__,module.__dict__,name,function.__defaults__)
                stack.enter_context(patch.object(module,name,restored))
        yield


def test_isolated_baseline_is_original():
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft import numint
    snapshot=json.loads(Path(__file__).with_name('algorithm_baseline.json').read_text())
    for label,module in [('df_jk',df_jk),('numint',numint)]:
        for name,source in snapshot[label]['functions'].items():
            namespace=dict(module.__dict__)
            exec(compile(source,'baseline_check.py','exec'),namespace)
            assert getattr(module,name).__code__.co_code==namespace[name].__code__.co_code
            assert getattr(module,name).__globals__ is module.__dict__


@pytest.mark.special
@pytest.mark.benchmark
def test_algorithm_components():
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft import numint
    output=Path(__file__).with_name('algorithm_component_results.json')
    report=dict(_metadata('gpu'),status='running',exchange=[],density=[],
        baseline_sha256=_file_identity(Path(__file__).with_name('algorithm_baseline.json'))['sha256'],
        candidate_sha256=_file_identity(Path(__file__).with_name('algorithm_candidates.py'))['sha256'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    rng=cp.random.RandomState(18)
    def measure(functions):
        samples={k:[] for k in functions}
        for f in functions.values(): f()
        for repeat in range(7):
            keys=list(functions)
            if repeat%2: keys.reverse()
            for k in keys:
                cp.cuda.Device().synchronize(); start=time.perf_counter()
                functions[k]()
                cp.cuda.Device().synchronize()
                samples[k].append(time.perf_counter()-start)
        return {k:dict(samples_seconds=v,median_seconds=statistics.median(v)) for k,v in samples.items()}
    for nao,nocc,block in [(208,44,256),(256,44,256),(384,100,256),(512,100,256),
                           (537,100,256),(537,169,66)]:
        x=rng.normal(size=(1,nao,nocc,block))
        out=cp.zeros((1,nao,nao))
        def original():
            return df_jk.contract('nikL,njkL->nij',x,x,beta=1.,out=out)
        def candidate(): return symmetric_gram(x.reshape(1,nao,-1),out)
        out.fill(0); reference=original().copy(); out.fill(0)
        cp.testing.assert_allclose(candidate(),reference,rtol=2e-12,atol=1e-9)
        times=measure(dict(original=original,candidate=candidate))
        report['exchange'].append(dict(nao=nao,nocc=nocc,block=block,timings=times))
        print('exchange',nao,nocc,block,{k:round(v['median_seconds']*1000,3) for k,v in times.items()},flush=True)
    for nao,nocc in [(128,100),(256,100),(384,100),(208,44),(537,100)]:
        ao=rng.normal(size=(4,nao,4096)); cpos=rng.normal(size=(nao,nocc))
        buf=cp.empty(max(nao,2*nocc)*4096)
        original=lambda:numint._eval_rho2(ao,cpos,'GGA',buf=buf)
        candidate=lambda:adaptive_rho(numint._eval_rho2)(ao,cpos,'GGA',buf=buf)
        cp.testing.assert_allclose(candidate(),original(),rtol=2e-11,atol=2e-9)
        times=measure(dict(original=original,candidate=candidate))
        report['density'].append(dict(nao=nao,nocc=nocc,timings=times))
        print('density',nao,nocc,{k:round(v['median_seconds']*1000,3) for k,v in times.items()},flush=True)
    report['status']='passed'
    output.write_text(json.dumps(report,indent=2)+'\n')


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('algorithm',['density','exchange'])
def test_algorithm_workflows(algorithm):
    output=Path(__file__).with_name('algorithm_'+algorithm+'_results.json')
    reference_path=Path(__file__).with_name('stage_profile_results.json')
    references=json.loads(reference_path.read_text())
    assert references['status']=='passed'
    report=dict(_metadata('gpu'),status='running',algorithm=algorithm,cases=[],
                baseline_sha256=_file_identity(Path(__file__).with_name('algorithm_baseline.json'))['sha256'],
                reference_sha256=_file_identity(reference_path)['sha256'],
                candidate_sha256=_file_identity(Path(__file__).with_name('algorithm_candidates.py'))['sha256'],
                note='Same native build/current code; original versus reversible algorithm substitution. Shared GB10. Three alternating pairs after warmup. Per-iteration synchronization retained; no production precision/grid changes.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    method='PBE' if algorithm=='density' else 'B3LYP'
    for case in ('vitamin_svp','tamoxifen_svp'):
        ref=next(c for c in references['cases'] if c['case']==case and c['method']==method)
        cpu=ref['cpu_reference']
        assert _file_identity(cpu['settings']['atom'])['sha256']==ref['geometry_sha256']
        for k in ('pyscf','numpy','cpu_libxc'): assert cpu[k]==report[k]
        assert ref['instrumented']['native_libraries']==report['native_libraries']
        row=dict(case=case,method=method,status='running',pairs=[],warmups=[])
        report['cases'].append(row)
        with patch.dict(SETTINGS,cpu['settings']):
            for version in ('original',algorithm):
                print(algorithm,case,version,'warmup',flush=True)
                with algorithm_version(version): value=run_case(method,'gpu')
                comparison=compare_cases(cpu,value)
                row['warmups'].append(dict(version=version,result=value,comparison=comparison))
                assert comparison['passed'],comparison
            for repeat in range(3):
                pair={}
                versions=['original',algorithm]
                if repeat%2: versions.reverse()
                for version in versions:
                    with algorithm_version(version): value=run_case(method,'gpu')
                    comparison=compare_cases(cpu,value)
                    pair[version]=dict(result=value,comparison=comparison)
                    output.write_text(json.dumps(report,indent=2)+'\n')
                    assert comparison['passed'],comparison
                assert pair['original']['result']['iterations']==pair[algorithm]['result']['iterations']
                row['pairs'].append(pair)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print(algorithm,case,'pair',repeat+1,'passed',flush=True)
            row['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+
                p[v]['result']['gradient_seconds'] for p in row['pairs']) for v in ('original',algorithm)}
            row['speedup']=row['combined_medians']['original']/row['combined_medians'][algorithm]
            row['status']='passed'
            print(algorithm,case,'speedup',row['speedup'],flush=True)
            output.write_text(json.dumps(report,indent=2)+'\n')
    report['status']='passed'
    output.write_text(json.dumps(report,indent=2)+'\n')
