"""Promoted layout algorithm versus saved previous GPU, with final profiling."""
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import cupy as cp
import pytest

from molecular import SETTINGS, _metadata, _file_identity, run_case, compare_cases
from test_exchange_layout_workflow import baseline
from test_stage_profile import stage_profile
from test_fine_stage_profile import fine_profile


@pytest.mark.special
@pytest.mark.benchmark
def test_layout_production_acceptance():
    from gpu4pyscf.df import df_jk
    from cupy_backends.cuda.libs import cublas
    assert hasattr(df_jk,'_exchange_transform')
    output=Path(__file__).with_name('exchange_layout_acceptance_results.json')
    refs_path=Path(__file__).with_name('stage_profile_results.json')
    refs=json.loads(refs_path.read_text())
    assert refs['status']=='passed'
    cpu=next(r['cpu_reference'] for r in refs['cases'] if r['case']=='tamoxifen_svp' and r['method']=='B3LYP')
    report=dict(_metadata('gpu'),status='running',case='tamoxifen_svp',method='B3LYP',pairs=[],warmups=[],
        baseline_sha256=_file_identity(Path(__file__).with_name('exchange_layout_baseline.json'))['sha256'],
        reference_sha256=_file_identity(refs_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        algorithm_source_hashes={p:_file_identity(p)['sha256'] for p in ['gpu4pyscf/df/df_jk.py','gpu4pyscf/dft/numint.py']},
        note='Current production versus saved previous optimized GPU get_jk, same native build. Shared GB10. Both warmed; three alternating pairs, then one instrumented production run. Grid level1 unpruned, unchanged precision/thresholds. Inclusive synchronized profiling perturbs scheduling; nested timings must not be summed.')
    for key in ['pyscf','numpy','cpu_libxc']: assert cpu[key]==report[key]
    ref=next(r for r in refs['cases'] if r['case']=='tamoxifen_svp' and r['method']=='B3LYP')
    assert ref['geometry_sha256']==_file_identity(cpu['settings']['atom'])['sha256']
    assert ref['instrumented']['native_libraries']==report['native_libraries']
    original=df_jk.get_jk
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    def checked(result):
        comparison=compare_cases(cpu,result)
        assert comparison['passed'],comparison
        return dict(result=result,comparison=comparison)
    def run(version):
        if version=='previous':
            with baseline(): result=run_case('B3LYP','gpu')
        else: result=run_case('B3LYP','gpu')
        assert df_jk.get_jk is original
        return checked(result)
    save()
    with patch.dict(SETTINGS,cpu['settings']):
        for version in ['previous','production']:
            print(version,'warmup',flush=True)
            report['warmups'].append(dict(version=version,**run(version))); save()
        for repeat in range(3):
            pair={}
            versions=['previous','production']
            if repeat%2: versions.reverse()
            for version in versions: pair[version]=run(version)
            assert pair['previous']['result']['iterations']==pair['production']['result']['iterations']
            report['pairs'].append(pair); save()
            print('production pair',repeat+1,'passed',flush=True)
        report['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+p[v]['result']['gradient_seconds'] for p in report['pairs']) for v in ['previous','production']}
        report['paired_ratios']=[(p['previous']['result']['scf_seconds']+p['previous']['result']['gradient_seconds'])/(p['production']['result']['scf_seconds']+p['production']['result']['gradient_seconds']) for p in report['pairs']]
        report['speedup']=report['combined_medians']['previous']/report['combined_medians']['production']
        print('production speedup',report['speedup'],'; final instrumented run',flush=True)
        exchange=dict(calls=0,seconds=0.)
        gram=dict(calls=0,seconds=0.)
        strided_calls=[]
        original_batched=cublas.dgemmStridedBatched
        def observed_batched(*args):
            strided_calls.append(True)
            return original_batched(*args)
        def timed(function, row):
            def wrapped(*args,**kwargs):
                cp.cuda.Device().synchronize(); start=time.perf_counter()
                try: return function(*args,**kwargs)
                finally:
                    cp.cuda.Device().synchronize()
                    row['calls']+=1; row['seconds']+=time.perf_counter()-start
            return wrapped
        with stage_profile() as coarse, fine_profile() as fine, \
             patch.object(df_jk,'_exchange_transform',timed(df_jk._exchange_transform,exchange)), \
             patch.object(df_jk,'_symmetric_gram',timed(df_jk._symmetric_gram,gram)), \
             patch.object(cublas,'dgemmStridedBatched',observed_batched):
            result=run_case('B3LYP','gpu')
        report.update(instrumented=checked(result),coarse=coarse,fine=fine,
                      exchange_transform=exchange,symmetric_gram=gram,
                      strided_gemm_calls=len(strided_calls))
        assert fine['bindings_restored']
        assert exchange['calls']>0 and gram['calls']>0
        assert strided_calls, 'No strided GEMM executed: performance qualification requires the fast path'
        assert all(p['production']['result']['iterations']==result['iterations'] for p in report['pairs'])
    report['status']='passed'; save()
