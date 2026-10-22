"""Actual production versus saved layout-stage GPU, with final profiling."""
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import cupy as cp
import pytest

from speed_memory_candidates import baseline
from molecular import SETTINGS,_metadata,_file_identity,run_case,compare_cases
from test_stage_profile import stage_profile
from test_fine_stage_profile import fine_profile


@pytest.mark.special
@pytest.mark.benchmark
def test_production_acceptance():
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.df.grad import rhf
    from cupy_backends.cuda.libs import cublas
    output=Path(__file__).with_name('speed_memory_acceptance_results.json')
    ref_path=Path(__file__).with_name('stage_profile_results.json')
    refs=json.loads(ref_path.read_text())
    assert refs['status']=='passed'
    report=dict(_metadata('gpu'),status='running',cases=[],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        baseline_sha256=_file_identity(Path(__file__).with_name('speed_memory_baseline.json'))['sha256'],
        reference_sha256=_file_identity(ref_path)['sha256'],
        algorithm_source_hashes={p:_file_identity(p)['sha256'] for p in
            ['gpu4pyscf/df/df_jk.py','gpu4pyscf/df/grad/rhf.py','gpu4pyscf/df/grad/uhf.py']},
        note='Saved approximately42-second layout-stage GPU versus actual production. Same native build, shared GB10, no memory instrumentation. Warm both; three alternating pairs large/one small smoke pair. One final large instrumented run. Inclusive synchronized timings perturb scheduling; do not sum nested scopes. No cumulative original-GPU ratio inferred. Gridlevel1 regression setting.')
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    for case in ['vitamin_svp','tamoxifen_svp']:
        ref=next(r for r in refs['cases'] if r['case']==case and r['method']=='B3LYP')
        cpu=ref['cpu_reference']
        assert _file_identity(cpu['settings']['atom'])['sha256']==ref['geometry_sha256']
        for key in ['pyscf','numpy','cpu_libxc']: assert cpu[key]==report[key]
        assert ref['instrumented']['native_libraries']==report['native_libraries']
        row=dict(case=case,method='B3LYP',warmups=[],pairs=[],status='running')
        report['cases'].append(row); save()
        original=df_jk.get_jk,rhf._jk_energy_per_atom
        def checked(result):
            comparison=compare_cases(cpu,result)
            assert comparison['passed'],comparison
            return dict(result=result,comparison=comparison)
        def run(version):
            if version=='previous':
                with baseline(): result=run_case('B3LYP','gpu')
            else: result=run_case('B3LYP','gpu')
            assert (df_jk.get_jk,rhf._jk_energy_per_atom)==original
            return checked(result)
        with patch.dict(SETTINGS,cpu['settings']):
            for version in ['previous','production']:
                print(case,version,'warmup',flush=True)
                row['warmups'].append(dict(version=version,**run(version))); save()
            for repeat in range(3 if case=='tamoxifen_svp' else 1):
                pair={}
                versions=['previous','production']
                if repeat%2: versions.reverse()
                for version in versions: pair[version]=run(version)
                assert pair['previous']['result']['iterations']==pair['production']['result']['iterations']
                row['pairs'].append(pair); save()
                print(case,'production pair',repeat+1,'passed',flush=True)
            row['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+p[v]['result']['gradient_seconds'] for p in row['pairs']) for v in ['previous','production']}
            row['paired_ratios']=[(p['previous']['result']['scf_seconds']+p['previous']['result']['gradient_seconds'])/(p['production']['result']['scf_seconds']+p['production']['result']['gradient_seconds']) for p in row['pairs']]
            row['speedup']=row['combined_medians']['previous']/row['combined_medians']['production']
            row['status']='passed'; save()
            print(case,'production speedup',row['speedup'],flush=True)
            if case=='tamoxifen_svp':
                timers={k:dict(calls=0,seconds=0.) for k in ['coulomb','gradient','exchange','gram']}
                def timed(function,key):
                    def wrapped(*args,**kwargs):
                        cp.cuda.Device().synchronize(); start=time.perf_counter()
                        try: return function(*args,**kwargs)
                        finally:
                            cp.cuda.Device().synchronize()
                            timers[key]['calls']+=1; timers[key]['seconds']+=time.perf_counter()-start
                    return wrapped
                gemv_calls=[]
                original_gemv=cublas.dgemv
                def observed_gemv(*args):
                    gemv_calls.append(True); return original_gemv(*args)
                with stage_profile() as coarse,fine_profile() as fine, \
                     patch.object(df_jk,'_coulomb_contract',timed(df_jk._coulomb_contract,'coulomb')), \
                     patch.object(df_jk,'_exchange_transform',timed(df_jk._exchange_transform,'exchange')), \
                     patch.object(df_jk,'_symmetric_gram',timed(df_jk._symmetric_gram,'gram')), \
                     patch.object(rhf,'_gradient_contract',timed(rhf._gradient_contract,'gradient')), \
                     patch.object(cublas,'dgemv',observed_gemv):
                    measured=run_case('B3LYP','gpu')
                assert fine['bindings_restored'] and gemv_calls
                assert all(p['production']['result']['iterations']==measured['iterations']
                           for p in row['pairs'])
                row.update(instrumented=checked(measured),coarse=coarse,fine=fine,
                           helpers=timers,actual_gemv_calls=len(gemv_calls)); save()
    report['status']='passed'; save()
