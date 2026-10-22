"""Final production algorithms versus original GPU Python paths in one build."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import pytest

from df_profile import contraction_version, BASELINE
from molecular import SETTINGS, _metadata, _file_identity, compare_cases, run_case


@contextmanager
def original_gpu():
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft import numint
    snapshot=json.loads(Path(__file__).with_name('algorithm_baseline.json').read_text())
    with contraction_version('baseline'), ExitStack() as stack:
        for label, module in [('df_jk',df_jk),('numint',numint)]:
            for name,source in snapshot[label]['functions'].items():
                namespace=dict(module.__dict__)
                exec(compile(source,'pre_algorithm_'+label+'.py','exec'),namespace)
                stack.enter_context(patch.object(module,name,namespace[name]))
        yield


@pytest.mark.special
@pytest.mark.benchmark
def test_production_acceptance():
    output=Path(__file__).with_name('algorithm_acceptance_results.json')
    references_path=Path(__file__).with_name('stage_profile_results.json')
    references=json.loads(references_path.read_text())
    assert references['status']=='passed'
    report=dict(_metadata('gpu'),status='running',cases=[],baseline_revision=BASELINE,
        baseline_sha256=_file_identity(Path(__file__).with_name('algorithm_baseline.json'))['sha256'],
        references_sha256=_file_identity(references_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        note='Original GPU contraction + saved original algorithm functions versus production defaults, same current native build. Shared GB10. Warm both; one pair per small case, three alternating pairs per large case. CPU gates on every run; per-iteration synchronization retained. Includes SCF/DF setup and gradient, excludes construction/initial guess/imports/metadata. Not independent package-release comparison.')
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft import numint
    report['algorithm_source_hashes']={f:_file_identity(f)['sha256'] for f in
        ['gpu4pyscf/dft/numint.py','gpu4pyscf/df/df_jk.py']}
    originals=(df_jk.get_jk,numint._eval_rho2)
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for case in ['vitamin_svp','tamoxifen_svp']:
        for method in ['PBE','B3LYP']:
            ref=next(c for c in references['cases'] if c['case']==case and c['method']==method)
            cpu=ref['cpu_reference']
            assert ref['geometry_sha256']==_file_identity(cpu['settings']['atom'])['sha256']
            for key in ['pyscf','numpy','cpu_libxc']: assert cpu[key]==report[key]
            assert ref['instrumented']['native_libraries']==report['native_libraries']
            row=dict(case=case,method=method,status='running',warmups=[],pairs=[])
            report['cases'].append(row); save()
            def run(version):
                if version=='original_gpu':
                    with original_gpu(): result=run_case(method,'gpu')
                else: result=run_case(method,'gpu')
                assert (df_jk.get_jk,numint._eval_rho2)==originals
                comparison=compare_cases(cpu,result)
                assert comparison['passed'],comparison
                return dict(result=result,comparison=comparison)
            with patch.dict(SETTINGS,cpu['settings']):
                for version in ['original_gpu','production']:
                    print(case,method,version,'warmup',flush=True)
                    row['warmups'].append(dict(version=version,**run(version))); save()
                for repeat in range(3 if case=='tamoxifen_svp' else 1):
                    pair={}
                    versions=['original_gpu','production']
                    if repeat%2: versions.reverse()
                    for version in versions: pair[version]=run(version)
                    assert pair['original_gpu']['result']['iterations']==pair['production']['result']['iterations']
                    row['pairs'].append(pair); save()
                    print(case,method,'pair',repeat+1,'passed',flush=True)
            row['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+
                p[v]['result']['gradient_seconds'] for p in row['pairs']) for v in ['original_gpu','production']}
            row['paired_ratios']=[(p['original_gpu']['result']['scf_seconds']+p['original_gpu']['result']['gradient_seconds'])/
                                 (p['production']['result']['scf_seconds']+p['production']['result']['gradient_seconds']) for p in row['pairs']]
            row['speedup']=row['combined_medians']['original_gpu']/row['combined_medians']['production']
            row['status']='passed'; save()
            print(case,method,'speedup',row['speedup'],flush=True)
    report['status']='passed'; save()
