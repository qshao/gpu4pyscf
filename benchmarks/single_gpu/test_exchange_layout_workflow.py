"""Paired energy and force qualification against the pre-layout GPU baseline."""
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
from types import FunctionType
from unittest.mock import patch

import pytest

import exchange_layout
from molecular import SETTINGS, _metadata, _file_identity, run_case, compare_cases


@contextmanager
def baseline():
    from gpu4pyscf.df import df_jk
    source = json.loads(Path(__file__).with_name('exchange_layout_baseline.json').read_text())['function']
    namespace = dict(df_jk.__dict__)
    exec(compile(source,'pre_layout_df_jk.py','exec'),namespace)
    function = namespace['get_jk']
    restored = FunctionType(function.__code__,df_jk.__dict__,'get_jk',function.__defaults__)
    with patch.object(df_jk,'get_jk',restored): yield


@pytest.mark.special
@pytest.mark.benchmark
def test_layout_workflows():
    from gpu4pyscf.df import df_jk
    output = Path(__file__).with_name('exchange_layout_workflow_results.json')
    refs_path = Path(__file__).with_name('stage_profile_results.json')
    refs = json.loads(refs_path.read_text())
    assert refs['status']=='passed'
    report = dict(_metadata('gpu'),status='running',cases=[],
        baseline_sha256=_file_identity(Path(__file__).with_name('exchange_layout_baseline.json'))['sha256'],
        reference_sha256=_file_identity(refs_path)['sha256'],
        candidate_sha256=_file_identity(exchange_layout.__file__)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        note='Pre-layout optimized GPU baseline versus reversible layout substitution. Both retain adaptive density and SYRK. Same native build; shared GB10. Warm both, three alternating pairs per case. CPU gates on every run; matching iterations. SCF plus gradient includes DF setup, excludes construction/guess/import/metadata. Grid level1 unpruned.')
    originals = df_jk.get_jk, df_jk.contract
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for case in ['vitamin_svp','tamoxifen_svp']:
        ref = next(r for r in refs['cases'] if r['case']==case and r['method']=='B3LYP')
        cpu = ref['cpu_reference']
        assert _file_identity(cpu['settings']['atom'])['sha256']==ref['geometry_sha256']
        for key in ['pyscf','numpy','cpu_libxc']: assert cpu[key]==report[key]
        assert ref['instrumented']['native_libraries']==report['native_libraries']
        row = dict(case=case,method='B3LYP',warmups=[],pairs=[],status='running')
        report['cases'].append(row)
        def run(version):
            with baseline():
                if version=='layout':
                    with exchange_layout.version(): result=run_case('B3LYP','gpu')
                else: result=run_case('B3LYP','gpu')
            assert (df_jk.get_jk,df_jk.contract)==originals
            comparison=compare_cases(cpu,result)
            assert comparison['passed'],comparison
            return dict(result=result,comparison=comparison)
        with patch.dict(SETTINGS,cpu['settings']):
            for version in ['current','layout']:
                print(case,version,'warmup',flush=True)
                row['warmups'].append(dict(version=version,**run(version))); save()
            for repeat in range(3):
                pair={}
                versions=['current','layout']
                if repeat%2: versions.reverse()
                for version in versions: pair[version]=run(version)
                assert pair['current']['result']['iterations']==pair['layout']['result']['iterations']
                row['pairs'].append(pair); save()
                print(case,'pair',repeat+1,'passed',flush=True)
        row['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+p[v]['result']['gradient_seconds'] for p in row['pairs']) for v in ['current','layout']}
        row['paired_ratios']=[(p['current']['result']['scf_seconds']+p['current']['result']['gradient_seconds'])/(p['layout']['result']['scf_seconds']+p['layout']['result']['gradient_seconds']) for p in row['pairs']]
        row['speedup']=row['combined_medians']['current']/row['combined_medians']['layout']
        row['status']='passed'; save()
        print(case,'speedup',row['speedup'],flush=True)
    report['status']='passed'; save()
