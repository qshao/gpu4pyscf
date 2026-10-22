"""Warm, uninstrumented alternating speed/memory algorithm comparison."""
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import pytest

import speed_memory_candidates as candidate
from molecular import SETTINGS,_metadata,_file_identity,run_case,compare_cases


@pytest.mark.special
@pytest.mark.benchmark
def test_combined_workflow():
    output=Path(__file__).with_name('speed_memory_workflow_results.json')
    ref_path=Path(__file__).with_name('stage_profile_results.json')
    refs=json.loads(ref_path.read_text())
    assert refs['status']=='passed'
    report=dict(_metadata('gpu'),status='running',cases=[],
        candidate_sha256=_file_identity(candidate.__file__)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        baseline_sha256=_file_identity(Path(__file__).with_name('speed_memory_baseline.json'))['sha256'],
        reference_sha256=_file_identity(ref_path)['sha256'],
        note='Saved layout-stage optimized GPU versus combined direct contractions and compact gradients. Same native build; shared GB10. No memory instrumentation. Both warmed, three alternating pairs. All scientific settings unchanged; CPU gates on every result. No cumulative original-GPU speedup inferred.')
    def save(): output.write_text(json.dumps(report,indent=2)+'\n')
    for case in ['vitamin_svp','tamoxifen_svp']:
        ref=next(r for r in refs['cases'] if r['case']==case and r['method']=='B3LYP')
        cpu=ref['cpu_reference']
        assert _file_identity(cpu['settings']['atom'])['sha256']==ref['geometry_sha256']
        for key in ['pyscf','numpy','cpu_libxc']: assert cpu[key]==report[key]
        assert ref['instrumented']['native_libraries']==report['native_libraries']
        row=dict(case=case,method='B3LYP',warmups=[],pairs=[],status='running')
        report['cases'].append(row); save()
        def run(version):
            with candidate.version(version): result=run_case('B3LYP','gpu')
            comparison=compare_cases(cpu,result)
            assert comparison['passed'],comparison
            return dict(result=result,comparison=comparison)
        with patch.dict(SETTINGS,cpu['settings']):
            for version in ['current','combined']:
                print(case,version,'warmup',flush=True)
                row['warmups'].append(dict(version=version,**run(version))); save()
            for repeat in range(3):
                pair={}
                versions=['current','combined']
                if repeat%2: versions.reverse()
                for version in versions: pair[version]=run(version)
                assert pair['current']['result']['iterations']==pair['combined']['result']['iterations']
                row['pairs'].append(pair); save()
                print(case,'pair',repeat+1,'passed',flush=True)
        row['combined_medians']={v:statistics.median(p[v]['result']['scf_seconds']+p[v]['result']['gradient_seconds'] for p in row['pairs']) for v in ['current','combined']}
        row['paired_ratios']=[(p['current']['result']['scf_seconds']+p['current']['result']['gradient_seconds'])/(p['combined']['result']['scf_seconds']+p['combined']['result']['gradient_seconds']) for p in row['pairs']]
        row['speedup']=row['combined_medians']['current']/row['combined_medians']['combined']
        row['status']='passed'; save()
        print(case,'speedup',row['speedup'],flush=True)
    report['status']='passed'; save()
