"""Fresh profiling after adaptive density and symmetric Gram promotion."""
import json
from pathlib import Path
import time
from unittest.mock import patch

import cupy as cp
import pytest

from molecular import SETTINGS, _metadata, _file_identity, run_case, compare_cases
from test_stage_profile import stage_profile
from test_fine_stage_profile import fine_profile


@pytest.mark.special
@pytest.mark.benchmark
def test_optimized_profile():
    from gpu4pyscf.df import df_jk
    output = Path(__file__).with_name('optimized_stage_profile_results.json')
    reference_path = Path(__file__).with_name('stage_profile_results.json')
    refs = json.loads(reference_path.read_text())
    assert refs['status'] == 'passed'
    report = dict(_metadata('gpu'), status='running', cases=[],
                  profiler_sha256=_file_identity(__file__)['sha256'],
                  reference_sha256=_file_identity(reference_path)['sha256'],
                  algorithm_source_hashes={p: _file_identity(p)['sha256'] for p in
                      ['gpu4pyscf/df/df_jk.py', 'gpu4pyscf/dft/numint.py']},
                  note='Current optimized defaults. Shared GB10; one warmup, one control and one instrumented run per method. Synchronized inclusive wall times perturb scheduling; nested timings must not be summed. density_mo includes adaptive dispatch and nests density_dm when selected. Grid level 1 unpruned; no production-grid convergence claim.')
    def save(): output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    for method in ['PBE', 'B3LYP']:
        ref = next(r for r in refs['cases'] if r['case']=='tamoxifen_svp' and r['method']==method)
        cpu = ref['cpu_reference']
        assert _file_identity(cpu['settings']['atom'])['sha256'] == ref['geometry_sha256']
        for key in ['pyscf', 'numpy', 'cpu_libxc']: assert cpu[key] == report[key]
        assert ref['instrumented']['native_libraries'] == report['native_libraries']
        row = dict(method=method, case='tamoxifen_svp', runs={})
        report['cases'].append(row)
        with patch.dict(SETTINGS, cpu['settings']):
            for label in ['warmup', 'control', 'instrumented']:
                print(method, label, flush=True)
                if label == 'instrumented':
                    gram = dict(calls=0, seconds=0.)
                    original = df_jk._symmetric_gram
                    def wrapped(*args, **kw):
                        cp.cuda.Device().synchronize(); start = time.perf_counter()
                        try: return original(*args, **kw)
                        finally:
                            cp.cuda.Device().synchronize()
                            gram['calls'] += 1
                            gram['seconds'] += time.perf_counter()-start
                    with stage_profile() as coarse, fine_profile() as fine, patch.object(df_jk, '_symmetric_gram', wrapped):
                        result = run_case(method, 'gpu')
                    assert fine['bindings_restored']
                    row.update(coarse=coarse, fine=fine, symmetric_gram=gram)
                else: result = run_case(method, 'gpu')
                comparison = compare_cases(cpu, result)
                row['runs'][label] = dict(result=result, comparison=comparison)
                save()
                assert comparison['passed'], comparison
            assert row['runs']['control']['result']['iterations'] == result['iterations']
            row['status'] = 'passed'
            print(method, 'passed', flush=True)
    report['status'] = 'passed'; save()
