"""Run each memory variant in a fresh pytest process."""
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_profile import peak_memory
from molecular import SETTINGS, _metadata, _file_identity, run_case, compare_cases


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('version',['current','compact','combined'])
def test_memory_qualification(version):
    import speed_memory_candidates as candidate
    output=Path(__file__).with_name('speed_memory_usage_'+version+'.json')
    reference_path=Path(__file__).with_name('stage_profile_results.json')
    references=json.loads(reference_path.read_text())
    assert references['status']=='passed'
    ref=next(r for r in references['cases'] if r['case']=='tamoxifen_svp' and r['method']=='B3LYP')
    cpu=ref['cpu_reference']
    report=dict(_metadata('gpu'),status='running',version=version,
        baseline_sha256=_file_identity(Path(__file__).with_name('speed_memory_baseline.json'))['sha256'],
        candidate_sha256=_file_identity(candidate.__file__)['sha256'] if version!='current' else None,
        benchmark_sha256=_file_identity(__file__)['sha256'],
        algorithm_source_hashes={p:_file_identity(p)['sha256'] for p in
            ['gpu4pyscf/df/df_jk.py','gpu4pyscf/df/grad/rhf.py','gpu4pyscf/lib/cutensor.py']},
        memory_profiler_sha256=_file_identity(Path(__file__).with_name('memory_profile.py'))['sha256'],
        note='Fresh process, cold complete run including construction/guess; allocation-hook instrumentation. Performance results require a separate uninstrumented paired benchmark. Regression gridlevel1, no production-grid qualification.')
    assert _file_identity(cpu['settings']['atom'])['sha256']==ref['geometry_sha256']
    for key in ['pyscf','numpy','cpu_libxc']: assert cpu[key]==report[key]
    assert ref['instrumented']['native_libraries']==report['native_libraries']
    output.write_text(json.dumps(report,indent=2)+'\n')
    context=nullcontext() if version=='current' else candidate.version(version)
    with patch.dict(SETTINGS,cpu['settings']),context,peak_memory() as memory:
        result=run_case('B3LYP','gpu')
    comparison=compare_cases(cpu,result)
    report.update(result=result,comparison=comparison,memory=memory,
                  status='passed' if comparison['passed'] else 'failed')
    output.write_text(json.dumps(report,indent=2)+'\n')
    assert comparison['passed'],comparison
    print(version,memory,flush=True)
