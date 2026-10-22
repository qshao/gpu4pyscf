"""Warm repeated timings without the memory allocation hook."""
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import subprocess
import weakref
from unittest.mock import patch

import pytest

from molecular import SETTINGS, _metadata, _file_identity, run_case, compare_cases


@pytest.mark.special
@pytest.mark.benchmark
def test_disk_speed(tmp_path):
    from gpu4pyscf.df import df
    from gpu4pyscf.df.disk import DiskTensor
    base = Path(__file__).parent
    ref = next(r for r in json.loads((base/'stage_profile_results.json').read_text())['cases']
               if r['case']=='vitamin_svp' and r['method']=='B3LYP')
    cpu = ref['cpu_reference']
    report = dict(_metadata('gpu'), status='running', case='vitamin_svp', warmups=[], rounds=[],
                  scratch_filesystem=subprocess.check_output(['df','-T',str(tmp_path)],text=True).strip(),
                  benchmark_sha256=_file_identity(__file__)['sha256'],
                  reference_sha256=_file_identity(base/'stage_profile_results.json')['sha256'],
                  algorithm_source_hashes={p:_file_identity(p)['sha256'] for p in
                    ['gpu4pyscf/df/df.py','gpu4pyscf/df/disk.py','gpu4pyscf/df/int3c2e_bdiv.py',
                     'gpu4pyscf/df/df_jk.py','gpu4pyscf/df/grad/rhf.py','gpu4pyscf/df/grad/uhf.py',
                     'gpu4pyscf/dft/numint.py']},
                  note='Shared GB10, regression grid1. One warmup per mode, 3 rotated rounds, no allocation hook. Entire SCF+gradient including fresh integral construction each run. OS page cache uncontrolled; logical transfer bytes are not physical bus traffic.128MiB staging, optional64MiB resident prefix. Weakrefs preserve normal CDERI release before gradients.')
    for key in ['pyscf','numpy','cpu_libxc']:
        assert cpu[key] == report[key]
    assert ref['instrumented']['native_libraries'] == report['native_libraries']
    output = base/'disk_df_speed_results.json'
    def save(): output.write_text(json.dumps(report, indent=2)+'\n')
    original = df._cholesky_eri
    def run(mode):
        print('starting',mode,flush=True)
        stores = []
        def build(*args, **kwargs):
            result = original(*args, **kwargs)
            stores.extend(dict(reference=weakref.ref(s),
                               stats=s.stats if isinstance(s,DiskTensor) else None) for s in result[0])
            return result
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(SETTINGS,cpu['settings']))
                for name,value in dict(use_gpu_memory=mode=='resident',
                        cderi_storage='disk' if mode in ['disk','cached'] else None,
                        cderi_directory=tmp_path,cderi_buffer_bytes=128*1024**2,
                        cderi_cache_bytes=64*1024**2 if mode=='cached' else 0).items():
                    stack.enter_context(patch.object(df.DF,name,value))
                stack.enter_context(patch.object(df,'_cholesky_eri',build))
                result = run_case('B3LYP','gpu')
            comparison = compare_cases(cpu,result)
            assert comparison['passed'],comparison
            assert all(s['reference']() is None for s in stores)
            print('finished',mode,result['iterations'],flush=True)
            return dict(result=result,comparison=comparison,
                        transfers=[s['stats'].copy() for s in stores if s['stats'] is not None])
        finally:
            for item in stores:
                s = item['reference']()
                if isinstance(s,DiskTensor): s.close()
            assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))
    modes = ['resident','host','disk','cached']
    for mode in modes:
        report['warmups'].append(dict(mode=mode,**run(mode))); save()
    for repeat in range(3):
        row = {}
        for mode in modes[repeat:]+modes[:repeat]:
            row[mode] = run(mode)
        assert len({row[m]['result']['iterations'] for m in ['resident','host','disk']}) == 1
        report['rounds'].append(row); save()
        print('storage timing round',repeat+1,'passed',flush=True)
    report['medians_seconds'] = {m:statistics.median(
        r[m]['result']['scf_seconds']+r[m]['result']['gradient_seconds'] for r in report['rounds']) for m in modes}
    report['cache_performance_status'] = ('not_qualified_iteration_regression' if any(
        r['cached']['result']['iterations'] != r['disk']['result']['iterations']
        for r in report['rounds']) else 'iteration_matched')
    report['iteration_counts'] = [{m:r[m]['result']['iterations'] for m in modes}
                                  for r in report['rounds']]
    report['status'] = 'passed'; save()
    print(report['medians_seconds'],flush=True)
