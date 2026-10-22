"""Fresh-process capacity/transfer qualification, separate from speed timing."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from memory_profile import peak_memory
from molecular import SETTINGS, _metadata, _file_identity, run_case, compare_cases


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('mode', ['resident', 'host', 'disk', 'cached'])
@pytest.mark.parametrize('case', ['vitamin_svp', 'tamoxifen_svp'])
def test_disk_memory(mode, case, tmp_path):
    from gpu4pyscf.df import df
    from gpu4pyscf.df.disk import DiskTensor
    base = Path(__file__).parent
    references = json.loads((base/'stage_profile_results.json').read_text())
    ref = next(r for r in references['cases'] if r['case']==case and r['method']=='B3LYP')
    cpu = ref['cpu_reference']
    report = dict(_metadata('gpu'), status='running', mode=mode, case=case,
                  scratch_filesystem=subprocess.check_output(['df','-T',str(tmp_path)],text=True).strip(),
                  algorithm_source_hashes={p: _file_identity(p)['sha256'] for p in
                    ['gpu4pyscf/df/df.py', 'gpu4pyscf/df/disk.py', 'gpu4pyscf/df/df_jk.py',
                     'gpu4pyscf/df/grad/rhf.py', 'gpu4pyscf/df/grad/uhf.py',
                     'gpu4pyscf/dft/numint.py', 'gpu4pyscf/df/int3c2e_bdiv.py']},
                  benchmark_sha256=_file_identity(__file__)['sha256'],
                  reference_sha256=_file_identity(base/'stage_profile_results.json')['sha256'],
                  memory_profiler_sha256=_file_identity(base/'memory_profile.py')['sha256'],
                  storage_buffer_bytes=(128 if case=='vitamin_svp' else 1024)*1024**2,
                  cache_budget_bytes=(64 if case=='vitamin_svp' else 384)*1024**2 if mode=='cached' else 0,
                  note='Fresh process, cold run, regression grid level1/shared GB10. Instrumented time diagnostic only. Counts are logical array-copy bytes, not physical bus traffic on unified memory. Host-mode streaming bytes inferred from complete loop yields; disk records actual set/get requests. OS file cache and native allocations not included in staging caps.')
    for key in ['pyscf','numpy','cpu_libxc']:
        assert cpu[key] == report[key]
    assert ref['instrumented']['native_libraries'] == report['native_libraries']
    stores, host_streamed = [], [0]
    original, original_loop = df._cholesky_eri, df.DF.loop
    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        stores.extend(dict(shape=list(s.shape), nbytes=s.nbytes,
                           stats=s.stats if isinstance(s, DiskTensor) else None,
                           reference=weakref.ref(s)) for s in result[0])
        return result
    def loop(obj, *args, **kwargs):
        blocks = original_loop(obj, *args, **kwargs)
        try:
            for dense, packed in blocks:
                if isinstance(obj._cderi[0], np.ndarray):
                    host_streamed[0] += packed.nbytes
                yield dense, packed
        finally:
            blocks.close()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(SETTINGS, cpu['settings']))
            stack.enter_context(patch.object(df.DF, 'use_gpu_memory', True if mode=='resident' else False))
            stack.enter_context(patch.object(df.DF, 'cderi_storage', 'disk' if mode in ['disk','cached'] else None))
            stack.enter_context(patch.object(df.DF, 'cderi_directory', tmp_path))
            stack.enter_context(patch.object(df.DF, 'cderi_buffer_bytes', report['storage_buffer_bytes']))
            stack.enter_context(patch.object(df.DF, 'cderi_cache_bytes', report['cache_budget_bytes']))
            stack.enter_context(patch.object(df, '_cholesky_eri', build))
            stack.enter_context(patch.object(df.DF, 'loop', loop))
            with peak_memory() as memory:
                result = run_case('B3LYP', 'gpu')
        comparison = compare_cases(cpu, result)
        assert all(s['reference']() is None for s in stores)
        report.update(result=result, comparison=comparison, memory=memory,
                      cderi_released_by_gradient=True,
                      storage=[dict(shape=s['shape'], nbytes=s['nbytes'],
                                    stats=s['stats'].copy() if s['stats'] is not None else None)
                               for s in stores], host_streamed_bytes_inferred=host_streamed[0],
                      status='passed' if comparison['passed'] else 'failed')
        if mode in ['disk','cached']:
            for s in stores:
                stats = s['stats']
                assert stats['peak_host_buffer_bytes'] <= report['storage_buffer_bytes']
                assert stats['peak_device_buffer_bytes'] <= report['storage_buffer_bytes']
                assert stats['d2h_bytes'] == s['nbytes']
                assert stats['cache_device_bytes'] <= report['cache_budget_bytes']
                assert s['reference']() is None  # gradient reset must release CDERI
        (base/('disk_df_memory_'+case+'_'+mode+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        assert comparison['passed'], comparison
        print(mode, report['storage'], memory, flush=True)
    finally:
        for item in stores:
            store = item['reference']()
            if isinstance(store, DiskTensor):
                store.close()
        assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))
