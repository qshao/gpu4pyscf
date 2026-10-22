"""Benchmark-only decomposition of XC and DF J/K on one GPU."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
import math
from pathlib import Path
import statistics
import threading
import time
from unittest.mock import patch

import pytest

from df_profile import ROOT
from molecular import SETTINGS, _file_identity, _metadata, compare_cases, run_case
from test_stage_profile import stage_profile


@contextmanager
def fine_profile():
    import cupy as cp
    from gpu4pyscf.df import df, df_jk, int3c2e_bdiv
    from gpu4pyscf.df.grad import rhf as df_rhf
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks
    from gpu4pyscf.hessian import rks as hess_rks

    local = threading.local()
    data = dict(timings={}, bindings_restored=False)
    originals = []

    def timed(original, label):
        @wraps(original)
        def wrapped(*args, **kwargs):
            key = label(*args, **kwargs) if callable(label) else label
            if key is None:
                return original(*args, **kwargs)
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                cp.cuda.Device().synchronize()
                row = data['timings'].setdefault(key, dict(calls=0, seconds=0., shapes=[]))
                row['seconds'] += time.perf_counter() - start
                row['calls'] += 1
                if '.contract:' in key:
                    shapes = [list(x.shape) for x in args[1:3]]
                    if shapes not in row['shapes'] and len(row['shapes']) < 12:
                        row['shapes'].append(shapes)
        return wrapped

    def scoped(original, scope):
        @wraps(original)
        def wrapped(*args, **kwargs):
            previous = getattr(local, 'scope', None)
            local.scope = scope
            try:
                return original(*args, **kwargs)
            finally:
                local.scope = previous
        return wrapped

    def xc_label(name):
        def label(*args, **kwargs):
            scope = getattr(local, 'scope', None)
            return scope + '.' + name if scope in ('scf.xc', 'gradient.xc') else None
        return label

    def install(stack, owner, name, replacement):
        originals.append((owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(owner, name, replacement))

    try:
        with ExitStack() as stack:
            for owner, name, scope in [(numint, '_nr_rks_task', 'scf.xc'),
                                        (rks, 'get_exc_full_response', 'gradient.xc'),
                                        (df_rhf, '_jk_energy_per_atom', 'gradient.jk')]:
                install(stack, owner, name, scoped(getattr(owner, name), scope))
            for owner, name, label in [
                (numint, 'eval_ao', xc_label('ao')),
                (numint, 'eval_rho', xc_label('density_dm')),
                (numint, '_eval_rho2', xc_label('density_mo')),
                (numint.NumInt, 'eval_xc_eff', xc_label('functional')),
                (numint, '_scale_ao', xc_label('scale_ao')),
                (numint, 'add_sparse', xc_label('sparse_scatter')),
                (rks, '_gga_grad_sum_', xc_label('gradient_matrix')),
                (hess_rks, 'get_dweight_dA', xc_label('grid_weight_derivative')),
                (df, 'fill_symmetric', 'scf.jk.unpack'),
                (df_jk, 'factorize_dm', 'scf.jk.factorize_dm'),
                (df_rhf, 'int2c2e', 'gradient.jk.metric_integrals'),
                (df_rhf, 'int2c2e_ip1_per_atom', 'gradient.jk.metric_derivative'),
                (df_rhf.libvhf_rys, 'sum_ejk_int3c2e_ip1', 'gradient.jk.derivative3c'),
            ]:
                install(stack, owner, name, timed(getattr(owner, name), label))
            for owner, prefix in [(df_jk, 'scf.jk'), (df_rhf, 'gradient.jk')]:
                install(stack, owner, 'contract', timed(owner.contract,
                    lambda expr, *a, prefix=prefix, **kw: prefix + '.contract:' + expr))

            original_loop = df.DF.loop
            def loop(*args, **kwargs):
                iterator = original_loop(*args, **kwargs)
                advance = timed(lambda: next(iterator), 'scf.jk.block_fetch_including_unpack')
                try:
                    while True:
                        try:
                            block = advance()
                        except StopIteration:
                            return
                        yield block
                finally:
                    iterator.close()
            install(stack, df.DF, 'loop', loop)

            original_solver = df_rhf._gen_metric_solver
            def solver(*args, **kwargs):
                apply = timed(original_solver, 'gradient.jk.metric_factorization')(*args, **kwargs)
                return timed(apply, 'gradient.jk.metric_solve')
            install(stack, df_rhf, '_gen_metric_solver', solver)

            original_evaluator = int3c2e_bdiv.Int3c2eOpt.int3c2e_evaluator
            def evaluator(*args, **kwargs):
                result = original_evaluator(*args, **kwargs)
                if getattr(local, 'scope', None) == 'gradient.jk':
                    return (timed(result[0], 'gradient.jk.integral3c'), *result[1:])
                return result
            install(stack, int3c2e_bdiv.Int3c2eOpt, 'int3c2e_evaluator', evaluator)
            yield data
    finally:
        data['bindings_restored'] = all(getattr(o, n) is f for o, n, f in originals)


@pytest.mark.special
@pytest.mark.benchmark
def test_fine_stage_profile():
    output = Path(__file__).with_name('fine_stage_profile_results.json')
    reference_path = Path(__file__).with_name('stage_profile_results.json')
    references = json.loads(reference_path.read_text())
    assert references['status'] == 'passed'
    report = dict(_metadata('gpu'), status='running', cases=[],
                  reference_report_sha256=_file_identity(reference_path)['sha256'],
                  profiler_sha256=_file_identity(__file__)['sha256'],
                  note='Shared single GB10. Cached matched CPU references. Three synchronized-iteration controls and one instrumented run after warmup. Fine timings synchronize each helper and perturb scheduling/overlap. Inclusive stages and generator retrieval must not be summed with nested children. No kernel-only or production-grid claims.')
    stage_files = ['dft/numint.py', 'grad/rks.py', 'df/df.py', 'df/df_jk.py',
                   'df/grad/rhf.py', 'df/int3c2e_bdiv.py', 'hessian/rks.py']
    report['profiled_source_hashes'] = {f: _file_identity(ROOT/'gpu4pyscf'/f)['sha256']
                                       for f in stage_files}
    output.write_text(json.dumps(report, indent=2) + '\n')
    for method in ('PBE', 'B3LYP'):
        ref = next(c for c in references['cases']
                   if c['case'] == 'tamoxifen_svp' and c['method'] == method)
        geometry = ROOT/'examples/Tamoxifen.xyz'
        assert _file_identity(geometry)['sha256'] == ref['geometry_sha256']
        cpu = ref['cpu_reference']
        for key in ('pyscf', 'numpy', 'cpu_libxc'):
            assert cpu[key] == report[key], key
        assert ref['instrumented']['source_hashes'] == report['source_hashes']
        assert ref['instrumented']['native_libraries'] == report['native_libraries']
        row = dict(method=method, case='tamoxifen_svp', status='running', controls=[])
        report['cases'].append(row)
        with patch.dict(SETTINGS, cpu['settings']):
            print(method + ': GPU warmup; reusing validated CPU reference', flush=True)
            row['warmup'] = run_case(method, 'gpu')
            row['warmup_comparison'] = compare_cases(cpu, row['warmup'])
            assert row['warmup_comparison']['passed']
            for repeat in range(3):
                result = run_case(method, 'gpu')
                comparison = compare_cases(cpu, result)
                row['controls'].append(dict(result=result, comparison=comparison))
                output.write_text(json.dumps(report, indent=2) + '\n')
                assert comparison['passed'], comparison
                print(f'{method}: control {repeat+1}/3', flush=True)
            print(method + ': instrumented run', flush=True)
            with stage_profile() as coarse, fine_profile() as fine:
                measured = run_case(method, 'gpu')
            comparison = compare_cases(cpu, measured)
            row.update(instrumented=measured, comparison=comparison, coarse=coarse, fine=fine)
            assert comparison['passed'], comparison
            assert fine['bindings_restored']
            assert all(math.isfinite(t['seconds']) and t['seconds'] >= 0
                       for t in fine['timings'].values())
            assert fine['timings']['scf.xc.ao']['calls'] > 0
            assert fine['timings']['gradient.xc.functional']['calls'] > 0
            if method == 'B3LYP':
                assert fine['timings']['scf.jk.contract:Lij,njk->nikL']['calls'] > 0
                assert fine['timings']['gradient.jk.integral3c']['calls'] > 0
            assert all(c['result']['iterations'] == measured['iterations'] for c in row['controls'])
            totals = [c['result']['scf_seconds'] + c['result']['gradient_seconds']
                      for c in row['controls']]
            row.update(control_median_seconds=statistics.median(totals),
                       control_range_seconds=[min(totals), max(totals)], status='passed')
            row['instrumented_over_control_ratio'] = (
                measured['scf_seconds'] + measured['gradient_seconds']) / row['control_median_seconds']
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{method}: passed, overhead ratio {row["instrumented_over_control_ratio"]:.3f}', flush=True)
    report['status'] = 'passed'
    output.write_text(json.dumps(report, indent=2) + '\n')
