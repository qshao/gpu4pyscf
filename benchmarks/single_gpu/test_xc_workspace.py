"""Fresh-process XC workspace measurements at identical corrected orbitals."""
from contextlib import nullcontext
import json
import traceback
from unittest.mock import patch
import statistics
import time

import numpy as np
import pytest

from corrected_optimization import BASE, ROOT, CASE, new_report, gradient_inputs, check_gradient, digest, save


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('variant', ['baseline', 'capped', 'full_capped'])
def test_xc_workspace_memory(variant):
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf.grad import rks
    from gpu4pyscf.hessian import rks as hessian_rks
    from gradient_snapshot import restore
    from memory_profile import peak_memory
    from xc_workspace import capped_workspace,original_workspace

    profile, identity = gradient_inputs()
    output, report = new_report(f'xc_{variant}_memory.json', __file__)
    report.update(variant=variant, profile_sha256=identity,
                  xc_source_sha256=digest(ROOT/'gpu4pyscf/grad/rks.py'),
                  candidate_sha256=digest(BASE/'xc_workspace.py'),
                  orbital_helper_sha256=digest(BASE/'gradient_snapshot.py'),
                  memory_profiler_sha256=digest(BASE/'memory_profile.py'),
                  orbitals=profile['orbitals'], batches=[], workspace_target_bytes=512*1024**2,
                  note='Fresh-process gradient allocations exclude preexisting/native storage; RSS and CuPy metrics are separate. The cap bounds an estimated batch target, not total memory. Timing under allocation instrumentation is diagnostic only.')
    save(output, report)
    try:
        assert cp.get_default_memory_pool().used_bytes() == 0
        with lib.with_omp_threads(4):
            mf = restore(CASE, profile['orbitals']['path'])
            original = hessian_rks.get_dweight_dA
            def weights(mol, grids, grid_range):
                report['batches'].append(dict(ngrids=grid_range[1]-grid_range[0],
                    estimated_bytes=(grid_range[1]-grid_range[0])*mol.natm*6*8))
                return original(mol, grids, grid_range)
            with patch.object(hessian_rks, 'get_dweight_dA', weights):
                with capped_workspace() if variant != 'baseline' else original_workspace():
                    with peak_memory() as observed:
                        if variant == 'full_capped':
                            grad = mf.nuc_grad_method()
                            grad.grid_response = grad.auxbasis_response = True
                            actual = grad.kernel()
                        else:
                            actual, _ = rks.get_exc_full_response(mf._numint, mf.mol, mf.grids,
                                                                 mf.xc, mf.make_rdm1())
        actual = np.asarray(actual)
        report.update(memory=observed, gradient=actual.tolist())
        save(output, report)  # Preserve observations even when an accuracy gate fails.
        if variant == 'full_capped':
            report['comparison'] = check_gradient(profile['result'], actual)
        elif variant == 'capped':
            path = output.parent/'xc_baseline_memory.json'
            baseline = json.loads(path.read_text())
            assert baseline['status'] == 'passed' and baseline['profile_sha256'] == identity
            assert baseline['xc_source_sha256'] == report['xc_source_sha256']
            error = float(np.max(np.abs(actual-np.asarray(baseline['gradient']))))
            report.update(baseline_sha256=digest(path), error_to_baseline=error)
            save(output, report)
            assert error <= 1e-9, error
        assert np.isfinite(actual).all()
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        save(output, report)
        raise
    save(output, report)
    print('XC workspace', variant, report['memory']['peak_tracked_live_bytes'], flush=True)


@pytest.mark.special
@pytest.mark.benchmark
def test_xc_workspace_pairs():
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf.grad import rks
    from gradient_snapshot import restore
    from xc_workspace import capped_workspace,original_workspace

    profile, identity = gradient_inputs()
    baseline_path = BASE/'optimization_results/xc_baseline_memory.json'
    capped_path = BASE/'optimization_results/xc_capped_memory.json'
    baseline = json.loads(baseline_path.read_text())
    capped = json.loads(capped_path.read_text())
    assert baseline['status'] == capped['status'] == 'passed'
    assert baseline['profile_sha256'] == capped['profile_sha256'] == identity
    assert capped['candidate_sha256'] == digest(BASE/'xc_workspace.py')
    output, report = new_report('xc_pairs.json', __file__)
    report.update(profile_sha256=identity, baseline_sha256=digest(baseline_path),
                  capped_sha256=digest(capped_path), candidate_sha256=digest(BASE/'xc_workspace.py'),
                  xc_source_sha256=digest(ROOT/'gpu4pyscf/grad/rks.py'), warmups=[], pairs=[],
                  note='XC component only; same converged orbitals, all grid response retained. Uninstrumented alternating timings after one warmup per variant. No whole-gradient or whole-SCF speed claim.')
    save(output, report)
    try:
        with lib.with_omp_threads(4):
            mf = restore(CASE, profile['orbitals']['path'])
            dm = mf.make_rdm1()
            def run(variant):
                with capped_workspace() if variant == 'capped' else original_workspace():
                    cp.cuda.Device().synchronize()
                    start = time.perf_counter()
                    actual, _ = rks.get_exc_full_response(mf._numint, mf.mol, mf.grids, mf.xc, dm)
                    cp.cuda.Device().synchronize()
                    elapsed = time.perf_counter()-start
                error = float(np.max(np.abs(np.asarray(actual)-np.asarray(baseline['gradient']))))
                assert np.isfinite(error) and error <= 1e-9, error
                return dict(seconds=elapsed, error_to_baseline=error)
            for variant in ['baseline', 'capped']:
                report['warmups'].append(dict(variant=variant, **run(variant)))
                save(output, report)
            for repeat in range(3):
                order = ['baseline', 'capped'] if repeat % 2 == 0 else ['capped', 'baseline']
                report['pairs'].append({variant:run(variant) for variant in order})
                save(output, report)
                print('XC pair', repeat+1, flush=True)
        report['medians'] = {variant:statistics.median(p[variant]['seconds'] for p in report['pairs'])
                             for variant in ['baseline', 'capped']}
        report.update(speedup=report['medians']['baseline']/report['medians']['capped'],
                      paired_ratios=[p['baseline']['seconds']/p['capped']['seconds'] for p in report['pairs']],
                      status='passed')
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        save(output, report)
        raise
    save(output, report)
