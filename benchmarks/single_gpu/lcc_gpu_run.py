"""One fresh-process original/current workflow on an LCC GPU (V100/A100/H200).

Usage: python lcc_gpu_run.py GPU SIZE VERSION MODE REPEAT
Same scientific settings and CPU references as test_original_comparison_run.py;
packages come from /scratch/qsh226/g4p_lcc/{original,current} built for sm70/80/90.
"""
import gc
import json
import platform
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PACKAGES = Path('/scratch/qsh226/g4p_lcc')
OUT = HERE / 'lcc_gpu_comparison' / 'results'


def cpu_reference(size, expected, digest, CASES, BASE):
    """Same checks as original_comparison.cpu_reference, on copies of the CPU references whose
    checkout path was rewritten from the original machine's to this machine's (no numbers changed)."""
    name = CASES[size] + ('_seeded_cpu.json' if size == 'large' else '_cpu.json')
    data = json.loads((HERE / 'lcc_gpu_comparison' / 'cpu_references' / name).read_text())
    assert data['status'] == 'passed' and data['result']['converged']
    assert data['geometry_sha256'] == digest(expected['atom'])
    actual = dict(data['result']['settings'], cpu_threads=expected['cpu_threads'])
    assert actual == expected
    return data


def main(gpu, size, version, mode, repeat):
    assert version in ('original', 'current', 'disk') and mode in ('time', 'memory')
    sys.path.insert(0, str(PACKAGES / ('original' if version == 'original' else 'current')))
    import numpy as np
    import cupy as cp
    import gpu4pyscf
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    from gpu4pyscf.df import df
    from pyscf import lib
    import pyscf
    from original_comparison import save, case_settings, configure_cpu, digest, CASES, BASE
    from memory_profile import peak_memory
    from seeded_reference import compare_execution_cases

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f'{gpu}_{size}_{version}_{mode}_{repeat}.json'
    if output.exists():
        raise FileExistsError(output)
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name']
    config = case_settings(size)
    reference = cpu_reference(size, config, digest, CASES, BASE)
    report = dict(status='running', gpu_label=gpu, size=size, version=version, mode=mode, repeat=repeat,
                  device_name=name.decode() if isinstance(name, bytes) else name,
                  device_total_bytes=int(cp.cuda.runtime.memGetInfo()[1]),
                  compute_capability=cp.cuda.Device().compute_capability,
                  gpu4pyscf_source=gpu4pyscf.__file__, pyscf=pyscf.__version__, numpy=np.__version__,
                  cupy=cp.__version__, host=platform.node(), settings=config,
                  cutensor_available=bool(__import__('cupy.cuda.cutensor', fromlist=['x']).available),
                  contract_engine=__import__('gpu4pyscf.lib.cutensor', fromlist=['x']).contract_engine or 'cutensor')
    save(output, report)
    stores = []
    paths = []
    build = df._cholesky_eri

    def observe(*args, **kwargs):
        value = build(*args, **kwargs)
        for store in value[0]:
            is_disk = hasattr(store, 'stats')
            stores.append(dict(shape=list(store.shape), nbytes=store.nbytes, is_disk=is_disk,
                               gpu_array=isinstance(store, cp.ndarray), stats=store.stats if is_disk else None))
            if is_disk:
                paths.append(store.path)
        return value

    mf = None
    try:
        import tempfile
        scratch = tempfile.mkdtemp(prefix='cderi-', dir='/scratch/qsh226/tmp_cderi') if version == 'disk' else None
        with lib.with_omp_threads(4), patch.object(df, '_cholesky_eri', observe):
            cp.cuda.Device().synchronize()
            with peak_memory() if mode == 'memory' else nullcontext({}) as memory:
                start = time.perf_counter()
                cpu = configure_cpu(size)
                dm = cpu.get_init_guess()
                mf = cpu.to_gpu()
                mf.grids.prune = nwchem_prune
                mf.with_df.use_gpu_memory = None if version == 'disk' else True
                if version != 'original':
                    mf.with_df.cderi_storage = 'disk' if version == 'disk' else None
                    mf.with_df.cderi_directory = scratch
                    mf.with_df.cderi_buffer_bytes = 4 * 1024**3
                    mf.with_df.cderi_cache_bytes = 0
                setup = time.perf_counter() - start
                scf_start = time.perf_counter()
                energy = mf.kernel(dm0=dm)
                cp.cuda.Device().synchronize()
                scf_seconds = time.perf_counter() - scf_start
                if not mf.converged:
                    raise RuntimeError('SCF failed to converge')
                grad = mf.nuc_grad_method()
                grad.grid_response = grad.auxbasis_response = True
                grad_start = time.perf_counter()
                gradient = grad.kernel()
                cp.cuda.Device().synchronize()
                gradient_seconds = time.perf_counter() - grad_start
                seconds = time.perf_counter() - start
                actual = np.asarray(gradient.get() if hasattr(gradient, 'get') else gradient)
                result = dict(method='B3LYP', settings=config, converged=bool(mf.converged),
                              energy_hartree=float(energy), gradient_hartree_per_bohr=actual.tolist())
        comparison = compare_execution_cases(reference['result'], result)
        report.update(result=result, setup_seconds=setup, scf_seconds=scf_seconds,
                      gradient_seconds=gradient_seconds, workflow_seconds=seconds, cycles=int(mf.cycles),
                      nao=mf.mol.nao, natm=mf.mol.natm, integrals=stores, memory=memory,
                      comparison=comparison,
                      status='passed' if comparison['passed'] else 'failed_accuracy')
        save(output, report)
        print('LCC', gpu, size, version, mode, repeat, round(seconds, 3), comparison, flush=True)
    except BaseException as error:
        oom = 'out of memory' in str(error).lower() or 'OutOfMemory' in type(error).__name__
        report.update(status='out_of_memory' if oom else 'failed_execution',
                      error=str(error), traceback=traceback.format_exc())
        save(output, report)
        print('LCC FAILED', gpu, size, version, mode, repeat, report['status'], flush=True)
    finally:
        try:
            if mf is not None:
                mf.with_df.reset()
            gc.collect()
        except Exception:
            pass


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
