"""Stage profile of one SCF + full-response gradient workflow on an LCC GPU, with an optional gradient-workspace cap.

Usage: python lcc_profile.py GPU SIZE VERSION WORKSPACE MODE [STORAGE]
  SIZE      small | medium | large | valinomycin (no CPU reference: a capacity observation, accuracy unchecked)
  STORAGE   gpu (default, resident CDERI) | host (use_gpu_memory=False, pinned host RAM) | disk (current only)
  VERSION   original | current
  WORKSPACE default | unlimited | <GiB as float>   (current only: gpu4pyscf.__config__.df_gradient_workspace_bytes)
  MODE      time   (synchronized stage timers, no allocator hook)
            memory (allocator hook around SCF and around the gradient separately; timings diagnostic only)
Results: lcc_profile_results/GPU_SIZE_VERSION_WORKSPACE_MODE.json. Accuracy is checked against the CPU reference.
"""
import json
import os
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PACKAGES = Path(os.environ.get('LCC_PACKAGES', '/scratch/qsh226/g4p_lcc'))
OUT = HERE / 'lcc_profile_results'


@contextmanager
def stage_timers(cp):
    """Synchronized, inclusive timers; nested scopes (e.g. df build inside get_jk) are labelled as such."""
    from gpu4pyscf.df import df, df_jk
    from gpu4pyscf.df.grad import rhf as df_rhf, rks as df_rks
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rhf, rks
    from gpu4pyscf.scf import hf
    targets = [
        (df_jk._DFHF, 'get_veff', 'scf.veff'),
        (df_jk._DFHF, 'get_jk', 'scf.jk'),
        (numint.NumInt, 'nr_rks', 'scf.xc_quadrature'),
        (hf.SCF, 'eig', 'scf.diagonalization'),
        (df.DF, 'build', 'scf.df_build_nested_in_jk'),
        (rhf.GradientsBase, '_hcore_energy', 'gradient.one_electron'),
        (df_rks.Gradients, 'energy_ee', 'gradient.two_electron'),
        (df_rks.Gradients, 'jk_energy_per_atom', 'gradient.df_jk_total'),
        (df_rhf, '_jk_energy_per_atom', 'gradient.df_jk_core_nested'),
        (rks, 'get_exc_full_response', 'gradient.xc_full_response'),
    ]
    result = {name: {'calls': 0, 'seconds': 0.} for _, _, name in targets}

    def timed(original, label):
        @wraps(original)
        def wrapped(*args, **kwargs):
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                cp.cuda.Device().synchronize()
                result[label]['seconds'] += time.perf_counter() - start
                result[label]['calls'] += 1
        return wrapped

    with ExitStack() as stack:
        for owner, name, label in targets:
            if hasattr(owner, name):
                stack.enter_context(patch.object(owner, name, timed(getattr(owner, name), label)))
            else:
                result.pop(label)
        yield result


LADDER = {'water008': 'water_clusters/008.xyz', 'water009': 'water_clusters/009.xyz', 'water010': 'water_clusters/010.xyz',
          **{f'gly{n}': f'gly/gly{n}.xyz' for n in range(30, 130, 10)}}


class GradTrace:
    """Allocator-level live-byte tracker: records the global live-storage peak reached while each named step runs.

    Only allocations made while installed are counted (like memory_profile.peak_memory), so persistent tensors the
    gradient creates itself (for example j3c_oo) are included and storage from earlier phases is not.
    """

    def __init__(self, cp):
        import threading
        self.cp, self.label, self.live, self.peaks = cp, 'gradient.other', 0, {}
        self.lock = threading.Lock()
        self.original = None

    def install(self):
        cp = self.cp
        self.original = cp.cuda.get_allocator()
        trace = self

        def allocate(size):
            pointer = trace.original(size)

            class Owner:
                def __init__(self_inner):
                    self_inner.pointer = pointer      # keeps the original allocation alive

                def __del__(self_inner):
                    with trace.lock:
                        trace.live -= size
            owner = Owner()
            with trace.lock:
                trace.live += size
                if trace.live > trace.peaks.get(trace.label, (0, 0))[0]:
                    trace.peaks[trace.label] = (trace.live, size)
            return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(pointer.ptr, size, owner, device_id=pointer.device_id), 0)
        cp.cuda.set_allocator(allocate)

    def uninstall(self):
        self.cp.cuda.Device().synchronize()
        self.cp.cuda.set_allocator(self.original)

    def wrap(self, owner, name, label=None):
        original = getattr(owner, name)
        label = label or name
        trace = self

        def wrapped(*args, **kwargs):
            previous = trace.label
            shapes = ','.join('x'.join(map(str, a.shape)) for a in args if hasattr(a, 'shape'))
            trace.label = f'{label}({args[0] if args and isinstance(args[0], str) else ""}) [{shapes}]'[:150]
            try:
                return original(*args, **kwargs)
            finally:
                trace.label = previous
        setattr(owner, name, wrapped)
        return original


def main(gpu, size, version, workspace, mode, storage='gpu'):
    assert version in ('original', 'current') and mode in ('time', 'memory') and storage in ('gpu', 'host', 'disk')
    assert storage != 'disk' or version == 'current'
    sys.path.insert(0, str(PACKAGES / version))
    import numpy as np
    import cupy as cp
    import gpu4pyscf
    from gpu4pyscf import __config__
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    from pyscf import lib
    from original_comparison import save, case_settings, configure_cpu, digest, CASES, BASE
    from lcc_gpu_run import cpu_reference
    from memory_profile import peak_memory
    from seeded_reference import compare_execution_cases

    import original_comparison
    import comprehensive
    original_comparison.CASES.setdefault('valinomycin', 'valinomycin_svp_b3lyp')
    for name, relative in LADDER.items():            # capacity-ladder systems: B3LYP/def2-SVP/def2-universal-jkfit, no CPU reference
        comprehensive.CASES.setdefault(name + '_svp_b3lyp', ('../' + relative, 'def2-svp', 'B3LYP'))
        original_comparison.CASES.setdefault(name, name + '_svp_b3lyp')
    OUT.mkdir(exist_ok=True)
    output = OUT / f'{gpu}_{size}_{version}_{workspace}_{mode}{"" if storage == "gpu" else "_" + storage}.json'
    if output.exists():
        raise FileExistsError(output)
    config = case_settings(size)
    try:
        reference = cpu_reference(size, config, digest, CASES, BASE)
    except FileNotFoundError:
        reference = None  # capacity observation: no independent CPU oracle exists for this system
    if workspace == 'default':
        cap = None
    else:
        assert version == 'current', 'the workspace cap exists only in the current code'
        cap = 1 << 60 if workspace == 'unlimited' else int(float(workspace) * 1024**3)
        __config__.df_gradient_workspace_bytes = cap
    # Build budgets (default: the library's 4 GiB). The capacity runner raises them when a rung's minimum requirement is larger.
    if os.environ.get('LCC_BUILD_BYTES'):
        __config__.df_cderi_build_workspace_bytes = int(os.environ['LCC_BUILD_BYTES'])
    staging_bytes = int(os.environ.get('LCC_STAGING_BYTES', 4 * 1024**3))
    name = cp.cuda.runtime.getDeviceProperties(0)['name']
    report = dict(status='running', gpu_label=gpu, size=size, version=version, workspace=workspace, mode=mode, storage=storage,
                  build_budget_bytes=int(os.environ['LCC_BUILD_BYTES']) if os.environ.get('LCC_BUILD_BYTES') else None,
                  staging_budget_bytes=int(os.environ.get('LCC_STAGING_BYTES', 4 * 1024**3)),
                  workspace_bytes=cap, device_name=name.decode() if isinstance(name, bytes) else name,
                  device_total_bytes=int(cp.cuda.runtime.memGetInfo()[1]), gpu4pyscf_source=gpu4pyscf.__file__)
    save(output, report)
    plans = []
    try:
        from gpu4pyscf.df.grad import rhf as df_rhf
        if hasattr(df_rhf, '_gradient_workspace_sizes'):
            original_plan = df_rhf._gradient_workspace_sizes

            def recording(*args, **kwargs):
                value = original_plan(*args, **kwargs)
                plans.append(dict(args=[int(a) for a in args], batch_aux=value[0], block_aux=value[1]))
                return value
            patcher = patch.object(df_rhf, '_gradient_workspace_sizes', recording)
        else:
            patcher = ExitStack()
        memory = {}
        trace = None
        if os.environ.get('LCC_TRACE_GRAD'):
            from gpu4pyscf.df.grad import rhf as _r
            trace = GradTrace(cp)
            for nm in ('contract', '_gradient_contract', '_apply_auxiliary_metric', 'cholesky', 'solve_triangular', 'asarray'):
                if hasattr(_r, nm):
                    trace.wrap(_r, nm)
        with lib.with_omp_threads(4), patcher, stage_timers(cp) as stages:
            cp.matmul(cp.ones((64, 64)), cp.ones((64, 64)))
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            cpu = configure_cpu(size)
            dm = cpu.get_init_guess()
            mf = cpu.to_gpu()
            mf.grids.prune = nwchem_prune
            mf.with_df.use_gpu_memory = {'gpu': True, 'host': False, 'disk': None}[storage]
            if version == 'current':
                mf.with_df.cderi_storage = 'disk' if storage == 'disk' else None
                mf.with_df.cderi_directory = os.environ.get('QM_SUITE_SCRATCH') if storage == 'disk' else None
                mf.with_df.cderi_buffer_bytes = staging_bytes
                mf.with_df.cderi_cache_bytes = 0
            setup = time.perf_counter() - start
            grid_info = {}
            if os.environ.get('LCC_PREBUILD_GRIDS'):
                # One-time grid construction timed on its own so it no longer sits inside the first XC call.
                g0 = time.perf_counter(); mf.grids.build(); cp.cuda.Device().synchronize()
                grid_info = dict(grid_build_seconds=time.perf_counter() - g0, grid_points=int(mf.grids.coords.shape[0]))

            def phase(label, work):
                if mode == 'memory':
                    with peak_memory() as data:
                        value = work()
                        cp.cuda.Device().synchronize()
                    memory[label] = dict(peak_live_bytes=data['peak_live_bytes'],
                                         peak_tracked_live_bytes=data['peak_tracked_live_bytes'],
                                         initial_live_bytes=data['initial_pool_live_bytes'])
                    return value
                value = work()
                cp.cuda.Device().synchronize()
                return value

            t0 = time.perf_counter()
            energy = phase('scf', lambda: mf.kernel(dm0=dm))
            scf_seconds = time.perf_counter() - t0
            if not mf.converged:
                raise RuntimeError('SCF failed to converge')
            grad = mf.nuc_grad_method()
            grad.grid_response = grad.auxbasis_response = True
            if trace is not None:
                trace.install()
            t0 = time.perf_counter()
            gradient = phase('gradient', grad.kernel)
            if trace is not None:
                trace.uninstall()
            gradient_seconds = time.perf_counter() - t0
            total = time.perf_counter() - start
            actual = np.asarray(gradient.get() if hasattr(gradient, 'get') else gradient)
        result = dict(method='B3LYP', settings=config, converged=True, energy_hartree=float(energy),
                      gradient_hartree_per_bohr=actual.tolist())
        comparison = compare_execution_cases(reference['result'], result) if reference else None
        rss = next((int(l.split()[1]) * 1024 for l in open('/proc/self/status') if l.startswith('VmHWM')), None)
        report.update(status=('capacity_observation' if comparison is None else 'passed' if comparison['passed'] else 'failed_accuracy'),
                      comparison=comparison, host_peak_rss_bytes=rss,
                      energy_hartree=float(energy), cycles=int(mf.cycles), setup_seconds=setup,
                      scf_seconds=scf_seconds, gradient_seconds=gradient_seconds, workflow_seconds=total,
                      stages=stages, workspace_plans=plans, memory=memory, grid_info=grid_info, grad_trace=(sorted(((round(v[0]/1e9, 3), round(v[1]/1e9, 3), k) for k, v in trace.peaks.items()), reverse=True)[:14] if trace else None),
                      gradient_digest=float(np.abs(actual).sum()))
    except BaseException as error:
        oom = 'out of memory' in str(error).lower() or 'OutOfMemory' in type(error).__name__
        report.update(status='out_of_memory' if oom else 'failed_execution', error=str(error),
                      traceback=traceback.format_exc(), workspace_plans=plans)
    save(output, report)
    print('PROFILE', gpu, size, version, workspace, mode, report['status'],
          report.get('scf_seconds'), report.get('gradient_seconds'), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:7])
