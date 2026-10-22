"""SCF cycle-count experiment: same method, grid, thresholds (conv_tol 1e-10, grad 1e-7), different route to convergence.

Usage: python lcc_scf_cycles.py GPU SIZE STRATEGY
  base      MINAO guess, default DIIS (what every campaign uses)
  atom      superposition of converged atomic densities as the guess
  huckel    extended-Huckel guess
  vsap      superposition of atomic potentials guess
  diis12    MINAO, DIIS space 12 (default 8)
  twostage  converge on grid level 1 to 1e-4 (cheap), then the full grid level 3 to the full thresholds from that density
  newton    second-order (SOSCF) from the MINAO guess
The only acceptance test is the final SCF energy against the independent CPU reference (<= 1e-8 Eh); cycle counts count
the full-quality (level 3, tight) cycles separately from the cheap pre-cycles.
"""
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/scratch/qsh226/g4p_lcc/current')
OUT = HERE / 'lcc_scf_cycle_results'


def main(gpu, size, strategy):
    import numpy as np
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    from original_comparison import save, case_settings, configure_cpu, digest, CASES, BASE
    from lcc_gpu_run import cpu_reference
    OUT.mkdir(exist_ok=True)
    output = OUT / f'{gpu}_{size}_{strategy}.json'
    if output.exists():
        raise FileExistsError(output)
    config = case_settings(size)
    reference = cpu_reference(size, config, digest, CASES, BASE)
    e_ref = reference['result']['energy_hartree']
    report = dict(status='running', gpu=gpu, size=size, strategy=strategy)
    save(output, report)
    try:
        cycles = []
        cp.matmul(cp.ones((64, 64)), cp.ones((64, 64))); cp.cuda.Device().synchronize()
        with lib.with_omp_threads(4):
            start = time.perf_counter()
            cpu = configure_cpu(size)
            key = {'base': 'minao', 'diis12': 'minao', 'twostage': 'minao', 'newton': 'minao'}.get(strategy, strategy)
            g0 = time.perf_counter()
            dm = cpu.get_init_guess(key=key)
            guess_seconds = time.perf_counter() - g0
            mf = cpu.to_gpu()
            mf.grids.prune = nwchem_prune
            mf.with_df.use_gpu_memory = True
            mf.callback = lambda env: cycles.append(dict(cycle=int(env['cycle']), e=float(env['e_tot']),
                                                         t=time.perf_counter() - start))
            if strategy == 'diis12':
                mf.diis_space = 12
            pre = None
            if strategy == 'twostage':
                mf.grids.level = 1
                mf.conv_tol, mf.conv_tol_grad = 1e-4, 1e-2
                s0 = time.perf_counter(); mf.kernel(dm0=dm); cp.cuda.Device().synchronize()
                pre = dict(seconds=time.perf_counter() - s0, cycles=len(cycles))
                dm = mf.make_rdm1()
                mf.grids.level = 3; mf.grids.reset() if hasattr(mf.grids, 'reset') else None
                mf.grids.prune = nwchem_prune
                mf.conv_tol, mf.conv_tol_grad = config['conv_tol'], config['conv_tol_grad']
            if strategy == 'newton':
                mf = mf.newton()
                mf.conv_tol, mf.conv_tol_grad = config['conv_tol'], config['conv_tol_grad']
            main_start = time.perf_counter()
            n_before = len(cycles)
            energy = mf.kernel(dm0=dm)
            cp.cuda.Device().synchronize()
            now = time.perf_counter()
        report.update(status='ok', converged=bool(mf.converged), energy_hartree=float(energy),
                      energy_error_hartree=abs(float(energy) - e_ref),
                      accepted=bool(mf.converged) and abs(float(energy) - e_ref) <= 1e-8,
                      guess_seconds=guess_seconds, prestage=pre,
                      full_quality_cycles=len(cycles) - n_before, total_cycles=len(cycles),
                      full_quality_seconds=now - main_start, scf_total_seconds=now - start,
                      iterations=cycles)
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    save(output, report)
    print('SCF', gpu, size, strategy, report['status'], report.get('full_quality_cycles'), report.get('scf_total_seconds'),
          report.get('energy_error_hartree'), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
