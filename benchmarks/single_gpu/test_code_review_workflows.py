"""Fresh scientific acceptance after the comprehensive code-review repairs."""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pytest

from original_comparison import (BASE,ROOT,case_settings,configure_cpu,
                                 cpu_reference,digest,python_sources,save)
from seeded_reference import compare_execution_cases

OUT=BASE/'code_review_results'


@pytest.mark.parametrize('xc,spin',[('wb97x',0),('wb97x',1),('tpss',1)])
def test_review_open_shell_and_range_separated_gradients(xc,spin):
    import cupy as cp
    from pyscf import dft,gto,lib
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
              basis='def2-svp',charge=spin,spin=spin,verbose=0)
    with lib.with_omp_threads(4):
        cpu=(dft.UKS if spin else dft.RKS)(mol,xc=xc).density_fit(
            auxbasis='def2-universal-jkfit')
        cpu.grids.level=1;cpu.grids.prune=None;cpu.small_rho_cutoff=0.
        cpu.conv_tol=1e-10;cpu.conv_tol_grad=1e-7;cpu.max_cycle=100
        density=cpu.get_init_guess()
        gpu=cpu.to_gpu()
        cpu.kernel(dm0=density);gpu.kernel(dm0=density)
        assert cpu.converged and gpu.converged
        assert abs(cpu.e_tot-gpu.e_tot)<=1e-8
        left=cpu.nuc_grad_method();right=gpu.nuc_grad_method()
        left.grid_response=right.grid_response=True
        left.auxbasis_response=right.auxbasis_response=True
        reference=left.kernel();actual=cp.asnumpy(right.kernel())
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual,reference,atol=1e-6,rtol=0)
        gpu.with_df.reset()


@pytest.mark.special
@pytest.mark.parametrize('size',['medium','large'])
def test_review_fresh_workflow(size):
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf import __config__
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    from molecular import _metadata
    output=OUT/f'{size}_workflow.json'
    if output.exists():
        raise FileExistsError(str(output))
    reference_path,cpu=cpu_reference(size)
    config=case_settings(size)
    previous_path=BASE/'resident_workspace_results'/(
        f'{size}_candidate_'+('memory' if size=='large' else 'validation')+'_0.json')
    previous=json.loads(previous_path.read_text())
    assert previous['status']=='passed' and previous['settings']==config
    sources=python_sources(ROOT/'gpu4pyscf')
    helpers={name:digest(BASE/name) for name in ['test_code_review_workflows.py',
        'original_comparison.py','molecular.py','comprehensive.py','seeded_reference.py']}
    report=dict(_metadata('gpu'),status='running',size=size,settings=config,
        algorithm_sources=sources,helpers=helpers,
        baseline_manifest_sha256=digest(OUT/'before_manifest.json'),
        cpu_reference_sha256=digest(reference_path),
        previous_input_report_sha256=digest(previous_path),
        geometry_sha256=digest(config['atom']),
        scope='Fresh atomic guess, resident DF build, converged SCF and full '
              'grid/auxiliary-response forces. One accuracy observation only; '
              'elapsed time does not establish a workflow speed ratio.')
    save(output,report)
    mf=None
    try:
        assert cp.get_default_memory_pool().used_bytes()==0
        assert int(getattr(__config__,'df_cderi_build_workspace_bytes',4*1024**3))==4*1024**3
        iterations=[]
        with lib.with_omp_threads(4):
            cp.cuda.Device().synchronize();start=time.perf_counter()
            cpu_mf=configure_cpu(size)
            density=cpu_mf.get_init_guess()
            initial=hashlib.sha256(np.ascontiguousarray(density).tobytes()).hexdigest()
            assert initial==previous['initial_density_sha256']
            mf=cpu_mf.to_gpu();mf.grids.prune=nwchem_prune
            assert mf.conv_tol==1e-10 and mf.conv_tol_grad==1e-7
            assert mf.max_cycle==100 and mf.small_rho_cutoff==0.
            assert mf.grids.level==3 and mf.grids.prune is nwchem_prune
            mf.with_df.use_gpu_memory=True;mf.with_df.cderi_storage=None
            setup=time.perf_counter()-start
            def callback(env):
                iterations.append(dict(cycle=int(env['cycle']),energy_hartree=float(env['e_tot'])))
                if len(iterations)%5==0:
                    print('Code review workflow',size,'cycle',len(iterations),flush=True)
            mf.callback=callback
            begin=time.perf_counter();energy=mf.kernel(dm0=density)
            cp.cuda.Device().synchronize();scf=time.perf_counter()-begin
            assert mf.converged
            assert all(isinstance(a,cp.ndarray) for a in mf.with_df._cderi)
            gradient=mf.nuc_grad_method()
            gradient.grid_response=gradient.auxbasis_response=True
            begin=time.perf_counter();value=gradient.kernel()
            cp.cuda.Device().synchronize();forces=time.perf_counter()-begin
            elapsed=time.perf_counter()-start
            actual=cp.asnumpy(value)
        result=dict(method='B3LYP',settings=config,converged=bool(mf.converged),
            energy_hartree=float(energy),gradient_hartree_per_bohr=actual.tolist())
        comparison=compare_execution_cases(cpu['result'],result)
        report.update(result=result,comparison=comparison,iterations=iterations,
            initial_density_sha256=initial,setup_seconds=setup,scf_seconds=scf,
            gradient_seconds=forces,workflow_seconds=elapsed,
            effective_build_workspace_bytes=4*1024**3)
        assert python_sources(ROOT/'gpu4pyscf')==sources
        assert all(digest(BASE/name)==sha for name,sha in helpers.items())
        assert comparison['passed'],comparison
        report['status']='passed'
    except BaseException:
        report.update(status='failed',traceback=traceback.format_exc())
        raise
    finally:
        if mf is not None:
            mf.with_df.reset()
        save(output,report)
    print('Review acceptance',size,elapsed,comparison,flush=True)
