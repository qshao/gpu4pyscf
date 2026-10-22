"""Fresh-process resident DF workspace qualification; one selected node per run."""
import ast
import copy
from contextlib import ExitStack, nullcontext
import gc
import hashlib
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import pytest

from original_comparison import (BASE, ROOT, OUT as FROZEN, configure_cpu,
    case_settings, cpu_reference, digest, save, python_sources)

OUT = BASE/'resident_workspace_results'


def control_builder(module):
    """Load only the hash-pinned pre-change build function into current globals."""
    import json
    frozen = json.loads((FROZEN/'manifest.json').read_text())['packages']['current']
    path = Path(frozen['path'])/'df/df.py'
    assert digest(path) == frozen['sources']['df/df.py']
    tree = ast.parse(path.read_text())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_cholesky_eri']
    namespace = dict(module.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_cholesky_eri']


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('version,size,mode,repeat',
    [('control','large','profile',0), ('candidate','large','memory',0)]
    +[(v,'large','time',i) for i in range(3) for v in ['control','candidate']]
    +[('candidate',s,'validation',0) for s in ['small','medium']])
def test_resident_workspace_run(tmp_path, version, size, mode, repeat):
    import cupy as cp
    from pyscf import lib
    from gpu4pyscf import __config__
    from gpu4pyscf.df import df, df_jk
    from gpu4pyscf.dft.gen_grid import nwchem_prune
    from molecular import _metadata
    from phase_memory import phase_memory
    from seeded_reference import compare_execution_cases
    from test_stage_profile import stage_profile
    from test_fine_stage_profile import fine_profile

    OUT.mkdir(exist_ok=True)
    output = OUT/f'{size}_{version}_{mode}_{repeat}.json'
    if output.exists():
        raise FileExistsError(str(output))
    config = case_settings(size)
    reference_path, cpu = cpu_reference(size)
    sources = python_sources(ROOT/'gpu4pyscf')
    helpers = {name:digest(BASE/name) for name in [
        'phase_memory.py','test_resident_workspace_run.py','test_stage_profile.py',
        'test_fine_stage_profile.py','original_comparison.py','seeded_reference.py']}
    report = dict(_metadata('gpu'), status='running', version=version, size=size,
        mode=mode, repeat=repeat, settings=config, algorithm_sources=sources,
        helpers=helpers, cpu_reference_sha256=digest(reference_path),
        effective_build_workspace_bytes=int(getattr(
            __config__, 'df_cderi_build_workspace_bytes', 4*1024**3)),
        geometry_sha256=digest(config['atom']),
        control_source_sha256=digest(FROZEN/'current/gpu4pyscf/df/df.py'),
        scope='Fresh atomic guess/setup, fresh DF construction, SCF and full '
              'grid/auxiliary-response gradient. Imports/provenance/final cleanup '
              'excluded. Profile and memory timings are diagnostic only. Named '
              'allocation and synchronized time stages are inclusive; do not sum.')
    save(output, report)
    chosen = control_builder(df) if version == 'control' else df._cholesky_eri
    initial_pool = cp.get_default_memory_pool().used_bytes()
    assert initial_pool == 0
    iterations = []
    coarse = {}; fine = {}; helper_times = {}
    mf = None
    try:
        assert report['effective_build_workspace_bytes'] == 4*1024**3
        with lib.with_omp_threads(4), ExitStack() as stack:
            stack.enter_context(patch.object(df, '_cholesky_eri', chosen))
            tracker = stack.enter_context(phase_memory()) if mode in ('profile','memory') else None
            phase = tracker.phase if tracker else lambda name:nullcontext()
            if mode == 'profile':
                coarse = stack.enter_context(stage_profile())
                fine = stack.enter_context(fine_profile())
                for name in ['_exchange_transform','_symmetric_gram','_coulomb_contract']:
                    original = getattr(df_jk,name)
                    def observed(*args, _fn=original, _name=name, **kwargs):
                        cp.cuda.Device().synchronize(); start=time.perf_counter()
                        try:
                            return _fn(*args,**kwargs)
                        finally:
                            cp.cuda.Device().synchronize()
                            row=helper_times.setdefault(_name,dict(seconds=0.,calls=0))
                            row['seconds']+=time.perf_counter()-start;row['calls']+=1
                    stack.enter_context(patch.object(df_jk,name,observed))
            if tracker:
                original_build = df.DF.build
                def build(*args,**kwargs):
                    with phase('df_build'):
                        return original_build(*args,**kwargs)
                stack.enter_context(patch.object(df.DF,'build',build))
            cp.cuda.Device().synchronize()
            start=time.perf_counter()
            with phase('setup'):
                cpu_mf=configure_cpu(size)
                density=cpu_mf.get_init_guess()
                density_sha=hashlib.sha256(np.ascontiguousarray(density).tobytes()).hexdigest()
                mf=cpu_mf.to_gpu()
                mf.grids.prune=nwchem_prune
                assert mf.grids.prune is nwchem_prune and mf.grids.level==3
                assert mf.conv_tol==1e-10 and mf.conv_tol_grad==1e-7
                assert mf.max_cycle==100 and mf.small_rho_cutoff==0.
                mf.with_df.use_gpu_memory=True
                mf.with_df.cderi_storage=None
            setup=time.perf_counter()-start
            def callback(environment):
                cp.cuda.Device().synchronize()
                iterations.append(dict(cycle=int(environment['cycle']),
                                       energy_hartree=float(environment['e_tot'])))
                if len(iterations)%5==0:
                    print('Workspace progress',version,mode,repeat,len(iterations),flush=True)
            mf.callback=callback
            scf_start=time.perf_counter()
            with phase('scf'):
                energy=mf.kernel(dm0=density)
                cp.cuda.Device().synchronize()
            scf_seconds=time.perf_counter()-scf_start
            assert mf.converged
            integral_storage=[dict(shape=list(a.shape),nbytes=a.nbytes,
                                   resident=isinstance(a,cp.ndarray)) for a in mf.with_df._cderi]
            assert all(a['resident'] for a in integral_storage)
            gradient=mf.nuc_grad_method()
            gradient.grid_response=gradient.auxbasis_response=True
            grad_start=time.perf_counter()
            with phase('gradient'):
                value=gradient.kernel()
                cp.cuda.Device().synchronize()
            gradient_seconds=time.perf_counter()-grad_start
            seconds=time.perf_counter()-start
            actual=np.asarray(value.get() if hasattr(value,'get') else value)
            result=dict(method='B3LYP',settings=config,converged=bool(mf.converged),
                energy_hartree=float(energy),gradient_hartree_per_bohr=actual.tolist())
        from pathlib import Path as _Path
        status=_Path('/proc/self/status').read_text().splitlines()
        rss=next(int(s.split()[1])*1024 for s in status if s.startswith('VmHWM:'))
        comparison=compare_execution_cases(cpu['result'],result)
        assert python_sources(ROOT/'gpu4pyscf')==sources
        assert all(digest(BASE/name)==sha for name,sha in helpers.items())
        report.update(result=result,comparison=comparison,nao=mf.mol.nao,natm=mf.mol.natm,
            initial_density_sha256=density_sha,iterations=iterations,integrals=integral_storage,
            setup_seconds=setup,scf_seconds=scf_seconds,gradient_seconds=gradient_seconds,
            workflow_seconds=seconds,phase_memory=copy.deepcopy(tracker.data) if tracker else {},
            initial_pool_live_bytes=initial_pool,host_address_space_peak_rss_bytes=rss,
            coarse=coarse,fine=fine,helper_times=helper_times)
        mf.with_df.reset()
        del gradient,mf;mf=None;gc.collect();cp.cuda.Device().synchronize()
        assert not list(tmp_path.glob('gpu4pyscf-cderi-*.bin'))
        assert comparison['passed'],comparison
        if mode=='profile': assert fine['bindings_restored']
        report.update(status='passed',scratch_clean=True)
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    finally:
        if mf is not None: mf.with_df.reset()
    save(output,report)
    print('Resident workspace',version,size,mode,repeat,seconds,comparison,
          report['phase_memory'],helper_times,flush=True)
