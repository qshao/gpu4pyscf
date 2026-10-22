"""Opt-in Taxol native-resident DF J/K block-size experiment, without SCF."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
from unittest.mock import patch

import numpy as np
import pytest

from original_comparison import (BASE, ROOT, configure_cpu, case_settings,
                                 cpu_reference, digest, python_sources, save)


@pytest.mark.special
@pytest.mark.benchmark
def test_resident_exchange_sweep():
    import cupy as cp
    import gpu4pyscf
    from pyscf import lib
    from gpu4pyscf.df import df
    from molecular import _metadata, _file_identity
    from seeded_reference import compare_execution_cases

    output = BASE/'resident_workspace_results/exchange_sweep.json'
    if output.exists():
        raise FileExistsError('Archive previous sweep explicitly: '+str(output))
    assert Path(gpu4pyscf.__file__).resolve() == ROOT/'gpu4pyscf/__init__.py'
    assert cp.cuda.runtime.getDeviceCount() == 1
    config = case_settings('large')
    reference_path, cpu = cpu_reference('large')
    profile_path = BASE/'optimization_results/corrected_profile.json'
    orbital_path = profile_path.with_suffix('.npz')
    profile = json.loads(profile_path.read_text())
    assert profile['status'] == 'passed' and profile['case'] == 'taxol_svp_b3lyp'
    assert profile['settings'] == config == profile['result']['settings']
    assert profile['result']['converged'] and profile['result']['method'] == 'B3LYP'
    assert profile['cpu_reference_sha256'] == digest(reference_path)
    snapshot_comparison = compare_execution_cases(cpu['result'],profile['result'])
    assert snapshot_comparison['passed'], snapshot_comparison
    assert profile['geometry_sha256'] == digest(config['atom'])
    assert Path(profile['orbitals']['path']).resolve() == orbital_path.resolve()
    assert profile['orbitals']['sha256'] == digest(orbital_path)
    assert profile['orbitals']['size_bytes'] == orbital_path.stat().st_size
    sources = python_sources(ROOT/'gpu4pyscf')
    helpers = {name:digest(BASE/name) for name in [
        'test_resident_exchange_sweep.py', 'original_comparison.py',
        'comprehensive.py', 'molecular.py', 'gradient_snapshot.py', 'seeded_reference.py']}
    native_sources = {str(p.relative_to(ROOT)):digest(p)
        for p in sorted((ROOT/'gpu4pyscf/lib').rglob('*'))
        if p.is_file() and 'deps' not in p.parts and (
            p.suffix in ('.cu', '.cuh', '.c', '.h', '.cpp', '.cmake')
            or p.name == 'CMakeLists.txt')}
    native_files = sorted(set((ROOT/'gpu4pyscf/lib').glob('*.so*'))
                          | set((ROOT/'gpu4pyscf/lib/deps/lib').glob('*.so*')))
    native = {str(p.relative_to(ROOT)):_file_identity(p) for p in native_files
              if p.is_file()}
    report = dict(_metadata('gpu'), status='running', case='taxol_svp_b3lyp',
        method='B3LYP', settings=config, algorithm_sources=sources,
        native_source_hashes=native_sources, native_file_identities=native,
        helpers=helpers, geometry_sha256=digest(config['atom']),
        snapshot_cpu_comparison=snapshot_comparison,
        inputs={name:_file_identity(path) for name,path in [
            ('geometry',Path(config['atom'])), ('cpu_reference',reference_path),
            ('corrected_profile',profile_path), ('orbitals',orbital_path)]},
        tolerance=dict(absolute=1e-9, relative=0., require_finite=True),
        control_block_size=256, requested_block_sizes=[128,256,512,1024],
        samples=[], variants={}, bindings_restored=False,
        scope='Fixed converged Taxol density; native resident CDERI built once. '
              'Synchronized full DF.get_jk includes factorization, unpack, '
              'occupied transform, Gram, Coulomb contractions and output assembly. '
              'Setup/build, patch installation, host copies and verification are '
              'excluded from J/K samples. One warmup per size then three pairs '
              'against 256 per candidate, alternating pair order. Ratios concern '
              'J/K only; no SCF, forces or workflow speedup is measured.')
    output.parent.mkdir(exist_ok=True)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    original_get_blksize = df.DF.get_blksize  # Unbound: never retain a DF instance.
    original_loop = df.DF.loop
    mf = density = baseline = None
    try:
        with lib.with_omp_threads(4):
            mf = configure_cpu('large').to_gpu()
            assert mf.xc.upper() == 'B3LYP'
            assert mf.conv_tol == 1e-10 and mf.conv_tol_grad == 1e-7
            mf.with_df.use_gpu_memory = True
            mf.with_df.cderi_storage = None
            # Same finite, real FP64/shape checks used by gradient_snapshot.restore.
            with np.load(orbital_path,allow_pickle=False) as archive:
                coeff,occ,energy = [archive[key] for key in
                                   ['mo_coeff','mo_occ','mo_energy']]
                assert coeff.ndim == 2 and coeff.shape[0] == mf.mol.nao
                assert occ.shape == (coeff.shape[1],) and energy.shape == occ.shape
                assert all(a.dtype == np.float64 and np.isfinite(a).all()
                           for a in [coeff,occ,energy])
                assert abs(occ.sum()-mf.mol.nelectron) < 1e-10
                assert np.isin(occ,[0.,2.]).all()
                snapshot_energy = float(archive['energy_hartree'])
                assert np.isfinite(snapshot_energy)
                assert abs(snapshot_energy-profile['result']['energy_hartree']) < 1e-10
                mf.mo_coeff,mf.mo_occ,mf.mo_energy = map(cp.asarray,[coeff,occ,energy])
            density = mf.make_rdm1()
            assert hasattr(density,'mo_coeff') and hasattr(density,'mo_occ')
            assert bool(cp.isfinite(density).all())
            density_sha = hashlib.sha256(cp.asnumpy(density).tobytes()).hexdigest()
            cp.cuda.Device().synchronize()
            started = time.perf_counter()
            mf.with_df.build()
            cp.cuda.Device().synchronize()
            report['df_build_seconds'] = time.perf_counter()-started
            assert len(mf.with_df._cderi) == 1
            resident = mf.with_df._cderi[0]
            assert isinstance(resident,cp.ndarray) and resident.dtype == cp.float64
            naux,nao = int(resident.shape[0]),int(mf.mol.nao)
            nocc = int(np.count_nonzero(occ))
            assert naux > 0 and mf.with_df.nao == nao
            report['unpatched_control_block_size'] = mf.with_df.get_blksize(mem_fraction=0.4)
            assert report['unpatched_control_block_size'] == min(256,naux)
            report.update(nao=nao,nocc=nocc,naux=naux,natm=mf.mol.natm,
                nelectron=mf.mol.nelectron,charge=mf.mol.charge,spin=mf.mol.spin,
                geometry_bohr=mf.mol.atom_coords().tolist(),
                atom_symbols=[mf.mol.atom_symbol(i) for i in range(mf.mol.natm)],
                density_sha256=density_sha,snapshot_energy_hartree=snapshot_energy,
                resident_cderi=dict(shape=list(resident.shape),nbytes=int(resident.nbytes)),
                workspace_scope='FP64 dense AO unpack plus one occupied transform '
                    'buffer only; excludes persistent CDERI, outputs, factors, '
                    'allocator reservation and native-library workspaces.')
            del resident
            for size in [128,256,512,1024]:
                block = min(size,naux)
                lengths = [min(block,naux-p) for p in range(0,naux,block)]
                report['variants'][str(size)] = dict(effective_block_size=block,
                    block_lengths=lengths,blocks_per_call=len(lengths),seconds=[],
                    paired_control_seconds=[],normalized_seconds=[],
                    workspace_bytes=dict(dense_ao_unpack=8*block*nao*nao,
                        occupied_transform=8*block*nao*nocc,
                        sum=8*block*nao*(nao+nocc)))

            def run(size,phase,pair=None):
                nonlocal baseline
                observed = []
                blocks = []
                def selected(dfobj,extra=0,nao=None,mem_fraction=0.3):
                    block = min(size,dfobj._cderi[cp.cuda.Device().id].shape[0])
                    observed.append(int(block))
                    return block
                def observed_loop(dfobj,*args,**kwargs):
                    iterator = original_loop(dfobj,*args,**kwargs)
                    try:
                        for dense,sparse in iterator:
                            blocks.append(int(sparse.shape[0]))
                            yield dense,sparse
                    finally:
                        iterator.close()
                with patch.object(df.DF,'get_blksize',selected), \
                     patch.object(df.DF,'loop',observed_loop):
                    cp.cuda.Device().synchronize()
                    start = time.perf_counter()
                    matrices = mf.with_df.get_jk(density,hermi=1,with_j=True,with_k=True)
                    cp.cuda.Device().synchronize()
                    seconds = time.perf_counter()-start
                assert df.DF.get_blksize is original_get_blksize
                assert df.DF.loop is original_loop
                assert observed and all(b == min(size,naux) for b in observed)
                assert blocks == report['variants'][str(size)]['block_lengths']
                actual = tuple(cp.asnumpy(a) for a in matrices)
                del matrices
                if baseline is None:
                    assert size == 256 and phase == 'warmup'
                    baseline = actual
                checks = {}
                for name,value,control in zip(['J','K'],actual,baseline):
                    assert value.shape == control.shape == (nao,nao)
                    finite = bool(np.isfinite(value).all() and np.isfinite(control).all())
                    error = float(np.max(np.abs(value-control))) if finite else None
                    checks[name] = dict(finite=finite,max_abs_error=error,
                                        passed=finite and error <= 1e-9)
                row = dict(block_size=size,phase=phase,pair=pair,seconds=seconds,
                    observed_get_blksize=observed,observed_block_lengths=blocks,comparison=checks,
                    passed=all(c['passed'] for c in checks.values()))
                report['samples'].append(row)
                save(output,report)
                print('Taxol exchange sweep',size,phase,pair,seconds,checks,flush=True)
                return seconds

            for size in [256,128,512,1024]:
                run(size,'warmup')
            for candidate in [128,512,1024]:
                variant = report['variants'][str(candidate)]
                for pair in range(3):
                    order = [256,candidate] if pair%2 == 0 else [candidate,256]
                    times = {size:run(size,'paired',dict(candidate=candidate,index=pair,
                                                        order=order)) for size in order}
                    variant['seconds'].append(times[candidate])
                    variant['paired_control_seconds'].append(times[256])
                    variant['normalized_seconds'].append(times[candidate]/times[256])
            report['variants']['256']['seconds'] = [row['seconds']
                for row in report['samples'] if row['block_size'] == 256
                and row['phase'] == 'paired']
            report['variants']['256']['normalized_seconds'] = [1.]*9
            for size,variant in report['variants'].items():
                variant['median_seconds'] = statistics.median(variant['seconds'])
                variant['median_normalized_seconds'] = statistics.median(
                    variant['normalized_seconds'])
                rows = [row for row in report['samples'] if row['block_size'] == int(size)]
                variant['passed'] = all(row['passed'] for row in rows)
                variant['max_abs_error'] = {name:max(
                    (row['comparison'][name]['max_abs_error'] for row in rows
                     if row['comparison'][name]['max_abs_error'] is not None),default=None)
                    for name in ['J','K']}
            assert density_sha == hashlib.sha256(cp.asnumpy(density).tobytes()).hexdigest()
            assert python_sources(ROOT/'gpu4pyscf') == sources
            assert all(digest(BASE/name) == sha for name,sha in helpers.items())
            assert all(digest(ROOT/name) == sha for name,sha in native_sources.items())
            assert all(_file_identity(ROOT/name) == value for name,value in native.items())
            assert all(_file_identity(value['path']) == value for value in report['inputs'].values())
            assert all(row['passed'] for row in report['samples']), 'J/K comparison failed'
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report['bindings_restored'] = (df.DF.get_blksize is original_get_blksize
                                       and df.DF.loop is original_loop)
        if not report['bindings_restored']:
            report.update(status='failed',error='DF.get_blksize/loop were not restored')
        if mf is not None:
            mf.with_df.reset()
        mf = density = baseline = None
        gc.collect()
        save(output,report)
    assert report['bindings_restored']
