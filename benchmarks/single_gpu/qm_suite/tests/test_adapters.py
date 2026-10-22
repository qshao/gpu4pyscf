"""Native CPU adapter acceptance; collection does not require CUDA."""
import importlib
import json
import sys
from contextlib import contextmanager

import numpy as np
import pytest


def adapter():
    return importlib.import_module('benchmarks.single_gpu.qm_suite.adapters')


@pytest.fixture
def water(tmp_path):
    geometry = tmp_path / 'water.xyz'
    geometry.write_text('3\nwater\nO 0 0 0\nH 0 -.757 .587\nH 0 .757 .587\n')
    return dict(id='tiny-water', family='core', group='test', task='energy_gradient',
                system='water', geometry=str(geometry), charge=0, spin=0,
                basis='sto-3g', auxbasis='weigend', xc='HF', df=True,
                settings={'conv_tol': 1e-10, 'conv_tol_grad': 1e-7,
                          'max_cycle': 100, 'grid_level': 0}, task_options={})


def test_collection_does_not_import_cupy():
    before = set(sys.modules)
    adapter()
    assert 'cupy' not in set(sys.modules) - before


def test_invalid_backend_and_task_rejected_before_native_import(water):
    module = adapter()
    with pytest.raises(ValueError, match='backend'):
        module.execute(water, 'typo')
    with pytest.raises(module.UnsupportedTask):
        module.execute(dict(water, task='made-up'), 'cpu')


def test_cpu_energy_gradient_is_native_serializable_and_reproducible(water):
    pytest.importorskip('pyscf')
    from pyscf import gto, scf
    import hashlib
    mol = gto.M(atom=water['geometry'], basis=water['basis'], verbose=0)
    initial = scf.RHF(mol).get_init_guess(key='minao')
    path = __import__('pathlib').Path(water['geometry']).with_name('initial.npy')
    np.save(path, initial)
    water['task_options'] = {'initial_density_file': str(path),
                             'initial_density_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = adapter().execute(water, 'cpu', mode='reference')
    again = adapter().execute(water, 'cpu', mode='reference')
    json.dumps(result, allow_nan=False)
    assert result['converged'] is True
    assert result['energy_hartree'] == pytest.approx(-74.963, abs=.002)
    gradient = np.asarray(result['gradient_hartree_per_bohr'])
    assert gradient.shape == (3, 3)
    assert np.max(np.abs(gradient.sum(axis=0))) < 1e-7
    assert result['initial_density_sha256'] == again['initial_density_sha256']
    assert result['initial_density_sha256'] == hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()
    assert result['metadata']['guess'] == 'frozen_common_cpu_atomic_minao'
    assert result['metadata']['backend'] == 'cpu'
    assert result['metadata']['placement']['scf'] == 'cpu'
    assert result['iterations'] > 0
    assert {'setup', 'scf', 'gradient', 'total'} <= result['phases_seconds'].keys()


def test_tracker_observes_phases_without_iteration_hooks(water):
    pytest.importorskip('pyscf')
    class Tracker:
        def __init__(self):
            self.events = []
        @contextmanager
        def phase(self, name):
            self.events.append(name)
            yield
    tracker = Tracker()
    result = adapter().execute(water, 'cpu', mode='memory', tracker=tracker)
    assert {'setup', 'scf', 'gradient'} <= set(tracker.events)
    assert result['metadata']['mode'] == 'memory'
    with pytest.raises(ValueError, match='timing'):
        adapter().execute(water, 'cpu', mode='timing', tracker=tracker)


def test_cpu_hessian_tensor_order_and_response_settings(water):
    pytest.importorskip('pyscf')
    result = adapter().execute(dict(water, task='hessian'), 'cpu', 'reference')
    hessian = np.asarray(result['hessian_hartree_per_bohr2'])
    assert hessian.shape == (3, 3, 3, 3)
    assert np.allclose(hessian, hessian.transpose(2, 3, 0, 1), atol=1e-7)
    assert result['metadata']['response']['auxbasis_response'] == 2
    assert result['metadata']['hessian_order'] == 'atom,xyz,atom,xyz'


def test_cpu_tda_reports_separate_root_convergence(water):
    pytest.importorskip('pyscf')
    result = adapter().execute(dict(water, task='tda', task_options={'nstates': 3}), 'cpu', 'reference')
    assert len(result['excitation_energies_hartree']) == 3
    assert len(result['oscillator_strengths']) == 3
    assert len(result['task_convergence']) == 3
    assert all(result['task_convergence'])


def test_cpu_mp2_uses_separate_correlation_fit_and_fp64(water):
    pytest.importorskip('pyscf')
    case = dict(water, task='dfmp2', task_options={'auxbasis': 'def2-svp-ri'})
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['correlation_energy_hartree'] < 0
    assert result['energy_hartree'] == pytest.approx(result['hf_reference_energy_hartree'] + result['correlation_energy_hartree'], abs=1e-12)
    assert result['metadata']['correlation']['auxbasis'] == 'def2-svp-ri'
    assert result['metadata']['correlation']['frozen'] == 0
    assert result['metadata']['correlation']['with_t2'] is False
    assert result['metadata']['correlation']['precision'] == 'float64'


def test_unqualified_extensions_raise_explicit_unsupported(water):
    module = adapter()
    for task in ['unknown_task']:
        with pytest.raises(module.UnsupportedTask):
            module.execute(dict(water, task=task), 'cpu')


def test_cpu_dft_pcm_full_gradient_response(water):
    pytest.importorskip('pyscf')
    case = dict(water, xc='b3lyp', task_options={'solvent': {
        'model': 'PCM', 'method': 'IEF-PCM', 'eps': 78.3553, 'lebedev_order': 5}})
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['converged']
    assert result['metadata']['response']['grid_response'] is True
    assert result['energy_hartree'] < -75
    assert np.isfinite(result['gradient_hartree_per_bohr']).all()


def test_requested_mp2_frozen_space_cannot_be_silently_changed(water):
    with pytest.raises(adapter().UnsupportedTask, match='frozen'):
        adapter().execute(dict(water, task='dfmp2', task_options={'frozen': 1}), 'cpu')


def test_cpu_periodic_gamma_energy_tiny_cell(water):
    pytest.importorskip('pyscf')
    case = dict(water, task='pbc_energy', xc='PBE', df=False, cell={
        'a': [[4, 0, 0], [0, 4, 0], [0, 0, 4]], 'atom': 'He 0 0 0',
        'basis': 'gth-szv', 'pseudo': 'gth-pbe', 'kmesh': [1, 1, 1],
        'mesh': [9, 9, 9], 'precision': 1e-6})
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['converged']
    assert result['energy_hartree'] < -2
    assert result['metadata']['periodic']['kmesh'] == [1, 1, 1]
    assert result['metadata']['placement']['pbc_energy'] == 'cpu'


def test_td_retains_phase_independent_ao_transition_density(water):
    pytest.importorskip('pyscf')
    result = adapter().execute(dict(water, task='tda', task_options={'nstates': 3}), 'cpu', 'reference')
    assert result['metadata']['transition_representation'] == 'ao_transition_density'
    assert np.asarray(result['transition_amplitudes']).shape == (3, 49)
    assert np.linalg.norm(result['transition_amplitudes'], axis=1).min() > 0


def test_cpu_open_shell_dfump2(water, tmp_path):
    pytest.importorskip('pyscf')
    xyz = tmp_path / 'oh.xyz'
    xyz.write_text('2\nOH\nO 0 0 0\nH 0 0 .97\n')
    case = dict(water, geometry=str(xyz), spin=1, task='dfump2')
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['converged'] and result['task_converged']
    assert result['correlation_energy_hartree'] < 0
    assert result['energy_hartree'] == pytest.approx(result['hf_reference_energy_hartree'] + result['correlation_energy_hartree'], abs=1e-12)
    assert np.asarray(result['mo_occupations']).shape[0] == 2


def test_cpu_direct_ccsd_reports_correlation_and_solver_convergence(water):
    pytest.importorskip('pyscf')
    result = adapter().execute(dict(water, task='ccsd', df=False), 'cpu', 'reference')
    assert result['correlation_energy_hartree'] < -0.01
    assert result['energy_hartree'] == pytest.approx(result['hf_reference_energy_hartree'] + result['correlation_energy_hartree'], abs=1e-12)
    assert result['energy_hartree'] < result['hf_reference_energy_hartree']
    assert result['task_converged']
    assert result['task_convergence'] is True
    assert result['workflow_seconds'] >= result['phases_seconds']['scf']


def test_optional_cpu_property_dependency_is_classified(water):
    pytest.importorskip('pyscf')
    import importlib.util
    if importlib.util.find_spec('pyscf.prop') is not None:
        pytest.skip('native property package installed')
    for task in ('polarizability', 'shielding'):
        with pytest.raises(adapter().DependencyMissing, match='pyscf.prop'):
            adapter().execute(dict(water, task=task), 'cpu')


def test_optional_optimizer_dependency_is_classified(water):
    pytest.importorskip('pyscf')
    import importlib.util
    if importlib.util.find_spec('geometric') is not None:
        pytest.skip('native optimizer installed')
    with pytest.raises(adapter().DependencyMissing, match='geometric'):
        adapter().execute(dict(water, task='optimization'), 'cpu')


def test_cpu_periodic_kpoint_result_serializes_complex_orbitals(water):
    pytest.importorskip('pyscf')
    case = dict(water, task='pbc_energy', xc='PBE', df=False, cell={
        'a': [[4, 0, 0], [0, 4, 0], [0, 0, 4]], 'atom': 'He 0 0 0',
        'basis': 'gth-szv', 'pseudo': 'gth-pbe', 'kmesh': [2, 1, 1],
        'mesh': [9, 9, 9], 'precision': 1e-6})
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['converged']
    assert result['metadata']['periodic']['kmesh'] == [2, 1, 1]
    json.dumps(result, allow_nan=False)


def test_full_dft_hessian_response_is_explicit(water):
    pytest.importorskip('pyscf')
    result = adapter().execute(dict(water, task='hessian', xc='b3lyp'), 'cpu', 'reference')
    assert result['metadata']['response']['grid_response'] is True
    assert result['metadata']['response']['auxbasis_response'] == 2
    assert result['metadata']['response']['cphf_grid'] == 'same_as_scf_grid'
    assert np.isfinite(result['hessian_hartree_per_bohr2']).all()


def test_gpu_storage_configuration_requires_real_capabilities(water):
    from types import SimpleNamespace
    module = adapter()
    df = SimpleNamespace(use_gpu_memory=None, cderi_storage=None,
                         cderi_buffer_bytes=0, cderi_cache_bytes=0)
    mf = SimpleNamespace(with_df=df)
    module._configure_gpu_df(mf, water)
    assert df.use_gpu_memory is True
    assert df.cderi_cache_bytes == 0
    old = SimpleNamespace(with_df=SimpleNamespace(use_gpu_memory=None))
    case = dict(water, task_options={'storage': 'disk'})
    with pytest.raises(module.UnsupportedTask, match='disk'):
        module._configure_gpu_df(old, case)


def test_owned_disk_tensors_close_on_cleanup():
    from types import SimpleNamespace
    class Store:
        closed = False
        def close(self):
            self.closed = True
    store = Store()
    nested_store = Store()
    nested = SimpleNamespace(_cderi=[nested_store], _rsh_df={})
    df = SimpleNamespace(_cderi=[store], _rsh_df={'omega': nested})
    adapter()._close_df(df)
    assert store.closed and nested_store.closed


def test_gpu_grid_pruning_uses_imported_gpu_implementation(water, monkeypatch):
    from types import SimpleNamespace
    gpu_prune = object()
    module = adapter()
    native_import = module._import
    monkeypatch.setattr(module, '_import', lambda name: SimpleNamespace(nwchem_prune=gpu_prune)
                        if name == 'gpu4pyscf.dft.gen_grid' else native_import(name))
    mf = SimpleNamespace(grids=SimpleNamespace(), cphf_grids=SimpleNamespace())
    module._settings(mf, water, backend='gpu')
    assert mf.grids.prune is gpu_prune
    assert mf.cphf_grids is mf.grids


def test_ecp_is_part_of_constructed_physical_model(water, tmp_path):
    pytest.importorskip('pyscf')
    xyz = tmp_path / 'nah.xyz'
    xyz.write_text('2\nNaH\nNa 0 0 0\nH 0 0 1.9\n')
    case = dict(water, geometry=str(xyz), basis='lanl2dz', df=False,
                task_options={'ecp': 'lanl2dz'})
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['converged']
    assert result['metadata']['state']['nelectron'] == 2


def test_hessian_reports_signed_harmonic_frequencies(water):
    pytest.importorskip('pyscf')
    result = adapter().execute(dict(water, task='hessian'), 'cpu', 'reference')
    assert len(result['frequencies_cm1']) == 3
    assert result['metadata']['frequency_convention'] == 'imaginary_modes_negative_cm-1'
    assert 'harmonic_analysis' in result['phases_seconds']


def test_frozen_density_artifact_integrity_and_state_shape(water, tmp_path):
    import hashlib
    path = tmp_path / 'initial.npy'
    np.save(path, np.eye(7, dtype=np.float64))
    case = dict(water, task_options={'initial_density_file': str(path),
                                    'initial_density_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    density = adapter()._frozen_density(case, (7, 7))
    assert hashlib.sha256(density.tobytes()).hexdigest() == hashlib.sha256(np.eye(7).tobytes()).hexdigest()
    with pytest.raises(ValueError, match='shape'):
        adapter()._frozen_density(case, (2, 7, 7))
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='hash'):
        adapter()._frozen_density(case, (7, 7))


@pytest.mark.parametrize('value', [np.eye(7, dtype=np.float32), np.full((7, 7), np.nan)])
def test_frozen_density_rejects_dtype_and_nonfinite(water, tmp_path, value):
    import hashlib
    path = tmp_path / 'invalid.npy'
    np.save(path, value)
    case = dict(water, task_options={'initial_density_file': str(path),
                                    'initial_density_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match='float64|finite'):
        adapter()._frozen_density(case, (7, 7))


@pytest.mark.parametrize('task', ['dfmp2', 'ccsd'])
def test_correlated_total_matches_independent_pyscf_total(water, task):
    pytest.importorskip('pyscf')
    from pyscf import gto, scf, df, cc
    from pyscf.mp import dfmp2
    case = dict(water, task=task, df=(task == 'dfmp2'))
    mol = gto.M(atom=case['geometry'], basis=case['basis'], verbose=0)
    native_scf = scf.RHF(mol)
    if case['df']:
        native_scf = native_scf.density_fit(auxbasis=case['auxbasis'])
    native_scf.conv_tol = 1e-12
    native_scf.kernel()
    if task == 'dfmp2':
        native = dfmp2.DFMP2(native_scf, frozen=0)
        native.with_df = df.DF(mol, auxbasis='def2-svp-ri')
        native.kernel(with_t2=False)
    else:
        native = cc.CCSD(native_scf, frozen=0)
        native.conv_tol = 1e-11
        native.kernel()
    result = adapter().execute(case, 'cpu', 'reference')
    assert result['energy_hartree'] == pytest.approx(native.e_tot, abs=1e-8)
