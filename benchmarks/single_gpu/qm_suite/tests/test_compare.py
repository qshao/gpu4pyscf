"""CPU-only acceptance and deliberate corruption tests for saved observables."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def compare(case, reference, candidate):
    path = Path(__file__).parents[1] / 'compare.py'
    assert path.exists(), 'scientific comparator has not been implemented'
    spec = importlib.util.spec_from_file_location('qm_compare_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_results(case, reference, candidate)


def sample(task='energy_gradient'):
    case = {'id': 'water', 'task': task, 'basis': 'def2-svp', 'spin': 0}
    result = {'case': copy.deepcopy(case), 'converged': True,
              'task_converged': True, 'energy_hartree': -75.,
              'gradient_hartree_per_bohr': [[0., 0., 0.], [0., 0., 0.]],
              'hessian_hartree_per_bohr2': np.eye(6).reshape(2, 3, 2, 3).tolist(),
              'correlation_energy_hartree': -0.2,
              'excitation_energies_hartree': [0.1, 0.2],
              'oscillator_strengths': [0.01, 0.02],
              'transition_amplitudes': [[1., 0.], [0., 1.]],
              'polarizability_au': np.eye(3).tolist(),
              'shielding_ppm': [np.eye(3).tolist()] * 2,
              'geometry_bohr': [[0., 0., 0.], [1., 0., 0.]]}
    return case, result, copy.deepcopy(result)


@pytest.mark.parametrize('task', ['energy_gradient', 'hessian', 'tda', 'tddft',
                                  'dfmp2', 'dfump2', 'ccsd', 'polarizability',
                                  'shielding', 'optimization', 'pbc_energy'])
def test_exact_reference_qualifies(task):
    case, ref, candidate = sample(task)
    out = compare(case, ref, candidate)
    assert out['passed'], out
    assert out['qualification_status'] == 'qualified'


@pytest.mark.parametrize('task,key,index,delta', [
    ('energy_gradient', 'energy_hartree', None, 1e-6),
    ('energy_gradient', 'gradient_hartree_per_bohr', (1, 2), 1e-4),
    ('hessian', 'hessian_hartree_per_bohr2', (1, 2, 1, 2), 1e-3),
    ('tda', 'excitation_energies_hartree', (1,), 1e-4),
    ('tddft', 'oscillator_strengths', (1,), 1e-3),
    ('dfmp2', 'correlation_energy_hartree', None, 1e-6),
    ('ccsd', 'energy_hartree', None, 1e-6),
    ('polarizability', 'polarizability_au', (1, 2), 1e-3),
    ('shielding', 'shielding_ppm', (1, 2, 1), 1.),
])
def test_corrupted_component_fails(task, key, index, delta):
    case, ref, candidate = sample(task)
    value = np.asarray(candidate[key]); value[index if index else ()] += delta
    candidate[key] = value.tolist()
    candidate['passed'] = True
    out = compare(case, ref, candidate)
    assert not out['passed']
    assert out['errors'][key]['max_abs'] >= delta * 0.99


@pytest.mark.parametrize('value', [[], [[0., 0.]], [[float('nan')] * 3] * 2,
                                   [[float('inf')] * 3] * 2, 'invalid'])
def test_invalid_arrays_fail(value):
    case, ref, candidate = sample()
    candidate['gradient_hartree_per_bohr'] = value
    assert not compare(case, ref, candidate)['passed']


@pytest.mark.parametrize('key', ['converged', 'task_converged', 'energy_hartree', 'case'])
def test_missing_required_data_fails(key):
    case, ref, candidate = sample()
    del candidate[key]
    assert not compare(case, ref, candidate)['passed']


def test_changed_settings_and_density_fail():
    case, ref, candidate = sample()
    candidate['case']['basis'] = 'sto-3g'
    assert not compare(case, ref, candidate)['passed']
    candidate['case'] = copy.deepcopy(case)
    ref['initial_density_sha256'] = 'a'
    candidate['initial_density_sha256'] = 'b'
    assert not compare(case, ref, candidate)['passed']


@pytest.mark.parametrize('section', ['state', 'effective_settings', 'response', 'correlation', 'units',
                                     'transition_representation'])
def test_changed_effective_scientific_metadata_fails(section):
    case, ref, candidate = sample()
    ref['metadata'] = {section: {'value': 1}}
    candidate['metadata'] = {section: {'value': 2}}
    assert not compare(case, ref, candidate)['passed']


def test_complex_observable_cannot_silently_discard_imaginary_part():
    case, ref, candidate = sample()
    candidate['energy_hartree'] = complex(-75., 1.)
    assert not compare(case, ref, candidate)['passed']


def test_td_roots_match_amplitudes_under_reordering_and_phase():
    case, ref, candidate = sample('tda')
    for key in ('excitation_energies_hartree', 'oscillator_strengths'):
        candidate[key].reverse()
    candidate['transition_amplitudes'] = [[0., -1.], [-1., 0.]]
    out = compare(case, ref, candidate)
    assert out['passed'], out
    assert out['diagnostics']['root_mapping'] == [1, 0]


def test_td_energy_only_is_limited_not_qualified():
    case, ref, candidate = sample('tda')
    del candidate['transition_amplitudes']
    out = compare(case, ref, candidate)
    assert not out['passed']
    assert out['qualification_status'] == 'limited'


def test_matching_but_truncated_td_root_sets_cannot_pass():
    case, ref, candidate = sample('tda')
    case['task_options'] = {'nstates': 5}
    ref['case'] = candidate['case'] = copy.deepcopy(case)
    assert not compare(case, ref, candidate)['passed']


def test_changed_td_state_cannot_pass_equal_energy():
    case, ref, candidate = sample('tda')
    candidate['transition_amplitudes'] = [[1., 1.], [1., -1.]]
    assert not compare(case, ref, candidate)['passed']


def test_degenerate_td_subspace_rotation_uses_oscillator_sum():
    case, ref, candidate = sample('tda')
    ref['excitation_energies_hartree'] = candidate['excitation_energies_hartree'] = [0.1, 0.1]
    candidate['transition_amplitudes'] = [[2**-0.5, 2**-0.5], [2**-0.5, -2**-0.5]]
    candidate['oscillator_strengths'] = [0.015, 0.015]
    out = compare(case, ref, candidate)
    assert out['passed'], out


def test_optimization_checks_both_force_endpoints_not_same_geometry():
    case, ref, candidate = sample('optimization')
    candidate['geometry_bohr'][1][0] += 0.01
    candidate['energy_hartree'] += 1e-5
    assert compare(case, ref, candidate)['passed']
    ref['gradient_hartree_per_bohr'][1][2] = 0.001
    assert not compare(case, ref, candidate)['passed']
