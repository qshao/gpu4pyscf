"""Registry and immutable identity checks; no CUDA or CuPy imports."""
import copy
import importlib
import json
from collections import Counter
from pathlib import Path
import pytest

NAME = 'benchmarks.single_gpu.qm_suite.manifest'


def api():
    try:
        return importlib.import_module(NAME)
    except ModuleNotFoundError as exc:
        pytest.fail(f'registry implementation missing: {exc}')


def test_exact_registry_counts_and_pilot():
    m = api()
    cases = m.load_cases()
    assert Counter(c['group'] for c in cases) == {'core': 60, 'extension': 56, 'direct': 6, 'periodic': 4, 'probe': 1, 'capacity': 12}
    assert len({c['id'] for c in cases}) == len(cases)
    assert len(m.pilot_cases()) == 10
    assert len(m.load_cases('core')) == 60
    assert m.pilot_cases()[2]['id'] == 'probe-oh-wb97x-svp'
    assert m.pilot_cases()[-1]['id'] == 'pbc-si-gamma'
    for case in cases:
        m.validate_case(case)


@pytest.mark.parametrize('mutation,match', [
    (lambda c: c.pop('charge'), 'charge'),
    (lambda c: c.pop('spin'), 'spin'),
    (lambda c: c.update(spin=1), 'parity'),
    (lambda c: c.update(task='magic'), 'task'),
    (lambda c: c.update(basis='def2-universal-jkfit'), 'orbital'),
    (lambda c: c.update(geometry_sha256='0'*64), 'hash'),
    (lambda c: c['task_options'].update(observables=[]), 'observables'),
])
def test_invalid_cases_rejected(mutation, match):
    m = api()
    c = m.get_case('core-h2o-b3lyp-svp')
    mutation(c)
    with pytest.raises(ValueError, match=match):
        m.validate_case(c)


def test_registry_copies_are_independent():
    m = api()
    c = m.get_case('core-h2o-b3lyp-svp')
    c['settings']['conv_tol'] = 1
    assert m.get_case(c['id'])['settings']['conv_tol'] == 1e-10


def test_correlation_basis_and_hessian_response():
    m = api()
    for c in m.load_cases('extension'):
        if c['task'] in ('dfmp2', 'dfump2'):
            assert c['auxbasis'] == 'def2-svp-ri'
            assert c['task_options']['frozen'] == 0
            assert c['task_options']['with_t2'] is False
        if c['task'] == 'hessian':
            assert c['task_options']['auxbasis_response'] == 2


def test_manifest_is_immutable_and_detects_source_changes(tmp_path):
    m = api()
    roots = {}
    for variant in ('G0', 'G1'):
        root = tmp_path / variant / 'gpu4pyscf'
        root.mkdir(parents=True)
        (root / '__init__.py').write_text('identity = 1\n')
        (root / 'native.so').write_bytes(b'library')
        roots[variant] = root.parent
    case = m.get_case('core-h2o-b3lyp-svp')
    out = tmp_path / 'manifest.json'
    result = m.write_manifest([case], roots, out)
    assert json.loads(out.read_text()) == result
    assert result['variants']['G0']['files']['gpu4pyscf/native.so'] == m.digest(roots['G0'] / 'gpu4pyscf/native.so')
    m.verify_manifest(result)
    with pytest.raises(FileExistsError):
        m.write_manifest([case], roots, out)
    (roots['G0'] / 'gpu4pyscf/__init__.py').write_text('identity = 2\n')
    with pytest.raises(ValueError, match='changed'):
        m.verify_manifest(result)


def test_duplicate_case_ids_rejected_in_manifest(tmp_path):
    m = api()
    case = m.get_case('core-h2o-b3lyp-svp')
    with pytest.raises(ValueError, match='duplicate'):
        m.write_manifest([case, case], {}, tmp_path/'bad.json')


def test_identity_ignores_id_but_tracks_scientific_settings():
    m = api()
    a = m.get_case('core-h2o-b3lyp-svp')
    b = copy.deepcopy(a)
    b['id'] = 'diagnostic'
    assert m.comparison_identity(a) == m.comparison_identity(b)
    b['settings']['grid_level'] = 4
    assert m.comparison_identity(a) != m.comparison_identity(b)


def test_basis_cpu_check_when_pyscf_available():
    pytest.importorskip('pyscf')
    from pyscf import gto
    m = api()
    for element in ('H', 'C', 'N', 'O'):
        assert gto.basis.load('def2-svp-ri', element)
    m.validate_case(m.get_case('ext-h2o-dfmp2'), check_basis=True)


def test_capacity_selection_uses_actual_ao_counts_and_unique_rungs():
    m = api()
    inventory = [
        {'geometry':'large_name.xyz','nao':1450},
        {'geometry':'tiny_name.xyz','nao':2550},
        {'geometry':'middle_name.xyz','nao':4100},
        {'geometry':'other.xyz','nao':900},
    ]
    selected = m.select_capacity_water_rungs(inventory)
    assert [row['geometry'] for row in selected] == ['large_name.xyz','tiny_name.xyz','middle_name.xyz']
    assert [row['target_nao'] for row in selected] == [1500,2500,4000]
    assert all(row['admission_required'] for row in selected)
    assert len(m.select_capacity_water_rungs(inventory[:1])) == 1


def test_changed_case_identity_in_frozen_manifest_rejected(tmp_path):
    m = api()
    c = m.get_case('core-h2o-b3lyp-svp')
    frozen = m.write_manifest([c], {}, tmp_path/'manifest.json')
    frozen['cases'][0]['settings']['grid_level'] = 4
    with pytest.raises(ValueError, match='identity'):
        m.verify_manifest(frozen)


def test_primary_storage_and_ccsd_reference_are_explicit():
    m = api()
    for case in m.load_cases():
        assert case['settings']['init_guess'] == 'minao'
        if case['group'] != 'capacity':
            assert case['settings']['storage'] == 'resident'
        if case['task'] == 'ccsd':
            assert case['df'] is False
            assert case['auxbasis'] is None
