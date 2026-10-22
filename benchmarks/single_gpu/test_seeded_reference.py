"""Thread counts are execution controls; physical settings and gates stay strict."""
import pytest


def test_compare_execution_controls_only():
    from seeded_reference import compare_execution_cases
    left=dict(settings=dict(basis='def2-svp',grid_level=3,cpu_threads=16),method='B3LYP',
              converged=True,energy_hartree=-1.,gradient_hartree_per_bohr=[[0.,0.,0.]])
    right=dict(left,settings=dict(left['settings'],cpu_threads=4))
    assert compare_execution_cases(left,right)['passed']
    with pytest.raises(ValueError):
        compare_execution_cases(left,dict(right,settings=dict(right['settings'],basis='def2-tzvp')))
    assert not compare_execution_cases(left,dict(right,energy_hartree=-1.+1e-7))['passed']
    assert not compare_execution_cases(left,dict(right,gradient_hartree_per_bohr=[[1e-5,0.,0.]]))['passed']


@pytest.mark.special
@pytest.mark.benchmark
def test_taxol_cpu_from_gpu_seed():
    from seeded_reference import execute
    report=execute('taxol_svp_b3lyp')
    result=report['result']
    print('Taxol CPU seeded reference',result['iterations'],result['scf_seconds'],
          result['gradient_seconds'],flush=True)
