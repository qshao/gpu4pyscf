"""A saved converged orbital state must reproduce the original full gradient."""
import pytest


@pytest.mark.special
def test_restore_converged_orbitals(tmp_path):
    from gradient_snapshot import write_orbitals,restore
    import cupy as cp
    from pyscf import lib
    import molecular
    from comprehensive import configured
    case='vitamin_svp_b3lyp'
    with lib.with_omp_threads(4),configured(case):
        mf,dm=molecular._mean_field('B3LYP','gpu')
        mf.kernel(dm0=dm)
        assert mf.converged
        grad=mf.nuc_grad_method()
        grad.grid_response=grad.auxbasis_response=True
        expected=grad.kernel()
        path=tmp_path/'orbitals.npz'
        write_orbitals(mf,path)
        replay=restore(case,path)
        actual=replay.nuc_grad_method()
        actual.grid_response=actual.auxbasis_response=True
        cp.testing.assert_allclose(actual.kernel(),expected,atol=1e-9,rtol=0)
        assert replay.e_tot==mf.e_tot
