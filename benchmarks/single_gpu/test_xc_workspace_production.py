"""Batch planning bounds and independent full-response molecular gradients."""
import numpy as np
import pytest


@pytest.mark.parametrize('available,natm', [(100*1024**3,113), (1024**3,3),
                                          (16*48*3*10,3), (1,113)])
def test_grid_response_batch_target(available,natm):
    from gpu4pyscf.grad.rks import _grid_response_batch_size
    block = _grid_response_batch_size(available,natm)
    target = min(int(available*.1),512*1024**2)
    unit = natm*6*8
    assert block % 16 == 0 and block >= 0
    assert block*unit <= target
    assert (block+16)*unit > target


@pytest.mark.parametrize('xc', ['lda,vwn', 'pbe', 'tpss'])
def test_full_grid_response_against_cpu(monkeypatch,xc):
    from pyscf import gto,dft,lib
    from gpu4pyscf.grad import rks
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='def2-svp',verbose=0)
    with lib.with_omp_threads(4):
        cpu = dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
        cpu.xc = xc
        cpu.grids.level = 1
        cpu.small_rho_cutoff = 0.
        cpu.conv_tol = 1e-12
        cpu.conv_tol_grad = 1e-9
        cpu.kernel()
        assert cpu.converged
        cg = cpu.nuc_grad_method()
        cg.grid_response = cg.auxbasis_response = True
        expected = cg.kernel()
        gpu = cpu.to_gpu()
        gpu.kernel()
        assert gpu.converged
        # Deliberately many small batches, retaining every grid/response term.
        monkeypatch.setattr(rks,'_grid_response_batch_size',lambda available,natm:256)
        gg = gpu.nuc_grad_method()
        gg.grid_response = gg.auxbasis_response = True
        actual = gg.kernel()
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
