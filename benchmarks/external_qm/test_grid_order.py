"""External functional integrators must retain slices for zero-AO grid blocks."""
import numpy as np
import pytest


@pytest.mark.parametrize("deriv", [0, 1])
def test_strict_grid_order_retains_zero_ao_block(deriv):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.dft import gen_grid, numint

    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    block = numint.MIN_BLK_SIZE
    grids = gen_grid.Grids(mol)
    grids.coords = cp.concatenate((cp.full((block, 3), 1e6), cp.zeros((block, 3))))
    grids.weights = cp.arange(1, 2 * block + 1, dtype=cp.float64)
    ni = numint.NumInt().build(mol, grids.coords)
    blocks = list(ni.block_loop(mol, grids, deriv=deriv, strict_grid_order=True))
    assert len(blocks) == 2
    assert blocks[0][1].size == 0, "First block must have no significant AOs"
    assert blocks[1][1].size > 0
    cp.testing.assert_array_equal(cp.concatenate([b[2] for b in blocks]), grids.weights)
    cp.testing.assert_array_equal(cp.concatenate([b[3] for b in blocks]), grids.coords)
    assert np.isfinite(blocks[1][0].get()).all()
