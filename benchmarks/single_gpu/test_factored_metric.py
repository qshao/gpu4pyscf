"""Independent oracle for metric application without forming its inverse."""
import numpy as np
import pytest


@pytest.mark.parametrize('columns',[1,7,64])
def test_factorized_metric_action(columns):
    import cupy as cp
    from factored_metric import apply_metric
    rng=np.random.default_rng(48)
    factor=rng.normal(size=(8,8)); matrix=factor@factor.T+np.eye(8)
    coeff=rng.normal(size=(11,8)); rhs=rng.normal(size=(11,3,5))
    expected=coeff@np.linalg.solve(matrix,coeff.T@rhs.reshape(11,-1))
    gpu_rhs=cp.asarray(rhs); gpu_coeff=cp.asarray(coeff)
    def solve(block):
        assert block.shape[1]<=columns
        return cp.linalg.solve(cp.asarray(matrix),block)
    actual=apply_metric(gpu_coeff,solve,gpu_rhs,columns=columns)
    np.testing.assert_allclose(actual.get().reshape(11,-1),expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(gpu_rhs.get(),rhs)
    np.testing.assert_array_equal(gpu_coeff.get(),coeff)
