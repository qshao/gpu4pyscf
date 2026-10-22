"""Regression for stable auxiliary-metric application in DF gradients."""
import numpy as np
import pytest


@pytest.mark.parametrize('decomposition',['CD','ED'])
@pytest.mark.parametrize('shape',[(11,3,5),(11,15),(11,17,19)])
def test_auxiliary_metric_matches_direct_solve(decomposition,shape):
    import cupy as cp
    from gpu4pyscf.df.grad.rhf import _apply_auxiliary_metric,_gen_metric_solver
    rng=np.random.default_rng(48)
    factor=rng.normal(size=(8,8)); matrix=factor@factor.T+np.eye(8)
    coeff=rng.normal(size=(11,8)); rhs=rng.normal(size=shape)
    expected=coeff@np.linalg.solve(matrix,coeff.T@rhs.reshape(11,-1))
    gpu_rhs=cp.asarray(rhs); gpu_coeff=cp.asarray(coeff)
    solve=_gen_metric_solver(cp.asarray(matrix),decomposition)
    actual=_apply_auxiliary_metric(gpu_coeff,solve,gpu_rhs)
    np.testing.assert_allclose(actual.get().reshape(11,-1),expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(gpu_rhs.get(),rhs)
    np.testing.assert_array_equal(gpu_coeff.get(),coeff)


def test_auxiliary_metric_can_fill_a_spin_slice():
    import cupy as cp
    from gpu4pyscf.df.grad.rhf import _apply_auxiliary_metric
    coeff=cp.eye(3); rhs=cp.arange(12,dtype=cp.float64).reshape(3,2,2)
    storage=cp.empty((2,3,2,2)); out=storage[1]
    assert _apply_auxiliary_metric(coeff,lambda x: x/2,rhs,out=out) is out
    cp.testing.assert_array_equal(out,rhs/2)


def test_nearly_dependent_metric_against_high_precision():
    from decimal import Decimal,localcontext
    import cupy as cp
    from gpu4pyscf.df.grad.rhf import _apply_auxiliary_metric,_gen_metric_solver
    # The Cholesky factor is exactly dyadic, isolating cancellation caused by
    # explicitly forming the inverse rather than factorization error itself.
    matrix=np.array([[1.,1.],[1.,1.+2.**-36]])
    coeff=np.eye(2)
    rhs=np.array([[1.234567891234,2.345678912345,.876543219876],
                  [1.234567891236,2.345678912349,.876543219879]])
    # Solve the exact binary input in high precision, independently of BLAS,
    # Cholesky and the production algorithm. Forming C J^-1 C.T first loses
    # accuracy on this small counterpart of the Taxol auxiliary metric.
    with localcontext() as context:
        context.prec=70
        def decimal_array(array):
            return np.array([[Decimal.from_float(float(x)) for x in row]
                             for row in array],dtype=object)
        metric=decimal_array(matrix); transform=decimal_array(coeff)
        projected=transform.T@decimal_array(rhs)
        augmented=np.concatenate((metric,projected),axis=1)
        size=len(matrix)
        for column in range(size):
            pivot=column+np.argmax(np.abs(augmented[column:,column]))
            augmented[[column,pivot]]=augmented[[pivot,column]]
            augmented[column,column:]/=augmented[column,column]
            for row in range(size):
                if row!=column:
                    augmented[row,column:]-=augmented[row,column]*augmented[column,column:]
        expected=np.array(transform@augmented[:,size:],dtype=float)
    solve=_gen_metric_solver(cp.asarray(matrix))
    actual=_apply_auxiliary_metric(cp.asarray(coeff),solve,cp.asarray(rhs))
    np.testing.assert_allclose(actual.get(),expected,atol=1e-12,rtol=0)
