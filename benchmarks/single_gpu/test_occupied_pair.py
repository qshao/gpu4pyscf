"""Independent math and eligibility tests for symmetric occupied-pair storage."""
import numpy as np
import pytest


@pytest.mark.special
def test_packing_preserves_symmetric_component():
    import occupied_pair
    assert hasattr(occupied_pair,'pack_oo'), 'symmetric-component packing missing'
    import cupy as cp
    rng=np.random.default_rng(29)
    full=rng.normal(size=(7,5,5))
    i,j=np.tril_indices(5)
    expected=((full+full.transpose(0,2,1))*.5)[:,i,j]
    out=cp.empty(expected.shape)
    result=occupied_pair.pack_oo(cp.asarray(full),cp.asarray(i),cp.asarray(j),out=out)
    assert result is out
    cp.testing.assert_array_equal(out,expected)
    transposed=cp.empty_like(out)
    occupied_pair.pack_oo(cp.asarray(full.transpose(0,2,1)),cp.asarray(i),cp.asarray(j),out=transposed)
    cp.testing.assert_array_equal(out,transposed)


@pytest.mark.special
@pytest.mark.parametrize('naux,nocc',[(7,3),(19,8)])
def test_packed_metric_trace_and_response(naux,nocc):
    import cupy as cp
    from occupied_pair import response_gram
    rng=np.random.default_rng(17)
    full=rng.normal(size=(naux,nocc,nocc))
    full=(full+full.transpose(0,2,1))*.5
    metric=rng.normal(size=(naux,naux))
    i,j=np.tril_indices(nocc)
    diagonal=np.cumsum(np.arange(1,nocc+1))-1
    packed=cp.asarray(full[:,i,j])
    transformed=cp.ascontiguousarray(cp.asarray(metric)@packed)
    expected=np.einsum('uv,vij->uij',metric,full)
    np.testing.assert_allclose(cp.asnumpy(transformed[:,diagonal].sum(axis=1)),
                               expected.trace(axis1=1,axis2=2),atol=1e-12,rtol=0)
    initial=rng.normal(size=(naux,naux))
    initial=(initial+initial.T)/2
    out=cp.asarray(initial)
    result=response_gram(transformed,cp.asarray(diagonal),alpha=-.37,beta=.6,out=out)
    assert result is out
    np.testing.assert_allclose(cp.asnumpy(result),
        -.37*np.einsum('rij,sij->rs',expected,expected)+.6*initial,atol=1e-10,rtol=0)


@pytest.mark.special
def test_response_gram_alias_and_complex():
    import cupy as cp
    from occupied_pair import response_gram
    rng=np.random.default_rng(19)
    p=rng.normal(size=(6,6))
    diagonal=np.array([0,2,5])
    expected=2*p@p.T-p[:,diagonal]@p[:,diagonal].T
    a=cp.asarray(p)
    result=response_gram(a,cp.asarray(diagonal),alpha=.2,beta=.4,out=a)
    np.testing.assert_allclose(cp.asnumpy(result),.2*expected+.4*p,atol=1e-12,rtol=0)
    p=p+1j*rng.normal(size=p.shape)
    result=response_gram(cp.asarray(p),cp.asarray(diagonal))
    np.testing.assert_allclose(cp.asnumpy(result),2*p@p.T-p[:,diagonal]@p[:,diagonal].T,atol=1e-12)


def test_candidate_restores_bindings():
    from gpu4pyscf.df.grad import rhf
    from occupied_pair import packed_gradient
    original=rhf._jk_energy_per_atom
    with packed_gradient():
        assert rhf._jk_energy_per_atom is not original
    assert rhf._jk_energy_per_atom is original


@pytest.mark.special
def test_complex_zero_beta_does_not_read_output():
    import cupy as cp
    from occupied_pair import response_gram
    p=cp.asarray(np.arange(18).reshape(3,6)+1j)
    diagonal=cp.asarray([0,2,5])
    out=cp.full((3,3),cp.nan+1j*cp.nan)
    response_gram(p,diagonal,beta=0,out=out)
    assert bool(cp.isfinite(out).all())


@pytest.mark.special
def test_response_output_shape_checked_before_native_call():
    from unittest.mock import patch
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
    from occupied_pair import response_gram
    with patch.object(cublas,'dsyrk',side_effect=AssertionError('unsafe native call')):
        with pytest.raises(ValueError,match='shape'):
            response_gram(cp.ones((3,6)),cp.asarray([0,2,5]),out=cp.empty((2,2)))


@pytest.mark.special
def test_response_strided_and_empty_fallbacks():
    import cupy as cp
    from occupied_pair import response_gram
    p=cp.arange(60,dtype=cp.float64).reshape(5,12)[:,::2]
    diagonal=cp.asarray([0,2,5])
    out=cp.full((10,10),2.)[::2,::2]
    expected=.7*(2*p@p.T-p[:,diagonal]@p[:,diagonal].T)+.6
    response_gram(p,diagonal,alpha=.7,beta=.3,out=out)
    cp.testing.assert_allclose(out,expected,atol=1e-12,rtol=0)
    empty=cp.empty((0,6))
    assert response_gram(empty,diagonal).shape==(0,0)


@pytest.mark.special
def test_response_nondefault_stream_and_pointer_mode():
    import cupy as cp
    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas
    from occupied_pair import response_gram
    stream=cp.cuda.Stream(non_blocking=True)
    host=np.arange(42,dtype=np.float64).reshape(7,6)/42
    expected=2*host@host.T-host[:,[0,2,5]]@host[:,[0,2,5]].T
    with stream:
        p=cp.asarray(host)
        diagonal=cp.asarray([0,2,5])
        handle=device.get_cublas_handle()
        previous=cublas.getPointerMode(handle)
        try:
            cublas.setPointerMode(handle,cublas.CUBLAS_POINTER_MODE_DEVICE)
            result=response_gram(p,diagonal)
            assert cublas.getPointerMode(handle)==cublas.CUBLAS_POINTER_MODE_DEVICE
        finally:
            cublas.setPointerMode(handle,previous)
    stream.synchronize()
    cp.testing.assert_allclose(result,expected,atol=1e-12,rtol=0)


@pytest.mark.special
@pytest.mark.parametrize('response',[False,True])
def test_forced_packed_molecular_gradient(response):
    from unittest.mock import patch
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf import dft
    from gpu4pyscf.df.grad import rhf
    from occupied_pair import packed_gradient
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='def2-svp',verbose=0)
    mf=dft.RKS(mol).density_fit()
    mf.xc='b3lyp'
    mf.grids.level=1
    mf.conv_tol=1e-11
    mf.kernel()
    assert mf.converged
    baseline=mf.nuc_grad_method()
    baseline.grid_response=True
    baseline.auxbasis_response=response
    expected=baseline.kernel()
    with packed_gradient(0,0),patch.object(rhf,'_occupied_unpack',wraps=rhf._occupied_unpack) as observed:
        grad=mf.nuc_grad_method()
        grad.grid_response=True
        grad.auxbasis_response=response
        actual=grad.kernel()
        assert observed.call_count>0
    np.testing.assert_allclose(cp.asnumpy(actual),cp.asnumpy(expected),atol=1e-9,rtol=0)


@pytest.mark.special
@pytest.mark.parametrize('hermi',[0,1])
def test_distinct_density_factors_keep_generic_path(hermi):
    from unittest.mock import patch
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf import scf
    from gpu4pyscf.df.grad import rhf
    from occupied_pair import packed_gradient
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='def2-svp',verbose=0)
    mf=scf.RHF(mol).density_fit()
    mf.conv_tol=1e-11
    mf.kernel()
    assert mf.converged
    dm=mf.make_rdm1().view(cp.ndarray)
    if hermi==0:
        dm=dm+cp.arange(dm.size).reshape(dm.shape)*1e-6
    grad=mf.nuc_grad_method()
    expected=grad.jk_energy_per_atom(dm,hermi=hermi)
    with packed_gradient(0,0),patch.object(rhf,'_occupied_unpack',side_effect=AssertionError('unsupported packing')):
        actual=grad.jk_energy_per_atom(dm,hermi=hermi)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)


@pytest.mark.special
@pytest.mark.parametrize('method',['HF','WB97X'])
def test_forced_packed_other_operators(method):
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf import scf,dft
    from occupied_pair import packed_gradient
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='def2-svp',verbose=0)
    mf=(scf.RHF(mol) if method=='HF' else dft.RKS(mol,xc=method)).density_fit()
    if method!='HF': mf.grids.level=1
    mf.conv_tol=1e-11
    mf.kernel()
    assert mf.converged
    def gradient():
        grad=mf.nuc_grad_method()
        grad.auxbasis_response=True
        if method!='HF': grad.grid_response=True
        return cp.asnumpy(grad.kernel())
    expected=gradient()
    with packed_gradient(0,0): actual=gradient()
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
