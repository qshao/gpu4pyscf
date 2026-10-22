"""NumPy-oracle tests for deterministic coalesced primitive recontraction."""
import numpy as np
import pytest


@pytest.mark.special
@pytest.mark.parametrize('naux',[1,17,513])
def test_gather_primitive_contributions(naux):
    import cupy as cp
    from recontract_gather import gather,csr_mapping
    rng=np.random.default_rng(43)
    data=rng.normal(size=(11,naux))
    inp=np.array([1,8,2,3,1,5,9,2,10],dtype=np.int32)
    out=np.array([2,0,2,4,0,2,4,4,2],dtype=np.int32)
    coef=np.array([.2,-.3,1.,-.7,0.,.4,.9,-1.,.05])
    expected=np.zeros((6,naux))
    for i,j,c in zip(inp,out,coef): expected[j]+=c*data[i]
    mapping=csr_mapping(inp,out,coef,6)
    array=cp.asarray(data)
    target=cp.full(expected.shape,cp.nan)
    actual=gather(array,mapping,out=target)
    assert actual is target
    cp.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    first=actual.copy()
    for _ in range(10): gather(array,mapping,out=target)
    assert bool(cp.array_equal(actual,first))


@pytest.mark.special
def test_gather_empty_and_nondefault_stream():
    import cupy as cp
    from recontract_gather import gather,csr_mapping
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        mapping=csr_mapping(np.array([0],np.int32),np.array([1],np.int32),np.array([2.]),3)
        data=cp.arange(19,dtype=cp.float64)[None,:]
        result=gather(data,mapping)
        empty=gather(cp.empty((1,0)),mapping)
    stream.synchronize()
    expected=np.zeros((3,19)); expected[1]=2*np.arange(19)
    cp.testing.assert_array_equal(result,expected)
    assert empty.shape==(3,0)


@pytest.mark.special
def test_gather_rejects_invalid_buffers():
    import cupy as cp
    from recontract_gather import gather,csr_mapping
    mapping=csr_mapping(np.array([0],np.int32),np.array([0],np.int32),np.array([1.]),1)
    data=cp.ones((1,3))
    with pytest.raises(ValueError,match='output'): gather(data,mapping,out=cp.empty((1,2)))
    with pytest.raises(ValueError,match='alias'): gather(data,mapping,out=data)
    with pytest.raises(ValueError,match='FP64'): gather(data.astype(cp.float32),mapping)


@pytest.mark.special
@pytest.mark.parametrize('basis',['def2-svp','cc-pvdz'])
def test_actual_density_fitting_recontraction(basis):
    from unittest.mock import patch
    import cupy as cp
    from pyscf import gto,scf,lib
    from gpu4pyscf.df import df,int3c2e_bdiv
    from recontract_gather import coalesced_recontraction
    with lib.with_omp_threads(2):
        mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis=basis,verbose=0)
        dm=cp.asarray(scf.RHF(mol).get_init_guess())
        original=df.DF(mol,auxbasis='def2-universal-jkfit')
        original.use_gpu_memory=True
        expected=original.get_jk(dm)
        before=int3c2e_bdiv._create_pair_recontractor
        calls=[]
        with coalesced_recontraction():
            function=int3c2e_bdiv._coalesced_gather
            def observed(*args,**kwargs):
                calls.append(True)
                return function(*args,**kwargs)
            with patch.object(int3c2e_bdiv,'_coalesced_gather',observed):
                candidate=df.DF(mol,auxbasis='def2-universal-jkfit')
                candidate.use_gpu_memory=True
                actual=candidate.get_jk(dm)
        assert int3c2e_bdiv._create_pair_recontractor is before
        assert bool(calls)==(basis=='cc-pvdz')
        for left,right in zip(actual,expected):
            cp.testing.assert_allclose(left,right,rtol=0,atol=1e-10)
