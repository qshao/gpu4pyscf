"""Production dispatch and complete DF exchange integration for layout GEMM."""
import cupy as cp
import pytest


@pytest.mark.parametrize('backend',['cupy','opt_einsum'])
def test_production_layout_exchange(monkeypatch,backend):
    from gpu4pyscf.df import df_jk
    assert hasattr(df_jk,'_exchange_transform'), 'production layout transform missing'
    from cupy_backends.cuda.libs import cublas
    original=cublas.dgemmStridedBatched
    calls=[]
    def observed(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(cublas,'dgemmStridedBatched',observed)
    monkeypatch.setattr(df_jk,'contract_engine',backend)
    # Independent diagonal-integral formula: sum_j A[L,i,j] B[n,j,k]
    # = (L+1)*B[n,i,k].  The final block is padded to exercise its fallback.
    work=cp.zeros((512,512,256))
    idx=cp.arange(512)
    work[idx,idx,:]=cp.arange(1,257,dtype=cp.float64)
    b=cp.arange(2*512*3,dtype=cp.float64).reshape(2,512,3)/1000
    for naux in [256,66]:
        a=work[:,:,:naux].transpose(2,0,1)
        out=cp.empty((2,512,3,naux))
        calls.clear()
        assert df_jk._exchange_transform(a,b,out) is out
        expected=b[:,:,:,None]*cp.arange(1,naux+1,dtype=cp.float64)
        cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)
        assert bool(calls)==(backend=='cupy' and naux==256)


def test_production_layout_get_jk(monkeypatch):
    from gpu4pyscf.df import df_jk
    assert hasattr(df_jk,'_exchange_transform'), 'production layout transform missing'
    import sys
    from types import SimpleNamespace
    from gpu4pyscf.lib.cupy_helper import tag_array
    original=df_jk._exchange_transform
    calls=[]
    def observed(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(df_jk,'_exchange_transform',observed)
    coeff=cp.arange(7*2,dtype=cp.float64).reshape(7,2)/100
    dm=tag_array(coeff@coeff.T,mo_coeff=coeff,mo_occ=cp.ones(2))
    work=cp.zeros((7,7,5))
    idx=cp.arange(7)
    work[idx,idx,0]=1.; work[idx,idx,1]=2.
    cderi=work[:,:,:2].transpose(2,0,1)
    obj=SimpleNamespace(mol=SimpleNamespace(stdout=sys.stdout,verbose=0),verbose=0,
        _cderi=True,get_blksize=lambda **kw:2,
        loop=lambda **kw:iter([(cderi,None)]))
    vj,vk=df_jk.get_jk(obj,dm,hermi=1,with_j=False)
    assert vj is None and calls
    cp.testing.assert_allclose(vk,5*dm,rtol=2e-12,atol=2e-12)
