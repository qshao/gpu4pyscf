"""Production dispatch at the measured short-block boundary."""
import pytest


@pytest.mark.parametrize('naux,nocc,backend,expected',[
    (32,128,'cupy',True),(31,128,'cupy',False),
    (32,127,'cupy',False),(32,128,'opt_einsum',False)])
def test_short_exchange_dispatch(monkeypatch,naux,nocc,backend,expected):
    import cupy as cp
    from gpu4pyscf.df import df_jk
    from cupy_backends.cuda.libs import cublas
    original=cublas.dgemmStridedBatched
    calls=[]
    def observed(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(cublas,'dgemmStridedBatched',observed)
    monkeypatch.setattr(df_jk,'contract_engine',backend)
    nao=1024
    work=cp.zeros((nao,nao,32))
    idx=cp.arange(nao)
    work[idx,idx,:]=cp.arange(1,33,dtype=cp.float64)
    a=work[:,:,:naux].transpose(2,0,1)
    b=cp.arange(nao*nocc,dtype=cp.float64).reshape(1,nao,nocc)/1000
    out=cp.empty((1,nao,nocc,naux))
    assert df_jk._exchange_transform(a,b,out) is out
    cp.testing.assert_allclose(out,b[:,:,:,None]*cp.arange(1,naux+1,dtype=cp.float64),
                               rtol=2e-12,atol=2e-12)
    assert bool(calls)==expected
