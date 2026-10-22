"""Numerical checks for auxiliary-contiguous exchange transformations."""
import cupy as cp
import numpy as np
import pytest

import exchange_layout as candidate


@pytest.mark.parametrize('batch,nao,nocc,naux,pitch', [
    (1,7,3,11,11), (2,7,3,5,11), (1,7,0,5,11),
    (1,0,3,5,11), (1,7,3,0,11), (0,7,3,5,11)])
def test_transform(batch, nao, nocc, naux, pitch):
    assert hasattr(candidate, 'transform'), 'layout-aware transform missing'
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(nao,nao,pitch))
    coeff = rng.normal(size=(batch,nao,nocc))
    expected = np.einsum('ijL,njk->nikL', raw[:,:,:naux], coeff)
    cderi = cp.asarray(raw)[:,:,:naux].transpose(2,0,1)
    out = cp.empty((batch,nao,nocc,naux))
    actual = candidate.transform(cderi, cp.asarray(coeff), out)
    assert actual is out
    np.testing.assert_allclose(out.get(), expected, rtol=2e-12, atol=2e-12)


def test_transform_stream_and_pointer_mode():
    assert hasattr(candidate, 'transform'), 'layout-aware transform missing'
    from cupy_backends.cuda.libs import cublas
    from cupy.cuda import device
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        a = cp.arange(7*7*11,dtype=cp.float64).reshape(7,7,11).transpose(2,0,1)
        b = cp.arange(2*7*3,dtype=cp.float64).reshape(2,7,3)
        out = cp.empty((2,7,3,11))
        handle = device.get_cublas_handle()
        mode = cublas.getPointerMode(handle)
        cublas.setPointerMode(handle,cublas.CUBLAS_POINTER_MODE_DEVICE)
        try:
            candidate.transform(a,b,out)
            assert cublas.getPointerMode(handle) == cublas.CUBLAS_POINTER_MODE_DEVICE
        finally: cublas.setPointerMode(handle,mode)
    stream.synchronize()
    np.testing.assert_allclose(out.get(),np.einsum('Lij,njk->nikL',a.get(),b.get()),rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('case',['eligible','complex','input_layout','output_layout','alias','scaled','accumulate','backend'])
def test_dispatch(monkeypatch,case):
    assert hasattr(candidate,'version'), 'layout candidate context missing'
    from gpu4pyscf.df import df_jk
    a = cp.arange(5*5*5,dtype=cp.float64).reshape(5,5,5).transpose(2,0,1)/100
    # Division may choose an arbitrary dense order; establish the DF layout.
    a = cp.ascontiguousarray(a.transpose(1,2,0)).transpose(2,0,1)
    b = cp.arange(25,dtype=cp.float64).reshape(1,5,5)/25
    out = cp.zeros((1,5,5,5))
    if case=='complex': a,b,out = a.astype(cp.complex128),b.astype(cp.complex128),out.astype(cp.complex128)
    if case=='input_layout': a=cp.ascontiguousarray(a)
    if case=='output_layout': out=out[:,:,:,::-1]
    if case=='alias': out=a.transpose(1,2,0)[None]
    alpha=.5 if case=='scaled' else 1.
    beta=1. if case=='accumulate' else 0.
    expected=alpha*cp.einsum('Lij,njk->nikL',a,b)+beta*out
    calls=[]
    def fallback(pattern,x,y,alpha=1.,beta=0.,out=None):
        calls.append(pattern)
        result=alpha*cp.einsum(pattern,x,y)+beta*out
        out[:]=result
        return out
    monkeypatch.setattr(df_jk,'contract',fallback)
    monkeypatch.setattr(df_jk,'contract_engine','opt_einsum' if case=='backend' else 'cupy')
    monkeypatch.setattr(candidate,'worthwhile',lambda a:True,raising=False)
    with candidate.version():
        df_jk.contract('Lij,njk->nikL',a,b,alpha=alpha,beta=beta,out=out)
    assert df_jk.contract is fallback
    assert bool(calls)==(case!='eligible')
    cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('nao,naux,expected',[(208,256,False),(537,66,False),(537,256,True),(512,256,True)])
def test_measured_size_guard(nao,naux,expected):
    assert hasattr(candidate,'worthwhile'), 'measured layout size guard missing'
    from types import SimpleNamespace
    assert candidate.worthwhile(SimpleNamespace(shape=(naux,nao,nao))) is expected
