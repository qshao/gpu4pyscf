"""Scientific and storage invariants for algorithm candidates."""
import cupy as cp
import numpy as np
import pytest
import algorithm_candidates as candidates


@pytest.mark.parametrize('batch,rows,rank', [(1, 7, 11), (2, 7, 11), (1, 7, 0), (1, 0, 5)])
def test_symmetric_gram(batch, rows, rank):
    assert hasattr(candidates, 'symmetric_gram'), 'symmetric Gram candidate missing'
    rng=np.random.default_rng(17)
    x=rng.normal(size=(batch, rows, rank))
    initial=rng.normal(size=(batch,rows,rows))
    initial=initial+initial.swapaxes(-1,-2)
    out=cp.asarray(initial)
    actual=candidates.symmetric_gram(cp.asarray(x),out)
    assert actual is out
    np.testing.assert_allclose(cp.asnumpy(out),initial+x@x.swapaxes(-1,-2),rtol=2e-12,atol=2e-12)


def test_symmetric_gram_input_alias():
    assert hasattr(candidates, 'symmetric_gram'), 'symmetric Gram candidate missing'
    x=cp.arange(49,dtype=cp.float64).reshape(1,7,7)/49
    x=x+x.swapaxes(-1,-2)
    expected=x.copy()+x@x.swapaxes(-1,-2)
    candidates.symmetric_gram(x,x)
    cp.testing.assert_allclose(x,expected,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('nao,nocc,ngrids',[(12,8,103),(80,8,103),(12,0,103),(0,8,103)])
def test_adaptive_density(nao,nocc,ngrids):
    assert hasattr(candidates, 'adaptive_rho'), 'adaptive density candidate missing'
    rng=np.random.default_rng(43)
    ao=rng.normal(size=(4,nao,ngrids))
    coeff=rng.normal(size=(nao,nocc))*np.sqrt(.3)
    mo=np.einsum('pi,xpg->xig',coeff,ao)
    expected=np.einsum('ig,xig->xg',mo[0],mo)
    expected[1:]*=2
    from gpu4pyscf.dft import numint
    evaluate=candidates.adaptive_rho(numint._eval_rho2)
    for size in (1, max(nao,2*nocc)):
        buf=cp.empty(size*ngrids)
        out=cp.empty((4,ngrids))
        result=evaluate(cp.asarray(ao),cp.asarray(coeff),'GGA',buf=buf,rho=out)
        assert result.data.ptr==out.data.ptr
        np.testing.assert_allclose(cp.asnumpy(result),expected,rtol=3e-12,atol=3e-11)


@pytest.mark.parametrize('case', ['eligible','small','complex','mixed','scaled','strided'])
def test_exchange_dispatch(monkeypatch,case):
    from gpu4pyscf.df import df_jk
    rows=511 if case=='small' else 512
    dtype=cp.complex128 if case=='complex' else cp.float64
    a=cp.arange(rows*6,dtype=cp.float64).reshape(1,rows,2,3).astype(dtype)/1000
    if case=='strided': a=a[:,:,:,::-1]
    b=a.copy() if case=='mixed' else a
    alpha=.5 if case=='scaled' else 1.
    out=cp.zeros((1,rows,rows),dtype=dtype)
    expected=alpha*cp.einsum('nikL,njkL->nij',a,b)
    calls=[]
    def original(pattern,x,y,alpha=1.,beta=0.,out=None):
        calls.append(pattern)
        value=alpha*cp.einsum(pattern,x,y)+beta*out
        out[:]=value
        return out
    monkeypatch.setattr(df_jk,'contract',original)
    with candidates.algorithm_version('exchange'):
        df_jk.contract('nikL,njkL->nij',a,b,alpha=alpha,beta=1.,out=out)
    assert bool(calls)==(case!='eligible')
    assert df_jk.contract is original
    cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)
