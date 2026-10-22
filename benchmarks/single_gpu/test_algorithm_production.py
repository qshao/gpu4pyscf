"""Production algorithm numerical and memory invariants."""
import cupy as cp
import numpy as np
import pytest


def test_production_symmetric_gram():
    from gpu4pyscf.df import df_jk
    assert hasattr(df_jk,'_symmetric_gram'), 'production symmetric accumulator missing'
    x=cp.arange(2*11*17,dtype=cp.float64).reshape(2,11,17)/100
    out=cp.ones((2,11,11))
    expected=out+x@x.swapaxes(-1,-2)
    assert df_jk._symmetric_gram(x,out) is out
    cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)


def test_production_gga_density_small_scratch():
    from gpu4pyscf.dft import numint
    rng=np.random.default_rng(91)
    ao=rng.normal(size=(4,12,103)); coeff=rng.normal(size=(12,8))
    mo=np.einsum('pi,xpg->xig',coeff,ao)
    expected=np.einsum('ig,xig->xg',mo[0],mo); expected[1:]*=2
    # DM-sized scratch fits this block without the two MO-sized workspaces.
    result=numint._eval_rho2(cp.asarray(ao),cp.asarray(coeff),'GGA',buf=cp.empty(12*103))
    np.testing.assert_allclose(cp.asnumpy(result),expected,rtol=2e-12,atol=2e-11)


@pytest.mark.parametrize('backend',['cupy','opt_einsum'])
def test_large_df_exchange_backend_guard(monkeypatch,backend):
    import sys
    from types import SimpleNamespace
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.lib.cupy_helper import tag_array
    assert hasattr(df_jk,'_symmetric_gram'), 'production symmetric accumulator missing'
    coeff=cp.arange(512*2,dtype=cp.float64).reshape(512,2)/1000
    dm=tag_array(coeff@coeff.T,mo_coeff=coeff,mo_occ=cp.ones(2))
    cderi=cp.stack([cp.eye(512),2*cp.eye(512)])
    obj=SimpleNamespace(mol=SimpleNamespace(stdout=sys.stdout,verbose=0),verbose=0,
        _cderi=True,get_blksize=lambda **kw:2,
        loop=lambda **kw:iter([(cderi,None)]))
    original=df_jk._symmetric_gram
    calls=[]
    def observed(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(df_jk,'_symmetric_gram',observed)
    monkeypatch.setattr(df_jk,'contract_engine',backend)
    vj,vk=df_jk.get_jk(obj,dm,hermi=1,with_j=False)
    assert vj is None
    assert bool(calls)==(backend=='cupy')
    cp.testing.assert_allclose(vk,5*dm,rtol=2e-12,atol=2e-12)
