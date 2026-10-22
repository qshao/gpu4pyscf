"""Independent numerical checks before any production changes."""
import cupy as cp
import numpy as np
import pytest

import speed_memory_candidates as candidate


def test_version_context_restores_bindings():
    assert hasattr(candidate,'version'), 'reversible speed/memory context missing'
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.df.grad import rhf
    originals=df_jk.contract,rhf.contract,rhf._jk_energy_per_atom
    for version in ['current','contractions','compact','combined']:
        with candidate.version(version): pass
        assert (df_jk.contract,rhf.contract,rhf._jk_energy_per_atom)==originals


@pytest.mark.parametrize('pattern',['np,Lp->nL','nL,Lp->np'])
@pytest.mark.parametrize('alpha,beta',[(1.,0.),(-.25,1.),(.5,.3)])
def test_coulomb(pattern,alpha,beta):
    assert hasattr(candidate,'coulomb'), 'direct Coulomb candidate missing'
    rng=np.random.default_rng(81)
    a=rng.normal(size=(1,11 if pattern.startswith('np') else 5))
    b=rng.normal(size=(5,11))
    ref=np.einsum(pattern,a,b)
    initial=rng.normal(size=ref.shape)
    out=cp.asarray(initial)
    result=candidate.coulomb(pattern,cp.asarray(a),cp.asarray(b),out,alpha,beta)
    assert result is out
    np.testing.assert_allclose(out.get(),alpha*ref+beta*initial,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('pattern',['pqr,pi->iqr','iqr,pi->pqr'])
@pytest.mark.parametrize('padded',[False,True])
def test_gradient(pattern,padded):
    assert hasattr(candidate,'gradient'), 'direct gradient candidate missing'
    rng=np.random.default_rng(82)
    p,q,i,r=7,9,3,5
    rows=p if pattern.startswith('pqr') else i
    pitch=11 if padded else r
    raw=rng.normal(size=(rows,q,pitch))
    a=raw[:,:,:r]
    b=rng.normal(size=(p,i))
    ref=np.einsum(pattern,a,b)
    initial=rng.normal(size=ref.shape)
    out=cp.asarray(initial)
    gpu_a=cp.asarray(raw)[:,:,:r]
    result=candidate.gradient(pattern,gpu_a,cp.asarray(b),out,-.2,.7)
    assert result is out
    np.testing.assert_allclose(out.get(),-.2*ref+.7*initial,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('case',['eligible','alias','complex','strided','backend','batch','empty'])
def test_coulomb_dispatch(monkeypatch,case):
    from gpu4pyscf.df import df_jk
    a=cp.arange(5,dtype=cp.float64)[None]
    b=cp.arange(25,dtype=cp.float64).reshape(5,5)/25
    out=cp.ones((1,5))
    if case=='alias': out=a
    if case=='complex': a,b,out=a.astype(cp.complex128),b.astype(cp.complex128),out.astype(cp.complex128)
    if case=='strided': b=b[:,::-1]
    if case=='batch': a=cp.vstack([a,a]); out=cp.ones((2,5))
    if case=='empty': a=a[:,:0]; b=b[:,:0]
    expected=cp.einsum('np,Lp->nL',a,b)+out
    calls=[]
    def original(pattern,x,y,alpha=1.,beta=0.,out=None):
        calls.append(True); value=alpha*cp.einsum(pattern,x,y)+beta*out; out[:]=value; return out
    monkeypatch.setattr(df_jk,'contract',original)
    monkeypatch.setattr(df_jk,'contract_engine','opt_einsum' if case=='backend' else 'cupy')
    with candidate.version('contractions'):
        df_jk.contract('np,Lp->nL',a,b,beta=1.,out=out)
    assert bool(calls)==(case!='eligible')
    cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)


def test_padded_coulomb_does_not_allocate_tensor():
    from memory_profile import peak_memory
    storage=cp.empty((13,11),order='F')
    b=storage[:5]; b[:]=cp.arange(55,dtype=cp.float64).reshape(5,11)/55
    a=cp.arange(11,dtype=cp.float64)[None]
    out=cp.empty((1,5))
    expected=cp.einsum('np,Lp->nL',a,b)
    candidate.coulomb('np,Lp->nL',a,b,out)
    with peak_memory() as memory:
        candidate.coulomb('np,Lp->nL',a,b,out)
    cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)
    assert memory['peak_tracked_live_bytes']==0, 'padded GEMV copied the integral block'


def test_coulomb_allocates_output_without_falling_back(monkeypatch):
    from gpu4pyscf.df import df_jk
    a=cp.arange(11,dtype=cp.float64)[None]
    b=cp.arange(55,dtype=cp.float64).reshape(5,11)
    expected=cp.einsum('np,Lp->nL',a,b)
    def fallback(*args,**kwargs):
        pytest.fail('single-density Coulomb without out fell back')
    monkeypatch.setattr(df_jk,'contract',fallback)
    monkeypatch.setattr(df_jk,'contract_engine','cupy')
    with candidate.version('contractions'):
        actual=df_jk.contract('np,Lp->nL',a,b,beta=7.)
    cp.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
