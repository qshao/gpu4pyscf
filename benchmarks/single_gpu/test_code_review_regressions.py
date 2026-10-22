"""Scientific/API edge cases found by the comprehensive source review."""
from pathlib import Path
import runpy
import subprocess
import sys
import weakref

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('command,wheel,expected',[
    ('--name',False,'gpu4pyscf'),('build_py',False,'gpu4pyscf'),
    ('sdist',False,'gpu4pyscf'),('--name',True,'gpu4pyscf-cuda13x')])
def test_default_and_explicit_package_names(monkeypatch,command,wheel,expected):
    import setuptools
    from wheel.bdist_wheel import bdist_wheel
    monkeypatch.setattr(bdist_wheel,'initialize_options',bdist_wheel.initialize_options)
    try:
        from setuptools.command.bdist_wheel import bdist_wheel as newer
        if newer is not bdist_wheel:
            monkeypatch.setattr(newer,'initialize_options',newer.initialize_options)
    except ImportError:
        pass
    monkeypatch.setattr(sys,'argv',['setup.py',command])
    monkeypatch.delenv('GPU4PYSCF_BUILD_CUDA_WHEEL',raising=False)
    if wheel:
        monkeypatch.setenv('GPU4PYSCF_BUILD_CUDA_WHEEL','1')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:b'nvcc release 13.0, V13.0.88')
    calls=[]
    monkeypatch.setattr(setuptools,'setup',lambda **kwargs:calls.append(kwargs))
    runpy.run_path(str(ROOT/'setup.py'),run_name='__main__')
    assert len(calls)==1 and calls[0]['name']==expected


@pytest.mark.parametrize('shape',[(4,1),(1,4),(1,1),(3,5)])
@pytest.mark.parametrize('order',['C','F','padded'])
@pytest.mark.parametrize('pattern',['np,Lp->nL','nL,Lp->np'])
def test_coulomb_singleton_and_padded_layouts(shape,order,pattern):
    import cupy as cp
    from gpu4pyscf.df.df_jk import _coulomb_contract
    rows,cols=shape
    if order=='padded':
        b=cp.empty((rows+2,cols),order='F')[:rows]
    else:
        b=cp.empty(shape,order=order)
    host=np.arange(rows*cols,dtype=float).reshape(shape)/7
    b[:]=cp.asarray(host)
    a=np.arange(cols if pattern=='np,Lp->nL' else rows,dtype=float)[None]+1
    expected=.7*np.einsum(pattern,a,host)+.3
    out=cp.ones(expected.shape)
    assert _coulomb_contract(pattern,cp.asarray(a),b,.7,.3,out) is out
    np.testing.assert_allclose(out.get(),expected,rtol=1e-12,atol=1e-12)


def test_coulomb_rejects_fractional_element_pitch(monkeypatch):
    import cupy as cp
    from gpu4pyscf.df import df_jk
    backing=cp.empty(100,dtype=cp.uint8)
    b=cp.ndarray((2,3),dtype=cp.float64,memptr=backing.data,strides=(8,25))
    a=cp.ones((1,3))
    sentinel=object()
    monkeypatch.setattr(df_jk,'contract',lambda *args,**kwargs:sentinel)
    from cupy_backends.cuda.libs import cublas
    def forbidden(*args,**kwargs):
        pytest.fail('fractional element pitch reached raw DGEMV')
    monkeypatch.setattr(cublas,'dgemv',forbidden)
    assert df_jk._coulomb_contract('np,Lp->nL',a,b) is sentinel


def test_exchange_rejects_fractional_element_pitch(monkeypatch):
    import cupy as cp
    from gpu4pyscf.df import df_jk
    from cupy_backends.cuda.libs import cublas
    pitch=256*8+1
    backing=cp.empty(512*512*pitch,dtype=cp.uint8)
    a=cp.ndarray((256,512,512),dtype=cp.float64,memptr=backing.data,
                 strides=(8,512*pitch,pitch))
    b=cp.empty((1,512,1));out=cp.empty((1,512,1,256))
    sentinel=object()
    monkeypatch.setattr(df_jk,'contract',lambda *args,**kwargs:sentinel)
    def forbidden(*args,**kwargs):
        pytest.fail('fractional element pitch reached raw DGEMM')
    monkeypatch.setattr(cublas,'dgemmStridedBatched',forbidden)
    assert df_jk._exchange_transform(a,b,out) is sentinel


@pytest.mark.parametrize('rhs_order',['C','F'])
@pytest.mark.parametrize('output',['default','C','F','strided'])
def test_auxiliary_metric_writes_requested_layout(rhs_order,output):
    import cupy as cp
    from gpu4pyscf.df.grad.rhf import _apply_auxiliary_metric
    host=np.arange(3*17*19,dtype=float).reshape(3,17,19)/7
    rhs=cp.array(host,order=rhs_order)
    out=None
    storage=None
    if output=='strided':
        storage=cp.full((3,34,38),-123.,dtype=cp.float64)
        out=storage[:,::2,::2]
    elif output!='default':
        out=cp.full(host.shape,-123.,order=output)
    actual=_apply_auxiliary_metric(cp.eye(3),lambda x:x/2,rhs,out)
    if out is not None:
        assert actual is out
    np.testing.assert_allclose(actual.get(),host/2,rtol=1e-12,atol=1e-12)
    cp.testing.assert_array_equal(rhs,cp.asarray(host))
    if storage is not None:
        cp.testing.assert_array_equal(storage[:,1::2,:],-123.)
        cp.testing.assert_array_equal(storage[:,:,1::2],-123.)


def test_grid_response_releases_previous_batch(monkeypatch):
    import cupy as cp
    from pyscf import dft,gto,lib
    from gpu4pyscf.grad import rks
    from gpu4pyscf.hessian import rks as hessian
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
              basis='def2-svp',verbose=0)
    previous=[]
    original=hessian.get_dweight_dA
    def observed(*args,**kwargs):
        assert not previous or previous[-1]() is None,'previous derivative batch remains live'
        result=original(*args,**kwargs)
        previous.append(weakref.ref(result))
        return result
    with lib.with_omp_threads(4):
        cpu=dft.RKS(mol,xc='pbe').density_fit(auxbasis='def2-universal-jkfit')
        cpu.grids.level=0;cpu.small_rho_cutoff=0.
        cpu.conv_tol=1e-11;cpu.conv_tol_grad=1e-8
        density=cpu.get_init_guess()
        cpu.kernel(dm0=density)
        assert cpu.converged
        reference=cpu.nuc_grad_method()
        reference.grid_response=reference.auxbasis_response=True
        expected=reference.kernel()
        gpu=cpu.to_gpu();gpu.kernel(dm0=density)
        assert gpu.converged
        monkeypatch.setattr(rks,'_grid_response_batch_size',lambda *args:256)
        monkeypatch.setattr(hessian,'get_dweight_dA',observed)
        gradient=gpu.nuc_grad_method()
        gradient.grid_response=gradient.auxbasis_response=True
        actual=cp.asnumpy(gradient.kernel())
    assert len(previous)>1 and all(ref() is None for ref in previous)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
