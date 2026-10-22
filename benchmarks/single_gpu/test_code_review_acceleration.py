"""Output ownership checks for the reviewed matrix-product optimization."""
import numpy as np
import pytest

from test_contraction import wrapper


@pytest.mark.parametrize('dtype',[np.float64,np.complex128])
@pytest.mark.parametrize('pattern',['ik,kj->ij','ki,jk->ij'])
def test_matrix_product_reuses_eligible_output(wrapper,monkeypatch,dtype,pattern):
    import cupy as cp
    host=np.arange(35,dtype=float).reshape(7,5)/35
    a=cp.asarray(host,dtype=dtype)
    b=cp.asarray(np.arange(15,dtype=float).reshape(5,3)/15,dtype=dtype)
    if dtype==np.complex128:
        a+=.3j;b-=.2j
    if pattern=='ki,jk->ij':
        a=a.T;b=b.T
    expected=np.einsum(pattern,a.get(),b.get())
    out=cp.empty(expected.shape,dtype=dtype)
    calls=[]
    matmul=cp.matmul
    def observed(left,right,out=None):
        calls.append(out)
        return matmul(left,right,out=out)
    monkeypatch.setattr(cp,'matmul',observed)
    assert wrapper._contract_einsum(pattern,a,b,1.,0.,out) is out
    np.testing.assert_allclose(out.get(),expected,rtol=1e-12,atol=1e-12)
    assert len(calls)==1 and calls[0] is out,'matrix output incurred a temporary/copy'


@pytest.mark.parametrize('destination',['a','b','overlap','F','strided','float32'])
def test_matrix_direct_output_guards(wrapper,monkeypatch,destination):
    import cupy as cp
    a=cp.arange(16,dtype=cp.float64).reshape(4,4)/16
    b=cp.eye(4)*2
    if destination=='a': out=a
    elif destination=='b': out=b
    elif destination=='overlap':
        backing=cp.arange(24,dtype=cp.float64)/24
        a=backing[:16].reshape(4,4);out=backing[8:].reshape(4,4)
    elif destination=='F': out=cp.empty((4,4),order='F')
    elif destination=='strided': out=cp.empty((8,8))[::2,::2]
    else: out=cp.empty((4,4),dtype=cp.float32)
    expected=a.get()@b.get()
    calls=[]
    matmul=cp.matmul
    def observed(left,right,out=None):
        calls.append(out)
        return matmul(left,right,out=out)
    monkeypatch.setattr(cp,'matmul',observed)
    assert wrapper._contract_einsum('ik,kj->ij',a,b,1.,0.,out) is out
    np.testing.assert_allclose(out.get(),expected,rtol=1e-6,atol=1e-7)
    assert len(calls)==1 and calls[0] is None


def test_matrix_direct_output_zero_contraction(wrapper):
    import cupy as cp
    out=cp.ones((3,4))
    assert wrapper._contract_einsum('ik,kj->ij',cp.empty((3,0)),
                                   cp.empty((0,4)),1.,0.,out) is out
    cp.testing.assert_array_equal(out,0.)


def test_grid_weight_reduction_avoids_full_temporary():
    import ast
    from pathlib import Path
    import cupy as cp
    from phase_memory import phase_memory
    path=Path(__file__).resolve().parents[2]/'gpu4pyscf/grad/rks.py'
    tree=ast.parse(path.read_text())
    function,=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name=='get_exc_full_response']
    loop,=[n for n in function.body if isinstance(n,ast.For)
           and isinstance(n.target,ast.Name) and n.target.id=='g0']
    updates=[n for n in loop.body if isinstance(n,ast.AugAssign)
             and isinstance(n.target,ast.Name)
             and n.target.id in ('dweight_dA','de_grid_response_weight')]
    code=compile(ast.Module(body=updates,type_ignores=[]),str(path),'exec')
    host=np.arange(113*3*4096,dtype=float).reshape(113,3,4096)/1e6
    right=np.linspace(-1,1,4096)
    derivative=cp.asarray(host)
    accumulator=cp.zeros((113,3))
    env=dict(cupy=cp,dweight_dA=derivative,dweightdA_right=cp.asarray(right),
             g0=0,g1=len(right),de_grid_response_weight=accumulator)
    # Warm kernels before measuring only the weighted-reduction temporaries.
    exec(code,env)
    derivative[:]=cp.asarray(host);accumulator.fill(0)
    cp.cuda.Device().synchronize()
    with phase_memory() as tracker:
        exec(code,env)
        cp.cuda.Device().synchronize()
    np.testing.assert_allclose(accumulator.get(),np.einsum('Adg,g->Ad',host,right),
                               rtol=1e-12,atol=1e-9)
    assert tracker.data['peak_live_bytes']<derivative.nbytes//8,tracker.data
