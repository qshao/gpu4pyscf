"""Benchmark-only direct contractions and bounded gradient workspaces."""
import cupy as cp
import numpy as np
from cupy import cublas
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from types import FunctionType
from unittest.mock import patch


def coulomb(pattern,a,b,out,alpha=1.,beta=0.):
    """Single-density Coulomb products with direct BLAS accumulation."""
    from cupy_backends.cuda.libs import cublas as backend
    from cupy.cuda import device
    forward=pattern=='np,Lp->nL'
    if b.strides[0]==8:
        m,n=b.shape; lda=b.strides[1]//8
        trans=backend.CUBLAS_OP_N if forward else backend.CUBLAS_OP_T
    else:
        n,m=b.shape; lda=b.strides[0]//8
        trans=backend.CUBLAS_OP_T if forward else backend.CUBLAS_OP_N
    handle=device.get_cublas_handle(); previous=backend.getPointerMode(handle)
    alpha,beta=np.array(alpha,dtype=np.float64),np.array(beta,dtype=np.float64)
    try:
        backend.setPointerMode(handle,backend.CUBLAS_POINTER_MODE_HOST)
        backend.dgemv(handle,trans,m,n,alpha.ctypes.data,b.data.ptr,lda,
                      a.data.ptr,1,beta.ctypes.data,out.data.ptr,1)
    finally: backend.setPointerMode(handle,previous)
    return out


def matrix_layout(b):
    return ((b.strides[0]==8 and b.strides[1]>=max(1,b.shape[0])*8)
            or (b.strides[1]==8 and b.strides[0]>=max(1,b.shape[1])*8))


def gradient(pattern,a,b,out,alpha=1.,beta=0.):
    """Direct output GEMM, packing only padded tensor input when necessary."""
    matrix=cp.ascontiguousarray(a).reshape(a.shape[0],-1)
    target=out.reshape(out.shape[0],-1)
    trans='T' if pattern=='pqr,pi->iqr' else 'N'
    cublas.gemm(trans,'N',b,matrix,out=target,alpha=alpha,beta=beta)
    return out


@contextmanager
def baseline():
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.df.grad import rhf
    from gpu4pyscf.lib import cutensor
    snapshot=json.loads(Path(__file__).with_name('speed_memory_baseline.json').read_text())
    modules=dict(df_jk=df_jk,df_grad_rhf=rhf,cutensor=cutensor)
    with ExitStack() as stack:
        for label,row in snapshot['modules'].items():
            module=modules[label]
            for name,source in row['functions'].items():
                scope=dict(module.__dict__)
                exec(compile(source,'pre_speed_memory_'+label+'.py','exec'),scope)
                f=scope[name]
                restored=FunctionType(f.__code__,module.__dict__,name,f.__defaults__)
                stack.enter_context(patch.object(module,name,restored))
        yield


@contextmanager
def version(name):
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.df.grad import rhf
    from gpu4pyscf.lib import cutensor
    assert name in ['current','contractions','compact','combined']
    with baseline(),ExitStack() as stack:
        if name in ['compact','combined']:
            snapshot=json.loads(Path(__file__).with_name('speed_memory_baseline.json').read_text())
            source=snapshot['modules']['df_grad_rhf']['functions']['_jk_energy_per_atom']
            batch='    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8))))'
            dense='    blksize = max(1, min(naux, int(mem_avail*.4/(nao*nao*2*8))//8*8))'
            scratch='    buf2 = cp.empty((blksize, nao, nao))'
            assert source.count(batch)==source.count(dense)==source.count(scratch)==1
            source=source.replace(batch,batch+'\n    batch_size = min(batch_size, 128)')
            source=source.replace(dense,dense+'\n    blksize = min(blksize, 128)')
            source=source.replace(scratch,'    buf2 = cp.empty((blksize, nocc if hermi == 1 else nao, nao))')
            scope=dict(rhf.__dict__)
            exec(compile(source,'compact_gradient_candidate.py','exec'),scope)
            f=scope['_jk_energy_per_atom']
            restored=FunctionType(f.__code__,rhf.__dict__,f.__name__,f.__defaults__)
            stack.enter_context(patch.object(rhf,f.__name__,restored))
        if name in ['contractions','combined']:
            original_jk,original_grad=df_jk.contract,rhf.contract
            def jk(pattern,a,b,alpha=1.,beta=0.,out=None):
                if (df_jk.contract_engine=='cupy' and pattern in ['np,Lp->nL','nL,Lp->np']
                        and a.ndim==b.ndim==2 and a.shape[0]==1
                        and a.size>0 and b.size>0
                        and a.shape[1]==b.shape[1 if pattern=='np,Lp->nL' else 0]
                        and a.dtype==b.dtype==cp.float64 and a.flags.c_contiguous and matrix_layout(b)):
                    shape=(1,b.shape[0 if pattern=='np,Lp->nL' else 1])
                    if out is None:
                        out=cp.empty(shape,dtype=cp.float64)
                        beta=0.
                    if (out.shape==shape and out.dtype==cp.float64 and out.flags.c_contiguous
                            and not cp.may_share_memory(a,out) and not cp.may_share_memory(b,out)):
                        return coulomb(pattern,a,b,out,alpha,beta)
                return original_jk(pattern,a,b,alpha=alpha,beta=beta,out=out)
            def grad(pattern,a,b,alpha=1.,beta=0.,out=None):
                if (cutensor.contract_engine=='cupy' and pattern in ['pqr,pi->iqr','iqr,pi->pqr']
                        and a.ndim==3 and b.ndim==2 and out is not None
                        and a.size>0 and b.size>0
                        and a.shape[0]==b.shape[0 if pattern=='pqr,pi->iqr' else 1]
                        and out.shape==(b.shape[1 if pattern=='pqr,pi->iqr' else 0],*a.shape[1:])
                        and a.dtype==b.dtype==out.dtype==cp.float64 and out.flags.c_contiguous
                        and not cp.may_share_memory(a,out) and not cp.may_share_memory(b,out)):
                    return gradient(pattern,a,b,out,alpha,beta)
                return original_grad(pattern,a,b,alpha=alpha,beta=beta,out=out)
            stack.enter_context(patch.object(df_jk,'contract',jk))
            stack.enter_context(patch.object(rhf,'contract',grad))
        yield
