"""Benchmark-only exchange transformations exploiting unpacked DF strides."""
import cupy as cp
import numpy as np
from contextlib import contextmanager
from unittest.mock import patch


def eligible(a, b, out):
    if not (a.ndim == b.ndim == 3 and out.ndim == 4): return False
    naux, ni, nj = a.shape
    batch, nj_b, nocc = b.shape
    return (a.dtype == b.dtype == out.dtype == cp.float64
            and ni == nj == nj_b and out.shape == (batch,ni,nocc,naux)
            and b.flags.c_contiguous and out.flags.c_contiguous
            and a.strides[0] == 8 and a.strides[2] >= max(naux,1)*8
            and a.strides[1] == nj*a.strides[2]
            and not cp.may_share_memory(a,out) and not cp.may_share_memory(b,out))


def worthwhile(a):
    # Partial blocks lose to the existing contraction on the qualified GB10.
    return a.shape[1] >= 512 and a.shape[0] >= 256


def transform(a, b, out):
    """Compute Lij,njk->nikL for padded auxiliary-contiguous real DF blocks."""
    naux, ni, nj = a.shape
    batch, _, nocc = b.shape
    if out.size == 0: return out
    assert eligible(a,b,out)
    from cupy_backends.cuda.libs import cublas
    from cupy.cuda import device
    handle = device.get_cublas_handle()
    previous = cublas.getPointerMode(handle)
    one, zero = np.array(1.,dtype=np.float64), np.array(0.,dtype=np.float64)
    try:
        cublas.setPointerMode(handle,cublas.CUBLAS_POINTER_MODE_HOST)
        for n in range(batch):
            # Each AO i supplies one column-major (L,j) matrix. Reuse B
            # across the AO batches and write column-major (L,k) in place.
            cublas.dgemmStridedBatched(handle, cublas.CUBLAS_OP_N,
                cublas.CUBLAS_OP_T, naux, nocc, nj, one.ctypes.data,
                a.data.ptr, a.strides[2]//8, a.strides[1]//8,
                b[n].data.ptr, nocc, 0, zero.ctypes.data,
                out[n].data.ptr, naux, nocc*naux, ni)
    finally: cublas.setPointerMode(handle,previous)
    return out


@contextmanager
def version():
    """Reversibly intercept the single qualified contraction pattern."""
    from gpu4pyscf.df import df_jk
    original = df_jk.contract
    def contract(pattern, a, b, alpha=1., beta=0., out=None):
        if (pattern == 'Lij,njk->nikL' and df_jk.contract_engine == 'cupy'
                and alpha == 1. and beta == 0. and out is not None
                and worthwhile(a) and eligible(a,b,out)):
            return transform(a,b,out)
        return original(pattern,a,b,alpha=alpha,beta=beta,out=out)
    with patch.object(df_jk,'contract',contract): yield
