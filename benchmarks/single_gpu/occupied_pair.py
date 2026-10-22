"""Benchmark-only exact occupied-pair symmetry candidate; not a default path."""
from contextlib import contextmanager
import inspect
import types
from unittest.mock import patch

import cupy as cp
import numpy as np


def pack_oo(full,i,j,out):
    """Retain the symmetric component, avoiding a one-sided roundoff bias.

    Both gathers fit the workspace planner's extra occupied-pair allowance.
    The existing output avoids an additional arithmetic temporary.
    """
    cp.add(full[:,i,j],full[:,j,i],out=out)
    out*=.5
    return out


def response_gram(packed,diagonal,alpha=1.,beta=0.,out=None):
    """Accumulate the full symmetric-matrix Gram from its packed lower half.

    An existing nonalias output must be symmetric. Raw SYRK consumes C storage
    through transpose flags, avoiding a full packed C-to-F copy.
    """
    if packed.ndim!=2 or diagonal.ndim!=1:
        raise ValueError('Expected matrix and diagonal-index vector shapes')
    shape=(packed.shape[0],packed.shape[0])
    if out is not None and out.shape!=shape:
        raise ValueError('Output shape must be '+str(shape))
    d=cp.ascontiguousarray(packed[:,diagonal])
    if out is None:
        out=cp.empty(shape,dtype=packed.dtype)
        beta=0.
    if (packed.dtype!=cp.float64 or out.dtype!=cp.float64
            or not packed.flags.c_contiguous or not out.flags.c_contiguous
            or packed.size==0 or cp.may_share_memory(packed,out)
            or not np.isrealobj(alpha) or not np.isrealobj(beta)):
        value=alpha*(2*(packed@packed.T)-d@d.T)
        out[:]=value if beta==0 else value+beta*out
        return out
    from cupy_backends.cuda.libs import cublas
    from cupy.cuda import device
    from gpu4pyscf.lib.cupy_helper import hermi_triu
    handle=device.get_cublas_handle()
    previous=cublas.getPointerMode(handle)
    first=np.array(2*alpha,dtype=np.float64)
    second=np.array(-alpha,dtype=np.float64)
    accum=np.array(beta,dtype=np.float64)
    one=np.array(1.,dtype=np.float64)
    n=packed.shape[0]
    try:
        cublas.setPointerMode(handle,cublas.CUBLAS_POINTER_MODE_HOST)
        cublas.dsyrk(handle,cublas.CUBLAS_FILL_MODE_UPPER,cublas.CUBLAS_OP_T,
                     n,packed.shape[1],first.ctypes.data,packed.data.ptr,
                     packed.shape[1],accum.ctypes.data,out.data.ptr,n)
        cublas.dsyrk(handle,cublas.CUBLAS_FILL_MODE_UPPER,cublas.CUBLAS_OP_T,
                     n,d.shape[1],second.ctypes.data,d.data.ptr,d.shape[1],
                     one.ctypes.data,out.data.ptr,n)
    finally:
        cublas.setPointerMode(handle,previous)
    hermi_triu(out)
    return out


def candidate_source(source,minimum_nao=512,minimum_nocc=64):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    replace("    nao, nocc = dm_factor_l.shape\n",
        f"    nao, nocc = dm_factor_l.shape\n"
        f"    packed_oo = (hermi == 1 and dm_factor_l is dm_factor_r and nao >= {minimum_nao}\n"
        f"                 and nocc >= {minimum_nocc} and dm_factor_l.dtype == cp.float64\n"
        f"                 and contract_engine == 'cupy')\n")
    replace("    j3c_oo = cp.empty((naux, nocc, nocc))\n",
        "    if packed_oo:\n"
        "        oi, oj = cp.tril_indices(nocc)\n"
        "        diagonal = cp.cumsum(cp.arange(1,nocc+1))-1\n"
        "        j3c_oo = cp.empty((naux,len(oi)))\n"
        "        oo_work = cp.empty((blksize,nocc,nocc))\n"
        "    else:\n"
        "        j3c_oo = cp.empty((naux,nocc,nocc))\n")
    replace("            contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])\n",
        "            if packed_oo:\n"
        "                contract('iqr,qj->rij', tmp, dm_factor_l, out=oo_work[:dk])\n"
        "                _occupied_pack(oo_work[:dk],oi,oj,out=j3c_oo[aux0:aux1])\n"
        "            else:\n"
        "                contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])\n")
    replace("    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None\n",
        "    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = oo_work = None\n")
    replace("        auxvec = dm_oo.trace(axis1=1, axis2=2)\n",
        "        auxvec = dm_oo[:,diagonal].sum(axis=1) if packed_oo else dm_oo.trace(axis1=1, axis2=2)\n")
    replace("    ejk_aux = cp.zeros((mol.natm, 3))\n",
        "    ejk_aux = cp.zeros((mol.natm, 3))\n"
        "    if packed_oo:\n"
        "        oo_work = cp.empty((blksize,nocc,nocc))\n")
    replace("            contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)\n",
        "            block_oo = (_occupied_unpack(dm_oo[aux0:aux1],out=oo_work)\n"
        "                        if packed_oo else dm_oo[aux0:aux1])\n"
        "            contract('rji,qj->iqr', block_oo, dm_factor_l, out=tmp)\n")
    replace("    buf = buf1 = buf2 = compressed = dm_tensor = dm_tensor1 = tmp = None\n",
        "    buf = buf1 = buf2 = compressed = dm_tensor = dm_tensor1 = tmp = oo_work = block_oo = None\n")
    replace("        if dm_factor_l is dm_factor_r:\n",
        "        if packed_oo:\n"
        "            dm_aux = _occupied_response(dm_oo,diagonal,\n"
        "                        alpha=-.5*k_factor,beta=j_factor,out=dm_aux)\n"
        "        elif dm_factor_l is dm_factor_r:\n")
    return source


@contextmanager
def packed_gradient(minimum_nao=512,minimum_nocc=64):
    from gpu4pyscf.df.grad import rhf
    from gpu4pyscf.lib.cupy_helper import unpack_tril
    original=rhf._jk_energy_per_atom
    source=candidate_source(inspect.getsource(original),minimum_nao,minimum_nocc)
    namespace=dict(rhf.__dict__)
    exec(compile(source,'<packed occupied-pair candidate>','exec'),namespace)
    compiled=namespace['_jk_energy_per_atom']
    candidate=types.FunctionType(compiled.__code__,rhf.__dict__,
                                 compiled.__name__,compiled.__defaults__)
    with patch.dict(rhf.__dict__,_occupied_response=response_gram,_occupied_unpack=unpack_tril,_occupied_pack=pack_oo), \
         patch.object(rhf,'_jk_energy_per_atom',candidate):
        yield
    assert rhf._jk_energy_per_atom is original
