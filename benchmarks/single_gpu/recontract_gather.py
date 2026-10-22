"""Benchmark-only FP64 coalesced gather for primitive-pair recontraction."""
from collections import namedtuple
from contextlib import contextmanager
import inspect
import types
from unittest.mock import patch

import cupy as cp
import numpy as np


Mapping=namedtuple('Mapping','indices coefficients offsets rows required_input_rows')
KERNEL=cp.RawKernel(r'''
extern "C" __global__ void recontract_coalesced(
    double *out, const double *input, const int *indices,
    const double *coef, const int *offsets, long long width, long long total)
{
    for (long long flat=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         flat<total; flat+=(long long)gridDim.x*blockDim.x) {
        long long row=flat/width;
        long long col=flat-row*width;
        double value=0.;
        for (int k=offsets[row]; k<offsets[row+1]; ++k)
            value+=coef[k]*input[(long long)indices[k]*width+col];
        out[flat]=value;
    }
}
''','recontract_coalesced',options=('--std=c++11','--fmad=false'))


def csr_mapping(inp,out,coef,rows):
    inp,out,coef=np.asarray(inp),np.asarray(out),np.asarray(coef)
    if (inp.ndim!=1 or inp.shape!=out.shape or inp.shape!=coef.shape
            or inp.dtype.kind not in 'iu' or out.dtype.kind not in 'iu'
            or np.iscomplexobj(coef) or rows<0):
        raise ValueError('Invalid recontraction mapping')
    limit=np.iinfo(np.int32).max
    if (len(inp)>limit or rows>limit or (len(inp) and
        (inp.min()<0 or inp.max()>limit or out.min()<0 or out.max()>=rows))):
        raise ValueError('Recontraction mapping indices are out of bounds')
    order=np.argsort(out,kind='stable')
    offsets=np.append(0,np.cumsum(np.bincount(out.astype(np.int64),minlength=rows)))
    return Mapping(cp.asarray(inp[order],dtype=cp.int32),
                   cp.asarray(coef[order],dtype=cp.float64),
                   cp.asarray(offsets,dtype=cp.int32),int(rows),
                   int(inp.max())+1 if len(inp) else 0)


def gather(data,mapping,out=None):
    if data.ndim!=2 or data.dtype!=cp.float64 or not data.flags.c_contiguous:
        raise ValueError('Recontraction input requires C-contiguous FP64 matrix')
    if data.shape[0]<mapping.required_input_rows:
        raise ValueError('Recontraction input has too few rows')
    shape=(mapping.rows,data.shape[1])
    if out is None: out=cp.empty(shape,dtype=cp.float64)
    if out.shape!=shape or out.dtype!=cp.float64 or not out.flags.c_contiguous:
        raise ValueError('Invalid recontraction output buffer')
    if cp.may_share_memory(data,out): raise ValueError('Recontraction input/output alias')
    if out.size:
        KERNEL((min(65535,(out.size+255)//256),),(256,),
               (out,data,mapping.indices,mapping.coefficients,mapping.offsets,
                np.int64(data.shape[1]),np.int64(out.size)))
    return out


@contextmanager
def coalesced_recontraction():
    from gpu4pyscf.df import int3c2e_bdiv as module
    if module.multi_gpu.num_devices!=1:
        raise NotImplementedError('Recontraction candidate requires a single GPU')
    original=module._create_pair_recontractor
    source=inspect.getsource(original)
    old='''        recontraction_params.append(
            (inp_idx[:count].copy(), out_idx[:count].copy(), coef[:count].copy()))'''
    assert source.count(old)==1
    source=source.replace(old,'''        recontraction_params.append(
            _gather_mapping(inp_idx[:count],out_idx[:count],coef[:count],cderi_npairs))''')
    first=source.index('    def recontract(')
    last=source.index('    return recontract,',first)
    source=source[:first]+'''    def recontract(batch_id,j3c,out=None):
        shape=(contracted_ao_pair_counts[batch_id],j3c.shape[1])
        target=ndarray(shape,buffer=out)
        return _coalesced_gather(j3c,recontraction_params[batch_id],out=target)
'''+source[last:]
    namespace=dict(module.__dict__)
    exec(compile(source,'<coalesced recontraction candidate>','exec'),namespace)
    compiled=namespace[original.__name__]
    candidate=types.FunctionType(compiled.__code__,module.__dict__,
                                 compiled.__name__,compiled.__defaults__)
    with patch.dict(module.__dict__,_gather_mapping=csr_mapping,_coalesced_gather=gather), \
         patch.object(module,'_create_pair_recontractor',candidate):
        yield
    assert module._create_pair_recontractor is original
