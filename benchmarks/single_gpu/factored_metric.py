"""Diagnostic metric action through factors, without an inverse-like matrix."""
from contextlib import contextmanager
import inspect
from unittest.mock import patch

import cupy as cp


def apply_metric(coeff,solve,rhs,columns=256):
    flat=rhs.reshape(rhs.shape[0],-1)
    out=cp.empty_like(rhs); target=out.reshape(flat.shape)
    for start in range(0,flat.shape[1],columns):
        end=min(start+columns,flat.shape[1])
        target[:,start:end]=coeff@solve(coeff.T@flat[:,start:end])
    return out


@contextmanager
def factored_metric():
    from gpu4pyscf.df.grad import rhf
    original=rhf._jk_energy_per_atom
    old="""    if mol.omega <= 0 and not auxmol.mol.cart:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'CD')(aux_coeff.T))
    else:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
    j2c = aux_coeff = None
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    metric = j3c_oo = None
"""
    new="""    if mol.omega <= 0 and not auxmol.mol.cart:
        solve = _gen_metric_solver(j2c, 'CD')
    else:
        solve = _gen_metric_solver(j2c, 'ED')
    dm_oo = _factored_apply(aux_coeff, solve, j3c_oo)
    solve = j2c = aux_coeff = j3c_oo = None
"""
    source=inspect.getsource(original)
    assert source.count(old)==1
    namespace=dict(original.__globals__,_factored_apply=apply_metric)
    exec(compile(source.replace(old,new),'<factored-metric>','exec'),namespace)
    with patch.object(rhf,'_jk_energy_per_atom',namespace[original.__name__]):
        yield
