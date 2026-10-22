"""Reversible candidates for algorithm experiments, qualified before promotion."""
from contextlib import contextmanager
from unittest.mock import patch
import cupy as cp
from cupy import cublas


def symmetric_gram(x, out):
    """Add real Gram matrices, evaluating one triangle then restoring symmetry."""
    if x.shape[-1] == 0 or x.shape[-2] == 0:
        return out
    if cp.may_share_memory(x, out):
        x = x.copy()
    from gpu4pyscf.lib.cupy_helper import hermi_triu
    for a, result in zip(x.reshape(-1, *x.shape[-2:]), out.reshape(-1, *out.shape[-2:])):
        cublas.syrk('N', a, out=result, beta=1., lower=True)
        hermi_triu(result)
    return out


def adaptive_rho(original, factor=4):
    """GGA cost crossover, including local DM formation; other paths unchanged."""
    from gpu4pyscf.dft import numint

    def evaluate(ao, cpos, xctype, with_lapl=False, buf=None, rho=None):
        nao, nocc = cpos.shape
        if (xctype != 'GGA' or ao.dtype != cp.float64 or cpos.dtype != cp.float64
                or nao >= factor*nocc):
            required=2*nocc*ao.shape[-1]*cpos.itemsize
            if buf is not None and buf.nbytes < required:
                buf=None
            return original(ao, cpos, xctype, with_lapl, buf=buf, rho=rho)
        dm=cpos@cpos.T
        if buf is not None and buf.nbytes < nao*ao.shape[-1]*8:
            buf=None
        value=numint.eval_rho(None, ao, dm, xctype='GGA', hermi=1, buf=buf)
        if rho is None:
            return value
        rho[:]=value
        return rho
    return evaluate


@contextmanager
def algorithm_version(version):
    """Temporary single-GPU, real ground-state algorithm substitution."""
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft import numint
    if version == 'original':
        yield
    elif version == 'density':
        with patch.object(numint, '_eval_rho2', adaptive_rho(numint._eval_rho2)):
            yield
    elif version == 'exchange':
        original=df_jk.contract
        def contract(pattern, a, b, alpha=1., beta=0., out=None):
            if (pattern=='nikL,njkL->nij' and a is b and alpha==beta==1.
                    and a.dtype==cp.float64 and a.flags.c_contiguous
                    and a.shape[1] >= 512
                    and out is not None and out.dtype==cp.float64 and out.flags.c_contiguous):
                return symmetric_gram(a.reshape(*a.shape[:2],-1),out)
            return original(pattern,a,b,alpha,beta,out)
        with patch.object(df_jk,'contract',contract):
            yield
    else:
        raise ValueError(version)
