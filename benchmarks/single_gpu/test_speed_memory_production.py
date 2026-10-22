"""Allocation bounds and production contraction correctness."""
import pytest
import cupy as cp


@pytest.mark.parametrize('nao,nocc,npair',[(565,100,110000),(2000,400,2000000),(4000,800,8000000)])
@pytest.mark.parametrize('hermi',[0,1])
def test_workspace_bound(nao,nocc,npair,hermi):
    from gpu4pyscf.df.grad import rhf
    assert hasattr(rhf,'_gradient_workspace_sizes'), 'bounded gradient workspace missing'
    budget=1024**3
    batch,block=rhf._gradient_workspace_sizes(nao,nocc,10000,npair,hermi,6,budget)
    # Conservative allowance for explicit buffers and contraction packing/output
    # temporaries; persistent auxiliary tensors and library workspaces excluded.
    scratch=nocc if hermi==1 else max(nao,nocc)
    dense=2*nao**2+nao*scratch+nao*nocc+2*nocc**2
    assert batch>=6 and block>=1
    assert 8*(batch*npair+block*dense)<=budget


def test_workspace_budget_too_small():
    from gpu4pyscf.df.grad import rhf
    assert hasattr(rhf,'_gradient_workspace_sizes'), 'bounded gradient workspace missing'
    with pytest.raises(MemoryError,match='workspace'):
        rhf._gradient_workspace_sizes(4000,800,10000,8000000,1,36,1024**2)


def test_gradient_keeps_complete_auxiliary_shell(monkeypatch):
    from gpu4pyscf.df.grad import rhf
    import numpy as np
    import sys
    from types import SimpleNamespace
    nao=565
    mol=SimpleNamespace(stdout=sys.stdout,verbose=0)
    aux=SimpleNamespace(nao=100,uniq_l_ctr=np.array([[4,1]]))
    class Planned(Exception): pass
    def evaluator(**kwargs):
        assert kwargs['aux_batch_size']>=15, 'old memory fraction truncated a complete auxiliary shell'
        raise Planned
    opt=SimpleNamespace(mol=mol,auxmol=aux,int3c2e_evaluator=evaluator,
        pair_and_diag_indices=lambda **kwargs:(np.arange(150000),np.array([])))
    coeff=cp.empty((nao,1))
    monkeypatch.setattr(rhf,'_factorize_dm',lambda *args:(coeff,coeff))
    monkeypatch.setattr(rhf,'_check_rsh_factors',lambda *args:(0,None,None))
    monkeypatch.setattr(rhf,'get_avail_mem',lambda **kwargs:33_000_000+100*8+nao*nao*8)
    with pytest.raises(Planned): rhf._jk_energy_per_atom(opt,None)


def test_production_direct_coulomb():
    from gpu4pyscf.df import df_jk
    assert hasattr(df_jk,'_coulomb_contract'), 'production direct Coulomb missing'
    a=cp.arange(11,dtype=cp.float64)[None]
    raw=cp.empty((13,11),order='F'); b=raw[:5]
    b[:]=cp.arange(55,dtype=cp.float64).reshape(5,11)/55
    expected=cp.einsum('np,Lp->nL',a,b)
    actual=df_jk._coulomb_contract('np,Lp->nL',a,b)
    cp.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)


def test_production_direct_gradient():
    from gpu4pyscf.df.grad import rhf
    assert hasattr(rhf,'_gradient_contract'), 'production direct gradient missing'
    a=cp.arange(3*7*5,dtype=cp.float64).reshape(3,7,5)/100
    b=cp.arange(11*3,dtype=cp.float64).reshape(11,3)/100
    out=cp.ones((11,7,5))
    expected=-.2*cp.einsum('iqr,pi->pqr',a,b)+.7*out
    rhf._gradient_contract('iqr,pi->pqr',a,b,-.2,.7,out=out)
    cp.testing.assert_allclose(out,expected,rtol=2e-12,atol=2e-12)


def test_open_shell_gradient_with_small_workspace(monkeypatch):
    from gpu4pyscf.df.grad import uhf
    assert hasattr(uhf,'_gradient_workspace_sizes'), 'open-shell workspace bounds missing'
    from pyscf import gto,dft,lib
    planner=uhf._gradient_workspace_sizes
    calls=[]
    def bounded(*args):
        batch,block=planner(*args)
        calls.append(True)
        return min(batch,48),min(block,8)
    monkeypatch.setattr(uhf,'_gradient_workspace_sizes',bounded)
    with lib.with_omp_threads(2):
        mol=gto.M(atom='O 0 0 0; H .94 0 0; H -.24 0 .91',
                  basis='def2-svp',charge=1,spin=1,verbose=0)
        cpu=dft.UKS(mol,xc='B3LYP').density_fit(auxbasis='def2-universal-jkfit')
        cpu.grids.level=1; cpu.grids.prune=None; cpu.small_rho_cutoff=0.
        cpu.conv_tol=1e-10; cpu.conv_tol_grad=1e-7
        cpu.max_cycle=100
        dm0=cpu.get_init_guess()
        gpu=cpu.to_gpu()
        e_cpu=cpu.kernel(dm0=dm0); e_gpu=gpu.kernel(dm0=dm0)
        assert cpu.converged and gpu.converged
        cpu_grad=cpu.nuc_grad_method(); gpu_grad=gpu.nuc_grad_method()
        cpu_grad.grid_response=gpu_grad.grid_response=True
        cpu_grad.auxbasis_response=gpu_grad.auxbasis_response=True
        reference=cpu_grad.kernel(); actual=cp.asnumpy(gpu_grad.kernel())
    assert calls
    assert abs(e_cpu-e_gpu)<1e-8
    import numpy as np
    np.testing.assert_allclose(actual,reference,rtol=0,atol=1e-6)
