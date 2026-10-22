"""CPU planner checks plus native small-molecule DF build regressions."""
import ast
from pathlib import Path

import pytest


def planner():
    source=Path(__file__).resolve().parents[2]/'gpu4pyscf/df/df.py'
    tree=ast.parse(source.read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
           and node.name=='_resident_build_batch_size']
    assert len(nodes)==1,'resident build workspace planner missing'
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['_resident_build_batch_size']


@pytest.mark.parametrize('extra',[False,True])
def test_resident_scratch_bound_and_existing_small_batches(extra):
    choose=planner()
    budget=4*1024**3
    rows=choose(1_000_000,8000,budget,extra)
    assert rows*8000*8*(1+extra)<=budget
    assert (rows+1)*8000*8*(1+extra)>budget
    assert choose(17,8000,budget,extra)==17


def test_resident_minimum_and_invalid_budgets():
    choose=planner()
    assert choose(0,100,800)==1
    with pytest.raises(MemoryError,match='workspace'):
        choose(10,100,799)
    with pytest.raises(MemoryError,match='workspace'):
        choose(10,100,1599,True)
    for budget in [0,-1]:
        with pytest.raises(ValueError,match='positive'):
            choose(10,100,budget)


@pytest.mark.special
@pytest.mark.parametrize('basis',['def2-svp','cc-pvdz'])
def test_bounded_resident_integrals_and_jk_match_control(basis,monkeypatch):
    import cupy as cp
    import numpy as np
    from pyscf import gto
    from gpu4pyscf import __config__
    from gpu4pyscf.df import df
    from test_resident_workspace_run import control_builder
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis=basis,verbose=0)
    monkeypatch.setattr(__config__,'df_cderi_build_workspace_bytes',1024**2,raising=False)
    old=control_builder(df)
    planned=[]
    choose=df._resident_build_batch_size
    def observe(*args,**kwargs):
        result=choose(*args,**kwargs)
        planned.append(result)
        return result
    monkeypatch.setattr(df,'_resident_build_batch_size',observe)
    reference=df.DF(mol,'weigend');reference.use_gpu_memory=True
    with monkeypatch.context() as patch:
        patch.setattr(df,'_cholesky_eri',old)
        reference.build()
    candidate=df.DF(mol,'weigend');candidate.use_gpu_memory=True
    try:
        candidate.build()
        assert bool(planned)==(basis=='def2-svp')
        assert candidate._cderi[0].shape==reference._cderi[0].shape
        cp.testing.assert_allclose(candidate._cderi[0],reference._cderi[0],atol=1e-10,rtol=0)
        cp.testing.assert_array_equal(candidate._cderi_idx[0],reference._cderi_idx[0])
        from pyscf.scf.hf import get_init_guess
        density=cp.asarray(get_init_guess(mol))
        a=reference.get_jk(density,hermi=1)
        b=candidate.get_jk(density,hermi=1)
        for x,y in zip(a,b):
            np.testing.assert_allclose(cp.asnumpy(x),cp.asnumpy(y),atol=1e-9,rtol=0)
    finally:
        reference.reset();candidate.reset()


@pytest.mark.special
def test_native_minimum_shell_batch_rejected_before_metric(monkeypatch):
    from pyscf import gto
    from pyscf.df.addons import make_auxmol
    from gpu4pyscf import __config__
    from gpu4pyscf.df import df
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
              basis='def2-svp',verbose=0)
    auxmol=make_auxmol(mol,'weigend')
    intopt=df.int3c2e_bdiv.Int3c2eOpt(mol,auxmol)
    intopt.build()
    # One AO pair fits, but the native evaluator's shell group is indivisible.
    budget=intopt.auxmol.nao*8
    monkeypatch.setattr(__config__,'df_cderi_build_workspace_bytes',budget,raising=False)
    evaluated=[]
    original=intopt.int3c2e_evaluator
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        evaluated.append((kwargs['ao_pair_batch_size'],max(result[2][1:]-result[2][:-1])))
        return result
    monkeypatch.setattr(intopt,'int3c2e_evaluator',observed)
    def forbidden(*args,**kwargs):
        pytest.fail('metric/output allocation reached despite impossible build budget')
    monkeypatch.setattr(df,'_decompose_j2c',forbidden)
    with pytest.raises(MemoryError,match='minimum shell-pair build workspace'):
        df._cholesky_eri(intopt,use_gpu_memory=True)
    assert len(evaluated)==1 and evaluated[0][0]==1 and evaluated[0][1]>1
