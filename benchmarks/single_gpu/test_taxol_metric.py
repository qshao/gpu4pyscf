"""Test a single metric-application hypothesis against CPU derivatives."""
import json
import time
import traceback
from unittest.mock import patch

import numpy as np
import pytest

from comprehensive import BASE, ROOT, RESULTS, SOURCES, save
from molecular import _file_identity, _metadata


@pytest.mark.special
@pytest.mark.benchmark
def test_factored_taxol_metric():
    import cupy as cp
    import scipy.linalg
    from pyscf import lib
    from gpu4pyscf.df.grad import rhf
    from gradient_snapshot import restore
    from factored_metric import factored_metric
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    parent=json.loads(parent_path.read_text())
    cpu_path=BASE/'accuracy_results/taxol_cpu_fixed_components.json'
    cpu=json.loads(cpu_path.read_text()); assert cpu['status']=='captured'
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    output=BASE/'accuracy_results/taxol_factored_metric.json'
    if output.exists(): raise FileExistsError('Archive existing hypothesis test explicitly')
    report=dict(_metadata('gpu'),status='running',
        parent_sha256=_file_identity(parent_path)['sha256'],cpu_sha256=_file_identity(cpu_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        candidate_sha256=_file_identity(BASE/'factored_metric.py')['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        note='Single hypothesis: apply the auxiliary metric through factors/solves, rather than constructing its inverse-like projected matrix. Same orbitals, density, native integrals, screening and quantum settings. CPU J/K derivative oracle includes auxiliary response. Matrix condition and CPU/GPU integral comparison are diagnostics only; no speed claim.')
    save(output,report)
    original_integral=rhf.int2c2e; original_solver=rhf._gen_metric_solver
    def integral(auxmol,*args,**kwargs):
        matrix=original_integral(auxmol,*args,**kwargs)
        reference=auxmol.mol.intor('int2c2e')
        host=cp.asnumpy(matrix)
        assert host.shape==reference.shape
        norm=np.linalg.norm(host,1)
        chol=scipy.linalg.cho_factor(host,lower=True)
        reciprocal,info=scipy.linalg.lapack.dpocon(chol[0],norm,uplo='L')
        assert info==0
        report['metric']=dict(shape=list(host.shape),max_cpu_gpu_difference=float(np.max(np.abs(host-reference))),
                              condition_estimate_1norm=float(1/reciprocal))
        print('Taxol metric',report['metric'],flush=True)
        return matrix
    def solver(*args,**kwargs):
        result=original_solver(*args,**kwargs)
        report['solver_closure']=list(result.__code__.co_freevars)
        return result
    try:
        with lib.with_omp_threads(4):
            mf=restore('taxol_svp_b3lyp',parent['orbitals']['path'])
            hybrid=mf._numint.rsh_and_hybrid_coeff(mf.xc,spin=mf.mol.spin)[2]
            grad=mf.nuc_grad_method(); grad.auxbasis_response=True
            begin=time.perf_counter()
            # Install these before compiling so its diagnostic globals agree.
            with patch.object(rhf,'int2c2e',integral),patch.object(rhf,'_gen_metric_solver',solver),factored_metric():
                actual=grad.jk_energy_per_atom(mf.make_rdm1(),j_factor=1.,k_factor=hybrid,hermi=1)
            actual=np.asarray(actual)
            error=float(np.max(np.abs(actual-np.asarray(cpu['jk']))))
            report.update(seconds=time.perf_counter()-begin,jk=actual.tolist(),cpu_error=error,
                          status='captured',meets_force_gate=(error<=1e-6))
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('Factored Taxol metric CPU force error',error,flush=True)
