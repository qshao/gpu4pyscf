"""Separate gradient arithmetic from SCF/grid effects after a failed large gate."""
import json
import numpy as np
import pytest

from comprehensive import BASE,ROOT,RESULTS,SOURCES,save
from molecular import _metadata,_file_identity


@pytest.mark.special
@pytest.mark.benchmark
def test_taxol_fixed_orbital_jk():
    from pyscf import lib
    from gpu4pyscf.df.grad import rhf
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    parent=json.loads(parent_path.read_text())
    assert parent['status']=='failed' and parent['error_type']=='AssertionError'
    assert parent['result']['converged']
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    output=RESULTS/'taxol_fixed_orbital_jk_diagnostic.json'
    if output.exists(): raise FileExistsError('Archive existing diagnostic explicitly')
    report=dict(_metadata('gpu'),status='running',
        parent_sha256=_file_identity(parent_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        helper_hashes={p:_file_identity(BASE/p)['sha256'] for p in ['gradient_snapshot.py','occupied_pair.py']},
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        note='Diagnostic only: two original DF J/K gradients and one packed gradient at identical converged disk-run orbitals. Excludes XC/grid response, one-electron and nuclear terms. Does not qualify the failed full-workflow comparison.')
    save(output,report)
    original=rhf._jk_energy_per_atom
    with lib.with_omp_threads(4):
        mf=restore('taxol_svp_b3lyp',parent['orbitals']['path'])
        omega,alpha,hybrid=mf._numint.rsh_and_hybrid_coeff(mf.xc,spin=mf.mol.spin)
        assert omega==0
        def run():
            grad=mf.nuc_grad_method()
            grad.auxbasis_response=True
            return grad.jk_energy_per_atom(mf.make_rdm1(),j_factor=1.,k_factor=hybrid,hermi=1)
        first=run(); second=run()
        with packed_gradient(): candidate=run()
    assert rhf._jk_energy_per_atom is original
    repeat_error=float(np.max(np.abs(first-second)))
    candidate_error=float(np.max(np.abs(first-candidate)))
    report.update(baseline_repeat_error=repeat_error,packed_error=candidate_error,
                  first=first.tolist(),second=second.tolist(),packed=candidate.tolist(),
                  status='captured')
    save(output,report)
    print('fixed orbital J/K repeat error',repeat_error,'packed error',candidate_error,flush=True)


@pytest.mark.special
@pytest.mark.benchmark
def test_taxol_fixed_orbital_full():
    from pyscf import lib
    from gradient_snapshot import restore
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    resident_path=RESULTS/'taxol_svp_b3lyp_capacity_resident_baseline.json'
    parent=json.loads(parent_path.read_text())
    resident=json.loads(resident_path.read_text())
    assert parent['result']['converged'] and parent['status']=='failed'
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    output=RESULTS/'taxol_fixed_orbital_full_diagnostic.json'
    if output.exists(): raise FileExistsError('Archive existing diagnostic explicitly')
    report=dict(_metadata('gpu'),status='running',
        parent_sha256=_file_identity(parent_path)['sha256'],
        baseline_sha256=_file_identity(resident_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        helper_sha256=_file_identity(BASE/'gradient_snapshot.py')['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        note='Diagnostic only: original full gradient at saved disk-run orbitals, fresh NumInt/grids. Compare saved packed gradient at same orbitals and saved resident full gradient at separately converged orbitals. No accuracy gate is relaxed.')
    save(output,report)
    with lib.with_omp_threads(4):
        mf=restore('taxol_svp_b3lyp',parent['orbitals']['path'])
        grad=mf.nuc_grad_method()
        grad.grid_response=grad.auxbasis_response=True
        actual=grad.kernel()
        actual=np.asarray(actual.get() if hasattr(actual,'get') else actual)
    disk=np.asarray(parent['result']['gradient_hartree_per_bohr'])
    original=np.asarray(resident['result']['gradient_hartree_per_bohr'])
    report.update(gradient=actual.tolist(),
        error_to_same_orbital_packed=float(np.max(np.abs(actual-disk))),
        error_to_separate_resident_scf=float(np.max(np.abs(actual-original))),status='captured')
    save(output,report)
    print('full replay error to same orbital packed',report['error_to_same_orbital_packed'],
          'separate resident SCF',report['error_to_separate_resident_scf'],flush=True)
