"""Large fixed-orbital regression for symmetric-component occupied-pair packing."""
import json
import numpy as np
import pytest

from comprehensive import BASE,ROOT,RESULTS,SOURCES,save
from molecular import _metadata,_file_identity


@pytest.mark.special
@pytest.mark.benchmark
def test_taxol_averaged_packing():
    from pyscf import lib
    from gradient_snapshot import restore
    from occupied_pair import packed_gradient
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    oracle_path=RESULTS/'taxol_fixed_orbital_jk_diagnostic.json'
    parent=json.loads(parent_path.read_text())
    oracle=json.loads(oracle_path.read_text())
    assert oracle['parent_sha256']==_file_identity(parent_path)['sha256']
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    output=RESULTS/'taxol_averaged_occupied_pair.json'
    if output.exists(): raise FileExistsError('Archive existing report explicitly')
    report=dict(_metadata('gpu'),status='running',
        benchmark_sha256=_file_identity(__file__)['sha256'],
        candidate_sha256=_file_identity(BASE/'occupied_pair.py')['sha256'],
        helper_sha256=_file_identity(BASE/'gradient_snapshot.py')['sha256'],
        oracle_sha256=_file_identity(oracle_path)['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        note='Same converged orbitals and original DF J/K gradient oracle. Average both orbital-pair orientations before packing to remove one-sided floating-point antisymmetry. Extra1e-9 roundoff regression target; original CPU energy/force gates unchanged.')
    assert report['native_libraries']==oracle['native_libraries']
    save(output,report)
    with lib.with_omp_threads(4):
        mf=restore('taxol_svp_b3lyp',parent['orbitals']['path'])
        omega,alpha,hybrid=mf._numint.rsh_and_hybrid_coeff(mf.xc,spin=mf.mol.spin)
        assert omega==0
        with packed_gradient():
            grad=mf.nuc_grad_method()
            grad.auxbasis_response=True
            actual=grad.jk_energy_per_atom(mf.make_rdm1(),j_factor=1.,k_factor=hybrid,hermi=1)
    error=float(np.max(np.abs(actual-np.asarray(oracle['first']))))
    report.update(gradient=actual.tolist(),max_error=error,status='passed' if error<=1e-9 else 'failed')
    save(output,report)
    assert error<=1e-9,error
    print('Taxol averaged packing J/K error',error,flush=True)
