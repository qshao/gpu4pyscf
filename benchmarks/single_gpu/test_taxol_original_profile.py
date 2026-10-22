"""Locate large-system gradient costs at a saved state, without repeating SCF."""
import json
import traceback

import numpy as np
import pytest

from comprehensive import BASE, ROOT, RESULTS, SOURCES, save
from molecular import _file_identity, _metadata


@pytest.mark.special
@pytest.mark.benchmark
def test_original_gradient_profile():
    from pyscf import lib
    from gradient_snapshot import restore
    from test_stage_profile import stage_profile
    from test_fine_stage_profile import fine_profile

    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    oracle_path=RESULTS/'taxol_fixed_orbital_full_diagnostic.json'
    parent=json.loads(parent_path.read_text())
    oracle=json.loads(oracle_path.read_text())
    assert parent['result']['converged'] and oracle['status']=='captured'
    assert oracle['parent_sha256']==_file_identity(parent_path)['sha256']
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    sources={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES}
    assert sources==oracle['algorithm_source_hashes']
    output=RESULTS/'taxol_original_gradient_profile.json'
    if output.exists(): raise FileExistsError('Archive existing profile explicitly')
    report=dict(_metadata('gpu'),status='running',
        parent_sha256=_file_identity(parent_path)['sha256'],
        oracle_sha256=_file_identity(oracle_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        helper_hashes={p:_file_identity(BASE/p)['sha256'] for p in
                      ['gradient_snapshot.py','test_stage_profile.py','test_fine_stage_profile.py']},
        algorithm_source_hashes=sources,
        note='Original gradient only, at saved converged orbitals. Full grid and auxiliary response. Inclusive synchronized profile scopes; do not sum nested timings. May overlap CPU reference; diagnostic bottleneck attribution, not speed acceptance. Matches an original full-gradient replay, not an independent CPU oracle or qualification of the failed packing workflow.')
    save(output,report)
    try:
        with lib.with_omp_threads(4):
            mf=restore('taxol_svp_b3lyp',parent['orbitals']['path'])
            grad=mf.nuc_grad_method()
            grad.grid_response=grad.auxbasis_response=True
            with stage_profile() as coarse, fine_profile() as fine:
                actual=grad.kernel()
                actual=np.asarray(actual.get() if hasattr(actual,'get') else actual)
        error=float(np.max(np.abs(actual-np.asarray(oracle['gradient']))))
        report.update(coarse=coarse,fine=fine,gradient=actual.tolist(),replay_error=error)
        assert fine['bindings_restored']
        assert error<=1e-9,error
        report['status']='passed_replay'
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('Taxol original gradient profile',report['replay_error'],coarse,flush=True)
