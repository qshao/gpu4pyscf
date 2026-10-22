"""Isolate full-grid XC force arithmetic at identical saved orbitals."""
import json
import time
import traceback

import numpy as np
import pytest

from comprehensive import BASE, ROOT, RESULTS, SOURCES, configured, save
from molecular import _file_identity, _metadata


@pytest.mark.special
@pytest.mark.benchmark
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_fixed_orbital_xc(backend):
    import molecular
    from pyscf import lib
    case='taxol_svp_b3lyp'
    parent_path=RESULTS/(case+'_capacity_disk_packed.json')
    parent=json.loads(parent_path.read_text())
    snapshot=parent['orbitals']
    assert _file_identity(snapshot['path'])==snapshot
    output=RESULTS/('taxol_fixed_orbital_xc_'+backend+'.json')
    if output.exists(): raise FileExistsError('Archive existing diagnostic explicitly')
    report=dict(_metadata(backend),status='running',backend=backend,
        parent_sha256=_file_identity(parent_path)['sha256'],orbitals=snapshot,
        benchmark_sha256=_file_identity(__file__)['sha256'],
        configuration_sha256=_file_identity(BASE/'comprehensive.py')['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        note='Diagnostic only: XC force including full grid response, at identical saved GPU orbitals/density. CPU formula is the XC subset of pyscf.grad.rhf.grad_elec plus exc1_grid; no J/K, nuclear, one-electron or SCF qualification. Unchanged grid/functional. Four CPU threads. CPU/GPU comparison made separately; timings diagnostic.')
    save(output,report)
    try:
        with lib.with_omp_threads(4),configured(case):
            mf,_=molecular._mean_field('B3LYP',backend)
            with np.load(snapshot['path'],allow_pickle=False) as data:
                coeff,occ=data['mo_coeff'],data['mo_occ']
            density=(coeff*occ)@coeff.T
            started=time.perf_counter()
            if backend=='cpu':
                from pyscf.grad.rks import get_vxc_full_response
                mf.grids.build(with_non0tab=True)
                grid_force,vxc=get_vxc_full_response(mf._numint,mf.mol,mf.grids,mf.xc,
                                                    density,max_memory=2000)
                force=np.asarray(grid_force).copy()
                for atom,(_,_,p0,p1) in enumerate(mf.mol.aoslice_by_atom()):
                    force[atom]+=2*np.einsum('xij,ij->x',vxc[:,p0:p1],density[p0:p1])
            else:
                import cupy as cp
                from gpu4pyscf.grad.rks import get_exc_full_response
                force,_=get_exc_full_response(mf._numint,mf.mol,mf.grids,mf.xc,cp.asarray(density))
                cp.cuda.Device().synchronize()
            elapsed=time.perf_counter()-started
        force=np.asarray(force)
        assert force.shape==(mf.mol.natm,3) and np.isfinite(force).all()
        report.update(force=force.tolist(),settings=parent['configuration']['scientific'],
                      seconds=elapsed,status='captured')
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('Taxol fixed-orbital XC',backend,report['seconds'],flush=True)
