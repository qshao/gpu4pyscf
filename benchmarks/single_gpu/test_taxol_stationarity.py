"""Measure true Fock stationarity before testing a tighter SCF solution."""
import json
import time
import traceback

import numpy as np
import pytest

from comprehensive import BASE, RESULTS, SOURCES, ROOT, configured, save
from molecular import _file_identity, _metadata


@pytest.mark.special
@pytest.mark.benchmark
def test_taxol_stationarity():
    import cupy as cp
    from pyscf import lib
    from gradient_snapshot import restore,write_orbitals
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    parent=json.loads(parent_path.read_text())
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    output=BASE/'accuracy_results/taxol_stationarity.json'
    if output.exists(): raise FileExistsError('Archive existing diagnostic explicitly')
    report=dict(_metadata('gpu'),status='running',
        parent_sha256=_file_identity(parent_path)['sha256'],
        benchmark_sha256=_file_identity(__file__)['sha256'],
        algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
        conv_tol=1e-12,conv_tol_grad=1e-10,
        note='Diagnostic: fresh unextrapolated Fock at saved converged orbitals, then explicitly tighter SCF from that density. Same Hamiltonian, basis and full grid/aux response. Different, stronger SCF thresholds are recorded honestly; this is not an unchanged-settings acceptance or a speed benchmark.')
    save(output,report)
    try:
        with lib.with_omp_threads(4):
            mf=restore('taxol_svp_b3lyp',parent['orbitals']['path'])
            def residual():
                dm=mf.make_rdm1()
                hcore=mf.get_hcore(); veff=mf.get_veff(dm=dm)
                fock=hcore+veff
                projected=mf.mo_coeff.T@fock@mf.mo_coeff
                return dict(orbital_gradient_norm=float(cp.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ,fock))),
                    canonical_fock_error=float(cp.max(cp.abs(projected-cp.diag(mf.mo_energy)))),
                    energy_hartree=float(mf.energy_tot(dm,hcore,veff)))
            report['initial_residual']=residual()
            print('Taxol initial stationarity',report['initial_residual'],flush=True)
            save(output,report)
            mf.conv_tol=report['conv_tol']; mf.conv_tol_grad=report['conv_tol_grad']
            observations=[]
            def callback(env):
                observation=dict(cycle=int(env['cycle']),energy=float(env['e_tot']),
                                 orbital_gradient_norm=float(env['norm_gorb']))
                observations.append(observation)
                print('Taxol refinement',observation,flush=True)
            mf.callback=callback; mf.max_cycle=60
            begin=time.perf_counter(); mf.kernel(dm0=mf.make_rdm1())
            report.update(scf_seconds=time.perf_counter()-begin,iterations=observations)
            assert mf.converged
            report['final_residual']=residual()
            orbitals=output.with_suffix('.npz'); write_orbitals(mf,orbitals)
            report['orbitals']=_file_identity(orbitals)
            grad=mf.nuc_grad_method(); grad.grid_response=grad.auxbasis_response=True
            begin=time.perf_counter(); force=grad.kernel()
            force=np.asarray(force.get() if hasattr(force,'get') else force)
            report.update(gradient_seconds=time.perf_counter()-begin,gradient=force.tolist())
            cpu=json.loads((RESULTS/'taxol_svp_b3lyp_seeded_cpu.json').read_text())['result']
            report.update(error_to_prior_cpu=float(np.max(np.abs(force-np.asarray(cpu['gradient_hartree_per_bohr'])))),
                          energy_error_to_prior_cpu=abs(mf.e_tot-cpu['energy_hartree']),status='captured')
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    print('Taxol tight SCF force error to previous CPU',report['error_to_prior_cpu'],flush=True)
