"""Decompose the Taxol force discrepancy without changing acceptance gates."""
import json
import time
import traceback

import numpy as np
import pytest

from comprehensive import BASE, RESULTS, configured, save
from molecular import _file_identity, _metadata

OUT=BASE/'accuracy_results'


@pytest.mark.special
@pytest.mark.benchmark
def test_cpu_fixed_orbital_components():
    import molecular
    from pyscf import lib
    parent_path=RESULTS/'taxol_svp_b3lyp_capacity_disk_packed.json'
    xc_path=RESULTS/'taxol_fixed_orbital_xc_cpu.json'
    parent=json.loads(parent_path.read_text()); xc=json.loads(xc_path.read_text())
    assert xc['status']=='captured' and xc['orbitals']==parent['orbitals']
    assert _file_identity(parent['orbitals']['path'])==parent['orbitals']
    OUT.mkdir(exist_ok=True)
    path=OUT/'taxol_cpu_fixed_components.json'
    if path.exists(): raise FileExistsError('Archive diagnostic explicitly')
    report=dict(_metadata('cpu'),status='running',
        benchmark_sha256=_file_identity(__file__)['sha256'],
        parent_sha256=_file_identity(parent_path)['sha256'],
        xc_sha256=_file_identity(xc_path)['sha256'],
        note='Independent CPU derivatives at identical saved GPU orbitals AND orbital energies. Reuses independently evaluated CPU full-grid XC at that density.16CPU threads; no SCF. J/K includes auxiliary response. Separates derivative arithmetic from separately converged SCF states; not a new converged CPU reference.')
    save(path,report)
    try:
        with lib.with_omp_threads(16),configured('taxol_svp_b3lyp'):
            mf,_=molecular._mean_field('B3LYP','cpu')
            with np.load(parent['orbitals']['path'],allow_pickle=False) as data:
                for name in ['mo_coeff','mo_occ','mo_energy']: setattr(mf,name,data[name])
            dm=mf.make_rdm1()
            grad=mf.nuc_grad_method(); grad.auxbasis_response=True
            dm=grad._tag_rdm1(dm,mf.mo_coeff,mf.mo_occ)
            begin=time.perf_counter()
            print('CPU Taxol fixed-orbital J/K started',flush=True)
            vj,vk=grad.get_jk(mf.mol,dm,hermi=1)
            hybrid=mf._numint.rsh_and_hybrid_coeff(mf.xc,spin=mf.mol.spin)[2]
            j=np.asarray(vj.aux).sum((0,1)).copy()
            k=np.asarray(vk.aux).sum((0,1)).copy()
            overlap=grad.get_ovlp(mf.mol)
            weighted=grad.make_rdm1e(mf.mo_energy,mf.mo_coeff,mf.mo_occ)
            hcore=grad.hcore_generator(mf.mol)
            one=np.zeros_like(j)
            for atom,(_,_,p0,p1) in enumerate(mf.mol.aoslice_by_atom()):
                j[atom]+=2*np.einsum('xij,ij->x',vj[:,p0:p1],dm[p0:p1])
                k[atom]+=2*np.einsum('xij,ij->x',vk[:,p0:p1],dm[p0:p1])
                one[atom]=np.einsum('xij,ij->x',hcore(atom),dm)-2*np.einsum('xij,ij->x',overlap[:,p0:p1],weighted[p0:p1])
            jk=j-.5*hybrid*k
            nuclear=grad.grad_nuc()
            full=jk+one+nuclear+np.asarray(xc['force'])
            report.update(seconds=time.perf_counter()-begin,j=j.tolist(),k=k.tolist(),
                jk=jk.tolist(),one=one.tolist(),nuclear=nuclear.tolist(),full=full.tolist())
            oracle=json.loads((RESULTS/'taxol_fixed_orbital_jk_diagnostic.json').read_text())
            full_oracle=json.loads((RESULTS/'taxol_fixed_orbital_full_diagnostic.json').read_text())
            report.update(jk_error=float(np.max(np.abs(jk-np.asarray(oracle['first'])))),
                full_error=float(np.max(np.abs(full-np.asarray(full_oracle['gradient'])))),status='captured')
            assert np.isfinite(full).all()
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(path,report)
        raise
    save(path,report)
    print('CPU fixed components',report['seconds'],report['jk_error'],report['full_error'],flush=True)
