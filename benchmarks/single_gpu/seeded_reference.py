"""Independent CPU SCF/forces seeded by saved GPU orbitals to avoid a cold SCF."""
import json
import traceback
from unittest.mock import patch

import numpy as np

import molecular
from comprehensive import BASE,ROOT,RESULTS,settings,configured,save
from molecular import _metadata,_file_identity,run_case,compare_cases,SETTINGS


def compare_execution_cases(cpu,gpu):
    """Keep all physical settings/gates strict; record thread counts separately."""
    left={k:v for k,v in cpu['settings'].items() if k!='cpu_threads'}
    right={k:v for k,v in gpu['settings'].items() if k!='cpu_threads'}
    if left!=right: raise ValueError('Cannot compare different scientific settings')
    return compare_cases(dict(cpu,settings=gpu['settings']),gpu)


def execute(case):
    if case!='taxol_svp_b3lyp': raise ValueError('Unknown seeded reference case')
    parent_path=RESULTS/(case+'_capacity_disk_packed.json')
    parent=json.loads(parent_path.read_text())
    assert parent['result']['converged']
    assert parent['geometry_sha256']==_file_identity(settings(case)['atom'])['sha256']
    snapshot=parent['orbitals']
    assert _file_identity(snapshot['path'])==snapshot
    with np.load(snapshot['path'],allow_pickle=False) as archive:
        coeff,occ=archive['mo_coeff'],archive['mo_occ']
        assert coeff.dtype==occ.dtype==np.float64
        factor=coeff[:,occ>0]*np.sqrt(occ[occ>0])
        density=factor@factor.T
    output=RESULTS/(case+'_seeded_cpu.json')
    if output.exists(): raise FileExistsError('Archive existing CPU report explicitly')
    config=dict(settings(case),cpu_threads=16)
    report=dict(_metadata('cpu'),status='running',case=case,settings=config,
        benchmark_sha256=_file_identity(__file__)['sha256'],
        configuration_sha256=_file_identity(BASE/'comprehensive.py')['sha256'],
        seed_report_sha256=_file_identity(parent_path)['sha256'],seed=snapshot,
        geometry_sha256=parent['geometry_sha256'],
        note='Independent CPU SCF and full grid/auxiliary-response gradients, unchanged physical settings and convergence thresholds. GPU orbitals used only to construct the initial density; CPU recomputes Hamiltonian/integrals and must converge.16CPU threads are an execution control, not a scientific change or tuned CPU performance baseline. For comparison, only cpu_threads may differ; all other settings and original1e-8Eh/1e-6Eh/Bohr gates remain strict.')
    save(output,report)
    try:
        with configured(case),patch.dict(SETTINGS,cpu_threads=16):
            original=molecular._mean_field
            def factory(method,backend):
                assert backend=='cpu'
                mf,_=original(method,backend)
                assert density.shape==(mf.mol.nao,mf.mol.nao)
                assert abs(occ.sum()-mf.mol.nelectron)<1e-10
                return mf,density.copy()
            with patch.object(molecular,'_mean_field',factory):
                result=run_case('B3LYP','cpu')
        assert result['converged'] and result['settings']==config
        report.update(result=result,status='passed')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        save(output,report)
        raise
    save(output,report)
    return report
