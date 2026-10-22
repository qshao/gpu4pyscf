"""Production-grid scaling campaign, isolated from historical benchmarks."""
from contextlib import contextmanager
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import molecular
from molecular import SETTINGS, _file_identity, _metadata, compare_cases, run_case

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
RESULTS=BASE/'comprehensive_results'
CASES={
    'vitamin_svp_pbe':('020_Vitamin_C.xyz','def2-svp','PBE'),
    'vitamin_svp_b3lyp':('020_Vitamin_C.xyz','def2-svp','B3LYP'),
    'tamoxifen_svp_pbe':('057_Tamoxifen.xyz','def2-svp','PBE'),
    'tamoxifen_svp_b3lyp':('057_Tamoxifen.xyz','def2-svp','B3LYP'),
    'tamoxifen_tzvp_b3lyp':('057_Tamoxifen.xyz','def2-tzvp','B3LYP'),
    'taxol_svp_b3lyp':('113_Taxol.xyz','def2-svp','B3LYP'),
    'valinomycin_svp_pbe':('168_Valinomycin.xyz','def2-svp','PBE'),
    'valinomycin_svp_b3lyp':('168_Valinomycin.xyz','def2-svp','B3LYP'),
}
SOURCES=['gpu4pyscf/df/df.py','gpu4pyscf/df/df_jk.py',
         'gpu4pyscf/df/grad/rhf.py','gpu4pyscf/df/grad/uhf.py',
         'gpu4pyscf/df/int3c2e_bdiv.py','gpu4pyscf/dft/numint.py',
         'gpu4pyscf/lib/cutensor.py']


def settings(case):
    geometry,basis,method=CASES[case]
    return dict(atom=str(ROOT/'benchmarks/molecules/organic'/geometry),
                unit='Angstrom',basis=basis,auxbasis='def2-universal-jkfit',
                conv_tol=1e-10,conv_tol_grad=1e-7,grid_level=3,
                grid_pruning='nwchem',small_rho_cutoff=0.,grid_response=True,
                auxbasis_response=True,cpu_threads=4,max_cycle=100)


@contextmanager
def configured(case):
    config=settings(case)
    original=molecular._mean_field
    def factory(method,backend):
        mf,dm0=original(method,backend)
        if backend=='cpu':
            from pyscf.dft.gen_grid import nwchem_prune
        else:
            from gpu4pyscf.dft.gen_grid import nwchem_prune
            mf.with_df.use_gpu_memory=True
            mf.with_df.cderi_cache_bytes=0
        mf.grids.prune=nwchem_prune
        mf.max_cycle=config['max_cycle']
        assert mf.grids.level==config['grid_level']
        return mf,dm0
    with patch.dict(SETTINGS,config),patch.object(molecular,'_mean_field',factory):
        yield config


def save(path,report):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(path)


def execute(case,mode):
    if mode not in ['cpu','gpu','profile','memory']:
        raise ValueError('Unknown comprehensive benchmark mode: '+mode)
    config=settings(case)
    method=CASES[case][2]
    RESULTS.mkdir(exist_ok=True)
    path=RESULTS/(case+'_'+mode+'.json')
    if path.exists():
        raise FileExistsError('Archive the existing report explicitly before rerunning: '+str(path))
    backend='cpu' if mode=='cpu' else 'gpu'
    report=dict(_metadata(backend),status='running',case=case,mode=mode,
                settings=config,driver_sha256=_file_identity(__file__)['sha256'],
                geometry_sha256=_file_identity(config['atom'])['sha256'],
                algorithm_source_hashes={p:_file_identity(ROOT/p)['sha256'] for p in SOURCES},
                helper_hashes={p:_file_identity(BASE/p)['sha256'] for p in
                               ['memory_profile.py','test_stage_profile.py','test_fine_stage_profile.py']},
                note='Level3 NWChem-pruned matched grid, not a grid-convergence study. FP64, full auxiliary/grid response. GPU-resident CDERI forced; experimental cache disabled. Initial single cold control establishes scaling, not repeated speedup. Profiles synchronize inclusive nested scopes; allocation run separate. CPU references are scientific oracles, not tuned CPU timing baselines.')
    save(path,report)
    try:
        cpu=None
        if backend=='gpu':
            cpu_report=json.loads((RESULTS/(case+'_cpu.json')).read_text())
            assert cpu_report['status']=='passed' and cpu_report['settings']==config
            assert cpu_report['geometry_sha256']==report['geometry_sha256']
            common='benchmarks/single_gpu/molecular.py'
            assert cpu_report['source_hashes'][common]==report['source_hashes'][common]
            cpu=cpu_report['result']
            for key in ['pyscf','numpy','cpu_libxc']:
                assert cpu[key]==report[key]
            report['cpu_reference_sha256']=_file_identity(RESULTS/(case+'_cpu.json'))['sha256']
        with configured(case):
            if mode=='profile':
                from test_stage_profile import stage_profile
                from test_fine_stage_profile import fine_profile
                with stage_profile() as coarse,fine_profile() as fine:
                    result=run_case(method,backend)
                assert fine['bindings_restored']
                report.update(coarse=coarse,fine=fine)
            elif mode=='memory':
                from memory_profile import peak_memory
                with peak_memory() as memory:
                    result=run_case(method,backend)
                assert memory['initial_pool_live_bytes']==0
                report['memory']=memory
            else:
                result=run_case(method,backend)
        assert result['converged']
        report['result']=result
        if cpu is not None:
            comparison=compare_cases(cpu,result)
            report['comparison']=comparison
            assert comparison['passed'],comparison
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
        save(path,report)
        raise
    save(path,report)
    return report
