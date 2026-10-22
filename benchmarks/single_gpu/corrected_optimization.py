"""Shared provenance and unchanged accuracy gates for corrected-code experiments."""
import json

import numpy as np

from comprehensive import BASE, ROOT, RESULTS, SOURCES, settings, save
from molecular import _file_identity, _metadata
from seeded_reference import compare_execution_cases

OUT=BASE/'optimization_results'
CASE='taxol_svp_b3lyp'


def digest(path):
    return _file_identity(path)['sha256']


def oracle():
    path=RESULTS/(CASE+'_seeded_cpu.json')
    report=json.loads(path.read_text())
    assert report['status']=='passed' and report['result']['converged']
    assert report['geometry_sha256']==digest(settings(CASE)['atom'])
    return path,report


def metadata(script):
    path,cpu=oracle()
    report=dict(_metadata('gpu'),status='running',case=CASE,
        benchmark_sha256=digest(script),cpu_reference_sha256=digest(path),
        configuration_sha256=digest(BASE/'comprehensive.py'),
        shared_helper_sha256=digest(__file__),
        comparison_helper_sha256=digest(BASE/'seeded_reference.py'),
        candidate_sha256=digest(BASE/'occupied_pair.py'),
        algorithm_source_hashes={p:digest(ROOT/p) for p in SOURCES},
        geometry_sha256=cpu['geometry_sha256'],settings=settings(CASE))
    for key in ['pyscf','numpy','cpu_libxc']: assert report[key]==cpu[key]
    OUT.mkdir(exist_ok=True)
    return report


def new_report(name,script):
    path=OUT/name
    if path.exists(): raise FileExistsError('Archive existing report explicitly: '+str(path))
    report=metadata(script); save(path,report)
    return path,report


def seed_density():
    accepted=BASE/'accuracy_results/taxol_resident_acceptance.json'
    parent=json.loads(accepted.read_text()); assert parent['status']=='passed'
    assert _file_identity(parent['seed']['path'])==parent['seed']
    with np.load(parent['seed']['path'],allow_pickle=False) as data:
        density=(data['mo_coeff']*data['mo_occ'])@data['mo_coeff'].T
    return density,parent['seed'],digest(accepted)


def gradient_inputs():
    path=OUT/'corrected_profile.json'
    profile=json.loads(path.read_text())
    assert profile['status']=='passed' and profile['result']['converged']
    assert profile['settings']==settings(CASE)
    assert profile['geometry_sha256']==digest(settings(CASE)['atom'])
    assert _file_identity(profile['orbitals']['path'])==profile['orbitals']
    assert profile['configuration_sha256']==digest(BASE/'comprehensive.py')
    assert profile['shared_helper_sha256']==digest(__file__)
    current=_metadata('gpu')
    for key in ['pyscf','numpy','cpu_libxc','cupy','native_libraries','source_hashes']:
        assert current[key]==profile[key],key
    archive=BASE/'runs/2026-10-18-pre-next-optimization'
    for source,sha in profile['algorithm_source_hashes'].items():
        allowed={digest(ROOT/source)}
        saved=archive/(source.replace('/','__')+'.source')
        if saved.exists(): allowed.add(digest(saved))
        assert sha in allowed,source
    return profile,digest(path)


def check_gradient(reference,gradient):
    actual=np.asarray(gradient.get() if hasattr(gradient,'get') else gradient)
    expected=np.asarray(reference['gradient_hartree_per_bohr'])
    assert actual.shape==expected.shape and np.isfinite(actual).all()
    error=float(np.max(np.abs(actual-expected)))
    assert error<=1e-9,error
    path,cpu=oracle()
    result=dict(reference,gradient_hartree_per_bohr=actual.tolist())
    comparison=compare_execution_cases(cpu['result'],result)
    assert comparison['passed'],comparison
    return dict(error_to_corrected_gpu=error,cpu_comparison=comparison,
                cpu_reference_sha256=digest(path),gradient=actual.tolist())
