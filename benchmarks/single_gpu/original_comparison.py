"""Immutable original/current GPU packages and common scientific configuration."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=BASE/'original_comparison'
REVISION='9af748a71ca8f4881c604332c2f0cc1632e033ac'
CASES={'small':'vitamin_svp_b3lyp','medium':'tamoxifen_svp_b3lyp',
       'large':'taxol_svp_b3lyp'}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): value.update(block)
    return value.hexdigest()


def save(path,data):
    path=Path(path)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def case_settings(size):
    from comprehensive import settings
    return settings(CASES[size])


def configure_cpu(size):
    from pyscf import gto,dft
    from pyscf.dft.gen_grid import nwchem_prune
    config=case_settings(size)
    mol=gto.M(atom=config['atom'],basis=config['basis'],unit=config['unit'],verbose=0)
    mf=dft.RKS(mol,xc='B3LYP').density_fit(auxbasis=config['auxbasis'])
    mf.grids.level=config['grid_level'];mf.grids.prune=nwchem_prune
    mf.small_rho_cutoff=config['small_rho_cutoff']
    mf.conv_tol=config['conv_tol'];mf.conv_tol_grad=config['conv_tol_grad']
    mf.max_cycle=config['max_cycle']
    return mf


def python_sources(package):
    return {str(path.relative_to(package)):digest(path)
            for path in sorted(package.rglob('*.py')) if '__pycache__' not in path.parts}


def cpu_reference(size):
    name=CASES[size]+('_seeded_cpu.json' if size=='large' else '_cpu.json')
    path=BASE/'comprehensive_results'/name
    data=json.loads(path.read_text())
    assert data['status']=='passed' and data['result']['converged']
    assert data['geometry_sha256']==digest(case_settings(size)['atom'])
    expected=case_settings(size)
    actual=dict(data['result']['settings'])
    actual['cpu_threads']=expected['cpu_threads']
    assert actual==expected
    return path,data


def prepare():
    manifest_path=OUT/'manifest.json'
    if manifest_path.exists(): raise FileExistsError('Comparison already frozen: '+str(manifest_path))
    OUT.mkdir(exist_ok=True)
    original=OUT/'original';current=OUT/'current'
    assert not original.exists() and not current.exists()
    original.mkdir()
    archive=subprocess.check_output(['git','-C',str(ROOT),'archive',REVISION,'gpu4pyscf'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(original,filter='data')
    shutil.copytree(ROOT/'gpu4pyscf',current/'gpu4pyscf',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.so.*','deps'))
    native={}
    for path in sorted((ROOT/'gpu4pyscf/lib').glob('*.so')):
        native[str(path.relative_to(ROOT))]=dict(path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size)
        for target in [original,current]:
            destination=target/path.relative_to(ROOT)
            assert not destination.exists()
            destination.symlink_to(path.resolve())
    dependencies=ROOT/'gpu4pyscf/lib/deps'
    for target in [original,current]:
        destination=target/'gpu4pyscf/lib/deps'
        assert not destination.exists()
        destination.symlink_to(dependencies.resolve(),target_is_directory=True)
    for path in sorted((dependencies/'lib').glob('*.so*')):
        native[str(path.relative_to(ROOT))]=dict(path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size)
    changed=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',REVISION],text=True).splitlines()
    assert not any(p.startswith('gpu4pyscf/lib/') and p.endswith(('.cu','.cpp','.c','.h','.cuh')) for p in changed)
    manifest=dict(original_revision=REVISION,
        packages={v:dict(path=str((OUT/v/'gpu4pyscf').resolve()),sources=python_sources(OUT/v/'gpu4pyscf'))
                  for v in ['original','current']},native_libraries=native,
        geometry_sha256={size:digest(case_settings(size)['atom']) for size in CASES},
        cpu_references={size:dict(path=str(cpu_reference(size)[0]),sha256=digest(cpu_reference(size)[0])) for size in CASES},
        runner_sha256=digest(BASE/'test_original_comparison_run.py'),
        configuration_sha256=digest(__file__),memory_profiler_sha256=digest(BASE/'memory_profile.py'),
        comparison_sha256=digest(BASE/'seeded_reference.py'),
        note='Original untouched Python algorithms at9af748a and frozen current sources use identical native numerical kernels, rebuilt forSM121/CUDA13. Build-architecture compatibility is shared; this is not a comparison with an incompatible released wheel. No native numerical kernel source changed. Complete frozen packages isolate all modifications, not just contraction code.')
    save(manifest_path,manifest)
    (OUT/'results').mkdir()
    print('Frozen original/current packages:',manifest_path,flush=True)


def manifest():
    path=OUT/'manifest.json'
    data=json.loads(path.read_text())
    assert data['configuration_sha256']==digest(__file__)
    assert data['runner_sha256']==digest(BASE/'test_original_comparison_run.py')
    assert data['memory_profiler_sha256']==digest(BASE/'memory_profile.py')
    assert data['comparison_sha256']==digest(BASE/'seeded_reference.py')
    return data


if __name__=='__main__': prepare()
