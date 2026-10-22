"""Prepare shared atomic guesses; never use a converged GPU density as an oracle."""
from pathlib import Path
import hashlib
import numpy as np


def prepare_density(case,path):
    from pyscf import gto,scf,lib
    root=Path(__file__).resolve().parents[3]
    with lib.with_omp_threads(4):
        if case['task']=='pbc_energy':
            from pyscf.pbc import gto as pgto,scf as pscf
            spec=case['cell']
            mol=pgto.Cell(a=spec['a'],atom=spec['atom'],basis=spec['basis'],pseudo=spec['pseudo'],
                         charge=case['charge'],spin=case['spin'],unit=spec.get('unit','Angstrom'),
                         precision=spec.get('precision',1e-8),verbose=0)
            mol.build()
            mesh=spec.get('kmesh',[1,1,1])
            if mesh==[1,1,1]:mf=(pscf.UHF if case['spin'] else pscf.RHF)(mol)
            else:mf=(pscf.KUHF if case['spin'] else pscf.KRHF)(mol,kpts=mol.make_kpts(mesh))
        else:
            mol=gto.M(atom=str(root/case['geometry']),basis=case['basis'],charge=case['charge'],spin=case['spin'],
                       ecp=case.get('task_options',{}).get('ecp',{}),unit=case['settings'].get('unit','Angstrom'),verbose=0)
            mf=(scf.UHF if case['spin'] else scf.RHF)(mol)
        density=np.asarray(mf.get_init_guess(key='minao'))
        if np.iscomplexobj(density):
            if np.max(np.abs(density.imag))>1e-13:raise ValueError('complex periodic atomic guess requires an explicitly complex-qualified protocol')
            density=density.real
        density=np.array(density,dtype=np.float64,order='C')
        if not np.isfinite(density).all():raise ValueError('nonfinite initial density')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.save(f,density,allow_pickle=False)
    from run import digest
    return {'initial_density_file':str(path.resolve()),'initial_density_file_sha256':digest(path),
            'array_sha256':hashlib.sha256(density.tobytes()).hexdigest(),'shape':list(density.shape),
            'provenance':'Independent CPU atomic MINAO density; fixed input preparation outside workflow. Loading/upload is included in setup.'}
