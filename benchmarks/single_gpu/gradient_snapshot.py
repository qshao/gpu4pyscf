"""Converged-orbital snapshots for isolated gradient comparisons."""
from pathlib import Path
import numpy as np


def write_orbitals(mf,path):
    import cupy as cp
    path=Path(path)
    if path.exists(): raise FileExistsError('Archive orbital snapshot explicitly: '+str(path))
    if not mf.converged: raise ValueError('Only converged orbitals can be saved')
    arrays={name:cp.asnumpy(getattr(mf,name)) for name in ['mo_coeff','mo_occ','mo_energy']}
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Snapshot requires finite real FP64 orbitals')
    with path.open('xb') as stream:
        np.savez(stream,energy_hartree=float(mf.e_tot),**arrays)


def restore(case,path):
    import cupy as cp
    import molecular
    from comprehensive import configured,CASES
    from pyscf.df.addons import make_auxmol
    with configured(case) as config:
        mf,_=molecular._mean_field(CASES[case][2],'gpu')
    with np.load(path,allow_pickle=False) as archive:
        coeff,occ,energy=[archive[key] for key in ['mo_coeff','mo_occ','mo_energy']]
        if (coeff.ndim!=2 or coeff.shape[0]!=mf.mol.nao
                or occ.shape!=(coeff.shape[1],) or energy.shape!=occ.shape
                or any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in [coeff,occ,energy])
                or abs(occ.sum()-mf.mol.nelectron)>1e-10):
            raise ValueError('Orbital snapshot does not match molecule')
        mf.mo_coeff,mf.mo_occ,mf.mo_energy=map(cp.asarray,[coeff,occ,energy])
        mf.e_tot=float(archive['energy_hartree'])
        if not np.isfinite(mf.e_tot): raise ValueError('Nonfinite snapshot energy')
    mf.converged=True
    mf.with_df.auxmol=make_auxmol(mf.mol,config['auxbasis'])
    return mf
