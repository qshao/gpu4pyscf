"""CPU-only check of the finite-difference Hessian reference against PySCF's analytic RHF Hessian."""
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import adapters


def test_finite_difference_hessian_matches_analytic_hf():
    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', basis='def2-svp', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    analytic = mf.Hessian().kernel()
    fd, diagnostics = adapters._finite_difference_hessian(mf, {'df': False, 'xc': 'HF'})
    assert np.abs(fd - analytic).max() < 5e-6
    assert diagnostics['step_stability_max_abs'] < 5e-6
    assert np.abs(fd - fd.transpose(1, 0, 3, 2)).max() < 1e-12
    assert diagnostics['gradient_evaluations'] == 36
