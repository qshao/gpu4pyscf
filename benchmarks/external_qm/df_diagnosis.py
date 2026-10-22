"""Frozen-density Coulomb derivative diagnostic, run explicitly with pytest."""
import json
import numpy as np
from test_preflight import ROOT


def test_identify_coulomb_derivative_mismatch():
    from pyscf import gto, lib
    from pyscf.grad.rhf import Gradients as DirectGradient
    from pyscf.df.grad.rhf import Gradients as DFGradient
    from skala.pyscf import SkalaKS

    lib.num_threads(2)
    findings = []
    for charge, spin in ((0, 0), (1, 1)):
        mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
                    basis="def2-svp", charge=charge, spin=spin, verbose=0)
        mf = SkalaKS(mol, xc="pbe", with_dftd3=False, with_density_fit=True,
                     auxbasis="def2-universal-jkfit")
        mf.grids.atom_grid = (30, 110)
        mf.grids.prune = None
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        assert mf.converged
        gradient = mf.nuc_grad_method().kernel()
        dm = mf.make_rdm1()
        dm = dm.sum(axis=0) if dm.ndim == 3 else dm
        direct = DirectGradient(mf).get_j(dm=dm)
        fitted = DFGradient(mf).get_j(dm=dm)
        correction = np.array([
            2 * np.einsum('xij,ij->x', (fitted-direct)[:,p0:p1], dm[p0:p1])
            for _, _, p0, p1 in mol.aoslice_by_atom()
        ]) + fitted.aux.sum(axis=(0, 1))
        findings.append({"charge": charge, "spin": spin,
                         "gradient": np.asarray(gradient).tolist(),
                         "df_minus_direct_j_derivative": correction.tolist()})
    (ROOT / "benchmarks/external_qm/df_diagnosis_current.json").write_text(
        json.dumps(findings, indent=2) + "\n")
