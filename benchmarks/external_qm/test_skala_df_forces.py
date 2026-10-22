"""All Cartesian DF forces against two finite-difference steps, no neural weights."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest
from test_preflight import ROOT


@pytest.mark.parametrize("charge,spin,auxbasis", [
    (0, 0, "def2-universal-jkfit"), (1, 1, "weigend")])
def test_all_df_gradient_components(charge, spin, auxbasis):
    path = ROOT / f"benchmarks/external_qm/df_forces_charge{charge}.json"
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(report, indent=2) + "\n")
    import cupy as cp
    import torch
    from pyscf import dft, gto, lib
    from skala.pyscf import SkalaKS as CpuKS
    from skala.gpu4pyscf import SkalaKS as GpuKS

    lib.num_threads(2)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    mol = gto.M(atom="O .05 .03 -.04; H .08 -.74 .63; H .13 .81 .55",
                basis="def2-svp", charge=charge, spin=spin, verbose=0)

    def make_ks(molecule, gpu=False):
        mf = (GpuKS if gpu else CpuKS)(molecule, xc="pbe", with_dftd3=False,
            with_density_fit=True, auxbasis=auxbasis)
        mf.grids.atom_grid = (30, 110)
        mf.grids.prune = None
        mf.grids.radi_method = dft.radi.treutler
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.max_cycle = 100
        return mf

    energies, gradients = [], []
    for gpu in (False, True):
        mf = make_ks(mol, gpu=gpu)
        energies.append(float(mf.kernel()))
        assert mf.converged
        grad = mf.nuc_grad_method()
        assert grad.auxbasis_response
        value = grad.kernel()
        gradients.append(value.get() if hasattr(value, "get") else np.asarray(value))
    cp.cuda.Stream.null.synchronize()
    coords = mol.atom_coords()
    finite_differences = []
    for step in (1e-3, 5e-4):
        fd = np.zeros((mol.natm, 3))
        for atom in range(mol.natm):
            for axis in range(3):
                values = []
                for direction in (-1, 1):
                    moved = coords.copy()
                    moved[atom, axis] += direction * step
                    displaced = mol.set_geom_(moved, unit="Bohr", inplace=False)
                    mf = make_ks(displaced)
                    values.append(float(mf.kernel()))
                    assert mf.converged
                fd[atom, axis] = (values[1] - values[0]) / (2 * step)
        finite_differences.append(fd)
    extrapolated = (4 * finite_differences[1] - finite_differences[0]) / 3
    errors = [float(np.max(np.abs(g - extrapolated))) for g in gradients]
    energy_error = abs(energies[1] - energies[0])
    gradient_error = float(np.max(np.abs(gradients[1] - gradients[0])))
    source_paths = [Path(__file__).resolve(),
        ROOT / "build/vendor/skala/src/skala/pyscf/gradients.py",
        ROOT / "build/vendor/skala/src/skala/gpu4pyscf/gradients.py"]
    report.update({
        "status": "passed" if (max(errors) < 1e-6 and energy_error < 1e-8
                               and gradient_error < 1e-6) else "failed",
        "functional": "Torch PBE, not neural Skala", "basis": mol.basis,
        "auxbasis": auxbasis, "charge": charge, "spin": spin,
        "coords_bohr": coords.tolist(), "atom_grid": [30, 110], "prune": None,
        "energies": energies, "gradients": [g.tolist() for g in gradients],
        "steps_bohr": [1e-3, 5e-4],
        "finite_differences": [g.tolist() for g in finite_differences],
        "extrapolated_gradient": extrapolated.tolist(),
        "cpu_gpu_energy_error": energy_error,
        "cpu_gpu_gradient_max_error": gradient_error,
        "finite_difference_max_errors": errors,
        "finite_difference_step_difference": float(np.max(np.abs(
            finite_differences[1] - finite_differences[0]))),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
    })
    path.write_text(json.dumps(report, indent=2) + "\n")
    assert abs(energies[1]-energies[0]) < 1e-8
    assert report["cpu_gpu_gradient_max_error"] < 1e-6
    assert max(errors) < 1e-6
