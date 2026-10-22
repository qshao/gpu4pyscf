"""Checkpoint-independent qualification of Skala's PBE SCF/force integration.

These tests exercise real Skala infrastructure, but do not qualify neural Skala.
"""
import json
import time
from datetime import datetime, timezone
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_preflight import ROOT  # applies the explicit read-only dependency overlay


@pytest.mark.parametrize("charge,spin", [(0, 0), (1, 1)])
@pytest.mark.parametrize("density_fit", [False, True])
def test_pbe_energy_and_gradient(charge, spin, density_fit):
    mode = "df" if density_fit else "direct"
    path = ROOT / f"benchmarks/external_qm/pbe_{mode}_charge{charge}_results.json"
    provenance = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "skala_revision": subprocess.check_output(
            ["git", "-C", str(ROOT / "build/vendor/skala"), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "gpu4pyscf/dft/numint.py", Path(__file__).resolve(),
                      ROOT / "build/vendor/skala/src/skala/pyscf/gradients.py",
                      ROOT / "build/vendor/skala/src/skala/gpu4pyscf/gradients.py")
        },
    }
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    import cupy as cp
    import torch
    from pyscf import dft, gto, lib
    from skala.pyscf import SkalaKS as CpuKS
    from skala.gpu4pyscf import SkalaKS as GpuKS
    from skala.pyscf.grids import SkalaGrids

    lib.num_threads(2)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
                basis="def2-svp", charge=charge, spin=spin, verbose=0)
    auxbasis = "def2-universal-jkfit" if density_fit else None
    native = (dft.RKS(mol) if spin == 0 else dft.UKS(mol))
    native.xc = "pbe"
    if density_fit:
        native = native.density_fit(auxbasis=auxbasis)
    native.grids = SkalaGrids(mol)
    constructors = {
        "native_cpu": lambda: native,
        "skala_pbe_cpu": lambda: CpuKS(mol, xc="pbe", with_dftd3=False,
                                        with_density_fit=density_fit, auxbasis=auxbasis),
        "skala_pbe_gpu": lambda: GpuKS(mol, xc="pbe", with_dftd3=False,
                                        with_density_fit=density_fit, auxbasis=auxbasis),
    }
    result = {**provenance, "functional": "PBE (not neural Skala)", "charge": charge,
              "spin": spin, "basis": mol.basis, "auxbasis": auxbasis,
              "density_fit": density_fit,
              "atom_grid": [30, 110], "prune": None, "with_dftd3": False,
              "conv_tol": 1e-11, "conv_tol_grad": 1e-8, "backends": {}}
    reference_coords = reference_weights = None
    for name, constructor in constructors.items():
        mf = constructor()
        mf.grids.atom_grid = (30, 110)
        mf.grids.prune = None
        mf.grids.radi_method = dft.radi.treutler
        mf.small_rho_cutoff = 0
        mf.conv_tol = result["conv_tol"]
        mf.conv_tol_grad = result["conv_tol_grad"]
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        energy = float(mf.kernel())
        cp.cuda.Stream.null.synchronize()
        scf_seconds = time.perf_counter() - start
        assert mf.converged, name
        coords, weights = mf.grids.coords, mf.grids.weights
        coords = coords.get() if hasattr(coords, "get") else np.asarray(coords)
        weights = weights.get() if hasattr(weights, "get") else np.asarray(weights)
        if reference_coords is None:
            reference_coords, reference_weights = coords, weights
        else:
            np.testing.assert_allclose(coords, reference_coords, rtol=0, atol=1e-12)
            np.testing.assert_allclose(weights, reference_weights, rtol=0, atol=1e-12)
        grad = mf.nuc_grad_method()
        # Skala differentiates quadrature positions and weights internally.
        # Its CPU UKS extra_force inherits PySCF and must retain its default flag.
        if name == "native_cpu":
            grad.grid_response = True
        grad.auxbasis_response = True
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gradient = grad.kernel()
        cp.cuda.Stream.null.synchronize()
        gradient_seconds = time.perf_counter() - start
        gradient = gradient.get() if hasattr(gradient, "get") else np.asarray(gradient)
        assert gradient.shape == (3, 3) and np.isfinite(gradient).all()
        assert np.isfinite(energy)
        result["backends"][name] = {
            "energy": energy, "gradient": gradient.tolist(),
            "scf_seconds": scf_seconds, "gradient_seconds": gradient_seconds,
            "converged": bool(mf.converged),
        }
    # Compare the same Torch functional across devices. The independent native
    # libxc PBE comparison is recorded separately; it is not assumed identical.
    ref = result["backends"]["skala_pbe_cpu"]
    for name, value in result["backends"].items():
        value["energy_error"] = abs(value["energy"] - ref["energy"])
        value["gradient_max_error"] = float(np.max(np.abs(
            np.asarray(value["gradient"]) - np.asarray(ref["gradient"]))))
    step = 1e-3  # Bohr; independent derivative check on oxygen z
    displaced_energies = []
    for direction in (-1, 1):
        coords = mol.atom_coords().copy()
        coords[0, 2] += direction * step
        displaced_mol = mol.set_geom_(coords, unit="Bohr", inplace=False)
        displaced = CpuKS(displaced_mol, xc="pbe", with_dftd3=False,
                          with_density_fit=density_fit, auxbasis=auxbasis)
        displaced.grids.atom_grid = (30, 110)
        displaced.grids.prune = None
        displaced.grids.radi_method = dft.radi.treutler
        displaced.conv_tol = result["conv_tol"]
        displaced.conv_tol_grad = result["conv_tol_grad"]
        displaced_energies.append(float(displaced.kernel()))
        assert displaced.converged
    fd = (displaced_energies[1] - displaced_energies[0]) / (2 * step)
    result["finite_difference"] = {
        "component": "oxygen_z", "step_bohr": step,
        "gradient": fd, "analytic_gradient": ref["gradient"][0][2],
        "absolute_error": abs(fd - ref["gradient"][0][2]),
    }
    result["status"] = "passed" if (
        result["backends"]["skala_pbe_gpu"]["energy_error"] < 1e-8
        and result["backends"]["skala_pbe_gpu"]["gradient_max_error"] < 1e-6
        and result["finite_difference"]["absolute_error"] < 1e-6
    ) else "failed"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, value in result["backends"].items():
        if name == "native_cpu":
            continue
        assert value["energy_error"] < 1e-8, name
        assert value["gradient_max_error"] < 1e-6, name
    assert result["finite_difference"]["absolute_error"] < 1e-6
