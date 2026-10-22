"""Matched single-GPU/CPU scientific qualification, with explicit provenance."""
from datetime import datetime, timezone
from pathlib import Path
import platform
import hashlib
import subprocess
import time
from unittest.mock import patch

import numpy as np
from pyscf import dft, gto, lib, scf
import pyscf

METHODS = ("HF", "PBE", "B3LYP", "WB97X")
SETTINGS = dict(
    atom="O 0 0 0; H 0.94 0 0; H -0.24 0 0.91",
    unit="Angstrom", basis="def2-svp", auxbasis="def2-universal-jkfit",
    conv_tol=1e-12, conv_tol_grad=1e-9, grid_level=1,
    grid_pruning=False, small_rho_cutoff=0., grid_response=True,
    auxbasis_response=True, cpu_threads=2)


def _synchronize(backend):
    if backend == "gpu":
        import cupy
        cupy.cuda.Device().synchronize()


def _timed(backend, func):
    _synchronize(backend)
    start = time.perf_counter()
    result = func()
    _synchronize(backend)
    return result, time.perf_counter() - start


def _host(array):
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def _mean_field(method, backend):
    if method not in METHODS:
        raise ValueError("Unknown method: " + method)
    if backend not in ("cpu", "gpu"):
        raise ValueError("Unknown backend: " + backend)
    mol = gto.M(atom=SETTINGS["atom"], basis=SETTINGS["basis"],
                unit=SETTINGS["unit"], verbose=0)
    if method == "HF":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=method)
        mf.grids.level = SETTINGS["grid_level"]
        mf.grids.prune = None
        mf.small_rho_cutoff = SETTINGS["small_rho_cutoff"]
    mf = mf.density_fit(auxbasis=SETTINGS["auxbasis"])
    mf.conv_tol = SETTINGS["conv_tol"]
    mf.conv_tol_grad = SETTINGS["conv_tol_grad"]
    dm0 = mf.get_init_guess()
    if backend == "gpu":
        import cupy
        import gpu4pyscf
        expected = Path(__file__).resolve().parents[2] / "gpu4pyscf/__init__.py"
        if Path(gpu4pyscf.__file__).resolve() != expected:
            raise RuntimeError("GPU benchmark imported a different GPU4PySCF checkout")
        if cupy.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError("Exactly one visible GPU is required")
        mf = mf.to_gpu()
    return mf, dm0


def _file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def _metadata(backend):
    info = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                backend=backend, machine=platform.machine(),
                pyscf=pyscf.__version__, numpy=np.__version__)
    root = Path(__file__).resolve().parents[2]
    info["source_revision"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    info["source_dirty"] = bool(subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    files = ("benchmarks/single_gpu/molecular.py",
             "benchmarks/single_gpu/test_molecular.py",
             "gpu4pyscf/lib/cutensor.py", "gpu4pyscf/lib/CMakeLists.txt",
             "gpu4pyscf/lib/cmake/CudaArchitectures.cmake")
    info["source_hashes"] = {name: _file_identity(root / name)["sha256"] for name in files}
    info["cpu_libxc"] = dft.libxc.__version__
    if backend == "gpu":
        import cupy
        import gpu4pyscf
        from gpu4pyscf.lib import cupy_helper
        from gpu4pyscf.dft import libxc
        from gpu4pyscf.df import int3c2e_bdiv
        info["gpu4pyscf_version"] = gpu4pyscf.__version__
        info["gpu4pyscf_source"] = gpu4pyscf.__file__
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        info["device_name"] = name.decode() if isinstance(name, bytes) else name
        info["cuda_driver_version"] = cupy.cuda.runtime.driverGetVersion()
        info["cuda_runtime_version"] = cupy.cuda.runtime.runtimeGetVersion()
        info["native_libraries"] = {
            "cupy_helper": _file_identity(cupy_helper.libcupy_helper._name),
            "libxc": _file_identity(libxc._libxc._name),
            "gdft": _file_identity(libxc.libgdft._name),
            "vhf_rys": _file_identity(int3c2e_bdiv.libvhf_rys._name),
        }
        info["cupy"] = cupy.__version__
        info["compute_capability"] = cupy.cuda.Device().compute_capability
    return info


def run_case(method, backend, hessian=False):
    with lib.with_omp_threads(SETTINGS["cpu_threads"]):
        mf, dm0 = _mean_field(method, backend)
        build_times, iterations = [], []
        df_class = type(mf.with_df)
        original_build = df_class.build
        def build(dfobj, *args, **kwargs):
            result, elapsed = _timed(backend, lambda: original_build(dfobj, *args, **kwargs))
            build_times.append(elapsed)
            return result
        previous = time.perf_counter()
        def callback(env):
            nonlocal previous
            _synchronize(backend)
            now = time.perf_counter()
            iterations.append(dict(cycle=int(env["cycle"]), seconds=now - previous,
                                   energy_hartree=float(env["e_tot"])))
            previous = now
        mf.callback = callback
        # Class binding preserves the receiver on copied range-separated DF objects.
        # Only timing is wrapped; the original implementation always executes.
        with patch.object(df_class, "build", build):
            energy, scf_seconds = _timed(backend, lambda: mf.kernel(dm0=dm0))
        if not mf.converged:
            raise RuntimeError(method + " did not converge")
        grad = mf.nuc_grad_method()
        grad.auxbasis_response = SETTINGS["auxbasis_response"]
        if method != "HF":
            grad.grid_response = SETTINGS["grid_response"]
        gradient, gradient_seconds = _timed(backend, grad.kernel)
        result = dict(_metadata(backend), method=method, settings=SETTINGS.copy(),
                      converged=bool(mf.converged), energy_hartree=float(energy),
                      scf_seconds=scf_seconds, iterations=len(iterations),
                      iteration_observations=iterations,
                      df_build_calls_seconds=build_times.copy(),
                      gradient_seconds=gradient_seconds,
                      gradient_hartree_per_bohr=_host(gradient).tolist(),
                      timing_note="SCF includes DF initialization; callback timings synchronize each iteration. No observed DF.build calls means setup was not separately captured.")
        if hessian:
            value, seconds = _timed(backend, lambda: mf.Hessian().kernel())
            result["hessian_seconds"] = seconds
            result["hessian_hartree_per_bohr2"] = _host(value).tolist()
        return result


def run_dfmp2(backend):
    with lib.with_omp_threads(SETTINGS["cpu_threads"]):
        mf, dm0 = _mean_field("HF", backend)
        _, scf_seconds = _timed(backend, lambda: mf.kernel(dm0=dm0))
        if not mf.converged:
            raise RuntimeError("HF reference did not converge")
        auxbasis = "def2-svp-ri"
        if backend == "cpu":
            from pyscf.mp.dfmp2 import DFMP2
            from pyscf import df
            def calculate():
                mp = DFMP2(mf)
                mp.with_df = df.DF(mf.mol, auxbasis=auxbasis)
                return mp.kernel(with_t2=False)
        else:
            from gpu4pyscf.mp.dfmp2 import DFMP2
            def calculate():
                mp = DFMP2(mf, auxbasis=auxbasis)
                return mp.kernel(with_t2=False, fp_type="FP64")
        value, seconds = _timed(backend, calculate)
        return dict(_metadata(backend), settings=SETTINGS.copy(),
                    mp2_auxbasis=auxbasis, precision="FP64",
                    timing_scope="constructor, auxiliary setup and kernel",
                    scf_converged=bool(mf.converged), scf_seconds=scf_seconds,
                    mp2_seconds=seconds, correlation_energy_hartree=float(value[0]))


def compare_cases(cpu, gpu):
    if cpu["settings"] != gpu["settings"] or cpu["method"] != gpu["method"]:
        raise ValueError("Cannot compare different scientific settings")
    if ("hessian_hartree_per_bohr2" in cpu) != ("hessian_hartree_per_bohr2" in gpu):
        raise ValueError("Hessian presence must match")
    def error(key):
        left, right = np.asarray(cpu[key]), np.asarray(gpu[key])
        if left.shape != right.shape:
            raise ValueError("Incompatible result shapes for " + key)
        return float(np.max(np.abs(left - right)))
    result = dict(energy_error_hartree=abs(cpu["energy_hartree"] - gpu["energy_hartree"]),
                  gradient_max_error=error("gradient_hartree_per_bohr"))
    passed = (cpu["converged"] and gpu["converged"] and
              result["energy_error_hartree"] <= 1e-8 and
              result["gradient_max_error"] <= 1e-6)
    if "hessian_hartree_per_bohr2" in cpu:
        result["hessian_max_error"] = error("hessian_hartree_per_bohr2")
        passed = passed and result["hessian_max_error"] <= 1e-5
    result["passed"] = bool(passed)
    return result
