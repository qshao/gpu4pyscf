"""Profile actual DF staging/unpacking and J/K on the tracked vitamin-C geometry."""
from contextlib import contextmanager
import gc
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import warnings

import numpy as np

from molecular import _metadata, _file_identity

BASELINE = "9af748a71ca8f4881c604332c2f0cc1632e033ac"
ROOT = Path(__file__).resolve().parents[2]


def _replace_contract(old, new):
    for name, module in list(sys.modules.items()):
        if name.startswith("gpu4pyscf") and isinstance(module, types.ModuleType):
            for attr, value in list(vars(module).items()):
                if value is old:
                    setattr(module, attr, new)


@contextmanager
def contraction_version(version):
    """Temporarily replace imported contraction bindings, including lazy imports."""
    from gpu4pyscf.lib import cutensor
    if cutensor.contract_engine != "cupy":
        raise RuntimeError("This comparison qualifies the CuPy fallback only")
    if version == "current":
        yield _file_identity(ROOT / "gpu4pyscf/lib/cutensor.py")["sha256"]
        return
    if version != "baseline":
        raise ValueError("Unknown contraction version")
    source = subprocess.check_output([
        "git", "-C", str(ROOT), "show", BASELINE + ":gpu4pyscf/lib/cutensor.py",
    ], text=True)
    baseline = types.ModuleType("baseline_cutensor")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exec(compile(source, "baseline_cutensor.py", "exec"), baseline.__dict__)
    original = cutensor.contract
    _replace_contract(original, baseline.contract)
    try:
        yield hashlib.sha256(source.encode()).hexdigest()
    finally:
        _replace_contract(baseline.contract, original)


def _measure(func, repeats=5):
    import cupy as cp
    func()
    cp.cuda.Device().synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        cp.cuda.Device().synchronize()
        samples.append(time.perf_counter() - start)
    return dict(median_seconds=statistics.median(samples), samples_seconds=samples)


def _paired_jk(obj, dm):
    timings = {name: dict(samples_seconds=[]) for name in ("baseline", "current")}
    outputs = {}
    for repeat in range(7):
        order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
        for version in order:
            with contraction_version(version) as source_hash:
                # Warm up after each binding switch before timing the next call.
                sample = _measure(lambda: obj.get_jk(dm, hermi=1), repeats=1)
                timings[version]["samples_seconds"].extend(sample["samples_seconds"])
                timings[version]["contraction_sha256"] = source_hash
                if repeat == 0:
                    outputs[version] = obj.get_jk(dm, hermi=1)
    for result in timings.values():
        result["median_seconds"] = statistics.median(result["samples_seconds"])
    return timings, outputs


def profile_df():
    import cupy as cp
    from pyscf import gto, lib, scf
    from gpu4pyscf.df.df import DF
    from gpu4pyscf.lib.cupy_helper import tag_array

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    geometry = ROOT / "examples/Vitamin_C.xyz"
    auxbasis = "def2-universal-jkfit"
    with lib.with_omp_threads(2):
        mol = gto.M(atom=str(geometry), basis="def2-svp", verbose=0)
        cpu = scf.RHF(mol).density_fit(auxbasis=auxbasis).run(conv_tol=1e-11)
        if not cpu.converged:
            raise RuntimeError("CPU HF reference did not converge")
        host_dm = cpu.make_rdm1()
        ref_j, ref_k = cpu.get_jk(dm=host_dm)
        dm = tag_array(cp.asarray(host_dm), mo_coeff=cp.asarray(cpu.mo_coeff),
                       mo_occ=cp.asarray(cpu.mo_occ))
        report = dict(_metadata("gpu"), baseline=BASELINE, natm=mol.natm,
                      nao=mol.nao, basis="def2-svp", auxbasis=auxbasis,
                      geometry_sha256=_file_identity(geometry)["sha256"],
                      profile_source_sha256=_file_identity(__file__)["sha256"],
                      cpu_scf_converged=bool(cpu.converged), storage=[],
                      note="Shared GPU. Seven alternating paired warm J/K samples. Loop-only timings exclude contractions. Block sizes are temporary overrides. Default-pool live/total bytes exclude large allocations routed directly to CUDA and are not peak memory.")
        for resident in (True, False):
            obj = DF(mol, auxbasis=auxbasis)
            obj.use_gpu_memory = resident
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            obj.build()
            cp.cuda.Device().synchronize()
            tensor = obj._cderi[0]
            row = dict(requested_storage="gpu" if resident else "host",
                       actual_storage="gpu" if isinstance(tensor, cp.ndarray) else "host",
                       setup_seconds=time.perf_counter() - start,
                       tensor_bytes=tensor.nbytes, naux=tensor.shape[0],
                       default_pool_live_bytes=cp.get_default_memory_pool().used_bytes(),
                       default_pool_total_bytes=cp.get_default_memory_pool().total_bytes(),
                       device_free_bytes=cp.cuda.runtime.memGetInfo()[0],
                       measurements=[])
            row["default_block_size"] = obj.get_blksize()
            actual_j, actual_k = obj.get_jk(dm, hermi=1)
            row["j_cpu_max_error"] = float(np.max(np.abs(cp.asnumpy(actual_j) - ref_j)))
            row["k_cpu_max_error"] = float(np.max(np.abs(cp.asnumpy(actual_k) - ref_k)))
            del actual_j, actual_k
            for block in (128, 256, 512):
                size = min(block, tensor.shape[0])
                def walk(unpack):
                    for dense, packed in obj.loop(blksize=size, unpack=unpack):
                        pass
                entry = dict(block_size=size,
                             packed_loop=_measure(lambda: walk(False)),
                             unpacked_loop=_measure(lambda: walk(True)), jk={})
                # This small workload has ample memory for each explicit size.
                # Do not use this override as a production memory heuristic.
                obj.get_blksize = lambda *args, **kwargs: size
                try:
                    entry["jk"], outputs = _paired_jk(obj, dm)
                    entry["baseline_current_j_error"] = float(cp.max(cp.abs(outputs["baseline"][0] - outputs["current"][0])).get())
                    entry["baseline_current_k_error"] = float(cp.max(cp.abs(outputs["baseline"][1] - outputs["current"][1])).get())
                    entry["speedup"] = entry["jk"]["baseline"]["median_seconds"] / entry["jk"]["current"]["median_seconds"]
                    del outputs
                finally:
                    # Restore class method lookup, avoiding a bound-method cycle.
                    del obj.get_blksize
                row["measurements"].append(entry)
            report["storage"].append(row)
            del tensor, obj
            gc.collect()
        return report
