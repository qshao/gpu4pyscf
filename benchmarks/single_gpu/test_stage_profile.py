"""Current-code stage profiling and basis/system scaling; no production changes."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import pytest
from pyscf import gto

from df_profile import ROOT
from molecular import SETTINGS, _file_identity, _metadata, compare_cases, run_case


@contextmanager
def stage_profile():
    import cupy as cp
    from gpu4pyscf.df import df, df_jk
    from gpu4pyscf.df.grad import rhf as df_rhf, rks as df_rks
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rhf, rks
    from gpu4pyscf.scf import hf

    targets = [
        (df_jk._DFHF, "get_veff", "scf.veff"),
        (df_jk._DFHF, "get_jk", "scf.jk"),
        (numint.NumInt, "nr_rks", "scf.xc_quadrature"),
        (hf.SCF, "eig", "scf.diagonalization"),
        (df.DF, "build", "scf.df_build_nested_in_jk"),
        (rhf.GradientsBase, "_hcore_energy", "gradient.one_electron"),
        (df_rks.Gradients, "energy_ee", "gradient.two_electron"),
        (df_rks.Gradients, "jk_energy_per_atom", "gradient.df_jk_total"),
        (df_rhf, "_jk_energy_per_atom", "gradient.df_jk_core_nested"),
        (rks, "get_exc_full_response", "gradient.xc_full_response"),
    ]
    result = {name: {"calls": 0, "seconds": 0.} for _, _, name in targets}

    def timed(original, label):
        @wraps(original)
        def wrapped(*args, **kwargs):
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                cp.cuda.Device().synchronize()
                result[label]["seconds"] += time.perf_counter() - start
                result[label]["calls"] += 1
        return wrapped

    with ExitStack() as stack:
        for owner, name, label in targets:
            stack.enter_context(patch.object(owner, name, timed(getattr(owner, name), label)))
        yield result


@pytest.mark.special
@pytest.mark.benchmark
def test_stage_profile_and_scaling():
    output = Path(__file__).with_name("stage_profile_results.json")
    output.write_text(json.dumps({"status": "running", "cases": []}) + "\n")
    report = dict(_metadata("gpu"), status="running", cases=[],
        profiler_sha256=_file_identity(__file__)["sha256"],
        helper_sha256=_file_identity(ROOT / "benchmarks/single_gpu/molecular.py")["sha256"],
        note="Shared GB10. Current code only. Each case has CPU reference, GPU warmup, three uninstrumented controls and one synchronized instrumented run. Instrumentation perturbs scheduling. Stage times are inclusive and nested; do not sum them. Timings exclude object construction, initial guess, imports and metadata. Grid level1 unpruned is a regression setting, not production convergence validation.")
    matrix = [("vitamin_svp", "Vitamin_C.xyz", "def2-svp"),
              ("vitamin_tzvp", "Vitamin_C.xyz", "def2-tzvp"),
              ("tamoxifen_svp", "Tamoxifen.xyz", "def2-svp")]
    for label, filename, basis in matrix:
        geometry = ROOT / "examples" / filename
        for method in ("PBE", "B3LYP"):
            row = dict(case=label, method=method, basis=basis, status="running",
                       geometry_sha256=_file_identity(geometry)["sha256"])
            report["cases"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            with patch.dict(SETTINGS, atom=str(geometry), basis=basis, cpu_threads=4,
                            conv_tol=1e-10, conv_tol_grad=1e-7):
                mol = gto.M(atom=str(geometry), basis=basis, verbose=0)
                row.update(natm=mol.natm, nao=mol.nao)
                print(f"{label} {method}: CPU reference ({mol.nao} AOs)", flush=True)
                cpu = run_case(method, "cpu")
                row["cpu_reference"] = cpu
                warmup = run_case(method, "gpu")
                assert compare_cases(cpu, warmup)["passed"]
                controls = []
                row["controls"] = controls
                for repeat in range(3):
                    current = run_case(method, "gpu")
                    comparison = compare_cases(cpu, current)
                    controls.append(dict(result=current, comparison=comparison))
                    if not comparison["passed"]:
                        row["status"] = report["status"] = "failed"
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    assert comparison["passed"], comparison
                    print(f"{label} {method}: control {repeat+1}/3", flush=True)
                with stage_profile() as stages:
                    instrumented = run_case(method, "gpu")
                comparison = compare_cases(cpu, instrumented)
                row.update(profile=stages, instrumented=instrumented,
                           instrumented_comparison=comparison,
                           status="passed" if comparison["passed"] else "failed")
                for phase in ("scf", "gradient"):
                    row[phase+"_median_seconds"] = statistics.median(
                        r["result"][phase+"_seconds"] for r in controls)
                combined = [r["result"]["scf_seconds"] + r["result"]["gradient_seconds"]
                            for r in controls]
                row["combined_median_seconds"] = statistics.median(combined)
                row["combined_control_range_seconds"] = [min(combined), max(combined)]
                row["instrumented_over_control_ratio"] = (
                    instrumented["scf_seconds"] + instrumented["gradient_seconds"]
                ) / row["combined_median_seconds"]
                row["status"] = "passed" if comparison["passed"] else "failed"
                if not comparison["passed"]:
                    report["status"] = "failed"
                output.write_text(json.dumps(report, indent=2) + "\n")
                assert comparison["passed"], comparison
                print(f"{label} {method}: {row['combined_median_seconds']:.3f}s; "
                      f"instrumentation ratio {row['instrumented_over_control_ratio']:.3f}", flush=True)
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2) + "\n")
