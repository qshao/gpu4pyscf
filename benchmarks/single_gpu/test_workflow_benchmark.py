"""Measure complete small-molecule SCF/gradient workflows, not just kernels."""
import json
from pathlib import Path
import statistics

import pytest

from df_profile import BASELINE, contraction_version
from molecular import _file_identity, _metadata, compare_cases, run_case


@pytest.mark.special
def test_baseline_switch_restores_imported_bindings():
    from gpu4pyscf.lib import cutensor
    from gpu4pyscf.df import df_jk
    original = cutensor.contract
    assert df_jk.contract is original
    with contraction_version("baseline"):
        assert cutensor.contract is not original
        assert df_jk.contract is cutensor.contract
    assert cutensor.contract is original
    assert df_jk.contract is original


@pytest.mark.special
def test_baseline_switch_restores_after_exception_and_lazy_import():
    import sys
    import types
    from gpu4pyscf.lib import cutensor
    original = cutensor.contract
    lazy = types.ModuleType("gpu4pyscf._qualification_lazy_import")
    try:
        with pytest.raises(RuntimeError, match="qualification failure"):
            with contraction_version("baseline"):
                lazy.contract = cutensor.contract
                sys.modules[lazy.__name__] = lazy
                raise RuntimeError("qualification failure")
        assert cutensor.contract is original
        assert lazy.contract is original
    finally:
        sys.modules.pop(lazy.__name__, None)


@pytest.mark.special
@pytest.mark.benchmark
def test_workflow_benchmark():
    report = dict(_metadata("gpu"), baseline=BASELINE,
                  benchmark_sha256=_file_identity(__file__)["sha256"], methods=[],
                  note="Shared GPU. Three alternating paired repetitions after method warmup. SCF/gradient measurements exclude import and result metadata collection.")
    for method in ("HF", "B3LYP"):
        cpu = run_case(method, "cpu")
        run_case(method, "gpu")
        runs = {"baseline": [], "current": []}
        for repeat in range(3):
            order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
            for version in order:
                with contraction_version(version) as source_hash:
                    result = run_case(method, "gpu")
                comparison = compare_cases(cpu, result)
                assert comparison["passed"], comparison
                runs[version].append(dict(result=result, comparison=comparison,
                                          contraction_sha256=source_hash))
        row = dict(method=method, runs=runs,
                   scf_median_seconds={}, gradient_median_seconds={},
                   combined_median_seconds={})
        for version, samples in runs.items():
            row["scf_median_seconds"][version] = statistics.median(s["result"]["scf_seconds"] for s in samples)
            row["gradient_median_seconds"][version] = statistics.median(s["result"]["gradient_seconds"] for s in samples)
            row["combined_median_seconds"][version] = statistics.median(
                s["result"]["scf_seconds"] + s["result"]["gradient_seconds"] for s in samples)
        row["combined_speedup"] = row["combined_median_seconds"]["baseline"] / row["combined_median_seconds"]["current"]
        report["methods"].append(row)
        print(method, "SCF+gradient speedup:", row["combined_speedup"])
    Path(__file__).with_name("workflow_results.json").write_text(json.dumps(report, indent=2) + "\n")
