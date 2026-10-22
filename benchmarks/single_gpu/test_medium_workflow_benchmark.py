"""Matched whole workflows on Vitamin C; no production code is modified."""
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import pytest

from df_profile import BASELINE, ROOT, contraction_version
from molecular import SETTINGS, _file_identity, _metadata, compare_cases, run_case


@pytest.mark.special
@pytest.mark.benchmark
def test_medium_workflow_benchmark():
    output = Path(__file__).with_name("medium_workflow_results.json")
    output.write_text(json.dumps({"status": "running"}) + "\n")
    geometry = ROOT / "examples/Vitamin_C.xyz"
    report = dict(_metadata("gpu"), baseline=BASELINE, status="running",
                  benchmark_sha256=_file_identity(__file__)["sha256"],
                  geometry_sha256=_file_identity(geometry)["sha256"],
                  pairs=7, methods=[],
                  note="Shared GPU. Baseline means the pinned contraction implementation in the same current native build, not an independently built old package. Warm up both variants. Seven alternating pairs. Timings include SCF/DF setup and complete gradient, excluding imports/metadata. CPU reference has matched settings; this is not a tuned CPU performance comparison.")
    output.write_text(json.dumps(report, indent=2) + "\n")
    with patch.dict(SETTINGS, atom=str(geometry), conv_tol=1e-10, conv_tol_grad=1e-7):
        for method in ("HF", "PBE", "B3LYP", "WB97X"):
            print(f"Vitamin C {method}: CPU reference", flush=True)
            cpu = run_case(method, "cpu")
            runs = {"baseline": [], "current": []}
            row = dict(method=method, cpu_reference=cpu, runs=runs)
            report["methods"].append(row)
            for version in runs:
                with contraction_version(version):
                    warmup = run_case(method, "gpu")
                assert compare_cases(cpu, warmup)["passed"]
            for repeat in range(report["pairs"]):
                order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                for version in order:
                    with contraction_version(version) as source_hash:
                        result = run_case(method, "gpu")
                    comparison = compare_cases(cpu, result)
                    runs[version].append(dict(result=result, comparison=comparison,
                                              contraction_sha256=source_hash))
                    if not comparison["passed"]:
                        report["status"] = "failed"
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    assert comparison["passed"], comparison
                print(f"Vitamin C {method}: pair {repeat+1}/7", flush=True)
            for component in ("scf", "gradient", "combined"):
                samples = {}
                for version in runs:
                    samples[version] = [
                        (s["result"]["scf_seconds"] + s["result"]["gradient_seconds"])
                        if component == "combined" else s["result"][component+"_seconds"]
                        for s in runs[version]
                    ]
                row[component+"_median_seconds"] = {
                    version: statistics.median(values) for version, values in samples.items()}
                row[component+"_paired_ratios"] = [
                    a/b for a,b in zip(samples["baseline"], samples["current"])]
            row["combined_speedup"] = (row["combined_median_seconds"]["baseline"] /
                                       row["combined_median_seconds"]["current"])
            print(f"Vitamin C {method}: {row['combined_speedup']:.3f}x", flush=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2) + "\n")
