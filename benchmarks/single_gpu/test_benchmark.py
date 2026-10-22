"""Explicit synchronized comparison against the pinned upstream fallback."""
import json
import hashlib
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import time
import types
import warnings

import cupy as cp
import numpy as np
import pytest
from test_contraction import wrapper

BASELINE = "9af748a71ca8f4881c604332c2f0cc1632e033ac"


def measure(func, repeats=30):
    stream = cp.cuda.get_current_stream()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    stream.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        func()
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - wall_start) / repeats
    return wall * 1e6, cp.cuda.get_elapsed_time(start, end) * 1000 / repeats


@pytest.mark.benchmark
def test_benchmark(wrapper):
    assert cp.cuda.runtime.getDeviceCount() == 1
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "-C", str(root), "show", BASELINE + ":gpu4pyscf/lib/cutensor.py"],
        text=True)
    baseline = types.ModuleType("baseline_cutensor")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exec(compile(source, "baseline_cutensor.py", "exec"), baseline.__dict__)
    rng = np.random.default_rng(5)
    cases = [
        ("small_matrix", "ik,kj->ij", (64, 64), (64, 64)),
        ("medium_matrix", "ik,kj->ij", (512, 512), (512, 512)),
        ("df_j", "np,Lp->nL", (4, 4096), (256, 4096)),
        ("df_j_accumulate", "nL,Lp->np", (4, 256), (256, 4096)),
        ("df_k_transform", "Lij,njk->nikL", (32, 64, 64), (2, 64, 16)),
    ]
    results = []
    for label, pattern, ashape, bshape in cases:
        host_a, host_b = rng.random(ashape), rng.random(bshape)
        a, b = cp.asarray(host_a), cp.asarray(host_b)
        memory_before = cp.get_default_memory_pool().used_bytes()
        reference = np.einsum(pattern, host_a, host_b)
        out = cp.zeros(reference.shape)
        beta = 1. if label.endswith("accumulate") else 0.
        funcs = {
            "baseline": lambda: baseline._contract_einsum(pattern, a, b, 1., beta, out),
            "modified": lambda: wrapper._contract_einsum(pattern, a, b, 1., beta, out),
        }
        cold, timings = {}, {name: [] for name in funcs}
        for name, func in funcs.items():
            out.fill(0)
            cold[name] = measure(func, repeats=1)[0]
            cp.testing.assert_allclose(out, reference, rtol=1e-12, atol=1e-12)
            for _ in range(5):
                func()
        for batch in range(7):
            names = list(funcs) if batch % 2 == 0 else list(reversed(funcs))
            for name in names:
                timings[name].append(measure(funcs[name]))
        row = dict(case=label, pattern=pattern, shapes=[ashape, bshape],
                   cold_wall_us=cold,
                   samples_us=timings,
                   pool_used_before_bytes=memory_before,
                   pool_used_after_bytes=cp.get_default_memory_pool().used_bytes(),
                   warm_wall_us={name: statistics.median(t[0] for t in ts) for name, ts in timings.items()},
                   warm_gpu_us={name: statistics.median(t[1] for t in ts) for name, ts in timings.items()})
        row["speedup"] = row["warm_wall_us"]["baseline"] / row["warm_wall_us"]["modified"]
        results.append(row)
    props = cp.cuda.runtime.getDeviceProperties(0)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  machine=platform.machine(), numpy=np.__version__,
                  driver_version=cp.cuda.runtime.driverGetVersion(),
                  runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                  pyscf=importlib.metadata.version("pyscf"),
                  modified_source_sha256=hashlib.sha256((root / "gpu4pyscf/lib/cutensor.py").read_bytes()).hexdigest(),
                  baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  baseline=BASELINE, cupy=cp.__version__, gpu=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                  compute_capability=cp.cuda.Device().compute_capability,
                  note="Shared GPU; synchronized per-call timings. Cold includes compilation/cache effects. No native molecular libraries required.",
                  results=results)
    output = Path(__file__).with_name("contraction_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(f"{row['case']}: {row['warm_wall_us']['baseline']:.2f} -> "
              f"{row['warm_wall_us']['modified']:.2f} us; {row['speedup']:.2f}x")
    print("Full report:", output)
