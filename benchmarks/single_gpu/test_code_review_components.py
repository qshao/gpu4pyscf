"""Opt-in, pinned pre-review versus current source component measurements.

Run exactly this node with -m benchmark. Imports at collection time do not
initialize CUDA. The output is
exclusive: archive an existing components.json explicitly before a fresh run.
"""
import ast
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
import types

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "code_review_results"
SOURCES = ("gpu4pyscf/lib/cutensor.py", "gpu4pyscf/grad/rks.py")
HELPERS = ("benchmarks/single_gpu/test_code_review_components.py",
           "benchmarks/single_gpu/phase_memory.py",
           "benchmarks/single_gpu/molecular.py")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_identity():
    manifest = RESULTS / "before_manifest.json"
    identity = _sha(manifest)
    sources = json.loads(manifest.read_text())["sources"]
    for name, expected in sources.items():
        assert _sha(RESULTS / "before" / name) == expected, name
    return dict(manifest_sha256=identity, verified_file_count=len(sources),
                source_hashes={name: sources[name] for name in SOURCES})


def _source_identity():
    return {name: _sha(ROOT / name) for name in SOURCES + HELPERS}


def _load_contract(source, filename, cp, np):
    """Compile the verbatim function ASTs, without importing another package."""
    tree = ast.parse(source, filename=filename)
    names = {"_matrix_product_axes", "_contract_einsum"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    module = types.ModuleType("component_" + Path(filename).stem)
    module.__dict__.update(cupy=cp, np=np, lru_cache=lru_cache)
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"),
         module.__dict__)
    return module._contract_einsum


def _grid_updates(source, filename):
    tree = ast.parse(source, filename=filename)
    function, = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "get_exc_full_response"]
    # Locate the actual batch loop by its response accumulator update, then
    # retain its weighted updates in production execution order.
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)
             and any(isinstance(statement, ast.AugAssign)
                     and isinstance(statement.target, ast.Name)
                     and statement.target.id == "de_grid_response_weight"
                     for statement in node.body)]
    loop, = loops
    updates = [node for node in loop.body if isinstance(node, ast.AugAssign)
               and isinstance(node.target, ast.Name)
               and node.target.id in ("dweight_dA", "de_grid_response_weight")]
    assert updates and updates[-1].target.id == "de_grid_response_weight"
    helper = ast.parse("def update(dweight_dA, dweightdA_right, "
                       "de_grid_response_weight, g0, g1):\n    pass\n").body[0]
    helper.body = updates + [ast.Return(value=ast.Name(
        id="de_grid_response_weight", ctx=ast.Load()))]
    tree = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    return compile(tree, filename, "exec"), "\n".join(map(ast.unparse, updates))


def _measure(cp, func, repeats, prepare=None):
    if prepare is not None:
        # Consumable batches require restoration before each call. Neither
        # reset, copy nor its synchronization enters either clock's interval.
        observations = []
        for _ in range(repeats):
            prepare()
            cp.cuda.Device().synchronize()
            observations.append(_measure(cp, func, 1))
        return dict(wall_us=statistics.median(s["wall_us"] for s in observations),
                    gpu_us=statistics.median(s["gpu_us"] for s in observations),
                    calls=observations)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.get_current_stream().synchronize()
    wall_start = time.perf_counter()
    start.record()
    result = None
    for _ in range(repeats):
        result = func()
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1e6 / repeats
    gpu_us = cp.cuda.get_elapsed_time(start, end) * 1000 / repeats
    # Keep the final returned storage alive until synchronization completes.
    del result
    return dict(wall_us=wall_us, gpu_us=gpu_us)


def _errors(np, actual, reference):
    assert actual.shape == reference.shape
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, max_absolute_error=None,
                    relative_to_reference_max=None, atol=1e-9, rtol=1e-12,
                    passed=False)
    difference = np.abs(actual - reference)
    absolute = float(np.max(difference, initial=0.))
    scale = float(np.max(np.abs(reference), initial=0.))
    passed = finite and bool(np.all(difference <= 1e-9 + 1e-12 * np.abs(reference)))
    return dict(finite=finite, max_absolute_error=absolute,
                relative_to_reference_max=absolute / max(scale, np.finfo(float).tiny),
                atol=1e-9, rtol=1e-12, passed=passed)


def _input_hash(np, *arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _run_component(cp, np, row, funcs, reference, inputs, repeats, prepares=None):
    from phase_memory import phase_memory

    row["input_sha256_before"] = _input_hash(np, *inputs)
    row["repeats_per_sample"] = repeats
    row["checks"] = {}
    values = {}
    for name, func in funcs.items():
        for _ in range(3):
            if prepares is not None:
                prepares[name]()
                cp.cuda.Device().synchronize()
            result = func()
        cp.cuda.Device().synchronize()
        values[name] = cp.asnumpy(result)
        row["checks"][name + "_reference"] = _errors(np, values[name], reference)
    row["checks"]["current_baseline"] = _errors(np, values["current"], values["baseline"])
    assert all(check["passed"] for check in row["checks"].values()), row["checks"]
    samples = {name: [] for name in funcs}
    row["samples"] = samples
    for pair in range(7):
        order = list(funcs) if pair % 2 == 0 else list(reversed(funcs))
        for name in order:
            samples[name].append(_measure(
                cp, funcs[name], repeats,
                None if prepares is None else prepares[name]))
    row["median_us"] = {
        name: {kind: statistics.median(s[kind] for s in observations)
               for kind in ("wall_us", "gpu_us")}
        for name, observations in samples.items()}
    row["speedup"] = {
        kind: row["median_us"]["baseline"][kind] / row["median_us"]["current"][kind]
        for kind in ("wall_us", "gpu_us")}
    row["requested_allocation"] = {}
    # Inputs, matrix destinations, references and warmed kernels precede hooks.
    # Each implementation gets an independent context; no timing uses hooks.
    for name, func in funcs.items():
        if prepares is not None:
            prepares[name]()
        cp.cuda.Device().synchronize()
        with phase_memory() as tracker:
            with tracker.phase(name):
                owner = func()
                cp.cuda.Device().synchronize()
            row["requested_allocation"][name] = json.loads(json.dumps(tracker.data))
        del owner
    row["input_sha256_after"] = _input_hash(np, *inputs)
    assert row["input_sha256_after"] == row["input_sha256_before"]


@pytest.mark.benchmark
def test_code_review_components():
    """One selected node records all component cases, including failures."""
    output = RESULTS / "components.json"
    # Exclusive creation protects historical evidence, including failed runs.
    with output.open("x") as destination:
        report = dict(status="failed", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      scope="Isolated contraction and grid weighted-reduction components only; "
                            "no molecular workflow timing or physical memory bound claims.",
                      baseline="Pinned source-only pre-review working-tree snapshot, not HEAD",
                      cases=[])
        try:
            report["baseline_identity_before"] = _baseline_identity()
            report["source_hashes_before"] = _source_identity()
            import cupy as cp
            import numpy as np
            from molecular import _metadata

            assert cp.cuda.runtime.getDeviceCount() == 1, "Exactly one visible GPU required"
            report["metadata_before"] = _metadata("gpu")
            source = {
                side: {name: (base / name).read_text() for name in SOURCES}
                for side, base in (("baseline", RESULTS / "before"), ("current", ROOT))}
            contracts = {side: _load_contract(files[SOURCES[0]], side + "/" + SOURCES[0], cp, np)
                         for side, files in source.items()}
            updates = {side: _grid_updates(files[SOURCES[1]], side + "/" + SOURCES[1])
                       for side, files in source.items()}
            report["grid_update_statements"] = {side: value[1] for side, value in updates.items()}
            grid_helpers = {}
            for name, (code, _) in updates.items():
                namespace = dict(cupy=cp)
                exec(code, namespace)
                grid_helpers[name] = namespace["update"]
            rng = np.random.default_rng(20261018)
            for n, dtype in ((64, np.float64), (512, np.float64), (1024, np.float64),
                             (512, np.complex128)):
                host = [rng.uniform(-1, 1, (n, n)).astype(dtype) for _ in range(2)]
                if np.issubdtype(dtype, np.complexfloating):
                    for value in host:
                        value += 1j * rng.uniform(-1, 1, value.shape)
                a, b = map(cp.asarray, host)
                outputs = {name: cp.empty((n, n), dtype=dtype) for name in contracts}
                funcs = {name: (lambda f=f, out=outputs[name]:
                                f("ik,kj->ij", a, b, 1., 0., out))
                         for name, f in contracts.items()}
                row = dict(case=f"matrix_{n}_{np.dtype(dtype).name}", shapes=[list(a.shape), list(b.shape)],
                           reference="Full GPU matmul control plus independent small NumPy matmul oracle")
                report["cases"].append(row)
                reference = cp.asnumpy(cp.matmul(a, b))
                # NumPy only computes a small matrix; no large CPU BLAS workload.
                small_a, small_b = a[:8, :11].copy(), b[:11, :7].copy()
                small_ref = host[0][:8, :11] @ host[1][:11, :7]
                row["cpu_oracle_checks"] = {}
                for name, function in contracts.items():
                    small_out = cp.empty((8, 7), dtype=dtype)
                    value = function("ik,kj->ij", small_a, small_b, 1., 0., small_out)
                    check = _errors(np, cp.asnumpy(value), small_ref)
                    row["cpu_oracle_checks"][name] = check
                    assert check["passed"], check
                _run_component(cp, np, row, funcs, reference, host, 30 if n == 64 else 5)
                assert _input_hash(np, cp.asnumpy(a), cp.asnumpy(b)) == row["input_sha256_before"]
            for natm, ngrids in ((3, 4096), (113, 32768)):
                host_d = rng.uniform(-1, 1, (natm, 3, ngrids))
                host_r = rng.uniform(-1, 1, ngrids)
                derivative, right = cp.asarray(host_d), cp.asarray(host_r)
                working = {name: cp.empty_like(derivative) for name in grid_helpers}
                accumulators = {name: cp.empty((natm, 3)) for name in grid_helpers}
                funcs = {name: (lambda f=helper, work=working[name], out=accumulators[name]:
                                f(work, right, out, 0, ngrids))
                         for name, helper in grid_helpers.items()}
                def restore(work, out):
                    cp.copyto(work, derivative)
                    out.fill(0)
                prepares = {name: (lambda work=working[name], out=accumulators[name]:
                                   restore(work, out)) for name in grid_helpers}
                row = dict(case=f"grid_{natm}_{ngrids}", shapes=[list(derivative.shape), list(right.shape)],
                           derivative_input_bytes=derivative.nbytes,
                           reference="Full independent NumPy weighted reduction",
                           preparation="Each implementation has a dedicated owned consumable derivative "
                                       "and accumulator. Before every call, copy immutable master to work, "
                                       "zero accumulator and synchronize outside timing/allocation hooks.",
                           sample_aggregation="Median of separately synchronized per-call observations; "
                                              "seven alternating baseline/current pairs")
                report["cases"].append(row)
                reference = np.einsum("Adg,g->Ad", host_d, host_r)
                _run_component(cp, np, row, funcs, reference, (host_d, host_r),
                               30 if natm == 3 else 5, prepares=prepares)
                assert _input_hash(np, cp.asnumpy(derivative), cp.asnumpy(right)) == row["input_sha256_before"]
            report["metadata_after"] = _metadata("gpu")
            report["source_hashes_after"] = _source_identity()
            report["baseline_identity_after"] = _baseline_identity()
            assert report["source_hashes_after"] == report["source_hashes_before"]
            assert report["baseline_identity_after"] == report["baseline_identity_before"]
            for key in ("native_libraries", "source_hashes"):
                assert report["metadata_before"][key] == report["metadata_after"][key], key
            report["status"] = "passed"
        except BaseException:
            report["error"] = traceback.format_exc()
            raise
        finally:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
    print("Component evidence:", output)
