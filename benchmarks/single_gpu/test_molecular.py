"""Explicit scientific reference and GPU qualification benchmarks."""
import json
from pathlib import Path

import numpy as np
import pytest

from molecular import METHODS, run_case, run_dfmp2, compare_cases


RESULTS = Path(__file__).with_name("molecular_results")


def record(name, result):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.benchmark
@pytest.mark.parametrize("method", METHODS)
def test_cpu_reference(method):
    result = run_case(method, "cpu", hessian=(method == "HF"))
    assert result["source_revision"]
    assert result["source_hashes"]["benchmarks/single_gpu/molecular.py"]
    assert result["converged"]
    assert result["iterations"] > 0
    assert np.isfinite(result["energy_hartree"])
    gradient = np.asarray(result["gradient_hartree_per_bohr"])
    assert gradient.shape == (3, 3)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-6)
    if method == "HF":
        hessian = np.asarray(result["hessian_hartree_per_bohr2"])
        assert hessian.shape == (3, 3, 3, 3)
        np.testing.assert_allclose(hessian, hessian.transpose(1, 0, 3, 2), atol=1e-8)
    record(method.lower() + "_cpu", result)


@pytest.mark.special
@pytest.mark.parametrize("method", METHODS)
def test_gpu_matches_cpu(method):
    # Run matched references in the same process and with the same settings.
    cpu = run_case(method, "cpu", hessian=(method == "HF"))
    gpu = run_case(method, "gpu", hessian=(method == "HF"))
    comparison = compare_cases(cpu, gpu)
    record(method.lower() + "_comparison", dict(cpu=cpu, gpu=gpu, comparison=comparison))
    assert comparison["passed"], comparison


@pytest.mark.benchmark
def test_cpu_dfmp2_reference():
    result = run_dfmp2("cpu")
    assert result["timing_scope"] == "constructor, auxiliary setup and kernel"
    assert result["scf_converged"]
    assert result["correlation_energy_hartree"] < 0
    assert np.isfinite(result["correlation_energy_hartree"])
    record("dfmp2_cpu", result)


@pytest.mark.special
def test_gpu_dfmp2_matches_cpu():
    cpu, gpu = run_dfmp2("cpu"), run_dfmp2("gpu")
    error = abs(cpu["correlation_energy_hartree"] - gpu["correlation_energy_hartree"])
    record("dfmp2_comparison", dict(cpu=cpu, gpu=gpu, absolute_error_hartree=error))
    assert cpu["scf_converged"] and gpu["scf_converged"]
    assert error <= 1e-8


def test_comparison_rejects_inaccurate_or_unconverged_results():
    good = dict(method="HF", settings={"basis": "def2-svp"}, converged=True,
                energy_hartree=-75., gradient_hartree_per_bohr=[[0., 0., 0.]])
    assert compare_cases(good, good)["passed"]
    bad = dict(good, energy_hartree=-74.)
    assert not compare_cases(good, bad)["passed"]
    bad = dict(good, converged=False)
    assert not compare_cases(good, bad)["passed"]
    bad = dict(good, settings={"basis": "sto-3g"})
    with pytest.raises(ValueError, match="settings"):
        compare_cases(good, bad)


def test_gradient_and_hessian_comparison_gates():
    good = dict(method="HF", settings={"basis": "def2-svp"}, converged=True,
                energy_hartree=-75., gradient_hartree_per_bohr=[[0., 0., 0.]],
                hessian_hartree_per_bohr2=np.zeros((1, 1, 3, 3)).tolist())
    bad = dict(good, gradient_hartree_per_bohr=[[0., 0., 1e-4]])
    assert not compare_cases(good, bad)["passed"]
    hessian = np.zeros((1, 1, 3, 3))
    hessian[0, 0, 0, 0] = 1e-3
    bad = dict(good, hessian_hartree_per_bohr2=hessian.tolist())
    assert not compare_cases(good, bad)["passed"]
    bad = dict(good, gradient_hartree_per_bohr=[[0., 0., float("nan")]])
    assert not compare_cases(good, bad)["passed"]
    bad = dict(good, gradient_hartree_per_bohr=np.zeros((2, 3)).tolist())
    with pytest.raises(ValueError, match="shapes"):
        compare_cases(good, bad)


def test_comparison_requires_symmetric_hessian_presence():
    good = dict(method="HF", settings={"basis": "def2-svp"}, converged=True,
                energy_hartree=-75., gradient_hartree_per_bohr=[[0., 0., 0.]])
    with_hessian = dict(good, hessian_hartree_per_bohr2=np.zeros((1, 1, 3, 3)).tolist())
    with pytest.raises(ValueError, match="Hessian"):
        compare_cases(good, with_hessian)
    with pytest.raises(ValueError, match="Hessian"):
        compare_cases(with_hessian, good)
