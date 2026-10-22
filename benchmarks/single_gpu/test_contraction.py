"""GPU component tests that do not require the native integral libraries."""
import importlib.util
from pathlib import Path
import sys
import types
import warnings

import cupy as cp
import numpy as np
import pytest


@pytest.fixture
def wrapper(monkeypatch):
    # The contraction module only imports this logger; no native code is needed.
    lib = types.ModuleType("gpu4pyscf.lib")
    lib.logger = types.ModuleType("logger")
    monkeypatch.setitem(sys.modules, "gpu4pyscf.lib", lib)
    path = Path(__file__).resolve().parents[2] / "gpu4pyscf/lib/cutensor.py"
    spec = importlib.util.spec_from_file_location("component_cutensor", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spec.loader.exec_module(module)
    return module


def test_gpu_environment():
    assert cp.cuda.runtime.getDeviceCount() == 1


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("alpha,beta", [(1., 0.), (2., 0.), (.5, .25), (1., 1.)])
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (True, False), (False, True), (True, True)])
def test_matrix_layouts(wrapper, dtype, alpha, beta, trans_a, trans_b):
    rng = np.random.default_rng(18)
    a = rng.random((7, 5)).astype(dtype)
    b = rng.random((5, 9)).astype(dtype)
    if dtype == np.complex128:
        a += 1j * rng.random(a.shape)
        b += 1j * rng.random(b.shape)
    a_modes, b_modes = "ik", "kj"
    if trans_a:
        a = a.T.copy()
        a_modes = a_modes[::-1]
    if trans_b:
        b = b.T.copy()
        b_modes = b_modes[::-1]
    pattern = a_modes + "," + b_modes + "->ij"
    out = cp.ones((7, 9), dtype=dtype)
    reference = alpha * np.einsum(pattern, a, b) + beta
    result = wrapper._contract_einsum(pattern, cp.asarray(a), cp.asarray(b), alpha, beta, out)
    assert result is out
    np.testing.assert_allclose(cp.asnumpy(result), reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("alias", ["a", "b"])
def test_accumulation_with_aliased_input(wrapper, alias):
    a = cp.arange(16, dtype=cp.float64).reshape(4, 4) / 16
    b = cp.eye(4) * 3
    out = a if alias == "a" else b
    reference = 2 * (a @ b) + .5 * out
    wrapper._contract_einsum("ik,kj->ij", a, b, 2., .5, out)
    cp.testing.assert_allclose(out, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pattern", ["ij,jk->ik", "ij,jk->ki", "ij,ij->i", "ij,ij->", "ii,j->ij"])
def test_generic_and_strided(wrapper, pattern):
    a = cp.arange(64, dtype=cp.float64).reshape(8, 8)[::2, ::2]
    b = cp.arange(4, dtype=cp.float64) if pattern == "ii,j->ij" else a.T
    ref = np.einsum(pattern, cp.asnumpy(a), cp.asnumpy(b))
    out = wrapper._contract_einsum(pattern, a, b, 1., 0.)
    assert out.flags.c_contiguous
    np.testing.assert_allclose(cp.asnumpy(out), ref, rtol=1e-12, atol=1e-12)


def test_zero_contraction_dimension(wrapper):
    a, b = cp.empty((3, 0)), cp.empty((0, 4))
    out = cp.ones((3, 4))
    wrapper._contract_einsum("ik,kj->ij", a, b, 2., .5, out)
    cp.testing.assert_allclose(out, .5)


def test_custom_engine_is_used(wrapper):
    calls = []
    def engine(pattern, a, b):
        calls.append(pattern)
        return cp.einsum(pattern, a, b)
    result = wrapper._contract_einsum("ij,jk->ik", cp.eye(4), cp.eye(4), 1., 0., einsum=engine)
    assert calls == ["ij,jk->ik"]
    cp.testing.assert_allclose(result, cp.eye(4))


def test_mixed_numpy_cupy(wrapper):
    a = np.eye(4)
    b = cp.eye(4)
    cp.testing.assert_allclose(wrapper._contract_einsum("ij,jk->ik", a, b, 1., 0.), b)


@pytest.mark.parametrize("dtype_a,dtype_b", [(np.float32, np.float64), (np.float64, np.complex128)])
def test_mixed_dtypes(wrapper, dtype_a, dtype_b):
    a = cp.arange(20, dtype=dtype_a).reshape(4, 5)
    b = cp.arange(30, dtype=dtype_b).reshape(5, 6)
    reference = np.einsum("ik,kj->ij", cp.asnumpy(a), cp.asnumpy(b))
    out = wrapper._contract_einsum("ik,kj->ij", a, b, 1., 0.)
    np.testing.assert_allclose(cp.asnumpy(out), reference, rtol=1e-12, atol=1e-12)


def test_noncontiguous_output_and_complex_scalars(wrapper):
    a = cp.arange(16, dtype=cp.complex128).reshape(4, 4)
    b = cp.eye(4, dtype=cp.complex128)
    backing = cp.ones((8, 8), dtype=cp.complex128)
    out = backing[::2, ::2]
    reference = (2 + 1j) * cp.asnumpy(a) + (.5 - 1j)
    wrapper._contract_einsum("ik,kj->ij", a, b, 2 + 1j, .5 - 1j, out)
    np.testing.assert_allclose(cp.asnumpy(out), reference, rtol=1e-12, atol=1e-12)
    cp.testing.assert_allclose(backing[1::2], 1)


def test_broadcast_contracted_dimension(wrapper):
    a, b = cp.ones((3, 1)), cp.arange(20, dtype=cp.float64).reshape(5, 4)
    reference = np.einsum("ik,kj->ij", cp.asnumpy(a), cp.asnumpy(b))
    out = wrapper._contract_einsum("ik,kj->ij", a, b, 1., 0.)
    cp.testing.assert_allclose(out, reference)


def test_negative_strides(wrapper):
    a = cp.arange(20, dtype=cp.float64).reshape(4, 5)[:, ::-1]
    b = cp.arange(30, dtype=cp.float64).reshape(5, 6)[::-1]
    reference = np.einsum("ik,kj->ij", cp.asnumpy(a), cp.asnumpy(b))
    cp.testing.assert_allclose(wrapper._contract_einsum("ik,kj->ij", a, b, 1., 0.), reference)


def test_invalid_dimensions(wrapper):
    with pytest.raises(ValueError):
        wrapper._contract_einsum("ik,kj->ij", cp.ones((2, 3)), cp.ones((4, 2)), 1., 0.)


def test_zero_output_dimension(wrapper):
    out = wrapper._contract_einsum("ik,kj->ij", cp.ones((0, 3)), cp.ones((3, 4)), 1., 0.)
    assert out.shape == (0, 4)


def test_overlapping_output_view(wrapper):
    backing = cp.arange(24, dtype=cp.float64).reshape(4, 6) / 24
    a, out = backing[:, :4], backing[:, 2:]
    b = cp.eye(4) * 3
    reference = 2 * (a @ b) + .5 * out.copy()
    wrapper._contract_einsum("ik,kj->ij", a, b, 2., .5, out)
    cp.testing.assert_allclose(out, reference, rtol=1e-12, atol=1e-12)


def test_generic_accumulation_alias(wrapper):
    a = cp.arange(16, dtype=cp.float64).reshape(4, 4) / 16
    b = cp.eye(4) * 3
    reference = 2 * cp.einsum("ik,kj->ji", a, b) + .5 * a
    wrapper._contract_einsum("ik,kj->ji", a, b, 2., .5, a)
    cp.testing.assert_allclose(a, reference, rtol=1e-12, atol=1e-12)
