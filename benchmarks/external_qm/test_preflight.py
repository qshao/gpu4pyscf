"""Real external-library probes; missing CUDA is a failure, not a skip."""
import importlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
# Read-only dependency reuse, explicitly opt-in. Existing scientific envs stay intact.
config_path = ROOT / "build/external-qm/extra_site_paths.json"
extra_paths = json.loads(config_path.read_text()) if config_path.exists() else []
extra_paths += os.environ.get("QM_EXTRA_SITE", "").split(os.pathsep)
for path in extra_paths:
    if path:
        sys.path.append(path)
sys.path.insert(0, str(ROOT / "build/vendor/skala/src"))
sys.path.insert(0, str(ROOT))


def test_dependency_imports():
    result = {"python": sys.version, "executable": sys.executable, "modules": {},
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    path = ROOT / "benchmarks/external_qm/preflight_imports.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        for name in ("numpy", "scipy", "pyscf", "cupy", "torch", "e3nn",
                     "opt_einsum_fx", "huggingface_hub", "skala"):
            module = importlib.import_module(name)
            result["modules"][name] = {
                "version": getattr(module, "__version__", None),
                "file": module.__file__,
            }
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")


def test_cuda_and_dlpack():
    import cupy as cp
    import torch
    result = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "torch_cuda_available": torch.cuda.is_available()}
    try:
        result["cupy_device_count"] = cp.cuda.runtime.getDeviceCount()
        assert result["torch_cuda_available"], "PyTorch CUDA unavailable"
        assert result["cupy_device_count"] == 1, "Expected one visible GPU"
        array = cp.arange(6, dtype=cp.float64)
        tensor = torch.from_dlpack(array)
        assert tensor.data_ptr() == array.data.ptr
        tensor.add_(1)
        torch.cuda.synchronize()
        cp.testing.assert_array_equal(array, cp.arange(1, 7))
        result["device"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = list(torch.cuda.get_device_capability(0))
        result["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        result["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
        result["dlpack_zero_copy"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (ROOT / "benchmarks/external_qm/preflight_cuda.json").write_text(
            json.dumps(result, indent=2) + "\n")


def test_skala_cpu_integration_import():
    from skala.pyscf import SkalaKS
    assert callable(SkalaKS)


def test_skala_gpu_integration_import():
    from skala.gpu4pyscf import SkalaKS
    assert callable(SkalaKS)
