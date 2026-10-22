"""Environment qualification; run explicitly with pytest -s."""
import importlib.metadata
import json
import platform
from pathlib import Path

def test_gpu_environment():
    import cupy as cp
    versions = {}
    for name in ("cupy-cuda12x", "cupy-cuda13x", "cutensor-cu12", "cutensor-cu13", "pyscf", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    print(json.dumps(dict(machine=platform.machine(), versions=versions,
                         device_count=cp.cuda.runtime.getDeviceCount(), device_name=name,
                         compute_capability=cp.cuda.Device(0).compute_capability,
                         memory_bytes=props["totalGlobalMem"]), indent=2))
    assert cp.cuda.runtime.getDeviceCount() == 1
    cp.testing.assert_allclose(cp.arange(4).sum(), 6)

def test_native_source_import():
    import gpu4pyscf
    print("gpu4pyscf source:", gpu4pyscf.__file__)
    expected = Path(__file__).resolve().parents[2] / "gpu4pyscf/__init__.py"
    assert Path(gpu4pyscf.__file__).resolve() == expected
