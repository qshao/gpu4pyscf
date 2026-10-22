"""Explicit configure-only integration check for pre-fetched build dependencies."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.special
def test_local_cutlass_source(tmp_path):
    if not shutil.which("nvcc"):
        pytest.skip("CUDA compiler required for configure integration test")
    root = Path(__file__).resolve().parents[2]
    local = tmp_path / "cutlass"
    header = local / "include/cutlass/cutlass.h"
    header.parent.mkdir(parents=True)
    header.write_text("// configure-only fixture; no kernel compilation\n")
    build = tmp_path / "build"
    result = subprocess.run([
        "cmake", "-S", str(root / "gpu4pyscf/lib"), "-B", str(build),
        "-DCMAKE_CUDA_ARCHITECTURES=80", "-DBUILD_LIBXC=OFF",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCUTLASS_SOURCE_DIR=" + str(local),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = json.loads((build / "compile_commands.json").read_text())
    grouped = [item for item in commands if item["file"].endswith("grouped_gemm.cu")]
    assert grouped, "CUTLASS kernels must remain enabled"
    response_files = "\n".join(path.read_text() for path in (build / "cupy_helper").rglob("*.rsp"))
    assert str(local / "include") in grouped[0]["command"] + response_files
    assert not (build / "cutlass-prefix").exists(), "Local source must not create a download project"
