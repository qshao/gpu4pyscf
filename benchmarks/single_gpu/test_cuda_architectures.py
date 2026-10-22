"""Exercise the CMake policy without requiring a CUDA compiler."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "gpu4pyscf/lib/cmake/CudaArchitectures.cmake"


@pytest.mark.parametrize("codes,expected", [
    ("sm_70\nsm_75\nsm_80\nsm_86", "70-real;80-real;80-virtual"),
    ("sm_70\nsm_75\nsm_80\nsm_90", "70-real;80-real;90-real;90-virtual"),
    ("sm_70\nsm_80\nsm_90\nsm_100\nsm_120", "70-real;80-real;90-real;100-real;120-real;120-virtual"),
    ("sm_75\nsm_80\nsm_90\nsm_100\nsm_120\nsm_121", "75-real;80-real;90-real;100-real;120-real;121-real;121-virtual"),
    ("sm_100a\nsm_120a\nsm_121a\nsm_80", "80-real;80-virtual"),
])
def test_defaults(tmp_path, codes, expected):
    assert POLICY.exists(), "CUDA target selection must be a separately testable policy"
    cmake = shutil.which("cmake")
    assert cmake, "CMake is required to validate the build policy"
    script = tmp_path / "check.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.21)\n'
        'include("' + str(POLICY) + '")\n'
        'gpu4pyscf_default_cuda_architectures(actual [=[' + codes + ']=])\n'
        'if(NOT actual STREQUAL "' + expected + '")\n'
        '  message(FATAL_ERROR "unexpected architectures: ${actual}")\n'
        'endif()\n')
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("settings,expected", [
    ('set(CUDA_ARCHITECTURES "121")', "121"),
    ('set(CMAKE_CUDA_ARCHITECTURES "80;90")', "80;90"),
    ('set(CMAKE_CUDA_ARCHITECTURES "80")\nset(CUDA_ARCHITECTURES "121")', "121"),
    ('set(ENV{CUDAARCHS} "121-real;121-virtual")', "121-real;121-virtual"),
])
def test_explicit_overrides(tmp_path, settings, expected):
    script = tmp_path / "override.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.21)\n'
        'include("' + str(POLICY) + '")\n'
        'set(CMAKE_CUDA_COMPILER "/must/not/be/queried")\n'
        + settings + '\n'
        'gpu4pyscf_select_cuda_architectures()\n'
        'if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "' + expected + '")\n'
        '  message(FATAL_ERROR "override was lost")\n'
        'endif()\n')
    result = subprocess.run([shutil.which("cmake"), "-P", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_compiler_query(tmp_path):
    import sys
    compiler = tmp_path / "nvcc"
    compiler.write_text(
        "#!" + sys.executable + "\n"
        "import sys\n"
        "assert sys.argv[1:] == [\"--list-gpu-code\"]\n"
        "print(\"sm_75\\nsm_80\\nsm_90\\nsm_100\\nsm_120\\nsm_121\")\n")
    compiler.chmod(0o755)
    script = tmp_path / "query.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.21)\n'
        'include("' + str(POLICY) + '")\n'
        'set(CMAKE_CUDA_COMPILER "' + str(compiler) + '")\n'
        'unset(ENV{CUDAARCHS})\n'
        'gpu4pyscf_select_cuda_architectures()\n'
        'if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75-real;80-real;90-real;100-real;120-real;121-real;121-virtual")\n'
        '  message(FATAL_ERROR "compiler query not reflected in defaults")\n'
        'endif()\n')
    result = subprocess.run([shutil.which("cmake"), "-P", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("codes", ["sm_52", "sm_100a", ""])
def test_no_generic_targets(tmp_path, codes):
    script = tmp_path / "unsupported.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.21)\n'
        'include("' + str(POLICY) + '")\n'
        'gpu4pyscf_default_cuda_architectures(actual [=[' + codes + ']=])\n')
    result = subprocess.run([shutil.which("cmake"), "-P", str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Set CMAKE_CUDA_ARCHITECTURES explicitly" in " ".join(result.stderr.split())


def test_unsupported_compiler_has_actionable_error(tmp_path):
    import sys
    compiler = tmp_path / "unsupported-compiler"
    compiler.write_text("#!" + sys.executable + "\nimport sys\nsys.exit(2)\n")
    compiler.chmod(0o755)
    script = tmp_path / "failed-query.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.21)\n'
        'include("' + str(POLICY) + '")\n'
        'set(CMAKE_CUDA_COMPILER "' + str(compiler) + '")\n'
        'unset(ENV{CUDAARCHS})\n'
        'gpu4pyscf_select_cuda_architectures()\n')
    result = subprocess.run([shutil.which("cmake"), "-P", str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    error = " ".join(result.stderr.split())
    assert "Set CMAKE_CUDA_ARCHITECTURES explicitly" in error
