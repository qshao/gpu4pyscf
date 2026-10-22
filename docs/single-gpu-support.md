# Single-GPU Support and Portability

## Verified local configuration

One NVIDIA GB10, SM121, ARM64, 130596093952 device-reported bytes.
CUDA compiler13.0.88, CMake3.28.3, GNU13.3 compilers.
Python3.11.15, CuPy CUDA13 14.1.1, cuTENSOR CUDA13 2.7.0,
NumPy2.4.6, PySCF2.13.1, pyscf-dispersion1.5.0.
Native CUDA libxc7.1.2 was built locally. Source baseline:
9af748a71ca8f4881c604332c2f0cc1632e033ac plus branch changes.
Existing GPU services remained running; all timings are from a shared GPU.

## Capability matrix

| Capability | Executed evidence | Status |
| --- | --- | --- |
| Native source on GB10 | Complete SM121 build and source import provenance | Verified |
| CuPy fallback | FP64/complex128, mixed inputs, layouts, empty dimensions and aliasing | Verified |
| CUDA targets | CMake policy tests and actual nvcc discovery/configuration | Verified |
| Local CUTLASS source | Configure integration test and compiled kernels | Verified |
| Molecular DF HF/PBE/B3LYP/WB97X | CPU/GPU energies and gradients | Verified for recorded water case |
| HF Hessian | Maximum error1.85e-7 Eh/Bohr squared | Verified for recorded water case |
| DF-MP2 | Correlation error8.48e-12 Eh and upstream tests | Verified for recorded restricted cases |
| DF memory paths | Vitamin C,208AOs, resident/host tensors, CPU J/K references | Verified |
| CUDA11/12 and other NVIDIA hardware | Policy tests only | Execution unqualified |
| AMD/Intel | Static audit only | Execution unqualified |
| Multi-GPU | Excluded by user | Outside scope |

Largest water energy error4.98e-13 Eh; largest gradient error6.00e-10 Eh/Bohr.
This is not basis/grid convergence proof or qualification of all upstream methods.
Existing cuTENSOR and explicitly selected alternative engines retain dispatch.
Installed cuTENSOR alone does not supply missing CuPy bindings; this CuPy14
environment uses the optimized CuPy fallback.

## Performance evidence

The fallback matrix fast path avoids general einsum parsing, removes redundant
scaling passes, and evaluates aliased input before mutating output.
Initial component ratios were about2.5x for64x64 matrices and7.2x for the tested
Coulomb accumulation. See raw samples/source hashes in contraction_results.json.

Vitamin C uses a164913408-byte DF tensor. Seven alternating paired warm samples:

The following table is the initial profile, archived under
benchmarks/single_gpu/runs/2026-10-17-before-rerun/. For the fresh profile and
larger complete workflows, see the latest campaign in
[the benchmark README](../benchmarks/single_gpu/README.md#measured-campaign-2026-10-17).

| Storage, block | Baseline J/K | Modified J/K | Ratio |
| --- | ---: | ---: | ---: |
| GPU,256 | 41.78ms | 41.41ms | 1.009x |
| Host,256 | 41.94ms | 39.41ms | 1.064x |
| GPU,128 | 40.29ms | 39.66ms | 1.016x |
| Host,128 | 40.40ms | 38.06ms | 1.062x |

Block512 was slower (about61-64ms). Default256 is retained because one molecule
does not establish a general memory heuristic. Pool live/total bytes exclude
large allocations routed directly to CUDA; neither is peak memory.

Small-water SCF+gradient timings varied: an earlier run was about1.0x for both
HF/B3LYP; a later run was1.00x HF/1.12x B3LYP. Three paired repetitions on a shared
GPU do not establish reliable overall uplift. Do not extrapolate component ratios
to complete calculations.

Reports are under benchmarks/single_gpu/: contraction_results.json,
df_profile_results.json, workflow_results.json and molecular_results/.

## Reproducing the native SM121 build

Build-time network access is restricted here; stage pinned sources first.
From a fresh repository root:

    git clone --depth 1 --branch v3.4.0 https://github.com/NVIDIA/cutlass.git build/vendor/cutlass
    git clone --depth 1 --branch 7.1.2 https://gitlab.com/libxc/libxc.git build/vendor/libxc
    git -C build/vendor/libxc apply "$PWD/gpu4pyscf/lib/libxc_prune.patch"
    cmake -S build/vendor/libxc -B build/libxc-sm121 \
      -DBUILD_SHARED_LIBS=ON -DENABLE_CUDA=ON -DBUILD_TESTING=OFF \
      -DENABLE_FORTRAN=OFF -DMAXORDER=3 -DDISABLE_FHC=ON \
      -DCMAKE_INSTALL_PREFIX="$PWD/gpu4pyscf/lib/deps" \
      -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_CUDA_ARCHITECTURES=121
    cmake --build build/libxc-sm121 -j 4
    cmake --install build/libxc-sm121
    cmake -S gpu4pyscf/lib -B build/single-gpu \
      -DCMAKE_CUDA_ARCHITECTURES=121 -DBUILD_LIBXC=OFF \
      -DCUTLASS_SOURCE_DIR="$PWD/build/vendor/cutlass"
    cmake --build build/single-gpu -j 4

CUTLASS v3.4.0 commit751eb9a8859ac36bfc77551f9e4a957c31a5a8b1;
libxc7.1.2 commitf1eb21e8d6743ed5ba519b8950c1ae02c3f5acd9 plus the existing
pruning patch. BUILD_LIBXC=OFF avoids a redundant download; CUDA libxc remains used.
CUDA13.0/Blackwell activates the existing ECP correctness workaround, retained here.

Actual automatic target selection:
75-real;80-real;90-real;100-real;120-real;121-real;121-virtual.
Only SM121 binaries were fully built/executed. Explicit CUDA_ARCHITECTURES,
CMAKE_CUDA_ARCHITECTURES and CUDAARCHS choices retain precedence.
Other CUDA compilers need an explicit target choice if they lack nvcc's query.
Source qualification does not establish wheel coverage.

The earlier bwrap startup failure resolved after the user loaded the Ubuntu
AppArmor profile. Shared Python environments were not modified.

## AMD feasibility audit

Static findings; no AMD execution is claimed.

| Layer | Current dependency | Next porting gate |
| --- | --- | --- |
| Build | CUDA language and CUDA libxc in lib/CMakeLists.txt | Explicit HIP build on pinned ROCm |
| Integrals | CUDA runtime, symbols/shared memory in lib/gvhf-rys/rys_contract_k.cu | HIP translation and integral reference tests |
| Runtime | Mapped/pinned buffers, RawKernel and shuffles in lib/cupy_helper.py | Validate wave size, masks and memory semantics |
| Solvers | Direct ctypes cusolverDnDsygvd/Zhegvd in lib/cusolver.py | Validated rocSOLVER backend |
| Contractions | NVIDIA bindings plus CuPy fallback in lib/cutensor.py | Fallback qualification; native hipTensor stride/layout tests |
| XC | Stream pointers into compiled CUDA library in dft/libxc.py | HIP XC build and stream validation |

Start with allocation/contractions/solvers, then integrals, RHF, LDA/GGA, hybrid DF
and gradients. CuPy array tests alone would not qualify an AMD GPU4PySCF backend.
Upstream discussion: https://github.com/pyscf/gpu4pyscf/issues/215
CuPy limitations: https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental
