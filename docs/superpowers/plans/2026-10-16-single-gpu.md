# Single-GPU Improvement Implementation Plan

> Execute inline using superpowers:executing-plans. Track executed evidence.
> Multi-GPU changes are excluded by the user.

**Goal:** Improve single-GPU performance and broaden GPU support based on the review.
**Architecture:** Optimize the CuPy fallback, modernize target selection, build and
qualify GB10, and measure DF memory and complete workflows. Retain an AMD
feasibility audit with explicit hardware gates.
**Tech stack:** Python, NumPy, CuPy, PySCF, CUDA, CMake, pytest.
**Baseline:**9af748a71ca8f4881c604332c2f0cc1632e033ac.
**Branch:**improve-single-gpu.

## Task1: Source and environment

- [x] Clone upstream, branch, read docs/source and check CodeGraph (absent).
- [x] Qualify single GB10/ARM64/SM121 and the read-only dft_oracle environment.
- [x] Record initial missing-library/PySCF and sandbox failures.
- [x] After user's AppArmor installation, verify normal command execution.
- [x] Stage pinned CUTLASS v3.4.0 and libxc7.1.2 in build/vendor.
- [x] Apply existing pruning patch; build/install CUDA libxc under the repo.
- [x] Complete actual SM121 native build and source import verification.
  Reproduction: docs/single-gpu-support.md. Existing GPU services and shared
  scientific environments were not modified.

## Task2: Contraction fallback

Files: gpu4pyscf/lib/cutensor.py; benchmarks/single_gpu/test_contraction.py,
test_benchmark.py and contraction_results.json.

- [x] Test real wrapper independently before native build.
- [x] Cover FP64/complex128, mixed NumPy/CuPy/dtypes, layouts, zero dimensions,
  scalars, accumulation, overlapping/aliased output, and custom engines.
- [x] Observe upstream alias failure before implementing the fix.
- [x] Add bounded matrix-pattern recognition and BLAS fast path.
- [x] Compute before mutation; avoid redundant scaling passes.
- [x] Retain generic einsum and explicitly selected alternative engines.
- [x] Measure repeated synchronized baseline/current samples with source hashes.
  Initial component ratios approximately2.5x small matrix and7.2x accumulation.

## Task3: Build support

Files: gpu4pyscf/lib/CMakeLists.txt, lib/cmake/CudaArchitectures.cmake,
MANIFEST.in, README.md; test_cuda_architectures.py and test_local_cutlass.py.

- [x] Test target selection, overrides, query and failure paths.
- [x] Select only nvcc-supported generic targets plus highest-target PTX.
- [x] Execute actual automatic discovery/configuration:
  75-real;80-real;90-real;100-real;120-real;121-real;121-virtual.
- [x] Compile/execute SM121 and retain existing ECP compiler workaround.
- [x] Add CUTLASS_SOURCE_DIR, retaining default downloads.
- [x] Verify local include propagation and actual compiled CUTLASS kernels.
- [x] Package the CMake helper and document source/wheel coverage separately.

## Task4: Scientific and performance qualification

Files under benchmarks/single_gpu/: molecular.py, test_molecular.py,
molecular_results/, df_profile.py, test_df_profile.py, df_profile_results.json,
test_workflow_benchmark.py, workflow_results.json.

- [x] Execute matched CPU/GPU DF HF/PBE/B3LYP/WB97X energies and gradients.
- [x] Separate observed DF setup, SCF totals/iterations, gradients and Hessian.
  Setup remains part of total SCF elapsed time, not an additive category.
- [x] Verify gates: energy1e-8 Eh, gradient1e-6 Eh/Bohr, Hessian1e-5 Eh/Bohr squared.
  Observed maxima:4.98e-13,6.00e-10,1.85e-7 respectively.
- [x] Verify closed-shell DF-MP2 against CPU: correlation error8.48e-12 Eh.
- [x] Test mismatched settings/shapes, asymmetric Hessians, inaccurate/nonfinite
  derivatives and unconverged reference states.
- [x] Include source/runtime/device and native binary fingerprints.
- [x] Include construction/auxiliary setup/kernel in MP2 timing on both backends.
- [x] Profile Vitamin C (20atoms,208AOs,164913408-byte tensor): resident/host
  paths, packed/unpacked loops, setup and CPU-validated J/K.
- [x] Use seven alternating paired warm J/K samples.
  Block256: resident41.78->41.41ms; host41.94->39.41ms.
- [x] Evaluate128/256/512 blocks. Retain256:512 was slower and one-molecule
  evidence does not justify a new global heuristic.
- [x] Record complete paired HF/B3LYP SCF+gradient samples.
  Shared-GPU variability prevents a reliable overall uplift claim.
- [x] Test binding restoration on normal exit, exceptions and lazy imports.
  Restore block overrides without retaining bound methods.
- [x] Correct memory labels: pool live/total exclude direct large allocations
  and are not peak memory.

## Task5: Review, documentation and completion audit

- [x] Document tested configuration and native build/test commands.
- [x] Complete static AMD runtime/kernel/solver/XC/contraction audit and hardware gates.
- [x] Record raw samples, numerical errors, memory metrics and capability matrix.
- [x] Obtain separate code/measurement reviews and resolve actionable findings.
- [x] Execute upstream cutensor (5passed) and helper/DF-MP2 tests (21passed).
- [x] Run final combined focused suite: 105 passed, 8 benchmark tests deselected,
  2 warnings, 9.36 seconds. Separate profile/workflow benchmark suite: 4 passed.
- [x] Verify git diff --check and branch improve-single-gpu; retain changes locally.
- [x] Audit each design deliverable against final artifacts; see final audit in
  the design document. All scoped deliverables are satisfied with stated limits.

## Retained limits

Other NVIDIA hardware/CUDA11/12 have policy tests, not compiled qualification.
AMD/Intel hardware was unavailable. Methods outside recorded cases are not
advertised as validated. No multi-GPU implementation changed. Performance
measurements ran with other GPU services active.
