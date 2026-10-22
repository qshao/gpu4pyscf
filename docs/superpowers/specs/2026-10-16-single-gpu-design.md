# Single-GPU Performance and Support Design

The user authorized developing and executing an improvement plan based on the
repository review. Multi-GPU is excluded because this machine has one GPU.

Baseline: upstream commit 9af748a71ca8f4881c604332c2f0cc1632e033ac.
Work branch: improve-single-gpu.

## Evidence and approach

This ARM64 machine exposes one NVIDIA GB10, compute capability 12.1, with
130596093952 bytes of device-reported memory. The existing dft_oracle environment
contains CuPy 14.1.1, cuTENSOR 2.7.0 (CUDA 13), and PySCF 2.13.1.
The original requirements target CUDA 12 and older CuPy. Initially importing the
source failed because its native libraries had not been built. The default Python
environment lacks PySCF. An existing environment is read-only for this task.

Three approaches were considered:
1. Optimize the usable CuPy contraction fallback and modernize build selection.
   This is directly measurable on the local GPU and is the first implementation.
2. Port the complete cuTENSOR C API to replace removed CuPy bindings. This has
   substantially greater maintenance and numerical-validation cost; investigate
   after measured fallback results.
3. Port all CUDA kernels to HIP. This requires AMD hardware and a qualified
   dependency stack; retain as a distinct hardware-gated workstream.

The chosen design preserves existing public contraction semantics and the
existing cuTENSOR path. Benchmark the current fallback before selecting a change.
A matrix-product fast path must use GPU BLAS, preserve alpha/beta/out behavior,
handle aliased arrays correctly, and retain generic einsum for unsupported
patterns and explicitly selected alternative contraction engines. Retain FP64
and complex128 accuracy. Avoid a global workspace cache without evidence.

Build improvements must retain user architecture overrides, avoid unsupported
architectures on CUDA 13, and include modern targets/PTX where the compiler
supports them. Verify decisions with table-driven configuration tests and
ultimately compile and execute the source on this GB10.

## Deliverables and gates

1. Reproducible GPU/environment metadata and contraction benchmark with cold and
   warm synchronized wall timings, correctness against NumPy, and memory data.
2. A measured single-GPU contraction optimization with regression tests for
   layouts, scalars, empty arrays, output accumulation, and aliasing.
3. Toolkit-aware architecture selection and documented single-GPU source build.
4. Molecular DF baselines and CPU comparisons for HF and pure/hybrid/RSH DFT;
   gradients and at least one Hessian comparison, plus DF-MP2 qualification.
5. A tested/documented capability matrix and AMD dependency/kernel audit.
   AMD execution remains unqualified until actual AMD tests are available.

Performance acceptance requires repeated before/after measurements on identical
inputs with compilation/warmup separated; report contention rather than claim
isolated timings. Reject changes that degrade target workloads or numerical
results. Proposed numerical gates: energy 1e-8 Hartree, gradients 1e-6
Hartree/Bohr, Hessians 1e-5 Hartree/Bohr squared, using matched basis, grid,
thresholds, and converged reference states. Check contraction relative and
absolute errors at 1e-12 for small well-conditioned FP64 inputs.

## Operational constraints

Initial commands failed before execution with the sandbox bootstrap error
"bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"; specifically,
the initial attempted CMake source build did not execute. The user's Ubuntu
AppArmor profile installation resolved this startup failure. Normal sandboxed
execution, native compilation and GPU numerical tests now succeed.
Do not stop the existing GPU services or modify other scientific environments.
Do not claim the complete plan achieved without native molecular validation.

## Final deliverable audit

1. Environment probes, source fingerprints, first-call and warm synchronized
   timings, numerical references and memory fields are recorded under
   benchmarks/single_gpu/. First-call timing shares library caches and is not
   represented as a clean-process cold-start comparison.
2. The matrix fast path and alias correction pass numerical regressions;
   alternating baseline/current component samples show measured gains.
3. Architecture policy tests pass. Native SM121 libraries compiled and executed;
   the local CUTLASS path supports builds without build-time downloads.
4. Matched water HF/PBE/B3LYP/WB97X energies/gradients, an HF Hessian and DF-MP2
   pass their CPU comparisons. Vitamin C validates resident and host DF J/K paths.
5. docs/single-gpu-support.md records the capability matrix, reproducible build,
   AMD static audit and explicit hardware gates. AMD execution remains unqualified.

The final focused suite passed 105 tests with 8 benchmark tests deselected.
Benchmarks were executed separately. Component improvements do not establish
reliable full-workflow uplift on this shared GPU. Default DF block sizing is
retained because the measured evidence does not justify a general change.
