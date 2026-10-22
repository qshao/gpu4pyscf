# External QM library evaluation — 2026-10-17

## Decision

Evaluate an optional cuEST backend before rewriting integral kernels. Treat Skala
as an experimental functional integration with the locally corrected DF force
path described below; the published release remains unchanged.
Retain Mako as a research lead until its artifact and license can be inspected.
The full library qualification remains incomplete because dependencies could
not be downloaded. No new speedup claim follows from this evaluation.

## Executed evidence on GB10

| Check | Result |
| --- | --- |
| Torch 2.11.0+cu130 and CuPy 14.1.1 real CUDA allocation | Passed |
| Zero-copy Torch/CuPy DLPack exchange | Passed |
| Skala v2026.9 CPU/GPU imports with source GPU4PySCF | Passed |
| Grid-loop compatibility and zero-AO block ordering, values/first derivatives | Passed after compatibility keyword addition |
| Torch PBE CPU/GPU SCF and gradients, water and doublet water cation | Within 1e-8 Eh / 1e-6 Eh/Bohr gates in direct and DF modes |
| Direct PBE oxygen-z finite-difference derivative | Passed; errors approximately 5.2–5.7e-8 Eh/Bohr |
| DF PBE oxygen-z finite-difference derivative | Initially failed; passes after local Skala patch |
| Patched DF forces, all 9 components, distorted neutral/cation water | Passed two-step finite differences with extrapolation, two auxiliary bases |
| Actual neural Skala-1.1 checkpoint execution | Not run: only LFS pointers available |
| cuEST 0.2.2 binary execution | Not run: package unavailable to local installer |
| Mako artifact inspection/execution | Not run: Zenodo inaccessible |

CPU/GPU Torch PBE energy differences are below 7e-12 Eh and gradient differences
below 2.2e-9 Eh/Bohr across recorded cases. Native libxc PBE energy differs from
Torch PBE by approximately 3.5e-6 Eh in both direct and DF modes; this is recorded
separately rather than asserted numerically identical. Full root-cause analysis
of that independent implementation difference remains open.

The DF force issue was isolated by repeating the same tests without DF. Skala's
DF energy was differentiated using direct Coulomb gradients. Replacing the direct
Coulomb derivative by its DF counterpart, including auxiliary response, predicts
the missing oxygen-z force: 7.682e-6 / 9.385e-6 Eh/Bohr. The remaining difference
at 1e-3 Bohr steps is the approximately 5e-8 finite-difference truncation seen in
the direct cases. The diagnostic before patching is retained in
benchmarks/external_qm/df_diagnosis_before_patch.json.

The local Skala patch now dispatches CPU Coulomb derivatives to PySCF's DF J
routine and carries its auxiliary response into extra_force. GPU calculations
use the existing DF RHF/UHF jk_energy_per_atom routines. Original direct/DF tests
pass with unchanged numerical gates. A broader test uses distorted water and its
doublet cation, def2-SVP, JKFIT/Weigend auxiliary bases, all nine Cartesian force
components and central differences at 1e-3 and 5e-4 Bohr. After extrapolation,
maximum force errors are below 1e-9 Eh/Bohr in recorded runs. This is a local
integration fix, not qualification of the actual learned functional.

Patch: [skala-v2026.9-df-gradients.patch](../benchmarks/external_qm/patches/skala-v2026.9-df-gradients.patch).
It applies to Skala c0487c9a241f652e3bcee9042677da835dde3dda and was tested with
PySCF 2.13.1 and this GPU4PySCF source based on
9af748a71ca8f4881c604332c2f0cc1632e033ac. It depends on the DF J auxiliary-response
layout and the GPU jk_energy_per_atom API. Arbitrary releases, custom DF backends
and range-separated DF configurations are not covered. Modified source hashes
are recorded; shared installed packages were not patched.

The production compatibility change only accepts strict_grid_order in NumInt's
block loop. Its existing behavior always preserves grid order and zero-AO blocks.
Real-GPU regression tests failed on the missing keyword before the change and
passed afterward. No numerical algorithm was changed.

Dependency reuse was read-only and process-local: PySCF/NumPy/CuPy from dft_oracle,
Torch/e3nn and related dependencies from labeling, and pinned Skala source under
build/vendor. Actual paths are saved in preflight_imports.json. This does not
establish a clean distributable environment. Existing scientific environments
and GPU services were left intact.

## cuEST changes our correlation roadmap

Correction to the preliminary assessment: cuEST v0.2 provides
cuestDFMOIntegralsCompute, not just contracted Fock matrices. It returns
metric-transformed DF MO tensors and the official sample repository includes
MP2 examples. It may therefore help DF-MP2 and future DF-CC infrastructure.
[MO integral API](https://docs.nvidia.com/cuda/cuest/python-api/methods/mo-integrals/index.html)

Source samples pinned at 58c811d2d3bccdcea6fcd8a2c45fddd19ca2fb00 were inspected.
The sample helper computes tensors with shape naux × nleft × nright, with
explicit query/allocate/compute workspace handling. Plan exchange and range-
separation factors apply to these tensors; an adapter must configure them
appropriately for correlation integrals. The variable scratch budget is not a
limit on total memory. Verify AO normalization/order and compare small tensors,
J/K and MP2 correlation energies against PySCF before advertising a backend.

Local CUDA SDK reports 13.0.3; NVIDIA requires >=13.0.2 for CUDA 13. ARM64 binaries
are documented, but GB10/SM121 is not in the published tested-configuration list.
The local Python 3.12 probe found neither cuest bindings nor libcuest. Initial
package-index access failed with DNS errors; subsequent pip attempts returned no
matching distribution. Neither outcome proves platform incompatibility.
[Installation and tested configurations](https://docs.nvidia.com/cuda/cuest/installation.html)

cuEST also offers GPU-dependent GEMM emulation and mixed precision by default,
with CUEST_NATIVE_FP64_MATH_MODE as an opt-out. Backend qualification should
compare native FP64 and default math modes explicitly, including convergence,
relative energies and forces. A returned FP64 array alone does not establish
FP64 internal arithmetic. Do not change global cuBLAS emulation settings around
cuEST; use its documented handle controls.
[Precision controls](https://docs.nvidia.com/cuda/cuest/gemm_emulation.html)

## Skala and Mako next gates

Skala release v2026.9 is pinned to c0487c9a241f652e3bcee9042677da835dde3dda.
Its package metadata supports our PySCF 2.13.1 and Torch >=2.9; current unpinned
main has different requirements. Use a release pin. Real neural models need
separate CPU and CUDA checkpoint hashes; see the experiment README.
[Official Skala repository](https://github.com/microsoft/skala)

Mako's SC25 paper describes matrix-oriented integral evaluation, stage-aware
precision and compiler specialization. The artifact DOI could not be accessed;
license, source contents, callable APIs and GB10 support remain unknown. Published
total-energy agreement criteria are insufficient by themselves for force/Hessian
qualification at our thresholds.
[Paper](https://www.likun.tech/pdf/sc25_62.pdf)

Recommended sequence:

1. Obtain cuEST 0.2.2, verify runtime on GB10, then qualify S/J/K and DF-MO tensors.
2. Retain the local DF correction and broaden force validation before production optimization.
3. Obtain hashed neural checkpoints and run matched CPU/GPU SCF/gradient tests,
   including open shells, charged systems and grid convergence.
4. Inspect Mako artifact/license before planning code reuse or a new kernel design.
5. Return to DF-MP2 gradients/double hybrids once the reusable backend is selected.

Reproduction commands and raw artifacts: [experiment README](../benchmarks/external_qm/README.md).

## Verification outcome

Initial external suite: 8 passed, 2 failed on the DF force gates. With the local
patch, the combined external suite passed all 12 tests (3 warnings, 39.98 seconds),
including the expanded force checks. Upstream numerical
integration suite: 23 passed, 1 failed on a hard-coded sparse-index fingerprint.
That fingerprint failure reproduces with the unchanged HEAD block-loop function:
observed 110.33487367511532 versus expected 103.60117204957997. It therefore
predates the compatibility keyword addition in this environment. No claim of a
fully passing upstream numint suite is made. Separate review found no correctness issue
in the compatibility change or local DF derivative assembly; provenance and report
status findings were corrected. The upstream sparse-fingerprint failure remains
outside this external integration fix.
