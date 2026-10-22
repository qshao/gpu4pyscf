# External QM library qualification

The user approved evaluating cuEST, Skala and Mako before implementing new
correlation methods. Execute inline; do not modify shared environments or
production GPU4PySCF behavior.

1. Fetch official source examples and pin revisions under ignored build/vendor.
2. Create repository-local environments. Probe real CUDA allocation, torch/CuPy
   DLPack exchange and imports before attempting numerical comparisons.
3. Obtain cuEST 0.2.2 and Skala 1.1 rev1 checkpoints through normal downloads.
   Record download failures explicitly; never replace missing models with a dummy.
4. Test Skala's checkpoint-independent PBE integration against native CPU PBE,
   then CPU/GPU Skala energies and gradients when the actual weights are available.
   Use matched def2-SVP/JKFIT, grid settings, convergence and dispersion.
5. Inspect cuEST's J/K, gradient and DF-MO APIs; run official examples once the
   binary is available. A successful source audit is not numerical qualification.
6. Inspect Mako's published artifact and licensing if obtainable. Retain the
   published precision criteria separately from our tighter derivative gates.
7. Save commands, revisions, machine-readable probe results and recommendations
   under benchmarks/external_qm and docs/external-qm-libraries.md.

Acceptance: energies within 1e-8 Eh and gradients within 1e-6 Eh/Bohr for matched
CPU/GPU implementations of the same functional. Different functionals require
scientific benchmark data rather than numerical equality. Full timing claims
require repeated synchronized complete calculations with initialization separate.

## Execution record

- Source retrieval/pinning, local venv creation and read-only dependency probes
  completed. No shared environment was modified.
- Added the minimal strict_grid_order compatibility keyword after observing the
  missing-keyword failure. Real GPU zero-AO-block tests passed for AO values and
  first derivatives. Existing loop semantics are unchanged.
- Torch PBE integration ran in direct/DF modes for restricted neutral water and
  unrestricted water cation. Same-functional CPU/GPU numerical gates passed.
- Direct finite-difference force checks passed; DF checks failed at 7.7e-6 and
  9.4e-6 Eh/Bohr. Retained failures block DF force qualification.
- cuEST source audit includes DF-MO APIs and MP2 examples, correcting the initial
  narrower assessment. Binary runtime remains unavailable.
- Actual neural checkpoints and Mako artifact could not be downloaded. Their
  numerical/source/license qualification remains pending, not complete.
- Saved metadata, raw outputs, commands and findings in benchmarks/external_qm
  and docs/external-qm-libraries.md. Separate review findings were addressed.
- External suite: 8 passed, 2 failed. Upstream numint: 23 passed, 1 baseline-
  reproduced sparse-fingerprint failure. git diff --check passed.

## Follow-up authorized by user's "proceed"

- Diagnosed the DF mismatch by independently computing DF-minus-direct Coulomb
  derivatives at the converged density. The correction explains the error.
- Patched only the local Skala source: CPU DF J and auxiliary response, GPU DF
  RHF/UHF Coulomb derivative dispatch. Shared environments remain unchanged.
- Saved the patch against pinned v2026.9 under benchmarks/external_qm/patches.
- Original four PBE direct/DF cases now pass unchanged force thresholds.
- Expanded to all nine components on distorted neutral/cation water, two auxiliary
  bases and two FD steps with extrapolation. Maximum force error below 1e-9 Eh/Bohr.
- Combined external suite: 12 passed, 3 warnings in 39.98 seconds.
- Code review found no derivative-assembly correctness issue in tested cases;
  corrected status/provenance findings and documented API/version constraints.
- Retried cuEST/checkpoint/Mako retrieval; still unavailable. Full backend and
  neural-functional qualification remains pending these external dependencies.
